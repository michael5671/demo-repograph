# This file is adapted from:
# - RepoMap (aider/repomap.py)
# - Agentless (get_repo_structure)
# - grep-ast + tree-sitter

import colorsys
import os
import random
import sys
import re
import warnings
from collections import namedtuple
from pathlib import Path

import ast
import json
import pickle

import networkx as nx
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser

Tag = namedtuple("Tag", "rel_fname fname line name kind category info".split())


class CodeGraph:
    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
    ):
        self.io = io
        self.verbose = verbose

        if not root:
            root = os.getcwd()
        self.root = root

        self.max_map_tokens = map_tokens
        self.max_context_window = max_context_window
        self.repo_content_prefix = repo_content_prefix

    # ========================
    # Top-level pipeline
    # ========================

    def get_code_graph(self, other_files, mentioned_fnames=None):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()

        tags = self.get_tag_files(other_files, mentioned_fnames)
        code_graph = self.tag_to_graph(tags)
        return tags, code_graph

    def get_tag_files(self, other_files, mentioned_fnames=None):
        try:
            tags = self.get_ranked_tags(other_files, mentioned_fnames)
            return tags
        except RecursionError:
            if self.io:
                self.io.tool_error("Disabling code graph, git repo too large?")
            self.max_map_tokens = 0
            return

    # ========================
    # Graph construction
    # ========================

    def tag_to_graph(self, tags):
        # Sanitize tags list: ensure only Tag objects or dict-like objects
        clean_tags = []
        for tag in tags:
            if isinstance(tag, Tag):
                clean_tags.append(
                    {
                        "name": tag.name,
                        "category": tag.category,
                        "info": tag.info,
                        "fname": tag.fname,
                        "line": tag.line,
                        "kind": tag.kind,
                    }
                )
            elif isinstance(tag, dict):
                clean_tags.append(tag)
            else:
                continue

        tags = clean_tags

        G = nx.MultiDiGraph()

        # Add nodes
        for tag in tags:
            G.add_node(
                tag["name"],
                category=tag["category"],
                info=tag["info"],
                fname=tag["fname"],
                line=tag["line"],
                kind=tag["kind"],
            )

        # Edges: class -> its methods (info = newline-separated method names)
        for tag in tags:
            if tag["category"] == "class" and tag["kind"] == "def":
                if not tag["info"]:
                    continue
                methods = [m.strip() for m in tag["info"].split("\n") if m.strip()]
                for m in methods:
                    G.add_edge(tag["name"], m)

        # Edges: reference -> definition
        tags_ref = [tag for tag in tags if tag["kind"] == "ref"]
        tags_def = [tag for tag in tags if tag["kind"] == "def"]

        defs_by_name = {}
        for td in tags_def:
            defs_by_name.setdefault(td["name"], []).append(td)

        for tr in tags_ref:
            name = tr["name"]
            if name in defs_by_name:
                for d in defs_by_name[name]:
                    G.add_edge(tr["name"], d["name"])

        return G

    # ========================
    # Utility helpers
    # ========================

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def split_path(self, path):
        path = os.path.relpath(path, self.root)
        return [path + ":"]

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            if self.io:
                self.io.tool_error(f"File not found error: {fname}")
            return None

    def _collect_defs_ast(self, code, codelines):
        """
        Dùng Python AST để lấy:
        - class_defs: name -> (start_line, end_line, [method_names])
        - func_defs: name -> (start_line, end_line, snippet)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}, {}

        class_defs = {}
        func_defs = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start is None or end is None:
                    continue
                # AST dùng 1-based line number
                start0, end0 = start - 1, end - 1

                method_names = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_names.append(item.name)

                class_defs[node.name] = {
                    "start": start0,
                    "end": end0,
                    "methods": method_names,
                }

            elif isinstance(node, ast.FunctionDef):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start is None or end is None:
                    continue
                start0, end0 = start - 1, end - 1
                snippet = "".join(codelines[start0 : end0 + 1])
                func_defs[node.name] = {
                    "start": start0,
                    "end": end0,
                    "snippet": snippet,
                }

        return class_defs, func_defs

    # ========================
    # Tag extraction (core)
    # ========================

    def get_tags(self, fname, rel_fname):
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []
        data = list(self.get_tags_raw(fname, rel_fname))
        return data

    def get_tags_raw(self, fname, rel_fname):
        # Only care about Python for now
        lang = filename_to_lang(fname)
        if lang != "python":
            return

        try:
            with open(str(fname), "r", encoding="utf-8") as f:
                code = f.read()
            with open(str(fname), "r", encoding="utf-8") as f:
                codelines = f.readlines()
        except (UnicodeDecodeError, FileNotFoundError):
            return

        code = code.replace("\ufeff", "")

        if not code.strip():
            return

        # AST-based defs
        class_defs, func_defs = self._collect_defs_ast(code, codelines)

        # 1) Emit Tags for all class defs
        for cname, info in class_defs.items():
            methods = info["methods"]
            line_nums = [info["start"], info["end"]]
            tag = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=cname,
                kind="def",
                category="class",
                info="\n".join(methods),
                line=line_nums,
            )
            yield tag

        # 2) Emit Tags for all function defs (bao gồm cả method)
        for fname_def, info in func_defs.items():
            line_nums = [info["start"], info["end"]]
            snippet = info["snippet"]
            tag = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=fname_def,
                kind="def",
                category="function",
                info=snippet,
                line=line_nums,
            )
            yield tag

        # 3) Tree-sitter-based call sites (refs)
        language = get_language("python")
        parser = get_parser("python")
        tree = parser.parse(bytes(code, "utf-8"))

        query_scm = """
        (call
          function: (identifier) @name.reference.call) @call

        (call
          function: (attribute
                      attribute: (identifier) @name.reference.call)) @call
        """
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            # capture_name is a bytes or str label
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            tag_name = node.text.decode("utf-8")

            # snippet = line chứa call
            if 0 <= start_line < len(codelines):
                snippet = codelines[start_line]
            else:
                snippet = ""

            tag = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=tag_name,
                kind="ref",
                category="function",
                info=snippet,
                line=[start_line, end_line],
            )
            yield tag

        # 4) Optional: nếu chỉ có def mà không có ref → backfill refs bằng pygments
        # (giống RepoGraph gốc)
        # Ở đây mình bỏ qua cho đơn giản, vì Requests có nhiều call rồi.

    # ========================
    # File discovery & ranking
    # ========================

    def get_ranked_tags(self, other_fnames, mentioned_fnames):
        tags_of_files = []
        personalization = {}

        fnames = sorted(set(other_fnames))
        if not fnames:
            return []

        personalize = 10 / len(fnames)

        for fname in tqdm(fnames):
            if not Path(fname).is_file():
                if fname not in self.warned_files:
                    if Path(fname).exists():
                        if self.io:
                            self.io.tool_error(
                                f"Code graph can't include {fname}, it is not a normal file"
                            )
                    else:
                        if self.io:
                            self.io.tool_error(
                                f"Code graph can't include {fname}, it no longer exists"
                            )

                self.warned_files.add(fname)
                continue

            rel_fname = self.get_rel_fname(fname)

            if fname in mentioned_fnames:
                personalization[rel_fname] = personalize

            tags = list(self.get_tags(fname, rel_fname))
            tags_of_files.extend(tags)

        return tags_of_files

    # ========================
    # Render tree (giữ lại cho đủ API, bạn không xài cũng được)
    # ========================

    def render_tree(self, abs_fname, rel_fname, lois):
        key = (rel_fname, tuple(sorted(lois)))
        if not hasattr(self, "tree_cache"):
            self.tree_cache = {}

        if key in self.tree_cache:
            return self.tree_cache[key]

        with open(str(abs_fname), "r", encoding="utf-8") as f:
            code = f.read() or ""

        if not code.endswith("\n"):
            code += "\n"

        context = TreeContext(
            rel_fname,
            code,
            color=False,
            line_number=False,
            child_context=False,
            last_line=False,
            margin=0,
            mark_lois=False,
            loi_pad=0,
            show_top_of_file_parent_scope=False,
        )

        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        tags = [tag for tag in tags if tag[0] not in chat_rel_fnames]
        tags = sorted(tags)

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        dummy_tag = (None,)
        for tag in tags + [dummy_tag]:
            this_rel_fname = tag[0]

            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if isinstance(tag, Tag):
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"
        return output

    # ========================
    # File discovery helpers
    # ========================

    def find_src_files(self, directory):
        if not os.path.isdir(directory):
            return [directory]

        src_files = []
        for root, dirs, files in os.walk(directory):
            # Skip non-source folders
            if any(skip in root for skip in ["docs", "tests", "test", "ext", "scripts", "examples"]):
                continue
            for file in files:
                if file.endswith(".py"):
                    src_files.append(os.path.join(root, file))
        return src_files

    def find_files(self, dirs):
        chat_fnames = []
        for fname in dirs:
            if Path(fname).is_dir():
                chat_fnames += self.find_src_files(fname)
            else:
                chat_fnames.append(fname)

        chat_fnames_new = []
        for item in chat_fnames:
            if not item.endswith(".py"):
                continue
            chat_fnames_new.append(item)
        return chat_fnames_new


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


if __name__ == "__main__":
    dir_name = sys.argv[1]
    code_graph = CodeGraph(root=dir_name)
    chat_fnames_new = code_graph.find_files([dir_name])

    tags, G = code_graph.get_code_graph(chat_fnames_new)

    print("---------------------------------")
    print(f"🏅 Successfully constructed the code graph for repo directory {dir_name}")
    print(f"   Number of nodes: {len(G.nodes)}")
    print(f"   Number of edges: {len(G.edges)}")
    print("---------------------------------")

    out_graph = os.path.join(os.getcwd(), "graph.pkl")
    out_tags = os.path.join(os.getcwd(), "tags.json")

    # overwrite cũ cho sạch
    if os.path.exists(out_graph):
        os.remove(out_graph)
    if os.path.exists(out_tags):
        os.remove(out_tags)

    with open(out_graph, "wb") as f:
        pickle.dump(G, f)

    with open(out_tags, "w", encoding="utf-8") as f:
        for tag in tags:
            line = json.dumps(
                {
                    "fname": tag.fname,
                    "rel_fname": tag.rel_fname,
                    "line": tag.line,
                    "name": tag.name,
                    "kind": tag.kind,
                    "category": tag.category,
                    "info": tag.info,
                }
            )
            f.write(line + "\n")

    print(f"🏅 Successfully cached code graph and node tags in directory '{os.getcwd()}'")
