import json
import networkx as nx
from pathlib import Path
from collections import Counter
import re
import pickle


##############################################################
# 1. LOAD GRAPH + TAGS
##############################################################

GRAPH_PATH = "graph.pkl"
TAGS_PATH = "tags.json"

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

tags = []
with open(TAGS_PATH, "r") as f:
    for line in f:
        tags.append(json.loads(line))


##############################################################
# 2. HELPER — lookup nodes by name
##############################################################

def find_nodes(name):
    return [n for n in G.nodes if n.lower() == name.lower()]


##############################################################
# 3. GREP RETRIEVER
##############################################################

def grep_search(keyword, repo_root):
    matches = []
    for py_file in Path(repo_root).rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf8") as f:
                code = f.read()
                if re.search(keyword, code):
                    matches.append(str(py_file))
        except:
            continue
    return matches


##############################################################
# 4. GRAPH RETRIEVER — k-hop ego graph
##############################################################

def graph_retrieve(node, hops=2):
    if node not in G.nodes:
        return []
    sub = nx.ego_graph(G, node, radius=hops)
    return list(sub.nodes)


##############################################################
# 5. METRICS
##############################################################

def recall(pred, gt):
    pred_set = set(pred)
    gt_set = set(gt)
    if len(gt_set) == 0:
        return 0
    return len(pred_set & gt_set) / len(gt_set)

def precision(pred, gt):
    pred_set = set(pred)
    gt_set = set(gt)
    if len(pred_set) == 0:
        return 0
    return len(pred_set & gt_set) / len(pred_set)

def noise_ratio(pred, gt):
    pred_set = set(pred)
    gt_set = set(gt)
    if len(pred_set) == 0:
        return 0
    return 1 - (len(pred_set & gt_set) / len(pred_set))


##############################################################
# 6. RUN EXPERIMENT FOR A GIVEN TARGET
##############################################################

def run_experiment(target, ground_truth_nodes):
    print("="*70)
    print(f"🎯 TARGET: {target}")
    print("="*70)

    # Find exact nodes in graph
    target_nodes = find_nodes(target)
    print(f"Found graph nodes: {target_nodes}")

    # Baseline 1: Grep
    grep_results = grep_search(target, repo_root="./requests")
    print(f"GREP results: {len(grep_results)} files")

    # Baseline 2: Graph K-hop
    graph_results = []
    for t in target_nodes:
        graph_results += graph_retrieve(t, hops=2)
    graph_results = list(set(graph_results))
    print(f"Graph results: {len(graph_results)} nodes")

    # Compute metrics
    print("\n📌 METRICS")
    print("Recall@graph:", recall(graph_results, ground_truth_nodes))
    print("Precision@graph:", precision(graph_results, ground_truth_nodes))
    print("Noise ratio:", noise_ratio(graph_results, ground_truth_nodes))

    print("\nDone.\n")


##############################################################
# 7. DEFINE YOUR EXPERIMENT CASES (GROUND TRUTH)
##############################################################

experiment_cases = {
    "Session": ["prepare_request", "send", "merge_environment_settings"],
    "Request": ["prepare", "copy"],
    "Response": ["json", "raise_for_status"],
}

##############################################################
# 8. RUN ALL EXPERIMENTS
##############################################################

for target, gt in experiment_cases.items():
    run_experiment(target, gt)
