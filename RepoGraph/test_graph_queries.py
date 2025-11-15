import pickle
import networkx as nx

with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

print("🔹 Nodes:", len(G.nodes()))
print("🔹 Edges:", len(G.edges()))

# Print sample nodes
print("\n📌 Sample 20 nodes:")
for i, (node, data) in enumerate(G.nodes(data=True)):
    print(f"{i}: {node} => {data}")
    if i >= 20:
        break
def search_by_name(name):
    matches = []
    for node, data in G.nodes(data=True):
        if data.get("name") == name:
            matches.append((node, data))
    return matches

targets = ["Session", "PreparedRequest", "HTTPAdapter", "Request", "Response"]

for t in targets:
    results = search_by_name(t)
    print(f"\n🔍 Results for {t}: {len(results)} nodes")
    for r in results:
        print("  →", r)
def get_ego(name, k=1):
    # find node id
    node_id = None
    for n, data in G.nodes(data=True):
        if data.get("name") == name:
            node_id = n
            break

    if node_id is None:
        print(f"❌ Node {name} not found")
        return None

    ego = nx.ego_graph(G, node_id, radius=k)
    print(f"\n🌐 {k}-hop ego graph for {name}:")
    print("   Nodes:", len(ego.nodes()))
    print("   Edges:", len(ego.edges()))
    return ego

get_ego("Session", 1)
get_ego("Session", 2)
