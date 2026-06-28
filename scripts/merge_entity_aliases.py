from pathlib import Path
import networkx as nx

GRAPH_PATH = Path(
    "indexes/lightrag_500_connected_v2/graph_chunk_entity_relation.graphml"
)

OUTPUT_PATH = Path(
    "indexes/lightrag_500_connected_v2/graph_chunk_entity_relation_merged.graphml"
)

G = nx.read_graphml(GRAPH_PATH)

ALIASES = {
    "Large Language Model (LLM)": "Large Language Models",
    "Large Language Model(S)": "Large Language Models",
    "LLM (Large Language Models)": "Large Language Models",

    "Vision Language Model (VLM)": "Vision-Language Models",
    "Graph Neural Network (GNN)": "Graph Neural Networks",
    "Artificial Intelligence (AI)": "Artificial Intelligence",
    "Natural Language Processing (NLP)": "Natural Language Processing",
}

for alias, canonical in ALIASES.items():

    if alias not in G:
        print(f"[SKIP] {alias}")
        continue

    if canonical not in G:
        print(f"[SKIP] {canonical}")
        continue

    print(f"[MERGE] {alias} -> {canonical}")

    G = nx.contracted_nodes(
        G,
        canonical,
        alias,
        self_loops=False
    )

print()
print("Nodes :", G.number_of_nodes())
print("Edges :", G.number_of_edges())

for node, data in G.nodes(data=True):
    if "contraction" in data:
        del data["contraction"]

for u, v, data in G.edges(data=True):
    if "contraction" in data:
        del data["contraction"]

nx.write_graphml(G, OUTPUT_PATH)

print()
print("Saved :", OUTPUT_PATH)