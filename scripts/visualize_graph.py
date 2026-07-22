"""
Génère une figure PNG du graphe de connaissances LightRAG, à utiliser
AVANT et APRÈS le post-traitement (résolution d'entités + arêtes de
co-occurrence) pour voir visuellement l'effet des scripts
resolve_entities_embeddings.py / apply_entity_merges.py / add_cooccurrence_edges.py.

5065 nœuds sont illisibles sur une seule figure : on affiche donc la
composante connexe géante, plafonnée à --max-nodes nœuds (les nœuds de plus
haut degré sont gardés en priorité).

Usage:
    python scripts/visualize_graph.py --out figures/fig9_graphe_avant.png
    python scripts/visualize_graph.py --out figures/fig10_graphe_apres.png --max-nodes 400
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

DEFAULT_GRAPHML = "indexes/lightrag_500_connected_v2/graph_chunk_entity_relation.graphml"

TYPE_COLORS = {
    "method": "#4C72B0", "concept": "#DD8452", "model": "#55A868",
    "dataset": "#C44E52", "metric": "#8172B2", "task": "#937860",
    "application": "#DA8BC3", "group": "#8C8C8C",
}
DEFAULT_COLOR = "#64B5CD"


def load_giant_component(graphml_path: Path, max_nodes: int):
    G = nx.read_graphml(graphml_path)
    UG = G.to_undirected()
    components = sorted(nx.connected_components(UG), key=len, reverse=True)
    giant_nodes = components[0] if components else set()
    H = G.subgraph(giant_nodes).copy()

    if H.number_of_nodes() > max_nodes:
        deg = dict(H.degree())
        top_nodes = sorted(deg, key=deg.get, reverse=True)[:max_nodes]
        H = H.subgraph(top_nodes).copy()

    return G, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphml", default=DEFAULT_GRAPHML)
    ap.add_argument("--out", required=True, help="Chemin de sortie, ex: figures/fig9_graphe.png")
    ap.add_argument("--max-nodes", type=int, default=300)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    graphml_path = Path(args.graphml)
    print(f"Chargement de {graphml_path} ...")
    G_full, H = load_giant_component(graphml_path, args.max_nodes)

    print(f"Graphe complet   : {G_full.number_of_nodes()} nœuds, {G_full.number_of_edges()} arêtes")
    print(f"Figure (comp. géante, plafonnée) : {H.number_of_nodes()} nœuds, {H.number_of_edges()} arêtes")

    deg = dict(H.degree())
    node_sizes = [80 + 25 * deg.get(n, 0) for n in H.nodes()]
    node_colors = [
        TYPE_COLORS.get(str(H.nodes[n].get("entity_type", "")).lower(), DEFAULT_COLOR)
        for n in H.nodes()
    ]

    pos = nx.spring_layout(H, k=0.35, iterations=50, seed=42)

    plt.figure(figsize=(16, 12), facecolor="white")
    nx.draw_networkx_edges(H, pos, alpha=0.25, width=0.6, edge_color="#999999")
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors,
                            edgecolors="white", linewidths=0.4, alpha=0.9)

    # Labels seulement pour les nœuds les mieux connectés (sinon illisible)
    top_labeled = sorted(deg, key=deg.get, reverse=True)[: max(15, args.max_nodes // 15)]
    labels = {n: n for n in top_labeled}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=7)

    title = args.title or (
        f"Graphe de connaissances LightRAG — composante géante "
        f"({H.number_of_nodes()}/{G_full.number_of_nodes()} nœuds affichés)"
    )
    plt.title(title, fontsize=13)
    plt.axis("off")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\n✓ Figure sauvegardée -> {out_path}")


if __name__ == "__main__":
    main()
