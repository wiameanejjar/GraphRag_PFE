"""
Mesure la qualité de connectivité du graphe LightRAG (avant / après correctifs).

Reproduit et automatise le calcul qui avait été fait à la main le 07/06
(figures/graph_statistics_20260607_025453.json), pour pouvoir comparer
objectivement l'effet de chaque étape de réparation (résolution d'entités,
arêtes de co-occurrence, etc.) sans avoir à relancer l'indexation.

Usage:
    python scripts/graph_quality_report.py
    python scripts/graph_quality_report.py --graphml indexes/lightrag_500_connected_v2/graph_chunk_entity_relation.graphml
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx


def report(graphml_path: Path) -> dict:
    G = nx.read_graphml(graphml_path)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)

    deg = dict(G.degree())
    degrees = list(deg.values())
    avg_degree = sum(degrees) / max(1, n_nodes)
    deg_counter = Counter(degrees)
    n_isolated = deg_counter.get(0, 0)
    n_deg1 = deg_counter.get(1, 0)

    UG = G.to_undirected()
    components = sorted(nx.connected_components(UG), key=len, reverse=True)
    giant = components[0] if components else set()
    giant_pct = 100 * len(giant) / max(1, n_nodes)

    try:
        clustering = nx.average_clustering(UG)
    except Exception:
        clustering = None

    out = {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": density,
        "avg_degree": avg_degree,
        "isolated_deg0": n_isolated,
        "isolated_deg0_pct": 100 * n_isolated / max(1, n_nodes),
        "deg1_pct": 100 * n_deg1 / max(1, n_nodes),
        "n_components": len(components),
        "giant_component_size": len(giant),
        "giant_component_pct": giant_pct,
        "avg_clustering": clustering,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--graphml",
        default="indexes/lightrag_500_connected_v2/graph_chunk_entity_relation.graphml",
    )
    ap.add_argument("--save-json", default=None, help="Chemin optionnel pour sauvegarder le rapport JSON")
    args = ap.parse_args()

    stats = report(Path(args.graphml))

    print(f"=== Rapport de connectivité — {args.graphml} ===")
    print(f"Nœuds                       : {stats['nodes']}")
    print(f"Relations                   : {stats['edges']}")
    print(f"Densité                     : {stats['density']:.6f}")
    print(f"Degré moyen                 : {stats['avg_degree']:.3f}")
    print(f"Nœuds isolés (deg=0)        : {stats['isolated_deg0']} ({stats['isolated_deg0_pct']:.1f}%)")
    print(f"Nœuds à degré 1             : {stats['deg1_pct']:.1f}%")
    print(f"Nb composantes connexes     : {stats['n_components']}")
    print(f"Composante géante           : {stats['giant_component_size']} nœuds ({stats['giant_component_pct']:.1f}%)")
    print(f"Clustering coefficient moy. : {stats['avg_clustering']:.4f}" if stats["avg_clustering"] is not None else "Clustering : N/A")

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\n✓ Sauvegardé -> {args.save_json}")


if __name__ == "__main__":
    main()
