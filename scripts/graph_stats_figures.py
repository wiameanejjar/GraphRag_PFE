"""
Figures de statistiques du graphe de connaissances, AVANT et APRES le
post-traitement du 22/07 (fusion des alias d'entites + aretes de co-occurrence).

Produit :
  1. figures/fig16_stats_graphe_avant_postTrait.png  (dashboard, graphe ancien)
  2. figures/fig17_stats_graphe_apres_postTrait.png  (dashboard, graphe actuel)
  3. figures/fig18_comparaison_graphe_avant_apres.png (comparatif + tableau)
  4. Eval_agentic/graphe_comparaison_avant_apres.csv  (tableau de resultats)

Le dashboard reprend la mise en page de figures/graph_statistics_v1.png
(deja produite le 07/06 sur l'ancien graphe) pour que les deux figures du
memoire soient directement comparables.

Usage:
    python scripts/graph_stats_figures.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────
GRAPH_AVANT = Path("indexes/lightrag_500_connected_v2_backup_before_merge_20260722_162530/graph_chunk_entity_relation.graphml")
GRAPH_APRES = Path("indexes/lightrag_500_connected_v2/graph_chunk_entity_relation.graphml")

FIG_DIR = Path("figures")
OUT_DIR = Path("Eval_agentic")

FIG_AVANT = FIG_DIR / "fig16_stats_graphe_avant_postTrait.png"
FIG_APRES = FIG_DIR / "fig17_stats_graphe_apres_postTrait.png"
FIG_COMP = FIG_DIR / "fig18_comparaison_graphe_avant_apres.png"
CSV_COMP = OUT_DIR / "graphe_comparaison_avant_apres.csv"

plt.rcParams.update({"font.size": 11, "figure.dpi": 110})


# ── Calcul des statistiques ───────────────────────────────────
def compute_stats(graphml_path: Path) -> dict:
    G = nx.read_graphml(graphml_path)
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()

    deg = dict(G.degree())
    degrees = np.array(list(deg.values())) if deg else np.array([0])
    counter = Counter(degrees.tolist())

    UG = G.to_undirected()
    components = sorted(nx.connected_components(UG), key=len, reverse=True)
    giant = len(components[0]) if components else 0

    top = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:15]

    return {
        "graphml": str(graphml_path),
        "nodes": n_nodes,
        "edges": n_edges,
        "density": nx.density(G),
        "avg_degree": float(degrees.mean()),
        "median_degree": float(np.median(degrees)),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "isolated_deg0": counter.get(0, 0),
        "isolated_deg0_pct": 100 * counter.get(0, 0) / max(1, n_nodes),
        "deg1_pct": 100 * counter.get(1, 0) / max(1, n_nodes),
        "n_components": len(components),
        "giant_component_size": giant,
        "giant_component_pct": 100 * giant / max(1, n_nodes),
        "avg_clustering": nx.average_clustering(UG),
        "degrees": degrees,
        "top_nodes": top,
    }


# ── Dashboard (meme mise en page que graph_statistics_v1.png) ──
def plot_dashboard(stats: dict, titre: str, sous_titre: str, out_path: Path) -> None:
    degrees = stats["degrees"]
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(titre, fontsize=20, fontweight="bold", y=0.97)

    # (1) Histogramme des degres
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(degrees, bins=60, color="#6fa8dc", edgecolor="#2c3e50", linewidth=0.5)
    ax1.axvline(stats["avg_degree"], color="red", linestyle="--", linewidth=2,
                label=f"Moyenne: {stats['avg_degree']:.2f}")
    ax1.axvline(stats["median_degree"], color="green", linestyle="--", linewidth=2,
                label=f"Mediane: {stats['median_degree']:.2f}")
    ax1.set_title("Distribution des Degres (Histogramme)", fontweight="bold", fontsize=14)
    ax1.set_xlabel("Degre")
    ax1.set_ylabel("Frequence")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (2) Distribution log-log (power law)
    ax2 = fig.add_subplot(2, 2, 2)
    c = Counter(degrees.tolist())
    xs = [d for d in sorted(c) if d > 0]
    ys = [c[d] for d in xs]
    ax2.scatter(xs, ys, s=70, color="#6aa84f", edgecolor="#38761d", alpha=0.85)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Distribution Log-Log (Power Law)", fontweight="bold", fontsize=14)
    ax2.set_xlabel("Degre (log scale)")
    ax2.set_ylabel("Frequence (log scale)")
    ax2.grid(alpha=0.3, which="both")

    # (3) Top 15 entites par degre
    ax3 = fig.add_subplot(2, 2, 3)
    names = [n for n, _ in stats["top_nodes"]][::-1]
    vals = [d for _, d in stats["top_nodes"]][::-1]
    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(vals)))
    bars = ax3.barh(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels([n[:32] for n in names], fontsize=10)
    ax3.set_title("Top 15 Entites par Degre", fontweight="bold", fontsize=14)
    ax3.set_xlabel("Degre")
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontweight="bold", fontsize=9)
    ax3.grid(alpha=0.3, axis="x")

    # (4) Resume statistique
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    resume = (
        "RESUME STATISTIQUE DU GRAPHE\n\n"
        f"Noeuds (Entites) : {stats['nodes']}\n"
        f"Aretes (Relations): {stats['edges']}\n"
        f"Densite: {stats['density']:.4f}\n\n"
        "Degres:\n"
        f"  - Moyenne: {stats['avg_degree']:.2f}\n"
        f"  - Mediane: {stats['median_degree']:.2f}\n"
        f"  - Min: {stats['min_degree']} | Max: {stats['max_degree']}\n"
        f"  - Noeuds isoles (deg=0): {stats['isolated_deg0']} "
        f"({stats['isolated_deg0_pct']:.1f}%)\n\n"
        "Structure:\n"
        f"  - Composantes connexes: {stats['n_components']}\n"
        f"  - Composante geante: {stats['giant_component_size']}/{stats['nodes']} "
        f"({stats['giant_component_pct']:.1f}%)\n"
        f"  - Clustering: {stats['avg_clustering']:.4f}\n\n"
        f"{sous_titre}"
    )
    ax4.text(0.03, 0.95, resume, va="top", ha="left", family="monospace", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.9", facecolor="#fdf3d8", edgecolor="#b8a06a"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


# ── Figure comparative avant / apres ──────────────────────────
COMPARE_ROWS = [
    ("Noeuds (entites)", "nodes", "{:.0f}", False),
    ("Aretes (relations)", "edges", "{:.0f}", True),
    ("Degre moyen", "avg_degree", "{:.3f}", True),
    ("Noeuds isoles (deg=0)", "isolated_deg0", "{:.0f}", False),
    ("Noeuds a degre 1 (%)", "deg1_pct", "{:.1f}", False),
    ("Composantes connexes", "n_components", "{:.0f}", False),
    ("Composante geante (noeuds)", "giant_component_size", "{:.0f}", True),
    ("Composante geante (%)", "giant_component_pct", "{:.1f}", True),
    ("Clustering moyen", "avg_clustering", "{:.4f}", True),
    ("Densite", "density", "{:.6f}", True),
]


def build_comparison_table(avant: dict, apres: dict) -> pd.DataFrame:
    rows = []
    for label, key, fmt, higher_is_better in COMPARE_ROWS:
        a, b = avant[key], apres[key]
        delta = b - a
        delta_pct = (100 * delta / a) if a else float("nan")
        if higher_is_better:
            sens = "amelioration" if delta > 0 else ("degradation" if delta < 0 else "stable")
        else:
            sens = "amelioration" if delta < 0 else ("degradation" if delta > 0 else "stable")
        rows.append({
            "metrique": label,
            "avant_postTraitement": fmt.format(a),
            "apres_postTraitement": fmt.format(b),
            "delta": fmt.format(delta) if not fmt.endswith("f}") or True else delta,
            "delta_pct": f"{delta_pct:+.1f}%" if delta_pct == delta_pct else "n/a",
            "sens": sens,
        })
    return pd.DataFrame(rows)


def plot_comparison(avant: dict, apres: dict, df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Graphe de connaissances : avant vs apres post-traitement\n"
                 "(fusion des alias d'entites + aretes de co-occurrence, 22/07/2026)",
                 fontsize=17, fontweight="bold", y=0.99)
    # 2 rangees : 4 graphiques en haut, tableau de resultats en bas (sur toute
    # la largeur) -> evite tout chevauchement entre le tableau et les axes.
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.45, wspace=0.28)

    col_avant, col_apres = "#e07b54", "#4a86c8"

    # (1) Volumetrie
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["Noeuds", "Aretes"]
    x = np.arange(len(labels))
    w = 0.35
    va = [avant["nodes"], avant["edges"]]
    vb = [apres["nodes"], apres["edges"]]
    ax1.bar(x - w / 2, va, w, label="Avant", color=col_avant, edgecolor="black", linewidth=0.6)
    ax1.bar(x + w / 2, vb, w, label="Apres", color=col_apres, edgecolor="black", linewidth=0.6)
    for xi, (a, b) in enumerate(zip(va, vb)):
        ax1.text(xi - w / 2, a, f"{a}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax1.text(xi + w / 2, b, f"{b}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Volumetrie du graphe", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")

    # (2) Connectivite
    ax2 = fig.add_subplot(gs[0, 1])
    labels2 = ["Composante\ngeante (%)", "Noeuds isoles\n(%)", "Noeuds\ndegre 1 (%)"]
    x2 = np.arange(len(labels2))
    va2 = [avant["giant_component_pct"], avant["isolated_deg0_pct"], avant["deg1_pct"]]
    vb2 = [apres["giant_component_pct"], apres["isolated_deg0_pct"], apres["deg1_pct"]]
    ax2.bar(x2 - w / 2, va2, w, label="Avant", color=col_avant, edgecolor="black", linewidth=0.6)
    ax2.bar(x2 + w / 2, vb2, w, label="Apres", color=col_apres, edgecolor="black", linewidth=0.6)
    for xi, (a, b) in enumerate(zip(va2, vb2)):
        ax2.text(xi - w / 2, a, f"{a:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax2.text(xi + w / 2, b, f"{b:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=10)
    ax2.set_ylabel("%")
    ax2.set_title("Connectivite", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    # (3) Degre moyen + clustering
    ax3 = fig.add_subplot(gs[0, 2])
    labels3 = ["Degre moyen", "Clustering\n(x10)"]
    x3 = np.arange(len(labels3))
    va3 = [avant["avg_degree"], avant["avg_clustering"] * 10]
    vb3 = [apres["avg_degree"], apres["avg_clustering"] * 10]
    ax3.bar(x3 - w / 2, va3, w, label="Avant", color=col_avant, edgecolor="black", linewidth=0.6)
    ax3.bar(x3 + w / 2, vb3, w, label="Apres", color=col_apres, edgecolor="black", linewidth=0.6)
    for xi, (a, b) in enumerate(zip(va3, vb3)):
        ax3.text(xi - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax3.text(xi + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels3, fontsize=10)
    ax3.set_title("Densification locale", fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.3, axis="y")

    # (4) Distributions de degres superposees
    ax4 = fig.add_subplot(gs[0, 3])
    bins = np.arange(0, 26, 1)
    ax4.hist(np.clip(avant["degrees"], 0, 25), bins=bins, alpha=0.6, label="Avant",
             color=col_avant, edgecolor="black", linewidth=0.4)
    ax4.hist(np.clip(apres["degrees"], 0, 25), bins=bins, alpha=0.6, label="Apres",
             color=col_apres, edgecolor="black", linewidth=0.4)
    ax4.set_title("Distribution des degres (tronquee a 25)", fontweight="bold")
    ax4.set_xlabel("Degre")
    ax4.set_ylabel("Frequence")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # (5-6) Tableau de resultats
    ax5 = fig.add_subplot(gs[1, :])
    ax5.axis("off")
    table_df = df[["metrique", "avant_postTraitement", "apres_postTraitement", "delta_pct", "sens"]]
    tbl = ax5.table(
        cellText=table_df.values,
        colLabels=["Metrique", "Avant post-traitement", "Apres post-traitement", "Variation", "Sens"],
        cellLoc="center", loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.75)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#9aa5b1")
        if r == 0:
            cell.set_facecolor("#1f3864")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            sens = df.iloc[r - 1]["sens"]
            if c == 4:
                cell.set_facecolor({"amelioration": "#d9ead3", "degradation": "#f4cccc"}.get(sens, "#efefef"))
            else:
                cell.set_facecolor("#ffffff" if r % 2 else "#f3f6fa")
    ax5.set_title("Tableau comparatif des statistiques du graphe", fontweight="bold",
                  fontsize=14, pad=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def main():
    for p in (GRAPH_AVANT, GRAPH_APRES):
        if not p.exists():
            raise SystemExit(f"Introuvable : {p}")

    print(f"Lecture du graphe AVANT : {GRAPH_AVANT}")
    avant = compute_stats(GRAPH_AVANT)
    print(f"  -> {avant['nodes']} noeuds / {avant['edges']} aretes")

    print(f"Lecture du graphe APRES : {GRAPH_APRES}")
    apres = compute_stats(GRAPH_APRES)
    print(f"  -> {apres['nodes']} noeuds / {apres['edges']} aretes")

    plot_dashboard(
        avant,
        "Analyse du Graphe de Connaissances - AVANT post-traitement",
        f"Index : {GRAPH_AVANT.parent.name}",
        FIG_AVANT,
    )
    plot_dashboard(
        apres,
        "Analyse du Graphe de Connaissances - APRES post-traitement",
        f"Index : {GRAPH_APRES.parent.name}",
        FIG_APRES,
    )

    df = build_comparison_table(avant, apres)
    plot_comparison(avant, apres, df, FIG_COMP)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_COMP, index=False, encoding="utf-8-sig")
    print(f"[OK] {CSV_COMP}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
