"""
Les 3 figures principales demandees par l'encadrant pour le rapport.

  1. figures/architecture/fig21_pipeline_global.png
     Schema du pipeline complet, du corpus arXiv jusqu'a la reponse de
     l'agent, avec les 3 systemes compares dans l'ablation study.
  2. figures/graphe/fig17_stats_graphe_apres_postTrait.png
     Deja produite par scripts/graph_stats_figures.py (rien a refaire ici).
  3. figures/evaluation/fig15_ablation_study_comparaison.png
     Bar chart de l'ablation study, refait proprement : la version
     precedente etait un plot matplotlib par defaut, sans valeurs affichees
     et avec un axe Y intitule "Score RAGAS" alors que les chiffres viennent
     de la grille manuelle a 3 criteres, pas de RAGAS.

Usage:
    python scripts/make_report_figures.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ABLATION_CSV = Path("Eval_agentic/ablation_study/plan_c_resultats_agreges.csv")
FIG_PIPELINE = Path("figures/architecture/fig21_pipeline_global.png")
FIG_ABLATION = Path("figures/evaluation/fig15_ablation_study_comparaison.png")

plt.rcParams.update({"font.size": 11, "figure.dpi": 110})

NAVY = "#1f3864"
BLUE = "#4a86c8"
ORANGE = "#e07b54"
GREEN = "#6aa84f"
GREY = "#7f8c8d"


# ── 1. Schema du pipeline global ──────────────────────────────
def box(ax, x, y, w, h, text, face, edge=NAVY, fontsize=10, weight="normal", tc="black"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=tc, wrap=True)


def arrow(ax, p1, p2, color=NAVY, style="-|>", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=15,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def plot_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(17, 10.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("Pipeline global — Agentic GraphRAG sur corpus arXiv",
                 fontsize=19, fontweight="bold", pad=16)

    # --- Etage 1 : donnees ---
    ax.text(2, 95.5, "1. DONNEES", fontsize=12, fontweight="bold", color=NAVY)
    box(ax, 2, 86, 20, 7.5, "Corpus arXiv\n500 articles", "#eaf1fa")
    box(ax, 26, 86, 20, 7.5, "Chunking\n1200 tokens / 200 overlap", "#eaf1fa")
    box(ax, 50, 86, 22, 7.5, "Benchmark true multi-hop\n76 questions validees", "#fdf3d8")
    arrow(ax, (22, 89.7), (26, 89.7))
    arrow(ax, (46, 89.7), (50, 89.7))

    # --- Etage 2 : indexation ---
    ax.text(2, 81, "2. INDEXATION", fontsize=12, fontweight="bold", color=NAVY)
    box(ax, 2, 70, 30, 8.5, "ChromaDB\nembeddings denses (vectoriel)", "#e6f2e0")
    box(ax, 38, 70, 34, 8.5, "LightRAG\nextraction entites + relations", "#e6f2e0")
    arrow(ax, (17, 86), (17, 78.5))
    arrow(ax, (36, 86), (55, 78.5))

    # --- Post-traitement du graphe ---
    box(ax, 76, 70, 22, 8.5,
        "Post-traitement du graphe\nfusion des alias\n+ aretes de co-occurrence", "#fde9e0")
    arrow(ax, (72, 74.2), (76, 74.2))
    ax.text(87, 66.5, "5065/5068  ->  4945/5593\n0 noeud isole, composante geante 43% -> 55%",
            ha="center", va="top", fontsize=8.5, style="italic", color=GREY)

    # --- Etage 3 : les 3 systemes compares ---
    ax.text(2, 60, "3. SYSTEMES COMPARES (ablation study)", fontsize=12,
            fontweight="bold", color=NAVY)
    box(ax, 2, 43, 28, 13,
        "RAG baseline\n\nrecherche vectorielle seule\n(ChromaDB) -> LLM",
        "#eaf1fa", fontsize=10.5)
    box(ax, 34, 43, 28, 13,
        "LightRAG hybride\n\nretrieval hybride\n(local + global), sans agent",
        "#eaf1fa", fontsize=10.5)
    box(ax, 66, 43, 32, 13,
        "Agentic GraphRAG\n\nretrieval hybride + boucle\nagentique (LangGraph)",
        "#d6e6f7", fontsize=10.5, weight="bold")
    arrow(ax, (16, 70), (16, 56))
    arrow(ax, (50, 70), (48, 56))
    arrow(ax, (60, 70), (82, 56))

    # --- Etage 4 : boucle agentique ---
    ax.text(2, 37.5, "4. BOUCLE AGENTIQUE (v3, LangGraph)", fontsize=12,
            fontweight="bold", color=NAVY)
    steps = [
        ("QUERY\nreformulation", "#eaf1fa"),
        ("HYBRID_SEARCH\nLightRAG", "#e6f2e0"),
        ("RESPONSE\ngeneration", "#eaf1fa"),
        ("CRITIQUE\njuge independant", "#fde9e0"),
        ("FINALIZE\nmeilleure reponse", "#e6f2e0"),
    ]
    w, gap, y0 = 17, 3, 22
    for i, (label, face) in enumerate(steps):
        x = 2 + i * (w + gap)
        box(ax, x, y0, w, 9, label, face, fontsize=9.5)
        if i:
            arrow(ax, (x - gap, y0 + 4.5), (x, y0 + 4.5))

    # Boucle de retour SELF_CORRECT
    x_crit = 2 + 3 * (w + gap) + w / 2
    x_resp = 2 + 2 * (w + gap) + w / 2
    ax.plot([x_crit, x_crit, x_resp, x_resp], [y0, y0 - 5, y0 - 5, y0],
            color=ORANGE, linewidth=1.6, linestyle="--")
    arrow(ax, (x_resp, y0 - 5), (x_resp, y0), color=ORANGE, ls="--")
    ax.text((x_crit + x_resp) / 2, y0 - 6.6, "SELF_CORRECT  (score < 0.7, max 2 iterations)",
            ha="center", va="top", fontsize=9, color=ORANGE, fontweight="bold")

    # --- Etage 5 : evaluation ---
    ax.text(2, 12, "5. EVALUATION", fontsize=12, fontweight="bold", color=NAVY)
    box(ax, 2, 2, 30, 8,
        "Grille manuelle (3 criteres)\ncorrect / ancre / clair-complet", "#fdf3d8")
    box(ax, 36, 2, 30, 8, "RAGAS\n4 metriques, juge Groq", "#fdf3d8")
    box(ax, 70, 2, 28, 8, "Comparaison des versions\nv1 / v2 / v3", "#fdf3d8")
    arrow(ax, (17, 22), (17, 10))
    arrow(ax, (51, 22), (51, 10))
    arrow(ax, (84, 22), (84, 10))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {out_path}")


# ── 2. Bar chart de l'ablation study ──────────────────────────
CRIT = ["correct_0_1", "ancre_contexte_0_1", "clair_complet_0_1"]
PRETTY = {
    "correct_0_1": "Correct",
    "ancre_contexte_0_1": "Ancre au contexte",
    "clair_complet_0_1": "Clair et complet",
}
SYS_ORDER = ["RAG baseline (ChromaDB)", "LightRAG hybride (sans agent)", "Agentic GraphRAG"]
SHORT = {
    "RAG baseline (ChromaDB)": "RAG\nbaseline",
    "LightRAG hybride (sans agent)": "LightRAG\nhybride",
    "Agentic GraphRAG": "Agentic\nGraphRAG",
}


def plot_ablation(out_path: Path) -> None:
    df = pd.read_csv(ABLATION_CSV)
    col_sys = "system" if "system" in df.columns else df.columns[0]
    df = df.set_index(col_sys).reindex(SYS_ORDER).reset_index()
    df["score_global"] = df[CRIT].mean(axis=1)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Ablation study — RAG baseline vs LightRAG hybride vs Agentic GraphRAG\n"
                 "30 questions du benchmark true multi-hop, evaluation manuelle a 3 criteres",
                 fontsize=16, fontweight="bold", y=1.0)
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 0.02], wspace=0.28)

    colors = [BLUE, ORANGE, GREEN]

    # (1) Les 3 criteres
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(CRIT))
    w = 0.8 / len(SYS_ORDER)
    for i, (label, col) in enumerate(zip(SYS_ORDER, colors)):
        vals = df.loc[df[col_sys] == label, CRIT].iloc[0].tolist()
        pos = x + (i - (len(SYS_ORDER) - 1) / 2) * w
        ax1.bar(pos, vals, w, label=label, color=col, edgecolor="black", linewidth=0.6)
        for p, v in zip(pos, vals):
            ax1.text(p, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([PRETTY[c] for c in CRIT])
    ax1.set_ylabel("Taux de reussite (0-1)")
    ax1.set_ylim(0, 1.12)
    ax1.set_title("Resultats par critere", fontweight="bold", fontsize=13)
    ax1.legend(fontsize=9.5, loc="upper left")
    ax1.grid(alpha=0.3, axis="y")

    # (2) Score global
    ax2 = fig.add_subplot(gs[0, 1])
    vals = df["score_global"].tolist()
    bars = ax2.bar([SHORT[s] for s in SYS_ORDER], vals, color=colors,
                   edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax2.set_ylim(0, max(vals) * 1.25)
    ax2.set_ylabel("Moyenne des 3 criteres")
    ax2.set_title("Score global", fontweight="bold", fontsize=13)
    ax2.tick_params(axis="x", labelsize=10)
    ax2.grid(alpha=0.3, axis="y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {out_path}")
    print(df[[col_sys] + CRIT + ["score_global"]].round(3).to_string(index=False))


def main():
    plot_pipeline(FIG_PIPELINE)
    plot_ablation(FIG_ABLATION)
    print("\nFigure 2 (stats du graphe post-traite) : "
          "figures/graphe/fig17_stats_graphe_apres_postTrait.png "
          "— produite par scripts/graph_stats_figures.py")


if __name__ == "__main__":
    main()
