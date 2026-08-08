# aggregate_manual_eval_v1_v3.py
"""
Agrege l'evaluation manuelle v1 vs v3 une fois la grille remplie, et produit
le tableau de resultats + la figure de comparaison.

Meme methode que l'ablation study (Plan C) : moyenne de chaque critere binaire
par version, puis score global = moyenne des 3 criteres.

Entree  : Eval_agentic/versions_agent/v1_v3_evaluation_manuelle.csv
Sorties : Eval_agentic/versions_agent/v1_v3_resultats_agreges.csv
          figures/evaluation/fig20_comparaison_v1_v3_manuelle.png

Usage:
    python aggregate_manual_eval_v1_v3.py
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

IN_CSV = Path("Eval_agentic/versions_agent/v1_v3_evaluation_manuelle.csv")
AGG_CSV = Path("Eval_agentic/versions_agent/v1_v3_resultats_agreges.csv")
FIG_PATH = Path("figures/evaluation/fig20_comparaison_v1_v3_manuelle.png")

V1_LABEL = "Agentic v1 (Sprint 3)"
V3_LABEL = "Agentic v3 (Sprint 4, actuel)"
ORDER = [V1_LABEL, V3_LABEL]
GRID = ["correct_0_1", "ancre_contexte_0_1", "clair_complet_0_1"]
PRETTY = {
    "correct_0_1": "Correct",
    "ancre_contexte_0_1": "Ancre au contexte",
    "clair_complet_0_1": "Clair et complet",
}

plt.rcParams.update({"font.size": 11, "figure.dpi": 110})


def load() -> pd.DataFrame:
    if not IN_CSV.exists():
        raise SystemExit(f"Introuvable : {IN_CSV}. Lancez d'abord build_manual_eval_v1_v3.py.")
    df = pd.read_csv(IN_CSV)
    for c in GRID:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def check(df: pd.DataFrame) -> None:
    """Refuse d'agreger une grille incomplete ou hors bornes : une moyenne
    calculee sur des lignes non notees serait fausse sans le montrer."""
    problems = []
    for label in ORDER:
        sub = df[df["system"] == label]
        if sub.empty:
            problems.append(f"{label} : aucune ligne")
            continue
        for c in GRID:
            n_missing = int(sub[c].isna().sum())
            if n_missing:
                problems.append(f"{label} / {c} : {n_missing} ligne(s) non notee(s)")
            bad = sub[~sub[c].isin([0, 1]) & sub[c].notna()]
            if len(bad):
                problems.append(f"{label} / {c} : {len(bad)} valeur(s) hors 0/1")
    if problems:
        print("Grille incomplete — agregation impossible :")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ORDER:
        sub = df[df["system"] == label]
        rec = {"version": label, "n_questions": len(sub)}
        for c in GRID:
            rec[c] = round(sub[c].mean(), 4)
        rec["score_global"] = round(float(np.mean([rec[c] for c in GRID])), 4)
        rows.append(rec)
    return pd.DataFrame(rows)


def plot(agg: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Evaluation manuelle — agent v1 (Sprint 3) vs agent v3 (Sprint 4)\n"
                 "memes 30 questions du benchmark true multi-hop, grille a 3 criteres",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.55], hspace=0.45, wspace=0.25)

    colors = ["#e07b54", "#4a86c8"]

    # (1) Les 3 criteres
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(GRID))
    w = 0.8 / len(ORDER)
    for i, (label, col) in enumerate(zip(ORDER, colors)):
        vals = [agg.loc[agg["version"] == label, c].iloc[0] for c in GRID]
        pos = x + (i - (len(ORDER) - 1) / 2) * w
        ax1.bar(pos, vals, w, label=label, color=col, edgecolor="black", linewidth=0.6)
        for p, v in zip(pos, vals):
            ax1.text(p, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                     fontweight="bold", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([PRETTY[c] for c in GRID])
    ax1.set_ylabel("Taux de reussite (0-1)")
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Resultats par critere", fontweight="bold", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")

    # (2) Score global
    ax2 = fig.add_subplot(gs[0, 2])
    vals = agg["score_global"].tolist()
    bars = ax2.bar(["v1\n(Sprint 3)", "v3\n(Sprint 4)"], vals, color=colors,
                   edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                 ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax2.set_ylim(0, max(0.6, max(vals) * 1.3) if vals else 1)
    ax2.set_ylabel("Moyenne des 3 criteres")
    ax2.set_title("Score global", fontweight="bold", fontsize=13)
    ax2.grid(alpha=0.3, axis="y")

    # (3) Tableau
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_df = agg.copy()
    for c in GRID + ["score_global"]:
        table_df[c] = table_df[c].map(lambda v: f"{v:.3f}")
    tbl = ax3.table(
        cellText=table_df[["version", "n_questions"] + GRID + ["score_global"]].values,
        colLabels=["Version", "N"] + [PRETTY[c] for c in GRID] + ["Score global"],
        cellLoc="center", loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)

    best = {}
    for j, c in enumerate(GRID + ["score_global"], start=2):
        best[j] = int(agg[c].idxmax())
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#9aa5b1")
        if r == 0:
            cell.set_facecolor("#1f3864")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if best.get(c) == r - 1:
                cell.set_facecolor("#d9ead3")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#ffffff" if r % 2 else "#f3f6fa")
    ax3.set_title("Tableau de resultats", fontweight="bold", fontsize=13, pad=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def main():
    df = load()
    check(df)
    agg = aggregate(df)
    AGG_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(AGG_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] {AGG_CSV}\n")
    print(agg.to_string(index=False))
    plot(agg, FIG_PATH)


if __name__ == "__main__":
    main()
