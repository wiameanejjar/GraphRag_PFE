# build_manual_eval_v1_v3.py
"""
Prepare la grille d'evaluation manuelle comparant l'agent v1 (Sprint 3) et
l'agent v3 (Sprint 4, actuel) sur les MEMES 30 questions du benchmark.

Meme grille que l'ablation study (Plan C), 3 criteres binaires par reponse :
    correct_0_1         : la reponse repond-elle correctement a la question,
                          au regard de la ground truth ?
    ancre_contexte_0_1  : la reponse s'appuie-t-elle sur le contexte recupere,
                          sans information inventee ?
    clair_complet_0_1   : la reponse est-elle claire, complete et exploitable ?

Ce script ne note rien : il assemble le fichier a remplir a partir des
reponses deja generees (Eval_agentic/agent_versions_checkpoint.json), sans
relancer les agents ni consommer de quota. La notation est faite a la main.

Le second script, aggregate_manual_eval_v1_v3.py, calcule les moyennes une
fois le fichier rempli.

Usage:
    python build_manual_eval_v1_v3.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from pathlib import Path

import pandas as pd

EVAL_ROOT = Path("Eval_agentic")
OUT_DIR = EVAL_ROOT / "versions_agent"
SOURCE_CHECKPOINT = EVAL_ROOT / "agent_versions_checkpoint.json"
OUT_CSV = OUT_DIR / "v1_v3_evaluation_manuelle.csv"

V1_LABEL = "Agentic v1 (Sprint 3)"
V3_LABEL = "Agentic v3 (Sprint 4, actuel)"
GRID = ["correct_0_1", "ancre_contexte_0_1", "clair_complet_0_1"]

# Extrait de contexte affiche pour pouvoir juger le critere "ancre au contexte"
# sans ouvrir le checkpoint. v3 renvoie ~195 passages : on en montre assez pour
# juger, pas la totalite (le fichier resterait illisible dans Excel).
CTX_CHARS = 1500


def main():
    if not SOURCE_CHECKPOINT.exists():
        raise SystemExit(f"Introuvable : {SOURCE_CHECKPOINT}")

    src = json.loads(SOURCE_CHECKPOINT.read_text(encoding="utf-8"))

    # On conserve l'ordre des questions de v1 pour que les deux versions
    # apparaissent dans le meme ordre -> comparaison ligne a ligne plus facile.
    order = list((src.get(V1_LABEL) or {}).keys())

    rows = []
    for label in (V1_LABEL, V3_LABEL):
        entries = src.get(label) or {}
        for qk in order:
            e = entries.get(qk)
            if e is None:
                continue
            contexts = e.get("contexts") or []
            excerpt = " || ".join(c.replace("\n", " ") for c in contexts)[:CTX_CHARS]
            row = {
                "system": label,
                "question": e["question"],
                "ground_truth": e.get("ground_truth", ""),
                "answer": e.get("answer", ""),
                "context_excerpt": excerpt,
            }
            for c in GRID:
                row[c] = ""     # a remplir a la main (0 ou 1)
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # utf-8-sig : sinon Excel affiche mal les accents des reponses.
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] {OUT_CSV} : {len(df)} lignes a noter")
    print(df["system"].value_counts().to_string())
    print(f"\nColonnes a remplir (0 ou 1) : {', '.join(GRID)}")
    print("Puis : python aggregate_manual_eval_v1_v3.py")


if __name__ == "__main__":
    main()
