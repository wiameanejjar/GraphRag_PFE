"""
Compare le score interne de l'agent (critique_score, produit par le nœud
CRITIQUE) aux 4 métriques RAGAS, pour détecter un éventuel biais de
surestimation (auto-évaluation) ou de sous-estimation.

Usage:
    python analyze_critique_bias.py --csv Eval_agentic/eval_agent_lightrag_only.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def compute_stats(critique, ragas_metric):
    mask = critique.notna() & ragas_metric.notna()
    c, r = critique[mask].to_numpy(), ragas_metric[mask].to_numpy()
    if len(c) < 3:
        return None
    pear_r, pear_p = pearsonr(c, r)
    spear_r, spear_p = spearmanr(c, r)
    mae = np.mean(np.abs(c - r))
    rmse = np.sqrt(np.mean((c - r) ** 2))
    bias = np.mean(c - r)  # >0 : le critique surestime par rapport à cette métrique
    return {"pearson": pear_r, "pearson_p": pear_p, "spearman": spear_r,
            "spearman_p": spear_p, "mae": mae, "rmse": rmse, "mean_bias": bias, "n": len(c)}


def confusion(df, seuil_critique, seuil_ragas_good):
    ragas_composite = df[RAGAS_METRICS].mean(axis=1)
    accepted = df["critique_score"] >= seuil_critique
    good     = ragas_composite >= seuil_ragas_good

    tp = int((accepted & good).sum())
    fp = int((accepted & ~good).sum())   # accepté par le critique MAIS mauvais selon RAGAS
    fn = int((~accepted & good).sum())   # rejeté par le critique MAIS bon selon RAGAS
    tn = int((~accepted & ~good).sum())
    return tp, fp, fn, tn, ragas_composite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Eval_agentic/eval_agent_lightrag_only.csv")
    ap.add_argument("--seuil-critique", type=float, default=0.75,
                     help="doit correspondre à CRITIQUE_SEUIL de graph_v3.py")
    ap.add_argument("--seuil-ragas-good", type=float, default=0.6,
                     help="moyenne des 4 métriques RAGAS au-dessus de laquelle une réponse est 'bonne'")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "critique_score" not in df.columns:
        raise SystemExit("Colonne 'critique_score' absente — utilisez le CSV produit par "
                          "eval_lightrag_ragas.py")

    print(f"{len(df)} questions chargées depuis {args.csv}\n")

    # ── 1. Corrélations / erreurs métrique par métrique ────────────────
    print("=== Critique interne vs métriques RAGAS ===")
    rows = []
    for metric in RAGAS_METRICS:
        stats = compute_stats(df["critique_score"], df[metric])
        if stats is None:
            print(f"  {metric:20s} : pas assez de données valides")
            continue
        rows.append({"metric": metric, **stats})
        print(f"  {metric:20s} | Pearson r={stats['pearson']:+.3f} (p={stats['pearson_p']:.3f}) "
              f"| Spearman ρ={stats['spearman']:+.3f} | MAE={stats['mae']:.3f} "
              f"| RMSE={stats['rmse']:.3f} | biais moyen={stats['mean_bias']:+.3f}")

    # ── 2. Faux positifs / faux négatifs ────────────────────────────────
    tp, fp, fn, tn, ragas_composite = confusion(df, args.seuil_critique, args.seuil_ragas_good)
    total = tp + fp + fn + tn
    print(f"\n=== Matrice de confusion (seuil critique={args.seuil_critique}, "
          f"seuil RAGAS 'bon'={args.seuil_ragas_good}) ===")
    print(f"  Vrais positifs  (acceptée, bonne)                : {tp}")
    print(f"  FAUX POSITIFS   (acceptée MAIS mauvaise/RAGAS)   : {fp}  <- surestimation du critique")
    print(f"  FAUX NÉGATIFS   (rejetée MAIS bonne/RAGAS)       : {fn}  <- sous-estimation du critique")
    print(f"  Vrais négatifs  (rejetée, mauvaise)               : {tn}")
    if total:
        print(f"  Taux de faux positifs : {fp/total*100:.1f}% | Taux de faux négatifs : {fn/total*100:.1f}%")

    # ── 3. Effet judge_independent, si la colonne existe ────────────────
    if "judge_independent" in df.columns:
        n_self = int((~df["judge_independent"].astype(bool)).sum())
        if n_self:
            print(f"\n⚠ {n_self}/{len(df)} lignes ont judge_independent=False (auto-évaluation).")
            print("  Statistiques recalculées sur le sous-ensemble à juge RÉELLEMENT indépendant :")
            df_indep = df[df["judge_independent"].astype(bool)]
            if len(df_indep) >= 3:
                for metric in RAGAS_METRICS:
                    s = compute_stats(df_indep["critique_score"], df_indep[metric])
                    if s:
                        print(f"    {metric:20s} | Pearson r={s['pearson']:+.3f} | biais moyen={s['mean_bias']:+.3f}")
            else:
                print("    Pas assez de lignes à juge indépendant pour recalculer.")

    # ── 4. Verdict ───────────────────────────────────────────────────────
    mean_bias_overall = np.mean([r["mean_bias"] for r in rows]) if rows else 0.0
    print("\n=== VERDICT ===")
    if mean_bias_overall > 0.15 and fp > fn:
        print(f"BIAIS D'AUTO-ÉVALUATION DÉTECTÉ : le critique surestime systématiquement "
              f"la qualité des réponses (biais moyen={mean_bias_overall:+.3f}, "
              f"{fp} faux positifs vs {fn} faux négatifs).")
    elif mean_bias_overall < -0.15 and fn > fp:
        print(f"Le critique est systématiquement PLUS SÉVÈRE que RAGAS "
              f"(biais moyen={mean_bias_overall:+.3f}) — sur-déclenche SELF_CORRECT inutilement.")
    else:
        print(f"Pas de biais systématique flagrant (biais moyen={mean_bias_overall:+.3f}, "
              f"FP={fp}, FN={fn}) — le critique semble raisonnablement calibré sur cet échantillon.")
    print(f"(NB : verdict basé sur n={len(df)} questions — à confirmer sur un échantillon plus "
          f"large avant conclusion définitive, cf. limite du benchmark 4% vraies questions multi-hop.)")


if __name__ == "__main__":
    main()
