# evaluate_agent_versions_ragas.py
"""
Evaluation RAGAS des 3 versions de l'agent Agentic GraphRAG (v1 / v2 / v3)
sur les MEMES 30 questions du benchmark true multi-hop.

Ce script n'appelle AUCUN agent : il reprend les reponses + contextes deja
generes et checkpointes par compare_agent_versions.py
(Eval_agentic/agent_versions_checkpoint.json), et se contente de les noter
avec RAGAS. Aucune notation manuelle, aucun fichier d'evaluation anterieur
(Plan C, eval_ragas*.py, ragas_age.py) n'est lu ni modifie.

Metriques RAGAS :
  - faithfulness       : la reponse est-elle ancree dans le contexte recupere ?
  - answer_relevancy   : la reponse repond-elle bien a la question ?
  - context_precision  : le contexte recupere est-il pertinent (peu de bruit) ?
  - context_recall     : le contexte couvre-t-il la ground truth ?

Robustesse (memes contraintes que le reste du projet) :
  - juge Groq en rotation sur les 3 comptes (src/utils/groq_rotation) ;
  - evaluation question par question, checkpointee apres chaque ligne
    -> un quota atteint n'annule pas le travail deja fait, il suffit de
       relancer le script ;
  - concurrence a 1 (max_workers=1, batch_size=1) : c'est la concurrence,
    pas le pipeline, qui produisait les NaN observes lors des runs precedents.

Sorties :
  - Eval_agentic/agent_versions_ragas_checkpoint.json  (reprise)
  - Eval_agentic/agent_versions_ragas_details.csv      (90 lignes notees)
  - Eval_agentic/agent_versions_ragas_resultats.csv    (tableau agrege)
  - figures/fig19_comparaison_versions_agentic_ragas.png

Usage:
    python evaluate_agent_versions_ragas.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from src.utils import groq_rotation
from src.utils.groq_rotation import AllGroqKeysExhaustedError, rotation_status

# ── Config ────────────────────────────────────────────────────
OUT_DIR = Path("Eval_agentic")
FIG_DIR = Path("figures")

SOURCE_CHECKPOINT = OUT_DIR / "agent_versions_checkpoint.json"
RAGAS_CHECKPOINT = OUT_DIR / "agent_versions_ragas_checkpoint.json"
DETAILS_CSV = OUT_DIR / "agent_versions_ragas_details.csv"
AGG_CSV = OUT_DIR / "agent_versions_ragas_resultats.csv"
FIG_PATH = FIG_DIR / "fig19_comparaison_versions_agentic_ragas.png"

VERSION_ORDER = [
    "Agentic v1 (Sprint 3)",
    "Agentic v2 (Sprint 3, corrige)",
    "Agentic v3 (Sprint 4, actuel)",
]
SHORT_LABELS = {
    "Agentic v1 (Sprint 3)": "v1\n(Sprint 3)",
    "Agentic v2 (Sprint 3, corrige)": "v2\n(Sprint 3, corrige)",
    "Agentic v3 (Sprint 4, actuel)": "v3\n(Sprint 4, actuel)",
}

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
JUDGE_MODEL = os.getenv("RAGAS_GROQ_MODEL", "llama-3.1-8b-instant")

# Budget de contexte envoye au juge, IDENTIQUE pour les 3 versions.
# Necessaire : v3 (retrieval hybride LightRAG) renvoie ~195 passages et
# ~36 000 caracteres par question, contre ~5 passages / ~4 000 caracteres
# pour v1 et v2. A cette taille, chaque appel RAGAS depasse la limite
# tokens/minute du palier gratuit Groq ; RAGAS avale l'erreur et renvoie NaN,
# ce qui rendait 100% des lignes v3 non notables. On tronque donc au budget
# ci-dessous en gardant les passages les mieux classes (ordre du retriever).
# Le meme plafond est applique a v1 et v2 pour que la comparaison reste
# equitable — en pratique il ne les affecte quasiment pas.
# A signaler comme limite : pour v3, context_precision et context_recall
# portent sur le haut du classement, pas sur la totalite du contexte recupere.
CTX_CHAR_BUDGET = int(os.getenv("RAGAS_CTX_CHAR_BUDGET", "6000"))
CTX_MAX_ITEMS = int(os.getenv("RAGAS_CTX_MAX_ITEMS", "20"))
# Pause entre deux lignes, pour rester sous la limite tokens/minute de Groq.
PACING_S = float(os.getenv("RAGAS_PACING_S", "8"))


def trim_contexts(contexts: list) -> list:
    """Tronque la liste de passages au budget commun (cf. CTX_CHAR_BUDGET)."""
    out, used = [], 0
    for c in (contexts or [])[:CTX_MAX_ITEMS]:
        c = (c or "").strip()
        if not c:
            continue
        remaining = CTX_CHAR_BUDGET - used
        if remaining <= 0:
            break
        out.append(c[:remaining])
        used += min(len(c), remaining)
    return out or ["No context retrieved."]


# ── Juge RAGAS (Groq + rotation multi-comptes) ────────────────
def build_judge():
    """Construit le juge RAGAS sur la cle Groq actuellement active.
    Retourne (ragas_llm, ragas_emb, index_de_cle)."""
    key, idx = groq_rotation.get_active_key()
    llm = ChatGroq(model=JUDGE_MODEL, api_key=key, temperature=0)
    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb), idx


def build_metrics(ragas_llm, ragas_emb):
    # strictness=1 : AnswerRelevancy demande par defaut n=strictness=3
    # completions en un seul appel ; l'API Groq refuse n>1
    # ("'n' : number must be at most 1" -> metrique systematiquement NaN).
    # Avec strictness=1 la metrique redevient calculable, au prix d'une
    # seule question generee par reponse au lieu de trois (a signaler comme
    # limite : estimation legerement plus bruitee).
    return [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb, strictness=1),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]


RUN_CONFIG = RunConfig(timeout=180, max_retries=5, max_wait=60, max_workers=1)


def _is_daily_quota_error(exc: Exception) -> bool:
    """Distingue le quota JOURNALIER (TPD, 500 000 tokens/jour/compte, non
    recuperable avant 24h -> il faut changer de cle) de la limite PAR MINUTE
    (TPM, 6 000 tokens/min, qui se reconstitue en quelques secondes -> il
    suffit d'attendre sur la meme cle).

    C'est la distinction qui manquait : une TPM prise pour un epuisement
    definitif retirait une cle parfaitement utilisable pour 24h."""
    msg = str(exc).lower()
    return "tokens per day" in msg or "tpd" in msg


def score_one(entry: dict) -> dict:
    """Note une seule ligne avec RAGAS, avec rotation reelle sur les cles.

    raise_exceptions=True (et non False) est essentiel : sinon RAGAS convertit
    silencieusement les 429 en NaN, on ne voit jamais l'erreur, et la rotation
    ne se declenche jamais. C'est exactement ce qui faisait echouer 100% des
    lignes v3 sur les 3 metriques qui envoient le contexte."""
    dataset = Dataset.from_dict({
        "question": [entry["question"]],
        "answer": [entry.get("answer", "") or ""],
        "contexts": [trim_contexts(entry.get("contexts"))],
        "ground_truth": [entry.get("ground_truth", "") or ""],
    })

    n_keys = max(1, len(groq_rotation._KEYS))
    tpm_waits = 0
    attempts = 0
    while attempts < n_keys + 3:
        attempts += 1
        ragas_llm, ragas_emb, key_idx = build_judge()  # leve si toutes epuisees
        try:
            result = evaluate(
                dataset,
                metrics=build_metrics(ragas_llm, ragas_emb),
                run_config=RUN_CONFIG,
                batch_size=1,
                raise_exceptions=True,
            )
        except Exception as exc:
            if _is_daily_quota_error(exc):
                groq_rotation.mark_exhausted(key_idx)  # bascule sur la cle suivante
                continue
            if groq_rotation._is_rate_limit_error(exc) and tpm_waits < 3:
                tpm_waits += 1
                print(f"      (limite par minute atteinte -> pause 60s, essai {tpm_waits}/3)")
                time.sleep(60)
                continue
            # Echec propre a cette ligne (JSON du juge invalide, timeout...) :
            # on enregistre NaN et on passe a la suivante.
            print(f"      (echec de notation : {type(exc).__name__} - {str(exc)[:110]})")
            return {m: None for m in METRICS}

        row = result.to_pandas().iloc[0]
        return {m: (None if pd.isna(row.get(m)) else float(row[m])) for m in METRICS}

    raise AllGroqKeysExhaustedError(
        "Les cles Groq configurees ont toutes atteint leur quota journalier."
    )


# ── Checkpoint ────────────────────────────────────────────────
def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_checkpoint(cp: dict) -> None:
    RAGAS_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    RAGAS_CHECKPOINT.write_text(json.dumps(cp, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Agregation + figure ───────────────────────────────────────
def build_details(source: dict, cp: dict) -> pd.DataFrame:
    rows = []
    for label in VERSION_ORDER:
        for qk, entry in (source.get(label) or {}).items():
            scores = (cp.get(label) or {}).get(qk, {})
            rows.append({
                "version": label,
                "question": entry["question"],
                "ground_truth": entry.get("ground_truth", ""),
                "answer": entry.get("answer", ""),
                "n_contexts": len(entry.get("contexts") or []),
                "faithfulness": scores.get("faithfulness"),
                "answer_relevancy": scores.get("answer_relevancy"),
                "context_precision": scores.get("context_precision"),
                "context_recall": scores.get("context_recall"),
                "critique_score": entry.get("critique_score"),
                "iterations": entry.get("iterations"),
                "latency_s": entry.get("latency_s"),
            })
    return pd.DataFrame(rows)


def build_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in VERSION_ORDER:
        sub = df[df["version"] == label]
        if sub.empty:
            continue
        rec = {"version": label, "n_questions": len(sub)}
        for m in METRICS:
            rec[m] = round(sub[m].mean(skipna=True), 4)
            rec[f"{m}_nan"] = int(sub[m].isna().sum())
        rec["ragas_score_moyen"] = round(float(np.nanmean([rec[m] for m in METRICS])), 4)
        rec["critique_score_moyen"] = round(sub["critique_score"].mean(skipna=True), 3)
        rec["iterations_moyennes"] = round(sub["iterations"].mean(skipna=True), 2)
        rec["latence_moyenne_s"] = round(sub["latency_s"].mean(skipna=True), 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_comparison(agg: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(19, 11))
    fig.suptitle("Comparaison RAGAS des 3 versions de l'agent Agentic GraphRAG\n"
                 "(memes 30 questions du benchmark true multi-hop, seed=42)",
                 fontsize=17, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.42, wspace=0.25)

    versions = agg["version"].tolist()
    short = [SHORT_LABELS.get(v, v) for v in versions]
    colors = ["#e07b54", "#f0c04a", "#4a86c8"][: len(versions)]

    # (1) Les 4 metriques RAGAS, groupees
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(METRICS))
    w = 0.8 / max(1, len(versions))
    for i, (v, c) in enumerate(zip(versions, colors)):
        vals = [agg.loc[agg["version"] == v, m].iloc[0] for m in METRICS]
        pos = x + (i - (len(versions) - 1) / 2) * w
        ax1.bar(pos, vals, w, label=v, color=c, edgecolor="black", linewidth=0.6)
        for p, val in zip(pos, vals):
            if val == val:
                ax1.text(p, val + 0.012, f"{val:.3f}", ha="center", va="bottom",
                         fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Faithfulness", "Answer\nrelevancy", "Context\nprecision", "Context\nrecall"])
    ax1.set_ylabel("Score RAGAS (0-1)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Metriques RAGAS par version", fontweight="bold", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, axis="y")

    # (2) Score RAGAS moyen
    ax2 = fig.add_subplot(gs[0, 2])
    vals = agg["ragas_score_moyen"].tolist()
    bars = ax2.bar(short, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, val in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, val + 0.012, f"{val:.3f}",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Moyenne des 4 metriques")
    ax2.set_ylim(0, max(0.6, max(vals) * 1.25) if vals else 1)
    ax2.set_title("Score RAGAS moyen", fontweight="bold", fontsize=14)
    ax2.grid(alpha=0.3, axis="y")

    # (3) Tableau de resultats
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_df = agg[["version", "n_questions"] + METRICS +
                   ["ragas_score_moyen", "critique_score_moyen", "iterations_moyennes",
                    "latence_moyenne_s"]].copy()
    for c in METRICS + ["ragas_score_moyen"]:
        table_df[c] = table_df[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    tbl = ax3.table(
        cellText=table_df.values,
        colLabels=["Version", "N", "Faithful.", "Ans. relev.", "Ctx precision",
                   "Ctx recall", "RAGAS moyen", "Score critique", "Iterations",
                   "Latence (s)"],
        cellLoc="center", loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)

    # Met en evidence la meilleure valeur de chaque colonne numerique.
    best_col = {}
    for j, m in enumerate(METRICS + ["ragas_score_moyen"], start=2):
        col = agg[m]
        if col.notna().any():
            best_col[j] = int(col.idxmax())

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#9aa5b1")
        if r == 0:
            cell.set_facecolor("#1f3864")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if best_col.get(c) == r - 1:
                cell.set_facecolor("#d9ead3")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#ffffff" if r % 2 else "#f3f6fa")
    ax3.set_title("Tableau de resultats — comparaison des versions de l'agent",
                  fontweight="bold", fontsize=14, pad=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


# ── Main ──────────────────────────────────────────────────────
def main():
    if not SOURCE_CHECKPOINT.exists():
        raise SystemExit(f"Introuvable : {SOURCE_CHECKPOINT}. "
                         f"Lancez d'abord compare_agent_versions.py.")

    source = load_json(SOURCE_CHECKPOINT, {})
    cp = load_json(RAGAS_CHECKPOINT, {})

    total = sum(len(source.get(v) or {}) for v in VERSION_ORDER)
    done = sum(len((cp.get(v) or {})) for v in VERSION_ORDER)
    print(f"Source     : {SOURCE_CHECKPOINT}")
    print(f"Juge RAGAS : Groq {JUDGE_MODEL} (max_workers=1, batch_size=1)")
    print(f"Embeddings : Ollama nomic-embed-text (local, hors quota)")
    print(f"Contexte   : plafonne a {CTX_CHAR_BUDGET} caracteres / {CTX_MAX_ITEMS} passages "
          f"(identique pour les 3 versions)")
    print(f"[GROQ ROTATION] {rotation_status()}")
    print(f"A noter    : {total} lignes ({done} deja dans le checkpoint RAGAS)\n")

    interrupted = False
    for label in VERSION_ORDER:
        entries = source.get(label) or {}
        if not entries:
            print(f"!! {label} : aucune donnee source, ignore.")
            continue
        cp.setdefault(label, {})
        print(f"=== {label} : {len(entries)} questions ===")

        for i, (qk, entry) in enumerate(entries.items(), 1):
            # Une ligne n'est consideree comme faite que si ses 4 metriques
            # sont renseignees : les lignes partiellement NaN (echec quota au
            # moment ou elles ont ete notees) sont automatiquement reprises.
            done = cp[label].get(qk)
            if done and all(done.get(m) is not None for m in METRICS):
                print(f"  [{i}/{len(entries)}] (checkpoint) {entry['question'][:60]}")
                continue
            if done:
                print(f"  [{i}/{len(entries)}] (reprise NaN) {entry['question'][:60]}")
            print(f"  [{i}/{len(entries)}] {entry['question'][:60]}")
            try:
                scores = score_one(entry)
            except AllGroqKeysExhaustedError as e:
                print(f"\n!! Quota Groq atteint sur les 3 cles : {e}")
                print(f"   Arret propre. Relancez ce script : il reprendra ici.")
                interrupted = True
                break
            cp[label][qk] = scores
            save_checkpoint(cp)
            time.sleep(PACING_S)  # respiration anti tokens/minute
            shown = " | ".join(
                f"{m.split('_')[0][:7]}={scores[m]:.2f}" if scores[m] is not None else f"{m.split('_')[0][:7]}=NaN"
                for m in METRICS
            )
            print(f"      {shown}")
        if interrupted:
            break
        print(f"  -> {len(cp[label])}/{len(entries)} notees.\n")

    # ── Export ────────────────────────────────────────────────
    df = build_details(source, cp)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DETAILS_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] {DETAILS_CSV} : {len(df)} lignes")

    agg = build_aggregate(df)
    agg.to_csv(AGG_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] {AGG_CSV}")

    print("\n=== Validite du run (NaN par metrique) ===")
    for label in VERSION_ORDER:
        sub = df[df["version"] == label]
        if sub.empty:
            continue
        nans = ", ".join(f"{m}={int(sub[m].isna().sum())}/{len(sub)}" for m in METRICS)
        print(f"  {label:34s} : {nans}")

    print("\n=== Resultats agreges ===")
    print(agg[["version", "n_questions"] + METRICS + ["ragas_score_moyen"]].to_string(index=False))

    if not agg.empty:
        plot_comparison(agg, FIG_PATH)

    if interrupted:
        print("\n!! Run incomplet (quota) : les chiffres ci-dessus sont partiels. "
              "Relancez le script pour les completer.")


if __name__ == "__main__":
    main()
