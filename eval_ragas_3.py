# eval_ragas_3.py
import json, random
from pathlib import Path
from scipy import stats

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness, AnswerRelevancy,
    ContextPrecision, ContextRecall,
)
from ragas.run_config import RunConfig

from src.agent.graph_v3 import run_agent

# ── Config ────────────────────────────────────────────────────
BENCHMARK_PATH = Path("data/processed/arxiv_multihop_v1.json")
OUTPUT_DIR     = Path("Eval_agentic")
OUTPUT_CSV     = OUTPUT_DIR / "eval_agent_s4_lightrag.csv"
OUTPUT_JSON    = OUTPUT_DIR / "agent_scores_s4_lightrag.json"

N_TRUE_2HOP  = 10    # toutes les vraies (il n'y en a que 10 dans le benchmark)
N_PSEUDO     = 4     # pseudo 2-hop
RANDOM_SEED  = 42

# ── Chargement benchmark ──────────────────────────────────────
def load_benchmark(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_eval_items(all_items, seed: int):
    random.seed(seed)

    true_2hop = [
        ex for ex in all_items
        if isinstance(ex.get("supporting_chunks"), list)
        and len(ex["supporting_chunks"]) >= 2
        and ex["supporting_chunks"][0] != ex["supporting_chunks"][1]
    ]
    pseudo_2hop = [
        ex for ex in all_items
        if isinstance(ex.get("supporting_chunks"), list)
        and len(ex["supporting_chunks"]) >= 2
        and ex["supporting_chunks"][0] == ex["supporting_chunks"][1]
    ]

    selected_true   = random.sample(true_2hop,   min(N_TRUE_2HOP, len(true_2hop)))
    selected_pseudo = random.sample(pseudo_2hop, min(N_PSEUDO,    len(pseudo_2hop)))
    selected        = selected_true + selected_pseudo
    random.shuffle(selected)

    print(f"Benchmark total   : {len(all_items)} questions")
    print(f"True 2-hop dispo  : {len(true_2hop)}")
    print(f"Pseudo 2-hop dispo: {len(pseudo_2hop)}")
    print(f"Sélectionnés      : {len(selected_true)} true + {len(selected_pseudo)} pseudo = {len(selected)} questions\n")
    return selected

def build_ragas_context(result: dict) -> list:
    """CORRIGÉ : contexte RAGAS = exclusivement ce que LightRAG a récupéré
    (result['lightrag_retrieved_contexts'], liste itemisée entités+relations+
    chunks — cf. graph_v3.node_hybrid_search). Avant, ce code lisait
    'vector_results' (Chroma) et 'lightrag_context' (string tronquée à 800
    caractères) — deux champs qui n'existent plus dans l'état de l'agent."""
    contexts = result.get("lightrag_retrieved_contexts") or []
    return contexts if contexts else ["No context retrieved."]

# ── Évaluation principale ──────────────────────────────────────
def run_evaluation():
    all_items  = load_benchmark(BENCHMARK_PATH)
    eval_items = build_eval_items(all_items, RANDOM_SEED)

    ragas_data = {
        "question":    [],
        "answer":      [],
        "contexts":    [],
        "ground_truth":[],
    }
    agent_scores = []

    for i, ex in enumerate(eval_items, 1):
        hop_type = (
            "true_2hop"
            if ex["supporting_chunks"][0] != ex["supporting_chunks"][1]
            else "pseudo_2hop"
        )
        print(f"[{i}/{len(eval_items)}] [{hop_type}] {ex['question'][:70]}")

        result = run_agent(ex["question"])

        ragas_data["question"].append(ex["question"])
        ragas_data["answer"].append(result.get("final_response", ""))
        ragas_data["contexts"].append(build_ragas_context(result))
        ragas_data["ground_truth"].append(ex.get("ground_truth", ""))

        agent_scores.append({
            "question":          ex["question"],
            "ground_truth":      ex.get("ground_truth", ""),
            "answer":            result.get("final_response", "")[:250],
            "agent_score":       result.get("critique_score", 0.0),
            "judge_independent": result.get("critique_judge_independent", False),
            "iterations":        result.get("iteration", 0),
            "hop_type":          hop_type,
        })
        print(f"  → agent_score={result.get('critique_score', 0):.2f} "
              f"| judge_independent={result.get('critique_judge_independent')} "
              f"| iter={result.get('iteration', 0)}\n")

    # ── RAGAS ─────────────────────────────────────────────────
    # CORRIGÉ : ce bloc avait disparu (git diff confirmé la fois précédente),
    # laissant "ragas_result" non défini plus bas -> NameError garanti.
    print("Lancement RAGAS...")
    llm_eval  = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434", temperature=0)
    emb_eval  = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    ragas_llm = LangchainLLMWrapper(llm_eval)
    ragas_emb = LangchainEmbeddingsWrapper(emb_eval)

    dataset = Dataset.from_dict(ragas_data)

    ragas_result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
        run_config=RunConfig(timeout=900),
        batch_size=1,
        raise_exceptions=False,
    )

    # ── Post-processing ────────────────────────────────────────
    df = ragas_result.to_pandas()
    df["hop_type"]          = [s["hop_type"]          for s in agent_scores]
    df["agent_score"]       = [s["agent_score"]       for s in agent_scores]
    df["judge_independent"] = [s["judge_independent"] for s in agent_scores]
    df["iterations"]        = [s["iterations"]        for s in agent_scores]

    pearson_r, p_val = stats.pearsonr(
        df["agent_score"],
        df["faithfulness"].fillna(0)
    )

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    sc      = (df["iterations"] > 0).sum()

    # ── Affichage ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("=== RAGAS Global ===")
    print("="*55)
    for m in metrics:
        print(f"  {m:25s} : {df[m].mean():.4f}")

    print("\n=== RAGAS par type de question ===")
    print(df.groupby("hop_type")[metrics].mean().to_string())

    print("\n=== Stats agent ===")
    print(f"  Score agent moyen              : {df['agent_score'].mean():.3f}")
    print(f"  Iterations moyennes            : {df['iterations'].mean():.2f}")
    print(f"  SELF_CORRECT déclenché         : {sc}/{len(df)} ({sc/len(df)*100:.0f}%)")
    print(f"  Pearson (agent / faithfulness) : {pearson_r:.3f}  (p={p_val:.3f})")

    n_self_eval = (~df["judge_independent"]).sum()
    if n_self_eval:
        print(f"  ⚠ {n_self_eval}/{len(df)} scores sont des auto-évaluations "
              f"(judge_independent=False) — interpréter la corrélation ci-dessus avec prudence")

    print("\n=== Comparaison Sprint 3 → Sprint 4 ===")
    s3 = {"faithfulness":       0.300,
          "context_precision":  0.000,
          "context_recall":     0.082,
          "pearson":           -0.082,
          "self_correct_%":     0.0}
    s4 = {"faithfulness":       df["faithfulness"].mean(),
          "context_precision":  df["context_precision"].mean(),
          "context_recall":     df["context_recall"].mean(),
          "pearson":            pearson_r,
          "self_correct_%":     sc / len(df) * 100}
    for k in s3:
        arrow = "↑" if s4[k] > s3[k] else "↓"
        print(f"  {k:20s} : {s3[k]:.3f} → {s4[k]:.3f} {arrow}")

    # ── Sauvegarde ────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(agent_scores, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {OUTPUT_CSV}")
    print(f"✓ {OUTPUT_JSON}")
    return df, pearson_r

if __name__ == "__main__":
    run_evaluation()
