# eval_ragas_3.py
import json, os, random
from pathlib import Path
from scipy import stats

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
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
BENCHMARK_PATH = Path("data/processed/benchmark_true_multihop.json")
OUTPUT_DIR     = Path("Eval_agentic")
OUTPUT_CSV     = OUTPUT_DIR / "eval_agent_s4_lightrag.csv"
OUTPUT_JSON    = OUTPUT_DIR / "agent_scores_s4_lightrag.json"


# Recommandation prof : réduire l'échelle le temps de stabiliser le scoring
# RAGAS (0 erreur / 0 NaN sur ce volume), puis remonter à 20 progressivement.
N_QUESTIONS  = 10
RANDOM_SEED  = 42

# ── Chargement benchmark ──────────────────────────────────────
def load_benchmark(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_eval_items(all_items, seed: int):
    random.seed(seed)
    selected = random.sample(all_items, min(N_QUESTIONS, len(all_items)))
    print(f"Benchmark total (true multi-hop validé) : {len(all_items)} questions")
    print(f"Sélectionnées                           : {len(selected)} questions\n")
    return selected

def build_ragas_context(result: dict) -> list:
    """ contexte RAGAS = exclusivement ce que LightRAG a récupéré
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
        hop_type = "true_multihop"
        print(f"[{i}/{len(eval_items)}] {ex['question'][:70]}")

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
    # Recommandations prof suite aux NaN observés avec batch_size=4 sur Groq :
    # le problème vient de la CONCURRENCE + de l'instabilité du juge (429 /
    # JSON mal formé), pas du pipeline lui-même. Corrections appliquées :
    #  1) concurrence RAGAS abaissée (max_workers=2, batch_size réduit)
    #  2) back-off plus généreux (max_retries, max_wait augmentés)
    #  3) juge API plus fiable en option (OpenAI gpt-4o-mini) si Groq
    #     reste instable — active via USE_OPENAI_JUDGE + OPENAI_API_KEY
    #  4) échelle réduite (N_QUESTIONS) le temps de stabiliser
    #  5) diagnostic NaN après coup (cf. plus bas)
    print("Lancement RAGAS...")
    use_nvidia_judge = os.getenv("USE_NVIDIA_JUDGE", "false").lower() == "true"
    nvidia_judge_key  = os.getenv("NVIDIA_API_KEY", "")
    use_gemini_judge = os.getenv("USE_GEMINI_JUDGE", "false").lower() == "true"
    gemini_key        = os.getenv("GEMINI_API_KEY", "")
    use_openai_judge = os.getenv("USE_OPENAI_JUDGE", "false").lower() == "true"
    openai_key        = os.getenv("OPENAI_API_KEY", "")
    use_groq   = os.getenv("USE_GROQ", "false").lower() == "true"
    groq_key   = os.getenv("GROQ_API_KEY", "")
    use_nvidia = os.getenv("USE_NVIDIA", "false").lower() == "true"
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")

    if use_nvidia_judge and nvidia_judge_key:
        nvidia_judge_model = os.getenv("NVIDIA_JUDGE_MODEL", "deepseek-ai/deepseek-v4-pro")
        print(f"[RAGAS JUDGE] NVIDIA {nvidia_judge_model} (DeepSeek, juge API)")
        llm_eval = ChatOpenAI(
            model=nvidia_judge_model,
            api_key=nvidia_judge_key,
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            temperature=0,
            max_tokens=16384,
            model_kwargs={"extra_body": {"chat_template_kwargs": {"thinking": False}}},
        )
        ragas_batch_size = 1
        ragas_max_workers = 1  # anti-429 : la clé NVIDIA/DeepSeek tolère mal la concurrence
    elif use_gemini_judge and gemini_key:
        gemini_judge_model = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.0-flash")
        print(f"[RAGAS JUDGE] Google Gemini {gemini_judge_model} (juge API fiable, gratuit, fallback Groq)")
        llm_eval = ChatGoogleGenerativeAI(model=gemini_judge_model, google_api_key=gemini_key, temperature=0,
                                           convert_system_message_to_human=True)
        ragas_batch_size = 2
        ragas_max_workers = 2
    elif use_openai_judge and openai_key:
        openai_judge_model = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
        print(f"[RAGAS JUDGE] OpenAI {openai_judge_model} (juge API fiable, fallback Groq)")
        llm_eval = ChatOpenAI(model=openai_judge_model, api_key=openai_key, temperature=0)
        ragas_batch_size = 2
        ragas_max_workers = 2
    elif use_groq and groq_key:
        ragas_groq_model = os.getenv("RAGAS_GROQ_MODEL", "llama-3.1-8b-instant")
        print(f"[RAGAS JUDGE] Groq {ragas_groq_model}")
        llm_eval = ChatGroq(model=ragas_groq_model, api_key=groq_key, temperature=0)
        ragas_batch_size = 2  # abaissé de 4 à 2 (recommandation prof, anti-429)
        ragas_max_workers = 2
    elif use_nvidia and nvidia_key:
        print(f"[RAGAS JUDGE] NVIDIA {os.getenv('NVIDIA_MODEL', 'z-ai/glm-5.2')}")
        llm_eval = ChatOpenAI(
            model=os.getenv("NVIDIA_MODEL", "z-ai/glm-5.2"),
            api_key=nvidia_key,
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            temperature=0,
        )
        ragas_batch_size = 1
        ragas_max_workers = 1
    else:
        print("[RAGAS JUDGE] Ollama llama3.1:8b (local)")
        llm_eval = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434", temperature=0)
        ragas_batch_size = 1
        ragas_max_workers = 1
    emb_eval  = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    ragas_llm = LangchainLLMWrapper(llm_eval)
    ragas_emb = LangchainEmbeddingsWrapper(emb_eval)
    print(f"[RAGAS CONFIG] batch_size={ragas_batch_size}, max_workers={ragas_max_workers}, max_retries=5, max_wait=60")

    dataset = Dataset.from_dict(ragas_data)

    ragas_result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
        run_config=RunConfig(timeout=180, max_retries=5, max_wait=60, max_workers=ragas_max_workers),
        batch_size=ragas_batch_size,
        raise_exceptions=False,
    )

    # ── Post-processing ────────────────────────────────────────
    df = ragas_result.to_pandas()

    # Recommandation prof (5) : compter les NaN par métrique avant toute
    # interprétation. Au-delà de 20-30% de NaN, les moyennes ne sont pas
    # exploitables -> il faut relancer plutôt que de commenter le résultat.
    print("\n=== Validité du run (NaN par métrique) ===")
    NAN_THRESHOLD_PCT = 10  # critere de succes du prof (plan de recuperation du 23/07)
    run_is_valid = True
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        n_nan = df[m].isna().sum()
        pct_nan = n_nan / len(df) * 100
        flag = "⚠ INEXPLOITABLE" if pct_nan > NAN_THRESHOLD_PCT else "OK"
        print(f"  {m:25s} : {n_nan}/{len(df)} NaN ({pct_nan:.0f}%) — {flag}")
        if pct_nan > NAN_THRESHOLD_PCT:
            run_is_valid = False
    if not run_is_valid:
        print(f"\n⚠ Plus de {NAN_THRESHOLD_PCT}% de NaN sur au moins une métrique : "
              f"ce run n'est PAS exploitable tel quel. Relancer (cf. RAGAS CONFIG "
              f"ci-dessus) avant d'utiliser ces chiffres dans le mémoire.")

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
