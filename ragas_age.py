import json
import random
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.run_config import RunConfig

from src.agent.graph_v1 import run_agent

BENCHMARK_PATH = Path("data/processed/arxiv_multihop_v1.json")
OUTPUT_DIR = Path("Eval_agentic")
OUTPUT_CSV = OUTPUT_DIR / "eval_agent_s4.csv"
OUTPUT_JSON = OUTPUT_DIR / "agent_score_s4.json"

N_MULTIHOP = 10
RANDOM_SEED = 42

def load_benchmark(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_eval_items(all_items):
    random.seed(RANDOM_SEED)
    multihop_items = [
        ex for ex in all_items
        if isinstance(ex.get("supporting_chunks"), list) and len(ex.get("supporting_chunks", [])) >= 2
    ]
    selected = random.sample(multihop_items, min(N_MULTIHOP, len(multihop_items)))
    return selected

def build_ragas_context(result):
    vector_chunks = result.get("vector_results", []) or []
    graph_context = result.get("fused_context", "")[:1200]
    contexts = []
    if vector_chunks:
        contexts.extend(vector_chunks[:4])
    if graph_context:
        contexts.append(graph_context)
    return contexts[:5] if contexts else ["No context retrieved."]

def run_evaluation():
    all_items = load_benchmark(BENCHMARK_PATH)
    eval_items = build_eval_items(all_items)

    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    agent_scores = []

    for i, ex in enumerate(eval_items, 1):
        print(f"\n[{i}/{len(eval_items)}] {ex['question'][:80]}")
        result = run_agent(ex["question"])

        ragas_data["question"].append(ex["question"])
        ragas_data["answer"].append(result.get("final_response", ""))
        ragas_data["contexts"].append(build_ragas_context(result))
        ragas_data["ground_truth"].append(ex.get("ground_truth", ""))

        agent_scores.append({
            "question": ex["question"],
            "ground_truth": ex.get("ground_truth", ""),
            "answer": result.get("final_response", "")[:250],
            "agent_score": result.get("critique_score", 0.0),
            "iterations": result.get("iteration", 0),
            "hop_type": "true_2hop",
        })

    llm_eval = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434", temperature=0)
    emb_eval = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
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

    df = ragas_result.to_pandas()
    df["agent_score"] = [s["agent_score"] for s in agent_scores]
    df["iterations"] = [s["iterations"] for s in agent_scores]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(agent_scores, f, ensure_ascii=False, indent=2)

    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())
    print("Saved:", OUTPUT_CSV, OUTPUT_JSON)

if __name__ == "__main__":
    run_evaluation()