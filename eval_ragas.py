import json
import os
import random
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from src.agent.graph_v2 import run_agent

BENCHMARK_PATH = Path("data/processed/arxiv_multihop_v1.json")
OUTPUT_DIR = Path("Eval_agentic")
OUTPUT_CSV = OUTPUT_DIR / "eval_agent_s3.csv"
OUTPUT_JSON = OUTPUT_DIR / "agent_scores_s3.json"

N_MULTIHOP = 10
N_CHALLENGE = 3
RANDOM_SEED = 42

CHALLENGE_QUESTIONS = [
    "What is the capital of France?",
    "What is the purpose of the moon in biology?",
    "What is the role of a transformer in a software project?",
]

def load_benchmark(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_eval_items(all_items):
    random.seed(RANDOM_SEED)

    multihop_items = [
        ex for ex in all_items
        if isinstance(ex.get("supporting_chunks"), list) and len(ex.get("supporting_chunks", [])) >= 2
    ]

    selected_multihop = random.sample(multihop_items, min(N_MULTIHOP, len(multihop_items)))

    items = []
    for ex in selected_multihop:
        items.append({
            "question": ex["question"],
            "ground_truth": ex.get("ground_truth", ""),
            "hop_type": "true_2hop",
            "source": ex,
        })

    for q in CHALLENGE_QUESTIONS:
        items.append({
            "question": q,
            "ground_truth": "not in corpus",
            "hop_type": "challenge",
            "source": None,
        })

    return items

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

    print(f"Évaluation sur {len(eval_items)} questions...")
    print("Questions sélectionnées :")
    for i, item in enumerate(eval_items, 1):
        print(f"{i}. {item['question']}")

    for i, item in enumerate(eval_items, 1):
        question = item["question"]
        print(f"\n[{i}/{len(eval_items)}] {question[:80]}")

        result = run_agent(question, use_phoenix=False)

        answer = result.get("final_response", "")
        contexts = build_ragas_context(result)

        ragas_data["question"].append(question)
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(item["ground_truth"])

        agent_scores.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "answer": answer[:250],
            "agent_score": result.get("critique_score", 0.0),
            "iterations": result.get("iteration", 0),
            "hop_type": item["hop_type"],
        })

        print(f"  → score={result.get('critique_score', 0.0):.2f} | iter={result.get('iteration', 0)}")

    print("\nLancement RAGAS...")
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

    print("\n=== RAGAS — Agentic GraphRAG ===")
    print(ragas_result)

    df = ragas_result.to_pandas()
    df["hop_type"] = [s["hop_type"] for s in agent_scores]
    df["agent_score"] = [s["agent_score"] for s in agent_scores]
    df["iterations"] = [s["iterations"] for s in agent_scores]

    print("\n=== RAGAS par type de question ===")
    print(
        df.groupby("hop_type")[
            ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        ].mean()
    )

    print("\n=== Stats agent ===")
    print(f"Score agent moyen               : {df['agent_score'].mean():.3f}")
    print(f"Iterations moyennes             : {df['iterations'].mean():.2f}")
    print(f"Questions résolues sans self_correct : {(df['iterations'] == 0).sum()}/{len(df)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(agent_scores, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {OUTPUT_CSV} sauvegardé")
    print(f"✓ {OUTPUT_JSON} sauvegardé")

if __name__ == "__main__":
    run_evaluation()