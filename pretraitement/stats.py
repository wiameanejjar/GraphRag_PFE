import json
from collections import Counter

print("Chargement des données...")

with open("data/chunks/arxiv_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open("data/cleaned/arxiv_cleaned.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

with open("data/hotpotqa/hotpotqa_2000.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

tokens_list = [c["token_count"] for c in chunks]
avg_tokens = sum(tokens_list) // len(tokens_list)

all_cats = []
for p in papers:
    cats = p["categories"]
    if isinstance(cats, list):
        all_cats.extend(cats)
    else:
        all_cats.extend(cats.split())
top_themes = Counter(all_cats).most_common(10)

q_types = Counter(q["type"] for q in questions)

print("\n" + "=" * 50)
print("   STATISTIQUES DESCRIPTIVES DU DATASET")
print("=" * 50)
print(f"\nCorpus arXiv CS.AI :")
print(f"  Nombre de documents    : {len(papers)}")
print(f"  Nombre de chunks       : {len(chunks)}")
print(f"  Tokens moyens/chunk    : {avg_tokens}")
print(f"  Tokens min/chunk       : {min(tokens_list)}")
print(f"  Tokens max/chunk       : {max(tokens_list)}")
print(f"\nDistribution des thèmes (top 10) :")
for cat, count in top_themes:
    print(f"  {cat:25s} : {count} articles")
print(f"\nBenchmark HotpotQA :")
print(f"  Nombre de questions    : {len(questions)}")
print(f"  Types de questions     : {dict(q_types)}")

stats = {
    "corpus_arxiv": {
        "total_docs": len(papers),
        "total_chunks": len(chunks),
        "avg_tokens_per_chunk": avg_tokens,
        "min_tokens": min(tokens_list),
        "max_tokens": max(tokens_list),
        "top_themes": [[c, n] for c, n in top_themes]
    },
    "benchmark_hotpotqa": {
        "total_questions": len(questions),
        "question_types": dict(q_types)
    }
}

with open("data/dataset_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"\nStatistiques sauvegardées : data/dataset_stats.json")
