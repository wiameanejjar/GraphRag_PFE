import os
import json
import asyncio
import requests
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./lightrag_storage_new"
PROGRESS_FILE = "progress.json"

# ---------------- CHECKPOINT ----------------
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        return json.load(open(PROGRESS_FILE))
    return {"last_index": 0}

def save_progress(i):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_index": i}, f)

# ---------------- GPU SAFE EMBEDDING ----------------
async def embedding_func(texts):
    embeddings = []

    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )

        emb = np.array(response.json()["embedding"], dtype=np.float32)
        embeddings.append(emb)

    return np.vstack(embeddings)

# ---------------- MAIN ----------------
async def main():
    print("Initialisation LightRAG (GPU Ollama mode)...")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="llama3.1:8b",
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {
                "num_ctx": 2048,
                "num_thread": 6,     #  CPU stable
                "num_gpu": 1         #  force GPU usage
            }
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=300,   #  IMPORTANT (évite crash)
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    print("LightRAG OK ")

    # ---------------- LOAD DATA ----------------
    with open("data/cleaned/arxiv_cleaned.json", "r", encoding="utf-8") as f:
        papers = json.load(f)

    papers_100 = papers[:100]

    progress = load_progress()
    start = progress["last_index"]

    print(f" Reprise à partir de : {start}")

    #  IMPORTANT: empêcher crash event loop
    for i in range(start, len(papers_100)):
        paper = papers_100[i]

        texte = f"""
Title: {paper['title']}
Abstract: {paper['abstract']}
Content: {paper['full_text'][:500]}
"""

        try:
            await rag.ainsert(texte)

            print(f" {i} - {paper['title'][:60]}")
            save_progress(i)

            #  STABLE SLEEP (IMPORTANT pour GPU + Ollama)
            await asyncio.sleep(2)

        except Exception as e:
            print(f" Erreur doc {i}: {str(e)[:120]}")
            save_progress(i)
            await asyncio.sleep(10)
            continue

    print(" Indexation terminée !")

    # ---------------- TEST ----------------
    question = "What are the main methods used in machine learning for classification?"

    print("\n--- NAIVE ---")
    try:
        res1 = await rag.aquery(question, param=QueryParam(mode="naive"))
        print(res1[:300])
    except Exception as e:
        print("naive error:", e)

    print("\n--- HYBRID ---")
    try:
        res2 = await rag.aquery(question, param=QueryParam(mode="hybrid"))
        print(res2[:300])
    except Exception as e:
        print("hybrid error:", e)

    # ---------------- SAVE ----------------
    results = {
        "question": question,
        "naive": str(res1),
        "hybrid": str(res2)
    }

    os.makedirs("data/lightrag_results", exist_ok=True)

    with open("data/lightrag_results/comparison_naive_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n Résultats sauvegardés")


# ---------------- SAFE RUN ----------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n Arrêt manuel sécurisé")