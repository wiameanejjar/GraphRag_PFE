import json
import os
import tiktoken

CHUNK_SIZE = 512
OVERLAP = 50

print("Chargement du tokenizer...")
encoder = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, doc_id):
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = encoder.decode(chunk_tokens)
        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
            "doc_id": doc_id,
            "text": chunk_text_str,
            "token_count": len(chunk_tokens),
            "start_token": start,
            "end_token": end
        })
        start += CHUNK_SIZE - OVERLAP
        chunk_idx += 1
    return chunks

print("Chargement des articles nettoyés...")
with open("data/cleaned/arxiv_cleaned.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Chunking de {len(papers)} articles...")
os.makedirs("data/chunks", exist_ok=True)

all_chunks = []
for i, paper in enumerate(papers):
    chunks = chunk_text(paper["full_text"], str(i))
    all_chunks.extend(chunks)
    if (i + 1) % 100 == 0:
        print(f"  Traité : {i + 1}/{len(papers)}")

with open("data/chunks/arxiv_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

avg_tokens = sum(c["token_count"] for c in all_chunks) // len(all_chunks)
print(f"\nRésultat :")
print(f"  Total chunks     : {len(all_chunks)}")
print(f"  Tokens moyens    : {avg_tokens}")
print(f"  Sauvegardé dans  : data/chunks/arxiv_chunks.json")