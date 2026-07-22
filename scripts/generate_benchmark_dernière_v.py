"""
Génère un benchmark étendu (cible : ~100 questions, dont une grosse majorité
de VRAIS 2-hop) à partir du graphe LightRAG actuel, sans toucher à l'index.

── Diagnostic (fait avant d'écrire ce script, chiffres réels) ──────────────
1) La génération originale (notebook Sprint 2, cellule 67) scannait les
   nœuds dans un ordre aléatoire et s'arrêtait dès que 2500 triplets A→B→C
   étaient collectés (`if len(two_hop_triplets) >= N_SAMPLES * 10: break`).
   Sur les 250 triplets finalement tirés au sort dans ce pool tronqué,
   seulement 10 étaient de VRAIS 2-hop (source_AB != source_BC).
   -> Un scan EXHAUSTIF (sans ce "break") du graphe actuel trouve en réalité
      49 662 triplets vrais 2-hop disponibles (sur 80 216 chaînes A→B→C
      totales). Le goulot d'étranglement n'était donc pas le graphe
      (trop peu de vrais 2-hop existants) mais l'arrêt prématuré du scan.
      -> Solution : scan exhaustif, pas besoin de "chunk_top_k" ici
         (ce paramètre est un réglage de RETRIEVAL au moment des requêtes
         LightRAG, il n'intervient pas dans cette génération qui est un
         pur parcours NetworkX du graphml, sans appel LightRAG).

2) HotpotQA (data/raw/hotpotqa_validation.json, 7405 questions) porte sur
   des sujets encyclopédiques génériques (films, jeux de société,
   géographie...), pas sur l'IA/le ML. Vérifié empiriquement :
   - correspondance par sous-chaîne dans le corpus arXiv (titre+abstract) :
     43/7405, presque toutes des mots génériques coïncidents (ex: "Blue",
     "Chess", "Friends", "United States") -> PAS de vrai recouvrement
     thématique.
   - correspondance exacte avec un nom d'entité extrait par LightRAG :
     seulement 2/7405 ("Power Grid", "Splendor (board game)" vs
     "Power Grid"), et même ces 2 sont des coïncidences lexicales, pas des
     questions dont la RÉPONSE se trouve dans le corpus arXiv.
   -> Il n'existe PAS 20 questions HotpotQA valides pour ce corpus. Ce
      script tente quand même le matching (au cas où le corpus aurait
      changé) et REPORTE HONNÊTEMENT le nombre trouvé au lieu d'en
      fabriquer 20 artificiellement.

Usage:
    python scripts/generate_benchmark_v2.py --n-true 90
    python scripts/generate_benchmark_v2.py --n-true 5   # test rapide
"""
import argparse
import asyncio
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import os

GRAPHML_PATH        = Path("indexes/lightrag_500_connected_v2/graph_chunk_entity_relation.graphml")
EXISTING_BENCHMARK  = Path("data/processed/arxiv_multihop_v1.json")
HOTPOT_PATH         = Path("data/raw/hotpotqa_validation.json")
ARXIV_CORPUS_PATH   = Path("data/processed/arxiv_cleaned.json")
OUTPUT_CSV          = Path("data/processed/benchmark_derniere_v.csv")
CHECKPOINT_PATH     = Path("data/processed/benchmark_derniere_v_checkpoint.json")

MAX_PER_BRIDGE      = 3     # diversité : pas plus de N questions partageant le même nœud-pont B
RANDOM_SEED         = 42

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama3.1:8b")


# ══════════════════════════════════════════════════════════════
# LLM pour la génération de questions (Groq si dispo, sinon Ollama local)
# ══════════════════════════════════════════════════════════════
async def _groq_generate(prompt: str) -> str:
    from groq import AsyncGroq
    client = AsyncGroq(api_key=GROQ_API_KEY)
    r = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=400,
    )
    return (r.choices[0].message.content or "").strip()


async def _ollama_generate(prompt: str) -> str:
    import httpx
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "options": {"temperature": 0.3}},
        )
        r.raise_for_status()
        return (r.json().get("message", {}).get("content") or "").strip()


async def _llm_generate(prompt: str) -> str:
    if GROQ_API_KEY:
        try:
            return await _groq_generate(prompt)
        except Exception as e:
            print(f"[LLM] Erreur Groq ({e}) — repli sur Ollama local")
    return await _ollama_generate(prompt)


QUESTION_PROMPT = """You are a researcher creating evaluation questions for a RAG system on AI/CS papers.

Given these two connected facts:
FACT 1: {A} → {B} : {desc_AB}
FACT 2: {B} → {C} : {desc_BC}

Generate ONE multi-hop question that:
- Requires knowing BOTH facts to answer
- Has a clear, short answer which is "{C}"
- Is specific and answerable from the facts above
- Does NOT mention "{C}" in the question

Output ONLY this JSON (no explanation, no markdown):
{{"question": "...", "answer": "{C}", "reasoning": "..."}}"""


# ══════════════════════════════════════════════════════════════
# 1. Scan exhaustif des triplets 2-hop VRAIS (cross-chunk)
# ══════════════════════════════════════════════════════════════
def find_true_2hop_triplets(graphml_path: Path):
    print(f"Chargement de {graphml_path} ...")
    G = nx.DiGraph(nx.read_graphml(str(graphml_path)))
    print(f"  {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    triplets = []
    for A in G.nodes():
        for B in G.successors(A):
            edge_AB = G.get_edge_data(A, B)
            src_ab = edge_AB.get("source_id", "")
            for C in G.successors(B):
                if C == A:
                    continue
                edge_BC = G.get_edge_data(B, C)
                src_bc = edge_BC.get("source_id", "")
                if src_ab and src_bc and src_ab != src_bc:  # VRAI 2-hop uniquement
                    triplets.append({
                        "A": A, "B": B, "C": C,
                        "desc_AB": edge_AB.get("description", "")[:300],
                        "desc_BC": edge_BC.get("description", "")[:300],
                        "source_AB": src_ab, "source_BC": src_bc,
                    })
    print(f"  {len(triplets)} triplets VRAIS 2-hop trouvés (scan exhaustif, sans plafond)")
    return triplets


def diversify(triplets: list, max_per_bridge: int, seed: int):
    random.seed(seed)
    random.shuffle(triplets)
    by_bridge = defaultdict(list)
    kept = []
    for t in triplets:
        if len(by_bridge[t["B"]]) < max_per_bridge:
            by_bridge[t["B"]].append(t)
            kept.append(t)
    print(f"  {len(kept)} triplets retenus après diversification "
          f"(max {max_per_bridge} par nœud-pont, {len(by_bridge)} ponts distincts)")
    return kept


def already_used_keys(existing_benchmark_path: Path):
    if not existing_benchmark_path.exists():
        return set()
    items = json.loads(existing_benchmark_path.read_text(encoding="utf-8"))
    keys = set()
    for ex in items:
        h1, h2 = ex.get("hop1", {}), ex.get("hop2", {})
        keys.add((h1.get("entity_A", ""), h1.get("entity_B", ""), h2.get("entity_C", "")))
    return keys


# ══════════════════════════════════════════════════════════════
# 2. Génération LLM des questions (checkpointée)
# ══════════════════════════════════════════════════════════════
async def generate_questions(candidates: list, n_target: int, checkpoint_path: Path):
    if checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        generated, start_index = ckpt["generated"], ckpt["last_index"] + 1
        print(f"  Reprise depuis l'index {start_index} ({len(generated)} déjà générées)")
    else:
        generated, start_index = [], 0

    i = start_index
    while len(generated) < n_target and i < len(candidates):
        t = candidates[i]
        prompt = QUESTION_PROMPT.format(A=t["A"], B=t["B"], C=t["C"],
                                         desc_AB=t["desc_AB"], desc_BC=t["desc_BC"])
        try:
            raw = await _llm_generate(prompt)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group()) if m else {}
            if data.get("question") and data.get("answer"):
                generated.append({
                    "question": data["question"],
                    "answer": data["answer"],
                    "ground_truth": data["answer"],
                    "reasoning": data.get("reasoning", ""),
                    "hop1": {"entity_A": t["A"], "entity_B": t["B"], "description": t["desc_AB"],
                              "chunk_id": t["source_AB"]},
                    "hop2": {"entity_B": t["B"], "entity_C": t["C"], "description": t["desc_BC"],
                              "chunk_id": t["source_BC"]},
                    "supporting_chunks": [t["source_AB"], t["source_BC"]],
                    "hop_type": "true_2hop",
                    "source": "generated_v2",
                })
                print(f"  [{len(generated)}/{n_target}] OK : {data['question'][:80]}")
            else:
                print(f"  [skip] JSON invalide pour triplet {t['A']} -> {t['B']} -> {t['C']}")
        except Exception as e:
            print(f"  [erreur] {e}")

        i += 1
        if i % 5 == 0:
            checkpoint_path.write_text(
                json.dumps({"generated": generated, "last_index": i - 1}, ensure_ascii=False),
                encoding="utf-8",
            )

    checkpoint_path.write_text(
        json.dumps({"generated": generated, "last_index": i - 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    return generated


# ══════════════════════════════════════════════════════════════
# 3. Matching HotpotQA <-> corpus arXiv (honnête, pas forcé à 20)
# ══════════════════════════════════════════════════════════════
def match_hotpotqa(hotpot_path: Path, arxiv_corpus_path: Path, graphml_path: Path, max_n: int):
    hotpot = json.loads(hotpot_path.read_text(encoding="utf-8"))
    arxiv = json.loads(arxiv_corpus_path.read_text(encoding="utf-8"))
    corpus_text = " ".join((d.get("title", "") + " " + d.get("abstract", "")) for d in arxiv).lower()

    G = nx.read_graphml(str(graphml_path))
    entity_names_lower = {n.lower() for n in G.nodes()}

    matches = []
    for q in hotpot:
        titles = q.get("supporting_facts", {}).get("title", [])
        titles = [t for t in titles if len(t.strip()) >= 5]  # écarte les titres trop courts/génériques
        entity_hits = [t for t in titles if t.lower().strip() in entity_names_lower]
        if entity_hits:
            matches.append({
                "question": q["question"],
                "answer": q["answer"],
                "ground_truth": q["answer"],
                "hop_type": "hotpotqa_crossdomain",
                "source": "hotpotqa_validation",
                "matched_entities": entity_hits,
                "supporting_chunks": ["hotpotqa", "hotpotqa"],
            })

    print(f"\n=== Matching HotpotQA <-> corpus arXiv ===")
    print(f"  {len(hotpot)} questions HotpotQA analysées")
    print(f"  {len(matches)} question(s) avec une entité EXACTEMENT présente dans le graphe LightRAG")
    if len(matches) < max_n:
        print(f"  ATTENTION : {max_n} demandées, seulement {len(matches)} trouvées de façon honnête.")
        print(f"  Cause : HotpotQA (sujets encyclopédiques génériques) et le corpus arXiv (papiers IA/ML)")
        print(f"  n'ont quasiment aucun recouvrement thématique — vérifié, pas supposé (cf. docstring).")
        print(f"  Les {len(matches)} trouvées restent des coïncidences lexicales, pas un vrai recouvrement")
        print(f"  de sujet : à vérifier une par une avant de les garder dans un benchmark scientifique.")
    return matches[:max_n]


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
async def main(n_true: int, n_hotpot: int):
    print("=" * 70)
    print("ÉTAPE 1 — Triplets vrais 2-hop (scan exhaustif du graphe)")
    print("=" * 70)
    all_triplets = find_true_2hop_triplets(GRAPHML_PATH)
    used_keys = already_used_keys(EXISTING_BENCHMARK)
    fresh_triplets = [t for t in all_triplets if (t["A"], t["B"], t["C"]) not in used_keys]
    print(f"  {len(fresh_triplets)} triplets non déjà présents dans {EXISTING_BENCHMARK.name}")
    candidates = diversify(fresh_triplets, MAX_PER_BRIDGE, RANDOM_SEED)

    print("\n" + "=" * 70)
    print(f"ÉTAPE 2 — Génération LLM de {n_true} nouvelles questions vrais 2-hop")
    print("=" * 70)
    new_questions = await generate_questions(candidates, n_true, CHECKPOINT_PATH)

    print("\n" + "=" * 70)
    print("ÉTAPE 3 — Questions vrais 2-hop déjà existantes (réutilisées telles quelles)")
    print("=" * 70)
    existing = json.loads(EXISTING_BENCHMARK.read_text(encoding="utf-8")) if EXISTING_BENCHMARK.exists() else []
    existing_true = [
        {**ex, "hop_type": "true_2hop", "source": "original_v1"}
        for ex in existing
        if ex["supporting_chunks"][0] != ex["supporting_chunks"][1]
    ]
    print(f"  {len(existing_true)} questions vrais 2-hop reprises de {EXISTING_BENCHMARK.name}")

    hotpot_rows = match_hotpotqa(HOTPOT_PATH, ARXIV_CORPUS_PATH, GRAPHML_PATH, n_hotpot)

    all_rows = existing_true + new_questions + hotpot_rows

    df = pd.DataFrame([{
        "question": r["question"],
        "ground_truth": r["ground_truth"],
        "hop_type": r["hop_type"],
        "source": r["source"],
        "supporting_chunks": json.dumps(r.get("supporting_chunks", [])),
        "reasoning": r.get("reasoning", ""),
    } for r in all_rows])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)
    print(df["hop_type"].value_counts().to_string())
    print(f"\nTotal : {len(df)} questions")
    print(f"✓ {OUTPUT_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-true", type=int, default=90,
                     help="Nombre de NOUVELLES questions vrais 2-hop à générer via LLM")
    ap.add_argument("--n-hotpot", type=int, default=20,
                     help="Nombre max de questions HotpotQA à inclure si trouvées (plafond, pas garanti)")
    args = ap.parse_args()
    asyncio.run(main(args.n_true, args.n_hotpot))
