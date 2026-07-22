"""
Étape 3 — Injecte des arêtes de co-occurrence pour désenclaver les nœuds
à degré 1 (62.8% du graphe), SANS aucun appel LLM d'extraction et SANS
réindexation.

Constat exact sur ce projet (indexes/lightrag_500_connected_v2) :
- kv_store_full_entities.json montre qu'un chunk donne en général 5-8
  entités extraites (ex: doc 2604.24758v1 -> 7 entités)
- kv_store_full_relations.json montre que ce même chunk ne produit souvent
  que 1-2 relations LLM
-> le LLM d'extraction énumère des entités mais ne les relie pas toutes
   entre elles : la majorité des entités d'un chunk restent orphelines
   du sous-graphe local, même si elles apparaissent dans le même passage.

Ce script ajoute, pour chaque chunk, des relations "co-occurs_with" entre
les entités qui partagent ce chunk mais ne sont reliées par AUCUN chemin
direct dans le graphe actuel. Ces relations sont marquées explicitement
(keywords="co_occurrence", weight=0.3, plus faible que les relations
sémantiques du LLM à weight=1.0) pour que le retrieval / reranking puisse
toujours distinguer une vraie relation sémantique d'un simple indice de
co-occurrence structurel.

Pour éviter une explosion combinatoire (un chunk à 8 entités a C(8,2)=28
paires possibles), on ne relie pas toutes les paires : chaque entité SANS
relation existante vers le "hub" du chunk (les entités qui participent
déjà à une relation LLM issue de ce chunk) est connectée au hub le plus
proche (au plus MAX_EDGES_PER_ENTITY nouvelles arêtes par entité).

Usage:
    python scripts/add_cooccurrence_edges.py --dry-run     # compte sans rien écrire
    python scripts/add_cooccurrence_edges.py                # applique
    python scripts/add_cooccurrence_edges.py --max-new-edges 2000  # limite le run
"""
import argparse
import asyncio
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import networkx as nx
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

INDEX_DIR = Path(os.getenv("INDEX_DIR", "indexes/lightrag_500_connected_v2"))
GRAPHML = INDEX_DIR / "graph_chunk_entity_relation.graphml"
ENTITY_CHUNKS_KV = INDEX_DIR / "kv_store_entity_chunks.json"
FULL_RELATIONS_KV = INDEX_DIR / "kv_store_full_relations.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

MAX_EDGES_PER_ENTITY = 2   # nb max de nouvelles arêtes ajoutées par entité orpheline
COOC_WEIGHT = 0.3          # < 1.0 (poids par défaut des relations LLM) exprès


async def embedding_func(texts):
    import httpx
    import numpy as np
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for text in texts:
            r = await client.post(f"{OLLAMA_URL}/api/embeddings",
                                   json={"model": EMBED_MODEL, "prompt": text})
            results.append(r.json()["embedding"])
    return np.vstack(results)


def build_chunk_to_entities() -> dict:
    """Inverse kv_store_entity_chunks.json : entity -> chunk_ids
    en chunk_id -> {entities}."""
    d = json.loads(ENTITY_CHUNKS_KV.read_text(encoding="utf-8"))
    chunk_to_entities = defaultdict(set)
    for entity_name, meta in d.items():
        for cid in meta.get("chunk_ids", []):
            chunk_to_entities[cid].add(entity_name)
    return chunk_to_entities


def build_chunk_hubs() -> dict:
    """chunk_id -> {entités qui participent déjà à >=1 relation LLM issue
    de ce chunk} — ce sont les entités 'bien connectées' localement,
    candidates prioritaires pour recevoir les nouvelles arêtes."""
    d = json.loads(FULL_RELATIONS_KV.read_text(encoding="utf-8"))
    hubs = defaultdict(set)
    for _key, rel in d.items():
        cid = rel.get("source_id")
        if cid:
            hubs[cid].add(rel["src_id"])
            hubs[cid].add(rel["tgt_id"])
    return hubs


def plan_new_edges(G: nx.Graph, chunk_to_entities: dict, chunk_hubs: dict):
    """Retourne la liste des (src, tgt, chunk_id) à créer, sans dépasser
    MAX_EDGES_PER_ENTITY nouvelles arêtes par entité orpheline."""
    new_edges = []
    seen_pairs = set()
    added_count = defaultdict(int)

    for cid, entities in chunk_to_entities.items():
        entities = [e for e in entities if e in G]  # ignore si fusionné/supprimé
        if len(entities) < 2:
            continue

        hubs = [e for e in chunk_hubs.get(cid, set()) if e in entities]
        if not hubs:
            hubs = sorted(entities)[:1]  # fallback déterministe

        for entity in entities:
            if added_count[entity] >= MAX_EDGES_PER_ENTITY:
                continue
            # entité déjà reliée à au moins un hub de ce chunk -> rien à faire
            if any(G.has_edge(entity, h) or G.has_edge(h, entity) for h in hubs if h != entity):
                continue
            for hub in hubs:
                if hub == entity:
                    continue
                pair = tuple(sorted([entity, hub]))
                if pair in seen_pairs:
                    continue
                if G.has_edge(entity, hub) or G.has_edge(hub, entity):
                    continue
                if added_count[entity] >= MAX_EDGES_PER_ENTITY:
                    break
                new_edges.append((entity, hub, cid))
                seen_pairs.add(pair)
                added_count[entity] += 1
                added_count[hub] += 1

    return new_edges


def backup_index(index_dir: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = index_dir.parent / f"{index_dir.name}_backup_before_cooc_{ts}"
    print(f"[BACKUP] {index_dir} -> {backup_dir}")
    shutil.copytree(index_dir, backup_dir)
    return backup_dir


async def main(dry_run: bool, max_new_edges: int, concurrency: int):
    print("Chargement du graphe et des index...")
    G = nx.read_graphml(GRAPHML)
    chunk_to_entities = build_chunk_to_entities()
    chunk_hubs = build_chunk_hubs()
    print(f"  {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes actuelles")
    print(f"  {len(chunk_to_entities)} chunks indexés")

    new_edges = plan_new_edges(G, chunk_to_entities, chunk_hubs)
    if max_new_edges:
        new_edges = new_edges[:max_new_edges]

    n_entities_touched = len({e for pair in new_edges for e in pair[:2]})
    print(f"\n{len(new_edges)} nouvelles arêtes de co-occurrence planifiées "
          f"({n_entities_touched} entités désenclavées)")

    if dry_run:
        for src, tgt, cid in new_edges[:20]:
            print(f"  [DRY-RUN] {src}  <-co_occurs_with->  {tgt}   (chunk={cid[:16]}...)")
        if len(new_edges) > 20:
            print(f"  ... et {len(new_edges) - 20} de plus")
        return

    if not new_edges:
        print("Rien à ajouter.")
        return

    backup_index(INDEX_DIR)

    async def dummy_llm(*a, **kw):
        return ""

    rag = LightRAG(
        working_dir=str(INDEX_DIR),
        llm_model_func=dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM, max_token_size=8192,
            model_name=EMBED_MODEL, func=embedding_func,
        ),
        llm_model_max_async=1,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()

    sem = asyncio.Semaphore(concurrency)
    ok, skipped = 0, 0

    async def _create(src, tgt, cid):
        nonlocal ok, skipped
        async with sem:
            try:
                await rag.acreate_relation(src, tgt, {
                    "description": (
                        f"'{src}' and '{tgt}' are mentioned in the same source "
                        f"passage (automatically inferred co-occurrence, not "
                        f"LLM-extracted)."
                    ),
                    "keywords": "co_occurrence",
                    "weight": COOC_WEIGHT,
                    "source_id": cid,
                })
                ok += 1
            except ValueError:
                skipped += 1  # arête déjà créée entre-temps (paire symétrique traitée 2x)
            except Exception as e:
                print(f"[ERREUR] {src} <-> {tgt} : {e}")

    print(f"\nCréation des arêtes (concurrency={concurrency})...")
    await asyncio.gather(*(_create(s, t, c) for s, t, c in new_edges))

    print(f"\n{ok} arêtes créées, {skipped} déjà existantes (ignorées).")
    await rag.finalize_storages()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-new-edges", type=int, default=0, help="0 = pas de limite")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    asyncio.run(main(args.dry_run, args.max_new_edges, args.concurrency))
