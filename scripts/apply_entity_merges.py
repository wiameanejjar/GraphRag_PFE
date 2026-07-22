"""
Étape 2 — Applique les clusters de doublons validés (data/processed/entity_merge_candidates.csv,
généré par scripts/resolve_entities_embeddings.py et relu à la main) au graphe LightRAG existant.

Utilise l'API officielle LightRAG `amerge_entities`, qui met à jour ATOMIQUEMENT :
- le graphe (graph_chunk_entity_relation.graphml) : réattache toutes les relations
  des entités sources vers l'entité cible, fusionne les descriptions
- la base vectorielle des entités (vdb_entities.json)
- la base vectorielle des relations (vdb_relationships.json)

AUCUN appel LLM d'extraction, AUCUNE réindexation du corpus : uniquement de la
réécriture de graphe. (Une fusion peut déclencher un unique appel LLM interne à
LightRAG si >8 fragments de description doivent être résumés — cf.
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE dans lightrag/constants.py — d'où la présence
d'un llm_model_func ci-dessous, jamais utilisé pour de l'extraction.)

Prérequis : avoir relu data/processed/entity_merge_candidates.csv et mis
keep_this_cluster à 0 pour tout cluster qui n'est pas un vrai doublon.

Usage:
    python scripts/apply_entity_merges.py                  # exécute les merges
    python scripts/apply_entity_merges.py --dry-run         # affiche sans rien modifier
"""
import argparse
import asyncio
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

INDEX_DIR = Path(os.getenv("INDEX_DIR", "indexes/lightrag_500_connected_v2"))
CANDIDATES_CSV = Path("data/processed/entity_merge_candidates.csv")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")


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


async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """Utilisé UNIQUEMENT si LightRAG doit résumer une description fusionnée
    trop longue (>8 fragments) — jamais pour de l'extraction d'entités."""
    import httpx
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in (history_messages or []):
        if isinstance(m, dict):
            messages.append(m)
    messages.append({"role": "user", "content": prompt})
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat",
                               json={"model": MODEL_NAME, "messages": messages, "stream": False})
        return r.json().get("message", {}).get("content", "")


def backup_index(index_dir: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = index_dir.parent / f"{index_dir.name}_backup_before_merge_{ts}"
    print(f"[BACKUP] {index_dir} -> {backup_dir}")
    shutil.copytree(index_dir, backup_dir)
    return backup_dir


def load_approved_clusters(csv_path: Path):
    clusters = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("keep_this_cluster(1/0)", "1").strip() not in ("1", "1.0", "true", "True"):
                continue
            members = [m.strip() for m in row["members"].split(";") if m.strip()]
            canonical = row["canonical_suggested"].strip()
            if canonical not in members:
                members.append(canonical)
            if len(members) < 2:
                continue
            clusters.append((canonical, members))
    return clusters


async def main(dry_run: bool):
    if not CANDIDATES_CSV.exists():
        raise SystemExit(f"{CANDIDATES_CSV} introuvable — lancer d'abord "
                          f"scripts/resolve_entities_embeddings.py")

    clusters = load_approved_clusters(CANDIDATES_CSV)
    print(f"{len(clusters)} clusters approuvés à fusionner (colonne keep_this_cluster=1)")
    if not clusters:
        print("Rien à faire.")
        return

    if dry_run:
        for canonical, members in clusters:
            sources = [m for m in members if m != canonical]
            print(f"  [DRY-RUN] {sources} -> '{canonical}'")
        return

    backup_index(INDEX_DIR)

    rag = LightRAG(
        working_dir=str(INDEX_DIR),
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM, max_token_size=8192,
            model_name=EMBED_MODEL, func=embedding_func,
        ),
        llm_model_max_async=1,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()

    ok, failed = 0, []
    for canonical, members in clusters:
        sources = [m for m in members if m != canonical]
        try:
            await rag.amerge_entities(
                source_entities=sources,
                target_entity=canonical,
                merge_strategy={
                    "description": "concatenate",
                    "entity_type": "keep_first",
                },
            )
            print(f"[MERGED] {sources} -> '{canonical}'")
            ok += 1
        except Exception as e:
            print(f"[ERREUR] {sources} -> '{canonical}' : {e}")
            failed.append((canonical, members, str(e)))

    print(f"\n{ok}/{len(clusters)} clusters fusionnés avec succès.")
    if failed:
        print(f"{len(failed)} échecs (voir ci-dessus) — le reste du graphe n'est pas affecté.")

    await rag.finalize_storages()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main(args.dry_run))
