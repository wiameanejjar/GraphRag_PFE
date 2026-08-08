# compare_agent_versions.py
"""
Comparaison des 3 versions de l'agent Agentic GraphRAG (graph_v1, graph_v2,
graph_v3) sur le meme benchmark et les memes questions.

- v1 (Sprint 3)      : QUERY -> GRAPH_SEARCH (Neo4j, Cypher brut) + VECTOR_SEARCH
                        (Chroma) -> FUSE -> RESPONSE -> CRITIQUE -> SELF_CORRECT.
                        Juge = meme modele que le generateur (pas d'independance).
- v2 (Sprint 3, corrige) : meme architecture que v1, avec gestion d'erreurs
                        Neo4j/Chroma, deduplication, contexte plus cible.
                        Juge toujours non independant du generateur.
- v3 (Sprint 4, actuel)  : retrieval hybride natif LightRAG (plus de Neo4j/Chroma
                        separes), juge INDEPENDANT du generateur, verification
                        obligatoire des refus, suivi de la meilleure reponse,
                        rotation Groq multi-comptes.

Important (a documenter comme limite methodologique) : v1/v2 interrogent le
graphe Neo4j ANCIEN (5065 entites/5068 relations, avant le post-traitement du
22/07 qui a fusionne les doublons et ajoute des aretes de co-occurrence),
tandis que v3 interroge l'index LightRAG post-traite (4945 noeuds/5593
relations). La comparaison mesure donc a la fois l'evolution architecturale
ET l'evolution du graphe sous-jacent, pas uniquement l'architecture isolee
(contrairement a l'ablation study RAG/LightRAG/Agentic qui isole strictement
la boucle agentique).

v3 n'est PAS relance : ses reponses sur les memes 30 questions (meme
benchmark, meme seed=42) existent deja dans Eval_agentic/plan_c_checkpoint_30q.json
et sont reutilisees telles quelles.

Ce script se contente de GENERER les reponses + contextes des 3 versions.
La notation est faite separement, avec RAGAS, par
evaluate_agent_versions_ragas.py (aucune grille manuelle ici).
"""
import sys

# Les logs des 3 agents utilisent des caracteres unicode (fleches "->" etc.).
# Sur Windows, stdout redirige vers un fichier retombe sur cp1252 (au lieu
# d'UTF-8), qui ne sait pas les encoder -> crash silencieux du print. Force
# UTF-8 explicitement, avant tout import qui pourrait imprimer quelque chose.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import hashlib
import json
import os
import random
import time
from pathlib import Path

import pandas as pd

from src.utils.groq_rotation import AllGroqKeysExhaustedError, rotation_status

# ── Config ────────────────────────────────────────────────────
BENCHMARK_PATH = Path("data/processed/benchmark_true_multihop.json")
N_QUESTIONS = int(os.getenv("EVAL_N_QUESTIONS", "30"))
SEED = 42

EVAL_ROOT = Path("Eval_agentic")            # etat runtime (checkpoints, non versionne)
OUTPUT_DIR = EVAL_ROOT / "versions_agent"   # livrables de la comparaison des versions
CHECKPOINT_PATH = EVAL_ROOT / "agent_versions_checkpoint.json"
DETAILS_CSV = OUTPUT_DIR / "agent_versions_comparaison_details.csv"
AGG_CSV = OUTPUT_DIR / "agent_versions_comparaison_resultats.csv"
PARAMS_CSV = OUTPUT_DIR / "agent_versions_parametres.csv"

V3_CHECKPOINT_PATH = EVAL_ROOT / "plan_c_checkpoint_30q.json"

VERSIONS_TO_RUN = ["Agentic v1 (Sprint 3)", "Agentic v2 (Sprint 3, corrige)"]
V3_LABEL = "Agentic v3 (Sprint 4, actuel)"
ALL_LABELS = VERSIONS_TO_RUN + [V3_LABEL]


def qkey(question: str) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()[:12]


def load_benchmark_sample() -> list:
    with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        all_items = json.load(f)
    random.seed(SEED)
    return random.sample(all_items, min(N_QUESTIONS, len(all_items)))


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(cp: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2, ensure_ascii=False), encoding="utf-8")


def build_context_list(result: dict) -> list:
    """Reconstruit une liste de passages a partir de l'etat retourne par
    graph_v1/graph_v2 (memes champs : vector_results, fused_context)."""
    vector_chunks = result.get("vector_results", []) or []
    # Attention : troncature a 1200 caracteres alors que fused_context fait
    # ~5900 caracteres pour v1/v2. Le contexte archive est donc partiel ; pour
    # juger si une reponse cite une source qu elle n a pas recuperee, il faut
    # rejouer la question et inspecter fused_context en entier (c est ce qui a
    # ete fait pour verifier les citations non ancrees de v1).
    graph_context = (result.get("fused_context", "") or "")[:1200]
    contexts = []
    if vector_chunks:
        contexts.extend(vector_chunks[:4])
    if graph_context:
        contexts.append(graph_context)
    return contexts if contexts else ["No context retrieved."]


def run_version(label: str, run_agent_fn, items: list, checkpoint: dict) -> bool:
    """Execute run_agent_fn sur chaque item, avec reprise sur checkpoint et
    arret propre si les 3 cles Groq sont epuisees. Retourne True si complet."""
    checkpoint.setdefault(label, {})
    n_done = len(checkpoint[label])
    print(f"\n=== {label} : {len(items)} questions ({n_done} deja en checkpoint) ===")
    print(f"[GROQ ROTATION] {rotation_status()}")

    for i, ex in enumerate(items, 1):
        qk = qkey(ex["question"])
        if qk in checkpoint[label]:
            print(f"[{i}/{len(items)}] (checkpoint) {ex['question'][:70]}")
            continue

        print(f"[{i}/{len(items)}] {ex['question'][:70]}")
        t0 = time.time()
        try:
            result = run_agent_fn(ex["question"])
        except AllGroqKeysExhaustedError as e:
            print(f"\n Les 3 cles Groq configurees sont epuisees.")
            print(f"   {e}")
            print(f"   Arret propre a la question {i}/{len(items)} de « {label} ».")
            print(f"   -> Relancez ce script plus tard : il reprendra a partir d'ici.")
            return False
        latency = round(time.time() - t0, 2)

        checkpoint[label][qk] = {
            "question": ex["question"],
            "ground_truth": ex.get("ground_truth", ""),
            "answer": result.get("final_response", ""),
            "contexts": build_context_list(result),
            "critique_score": result.get("critique_score", 0.0),
            "iterations": result.get("iteration", 0),
            "latency_s": latency,
        }
        save_checkpoint(checkpoint)
        print(f"  -> score={result.get('critique_score', 0):.2f} | "
              f"iter={result.get('iteration', 0)} ({latency}s)")

    print(f"OK {label} : {len(checkpoint[label])}/{len(items)} questions completes.")
    return True


def load_v3_reused(items: list) -> dict:
    """Reutilise les reponses v3 deja calculees (meme benchmark/seed/N),
    sans relancer l'agent ni consommer de quota."""
    entries = {}
    if not V3_CHECKPOINT_PATH.exists():
        print(f"ATTENTION : {V3_CHECKPOINT_PATH} introuvable -> v3 ne sera pas inclus.")
        return entries

    raw = json.loads(V3_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    agentic_raw = raw.get("Agentic GraphRAG", {})

    for ex in items:
        qk = qkey(ex["question"])
        if qk not in agentic_raw:
            continue
        e = agentic_raw[qk]
        entries[qk] = {
            "question": e["question"],
            "ground_truth": e["ground_truth"],
            "answer": e["answer"],
            "contexts": e.get("contexts", []),
            "critique_score": e.get("critique_score"),
            "iterations": e.get("iterations"),
            "latency_s": e.get("latency_s"),
        }
    return entries


def main():
    items = load_benchmark_sample()
    print(f"Benchmark : {BENCHMARK_PATH} | {N_QUESTIONS} questions (seed={SEED})")

    checkpoint = load_checkpoint()

    # v3 : reutilise, pas de nouvel appel LLM
    v3_entries = load_v3_reused(items)
    checkpoint[V3_LABEL] = v3_entries
    save_checkpoint(checkpoint)
    print(f"\n=== {V3_LABEL} : {len(v3_entries)}/{len(items)} reponses reutilisees "
          f"depuis {V3_CHECKPOINT_PATH.name} ===")

    # v1 et v2 : execution reelle
    from src.agent.graph_v1 import run_agent as run_agent_v1
    complete_v1 = run_version(VERSIONS_TO_RUN[0], run_agent_v1, items, checkpoint)

    from src.agent.graph_v2 import run_agent as run_agent_v2
    complete_v2 = run_version(VERSIONS_TO_RUN[1], run_agent_v2, items, checkpoint)

    # ── Export CSV detaille (reponses brutes ; la notation est faite par
    #    evaluate_agent_versions_ragas.py) ──────────────────────────
    rows = []
    for label in ALL_LABELS:
        for ex in items:
            qk = qkey(ex["question"])
            e = checkpoint.get(label, {}).get(qk)
            if e is None:
                continue
            contexts = e.get("contexts", [])
            rows.append({
                "system": label,
                "question": e["question"],
                "ground_truth": e["ground_truth"],
                "answer": e["answer"],
                "context_excerpt": (contexts[0][:500] if contexts else ""),
                "critique_score": e.get("critique_score"),
                "iterations": e.get("iterations"),
                "latency_s": e.get("latency_s"),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(DETAILS_CSV, index=False)
    print(f"\n{DETAILS_CSV} : {len(df)} lignes")

    if not (complete_v1 and complete_v2):
        print("\nATTENTION : v1 et/ou v2 incomplets (quota Groq epuise) -> "
              "relancer ce script plus tard pour terminer avant de noter.")
    else:
        print("\nEtape suivante : python evaluate_agent_versions_ragas.py")


if __name__ == "__main__":
    main()
