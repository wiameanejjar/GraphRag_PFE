"""
Validation de la qualité multi-hop d'un benchmark — SANS jamais se fier au
champ 'hop_type' (ou tout champ d'étiquette pré-existant), uniquement sur le
contenu réel de chaque exemple : question, ground_truth, reasoning, hop1/hop2
(entités + descriptions + chunk_id), supporting_chunks, et supporting_documents
si résolvables.

── Critères (voir explication complète donnée à l'utilisateur avant ce script) ──
Un exemple est TRUE_MULTI_HOP seulement s'il passe TOUT :
  1. entity_A (départ de hop1) est référencé dans la question
     -> sinon hop1 est inutile, la question est répondable via hop2 seul.
  2. ground_truth n'apparaît pas déjà dans la question (pas de fuite de réponse)
  3. hop1.description et hop2.description ne sont pas quasi-identiques
     (similarité de Jaccard < seuil) -> sinon ce ne sont pas 2 faits distincts
  4. supporting_chunks[0] != supporting_chunks[1] (chunks réellement distincts)
     -> sinon un seul passage récupéré suffit (pas un test de retrieval multi-hop)

Échec de 1, 2 ou 3   -> SINGLE_HOP
Échec de 4 seulement -> PSEUDO_MULTI_HOP
Tout passe           -> TRUE_MULTI_HOP

Réutilisable sur n'importe quel autre benchmark ayant le même schéma
(question, ground_truth, hop1{entity_A, entity_B, description, chunk_id},
hop2{entity_B, entity_C, description, chunk_id}, supporting_chunks).

Usage:
    python scripts/validate_multihop_benchmark.py
    python scripts/validate_multihop_benchmark.py --input mon_benchmark.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Sources par défaut : fusion des 250 questions d'origine + les 90
# nouvelles générées (checkpoint), reconstituant le pool complet de
# candidats 2-hop tel qu'il existe aujourd'hui, SANS présupposer lesquelles
# sont "vraies" (le hop_type de ces fichiers est ignoré, jamais lu).
DEFAULT_SOURCES = [
    "data/processed/arxiv_multihop_v1.json",
    "data/processed/benchmark_derniere_v_checkpoint.json",  # clé "generated"
]
DEFAULT_TEXT_CHUNKS_KV = "indexes/lightrag_500_connected_v2/kv_store_text_chunks.json"
DEFAULT_OUTPUT     = "data/processed/benchmark_true_multihop.json"
DEFAULT_AUDIT_CSV  = "data/processed/benchmark_multihop_audit.csv"
DEFAULT_FIGURES_DIR = "figures"

JACCARD_REDUNDANCY_THRESHOLD = 0.75  # au-dessus : hop1/hop2 jugés redondants
STOPWORDS = {
    "a", "an", "the", "of", "for", "in", "on", "with", "and", "or", "is",
    "are", "to", "by", "this", "that", "as", "its", "it", "be", "was", "were",
}


# ══════════════════════════════════════════════════════════════
# Chargement / fusion des sources (schéma commun : question, ground_truth,
# reasoning, hop1, hop2, supporting_chunks — hop_type/source ignorés)
# ══════════════════════════════════════════════════════════════
def _extract_records(raw) -> list:
    """Gère les deux formats rencontrés : liste brute, ou
    {"generated": [...], "last_index": ...} (checkpoint)."""
    if isinstance(raw, dict) and "generated" in raw:
        return raw["generated"]
    if isinstance(raw, list):
        return raw
    return []


def load_and_merge(source_paths: list) -> list:
    merged = []
    seen_keys = set()
    for path_str in source_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"  [skip] {path} introuvable")
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = _extract_records(raw)
        added = 0
        for ex in records:
            h1, h2 = ex.get("hop1", {}), ex.get("hop2", {})
            key = (h1.get("entity_A", ""), h1.get("entity_B", ""), h2.get("entity_C", ""),
                   ex.get("question", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            # Copie SANS les champs d'étiquette pré-existants (hop_type, source)
            # pour garantir que le classifieur ne peut pas s'appuyer dessus.
            clean = {k: v for k, v in ex.items() if k not in ("hop_type", "source")}
            merged.append(clean)
            added += 1
        print(f"  {path} : {added} exemples ajoutés ({len(records)} lus)")
    print(f"Total fusionné (dédupliqué) : {len(merged)} exemples\n")
    return merged


# ══════════════════════════════════════════════════════════════
# Résolution chunk -> document (optionnelle)
# ══════════════════════════════════════════════════════════════
def load_chunk_to_doc(text_chunks_kv_path: str):
    path = Path(text_chunks_kv_path)
    if not path.exists():
        print(f"  (info) {path} introuvable -> pas de résolution document, "
              f"seule la distinction au niveau chunk sera utilisée.\n")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {cid: meta.get("file_path", "") for cid, meta in data.items()}


# ══════════════════════════════════════════════════════════════
# Helpers texte
# ══════════════════════════════════════════════════════════════
def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set:
    return {t for t in _normalize(text).split() if t not in STOPWORDS and len(t) > 2}


def entity_mentioned_in_question(entity: str, question: str) -> bool:
    """Vrai si l'entité (ou une forme très proche) apparaît dans la question."""
    norm_entity = _normalize(entity)
    norm_question = _normalize(question)
    if not norm_entity:
        return False
    if norm_entity in norm_question:
        return True
    ent_tokens = _tokens(entity)
    if not ent_tokens:
        return False
    q_tokens = _tokens(question)
    overlap = len(ent_tokens & q_tokens) / len(ent_tokens)
    return overlap >= 0.6


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ══════════════════════════════════════════════════════════════
# Classification (le cœur : contenu réel, jamais hop_type)
# ══════════════════════════════════════════════════════════════
def classify(example: dict, chunk_to_doc: dict) -> dict:
    question = example.get("question", "")
    ground_truth = str(example.get("ground_truth", example.get("answer", "")))
    hop1 = example.get("hop1", {})
    hop2 = example.get("hop2", {})
    supporting_chunks = example.get("supporting_chunks", [])

    if not hop1 or not hop2:
        return {
            "label": "SINGLE_HOP",
            "reason": "hop1/hop2 absents : format non decomposable, classe par defaut",
            "entity_A_mentioned": None, "answer_leaked": None,
            "hops_redundant": None, "chunks_distinct": None, "docs_distinct": None,
        }

    entity_A = hop1.get("entity_A", "")
    entity_A_mentioned = entity_mentioned_in_question(entity_A, question)
    answer_leaked = _normalize(ground_truth) != "" and _normalize(ground_truth) in _normalize(question)
    hops_redundant = jaccard_similarity(hop1.get("description", ""), hop2.get("description", "")) \
        >= JACCARD_REDUNDANCY_THRESHOLD

    chunks_distinct = (
        len(supporting_chunks) >= 2 and supporting_chunks[0] != supporting_chunks[1]
    )

    docs_distinct = None
    if chunk_to_doc and len(supporting_chunks) >= 2:
        doc_a = chunk_to_doc.get(supporting_chunks[0])
        doc_b = chunk_to_doc.get(supporting_chunks[1])
        if doc_a and doc_b:
            docs_distinct = doc_a != doc_b

    if not entity_A_mentioned:
        label, reason = "SINGLE_HOP", "entity_A absente de la question -> hop1 inutile"
    elif answer_leaked:
        label, reason = "SINGLE_HOP", "ground_truth deja present dans la question"
    elif hops_redundant:
        label, reason = "SINGLE_HOP", "hop1 et hop2 quasi-identiques -> pas 2 faits distincts"
    elif not chunks_distinct:
        label, reason = "PSEUDO_MULTI_HOP", "2 faits reels mais meme chunk source"
    else:
        label, reason = "TRUE_MULTI_HOP", "entity_A referencee, faits distincts, chunks distincts"

    return {
        "label": label, "reason": reason,
        "entity_A_mentioned": entity_A_mentioned, "answer_leaked": answer_leaked,
        "hops_redundant": hops_redundant, "chunks_distinct": chunks_distinct,
        "docs_distinct": docs_distinct,
    }


# ══════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════
def make_visualizations(audit_df: pd.DataFrame, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    order = ["TRUE_MULTI_HOP", "PSEUDO_MULTI_HOP", "SINGLE_HOP"]
    colors = {"TRUE_MULTI_HOP": "#4C9F70", "PSEUDO_MULTI_HOP": "#E8A33D", "SINGLE_HOP": "#C44E52"}
    counts = audit_df["label"].value_counts().reindex(order, fill_value=0)

    # 1. Bar chart
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values, color=[colors[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values) * 0.01, str(v), ha="center", fontweight="bold")
    plt.title("Classification multi-hop (analyse de contenu, sans hop_type)")
    plt.ylabel("Nombre de questions")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig11_multihop_bar_chart.png", dpi=150)
    plt.close()

    # 2. Pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=[colors[k] for k in counts.index], startangle=90)
    plt.title("Répartition TRUE / PSEUDO / SINGLE-hop")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig12_multihop_pie_chart.png", dpi=150)
    plt.close()

    # 3. Histogramme : nombre de CHUNKS DISTINCTS par question
    # (le nombre brut de supporting_chunks vaut toujours 2 par construction ;
    # ce qui est informatif, c'est len(set(...)) : 1 = meme chunk, 2 = distincts)
    n_distinct_chunks = audit_df["n_distinct_chunks"]
    plt.figure(figsize=(6, 5))
    bins = sorted(n_distinct_chunks.unique())
    plt.hist(n_distinct_chunks, bins=[b - 0.5 for b in bins] + [bins[-1] + 0.5],
             rwidth=0.6, color="#4C72B0")
    plt.xticks(bins)
    plt.title("Distribution du nombre de chunks distincts par question")
    plt.xlabel("Chunks distincts (1 = meme passage, 2 = passages differents)")
    plt.ylabel("Nombre de questions")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig13_distribution_chunks_distincts.png", dpi=150)
    plt.close()

    # 4. Tableau récapitulatif (figure matplotlib)
    total = len(audit_df)
    summary_rows = [
        [label, str(counts[label]), f"{counts[label] / total * 100:.1f}%"]
        for label in order
    ]
    summary_rows.append(["TOTAL", str(total), "100.0%"])

    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=summary_rows,
        colLabels=["Catégorie", "Nombre", "Pourcentage"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    plt.title("Tableau récapitulatif — validation multi-hop", pad=20)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig14_tableau_recapitulatif.png", dpi=150)
    plt.close()

    print(f"✓ Figures sauvegardées dans {figures_dir}/ "
          f"(fig11_multihop_bar_chart.png, fig12_multihop_pie_chart.png, "
          f"fig13_distribution_chunks_distincts.png, fig14_tableau_recapitulatif.png)")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                     help="Fichiers JSON à fusionner et analyser")
    ap.add_argument("--text-chunks-kv", default=DEFAULT_TEXT_CHUNKS_KV)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--audit-csv", default=DEFAULT_AUDIT_CSV)
    ap.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    args = ap.parse_args()

    print("=" * 70)
    print("Chargement et fusion des sources (hop_type ignoré dès le chargement)")
    print("=" * 70)
    examples = load_and_merge(args.sources)
    chunk_to_doc = load_chunk_to_doc(args.text_chunks_kv)

    print("=" * 70)
    print("Classification (analyse de contenu uniquement)")
    print("=" * 70)
    audit_rows = []
    kept_examples = []
    for ex in examples:
        result = classify(ex, chunk_to_doc)
        supporting_chunks = ex.get("supporting_chunks", [])
        audit_rows.append({
            "question": ex.get("question", ""),
            "ground_truth": ex.get("ground_truth", ex.get("answer", "")),
            "label": result["label"],
            "reason": result["reason"],
            "entity_A_mentioned": result["entity_A_mentioned"],
            "answer_leaked": result["answer_leaked"],
            "hops_redundant": result["hops_redundant"],
            "chunks_distinct": result["chunks_distinct"],
            "docs_distinct": result["docs_distinct"],
            "n_distinct_chunks": len(set(supporting_chunks)) if supporting_chunks else 0,
        })
        if result["label"] == "TRUE_MULTI_HOP":
            kept_examples.append(ex)  # format d'origine, intact, aucun champ ajouté

    audit_df = pd.DataFrame(audit_rows)

    # ── Sauvegarde du benchmark filtré (format identique à l'original) ──
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(kept_examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit_df.to_csv(args.audit_csv, index=False)

    # ── Rapport ──────────────────────────────────────────────────────
    total = len(audit_df)
    counts = audit_df["label"].value_counts()
    n_true = int(counts.get("TRUE_MULTI_HOP", 0))
    n_pseudo = int(counts.get("PSEUDO_MULTI_HOP", 0))
    n_single = int(counts.get("SINGLE_HOP", 0))

    print("\n" + "=" * 70)
    print("RAPPORT")
    print("=" * 70)
    print(f"Nombre total de questions       : {total}")
    print(f"Vraies questions multi-hop      : {n_true}")
    print(f"Pseudo multi-hop                : {n_pseudo}")
    print(f"Single-hop                      : {n_single}")
    print(f"Pourcentage conservé (TRUE)     : {n_true/total*100:.1f}%")
    print(f"Pourcentage supprimé            : {(n_pseudo+n_single)/total*100:.1f}%")

    make_visualizations(audit_df, Path(args.figures_dir))

    print(f"\n✓ Benchmark filtré -> {args.output} ({n_true} questions, format d'origine intact)")
    print(f"✓ Audit détaillé   -> {args.audit_csv}")


if __name__ == "__main__":
    main()
