"""
Étape 1 — Résolution d'entités à l'échelle du graphe, SANS réindexation
et SANS appel LLM.

Constat (confirmé sur indexes/lightrag_500_connected_v2) :
- 62.8% des 5065 nœuds ont un degré == 1 (mini-étoiles locales à un chunk)
- 534 composantes connexes, seulement 43% des nœuds dans la composante géante
- scripts/merge_entity_aliases.py ne traite que 7 alias écrits à la main

IMPORTANT — pourquoi ce script n'utilise PAS la similarité cosinus comme
critère principal (contrairement à une première version testée) :
mesuré empiriquement sur vdb_entities.json (nomic-embed-text, textes courts
"nom + description") :
  - paires aléatoires (probablement non liées)   : cosine médian = 0.58,
    mais p99 = 0.9987 (!) -> l'espace d'embedding a une queue de similarités
    parasites très élevées entre entités SANS RAPPORT.
  - vrais doublons connus ("Large Language Models" vs
    "Large Language Model (LLM)")                : cosine = 0.92
  - vrais doublons connus ("Large Language Models" vs
    "Large Language Model(S)")                   : cosine = 0.85
Autrement dit, avec ce modèle d'embedding sur des noms d'entités courts,
des paires SANS AUCUN rapport peuvent avoir un score plus élevé (0.999)
que de vrais doublons (0.85-0.92). Un seuil cosinus, quel qu'il soit,
produit donc soit trop de faux positifs, soit rate les vrais doublons.

-> Stratégie retenue : blocage LEXICAL (normalisation, pluriel, acronymes,
   parenthèses) comme signal PRINCIPAL et décisif, la similarité cosinus
   n'étant conservée que comme colonne d'information dans le CSV de revue
   (jamais comme critère d'acceptation).

Il ne merge RIEN automatiquement : il écrit un CSV de candidats à relire
(quelques minutes pour ~50-150 lignes), consommé ensuite par
scripts/apply_entity_merges.py.

Usage:
    python scripts/resolve_entities_embeddings.py
"""
import argparse
import base64
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

VDB_PATH = Path("indexes/lightrag_500_connected_v2/vdb_entities.json")
OUT_CSV = Path("data/processed/entity_merge_candidates.csv")

# Mots de tête trop génériques pour, à eux seuls, prouver un doublon
# ("Insurance Dataset" et "Taobao Dataset" partagent "dataset" mais ne
#  sont PAS la même entité).
GENERIC_TOKENS = {
    "dataset", "datasets", "model", "models", "method", "methods",
    "approach", "approaches", "agent", "agents", "technique", "techniques",
    "framework", "frameworks", "baseline", "baselines", "system", "systems",
    "task", "tasks", "metric", "metrics", "algorithm", "algorithms",
    "module", "modules", "component", "components", "strategy", "strategies",
}
STOPWORDS = {"a", "an", "the", "of", "for", "in", "on", "with", "and", "or"}

_VERSION_RE = re.compile(r"\d+(?:\.\d+)?")


def _normalize(name: str) -> str:
    s = name.lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_keep_dots(name: str) -> str:
    """Comme _normalize mais préserve les points décimaux ('4.6' reste
    '4.6' et pas '4 6') — indispensable pour détecter les numéros de
    version (Claude Sonnet 4 / 4.5 / 4.6, GPT-4 / GPT-5, Llama-2 / Llama-3)."""
    s = name.lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9.\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _singularize(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("es") and len(tok) > 3:
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def _meaningful_tokens(name: str) -> frozenset:
    toks = [_singularize(t) for t in _normalize(name).split() if t not in STOPWORDS]
    return frozenset(t for t in toks if t not in GENERIC_TOKENS)


def _all_tokens_singularized(name: str) -> frozenset:
    return frozenset(_singularize(t) for t in _normalize(name).split() if t not in STOPWORDS)


def _differs_only_by_version(a: str, b: str) -> bool:
    """GPT-4 vs GPT-5, Llama-2 vs Llama-3, Claude Sonnet 4 vs 4.5 vs 4.6 :
    NE JAMAIS fusionner automatiquement. Opère sur une normalisation qui
    préserve les points décimaux (sinon '4.6' devient '4 6' et le garde-fou
    ne détecte plus rien — bug constaté et corrigé après test)."""
    na, nb = _normalize_keep_dots(a), _normalize_keep_dots(b)
    nums_a = _VERSION_RE.findall(na)
    nums_b = _VERSION_RE.findall(nb)
    if not nums_a and not nums_b:
        return False
    stripped_a = _VERSION_RE.sub("#", na)
    stripped_b = _VERSION_RE.sub("#", nb)
    return stripped_a == stripped_b and nums_a != nums_b


def _acronym_of(name: str) -> str:
    words = [w for w in re.split(r"[\s\-]+", name) if w]
    return "".join(w[0] for w in words if w[0].isalpha()).lower()


def _blocking_key(name: str):
    """Retourne une clé de blocage si le nom est un candidat évident
    (forme normalisée+singularisée, ou acronyme), sinon None."""
    all_sing = _all_tokens_singularized(name)
    if len(all_sing) >= 2:
        return ("tokset", all_sing)
    return None


def load_entity_vectors(vdb_path: Path):
    d = json.loads(vdb_path.read_text(encoding="utf-8"))
    dim = d["embedding_dim"]
    raw = base64.b64decode(d["matrix"])
    matrix = np.frombuffer(raw, dtype=np.float32).reshape(-1, dim)
    names = [row["entity_name"] for row in d["data"]]
    descriptions = [row.get("content", "")[:160] for row in d["data"]]
    assert matrix.shape[0] == len(names), "matrix/data mismatch"
    return names, matrix, descriptions


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def is_duplicate_pair(name_i: str, name_j: str) -> tuple[bool, str]:
    """Décision lexicale : True + raison si (name_i, name_j) sont très
    probablement la même entité."""
    if _differs_only_by_version(name_i, name_j):
        return False, "version_diff"

    norm_i, norm_j = _normalize(name_i), _normalize(name_j)
    all_i, all_j = _all_tokens_singularized(name_i), _all_tokens_singularized(name_j)

    # 1) identique après normalisation + singularisation complète
    #    ("LLM" == "LLMs", "Large Language Model" == "Large Language Models")
    if all_i == all_j and len(all_i) >= 1:
        return True, "normalized_equal"

    # 2) acronyme <-> forme longue ("LLM" <-> "Large Language Model(s)")
    acr_i, acr_j = _acronym_of(name_i), _acronym_of(name_j)
    flat_i, flat_j = norm_i.replace(" ", ""), norm_j.replace(" ", "")
    if len(flat_i) <= 6 and acr_j == flat_i and len(acr_j) >= 2:
        return True, "acronym_match"
    if len(flat_j) <= 6 and acr_i == flat_j and len(acr_i) >= 2:
        return True, "acronym_match"

    # 3) forte similarité de chaîne (fautes de frappe, tirets, casse)
    #    ET recouvrement de tokens significatifs (pas seulement génériques)
    meaningful_i, meaningful_j = _meaningful_tokens(name_i), _meaningful_tokens(name_j)
    if meaningful_i and meaningful_j:
        jaccard_meaningful = len(meaningful_i & meaningful_j) / len(meaningful_i | meaningful_j)
        ratio = SequenceMatcher(None, norm_i, norm_j).ratio()
        if jaccard_meaningful >= 0.75 and ratio >= 0.80:
            return True, "high_lexical_overlap"

    return False, ""


def main():
    print(f"Chargement des embeddings depuis {VDB_PATH} ...")
    names, matrix, descriptions = load_entity_vectors(VDB_PATH)
    print(f"  {len(names)} entités, dim={matrix.shape[1]}")

    # --- Blocage : on ne compare que les entités qui partagent au moins
    # un token significatif (évite le O(n^2) complet tout en restant exhaustif
    # sur les vrais candidats, contrairement au NN sur embeddings).
    token_index = defaultdict(list)
    for i, name in enumerate(names):
        for tok in _all_tokens_singularized(name):
            token_index[tok].append(i)

    candidate_pairs = set()
    for tok, idxs in token_index.items():
        if len(idxs) < 2 or len(idxs) > 60:  # >60 = token trop générique, ignoré
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                candidate_pairs.add((min(i, j), max(i, j)))

    print(f"  {len(candidate_pairs)} paires candidates après blocage lexical")

    uf = UnionFind(range(len(names)))
    reasons = {}
    rejected_version_pairs = []

    for i, j in candidate_pairs:
        ok, reason = is_duplicate_pair(names[i], names[j])
        if reason == "version_diff":
            rejected_version_pairs.append((names[i], names[j]))
            continue
        if ok:
            uf.union(i, j)
            reasons[(i, j)] = reason

    clusters = defaultdict(list)
    for i in range(len(names)):
        clusters[uf.find(i)].append(i)
    clusters = {root: members for root, members in clusters.items() if len(members) > 1}

    print(f"\n{len(clusters)} clusters de doublons détectés "
          f"({sum(len(m) for m in clusters.values())} entités concernées sur {len(names)})")
    print(f"{len(rejected_version_pairs)} paires écartées automatiquement (numéros de version différents)")

    def cosine(i, j):
        return float(matrix[i] @ matrix[j])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "keep_this_cluster(1/0)", "canonical_suggested",
                    "members", "n_members", "match_reasons", "avg_cosine_info_only",
                    "sample_descriptions"])
        for cid, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
            member_names = [names[m] for m in members]
            canonical = min(member_names, key=len)
            sims, rsns = [], set()
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    key = (min(members[a], members[b]), max(members[a], members[b]))
                    sims.append(cosine(*key))
                    if key in reasons:
                        rsns.add(reasons[key])
            avg_cos = sum(sims) / len(sims) if sims else 0.0
            samples = " | ".join(descriptions[m][:80] for m in members[:3])
            w.writerow([cid, 1, canonical, "; ".join(member_names), len(member_names),
                        ",".join(sorted(rsns)), round(avg_cos, 4), samples])

    print(f"\n✓ Candidats écrits -> {OUT_CSV}")
    print("  ÉTAPE MANUELLE (quelques minutes) : ouvrir ce CSV, mettre 'keep_this_cluster' à 0")
    print("  pour les clusters qui ne sont PAS de vrais doublons, ajuster 'canonical_suggested'")
    print("  si besoin, puis lancer scripts/apply_entity_merges.py")
    print("  (colonne avg_cosine_info_only = indicatif uniquement, PAS un critère de décision")
    print("   — voir le commentaire en tête de fichier pour comprendre pourquoi)")


if __name__ == "__main__":
    main()
