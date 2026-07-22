# État réel du projet — graphe, RAG, et intégration React

**Date :** 21 juillet 2026
**But de ce document :** répondre sans ambiguïté à "qu'est-ce qui a été réellement modifié", "qu'est-ce que je dois exécuter", "mon RAG fonctionne-t-il bien", et "comment brancher une interface React".

---

## TL;DR (réponses directes)

| Question | Réponse |
|---|---|
| Le graphe LightRAG a-t-il été modifié ? | **NON.** Aucun fichier de `indexes/lightrag_500_connected_v2/` n'a changé (vérifié : dates de modification inchangées, aucun dossier de backup créé). |
| Des améliorations ont-elles été préparées ? | **OUI** — 4 scripts créés et testés, mais seulement en mode `--dry-run` (simulation). Rien n'a été appliqué pour de vrai. |
| Dois-je exécuter des fichiers ? | **OUI, 2 commandes**, voir §2. Sans ça, le graphe reste exactement comme avant cette conversation. |
| `graph_v3.py` a-t-il changé ? | **Partiellement.** Le correctif `top_k`/`chunk_top_k` est sur le disque (je l'ai appliqué directement). Les correctifs plus récents (bug Groq, suppression de Chroma, juge indépendant) n'existent que dans le chat — pas encore copiés dans le fichier. |
| `eval_ragas_3.py` a-t-il changé ? | **Oui, entièrement** — le diff montre que vous avez déjà copié la version corrigée que je vous ai donnée. Ce fichier est à jour. |
| Ai-je un RAG qui fonctionne bien ? | **Non, pas encore validé.** Voir §4 — les trois systèmes tournent mécaniquement, mais aucun n'a de score RAGAS propre et récent qui prouve une bonne qualité. |
| Faut-il des endpoints pour une interface React ? | **Oui, obligatoire.** Voir §5. |

---

## 1. Ce qui a été réellement fait cette session (preuve, pas promesse)

Vérifié par `git status` / dates de fichiers, pas de mémoire :

### Fichiers créés sur disque
- `scripts/graph_quality_report.py` — mesure densité/degré/composantes. **Exécuté**, résultat confirmé : 5065 nœuds, 5068 relations, densité 0.000395, composante géante 43.3%.
- `scripts/resolve_entities_embeddings.py` — détecte les doublons d'entités. **Exécuté pour de vrai** (c'est une simple analyse, ça ne touche pas le graphe) → a produit `data/processed/entity_merge_candidates.csv` (103 clusters de doublons, à relire).
- `scripts/apply_entity_merges.py` — fusionne les doublons validés dans le CSV. **Testé uniquement en `--dry-run`.** Aucune fusion réelle n'a eu lieu.
- `scripts/add_cooccurrence_edges.py` — ajoute des relations entre entités qui partagent un chunk. **Testé uniquement en `--dry-run`.** Aucune arête réelle n'a été ajoutée.
- `scripts/visualize_graph.py` — **nouveau, ajouté maintenant**, génère une figure PNG du graphe.
- `scripts/export_graph_to_neo4j.py` — **nouveau, ajouté maintenant**, exporte le `.graphml` vers Neo4j.

### Fichiers modifiés sur disque
- `src/agent/graph_v3.py` : seul changement réel appliqué : `TOP_K_LIGHTRAG` 5→40, `CHUNK_TOP_K` 3→20, `enable_rerank` commenté, `max_total_tokens` ajouté. **Tout le reste discuté (bug Groq câblé en dur, suppression de Chroma, `aquery_data`, juge indépendant) n'a été donné qu'en texte dans le chat — pas appliqué au fichier.**
- `eval_ragas_3.py` : le `git diff` montre que la version corrigée complète (import `ragas.metrics`, appel `evaluate()` restauré, `judge_independent`) est bien sur le disque — vous l'avez appliquée vous-même entre-temps.

### Ce qui n'existe QUE dans le chat (jamais écrit sur disque)
- La version complète corrigée de `graph_v3.py` (bug Groq, suppression Chroma, `aquery_data`).
- `eval_lightrag_ragas.py`
- `analyze_critique_bias.py`

**Si vous voulez ces trois-là, il faut les copier vous-même depuis la conversation — je ne les ai pas écrits sur le disque, comme demandé ("ne corrige pas dans le code").**

---

## 2. Fichiers à exécuter maintenant, dans l'ordre

Tant que vous ne lancez pas ces 2 commandes, **le graphe reste inchangé** :

```bash
# 1. Relire (2-5 min) data/processed/entity_merge_candidates.csv,
#    mettre keep_this_cluster(1/0) à 0 pour les clusters qui ne sont pas
#    de vrais doublons (déjà fait une fois — 103 clusters, quelques faux
#    positifs repérés comme "3D Convolutional Neural Networks" absorbant "CNN")

# 2. Appliquer les fusions validées (backup automatique de l'index avant écriture)
python scripts/apply_entity_merges.py

# 3. Ajouter les arêtes de co-occurrence (backup automatique avant écriture)
python scripts/add_cooccurrence_edges.py

# 4. Vérifier l'effet réel (pas simulé cette fois)
python scripts/graph_quality_report.py --save-json data/processed/graph_stats_after.json
```

D'après la simulation faite précédemment (en mémoire, sur les mêmes données), l'effet attendu :

| Métrique | Avant | Après (attendu) |
|---|---|---|
| Composante géante | 43.3% | ~54.8% |
| Degré moyen | 2.00 | ~2.25 |
| Composantes connexes | 534 | ~390 |

C'est une vraie amélioration, pas un miracle — le graphe restera épars (voir `ANALYSE_PROJET.md` pour la cause racine : chaque abstract arXiv introduit une méthode unique qui ne peut structurellement pas se reconnecter à un autre papier).

---

## 3. Nouveaux outils : visualiser et exporter le graphe

### Visualisation (figure PNG, dossier `figures/`)

```bash
# AVANT le post-traitement (déjà généré comme preuve, voir figures/fig9_graphe_knowledge_avant_posttraitement.png)
python scripts/visualize_graph.py --out figures/fig9_graphe_avant_posttraitement.png --title "Graphe LightRAG avant post-traitement"

# APRÈS avoir appliqué les étapes 2-3 ci-dessus
python scripts/visualize_graph.py --out figures/fig10_graphe_apres_posttraitement.png --title "Graphe LightRAG après post-traitement"
```

J'ai déjà exécuté la première commande : la figure **`figures/fig9_graphe_knowledge_avant_posttraitement.png`** existe déjà sur votre disque. Elle montre clairement le phénomène décrit dans `ANALYSE_PROJET.md` : un unique hub massif ("Large Language Models") au centre, entouré d'un nuage d'entités faiblement connectées entre elles (une seule arête chacune, pas de maillage).

5065 nœuds étant illisibles d'un coup, le script affiche la composante connexe géante, plafonnée à 300 nœuds (les plus connectés en priorité) — utilisez `--max-nodes` pour ajuster.

### Export vers Neo4j

⚠️ **Neo4j n'est actuellement PAS démarré sur votre machine** (j'ai testé la connexion : `Connection refused` sur le port 7687). Il faut d'abord le relancer, comme dans le journal Sprint 1 :
```bash
docker run -p 7474:7474 -p 7687:7687 neo4j:5.13.0
```
Puis :
```bash
# Réutilise le même schéma Cypher que vos requêtes existantes :
# (:Entity {name, description, type})-[:RELATES_TO]->(:Entity)
python scripts/export_graph_to_neo4j.py

# Pour repartir d'un Neo4j propre (efface l'ancien graphe avant d'importer) :
python scripts/export_graph_to_neo4j.py --clear
```
Le script utilise `MERGE` (pas `CREATE`), donc le relancer plusieurs fois ne crée pas de doublons — vous pouvez l'exécuter une fois avant le post-traitement et une fois après, pour comparer visuellement dans Neo4j Browser avec les requêtes déjà connues :
```cypher
MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 50
MATCH (n) RETURN n.name, count{(n)--()} AS degree ORDER BY degree DESC LIMIT 10
```

Je n'ai **pas exécuté** ce script moi-même (ni le `--clear` ni l'import normal) : c'est une écriture dans votre base Neo4j partagée, à vous de la lancer en connaissance de cause.

---

## 4. Est-ce que le RAG fonctionne bien ? (analyse honnête, système par système)

### RAG baseline (ChromaDB + Ollama)
- L'index existe et est peuplé (~19.7 Mo, 7 fichiers).
- Le retriever fonctionne mécaniquement (il renvoie des chunks).
- **Mais il n'a jamais été évalué proprement, seul, avec des métriques RAGAS fiables.** Le journal (12-14/06) documente une tentative où `context_precision` était invalidé par un problème d'alignement des chunks. Aucun script du projet ne produit aujourd'hui un score RAGAS propre pour ce système isolé.
- **Verdict : fonctionne, mais qualité non prouvée.**

### LightRAG (graphe + vecteurs, mode hybrid)
- Historique documenté de retrieval "cassé" (`[no-context]` sur toutes les requêtes).
- Cause probable identifiée cette session : `_build_lightrag()` câblait `llm_model_func=groq_llm_func` **en dur**, indépendamment du flag `USE_GROQ` — si la clé Groq est absente/invalide, l'extraction de mots-clés (obligatoire en mode hybrid) échoue silencieusement → contexte vide à chaque fois. **Ce correctif n'est pas encore appliqué sur le disque** (seulement montré dans le chat).
- Le graphe lui-même reste épars (densité 0.0004) tant que §2 n'est pas exécuté.
- **Verdict : la cause probable du problème historique est identifiée mais pas encore corrigée dans le fichier réel. Pas de run récent et propre pour confirmer.**

### Agentic GraphRAG (l'agent LangGraph complet)
- Derniers scores RAGAS mesurés (`Eval_agentic/eval_agent_s4.csv`, `eval_agent_6_questions.csv`) : Faithfulness 0.20–0.63, Context Precision/Recall souvent proches de 0, un run récent avec `answer_relevancy` = NaN sur 100% des lignes.
- Ces chiffres datent d'AVANT tous les correctifs discutés cette session (aucun n'a encore été suivi d'une réévaluation).
- **Verdict : pas bon aujourd'hui, mais mesuré sur une version antérieure aux correctifs. Il faut relancer une évaluation après avoir appliqué les fixes pour savoir où on en est réellement.**

### Conclusion générale
**Non, il n'y a pas aujourd'hui de système dont on puisse dire "il fonctionne bien" avec des preuves RAGAS à l'appui.** Tout tourne mécaniquement (aucun crash), mais aucun score récent et fiable ne confirme une bonne qualité pour l'un des trois systèmes. La priorité n°1 avant toute nouvelle fonctionnalité (React, etc.) est de :
1. Appliquer réellement les correctifs déjà écrits (§2 + copier le `graph_v3.py` corrigé du chat),
2. Relancer une évaluation RAGAS propre,
3. Regarder les vrais chiffres avant de décider de la suite.

---

## 5. Ajouter une interface React — faut-il des endpoints ?

**Oui, obligatoirement.** React tourne dans le navigateur ; il ne peut pas importer directement `run_agent()` (du code Python qui charge des modèles, se connecte à Neo4j/Ollama/Groq). Il faut un serveur HTTP entre les deux :

```
React (navigateur)  --HTTP-->  API Python (FastAPI)  --appelle-->  run_agent()
```

### Squelette minimal (FastAPI, cohérent avec le code déjà 100% async du projet)

```python
# api_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent.graph_v3 import run_agent

app = FastAPI(title="Agentic GraphRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # URL de votre app React (Vite par défaut)
    allow_methods=["*"], allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(req: QueryRequest):
    result = run_agent(req.question)
    return {
        "answer": result["final_response"],
        "critique_score": result["critique_score"],
        "judge_independent": result.get("critique_judge_independent"),
        "iterations": result["iteration"],
        "trace": result["trace"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```
```bash
pip install fastapi uvicorn
uvicorn api_server:app --reload --port 8000
```

Côté React, un simple `fetch("http://localhost:8000/query", {method:"POST", body: JSON.stringify({question})})`.

**Points d'attention spécifiques à ce projet :**
- `run_agent()` charge LightRAG/Neo4j/Ollama **à l'import du module** (variables globales `rag_instance`, `llm`, etc. dans `graph_v3.py`) — donc le serveur FastAPI doit rester up en continu (pas de rechargement à chaque requête), sinon chaque appel re-télécharge/reconnecte tout.
- Une requête peut prendre 30-90s (retrieval + génération + éventuel self-correct ×3) → prévoir un indicateur de chargement côté React, voire passer en streaming (Server-Sent Events) si vous voulez afficher le trace en temps réel (`[QUERY]`, `[HYBRID_SEARCH]`, `[CRITIQUE]`...).
- N'exposez `NEO4J_PASSWORD`/`GROQ_API_KEY` que côté serveur (`.env`), jamais dans le code React.

---

## 6. Prochaines étapes recommandées, dans l'ordre

1. Relire `data/processed/entity_merge_candidates.csv`, lancer `apply_entity_merges.py` puis `add_cooccurrence_edges.py` pour de vrai.
2. Copier dans `src/agent/graph_v3.py` les correctifs donnés en chat (bug Groq, suppression Chroma, `aquery_data`, juge indépendant).
3. Relancer une évaluation RAGAS propre (`eval_lightrag_ragas.py` ou la version corrigée de `eval_ragas_3.py`).
4. Lancer `analyze_critique_bias.py` sur le nouveau CSV pour savoir si le critique interne est fiable.
5. Regarder les vrais chiffres → décider si le système est prêt pour une interface, ou s'il faut encore itérer.
6. Si prêt : ajouter `api_server.py` (FastAPI) + démarrer le front React.
