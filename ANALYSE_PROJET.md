# Analyse complète du projet — PFE Agentic GraphRAG

**Date de l'analyse :** 16 juillet 2026
**Base :** JOURNAL.md (jusqu'au 28 juin 2026), code source (`src/`, `eval_ragas*.py`, `ragas_age.py`, `scripts/`), index LightRAG (`indexes/lightrag_500_connected_v2`), résultats d'évaluation (`Eval_agentic/`)

---

## 0. Résumé exécutif

Le projet a trois briques censées fonctionner ensemble : un **RAG baseline** (ChromaDB), un **LightRAG** (graphe + vecteurs), et un **Agentic GraphRAG** (LangGraph, censé orchestrer les deux). D'après le journal et les fichiers présents, aucune des trois ne produit aujourd'hui des scores RAGAS satisfaisants, et ce n'est **pas une question de chance ou de LLM trop faible** : ce sont des causes structurelles précises, identifiables dans le code.

Les 3 blocages principaux, par ordre d'impact :

1. **Le graphe LightRAG est structurellement sous-connecté** (densité 0.0004, degré moyen 2, 43 % seulement des nœuds dans la composante géante) → le mode `hybrid`/`local`/`global` n'a presque rien à traverser → contexte pauvre → réponses "not in corpus" en boucle.
2. **Le benchmark d'évaluation n'a presque aucune vraie question multi-hop** (10 vraies sur 250, soit 4 %) → les métriques RAGAS actuelles ne mesurent pas ce que le PFE prétend démontrer (raisonnement multi-sauts).
3. **Le code d'agent est dupliqué en 4 versions non synchronisées** (`graph_v1.py`, `graph_v2.py`, `graph_v2_corrigé.py`, `graph_v3.py`), avec des scripts d'évaluation qui importent la mauvaise version — dont un bug actif dans le fichier actuellement ouvert (`eval_ragas_3.py`), qui plante systématiquement.

Le reste du document détaille chaque problème avec sa localisation exacte, sa cause racine, et une solution concrète.

---

## 1. État actuel du projet — vue d'ensemble

### 1.1 Trois systèmes visés

| Système | Statut réel | Fichiers |
|---|---|---|
| **RAG baseline** (ChromaDB + Ollama) | Fonctionnel mais mono-hop, jamais évalué isolément avec un vrai benchmark aligné | `scripts/build_chroma_pfe500.py`, `docker-compose.rag-baseline.yml` |
| **LightRAG** (graphe + vecteurs, mode hybrid) | Indexé (5065 nœuds / 5068 relations) mais graphe peu connecté ; retrieval a été "cassé" (no-context) à plusieurs reprises, jamais confirmé stable | `indexes/lightrag_500_connected_v2/`, `src/agent/graph_v3.py` |
| **Agentic GraphRAG** (LangGraph, 7-8 nœuds, self-critique) | Fonctionne de bout en bout mais scores RAGAS faibles (Faithfulness ~0.20–0.63, Context Precision/Recall souvent proches de 0) | `src/agent/graph_v1.py`, `graph_v2.py`, `graph_v2_corrigé.py`, `graph_v3.py` |

### 1.2 Métriques RAGAS actuellement obtenues (mesurées, pas estimées)

Calculées directement à partir des CSV présents dans `Eval_agentic/` :

| Run | n | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|
| `eval_agent_s3.csv` (agent → `graph_v2`, Neo4j direct) | 10 | 0.576 | 0.568 | 0.381 | 0.291 |
| `eval_agent_s4.csv` (agent → `graph_v1`) | 10 | 0.630 | 0.494 | 0.314 | 0.331 |
| `eval_agent_6_questions.csv` (run le plus récent, 29/06) | 6 | 0.197 | **NaN (6/6)** | 0.000 | 0.000 |

**Lecture** : la tendance est à la dégradation, pas à l'amélioration. Le run le plus récent (`6_questions`) a une `answer_relevancy` totalement nulle (NaN sur 100 % des lignes — signe que le calcul RAGAS a échoué silencieusement, probablement une erreur LLM/embeddings non levée grâce à `raise_exceptions=False`) et un `context_precision`/`context_recall` à 0 exact, ce qui correspond au symptôme "no-context" documenté dans le journal du 05/06.

Pour un PFE qui doit "démontrer un bon score RAGAS", ces chiffres sont largement en dessous des seuils généralement attendus (Faithfulness et Context Precision devraient viser >0.7).

---

## 2. Bugs de code identifiés (concrets, avec localisation)

### 2.1 `eval_ragas_3.py` — script cassé, actuellement ouvert dans l'IDE

Deux bugs cumulés, confirmés par `git diff` (changements non commités) :

- **Bug A — appel RAGAS manquant.** Entre la boucle de collecte (ligne ~110) et le post-traitement (`df = ragas_result.to_pandas()`, ligne 114), le bloc qui appelait `evaluate(...)` et assignait `ragas_result` a été supprimé (visible dans `git diff` : 22 lignes supprimées, remplacées par une ligne vide). Résultat : `NameError: name 'ragas_result' is not defined` à coup sûr à l'exécution.
- **Bug B — mauvais agent importé.** Le script importe `from src.agent.graph_v2 import run_agent` (ligne 18), mais construit le contexte RAGAS avec `result.get("lightrag_context", "")` (ligne 65) et attend un champ `vector_results` produit par un nœud `chroma_for_ragas`. Or `graph_v2.py` n'a **aucun** champ `lightrag_context` dans son `AgentState` — c'est `graph_v3.py` (l'agent basé sur LightRAG hybrid search) qui possède ces champs exacts (`hybrid_search`, `chroma_for_ragas`, `lightrag_context`). L'import pointe donc sur le mauvais fichier ; même en réparant le bug A, le script tournera sur le mauvais agent et `build_ragas_context()` ne recevra jamais de `lightrag_context`.

**Correctif** : remplacer l'import par `from src.agent.graph_v3 import run_agent`, et restaurer l'appel `evaluate()` (le bloc existe encore tel quel dans `eval_ragas.py` / `ragas_age.py`, à adapter avec les mêmes métriques `ragas.metrics.collections`).

### 2.2 Duplication non maîtrisée de l'agent (4 fichiers, 3 architectures différentes)

| Fichier | Lignes | Architecture | Backend retrieval | Importé par |
|---|---|---|---|---|
| `graph_v1.py` | 622 | Query→Graph+Vector→Fuse→Response→Critique→SelfCorrect | Neo4j Cypher direct + Chroma | `ragas_age.py` |
| `graph_v2.py` | 656 | idem (8 nœuds, "version corrigée", dédup, docstrings) | Neo4j Cypher direct + Chroma | `eval_ragas.py`, `eval_ragas_3.py` (à tort) |
| `graph_v2_corrigé.py` | 385 | inconnue (jamais tracée) | inconnu | **aucun fichier ne l'importe** — code mort |
| `graph_v3.py` | 449 | Query→HybridSearch(LightRAG)→ChromaForRagas→Response→Critique→SelfCorrect | **LightRAG `aquery(mode="hybrid")`** | **aucun fichier ne l'importe** (hors le bug 2.1) |

C'est le problème racine derrière les résultats incohérents d'un run à l'autre : chaque évaluation dans `Eval_agentic/` a probablement tourné sur une version différente de l'agent sans que ce soit tracé (le nom du fichier CSV ne dit pas quelle version de `graph_vX.py` a été utilisée). Il est aujourd'hui **impossible de dire objectivement laquelle des 4 architectures est la meilleure**, alors que c'est exactement la question qu'un PFE doit trancher.

`graph_v3.py` est architecturalement le plus intéressant (il utilise réellement LightRAG au lieu de réimplémenter du retrieval Cypher à la main), mais c'est le seul qui n'est jamais exécuté par un script d'évaluation fonctionnel.

### 2.3 Dépendance `ragas` absente de `requirements.txt`

`requirements.txt` (155 paquets figés) ne contient **ni `ragas` ni `scipy`** (utilisé dans `eval_ragas_3.py` pour `pearsonr`), alors que tous les scripts d'évaluation en dépendent. Le journal (17-18/05) documente d'ailleurs des heures perdues à cause d'installations ad hoc de `ragas` en dehors de `requirements.txt`, avec des conflits de versions (`langchain-community`). C'est le même problème qui va se reproduire à chaque nouvelle machine/environnement.

### 2.4 Cache LightRAG qui semble ne jamais s'invalider correctement

Journal (08/06) : après suppression et reconstruction complète des index, LightRAG "continuait à retourner les anciennes réponses" — problème noté comme non résolu. `kv_store_llm_response_cache.json` existe toujours dans l'index (mis à jour 06/07). Si ce cache mélange des réponses générées avec d'anciens prompts/paramètres (avant les correctifs de prompt strict anti-hallucination), toute évaluation ultérieure est polluée sans avertissement.

---

## 3. Pourquoi le graphe LightRAG donne de mauvais résultats (cause racine, pas juste le symptôme)

Statistiques réelles du graphe (`figures/graph_statistics_20260607_025453.json`) :

- **5065 nœuds, 5068 relations** → un graphe presque "arbre" (nombre d'arêtes ≈ nombre de nœuds, alors qu'un graphe de connaissances bien formé a en général 2 à 5× plus de relations que d'entités).
- **Densité = 0.0004**, **degré moyen = 2**, **degré médian = 1**.
- **Composante géante = 43 % des nœuds seulement** → 57 % du graphe est déconnecté du reste, donc invisible pour toute traversée multi-hop.
- `diameter: "N/A (graphe non connexe)"` — confirmation directe que le graphe n'est pas exploitable pour du raisonnement multi-sauts au-delà de la composante principale.

**Cause racine, pas juste "le LLM local hallucine"** : le pipeline d'extraction LightRAG tourne **chunk par chunk, indépendamment**, avec un LLM local (`llama3.1:8b`) qui n'a aucune vue sur les entités déjà extraites dans les autres chunks. Sans étape de résolution d'entités (entity resolution / linking) globale, chaque chunk "invente" ses propres nœuds locaux, qui ne se reconnectent au reste du graphe que si le LLM réutilise *exactement* la même chaîne de caractères pour désigner une entité déjà vue — ce qui explique aussi pourquoi les alias manuels (`scripts/merge_entity_aliases.py`) ne peuvent traiter que 7 paires à la main, une goutte d'eau face à 5065 nœuds.

Deux tentatives ont été faites pour compenser (alias canoniques, augmentation du gleaning à 6 passes, filtrage) mais elles agissent en aval du problème, pas sur la cause. Le gleaning à 6 passes a même été identifié comme cause probable de bruit/hallucination supplémentaire (journal 05/06, Solution 3A).

### Ce qui manque structurellement ici

- Pas d'étape de **résolution d'entités globale post-extraction** (au-delà des 7 alias en dur) : ni fuzzy matching, ni clustering par embedding de nom d'entité, ni passage par un LLM dédié à la déduplication sur l'ensemble du graphe.
- Pas de **filtrage des nœuds de degré 0-1 orphelins** avant l'indexation vectorielle — un nœud isolé n'apporte rien au mode `local`/`global`/`hybrid` mais dilue le top_k du retrieval.
- Pas de **mesure de qualité de graphe automatisée dans la boucle** (la densité n'est calculée qu'une fois manuellement le 07/06, pas à chaque réindexation) — il faudrait un script qui recalcule densité/composante géante/degré moyen après chaque run et bloque la mise en prod du graphe si les seuils ne sont pas atteints.

---

## 4. Pourquoi le benchmark d'évaluation ne mesure pas ce qu'il devrait

Vérifié directement sur `data/processed/arxiv_multihop_v1.json` : **250 questions au total, dont seulement 10 sont "vraies" 2-hop** (`supporting_chunks[0] != supporting_chunks[1]`), soit **4 %**. Les 240 autres sont des questions "pseudo 2-hop" (les deux chunks de support sont identiques — donc en réalité des questions mono-hop déguisées).

Tous les scripts d'évaluation (`eval_ragas.py`, `ragas_age.py`, `eval_ragas_3.py`) n'échantillonnent que 10 à 14 questions au total sur ce jeu — un échantillon minuscule (`N=6` à `N=10` dans les CSV présents), avec seed fixe (42), donc **aucune variance mesurée**, pas d'intervalle de confiance, pas de robustesse statistique. Une seule question qui échoue fait varier `context_recall` de 10 points.

C'est un problème majeur pour un PFE dont l'angle central est "Agentic GraphRAG pour le Raisonnement Complexe" : avec seulement 10 vraies questions multi-hop disponibles dans tout le benchmark, il est impossible de démontrer statistiquement un avantage du raisonnement multi-sauts par rapport à un RAG naïf.

### Ce qui manque

- Un vrai générateur de questions multi-hop *garanties* (ex : sélectionner deux chunks reliés par une entité pivot dans le graphe LightRAG lui-même, puis générer une question qui exige les deux — au lieu de générer une question puis vérifier après coup si elle est multi-hop).
- Un jeu de test d'au moins 50-100 vraies questions multi-hop pour que les moyennes RAGAS soient statistiquement interprétables.
- Une séparation claire train/validation pendant le tuning des prompts (actuellement, les mêmes 10-20 questions ont probablement servi à la fois à diagnostiquer les problèmes et à "valider" les correctifs — risque de sur-ajustement au jeu de test).

---

## 5. Pourquoi les réponses de l'agent sont si souvent "corpus does not contain information"

C'est visible en clair dans `Eval_agentic/agent_score_s4.json` : sur 10 questions, 5 réponses commencent par "The corpus does not contain information...". Comme le benchmark est généré **à partir du corpus lui-même**, la réponse existe forcément quelque part dans les documents — donc ces 5 cas sont des **faux négatifs de retrieval ou de génération**, pas des cas légitimes de refus.

Deux causes cumulées dans `graph_v3.py` :

1. **`top_k` trop restrictif** : `TOP_K_LIGHTRAG = 5`, `CHUNK_TOP_K = 3` (lignes 34-35). Le journal lui-même recommandait de monter à `top_k=100`, `chunk_top_k=80` en diagnostic (05/06, Solution 1C) mais ces valeurs n'ont jamais été reportées dans le code de l'agent — elles ne sont apparues que dans un bloc de code de diagnostic jamais intégré.
2. **Prompt de génération trop strict combiné à un juge qui récompense le refus** : le prompt système de `node_response` interdit toute extrapolation ("If the answer is not in the context, respond EXACTLY..."), et le prompt de `node_critique` donne un score ≥0.8 quand l'agent dit correctement "not in corpus" (`graph_v3.py` ligne 293). Résultat pervers : **le nœud CRITIQUE valide (score élevé) des réponses "je ne sais pas" qui sont en réalité incorrectes** parce que l'information existait mais n'a pas été récupérée — le juge n'a aucun moyen de savoir que l'info existe ailleurs dans le corpus (il ne voit que le contexte tronqué transmis, pas le corpus entier). C'est cohérent avec le journal (26/06) : *"l'évaluation de la qualité avant le critique donne un bon score"* mais les scores RAGAS restent mauvais — le juge interne et RAGAS ne mesurent pas la même chose, et le juge interne est structurellement biaisé en faveur du refus.

C'est la cause directe des faibles `context_recall` (0.29-0.33) : quand l'agent refuse de répondre, RAGAS considère à juste titre qu'aucune information pertinente utile n'a été extraite du contexte.

---

## 6. Ce qui manque au projet (au-delà des bugs)

- **Pas de configuration centralisée** : chaque agent (`graph_v1` à `v3`) redéfinit ses propres variables d'environnement, ses propres valeurs par défaut de `top_k`, `CRITIQUE_SEUIL`, etc., directement en dur dans le fichier. Un changement de paramètre doit être répété 4 fois, avec risque d'incohérence (déjà visible : `CRITIQUE_SEUIL=0.7` dans `graph_v1.py` vs `0.75` dans `graph_v2.py`/`graph_v3.py`).
- **Pas de traçabilité entre un run d'évaluation et la config utilisée** : les CSV de sortie (`eval_agent_s3.csv`, `s4.csv`, `6_questions.csv`) ne contiennent aucune métadonnée sur la version de l'agent, les paramètres LightRAG, le hash du graphe utilisé. Impossible de reproduire un résultat a posteriori.
- **Pas de tests automatisés** (aucun `pytest`, aucun test unitaire sur les nœuds de l'agent, la fonction d'embedding, le parsing du critique JSON) — hors les scripts `src/test_dim.py`, `test_embed.py`, `test_llm.py`, `test_neo4j.py` qui sont des scripts de vérification manuelle de l'environnement, pas des tests automatisés.
- **`ragas` et `scipy` absents de `requirements.txt`** (cf. §2.3).
- **Pas de script de contrôle qualité du graphe** exécuté systématiquement après chaque réindexation (densité, composante géante, taux de nœuds orphelins) — actuellement fait une fois à la main.
- **Aucune baseline "RAG naïf pur" évaluée avec le même benchmark et les mêmes 4 métriques RAGAS que l'agent**, ce qui rend impossible de démontrer objectivement que l'Agentic GraphRAG apporte un gain (le journal mentionne une évaluation du RAG baseline le 12-14/06 mais avec un problème d'alignement des chunks qui invalide les résultats de `context_precision`).
- **Pas de gestion d'erreurs silencieuses dans RAGAS** : tous les appels `evaluate(...)` utilisent `raise_exceptions=False`, ce qui masque les erreurs individuelles en `NaN` sans log clair (cf. `answer_relevancy` = NaN sur 100 % du run `6_questions.csv`, jamais investigué).
- **Documentation d'architecture absente** : aucun README/schéma qui explique, à la date du jour, laquelle des 4 versions d'agent est "the one", quel index LightRAG est le bon (`lightrag`, `lightrag_500_V1`, `lightrag_500_connected_v2` coexistent dans `indexes/`), et quel est le pipeline de bout en bout actuel.

---

## 7. Plan d'action priorisé

### Phase 1 — Débloquer immédiatement (1-2 jours)

1. **Corriger `eval_ragas_3.py`** : réparer l'import (`graph_v3`, pas `graph_v2`) et restaurer l'appel `evaluate(...)` (cf. §2.1). C'est la seule chose qui empêche de lancer une évaluation propre de l'agent LightRAG dès maintenant.
2. **Ajouter `ragas` et `scipy` à `requirements.txt`** avec versions figées, pour arrêter les installs ad hoc.
3. **Choisir une seule version d'agent** parmi les 4, supprimer ou archiver les 3 autres (`graph_v2_corrigé.py` est déjà mort — code jamais importé, à supprimer sans risque). Recommandation : partir de `graph_v3.py` (seul à utiliser réellement LightRAG au lieu de réimplémenter du Cypher à la main) et y fusionner les bonnes idées de `graph_v2.py` (dédup, docstrings).
4. **Purger le cache LightRAG** (`kv_store_llm_response_cache.json`) avant toute nouvelle campagne d'évaluation, pour être sûr que les réponses reflètent le code/prompt actuel.

### Phase 2 — Réparer le retrieval (3-5 jours)

5. **Augmenter `TOP_K_LIGHTRAG`/`CHUNK_TOP_K`** dans l'agent retenu (au moins `top_k=30-40`, `chunk_top_k=15-20` pour commencer — pas besoin d'aller jusqu'à 100/80, ce sont des valeurs de diagnostic extrême) et re-mesurer l'impact sur `context_recall` avant d'aller plus loin.
6. **Assouplir le prompt de réponse** : au lieu d'un refus binaire "the corpus does not contain...", demander au LLM de citer explicitly l'entité/chunk le plus proche même en cas d'incertitude partielle, et **désynchroniser le score du CRITIQUE de RAGAS** — ne pas laisser le juge interne récompenser les refus sans vérifier qu'ils sont justifiés.
7. **Ajouter un contrôle qualité du graphe automatisé** (script à lancer après chaque réindexation) qui calcule densité / % composante géante / degré moyen, et fixe un seuil minimal avant d'autoriser l'utilisation du graphe en évaluation (ex : composante géante > 70 %, degré moyen > 3).
8. **Résolution d'entités globale** post-extraction : remplacer les 7 alias en dur de `merge_entity_aliases.py` par un clustering automatique (embedding du nom d'entité + seuil de similarité cosinus, ou fuzzy string matching type `rapidfuzz`) appliqué à l'ensemble des 5065 nœuds.

### Phase 3 — Réparer le benchmark (2-3 jours)

9. **Reconstruire le générateur de questions multi-hop** pour garantir des vraies questions 2-hop : partir d'une entité pivot du graphe LightRAG connectée à ≥2 chunks distincts, générer la question à partir de cette structure plutôt que de générer d'abord et vérifier après.
10. **Porter le jeu de vraies questions multi-hop à 50+** avant de tirer des conclusions statistiques.
11. **Évaluer les 3 systèmes (RAG baseline / LightRAG seul / Agentic GraphRAG) sur exactement le même jeu de questions et les mêmes 4 métriques RAGAS**, dans un seul script versionné, avec le hash de la config et de l'index enregistré dans chaque CSV de sortie.

### Phase 4 — Consolidation (1-2 jours)

12. Centraliser la config (un seul `config.py` ou `.env` partagé, pas de valeurs dupliquées entre fichiers agent).
13. Nettoyer `indexes/` : documenter/supprimer les index obsolètes (`lightrag`, `lightrag_500_V1`) pour ne garder que celui réellement utilisé.
14. Ajouter quelques tests automatisés minimaux (parsing du JSON du critique, format du contexte RAGAS, embedding shape) pour éviter les régressions silencieuses déjà vécues (dimension mismatch, `list` vs `np.array`, etc. — cf. journal Sprint 1).

---

## 8. Ce qu'il ne faut *pas* faire

- Ne pas changer de LLM local une nouvième fois pour "espérer" une amélioration (déjà testé : mistral, llama3:8b, llama3.1:8b, sans effet significatif d'après le journal du 25/04). Le goulot d'étranglement documenté est le **retrieval et la connectivité du graphe**, pas le modèle de génération.
- Ne pas ré-augmenter le gleaning ou le contexte d'extraction sans mesurer d'abord l'impact sur la densité du graphe — le journal montre que l'augmentation à 6 passes de gleaning a coïncidé avec une baisse de qualité perçue (plus de bruit), pas une amélioration.
- Ne pas lancer de nouvelle campagne d'évaluation RAGAS tant que les bugs de la Phase 1 ne sont pas corrigés — les résultats actuels ne sont pas interprétables (mauvais agent importé, cache non purgé, `NaN` silencieux).
