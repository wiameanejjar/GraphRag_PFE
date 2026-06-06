# JOURNAL DE BORD — PFE Agentic GraphRAG

**Étudiante :** Wiame Anejjar  
**Encadrant :** Pr. Abdelaaziz Hessane  
**Université :** Moulay Ismail — FSM  
**Parcours :** Master SDIA 2025–2026

---

## SPRINT 1 — Semaine 1 (07–14 Avril 2026)

### Lecture des papers clés

**07/04/2026**

- Réception du sujet PFE de Pr. Hessane : "Agentic GraphRAG pour le Raisonnement Complexe sur des Corpus Techniques"
- Lecture du paper 1 : **RAG Survey** (arxiv:2312.10997) — Gao et al., 2023
  - Compris : 3 paradigmes RAG (Naive, Advanced, Modular)
  - Métriques clés : Faithfulness, Answer Relevance, Context Relevance
  - Métriques d'évaluation RAGAS notées pour Sprint 4

**08/04/2026**

- Lecture du paper 2 : **GraphRAG Microsoft** (arxiv:2404.16130) — Edge et al., 2024
  - Pipeline : Documents → Chunks → Entités/Relations (LLM) → Communautés (Leiden) → Résumés
  - Limitation principale : 610K tokens par requête (très coûteux)
  - Win rate 72-83% vs RAG classique sur questions de compréhension globale

**09/04/2026**

- Lecture du paper 3 : **LightRAG** (arxiv:2410.05779) — HKUDS, 2024
  - Solution au problème de coût de GraphRAG : 6000x moins de tokens
  - Dual-level retrieval : low-level (entités précises) + high-level (thèmes globaux)
  - 4 modes : naive, local, global, hybrid
  - C'est l'outil principal de mon PFE (pip install lightrag-hku)

**10/04/2026**

- Lecture du paper 4 : **LangGraph** — blog.langchain.dev
  - Framework Python pour créer des agents stateful avec graphe d'états
  - Chaque nœud = fonction Python + appel LLM
  - C'est ma contribution principale : l'agent de raisonnement multi-sauts

**11/04/2026**

- Lecture du paper 5 : **HotpotQA** — Yang et al., 2018
  - 113K questions nécessitant 2 documents pour répondre
  - 2 types : bridge (traverser entre docs) et comparison (comparer entités)
  - Benchmark parfait pour évaluer le raisonnement multi-sauts

**12/04/2026**

- Lecture du paper 6 : **RAGAS** — Es et al., 2023
  - Framework d'évaluation automatique pour les systèmes RAG
  - 4 métriques : Faithfulness, Answer Relevancy, Context Precision, Context Recall
  - Sera utilisé au Sprint 4 pour l'évaluation quantitative

**14/04/2026**

- Lecture du paper 7 : **Self-RAG** — Asai et al., 2023
  - Introduit les "reflection tokens" : le LLM décide quand récupérer de l'info
  - Inspirera le nœud CRITIQUE de mon agent LangGraph
- Rédaction du tableau comparatif RAG vs GraphRAG vs Agentic RAG

---

## SPRINT 1 — Semaine 2 (15–21 Avril 2026)

### Installation de l'environnement

**15/04/2026**

- Installation de Python 3.12 via pyenv
- Création du venv : `python -m venv venv`
- Installation de VSCode + extensions Python, Jupyter

**16/04/2026**

- Installation de Docker Desktop sur Windows
- Lancement de Neo4j Community Edition via Docker :
  `docker run -p 7474:7474 -p 7687:7687 neo4j:5.13.0`
- Validation : interface Neo4j accessible sur http://localhost:7474

**17/04/2026**

- Installation d'Ollama : https://ollama.com
- Téléchargement de Llama-3.1-8B : `ollama pull llama3.1:8b` (4.7 Go, ~45 min)
- Test via curl :
  ```bash
  curl http://localhost:11434/api/generate -d '{"model": "llama3.1:8b", "prompt": "Hello"}'
  ```
- Llama-3.1-8B répond correctement en local

**18/04/2026**

- Installation de LightRAG : `pip install lightrag-hku`
- Installation des dépendances : pandas, matplotlib, seaborn, tiktoken
- Création du compte Groq sur console.groq.com
- Obtention de la clé API Groq (tier gratuit : 500K tokens/jour)

**21/04/2026**

- Validation complète de l'environnement
- Test de connexion Neo4j depuis Python via neo4j-driver
- Test de l'API Groq avec llama-3.1-8b-instant
- Tout l'environnement est opérationnel

---

## SPRINT 1 — Semaine 3 (22–28 Avril 2026)

### Première tentative d'indexation avec PDFs

**22/04/2026 — Téléchargement du corpus arXiv (PDFs)**

- Décision initiale : télécharger les PDFs complets des articles arXiv CS.AI
- Problème rencontré : l'API arXiv bloque les requêtes après ~400 articles
  → Erreur : `HTTP Error 429 Too Many Requests`
- Solution appliquée : téléchargement des métadonnées via Kaggle
  (dataset officiel Cornell University, ~4 Go), puis téléchargement
  des PDFs via l'API arXiv avec pauses de 90 secondes tous les 25 articles
- Résultat : 493/500 PDFs téléchargés et 7 introuvables (supprimés d'arXiv)

**23/04/2026 — Parsing des PDFs bruité + nettoyage + Chunking avec overlap**

- Parsing avec PyMuPDF : extraction du texte brut
- Problème : texte contient des numéros de pages, en-têtes LaTeX, formules
- Exemple : `arXiv:0704.0047v1 [cs.NE] 1 Apr 2007
1
Intelligent location...`
- Solution partielle : nettoyage regex (suppression \n, URLs, caractères spéciaux)

**Chunking avec overlap**

- Implémentation du découpage en chunks de 512 tokens avec overlap 50 tokens
- Résultat : 14 085 chunks pour 493 articles
- Tokens moyens/chunk : 502
- Justification du chunking : compromis entre GraphRAG (600 tokens) et LightRAG (1200 tokens)
- Justification de l'overlap : 50 tokens est environ 10% du chunk, standard dans la littérature RAG

**24/04/2026 — Première tentative d'indexation avec texte long (8000 tokens)**

- Tentative d'indexer chaque document avec : titre + abstract + 8000 tokens du full_text
- Problème : le processus bloque complètement ,timeout LLM, mémoire insuffisante
- Réduction à 800 tokens celà bloque encore
- Réduction à 500 tokens -> fonctionne enfin
- Conclusion : seuls 500 tokens du full_text sont exploitables avec Llama-3.1-8B local

**24/04/2026 — Première tentative avec Groq API + checkpointing**

- Décision : utiliser Groq API (llama-3.1-8b-instant) pour accélérer l'extraction
- Implémentation d'un système de checkpointing pour reprendre en cas d'interruption
- Problème rencontré : la limite journalière Groq (500K tokens/jour) s'épuise
  très rapidement car chaque document envoie titre + abstract + 500 tokens PDF
  -> Calcul : environ 4500 tokens/doc × 100 docs = 450K tokens/jour, donc on obtient un quota épuisé
- Résultat : après presque 2 jours de tentatives, seulement 20 premiers
  documents indexés ce qui est inacceptable pour 100 documents

**24/04/2026 — Deuxième tentative : Ollama local (Llama-3.1-8B)**

- Décision : abandonner Groq temporairement, utiliser Ollama 100% local
  pour ne pas avoir de limite de quota
- LLM : llama3.1:8b via Ollama
- Embedding : nomic-embed-text via Ollama
- Résultat : le processus fonctionne mais est extrêmement lent
  -> 1 journée entière (nuit + jour) pour indexer 100 documents
  -> Nombreux warnings LLM pendant tout le processus

**24/04/2026 — Problème 1 : Conflit de dimension d'embedding**

- Erreur : `Embedding dimension mismatch detected: total elements (768) 
cannot be evenly divided by expected dimension (1024)`
- Cause : Dans mon premier code j'ai utilisé `embedding_dim=1024` mais nomic-embed-text
  produit des vecteurs de 768 dimensions
- Diagnostic : test direct de l'API Ollama via `requests.post`
  -> dimension réelle confirmée = 768
- Solution : correction `embedding_dim=1024` à `embedding_dim=768`
  - suppression du dossier `lightrag_storage/` (métadonnées corrompues)
  - réinstallation propre de LightRAG

**24/04/2026 — Problème 2 : Fonction embedding retourne une liste Python**

- Erreur : `AttributeError: 'list' object has no attribute 'size'`
- Cause : LightRAG attend un tableau NumPy 2D mais la fonction
  personnalisée retournait une liste Python standard
- Solution : remplacement de `return embeddings` par
  `return np.vstack(embeddings)` qui retourne un tableau NumPy 2D de shape (n_textes, 768)

**25/04/2026 — Problème 3 : Warnings LLM format errors (tentative de résolution)**

- Warning récurrent : `LLM output format error; found 5/4 fields on ENTITY`
- Cause : Llama-3.1-8B génère parfois des entités avec plus de champs
  que le format attendu par LightRAG
- Tentative 1 : changer de modèle LLM
  -> J'ai essayé de tester : mistral, llama3:8b et llama3.1:8b, plusieurs variantes
  -> Résultat : problème non résolu, résultats parfois pires qu'avec llama3.1:8b
- Tentative 2 : ajouter un prompt personnalisé au LLM pour préciser
  son rôle exact et le format attendu
  -> Résultat : warnings toujours présents, amélioration marginale insuffisante
- Décision finale : garder llama3.1:8b + accepter les warnings
  -> Impact réel faible : LightRAG ignore automatiquement les entités
  mal formatées et continue l'indexation normalement
  -> Les entités bien formatées (majorité) sont correctement indexées
  -> Amélioration des prompts prévue au Sprint 2

**26/04/2026 — Résultats de la première indexation complète**

- Indexation terminée avec Ollama local (llama3.1:8b + nomic-embed-text)
- Graphe exporté vers Neo4j Browser avec succès
- Résultats du graphe :
  -> Nœuds avec nom, id, description corrects
  -> Relations RELATES_TO entre entités visibles
  -> Requêtes Cypher exécutées avec succès :
  `MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 50`
  `MATCH (n) RETURN n.name, count(r) ORDER BY degree DESC LIMIT 10`
- Évaluation : bon résultat pour une première tentative avec PDFs,
  mais qualité insuffisante dans l'extraction des entités et relation

**26/04/2026 — Résumé des problèmes et solutions Semaine 3**

| Problème                     | Solution retenue                                |
| ---------------------------- | ----------------------------------------------- |
| HTTP 429 arXiv               | Kaggle + pauses 90s                             |
| 8000 tokens -> blocage       | Réduction à 500 tokens                          |
| Groq quota épuisé en 20 docs | Ollama local + checkpoint                       |
| embedding_dim=1024 vs 768    | Correction + réinstallation LightRAG            |
| list vs numpy array          | np.vstack(embeddings)                           |
| Warnings LLM format          | Accepté (impact faible) + amélioration Sprint 2 |

## SPRINT 1 — Semaine 4 (29 Avril – 04 Mai 2026)

### Amélioration d'indexation LightRAG + Analyse graphe Neo4j (en utilisant que le titre + abstract)

**28/04/2026 — Décision : abandonner les PDFs**

- Après analyse des résultats et lecture des recommandations
  du Prof. Hessane (document du 28/04/2026)
- Constat : les PDFs apportent du bruit sans valeur ajoutée réelle
  -> Parsing bruité : formules, numéros de pages, en-têtes LaTeX
  -> Stockage excessif : 493 PDFs ≈ 5-15 Go sur SSD
  -> Quota Groq épuisé car full_text trop long
  -> L'abstract seul contient toutes les entités clés pour le Knowledge Graph
- Décision finale : refaire le projet depuis zéro avec :
  -> Abstracts uniquement via l'API arXiv (0 PDF téléchargé)
  -> Structure de projet conforme aux recommandations de Prof
  -> Groq API avec retry automatique avec backoff exponentiel et envoi uniquement du titre + abstract
  -> Chunk size réduit : ~300-500 tokens/doc au lieu de 4500

**29/04/2026 - Mise en place de l'infrastructure et Collecte**

- Restructuration du projet : Refonte complète de l'arborescence pour suivre les standards de développement.
- Préparation de l'environnement :
  - Configuration du Notebook de travail (Sprint1_Fondations.ipynb) et installation des dépendances nécessaires.
- Acquisition des données :
  - Développement du script de téléchargement pour arXiv : récupération de 500 abstracts ciblés sur l'IA (cs.AI).
  - Téléchargement du benchmark HotpotQA via HuggingFace (7405 questions multi-sauts) pour les futures phases d'évaluation.
- Prétraitement des données :
  - Nettoyage du corpus arXiv (suppression des caractères spéciaux, normalisation).
  - Mise en place du Chunking : segmentation intelligente des textes pour optimiser l'extraction par
    le LLM (512 tokens avec un overlap de 50 tokens).
- Indexation (Phase 1) : Initialisation et lancement de la première phase d'indexation LightRag en utilisant l'API Groq
  pour l'extraction rapide des entités et des relations de base.

**30/04/2026**

- Indexation (Phase 2) : Lancement de la deuxième phase d'indexation avec LightRAG. Enrichissement du graphe
  de connaissances et gestion de la persistance des index (VDB et GraphML).
- Gestion du code :
  - Nettoyage des scripts de test.
  - Push Git : Mise à jour du dépôt distant avec la nouvelle structure de projet et le dernier notebook.
- Documentation et Rapport :Début de la rédaction du rapport de Sprint 1 .

- Pour gérer les interruptions dues aux limites de l’API Groq (rate limits), à chaque fois que le processus se bloque,
  j’arrête l’exécution puis je relance l’indexation à partir du point où elle s’était arrêtée, après le délai de réinitialisation des quotas en utilisant le mécanisme de checkpointing.

- Une tentative d’amélioration du pipeline a été réalisée en explorant l’intégration de ChromaDB comme base
  vectorielle afin de stocker directement les embeddings avec un LLM local (Ollama) et renforcer la structuration du graphe.
- Problème rencontré: la version de LightRAG utilisée ne supporte pas nativement ChromaDB, ce qui a empêché
  son intégration directe.
- La solution adoptée: consiste donc à conserver le pipeline LightRAG existant pour la génération du graphe
  de connaissances, tout en ajoutant une couche supplémentaire d’export des chunks vers une base vectorielle ChromaDB afin d’assurer la persistance et l’exploitation hybride des données.

**01/05/2026 – Analyse du graphe et amélioration du pipeline LightRAG**

### Avancement du pipeline

- 100 documents indexés avec succès dans LightRAG
- Graphe généré : 1355 nœuds, 620 relations
- Export vers Neo4j réussi
- Tests Naive et Hybrid fonctionnels

### Architecture & stockage

- ChromaDB utilisé uniquement pour stocker les embeddings, mais Retrieval est basé uniquement sur LightRAG c'est à dire qu'ont utilise graph + vector interne pour faire la recherche
- ChromaDB pas encore intégré dans la recherche

### LLM utilisé

- Remplacement de Groq par un LLM local
- Objectifs :
  - éviter les rate limits
  - améliorer la reproductibilité
  - stabiliser le pipeline

### Problèmes identifiés

- Graphe peu dense (0.0007) malgré un grand nombre de nœuds
- Peu de relations entre entités
- Forte dépendance du mode Hybrid aux relations du graphe

### Impact

- Réponses Hybrid plus courtes, mais plus précises
- Manque de contexte riche à cause d’un graphe sous-connecté

---

## Comparaison Naive vs Hybrid

| Critère           | Naive          | Hybrid             |
| ----------------- | -------------- | ------------------ |
| Type              | Vector search  | Graph + Vector     |
| Précision         | Bonne          | Très bonne         |
| Contexte          | Moyen          | Faible à moyen     |
| Longueur          | Plus détaillée | Plus courte        |
| Multi-hop         | Faible         | Meilleur potentiel |
| Dépendance graphe | Non            | Forte              |

### Analyse des résultats et synthése de causes du problème Hybrid

- Forte dépendance au graphe de connaissances
- Peu de relations extraites entre entités
- Chunking et extraction limités
- Filtrage trop strict des relations

### Solutions et améliorations testées / envisagées

- Optimisation de chunking (taille + overlap) et réduction de filtrage des relations faibles -> c'est la solution que j'ai testé

-> Pour les solutions prochaines :

- Améliorer le prompt LLM (relations explicites + implicites) même si j'ai déjà essayé avec un prompt mais j'ai pas eu d'amélioration
- Renforcer l’architecture LightRAG pour repenser la construction du graphe avec une meilleure fusion entités / relations
- Intégrer ChromaDB dans le retrieval (vrai hybride)

### Résumé de prochaines étapes

- Améliorer extraction entités/relations
- Enrichir la densité du graphe
- Intégrer ChromaDB dans le système de recherche
- Transformer le pipeline en véritable système hybride Graph + Vector(chromadb)

**02/05/2026**

- Résout d'un problème sur mon PC
- Términer le rapport de sprint 1

**03/05/2026**

- Préparation de la présentation et vérification de tous les livrables demandés

**04/05/2026**

- Faire le dérnier push des modifications
- Envoyer le rapport et les livrables demandés pour le Sprint 1

---

## Tableau comparatif RAG vs GraphRAG vs Agentic RAG

| Critère             | Naive RAG            | GraphRAG            | Agentic RAG (mon PFE) |
| ------------------- | -------------------- | ------------------- | --------------------- |
| Précision multi-hop | Faible (1 recherche) | Bonne (communautés) | Excellente (itératif) |
| Coût tokens/requête | Faible (~1K tokens)  | Très élevé (610K)   | Moyen (2-5K)          |
| Complexité impl.    | Simple               | Complexe            | Très complexe         |
| Mise à jour corpus  | Difficile            | Difficile           | Facile (LightRAG)     |
| Raisonnement        | Aucun                | Partiel             | Complet (auto-eval)   |
| Outil principal     | ChromaDB             | GraphRAG Microsoft  | LightRAG + LangGraph  |

---

## SPRINT 2 - Pipeline d'indexation hybride : Vecteurs + Graphe de connaissances

### Semaine 1 (05–11 Mai 2026)

## 05/05/2026 — Analyse approfondie de LightRAG

- Étude détaillée des différents modes de recherche proposés par LightRAG :
  - Naive ,Local ,Global ,Hybrid
- Analyse de l'architecture interne :
  - Pipeline d'extraction des entités et relations
  - Construction du graphe de connaissances
  - Fonctionnement du retrieval vectoriel interne
  - Mécanisme de fusion des informations issues du graphe et des vecteurs

- Lecture du code source LightRAG afin de comprendre :
  - les paramètres d'extraction
  - les mécanismes de chunking
  - les stratégies de récupération de contexte

### 06/05/2026 — Premiers tests d'amélioration du graphe et augmentation de la couverture d'extraction

Après analyse des résultats du Sprint 1, plusieurs problèmes persistent :

- faible densité du graphe
- nombreuses entités isolées
- relations peu informatives
- warnings fréquents lors de l'extraction

Afin de valider rapidement les modifications, tous les tests sont d'abord réalisés sur :

- 50 documents , puis 100 documents

### Objectif

- vérifier la stabilité du pipeline
- mesurer l'impact des modifications avant de lancer une indexation complète

### Modifications des paramètres d'extraction

| Paramètre                   | Ancienne valeur | Nouvelle valeur |
| --------------------------- | --------------- | --------------- |
| max_tokens_extract          | 300             | 1800            |
| entity_extract_max_gleaning | 0               | 4               |
| chunk_token_size            | 512             | 700             |
| max_extract_input_tokens    | 512             | 6000            |

### Objectif

- permettre au LLM d'extraire davantage d'entités
- augmenter le nombre de relations
- réduire le nombre de nœuds isolés
  J'ai utilisé das ce jour là Groq API mais à cause de ces limites j'ai décidé d'utiliser le LLM local qui est plus stable et donne moins de warnings

### 07 - 09 Mai 2026

- L'indexation de graphe en local a pris beaucoup de temps pour tous le corpus (500)
- Une fois l'indexation s'est terminer j'ai remarqué encore qu'il existe beaucoup des noeuds isolés comme si chaque chunks fait son graph isolé

### 09/05/2026 — Normalisation des entités et réduction des warnings de format LightRAG

##### Problème observé

La même entité apparaissait sous plusieurs formes :

- LLM ,LLMs ,Large Language Model , Large Language Models

#### Conséquence

- fragmentation du graphe
- création de plusieurs nœuds représentant le même concept

#### Solution testée

- mise en place d'une table de synonymes (_Canonical Aliases_)
- normalisation systématique des noms d'entités avant insertion dans le graphe

### Réduction des warnings de format LightRAG

#### Problème

Warnings récurrents :

```text
LLM output format error; found 5/4 fields on ENTITY
```

#### Analyse

Certaines sorties générées par Llama-3.1-8B ne respectent pas exactement le format attendu par LightRAG.

#### Solutions testées

- nettoyage automatique des sorties LLM
- correction des lignes mal formées
- ajout automatique des délimiteurs manquants
- contrôle du format avant insertion dans le graphe

#### Résultat

- nette diminution des warnings sur les tests 50 puis 100 documents
- pipeline plus stable

### 09 - 12 Mai 2026

- Puis j'ai relancé la réindéxation de tous le corpus

### Correction API Ollama et réindexation complète du corpus

### Correction d'un problème d'API Ollama

#### Problème

```text
404 Client Error:
http://localhost:11434/v1/chat/completions
```

#### Cause

- endpoint OpenAI non compatible avec la version installée d'Ollama

#### Solution

- utilisation exclusive de l'endpoint natif : /api/chat
- reprise normale de l'indexation

---

### 13/05/2026 — Réindexation complète du corpus

Les améliorations suivantes sont conservées :

- augmentation du nombre de tokens d'extraction
- passes multiples de gleaning
- normalisation des entités
- filtrage des faux nœuds
- amélioration des prompts
- nettoyage des sorties LLM

#### Conséquence

- augmentation importante du coût d'extraction
- davantage d'entités et relations générées

#### Temps d'exécution observé

- ancienne version : environ 2 jours continus
- nouvelle version : environ 3 à 4 jours continus (jour et nuit)

#### Cause de l'augmentation du temps

- hausse du nombre de tokens
- multiples passes d'extraction
- traitements de normalisation supplémentaires

### 16 Mai 2026

- Import de graphe dns Neo4j et analyse de graphes avec les requêtes
- J'ai remarqué que le degré moyen est encore < 3 (=1) et que le graphe est encore n'est pas connécté

### 17 - 18 Mai 2026

## Problèmes rencontrés avec RAGAS (installation + environnement) et solutions

### 1) `ModuleNotFoundError: No module named 'ragas'`

- **Contexte**: import `from ragas import evaluate` dans le notebook.
- **Cause**: package `ragas` non installé dans l’environnement du kernel Jupyter (venv du projet).
- **Solution testée**:
  - Installer via l’interpréteur du kernel (pas via `pip` du terminal) :
    - `!{sys.executable} -m pip install -U ragas`
- **Résultat**: `ragas` importable après installation.

---

### 2) `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` au moment de `from ragas import evaluate`

- **Contexte**: RAGAS plante à l’import dans `ragas.llms.base` (import de `ChatVertexAI`).
- **Cause**: incompatibilité entre la version de `ragas` et les versions installées de `langchain-community` (module déplacé/supprimé selon versions).
- **Solutions testées / proposées**:
  - Tentative de réinstallation/upgrade de `ragas`:
    - `!{sys.executable} -m pip install -U --no-cache-dir --force-reinstall ragas`
  - Workaround “pin” côté `langchain-community`:
    - `!{sys.executable} -m pip install --no-deps --ignore-installed --no-cache-dir "langchain-community==0.4.1"`
    - `langchain-community==0.4.0`
- **Résultat**: l’import de `ragas` a fini par fonctionner après alignement des versions (pin / réinstall).

---

### 3) `OpenAIError: Missing credentials ... set OPENAI_API_KEY ...` lors de `evaluate(...)`

- **Contexte**: `ragas_result = evaluate(dataset, metrics=[...])` appelé sans `llm`/`embeddings`.
- **Cause**: si `llm` est `None`, RAGAS utilise OpenAI par défaut (il instancie `OpenAI()`), donc exige une clé API.
- **Solutions testées**:
  - **Solution (locale, sans OpenAI)**: fournir explicitement le LLM et les embeddings (Ollama) à `evaluate()` :
    - `evaluate(..., llm=llm, embeddings=emb_fn)`
- **Résultat**: plus de demande de `OPENAI_API_KEY` (RAGAS utilise Ollama).

---

### 4) `DeprecationWarning` sur `from ragas.metrics import ...`

- **Contexte**: warnings indiquant que l’import depuis `ragas.metrics` sera supprimé en v1.0.
- **Cause**: changement d’API côté RAGAS.
- **Solution testée**:
  - Remplacer par :
    - `from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall`
- **Résultat**: warnings supprimés (non bloquant).

---

### 19 Mai 2026

- J'ai fais l'evaluation de lightrag en mode hybrid et les résultats ont été nuls et aucun contexte était récupéré
- Décision de refaire l'indexation de graphe dés le début avec d'autre amélioration , mais aprés faire l'extraction NER + spacy et Rebell pour ne pas perdre de temps

### Étude des approches NER et Relation Extraction

Objectif :

- obtenir davantage de relations
- comparer qualitativement les triplets avec ceux produits par LightRAG
- améliorer la couverture relationnelle du graphe
  Technologies étudiées :
- spaCy , GLiNER ,REBEL

Décision finale :

- utiliser une approche : spaCy + REBEL + rappel++
- afin d'obtenir un pipeline totalement indépendant de LightRAG , j'ai tésté d'abord un pipeline spacy + REBEL avant cette décision finale

---

## 20/05/2026 — Développement du pipeline spaCy + REBEL

Architecture retenue :

- segmentation des documents avec spaCy
- extraction des relations avec REBEL
- récupération automatique des entités à partir des champs head et tail
- export JSON compatible avec le format utilisé dans le projet

Pipeline :

Document -> spaCy Sentencizer -> Fenêtres de phrases (rappel++) -> REBEL -> Triplets (head, relation, tail) -> JSON

### Problèmes rencontrés

#### 1. REBEL retourne 0 triplet

- Cause : mauvais parsing du format de sortie
- Solution : correction complète du parser selon le format officiel Babelscape

#### 2. Triplets trop peu nombreux

- Cause : contexte trop limité
- Solution :
  - mise en place de la stratégie rappel++
  - fenêtres de phrases avec overlap

---

## 21/05/2026 — Validation du pipeline

- génération des fichiers JSON de triplets
- comparaison qualitative avec les résultats de LightRAG

### Résultat

- augmentation importante du nombre de triplets extraits par document
- pipeline spaCy + REBEL fonctionnel et stable

### Conclusion

Le pipeline NER + REBEL permet d'obtenir une extraction complémentaire et offre une base solide pour comparer la qualité des relations générées par LightRAG lors de la suite du sprint 2.

### 22 - 23 Mai 2026 - 1ére tentation de comparaison et indexation de corpus

- J'ai fait une première tentation de comparaison et validation mannuelle entre les triplets de lightrag et les triplets extrait avec spacy, mais puisque le graphe n'est pas encore bon les résultats sont nuls.
- Aussi l'indexation prends beaucoup du temps , donc j'ai décidé de faire les embeddings de tous le corpus et les stocker dans chromadb pour les préparer pour le RAG baseline , puis implémenter un Rag baseline et l'valuer sur les 20 questions.

---

## 23/05/2026 — Indexation de corpus avec ChromaDB et Docker Compose

### Objectif

Construire un pipeline complet d'indexation de corpus avec stockage des embeddings en **ChromaDB**, orchestré via **Docker Compose** pour garantir une reproductibilité et une scalabilité du système.

### Architecture retenue

**Pipeline global:**
Corpus arXiv (500 abstracts) -> Chunking & Tokenization -> Embedding (Ollama: nomic-embed-text) -> ChromaDB (persistent storage) -> Vector Index (prêt pour RAG/Retrieval)

### Technologie utilisée

| Composant          | Rôle            | Détail                             |
| ------------------ | --------------- | ---------------------------------- |
| **Docker**         | Orchestration   | Isolation des services             |
| **Docker Compose** | Composition     | `docker-compose.rag-baseline.yml`  |
| **Ollama**         | Embedding Model | `nomic-embed-text` (768 dim)       |
| **ChromaDB**       | Vector Store    | Stockage persistant des embeddings |
| **Python Script**  | ETL             | `build_chroma_pfe500.py`           |

### Démarrage du système

#### Commande de lancement

```bash
# Dans le répertoire racine du projet
docker compose -f docker-compose.rag-baseline.yml up --build

# Explications:
# - up : démarrer les services définis
# - --build : reconstruire les images Docker
```

### 27 Mai 2026 - Amélioration de la qualité des entités et relations pour lightrag

### Problèmes identifiés

#### 1. Titres de papers extraits comme entités

Exemples : titres complets d'articles transformés en nœuds

#### Conséquences :

- augmentation artificielle du nombre de nœuds
- faible connectivité du graphe

#### Solution :

- ajout de filtres supprimant les titres trop longs
- enrichissement du prompt d'extraction

---

#### 2. Relations trop génériques

Exemple : related_to

#### Solution :

- ajout d'instructions explicites dans le prompt
- demande de relations plus informatives :
  - uses
  - improves
  - trained_on
  - evaluates_on
  - based_on
  - applied_to

---

#### — Réindexation complète et bilan

Après validation sur :

- 50 documents
- puis 100 documents

### Stratégie de validation

- test initial sur **50 documents**
  - vérification des problèmes de warnings
  - correction des erreurs de parsing et de code

- test intermédiaire sur **100 documents**
  - validation de la stabilité du pipeline

- si validation correcte :
  - lancement de l’indexation complète (500 documents)

---

## Paramètres finaux utilisés

| Paramètre                   | Valeur |
| --------------------------- | ------ |
| CHUNK_TOKEN_SIZE            | 700    |
| CHUNK_OVERLAP               | 120    |
| MAX_EXTRACT_INPUT_TOKENS    | 6000   |
| ENTITY_EXTRACT_MAX_GLEANING | 6      |
| MAX_TOKENS_EXTRACT          | 1800   |
| OLLAMA_NUM_CTX              | 8192   |
| LLM_MAX_ASYNC               | 1      |
| MAX_PARALLEL_INSERT         | 1      |
| SLEEP_BETWEEN_DOCS_S        | 0.2    |

---

### 30 - 01 Juin 2026 - Réindexation compléte de tous le corpus

- Lancer l'indéxation sur tous le corpus

## Résumé du pipeline final

- extraction améliorée des entités (filtrage des titres)
- relations enrichies et plus spécifiques
- normalisation des entités (aliases canoniques)
- suppression des nœuds bruités
- correction des erreurs de parsing LLM
- réindexation complète stable du corpus

---

## Résultats obtenus

- amélioration de la qualité globale du graphe
- diminution d'une partie des warnings
- meilleure cohérence des entités
- graphe plus connecté
- degré moyen du graphe augmenté à **2**

---

### 02 - 04 Juin 2026

- Problème technique dans le PC

### 05 Juin 2026

- J'ai refais l'évaluation de lightrag et la comparaison entre les triplets des 2 méthodes , j'ai remarqué que les résultats réstent encore nuls , voici les problèmes remarqués :

### Diagnostic critique : Évaluation LightRAG Hybrid + Comparaison avec REBEL

**ANALYSE CRITIQUE DES RÉSULTATS**

#### Trois problèmes majeurs identifiés après évaluation

---

## PROBLÈME 1 : Retrieval complètement cassé - [no-context] sur tous les queries

### Description

- **Symptôme** : Toutes les questions retournent `"Sorry, I'm not able to provide an answer to that question.[no-context]"`
- **Latences** : 30-39 secondes (OK)
- **Contexte retrouvé** : 0
- **Nodes traversed** : 0 (le graphe n'est pas consulté du tout)

### Causes probables

1. Index LightRAG mal indexé (malgré 5433 entités et 5069 relations dans les fichiers)
2. Mismatch entre embedding_func utilisée lors de l'indexation et celle du retriever
3. Paramètres retrieval insuffisants (top_k trop bas, chunk_top_k limité)
4. Erreur silencieuse lors de l'insertion dans le VDB (Vector Database)

### Solutions à tester

#### Solution 1A : Vérifier l'intégrité de l'index

```python
import os

# Vérifier la taille de l'index
index_size = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, filenames in os.walk(INDEX_DIR)
    for f in filenames
)
print(f"Index size: {index_size / 1024 / 1024:.2f} MB")

# Charger les stores directement
import json
rel_store = json.loads((INDEX_DIR / "kv_store_full_relations.json").read_text())
ent_store = json.loads((INDEX_DIR / "kv_store_full_entities.json").read_text())
text_store = json.loads((INDEX_DIR / "kv_store_text_chunks.json").read_text())

print(f"Relations stored: {len(rel_store)}")
print(f"Entities stored: {len(ent_store)}")
print(f"Text chunks stored: {len(text_store)}")

# Si 0 -> l'indexation a échoué silencieusement
```

#### Solution 1B : Tester le retriever directement

```python
# Test direct sans passer par RAG.aquery
test_query = "machine learning classification methods"

# 1. Vérifier l'embedding
test_emb = await embedding_func([test_query])
print(f"Query embedding shape: {test_emb.shape}")  # doit être (1, 768)

# 2. Vérifier la recherche dans le VDB
try:
    results = await rag.aquery(
        test_query,
        param=QueryParam(mode='local', top_k=10)  # plus simple que hybrid
    )
    print(f"Query results: {results}")
except Exception as e:
    print(f"Error: {e}")
```

#### Solution 1C : Augmenter les paramètres de recherche

```python
# Dans run_benchmark(), augmenter les paramètres
async def run_benchmark_fixed(rag):
    rows = []
    for q in QUESTIONS:
        t0 = time.time()
        # ↑ Augmenter ces paramètres
        res = await rag.aquery(
            q,
            param=QueryParam(
                mode='hybrid',
                top_k=100,              #  de 40 à 100
                chunk_top_k=80,         #  de 20 à 80
                enable_rerank=True,     #  activer reranking
                similarity_threshold=0.1 #  baisser le seuil
            )
        )
        lat = round(time.time() - t0, 2)

```

#### Solution 1D : Vérifier la compatibilité des embeddings

```python
# Tester que l'embedding_func fonctionne bien
test_texts = ["test query", "another test", "third sample"]
test_embs = await embedding_func(test_texts)

print(f"Embeddings shape: {test_embs.shape}")  # doit être (3, 768)
print(f"Embedding dtype: {test_embs.dtype}")   # doit être float32
print(f"Embedding norms: {[np.linalg.norm(e) for e in test_embs]}")  # vérifier non-zero
```

---

## PROBLÈME 2 : Aucun overlap entre LightRAG et REBEL (0 exact matches)

### Description

| Métrique                        | Valeur        |
| ------------------------------- | ------------- |
| **LightRAG triplets uniques**   | 5066          |
| **REBEL triplets uniques**      | 2788          |
| **Exact overlap**               | **0**         |
| **Entity pair overlap (loose)** | **31** (0.6%) |

**Conclusion** : Les deux systèmes extraient des relations **complètement différentes**.

### Causes probables

1. Relations LightRAG sont verbeux et mal structurés (7-8 mots par relation)
2. REBEL extrait des relations propres et standardisées (1-2 mots)
3. Le LLM local crée des relations contextuelles au lieu de sémantiques
4. Pas de canonicalization/nettoyage après extraction du LLM

### Solutions à tester

#### Solution 2A : Enforcer les prompts d'extraction (PRIORITÉ HAUTE)

```python
#  Logger JSON strict + taux de parsing"
try:
    from lightrag.prompt import PROMPTS
    TUPLE_DELIM = PROMPTS.get('DEFAULT_TUPLE_DELIMITER', '<|#|>')
    COMPLETE_DELIM = PROMPTS.get('DEFAULT_COMPLETION_DELIMITER', '<|COMPLETE|>')

    #  NOUVEAU PROMPT STRICTE
    PROMPTS['entity_extraction_system_prompt'] += """
---STRICT PREDICATE LIST---
You MUST use ONLY these precise predicates:
- uses (Entity A uses Entity B)
- improves (Entity A improves Entity B)
- evaluates_on (Entity A evaluates_on Entity B)
- trained_on (Entity A trained_on Entity B)
- compares_with (Entity A compares_with Entity B)
- extends (Entity A extends Entity B)
- applies_to (Entity A applies_to Entity B)
- related_to (ONLY as fallback)

FORBIDDEN:
- Multi-word relations (NO "model application task")
- Contextual relations (NO embedding context in predicate)
- Vague relations

STRICT FORMAT:
relation<|#|>Source Entity<|#|>Target Entity<|#|>PREDICATE<|#|>brief_description<|COMPLETE|>

Example CORRECT: relation<|#|>Graph Neural Networks<|#|>Node Embedding<|#|>improves<|#|>GNNs improve node embeddings
Example WRONG: relation<|#|>Model<|#|>Task<|#|>model application task<|#|>...
"""

except Exception:
    TUPLE_DELIM = '<|#|>'
    COMPLETE_DELIM = '<|COMPLETE|>'

print(' Strict prompts loaded')
```

#### Solution 2B : Renforcer la sanitization

```python
# Améliorer sanitize_extraction_output() fonction

VALID_RELATIONS = {
    "uses", "improves", "evaluates_on", "trained_on",
    "compares_with", "extends", "applies_to", "related_to"
}

def sanitize_extraction_output_v2(text):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    entities = {}
    relations = []

    for ln in lines:
        if ln == COMPLETE_DELIM:
            break

        parts = ln.split(TUPLE_DELIM)
        if not parts:
            continue

        if parts[0] == "entity":
            if len(parts) == 3:
                parts = parts + [""]
            if len(parts) < 4:
                continue

            name = canonicalize_entity_name(parts[1])
            if is_bad_entity(name):
                continue

            ent_type = parts[2].strip() or "Concept"
            desc = parts[3].strip()
            entities[name] = (ent_type, desc)

        elif parts[0] == "relation":
            if len(parts) == 4:
                parts = parts + [""]
            if len(parts) < 5:
                continue

            src = canonicalize_entity_name(parts[1])
            tgt = canonicalize_entity_name(parts[2])

            if src == tgt or is_bad_entity(src) or is_bad_entity(tgt):
                continue

            pred = parts[3].strip().lower()

            #  NOUVEAU : VALIDER LE PREDICAT
            if pred not in VALID_RELATIONS:
                pred = "related_to"  # fallback strict

            desc = parts[4].strip()
            relations.append((src, tgt, pred, desc))

    # Garder seulement les entités utilisées dans les relations
    used_entities = {s for s, _, _, _ in relations} | {t for _, t, _, _ in relations}

    fixed = []
    for name in sorted(used_entities):
        ent_type, desc = entities.get(
            name,
            ("Concept", f"{name} is mentioned in a relation."),
        )
        fixed.append(f"entity{TUPLE_DELIM}{name}{TUPLE_DELIM}{ent_type}{TUPLE_DELIM}{desc}")

    seen_rel = set()
    for src, tgt, pred, desc in relations:
        key = tuple(sorted([src, tgt])) + (pred,)
        if key in seen_rel:
            continue
        seen_rel.add(key)
        fixed.append(f"relation{TUPLE_DELIM}{src}{TUPLE_DELIM}{tgt}{TUPLE_DELIM}{pred}{TUPLE_DELIM}{desc}")

    fixed.append(COMPLETE_DELIM)
    return "\n".join(fixed)

# Remplacer l'appel dans local_llm_func
if extraction:
    out = sanitize_extraction_output_v2(out)
    triplet_logger.parse_extraction(out, doc_id=CURRENT_DOC_ID)
```

#### Solution 2C : Réduire la longueur des extractions

```python
# Dans local_llm_func, réduire num_predict pour éviter l'hallucination
async def local_llm_func(prompt, system_prompt=None,
                          history_messages=[], **kwargs):
    # ici je conserve le même code existant

    extraction = is_extraction_call(system_prompt)
    #  Réduire de 1800 à 800 tokens
    num_predict = 800 if extraction else MAX_TOKENS_QUERY

```

---

## PROBLÈME 3 : Qualité inférieure de LightRAG vs REBEL

### Description

| Aspect                | LightRAG                          | REBEL                  |
| --------------------- | --------------------------------- | ---------------------- |
| **Entités totales**   | 5433                              | 3946                   |
| **Relations totales** | 5069                              | 2821                   |
| **Exact overlap**     | 0%                                | N/A                    |
| **Qualité relations** | **Mauvaise**                      | **Bonne**              |
| **Top relation**      | "model application task" (7 mots) | "instance of" (2 mots) |
| **Parsing success**   | ~100%                             | 100%                   |

**Analyse** : LightRAG extrait PLUS mais de QUALITÉ BEAUCOUP PLUS FAIBLE.

### Causes

1. LightRAG dépend du LLM local (llama3.1:8b) qui crée du bruit
2. REBEL est un modèle entraîné spécifiquement (Sequence-to-Sequence), ce qui est plus robuste
3. Paramètres gleaning trop élevés (6) -> trop de passe LLM = hallucinations
4. Context size trop grand (6000 tokens)

### Solutions à tester

#### Solution 3A : Réduire les paramètres d'hallucination

```python
# Dans la cellule de configuration

#  Réduire le gleaning (moins de pass LLM = moins d'hallucinations)
ENTITY_EXTRACT_MAX_GLEANING = 3  #  de 6 à 3

#  Réduire le contexte d'extraction (moins de contexte = moins de bruit)
MAX_EXTRACT_INPUT_TOKENS = 3000  #  de 6000 à 3000

#  Augmenter la taille des chunks (plus de contexte local = moins de fragments)
CHUNK_TOKEN_SIZE = 1000  #  de 700 à 1000

# Réduire l'overlap
CHUNK_OVERLAP = 100  #  de 120 à 100

print(" Configuration optimisée pour qualité (moins d'hallucinations)")
```

#### Solution 3C : Utiliser un meilleur modèle local

```python
# Remplacer le modèle dans .env ou directement
# Option 1 : Neural Chat (meilleur pour extraction)
MODEL_NAME = "neural-chat:7b"

# Option 2 : Mistral (plus puissant)
MODEL_NAME = "mistral:7b"

# Option 3 : Llama 3 plus puissant (si ressources)
MODEL_NAME = "llama2:13b"
print(" Modèle local changé pour meilleure qualité d'extraction")
```

---

## PLAN D'ACTION — Les solutions que je voulais tester aprés l'aide du prof

- [ ] **Solution 1** : Vérifier l'intégrité de l'index
- [ ] **Solution 2** : Tester retriever directement
- [ ] **Solution 3** : Augmenter paramètres de recherche + Enforcer prompts + Renforcer sanitization
- [ ] **Solution 4** : Réduire length extractions + Réduire paramètres hallucination
- [ ] Ré-indexer avec nouvelles configurations
- [ ] **Solution 5** : Tester modèles alternatifs + Évaluer les résultats avec RAGAS +Comparer amélioration

---

## Résumé des problèmes et solutions

| #   | Problème                     | Cause                     | Complexité |
| --- | ---------------------------- | ------------------------- | ---------- |
| 1   | Retrieval cassé [no-context] | Index cassé               | Moyenne    |
| 2   | 0 overlap LightRAG vs REBEL  | Relations mal structurées | Élevée     |
| 3   | Qualité inférieure LightRAG  | Hallucinations LLM        | Moyenne    |

---

## Problème — Scores à 0 avec HotpotQA (le RAG utilise mon corpus)

### Symptôme

- Pour toutes les questions : `retrieval_multi_hop_failure` et `recall = 0.0`
- `EM = 0.0` et métriques proches de 0

### Cause (racine)

- Le retriever interroge bien **uniquement mon index Chroma (corpus PFE)**.
- Mais le dataset d’évaluation utilisé (`hotpot_qa`) est basé sur **Wikipédia** :
  - `supporting_facts["title"]` et `answer` correspondent à des pages Wikipédia.
- Mon calcul de recall compare :
  - `retrieved_titles` (metadata/titres de mon corpus PFE, souvent `None` ou non-Wiki)
  - vs `supporting_facts["title"]` (titres Wikipédia)
- Donc l’intersection est vide , `recall = 0` partout.
- Plus généralement : beaucoup de questions HotpotQA n’ont **pas de réponse dans mon corpus**, donc l’évaluation n’est pas alignée.

### Impact

- Les métriques `recall/EM` calculées avec HotpotQA ne reflètent pas la qualité du RAG sur mon corpus.

### Solution / Correctif à appliquer

- Utiliser un dataset de test **construit à partir de mon corpus** :
  - questions + `ground_truth` provenant des documents PFE (même 20 questions manuelles), ou
  - générer un dataset de test depuis le corpus puis évaluer avec RAGAS.

- Terminer le rapport + livrables demandés
  **Statut** : À tester  
  **J'ai laissé ces codes dans le journal à fin de les tester, je voulais d'abord votre aide et remarques pour surmonter ce bloquage et aussi de ne pas être trés en retard dans les livrables\***
  **Prochaine étape** : Exécuter ces Solutions et avancer dans le projet pour récupérer le temps pérdus

_Journal mis à jour quotidiennement — Wiame Anejjar_
_Dernière mise à jour : 05 Juin 2026_
