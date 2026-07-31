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

# SPRINT 3 — Semaine 1 (08–14 Juin 2026)

## Résolution des problèmes du Sprint 2

L'objectif principal de cette semaine était de corriger les principaux problèmes identifiés lors de l'évaluation de LightRAG afin de pouvoir poursuivre le développement de l'Agentic GraphRAG sur une base plus stable.

---

## 08/06/2026 — Correction des problèmes LightRAG

### Problème 1 — Réponses tronquées du LLM

#### Problème

Les réponses générées par Ollama étaient parfois limitées à quelques mots tels que :

- "Based"
- "The"

#### Diagnostic

- Vérification du prompt envoyé au LLM.
- Vérification des paramètres `num_predict`.
- Analyse des logs Ollama.

#### Cause

Le nombre maximal de tokens générés était insuffisant.

#### Solution

- augmentation de `num_predict`
- ajustement des paramètres de génération

#### Résultat

Les réponses sont désormais complètes mais pas assez suffisant et pas assez correct.

---

### Problème 2 — Cache LightRAG

#### Problème

Après modification du code, LightRAG continuait à retourner les anciennes réponses.

#### Diagnostic

- analyse du cache LightRAG
- suppression puis reconstruction du cache
- vérification des clés utilisées

#### Solution

- nettoyage du cache
- reconstruction complète des index

-  Résultat : les mêmes réponses de cache sont encores là même si tous est supprimés et redémarés  .

---

### Problème 3 — Canonicalisation des entités

#### Problème

Une même entité apparaissait sous plusieurs formes.

Exemples :

- LLM
- LLMs
- Large Language Model
- Large Language Models

#### Diagnostic

Analyse des entités présentes dans le graphe Neo4j.

#### Solution

- amélioration de la table des alias canoniques
- normalisation des entités avant insertion dans le graphe

#### Résultat

Réduction des doublons et amélioration de la cohérence du graphe.

---

## 10/06/2026 — Étude de LangGraph

Afin de préparer le développement de l'Agentic GraphRAG, une étude du framework LangGraph a été réalisée.

### Travaux effectués

- étude de LangChain Academy (pas tout le cours car j'ai pas arrivé à comprendre dans ce  cours mais j'ai utilisé des vidéos youtube et ainsi des llms pour comprendre)
- compréhension du fonctionnement des agents
- étude du StateGraph
- compréhension des nœuds
- étude du routage conditionnel
- compréhension du partage d'état entre les nœuds

Cette étude a permis de préparer l'architecture de l'agent développée dans la suite du sprint.

---

## 11/06/2026 — Génération du benchmark d'évaluation

### Première version

Un premier générateur de benchmark a été développé à partir du corpus.

#### Résultat

Le benchmark obtenu contenait uniquement des questions pseudo multi-hop.

#### Limite

Les questions ne nécessitaient pas réellement plusieurs documents pour répondre.

---

### Deuxième version

Le générateur a été entièrement revu afin de produire de véritables questions multi-hop.

#### Résultat obtenu

Benchmark final :

- 150 questions
- 10 vraies questions multi-hop
- 140 questions pseudo multi-hop

Ce benchmark sera utilisé pour les évaluations quantitatives du projet, mais je vais l'améliorer ensuite.

---

## 12–14/06/2026 — Évaluation du RAG Baseline

Le benchmark généré a ensuite été utilisé pour évaluer le RAG vectoriel construit avec ChromaDB.

#### Diagnostic

- Analyse des documents récupérés par le retriever.
- Le système retrouve généralement un seul document support au lieu des deux documents nécessaires.
- Le RAG vectoriel classique ne réalise pas de raisonnement multi-hop.

---

#### Diagnostic pour savoir pourquoi Context Precision nul

- Analyse des chunks transmis à RAGAS.

#### Résultat

- Les chunks récupérés par ChromaDB ne correspondent pas aux chunks de référence utilisés dans le benchmark.
- Le problème provient d'une incompatibilité entre les index utilisés pour l'évaluation.

---

### Problème 3 — Difficulté à évaluer LightRAG

Une première tentative d'évaluation de LightRAG a été réalisée sur le benchmark.

#### Problèmes observés

- réponses incorrectes
- réponses incomplètes
- comportement incohérent selon les requêtes

#### Diagnostics réalisés

- vérification du retrieval
- vérification du contexte construit
- vérification de la transmission du contexte au LLM
- analyse des paramètres de requête
- analyse du fonctionnement du cache

#### Résultats

Les diagnostics montrent que :

- les entités sont correctement retrouvées ;
- les relations sont correctement retrouvées ;
- les chunks sont correctement récupérés ;
- le contexte final est correctement construit ;
- le contexte est correctement transmis au LLM.

#### Conclusion

Les problèmes observés ne proviennent pas principalement du module de retrieval mais plutôt de la génération des réponses ainsi que de certains comportements liés au cache et aux anciennes versions de `local_llm_func`.

---

## Bilan de la semaine

### Travaux réalisés

- Essaies de correction des problèmes identifiés au Sprint 2 ;
- amélioration de la canonicalisation des entités ;
- étude de LangChain Academy et de LangGraph ;
- génération d'un benchmark multi-hop adapté au corpus ;
- évaluation quantitative du RAG Baseline ;
- premiers diagnostics approfondis sur le comportement de LightRAG.

# SPRINT 3 — Semaine 2 (15–21 Juin 2026)

## Compréhension approfondie de LightRAG, développement de la première version de l'Agentic GraphRAG et poursuite des diagnostics

### 15/06/2026 — Poursuite des diagnostics sur LightRAG

Après les premiers diagnostics réalisés la semaine précédente, les résultats de LightRAG restaient insuffisants malgré un retrieval fonctionnel.

#### Travaux réalisés

- Analyse approfondie du pipeline de génération de LightRAG.
- Vérification du fonctionnement de `local_llm_func` et ajout de quelque print pour comprendre le processus.
- Comparaison entre les réponses produites par le RAG baseline et celles générées par LightRAG.
- Étude du code source afin d'identifier l'origine des mauvaises réponses.

#### Constat

- Le retrieval récupère correctement les entités, les relations et les chunks.
- Les réponses générées restent néanmoins incomplètes ou incorrectes.


---

### 17/06/2026 — Première version de l'Agentic GraphRAG

#### Travaux réalisés

Développement de la première architecture fonctionnelle de l'agent.

Architecture implémentée :

```text
QUERY
   │
GRAPH SEARCH
   │
RESPONSE
```

Trois premiers nœuds ont été développés :

- Query
- Graph Search
- Response

#### Résultat

- exécution complète du graphe fonctionnelle ;
- transitions entre les nœuds correctement réalisées ;
- premier workflow LangGraph opérationnel.

Cette première version constitue la base de l'architecture finale de l'Agentic GraphRAG.

---

### 18/06/2026 — Intégration de Phoenix pour le tracing

#### Travaux réalisés

Afin d'analyser le comportement de l'agent, Phoenix a été intégré au projet.

Actions réalisées :

- installation et configuration de Phoenix ;
- activation du tracing des appels LLM ;
- génération des premières traces d'exécution.

#### Étude réalisée

Une phase d'analyse a ensuite été consacrée à la compréhension des informations affichées dans Phoenix :

- déroulement des appels LLM ;
- visualisation des différentes étapes du workflow ;
- compréhension des traces générées par LangGraph.

Cette étape avait principalement pour objectif de se familiariser avec l'outil avant son utilisation pour le diagnostic des performances de l'agent.

---

### 19–20/06/2026 — Reprise de l'évaluation du RAG et de LightRAG

Malgré les corrections précédentes, les résultats obtenus restaient insuffisants.

#### Travaux réalisés

- nouvelle évaluation du RAG baseline sur le benchmark construit à partir du corpus et avec les nouveaux paramètres ;
- nouvelles expérimentations sur LightRAG ;
- comparaison détaillée entre les réponses attendues et les réponses générées ;
- analyse des métriques RAGAS.

Les résultats obtenus montrent que les performances restent encore insuffisantes et faibles pour une évaluation fiable.

---

## Problèmes rencontrés

### Problème 1 — Les réponses générées par LightRAG restent incorrectes malgré un retrieval valide

#### Diagnostic réalisé

- vérification des étapes de retrieval ;
- inspection du contexte transmis au LLM et tester avec un autre llm ;
- un diagnostic a été fait pour savoir est ce que llm qui se bloque lors de la génération de la réponse
- comparaison avec les réponses du RAG baseline.

#### Résultat

- retrieval correct ;
- contexte correctement construit ;
- problème localisé au niveau de la génération.

#### Solution

- poursuite de l'analyse du pipeline de génération ;
- étude approfondie de `local_llm_func` et de pipeline lightrag;
- préparation de nouveaux tests de génération.

---

### Problème 2 — Difficulté à comprendre les informations affichées dans Phoenix

#### Diagnostic réalisé

Après l'intégration de Phoenix, plusieurs informations (traces, spans, appels LLM, timings) étaient disponibles mais leur interprétation n'était pas encore maîtrisée.

#### Solution

- étude de la documentation Phoenix ;
- analyse progressive des différentes traces ;

---

### Problème 3 — Les métriques RAGAS restent faibles

#### Diagnostic réalisé

- réévaluation du benchmark et essayé de l'améliorer pour avoir un bon benchmark  ;
- analyse des réponses générées ;
- comparaison avec les réponses de référence.

---

# SPRINT 3 — Semaine 3 (22–28 Juin 2026)

## Développement de la version complète de l'Agentic GraphRAG et première évaluation

### 22/06/2026

- Reprise du développement de l'architecture de l'Agentic GraphRAG.
- Ajout des nœuds restants afin d'obtenir un workflow plus complet.
- Définition des transitions entre les différents nœuds.
- Mise en place des conditions permettant à l'agent de décider de poursuivre ou d'arrêter le raisonnement.
- Validation du bon fonctionnement de l'ensemble du graphe.

L'architecture obtenue est composée de cinq nœuds principaux :

- Query
- Graph Search + VECTOR_SEARCH
- Response
- Critique
- Finalize

---

### 23/06/2026

- Finalisation de l'implémentation de l'agent.
- Tests de plusieurs questions sur le benchmark.
- Vérification de l'exécution complète du workflow.
- Validation des transitions entre les nœuds.
- Vérification de la production des réponses finales.

L'agent est désormais capable de :

- Extraire les mots-clés de la question et interroger Neo4j via des requêtes Cypher multi-hop (0-hop entités seed, 1-hop relations directes, 2-hop multi-sauts)
- Effectuer une recherche sémantique dans ChromaDB pour récupérer les passages textuels pertinents
- Fusionner les deux sources (graphe + vecteurs) dans un contexte structuré via le nœud FUSE
- Générer une réponse synthétique via Ollama llama3.1:8b (ou Groq en option) dans le nœud RESPONSE
- Évaluer automatiquement la qualité de la réponse (complétude, fidélité, clarté) via le nœud CRITIQUE qui retourne un score entre 0 et 1
- Reformuler la question et relancer le cycle si le score est inférieur à 0.7, jusqu'à 3 itérations maximum (SELF_CORRECT)
- Finaliser et retourner la réponse validée via le nœud FINALIZE

---

### 24/06/2026

- Préparation du benchmark pour l'évaluation de l'Agentic GraphRAG.
- Adaptation du pipeline d'évaluation afin de pouvoir mesurer les performances de l'agent.
- Intégration des métriques RAGAS dans le processus d'évaluation.
- Lancement des premières expérimentations.

---

### 25/06/2026

- Évaluation complète de l'Agentic GraphRAG sur le benchmark construit à partir du corpus.
- Calcul des métriques RAGAS :
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall
- Analyse des réponses générées par l'agent.
- Comparaison qualitative avec les résultats obtenus précédemment sur le RAG baseline et LightRAG, même si les réponses de ces dérniers ne sont pas assez corrects.

---

### 26/06/2026

- Analyse détaillée des résultats obtenus.
- Étude des réponses générées pour chaque question du benchmark.
- Analyse de l'impact du nœud Critique sur le comportement global de l'agent.

---

## Problèmes rencontrés

### Problème 1 — Les performances de l'Agentic GraphRAG restent limitées

#### Diagnostic réalisé

- Analyse des métriques RAGAS obtenues.
- Étude des réponses générées pour chaque question.

#### Résultat

- certaines réponses sont pertinentes ;
- plusieurs réponses restent incomplètes ;
- les scores RAGAS demeurent insuffisants pour valider définitivement l'approche, même si l'evaluation de la qualité avant le critique donne un bon score.

#### Solution

- améliorer le benchmark d'évaluation ;
- poursuivre l'amélioration de LightRAG afin de fournir un meilleur contexte à l'agent puis l'utiliser au lieu de requête Neo4j;
- enrichir progressivement la stratégie de raisonnement de l'agent.

---

### Problème 2 — Les performances de l'agent dépendent fortement de la qualité du contexte récupéré

#### Diagnostic réalisé

Les analyses montrent que lorsque le contexte retourné est incomplet ou peu pertinent, la qualité des réponses de l'agent diminue également.

#### Résultat

Le raisonnement de l'agent reste limité par les informations fournies lors de la phase de retrieval.

#### Solution

- poursuivre les diagnostics de LightRAG ;
- améliorer la qualité du retrieval avant de réaliser une nouvelle campagne d'évaluation de l'agent.

---

## Bilan de la semaine

Au cours de cette semaine :

- développement de la version 1 complète de l'Agentic avec cinq nœuds ;
- validation du fonctionnement global du workflow LangGraph ;
- première évaluation quantitative de l'Agentic GraphRAG à l'aide des métriques RAGAS ;
- analyse détaillée des résultats obtenus ;
- identification des limites actuelles du système et des axes d'amélioration pour la suite du projet.

### Semaine 1 (29/06/2026 – 06/07/2026)

#### Travaux réalisés

- Analyse et diagnostic de l'état actuel du projet à partir des résultats obtenus lors du Sprint 3 et des recommandations de l'encadrant.
- Identification des principaux problèmes impactant les performances de l'Agentic GraphRAG, notamment le biais du nœud **CRITIQUE**, les hallucinations du LLM et le manque de cohérence entre les scores internes de l'agent et les métriques RAGAS.
- Modification du nœud **CRITIQUE** afin d'utiliser un modèle de jugement indépendant du modèle générateur, dans le but d'obtenir un score d'évaluation plus objectif et d'éviter le biais d'auto-évaluation.
- Amélioration du prompt de génération pour limiter les hallucinations en imposant au modèle de répondre uniquement à partir des informations présentes dans le contexte récupéré par LightRAG.
- Réorganisation des composants **LLM Generator** et **LLM Judge** afin de séparer leurs rôles et d'améliorer la fiabilité des scores produits par l'agent.
- Conservation des deux versions de l'agent (**Version 1** et **Version 2**) tout en intégrant progressivement les nouvelles améliorations afin de pouvoir comparer leurs performances.
- Exécution d'une nouvelle campagne d'évaluation avec **RAGAS** pour mesurer l'impact des premières modifications.
- Reprise du développement de l'agent et réorganisation de son architecture afin de préparer une nouvelle version de l'Agentic GraphRAG basée directement sur le mécanisme de retrieval de **LightRAG**.
- Vérification de la cohérence entre les différentes versions de l'agent avant la poursuite des expérimentations.
- Intégration du script **`merge_entity_aliases.py`** dans le pipeline afin de préparer le post-traitement du graphe et la fusion des entités dupliquées sans réindexer le corpus.

#### Résultats obtenus

- Les hallucinations du modèle ont été fortement réduites grâce à l'amélioration du prompt de génération.
- Les versions 1 et 2 de l'agent ont produit des réponses plus pertinentes en exploitant les requêtes **Cypher** sur le graphe de connaissances.
- Le nœud **SELF_CORRECT** s'est déclenché automatiquement lorsque la première réponse obtenue ne satisfaisait pas le seuil de qualité défini.
- Les premiers résultats d'évaluation avec **RAGAS** ont été obtenus et ont servi de base pour identifier les améliorations à apporter au système lors des semaines suivantes.

### 07/07/2026

- Début des travaux sur la qualité du benchmark utilisé pour l'évaluation de l'Agentic GraphRAG.
- Analyse de la structure des questions générées et vérification de leur conformité avec l'objectif d'évaluer un système de raisonnement multi-hop.
- Génération de la première version du benchmark contenant **100 questions**, dont **2 questions issues directement de HotpotQA** utilisées comme référence, car dans HotpotQA(vérifié empiriquement et pas supposé)  sur 7405 questions, seulement 2 ont une entité qui existe exactement dans le graphe LightRAG, et ce sont des coïncidences lexicales sans rapport thématique réel (jeux de société type "Power Grid"/"Splendor", rien à voir avec l'IA/ML), donc il est impossible d'en trouver 20 de façon honnête , le corpus arXiv (papiers IA) et HotpotQA (culture générale/Wikipédia) n'ont quasiment aucun recouvrement de sujet.

#### Résultats obtenus

- Obtention d'une première version exploitable du benchmark destinée à l'évaluation de l'agent.

---

### 08/07/2026
- Développement d'un script de diagnostic permettant d'analyser automatiquement la qualité du benchmark.
- Mise en place de critères de validation afin d'identifier les questions nécessitant réellement plusieurs étapes de raisonnement.
- Vérification de chaque question indépendamment de son étiquette (`hop_type`) afin de distinguer les vraies questions multi-hop des questions pseudo multi-hop.

### Critères de validation des questions multi-hop

| Critère de validation | Description | Décision si le critère n'est pas respecté |
|------------------------|-------------|-------------------------------------------|
| **Présence de l'entité de départ (`entity_A`) dans la question** | Vérifie que la question fait explicitement référence à l'entité de départ (`entity_A`) ou à un équivalent. Si cette entité n'est pas nécessaire pour comprendre la question, le premier saut est considéré comme inutile. | **SINGLE_HOP** |
| **Absence de fuite de la réponse (`ground_truth`)** | Vérifie que la réponse attendue n'est pas déjà présente dans l'énoncé de la question. Si la réponse apparaît directement dans la question, aucun raisonnement n'est nécessaire. | **SINGLE_HOP** |
| **Non-redondance des deux sauts (`hop1` et `hop2`)** | Compare les descriptions des deux relations à l'aide d'une similarité de Jaccard afin de vérifier qu'elles représentent deux faits distincts. Si les deux descriptions sont quasiment identiques, il n'existe pas de véritable raisonnement multi-hop. | **SINGLE_HOP** |
| **Origine des informations dans des passages distincts** | Vérifie que les deux relations proviennent de **chunks différents** (et, lorsque l'information est disponible, de documents différents). Si les deux faits sont contenus dans le même passage, une seule récupération suffit pour répondre à la question. | **PSEUDO_MULTI_HOP** |

### Règles de classification

| Résultat de la validation | Interprétation |
|----------------------------|----------------|
| **TRUE_MULTI_HOP** | Tous les critères sont satisfaits. La question nécessite réellement de combiner plusieurs informations provenant de passages (et idéalement de documents) différents. |
| **PSEUDO_MULTI_HOP** | Deux faits distincts existent, mais ils proviennent du même passage. La question ne teste donc pas réellement la capacité de retrieval multi-hop. |
| **SINGLE_HOP** | Au moins un des trois premiers critères n'est pas satisfait. La question peut être résolue sans véritable raisonnement multi-hop. |

### Résultat obtenu

Le script **`validate_multihop_benchmark.py`** a permis de filtrer automatiquement le benchmark généré à partir du graphe de connaissances. Seules les questions classées **TRUE_MULTI_HOP** ont été conservées afin de constituer un benchmark plus représentatif des capacités de raisonnement multi-hop d'un système **Agentic GraphRAG**.

---

### 09/07/2026

- Filtrage automatique du benchmark à partir des résultats du diagnostic.
- Suppression des questions pseudo multi-hop.
- Génération d'une nouvelle version du benchmark contenant uniquement les questions nécessitant réellement plusieurs étapes de raisonnement.

#### Résultats obtenus

- Obtention d'un benchmark final composé de **76 vraies questions multi-hop**, utilisé pour les nouvelles campagnes d'évaluation.

---

### 10/07/2026

- Réalisation du post-traitement du graphe de connaissances sans réindexer le corpus.
- Fusion des entités dupliquées à l'aide du script `merge_entity_aliases.py`.
- Ajout automatique de nouvelles relations de co-occurrence entre les entités lorsque cela était pertinent.
- Réduction des composantes isolées et amélioration de la connectivité globale du graphe.
- Vérification des statistiques du graphe avant et après le post-traitement afin de mesurer les améliorations obtenues.

#### Résultats obtenus

- Fusion de **103 groupes d'entités dupliquées**.
- Augmentation du nombre de relations de **5068 à 5593**.
- Amélioration de la densité du graphe.
- Augmentation du degré moyen des nœuds.
- Suppression complète des nœuds isolés.
- Augmentation de la composante géante de **43,3 % à 55,3 %**.
- Réduction du nombre de composantes connexes de **534 à 385**, améliorant ainsi la navigation dans le graphe.

---

### 11/07/2026

- Vérification de la cohérence du graphe après le post-traitement et préparation d'une nouvelle campagne d'évaluation.
- Vérification du fonctionnement du mécanisme de récupération de contexte après les améliorations du graphe.
- Préparation d'une nouvelle campagne d'évaluation de l'Agentic GraphRAG.

#### Résultats obtenus

- Validation du bon fonctionnement du graphe après le post-traitement.
- Confirmation de l'amélioration de la structure du graphe avant l'évaluation.

---

### 12/07/2026

- Exécution d'une nouvelle campagne d'évaluation avec RAGAS en utilisant le benchmark filtré.
- Analyse des métriques obtenues (Faithfulness, Answer Relevancy, Context Precision et Context Recall).
- Comparaison des nouveaux résultats avec ceux obtenus avant le post-traitement.

#### Résultats obtenus
- Les performances liées au retrieval (Context Precision et Context Recall) demeurent cependant insuffisantes malgré l'amélioration de la structure du graphe (réstent encore null).

---

### 13/07/2026
- Analyse détaillée des résultats obtenus après l'évaluation.
- Identification des limitations restantes du système de récupération de contexte.
- Préparation des prochaines améliorations visant à optimiser le retrieval de LightRAG et à améliorer les performances globales de l'Agentic GraphRAG.

#### Résultats obtenus

- Mise en évidence que le principal axe d'amélioration concerne désormais la qualité du contexte récupéré par LightRAG et les performances du générateur.

### 14/07/2026

- Lancement d'une nouvelle campagne d'évaluation de LightRAG afin de mesurer l'impact des améliorations apportées au graphe après le post-traitement.
- Analyse des nouvelles métriques RAGAS obtenues et comparaison avec les résultats des évaluations précédentes.
- Constat que les métriques **Context Precision** et **Context Recall** demeuraient très faibles malgré l'amélioration de la structure du graphe.
- Début de l'analyse du pipeline de retrieval afin d'identifier l'origine de ce problème.

#### Résultats obtenus

- Les résultats ont montré que le problème ne provenait probablement plus de mécanisme de récupération des contextes , mais de la qualité du graphe.

---

### 15/07/2026

- Développement d'une nouvelle version de l'Agentic GraphRAG basée directement sur le mécanisme de retrieval de LightRAG.
- Suppression des composants devenus inutiles afin d'utiliser directement les contextes retournés par LightRAG.
- Première campagne d'évaluation de cette nouvelle version de l'agent.

#### Résultats obtenus

- Les performances liées au retrieval sont restées faibles, avec des valeurs de **Context Precision** et **Context Recall** toujours insuffisantes, indiquant qu'un dysfonctionnement persistait dans le pipeline de récupération des contextes.

---

### 16/07/2026

- Analyse détaillée du fonctionnement interne du pipeline de retrieval dans `graph_v3.py`.
- Inspection des fonctions responsables de la génération des embeddings et de la recherche vectorielle.
- Vérification des journaux d'exécution afin d'identifier la cause des contextes vides retournés par LightRAG.

#### Résultats obtenus

- Identification d'une exception levée lors de la recherche vectorielle (`AttributeError: 'list' object has no attribute 'size'`).
- Mise en évidence que cette erreur était interceptée silencieusement dans `node_hybrid_search`, empêchant la récupération des contextes sans interrompre l'exécution de l'agent.

---

### 17/07/2026


- Analyse de la fonction `embedding_func()` afin de comprendre l'origine de l'erreur.
- Vérification du type de données retourné par la fonction d'embedding.
- Développement du correctif consistant à retourner les embeddings sous forme d'un tableau **NumPy** (`np.vstack(results)`) conformément aux attentes de LightRAG.

#### Résultats obtenus

- Le diagnostic a confirmé que le problème provenait de la fonction d'embedding et non de la qualité du graphe ou du mécanisme de critique.
- Le correctif a été intégré afin de préparer une nouvelle campagne d'évaluation.

---

### 18/07/2026

- Préparation d'une nouvelle campagne de tests après correction du pipeline de retrieval.
- Vérification de la cohérence de la nouvelle architecture de l'Agentic GraphRAG et des différentes étapes du pipeline avant de relancer les évaluations.

#### Résultats obtenus

- L'agent est prêt pour une nouvelle phase d'évaluation afin de vérifier l'impact du correctif sur les performances de retrieval.

### Période du 19/07/2026 au 22/07/2026

- Réalisation de nouvelles campagnes d'évaluation de **LightRAG** et de la nouvelle version de l'**Agentic GraphRAG** à l'aide du benchmark validé et des métriques **RAGAS**.
- Analyse détaillée des résultats obtenus afin d'identifier les principales causes des faibles performances observées sur les métriques **Context Precision** et **Context Recall**.
- Diagnostic des réponses générées et classification des erreurs en trois catégories : **Retrieval Miss**, **Generation Miss** et **Context Insufficient**, afin d'orienter les prochaines optimisations du système.
- Optimisation des paramètres de récupération de **LightRAG** par la réduction des valeurs de `top_k` et `chunk_top_k` (de **40/20** à **15/10**) afin de diminuer le temps de recherche tout en conservant des contextes pertinents.
- Intégration et test du modèle **NVIDIA GLM-5.2** comme générateur alternatif. Les expérimentations ont montré un temps de réponse compris entre **60 et 124 secondes par requête**, ce qui le rend inadapté à une évaluation de grande échelle. Ce modèle a donc été conservé uniquement comme possibilité d'expérimentation.
- Mise en place d'un mécanisme garantissant l'indépendance entre le **LLM générateur** et le **LLM juge**, en empêchant l'utilisation du même modèle pour les deux rôles afin de limiter les biais d'auto-évaluation.
- Correction du script **`eval_ragas_3.py`** afin que le juge utilisé par **RAGAS** exploite également les modèles **Groq/NVIDIA**, en remplacement de l'utilisation exclusive d'Ollama local afin de réduire le temps d'inférence lors des campagnes d'évaluation.
- Création du notebook **`Sprint4_Ablation_Study.ipynb`** permettant de comparer les performances des trois approches étudiées (**RAG vectoriel**, **LightRAG Hybride** et **Agentic GraphRAG**) sur les mêmes questions du benchmark, avec les mêmes métriques RAGAS, ainsi qu'un tableau récapitulatif des paramètres expérimentaux.
- Correction de la configuration de l'environnement **Jupyter Notebook**, dont le noyau utilisait un interpréteur Python différent de celui de l'environnement virtuel du projet, provoquant des incohérences lors des expérimentations.
- Réalisation de plusieurs campagnes de tests et de diagnostics afin de vérifier le bon fonctionnement des différentes modifications avant le lancement de l'évaluation finale.

#### Résultats obtenus

- Les optimisations réalisées ont permis de réduire le temps d'exécution des expérimentations tout en conservant la qualité des réponses générées.
- Les diagnostics ont confirmé que les principales limitations restantes concernent toujours la qualité du **retrieval**, en particulier les métriques **Context Precision** et **Context Recall**.
- Les différentes architectures d'évaluation (RAG, LightRAG et Agentic GraphRAG) sont désormais prêtes à être comparées dans des conditions expérimentales identiques.
- Les scripts d'évaluation et l'environnement de développement ont été stabilisés en préparation de la campagne finale d'expérimentation.

#### Travaux restants

- Comparer les résultats finaux du **RAG vectoriel**, de **LightRAG** et de **l'Agentic GraphRAG**.
- Analyser les résultats obtenus avec **RAGAS** et préparer la synthèse des performances pour le rapport de PFE.




### 24/07/2026

- Ajout de juges RAGAS alternatifs à Groq (Google Gemini `gemini-2.0-flash`) dans `eval_ragas_3.py` et le notebook `Sprint4_Ablation_Study.ipynb`, activables via `USE_GEMINI_JUDGE`/`USE_OPENAI_JUDGE`, suite aux NaN déjà observés avec Groq en juge à `batch_size=4`.
- Réduction de la concurrence RAGAS (`max_workers=2`, `batch_size` abaissé à 1-2 selon le juge) et allongement du back-off (`max_retries=5`, `max_wait=60`) pour limiter les erreurs 429 / JSON mal formé.
- Ajout d'un diagnostic automatique post-run comptant le pourcentage de NaN par métrique (seuil 25 %) afin de détecter immédiatement un run inexploitable plutôt que d'interpréter des moyennes faussées.
- Réduction temporaire de `N_QUESTIONS` à 10 (au lieu de 20) le temps de stabiliser le scoring avant de remonter l'échelle.
- Première exécution complète de l'ablation study (`v1_ablation_study`) comparant les 3 systèmes (RAG baseline ChromaDB, LightRAG hybride sans agent, Agentic GraphRAG) sur le même benchmark, avec juge Google Gemini.

#### Résultats obtenus

- Le run `v1_ablation_study` a produit **100 % de NaN** sur les 4 métriques RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) pour les 3 systèmes , résultat inexploitable en l'état.
- Diagnostic des logs : chaque appel au juge Gemini échoue avec `429 RESOURCE_EXHAUSTED` et un quota `limit: 0` sur le tier gratuit, épuisant les 5 tentatives de retry sans jamais obtenir de réponse.
- Cause identifiée : 
- Le protocole d'ablation study en lui-même (isolation retrieval/boucle agentique, mêmes questions, même générateur) est validé et prêt ; seul le juge RAGAS doit être corrigé avant de pouvoir exploiter les résultats.

### 25/07/2026

- Remplacement du juge RAGAS défaillant (Gemini) par **DeepSeek** (`deepseek-ai/deepseek-v4-pro`) via l'API NVIDIA, déjà utilisée comme option de repli dans `eval_ragas_3.py`.
- Ajout d'un branchement dédié `USE_NVIDIA_JUDGE`/`NVIDIA_JUDGE_MODEL` (indépendant du générateur principal) dans `eval_ragas_3.py` et dans le notebook d'ablation study, avec priorité la plus haute dans la sélection du juge.
- Désactivation du juge Gemini (`USE_GEMINI_JUDGE=false`) dans `.env` suite au diagnostic de la veille, et configuration de la nouvelle clé NVIDIA fournie pour DeepSeek.
- Test de validation isolé (1 appel) de la clé DeepSeek/NVIDIA hors notebook : réponse JSON propre en 8,2 s — clé et modèle fonctionnels.
- Relance complète du notebook `Sprint4_Ablation_Study.ipynb` avec le juge DeepSeek (`batch_size=1`, `max_workers=2` hérité de la config précédente).
- Réception du plan de récupération du prof : confirme que le pipeline agentique est correct (résultats Agentic complets et cohérents), situe le problème uniquement au niveau du juge RAGAS, et propose un plan A/B/C (Gemini avec `convert_system_message_to_human=True`, repli 5 questions/`max_workers=1`, ou évaluation manuelle en dernier recours) ainsi qu'un correctif spécifique pour Gemini.
- Correction de `max_workers` (fixé à 1, spécifiquement pour le juge NVIDIA/DeepSeek) et ajout de `convert_system_message_to_human=True` sur la branche Gemini (recommandation du plan du prof), dans `eval_ragas_3.py` et le notebook.

#### Résultats obtenus

- Premier run DeepSeek (avant le correctif `max_workers`) : nette amélioration par rapport à Gemini (60-70 % de NaN au lieu de 100 %), mais toujours inexploitable , cause différente cette fois : `429 Too Many Requests` réel de l'API NVIDIA, provoqué par `max_workers=2` qui fait tourner des requêtes en parallèle malgré `batch_size=1`.
- Diagnostic confirmé : ce n'est plus un problème de clé/quota (comme avec Gemini) mais un problème de concurrence pure sur cette clé DeepSeek.
- Correctif appliqué (`max_workers=1` dédié au juge NVIDIA) dans `eval_ragas_3.py` et dans le notebook (via patch direct du fichier `.ipynb`) ; nouvelle exécution à faire pour confirmer 0 % de NaN.

- Relance du notebook avec le juge Gemini reconfiguré selon le plan du prof (`convert_system_message_to_human=True`, `batch_size=2`, `max_workers=2`, seuil NaN à 10 % au lieu de 25 %).
- Ajout du Plan B (repli via `.env` : `EVAL_N_QUESTIONS`, `RAGAS_BATCH_SIZE`, `RAGAS_MAX_WORKERS`, sans toucher au code) et du Plan C (évaluation manuelle) dans `eval_ragas_3.py` et le notebook : 3 nouvelles cellules après la section RAGAS (export CSV à noter à la main, puis agrégation par système).

#### Résultats obtenus (suite)

- Run Gemini (Plan A du prof, avec tous les correctifs appliqués) : **échec identique au run du 24/07**, `429 RESOURCE_EXHAUSTED` avec `limit: 0` sur toutes les métriques de quota (input tokens, requêtes/minute, requêtes/jour), dès la première requête.
- Diagnostic : ce n'est ni un problème de code, ni de concurrence, ni de format de sortie (JSON) , c'est un blocage total au niveau du compte/projet Google associé à cette clé (zéro quota alloué). Le Plan B (réduction à 5 questions, `max_workers=1`) ne peut donc pas non plus fonctionner : un `limit: 0` bloque une requête unique aussi bien que quarante.
- Décision : passage direct au **Plan C** (évaluation manuelle, cellules déjà en place dans le notebook) plutôt que de perdre du temps sur le Plan B, qui échouerait pour la même raison.

- Exécution du Plan C : notation manuelle des 30 réponses (3 systèmes × 10 questions) sur 3 critères (correct / ancré au contexte / clair-complet), export vers `Eval_agentic/plan_c_evaluation_manuelle.csv` et agrégation dans `Eval_agentic/plan_c_resultats_agreges.csv`.
- Correction de la cellule "Tableau comparatif" du notebook pour basculer automatiquement sur les résultats du Plan C si les DataFrames RAGAS (Gemini/DeepSeek) n'existent pas au lieu de planter avec un `NameError`.
- **Analyse des résultats du Plan C et modifications pour améliorer l'Agentic GraphRAG** : le score manuel montre LightRAG seul (0.567) légèrement devant l'Agentic GraphRAG (0.433), contraire à l'hypothèse de départ. Cause identifiée sur plusieurs cas (ex. question MindTrellis/Quantum Knowledge Graph) : l'agent déclare une information "absente du contexte" alors qu'elle y figure explicitement. Deux corrections apportées dans `src/agent/graph_v3.py` :
  1. **Élargissement progressif du retrieval au SELF_CORRECT** (`TOP_K_LIGHTRAG_STEP=10`, `CHUNK_TOP_K_STEP=5`) : auparavant, seule la question était reformulée entre deux itérations le budget de recherche (`top_k`/`chunk_top_k`) restant identique , ne compensait pas un retrieval initial incomplet sur un corpus restreint (500 documents).
  2. **Alignement du contexte vu par le juge de critique sur celui du générateur** : le juge n'évaluait que les 800 premiers caractères du contexte contre 6000 pour le générateur, le rendant aveugle à des informations pourtant fournies au générateur , source probable de faux jugements de conformité. Troncature de la réponse à évaluer également augmentée (600 → 2000 caractères) pour ne pas couper la conclusion de réponses longues.
- Mise à jour du notebook (section 5 et "Limites identifiées") pour documenter ce diagnostic et ces corrections.

- Les corrections apportées à l'agent (élargissement du retrieval au SELF_CORRECT, alignement du contexte du juge) n'ont pas amélioré le score manuel (Agentic GraphRAG toujours à 0.433, une régression détectée sur une question) : **restauration du code de l'agent (`src/agent/graph_v3.py`) et des résultats du tout premier run**, tels qu'obtenus avant ces modifications, en attendant une nouvelle piste d'amélioration.

#### Travaux restants
- Retenter une amélioration de l'Agentic GraphRAG.

### 26-30/07/2026

#### Reprise du diagnostic sur le prompt RESPONSE

- Nouvelle lecture du retour du prof du 25/07 (analyse question par question du fichier `eval_agent_s4.csv`et `plan_c_evaluation_manuelle.csv`) : sur plusieurs questions (MindTrellis/Quantum Knowledge Graph, PyPOTS, QED...), l'agent affirme "The corpus does not contain information..." alors que l'information est bien présente, parfois même citée juste après dans la même réponse , contradiction flagrante causée par la règle 2 du prompt RESPONSE, trop stricte sur les cas limites (trade-off precision/recall en faveur du refus).
- **Option A du prof appliquée** : remplacement de la règle 2 par 3 règles plus nuancées dans `RESPONSE_SYSTEM_PROMPT` (`src/agent/graph_v3.py`) , extraire les faits présents avant de refuser, ne réserver le refus qu'au cas où aucun document ne mentionne le sujet.
- J'ai relancé le notebook `Sprint4_Ablation_Study.ipynb` avec le prompt corrigé : RAGAS toujours indisponible (Gemini toujours en `quota=0`, DeepSeek/NVIDIA en `429`).

#### Bug identifié : troncature du contexte trop agressive

- Diagnostic du contexte réellement envoyé au générateur : en moyenne 30 000 caractères pour LightRAG/Agentic (jusqu'à 39 700), tronqué à 6000 caractères pour éviter l'erreur `413 Request too large` de Groq.
- Cause : dans `_format_context_from_raw`, l'ordre était entités → relations → passages de documents , la troncature à 6000 caractères coupait donc quasi systématiquement avant d'atteindre les passages contenant le texte source, laissant le générateur avec seulement des listes d'entités/relations.
- **Correctif appliqué** : réordonnancement (passages de documents en premier, puis relations, puis entités) + augmentation de la limite à 8000 caractères.

#### Test d'un juge externe via OpenRouter

- Test d'un modèle GPT via OpenRouter (GPT-OSS-20B (free)) comme juge RAGAS alternatif, après les échecs de Gemini et DeepSeek/NVIDIA : RAGAS reste trop lent avec ce juge également.
- Décision : abandon des tentatives de juge RAGAS automatique, retour définitif à l'évaluation manuelle (Plan C) comme méthode d'évaluation retenue pour l'ablation study.

#### Nouvelle notation Plan C (post-correctifs)

- J'ai relancé le notebook avec les 2 correctifs (prompt + contexte) : ré-export de `plan_c_evaluation_manuelle.csv` avec les nouvelles réponses des 3 systèmes.
- Notation selon la grille du prof (correct / ancré au contexte / clair-complet) : score moyen RAG baseline 0.333, LightRAG 0.500, Agentic GraphRAG 0.400.
- Une réponse de LightRAG (question EPM-RL) s'est révélée être une erreur `429 rate limit` de Groq plutôt qu'une vraie génération.
- Constat : le correctif du prompt (Option A) ne montre pas d'amélioration nette sur cet échantillon de 10 questions (Agentic toujours en dessous de LightRAG), à documenter comme résultat honnête plutôt qu'à masquer.

#### Travaux restants
- Isoler et relancer la question EPM-RL (LightRAG) pour un vrai score.
- Décider entre creuser davantage la cause de la sous-performance de l'Agentic ou documenter le résultat tel quel (Option B du prof proposée le 25/07) dans le mémoire.


- Reprendre la rédaction du mémoire final (chapitres État de l'art / Méthodologie), deadline de rédaction fixée au 30 août.

### 30/07/2026 

#### Nouvelle piste sur la sous-performance de l'Agentic

- Diagnostic complémentaire : le nœud CRITIQUE tronquait toujours le contexte à 800 caractères (`node_critique`) alors que le générateur en reçoit désormais 8000 (`MAX_GENERATOR_CONTEXT_CHARS`) , le juge interne jugeait donc sur un contexte quasi vide comparé à ce que le générateur voyait réellement, ce qui pouvait déclencher des `SELF_CORRECT` injustifiés.
- Autre défaut trouvé dans `FINALIZE` : l'agent renvoyait systématiquement la réponse de la **dernière** itération, sans garantie qu'elle soit meilleure qu'une itération précédente (aucun mécanisme ne comparait les scores entre itérations).
- **Corrections appliquées** dans `src/agent/graph_v3.py` :
  1. Contexte du juge aligné sur `MAX_GENERATOR_CONTEXT_CHARS` (800 → 8000).
  2. Suivi de la **meilleure réponse** sur toutes les itérations (`best_response`/`best_score`/`best_retrieved_contexts` dans l'état de l'agent) : `FINALIZE` renvoie désormais la meilleure itération, pas la dernière.
  - Note : l'alignement du contexte du juge avait déjà été tenté le 25/07 puis annulé (pas d'amélioration, une régression) , cette fois combiné au correctif de réordonnancement du contexte (passages en premier) et au suivi de meilleure réponse, jamais testés ensemble auparavant.

#### Nouveau juge RAGAS : OpenRouter (Gemma, gratuit)

- Ajout d'un juge RAGAS supplémentaire via OpenRouter (`google/gemma-3-27b-it:free`), en tête de priorité dans `eval_ragas_3.py` et le notebook, activable via `USE_OPENROUTER_JUDGE`/`OPENROUTER_API_KEY` dans `.env`.
- Précision importante : ce nouveau juge concerne uniquement le **scoring RAGAS** (mesure externe), pas l'agent lui-même. Dans `graph_v3.py`, le générateur (Groq `llama-3.1-8b-instant`) et le juge de critique interne (Groq `llama-3.3-70b-versatile`) restent inchangés.

#### Deux nouveaux problèmes trouvés après exécution du notebook complet

1. **RAGAS échoue à 100% avec OpenRouter/Gemma** : `google/gemma-3-27b-it:free` renvoie une erreur 404 (*"This model is unavailable for free... use this slug instead: google/gemma-3-27b-it"*) , OpenRouter a retiré ce modèle de son offre gratuite, remplacé par une version payante. Résultat : 100% de NaN sur les 4 métriques, sur les 3 systèmes.
   - **Solution retenue** : changement du modèle dans `.env` vers `meta-llama/llama-3.3-70b-instruct:free` (un autre modèle gratuit d'OpenRouter, plus adapté à l'évaluation RAGAS de par sa taille).

2. **Cause principale (probable) de la stagnation de l'Agentic, trouvée en lisant les logs du CRITIQUE** : sur les 10 questions, les 10 scores de critique valent exactement 0.90, y compris pour des réponses qui refusent alors que l'information est présente dans le contexte. Le prompt du juge disait "si la réponse dit correctement 'pas dans le corpus' → score ≥ 0.8" mais **le juge ne vérifiait jamais réellement si l'info était vraiment absente** , il faisait confiance à la réponse elle-même. Résultat : `SELF_CORRECT` ne se déclenche quasiment jamais, même sur des refus injustifiés.
   - **Correction appliquée** dans `node_critique` (`src/agent/graph_v3.py`) : ajout d'une étape de vérification obligatoire dans le prompt du juge , avant de noter un refus, le juge doit lui-même re-vérifier si le sujet/les entités de la question apparaissent dans le contexte, et noter < 0.4 si c'est le cas (au lieu de faire confiance à la réponse).

#### Travaux restants
- Relancer `eval_ragas_3.py` / le notebook avec l'ensemble des correctifs combinés (prompt Option A + contexte réordonné 8000 + juge critique aligné + meilleure réponse + juge critique qui vérifie les refus + juge RAGAS OpenRouter corrigé) pour une vraie mesure.
- Si RAGAS échoue encore : je vais refaire l'évaluation manuelle (Plan C), déjà en place dans le notebook.

#### Abandon de RAGAS via OpenRouter — décision de basculer sur l'évaluation manuelle

- Deuxième échec OpenRouter : après `google/gemma-3-27b-it:free` (retiré de l'offre gratuite), le modèle de repli `meta-llama/llama-3.3-70b-instruct:free` échoue avec la **même erreur 404** ("model unavailable for free, use meta-llama/llama-3.3-70b-instruct instead") , OpenRouter a visiblement retiré ces deux modèles de son offre gratuite même s'il est écrit le mot free à côté. Résultat : de nouveau 100% de NaN sur les 4 métriques, sur les 3 systèmes.
- **Décision** : compte tenu du temps déjà passé à tester Groq, Gemini, DeepSeek/NVIDIA puis deux modèles OpenRouter , tous instables ou indisponibles pour un usage gratuit , j'ai décidé d'abandonner la mesure automatique RAGAS pour cette échéance et de **basculer définitivement sur l'évaluation manuelle (Plan C)**, déjà en place et déjà fonctionnelle dans le notebook (repli automatique déjà codé).
- Prochaine étape : relancer le notebook pour régénérer les réponses des 3 systèmes avec tous les correctifs (prompt Option A, contexte réordonné, juge critique qui vérifie les refus, meilleure réponse), puis noter manuellement les nouvelles réponses dans `plan_c_evaluation_manuelle.csv`.

### Bilan de l'ablation study 

- Avec les correctifs appliqués (prompt Option A, contexte réordonné, juge critique qui vérifie les refus, meilleure réponse conservée), l'écart entre l'Agentic GraphRAG et LightRAG hybride seul a disparu sur l'échantillon de 10 questions (0.500 vs 0.500, contre 0.433 vs 0.567 avant correctifs) , la couche agentique n'est plus pénalisante.
- J'ai voulu passer à 30 questions pour avoir un échantillon plus large et vérifier si la contribution de la couche agentique (CRITIQUE + SELF_CORRECT) devient positivement visible à plus grande échelle plutôt que de rester à l'égalité observée sur seulement 10 questions.
- **Limite rencontrée** : le quota gratuit Groq (100 000 tokens/jour sur le modèle juge) est systématiquement dépassé à 30 questions à cause du nombre d'appels multiplié par les itérations de SELF_CORRECT. Décision : rester sur un échantillon de 10 questions pour cette échéance, et documenter ce plafond d'API gratuite comme une limite matérielle du projet (cf. section Limites du mémoire), plutôt que de perdre du temps supplémentaire à contourner ce plafond.
- Hypothèse à documenter dans le mémoire : la valeur ajoutée de la boucle agentique (capacité à reformuler et relancer une recherche après un premier échec) pourrait se manifester plus nettement sur un échantillon plus larges, là où les erreurs ponctuelles d'un système sans boucle de correction pèsent proportionnellement plus lourd , hypothèse cohérente avec l'égalité observée sur ce petit échantillon, mais non vérifiée faute de temps/quota pour la tester à plus grande échelle.

_Journal mis à jour quotidiennement — Wiame Anejjar_
_Dernière mise à jour : 30 Juillet 2026_
