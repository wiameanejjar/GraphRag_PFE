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

**01/05/2026**

**02/05/2026**

**03/05/2026**

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

_Journal mis à jour quotidiennement — Wiame Anejjar_
_Dernière mise à jour : 30 Avril 2026_
