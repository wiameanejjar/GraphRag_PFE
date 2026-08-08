# Eval_agentic — organisation

Resultats d'evaluation, ranges par etude. Les fichiers d'etat runtime
(checkpoints, rotation de cles) restent a la racine et ne sont pas versionnes.

| Dossier | Contenu | Produit par |
|---|---|---|
| `ablation_study/` | Ablation study RAG baseline / LightRAG hybride / Agentic GraphRAG sur 30 questions : parametres, detail par question, tableau comparatif, et l'evaluation manuelle Plan C (grille correct / ancre au contexte / clair-complet) avec ses moyennes agregees | `Sprint4_Ablation_Study.ipynb` |
| `versions_agent/` | Comparaison des 3 versions de l'agent (v1, v2, v3) sur les memes 30 questions : reponses brutes, notation RAGAS detaillee et agregee | `compare_agent_versions.py`, `evaluate_agent_versions_ragas.py` |
| `graphe/` | Statistiques du graphe de connaissances avant / apres post-traitement | `scripts/graph_stats_figures.py` |
| `historique_sprints/` | Evaluations RAGAS des sprints precedents (s3, s4, lightrag, run a 6 questions), conservees pour la comparaison Sprint 3 -> Sprint 4 | `eval_ragas.py`, `ragas_age.py`, `eval_ragas_3.py` |

## Fichiers d'etat (racine, non versionnes)

| Fichier | Role |
|---|---|
| `groq_key_state.json` | Cle Groq active + horodatages d'epuisement (rotation sur 3 comptes) |
| `plan_c_checkpoint_30q.json` | Reprise de l'ablation study a 30 questions |
| `agent_versions_checkpoint.json` | Reponses generees des versions v1 / v2 / v3 |
| `agent_versions_ragas_checkpoint.json` | Reprise de la notation RAGAS des versions |
