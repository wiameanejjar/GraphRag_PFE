"""
Exporte le graphe LightRAG (.graphml) vers Neo4j, avec le même schéma que
celui déjà utilisé dans ce projet (cf. src/agent/graph_v1.py / graph_v2.py) :

    (:Entity {name, description, type})-[:RELATES_TO {description, weight, keywords}]->(:Entity)

Ce qui permet de réutiliser telles quelles les requêtes Cypher déjà présentes
dans le JOURNAL.md, ex:
    MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 50
    MATCH (n) RETURN n.name, count{(n)--()} AS degree ORDER BY degree DESC LIMIT 10

Usage:
    python scripts/export_graph_to_neo4j.py
    python scripts/export_graph_to_neo4j.py --clear                # vide Entity/RELATES_TO avant d'importer
    python scripts/export_graph_to_neo4j.py --graphml indexes/lightrag_500_connected_v2/graph_chunk_entity_relation.graphml
"""
import argparse
import os
import sys
from pathlib import Path

# Sous Windows, stdout redirige vers un fichier retombe sur cp1252, qui ne sait
# pas encoder les caracteres accentues/symboles des messages ci-dessous.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import networkx as nx
from neo4j import GraphDatabase

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

DEFAULT_GRAPHML = "indexes/lightrag_500_connected_v2/graph_chunk_entity_relation.graphml"
BATCH_SIZE = 500


def batched(iterable, n):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def export(graphml_path: Path, clear: bool):
    print(f"Chargement de {graphml_path} ...")
    G = nx.read_graphml(graphml_path)
    print(f"  {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes à exporter")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("RETURN 1")  # vérifie la connexion avant de commencer
        print(f"[NEO4J] Connecté à {NEO4J_URI}")

        if clear:
            print("[NEO4J] --clear : suppression des Entity/RELATES_TO existants...")
            session.run("MATCH (n:Entity) DETACH DELETE n")

        session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                     "FOR (e:Entity) REQUIRE e.name IS UNIQUE")

        # ── Nœuds ────────────────────────────────────────────────
        node_rows = [
            {
                "name": name,
                "description": (attrs.get("description") or "")[:8000],
                "type": attrs.get("entity_type", "concept"),
                "file_path": attrs.get("file_path", ""),
            }
            for name, attrs in G.nodes(data=True)
        ]
        total = 0
        for batch in batched(node_rows, BATCH_SIZE):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {name: row.name})
                SET e.description = row.description,
                    e.type = row.type,
                    e.file_path = row.file_path
                """,
                rows=batch,
            )
            total += len(batch)
            print(f"  Nœuds importés : {total}/{len(node_rows)}", end="\r")
        print(f"  Nœuds importés : {total}/{len(node_rows)}")

        # ── Relations ────────────────────────────────────────────
        edge_rows = [
            {
                "src": src,
                "tgt": tgt,
                "description": (attrs.get("description") or "")[:4000],
                "keywords": attrs.get("keywords", ""),
                "weight": float(attrs.get("weight", 1.0)),
            }
            for src, tgt, attrs in G.edges(data=True)
        ]
        total = 0
        for batch in batched(edge_rows, BATCH_SIZE):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Entity {name: row.src})
                MATCH (b:Entity {name: row.tgt})
                MERGE (a)-[r:RELATES_TO]->(b)
                SET r.description = row.description,
                    r.keywords = row.keywords,
                    r.weight = row.weight
                """,
                rows=batch,
            )
            total += len(batch)
            print(f"  Relations importées : {total}/{len(edge_rows)}", end="\r")
        print(f"  Relations importées : {total}/{len(edge_rows)}")

        counts = session.run(
            "MATCH (n:Entity) OPTIONAL MATCH (n)-[r:RELATES_TO]->() "
            "RETURN count(DISTINCT n) AS n_nodes, count(r) AS n_edges"
        ).single()
        print(f"\n[NEO4J] État final : {counts['n_nodes']} nœuds Entity, "
              f"{counts['n_edges']} relations RELATES_TO")

    driver.close()
    print(f"\n✓ Export terminé -> {NEO4J_URI}")
    print("  Requêtes utiles dans Neo4j Browser :")
    print("    MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 50")
    print("    MATCH (n) RETURN n.name, count{(n)--()} AS degree ORDER BY degree DESC LIMIT 10")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphml", default=DEFAULT_GRAPHML)
    ap.add_argument("--clear", action="store_true",
                     help="Supprime les Entity/RELATES_TO existants avant l'import "
                          "(sinon MERGE : ré-exécutable sans dupliquer)")
    args = ap.parse_args()
    export(Path(args.graphml), args.clear)
