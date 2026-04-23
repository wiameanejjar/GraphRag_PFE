import os
from neo4j import GraphDatabase
import networkx as nx

# ---------------- CONFIG ----------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

GRAPHML_PATH = "./lightrag_storage_new/graph_chunk_entity_relation.graphml"

# ---------------- DRIVER ----------------
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ---------------- IMPORT FUNCTION ----------------
def clear_and_import(tx, graph):
    print(" Suppression ancien graphe...")
    tx.run("MATCH (n) DETACH DELETE n")

    print(" Insertion des noeuds...")
    for node, data in graph.nodes(data=True):
        tx.run(
            """
            MERGE (e:Entity {id: $id})
            SET e.name = $name,
                e.type = $type,
                e.description = $desc
            """,
            id=node,
            name=data.get("name") or node,   
            type=data.get("type", ""),
            desc=data.get("description", "")
        )

    print(" Insertion des relations...")
    for source, target, data in graph.edges(data=True):
        tx.run(
            """
            MATCH (a:Entity {id: $src})
            MATCH (b:Entity {id: $tgt})
            MERGE (a)-[r:RELATES_TO]->(b)
            SET r.description = $desc
            """,
            src=source,
            tgt=target,
            desc=data.get("description", "")
        )

# ---------------- MAIN ----------------
def main():
    if not os.path.exists(GRAPHML_PATH):
        print(f" Fichier introuvable : {GRAPHML_PATH}")
        return

    print(" Chargement du fichier GraphML...")
    graph = nx.read_graphml(GRAPHML_PATH)

    print(f" Noeuds : {len(graph.nodes())}")
    print(f" Relations : {len(graph.edges())}")

    with driver.session() as session:
        session.execute_write(clear_and_import, graph)

    print("\n Export terminé avec succès !")
    print(" Ouvre http://localhost:7474")
    print(" Lance : MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 50")

    driver.close()

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()