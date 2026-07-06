import json
import os
import re
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph
from neo4j import GraphDatabase

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
CHROMA_DIR = os.getenv("CHROMA_DIR", "indexes/chroma_pfe500_baseline")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pfe_500_baseline")

TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "5"))
TOP_K_GRAPH = int(os.getenv("TOP_K_GRAPH", "8"))
CRITIQUE_SEUIL = float(os.getenv("CRITIQUE_SEUIL", "0.75"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

class AgentState(TypedDict):
    question: str
    reformulated_q: str
    graph_results: list
    vector_results: list
    fused_context: str
    response: str
    critique_score: float
    critique_reason: str
    iteration: int
    final_response: str
    trace: list

_STOPWORDS = {
    "what", "is", "the", "a", "an", "how", "does", "which", "who", "why",
    "when", "in", "for", "of", "to", "and", "or", "are", "was", "were",
    "used", "tell", "give", "this", "that", "from", "with", "about", "can",
    "should", "their", "them", "it", "its"
}

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()

def _keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS][:8]

def _score_relevance(query: str, text: str) -> float:
    qk = set(_keywords(query))
    tk = set(_keywords(text))
    if not qk:
        return 0.0
    return round(len(qk & tk) / max(1, len(qk)), 3)

def _deduplicate(items: List[Dict[str, Any]], key_name: str = "name") -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = item.get(key_name, "") or item.get("description", "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL, temperature=0)
judge_llm = llm

neo4j_driver = None
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with neo4j_driver.session() as session:
        session.run("RETURN 1")
    print("[NEO4J] OK")
except Exception as e:
    print(f"[NEO4J] unavailable: {e}")
    neo4j_driver = None

chroma_retriever = None
try:
    emb_fn = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=emb_fn,
        collection_name=COLLECTION_NAME,
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_VECTOR})
    print("[CHROMA] OK")
except Exception as e:
    print(f"[CHROMA] unavailable: {e}")

def _llm_call(system: str, user: str) -> str:
    try:
        r = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return (r.content or "").strip()
    except Exception as e:
        return f"[LLM ERROR] {e}"

def node_query(state: AgentState) -> AgentState:
    """Initialise la question et la reformulation."""
    q = state.get("reformulated_q") or state["question"]
    iteration = state.get("iteration", 0)
    trace = state.get("trace", [])
    trace.append(f"[QUERY iter={iteration}] {q}")
    return {**state, "reformulated_q": q, "iteration": iteration, "trace": trace}

def node_graph_search(state: AgentState) -> AgentState:
    """Recherche graphe dans Neo4j avec gestion d’erreur et déduplication."""
    q = state["reformulated_q"]
    trace = state.get("trace", [])
    kws = _keywords(q)
    results = []

    if neo4j_driver is None:
        trace.append("[GRAPH_SEARCH] Neo4j non disponible")
        return {**state, "graph_results": [], "trace": trace}

    try:
        with neo4j_driver.session() as session:
            seed_names = []
            for kw in kws:
                rows = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS $kw
                       OR toLower(e.description) CONTAINS $kw
                    RETURN e.name AS name, e.description AS description, e.type AS type
                    LIMIT $lim
                """, kw=kw, lim=TOP_K_GRAPH).data()

                for row in rows:
                    name = row.get("name", "")
                    if name and name not in seed_names:
                        seed_names.append(name)
                        results.append({
                            "name": name,
                            "description": _normalize_text(row.get("description", "")),
                            "type": row.get("type", "entity"),
                            "hop": 0,
                        })

            if seed_names:
                hop1 = session.run("""
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE a.name IN $names
                    RETURN a.name AS src, b.name AS tgt, b.description AS description,
                           b.type AS type, r.description AS rel_desc
                    LIMIT 12
                """, names=list(seed_names)).data()

                for row in hop1:
                    src = row.get("src", "")
                    tgt = row.get("tgt", "")
                    rel_desc = _normalize_text(row.get("rel_desc", ""))
                    desc = _normalize_text(row.get("description", ""))
                    if tgt:
                        results.append({
                            "name": f"{src} → {tgt}",
                            "description": f"Relation: {rel_desc} | {tgt}: {desc}",
                            "type": "relation",
                            "hop": 1,
                        })

            results = _deduplicate(results, key_name="name")
            results = sorted(results, key=lambda r: (r.get("hop", 0), len(r.get("description", ""))), reverse=False)

    except Exception as e:
        trace.append(f"[GRAPH_SEARCH] Erreur Neo4j : {e}")
        results = []

    trace.append(f"[GRAPH_SEARCH] {len(results)} résultats")
    return {**state, "graph_results": results[:TOP_K_GRAPH], "trace": trace}

def node_vector_search(state: AgentState) -> AgentState:
    """Recherche vectorielle plus ciblée et filtrée."""
    q = state["reformulated_q"]
    trace = state.get("trace", [])
    texts = []

    if chroma_retriever is None:
        trace.append("[VECTOR_SEARCH] Chroma non disponible")
        return {**state, "vector_results": [], "trace": trace}

    try:
        docs = chroma_retriever.invoke(q)
        scored = []
        for d in docs:
            txt = _normalize_text(d.page_content)
            if not txt:
                continue
            scored.append((_score_relevance(q, txt), txt))

        scored = sorted(scored, key=lambda x: x[0], reverse=True)
        texts = [txt for _, txt in scored[:TOP_K_VECTOR]]
    except Exception as e:
        trace.append(f"[VECTOR_SEARCH] Erreur : {e}")

    trace.append(f"[VECTOR_SEARCH] {len(texts)} chunks")
    return {**state, "vector_results": texts, "trace": trace}

def node_fuse(state: AgentState) -> AgentState:
    """Fusionne le contexte de façon plus sélective et compacte."""
    graph = state.get("graph_results", [])
    vectors = state.get("vector_results", [])
    trace = state.get("trace", [])

    parts = []

    graph_items = []
    for item in graph:
        if isinstance(item, dict):
            desc = _normalize_text(item.get("description", ""))
            if desc:
                graph_items.append(desc)

    if graph_items:
        parts.append("=== Evidence graphe ===")
        for i, item in enumerate(graph_items[:4], 1):
            parts.append(f"[Graph {i}] {item}")

    if vectors:
        parts.append("\n=== Passages pertinents ===")
        for i, chunk in enumerate(vectors[:4], 1):
            parts.append(f"[Doc {i}] {chunk[:800]}")

    fused = "\n".join(parts) if parts else "Aucun contexte disponible."
    trace.append(f"[FUSE] {len(fused)} chars | graph={len(graph_items)} | vectors={len(vectors)}")
    return {**state, "fused_context": fused, "trace": trace}

def node_response(state: AgentState) -> AgentState:
    """Génère une réponse stricte, fondée uniquement sur le contexte."""
    q = state["reformulated_q"]
    context = state.get("fused_context", "")
    iteration = state.get("iteration", 0)
    trace = state.get("trace", [])

    system = """You are a strict research assistant.
Use ONLY the provided context.
If the answer is not in the context, reply exactly:
The corpus does not contain information to answer this question.
Always cite the source as [Graph] or [Doc N].
Do not use external knowledge.
If partial information exists, say what is known and mark the missing part as [not in corpus].
"""
    user = f"""Context:
{context}

Question: {q}

Provide a concise answer grounded in the context above.
"""

    answer = _llm_call(system, user)
    trace.append(f"[RESPONSE iter={iteration}] {answer[:140]}...")
    return {**state, "response": answer, "trace": trace}

def node_critique(state: AgentState) -> AgentState:
    """Évalue la réponse avec un critère stricte de grounding."""
    q = state["question"]
    response = state.get("response", "")
    context = state.get("fused_context", "")[:1200]
    iteration = state.get("iteration", 0)
    trace = state.get("trace", [])

    system = """You are a strict RAG evaluator.
Return ONLY valid JSON:
{"score": 0.0, "reason": "short explanation"}
Score 1.0 = fully correct and fully grounded.
Below 0.75 = needs improvement.
"""
    user = f"""Question: {q}

Context:
{context}

Answer:
{response[:800]}

Evaluate the answer.
"""

    raw = judge_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        score = max(0.0, min(1.0, float(data.get("score", 0.5))))
        reason = str(data.get("reason", raw[:150]))
    except Exception:
        score, reason = 0.5, raw[:150]

    trace.append(f"[CRITIQUE iter={iteration}] score={score:.2f} | {reason[:80]}")
    return {**state, "critique_score": score, "critique_reason": reason, "trace": trace}

def node_self_correct(state: AgentState) -> AgentState:
    """Réformule la question si la réponse est trop faible."""
    q = state["question"]
    response = state.get("response", "")[:300]
    reason = state.get("critique_reason", "")
    iteration = state.get("iteration", 0)
    trace = state.get("trace", [])

    system = """You are a search-query optimizer.
Reformulate the question to be more specific and more answerable from the corpus.
Return ONLY the reformulated question.
"""
    user = f"""Original question: {q}

Previous answer: {response}

Why it failed: {reason}

Better reformulated question:
"""
    new_q = _llm_call(system, user).strip() or q
    new_iter = iteration + 1
    trace.append(f"[SELF_CORRECT iter={new_iter}] {new_q}")
    return {**state, "reformulated_q": new_q, "iteration": new_iter, "trace": trace}

def node_finalize(state: AgentState) -> AgentState:
    """Retourne la réponse finale."""
    score = state.get("critique_score", 0.0)
    iter_ = state.get("iteration", 0)
    trace = state.get("trace", [])
    trace.append(f"[FINALIZE] score={score:.2f}, iterations={iter_}")
    return {**state, "final_response": state.get("response", ""), "trace": trace}

def route_after_critique(state: AgentState) -> str:
    """Décide de finaliser ou de relancer la self-correction."""
    score = state.get("critique_score", 0.0)
    iter_ = state.get("iteration", 0)
    if score >= CRITIQUE_SEUIL:
        return "finalize"
    if iter_ < MAX_ITERATIONS:
        return "self_correct"
    return "finalize"

def build_agent():
    """Construit le graphe LangGraph avec 8 nœuds séparés."""
    g = StateGraph(AgentState)
    g.add_node("query", node_query)
    g.add_node("graph_search", node_graph_search)
    g.add_node("vector_search", node_vector_search)
    g.add_node("fuse", node_fuse)
    g.add_node("response", node_response)
    g.add_node("critique", node_critique)
    g.add_node("self_correct", node_self_correct)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("query")
    g.add_edge("query", "graph_search")
    g.add_edge("graph_search", "vector_search")
    g.add_edge("vector_search", "fuse")
    g.add_edge("fuse", "response")
    g.add_edge("response", "critique")

    g.add_conditional_edges("critique", route_after_critique, {"finalize": "finalize", "self_correct": "self_correct"})
    g.add_edge("self_correct", "query")
    g.add_edge("finalize", END)
    return g.compile()

def run_agent(question: str) -> dict:
    """Exécute l’agent sur une question."""
    agent = build_agent()
    init: AgentState = {
        "question": question,
        "reformulated_q": question,
        "graph_results": [],
        "vector_results": [],
        "fused_context": "",
        "response": "",
        "critique_score": 0.0,
        "critique_reason": "",
        "iteration": 0,
        "final_response": "",
        "trace": [],
    }
    return agent.invoke(init)