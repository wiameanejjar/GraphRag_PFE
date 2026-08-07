"""
Agentic GraphRAG - version corrigée.

Architecture :
1. QUERY
2. GRAPH_SEARCH
3. VECTOR_SEARCH
4. FUSE
5. RESPONSE
6. CRITIQUE
7. SELF_CORRECT
8. FINALIZE

Objectifs de cette version :
- séparation claire en 8 nœuds
- docstrings systématiques
- gestion d'erreurs Neo4j
- déduplication des résultats
- contexte plus ciblé et moins bruité
- réponse stricte, fidèle au contexte
- boucle de correction avec self-correction
"""

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

from src.utils.groq_rotation import groq_chat_completion, AllGroqKeysExhaustedError

load_dotenv()

# =========================
# CONFIGURATION
# =========================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

CHROMA_DIR = os.getenv("CHROMA_DIR", "indexes/chroma_pfe500_baseline")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pfe_500_baseline")

USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
USE_GROQ_JUDGE = os.getenv("USE_GROQ_JUDGE", "true").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "5"))
TOP_K_GRAPH = int(os.getenv("TOP_K_GRAPH", "8"))
CRITIQUE_SEUIL = float(os.getenv("CRITIQUE_SEUIL", "0.75"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

# =========================
# CLIENTS
# =========================
def _get_llm():
    """Initialise le LLM principal."""
    if USE_GROQ and GROQ_API_KEY:
        print("[LLM] Groq llama-3.3-70b-versatile (rotation GROQ_API_KEY/_2/_3)")

        class _GroqLLM:
            def invoke(self, messages):
                msgs = []
                for m in messages:
                    role = "system" if isinstance(m, SystemMessage) else "user"
                    msgs.append({"role": role, "content": m.content})
                r = groq_chat_completion(
                    model="llama-3.3-70b-versatile",
                    messages=msgs,
                    temperature=0,
                    max_tokens=800,
                )
                class _R:
                    content = r.choices[0].message.content
                return _R()

        return _GroqLLM()

    print(f"[LLM] Ollama {MODEL_NAME}")
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_URL,
        temperature=0,
        num_ctx=OLLAMA_NUM_CTX,
    )


def _get_judge_llm():
    """Initialise le LLM de critique."""
    if USE_GROQ_JUDGE and GROQ_API_KEY:
        class _GroqJudge:
            def invoke(self, messages):
                msgs = []
                for m in messages:
                    role = "system" if isinstance(m, SystemMessage) else "user"
                    msgs.append({"role": role, "content": m.content})
                r = groq_chat_completion(
                    model="llama-3.3-70b-versatile",
                    messages=msgs,
                    temperature=0,
                    max_tokens=300,
                )
                class _R:
                    content = r.choices[0].message.content
                return _R()

        return _GroqJudge()

    return llm


llm = _get_llm()
judge_llm = _get_judge_llm()

print("\n========== GENERATION LLM ==========")
print("Type :", type(llm))
if hasattr(llm, "model"):
    print("Model :", llm.model)
if hasattr(llm, "base_url"):
    print("Base URL :", llm.base_url)

print("\n========== JUDGE LLM ==========")
print("Type :", type(judge_llm))
if hasattr(judge_llm, "model"):
    print("Model :", judge_llm.model)
if hasattr(judge_llm, "base_url"):
    print("Base URL :", judge_llm.base_url)

# Neo4j
neo4j_driver = None
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with neo4j_driver.session() as session:
        session.run("RETURN 1")
    print("[NEO4J] Connecté")
except Exception as e:
    print(f"[NEO4J] Non disponible : {e}")
    neo4j_driver = None

# Chroma
vectorstore = None
chroma_retriever = None
try:
    emb_fn = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=emb_fn,
        collection_name=COLLECTION_NAME,
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_VECTOR})
    print("[CHROMA] Chargé")
except Exception as e:
    print(f"[CHROMA] Non disponible : {e}")


# =========================
# STATE
# =========================
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


# =========================
# HELPERS
# =========================
_STOPWORDS = {
    "what", "is", "the", "a", "an", "how", "does", "which", "who", "why",
    "when", "in", "for", "of", "to", "and", "or", "are", "was", "were",
    "used", "tell", "give", "does", "do", "this", "that", "from", "with",
    "about", "can", "should", "their", "them", "it", "its"
}


def _normalize_text(text: str) -> str:
    """Nettoie un texte brute avant utilisation."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def _keywords(text: str) -> List[str]:
    """Extrait des mots-clés utiles d'une question."""
    words = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS][:8]


def _score_relevance(query: str, text: str) -> float:
    """Score de similarité simple basé sur les mots-clés communs."""
    qk = set(_keywords(query))
    tk = set(_keywords(text))
    if not qk:
        return 0.0
    overlap = len(qk & tk)
    return round(overlap / max(1, len(qk)), 3)


def _deduplicate(items: List[Dict[str, Any]], key_name: str = "name") -> List[Dict[str, Any]]:
    """Déduplique une liste de dicts sur la base d'une clé."""
    seen = set()
    unique = []
    for item in items:
        key = item.get(key_name, "") or item.get("description", "")
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _llm_call(system: str, user: str) -> str:
    """Appel LLM sécurisé."""
    try:
        r = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return (r.content or "").strip()
    except AllGroqKeysExhaustedError:
        raise
    except Exception as e:
        return f"[LLM ERROR] {e}"


# =========================
# NŒUD 1 : QUERY
# =========================
def node_query(state: AgentState) -> AgentState:
    """Initialise la question active et la reformulation."""
    q = state.get("reformulated_q") or state["question"]
    iteration = state.get("iteration", 0)
    trace = state.get("trace", [])
    trace.append(f"[QUERY iter={iteration}] {q}")
    print(f"\n[QUERY] iter={iteration} | {q}")
    return {**state, "reformulated_q": q, "iteration": iteration, "trace": trace}


# =========================
# NŒUD 2 : GRAPH_SEARCH
# =========================
def node_graph_search(state: AgentState) -> AgentState:
    """Recherche des entités et relations pertinentes dans Neo4j."""
    q = state["reformulated_q"]
    trace = state.get("trace", [])
    kws = _keywords(q)

    results: List[Dict[str, Any]] = []

    if neo4j_driver is None:
        trace.append("[GRAPH_SEARCH] Neo4j non disponible")
        print("[GRAPH_SEARCH] Neo4j non disponible")
        return {**state, "graph_results": [], "trace": trace}

    try:
        with neo4j_driver.session() as session:
            seed_names: List[str] = []

            for kw in kws:
                rows = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS $kw
                       OR toLower(e.description) CONTAINS $kw
                    RETURN e.name AS name, e.description AS description, e.type AS type
                    LIMIT $lim
                    """,
                    kw=kw,
                    lim=TOP_K_GRAPH,
                ).data()

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
                hop1 = session.run(
                    """
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE a.name IN $names
                    RETURN a.name AS src, b.name AS tgt, b.description AS description,
                           b.type AS type, r.description AS rel_desc
                    LIMIT 12
                    """,
                    names=list(seed_names),
                ).data()

                for row in hop1:
                    src = row.get("src", "")
                    tgt = row.get("tgt", "")
                    desc = _normalize_text(row.get("description", ""))
                    rel_desc = _normalize_text(row.get("rel_desc", ""))
                    if tgt:
                        results.append({
                            "name": f"{src} → {tgt}",
                            "description": f"Relation: {rel_desc} | {tgt}: {desc}",
                            "type": "relation",
                            "hop": 1,
                        })

                    if src and tgt and " " in tgt:
                        pass

            results = _deduplicate(results, key_name="name")
            results = sorted(results, key=lambda r: (r.get("hop", 0), len(r.get("description", ""))), reverse=False)

    except Exception as e:
        trace.append(f"[GRAPH_SEARCH] Erreur Neo4j : {e}")
        print(f"[GRAPH_SEARCH] Erreur : {e}")
        results = []

    trace.append(f"[GRAPH_SEARCH] {len(results)} résultats (kws={kws})")
    print(f"[GRAPH_SEARCH] {len(results)} résultats | kws={kws}")
    return {**state, "graph_results": results[:TOP_K_GRAPH], "trace": trace}


# =========================
# NŒUD 3 : VECTOR_SEARCH
# =========================
def node_vector_search(state: AgentState) -> AgentState:
    """Recherche sémantique dans ChromaDB et filtre les chunks les plus pertinents."""
    q = state["reformulated_q"]
    trace = state.get("trace", [])

    texts: List[str] = []

    if chroma_retriever is None:
        trace.append("[VECTOR_SEARCH] Chroma non disponible")
        print("[VECTOR_SEARCH] Chroma non disponible")
        return {**state, "vector_results": [], "trace": trace}

    try:
        docs = chroma_retriever.invoke(q)
        scored = []
        for d in docs:
            txt = _normalize_text(d.page_content)
            if not txt:
                continue
            score = _score_relevance(q, txt)
            scored.append((score, txt))

        scored = sorted(scored, key=lambda x: x[0], reverse=True)
        texts = [txt for _, txt in scored[:TOP_K_VECTOR]]
    except Exception as e:
        trace.append(f"[VECTOR_SEARCH] Erreur : {e}")
        print(f"[VECTOR_SEARCH] Erreur : {e}")

    trace.append(f"[VECTOR_SEARCH] {len(texts)} chunks")
    print(f"[VECTOR_SEARCH] {len(texts)} chunks")
    return {**state, "vector_results": texts, "trace": trace}


# =========================
# NŒUD 4 : FUSE
# =========================
def node_fuse(state: AgentState) -> AgentState:
    """Fusionne les résultats graphe + vecteurs en un contexte court et pertinent."""
    graph = state.get("graph_results", [])
    vectors = state.get("vector_results", [])
    trace = state.get("trace", [])

    parts: List[str] = []

    graph_items = []
    for item in graph:
        if isinstance(item, dict):
            description = _normalize_text(item.get("description", ""))
            if description:
                graph_items.append(description)

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
    print(f"[FUSE] {len(fused)} chars | graph={len(graph_items)} | vectors={len(vectors)}")
    return {**state, "fused_context": fused, "trace": trace}


# =========================
# NŒUD 5 : RESPONSE
# =========================
def node_response(state: AgentState) -> AgentState:
    """Génère une réponse stricte et fondée sur le contexte."""
    q = state["reformulated_q"]
    context = state.get("fused_context", "")
    iteration = state.get("iteration", 0)
    trace = state.get("trace", [])

    system = """You are a strict research assistant.
You MUST use ONLY the provided context.
Rules:
1. Use only information explicitly present in the context.
2. If the context does not contain the answer, reply exactly:
   The corpus does not contain information to answer this question.
3. Always cite the source as [Graph] or [Doc N].
4. Do not use external knowledge.
5. If only partial information is available, say what is known and clearly mark the missing part as [not in corpus].
"""
    user = f"""Context:
{context}

Question: {q}

Provide a concise answer grounded in the context above.
"""

    answer = _llm_call(system, user)
    trace.append(f"[RESPONSE iter={iteration}] {answer[:140]}...")
    print(f"[RESPONSE iter={iteration}] {answer[:150]}")
    return {**state, "response": answer, "trace": trace}


# =========================
# NŒUD 6 : CRITIQUE
# =========================
def node_critique(state: AgentState) -> AgentState:
    """Évalue la réponse selon la fidélité, la complétude et la clarté."""
    q = state["question"]
    response = state.get("response", "")
    context = state.get("fused_context", "")[:1200]
    iteration = state.get("iteration", 0)
    trace = state.get("trace", [])

    system = """You are a strict RAG evaluator.
Return ONLY valid JSON:
{"score": 0.0, "reason": "short explanation"}

Scoring rules:
- 1.0 = fully correct, fully grounded, complete
- 0.75+ = acceptable
- below 0.75 = needs improvement
- very low if the answer is speculative, unsupported, or too vague
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
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        score = max(0.0, min(1.0, float(data.get("score", 0.5))))
        reason = str(data.get("reason", raw[:150]))
    except Exception:
        score, reason = 0.5, raw[:150]

    trace.append(f"[CRITIQUE iter={iteration}] score={score:.2f} | {reason[:80]}")
    print(f"[CRITIQUE iter={iteration}] score={score:.2f} | {reason[:80]}")
    return {**state, "critique_score": score, "critique_reason": reason, "trace": trace}


# =========================
# NŒUD 7 : SELF_CORRECT
# =========================
def node_self_correct(state: AgentState) -> AgentState:
    """Réformule la question si la réponse a échoué ou manque de précision."""
    q = state["question"]
    response = state.get("response", "")[:300]
    reason = state.get("critique_reason", "")
    iteration = state.get("iteration", 0)
    trace = state.get("trace", [])

    system = """You are a search-query optimizer for a RAG system.
Given a question that did not get a good answer, reformulate it to be:
1. More specific
2. More answerable from the available corpus
3. Better aligned with the relevant entities or concepts
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
    print(f"[SELF_CORRECT] iter {iteration}→{new_iter}")
    print(f"  Original  : {q}")
    print(f"  Reformulé : {new_q}")

    return {**state, "reformulated_q": new_q, "iteration": new_iter, "trace": trace}


# =========================
# NŒUD 8 : FINALIZE
# =========================
def node_finalize(state: AgentState) -> AgentState:
    """Valide et retourne la réponse finale."""
    score = state.get("critique_score", 0.0)
    iter_ = state.get("iteration", 0)
    trace = state.get("trace", [])
    trace.append(f"[FINALIZE] score={score:.2f}, iterations={iter_}")
    print(f"[FINALIZE] score={score:.2f} après {iter_} iteration(s)")
    return {
        **state,
        "final_response": state.get("response", ""),
        "trace": trace,
    }


# =========================
# ROUTEUR
# =========================
def route_after_critique(state: AgentState) -> str:
    """Choisit entre finaliser ou relancer une self-correction."""
    score = state.get("critique_score", 0.0)
    iter_ = state.get("iteration", 0)

    if score >= CRITIQUE_SEUIL:
        print(f"[ROUTE] score={score:.2f} >= {CRITIQUE_SEUIL} → FINALIZE")
        return "finalize"
    if iter_ < MAX_ITERATIONS:
        print(f"[ROUTE] score={score:.2f} < {CRITIQUE_SEUIL}, iter={iter_}/{MAX_ITERATIONS} → SELF_CORRECT")
        return "self_correct"
    print(f"[ROUTE] score={score:.2f} mais MAX_ITERATIONS atteint → FINALIZE")
    return "finalize"


# =========================
# CONSTRUCTION DU GRAPHE
# =========================
def build_agent():
    """Construit le graphe LangGraph avec les 8 nœuds."""
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

    g.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "finalize": "finalize",
            "self_correct": "self_correct",
        },
    )

    g.add_edge("self_correct", "query")
    g.add_edge("finalize", END)

    return g.compile()


# =========================
# API PUBLIQUE
# =========================
def run_agent(question: str, use_phoenix: bool = False) -> dict:
    """Exécute l’agent sur une question donnée."""
    if use_phoenix:
        try:
            import phoenix as px
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter("http://localhost:6006/v1/traces")))
            otel_trace.set_tracer_provider(provider)
            LangChainInstrumentor().instrument()
            px.launch_app()
            print("[PHOENIX] Dashboard activé")
        except Exception as e:
            print(f"[PHOENIX] Non disponible : {e}")

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

    print(f"\n{'=' * 60}")
    print(f"QUESTION : {question}")
    print(f"{'=' * 60}")

    result = agent.invoke(init)

    print(f"\n--- RÉPONSE FINALE ---")
    print(result["final_response"])
    print(f"\n--- SCORE / ITERATIONS ---")
    print(f"Score : {result['critique_score']:.2f} | Iterations : {result['iteration']}")
    print(f"Raison : {result['critique_reason'][:120]}")

    return result