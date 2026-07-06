"""
src/agent/graph_v1_corrigé.py
Agentic GraphRAG - Sprint 4 (corrections critiques)
Fix 1 : Prompt RESPONSE strict anti-hallucination
Fix 2 : LLM juge séparé (Groq) pour CRITIQUE
Fix 3 : USE_GROQ_JUDGE indépendant de USE_GROQ
"""

import os, json, re
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from neo4j import GraphDatabase

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://localhost:11434")
MODEL_NAME      = os.getenv("MODEL_NAME",      "llama3.1:8b")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "nomic-embed-text")
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
NEO4J_URI       = os.getenv("NEO4J_URI",       "bolt://localhost:7687")
NEO4J_USER      = os.getenv("NEO4J_USER",      "neo4j")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD",  "")
CHROMA_DIR      = os.getenv("CHROMA_DIR",      "indexes/chroma_pfe500_baseline")
COLLECTION_NAME = "pfe_500_baseline"

# ── LLM principal (RESPONSE + SELF_CORRECT) ──────────────────
USE_GROQ        = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

# ── LLM juge séparé (CRITIQUE uniquement) ────────────────────
# FIX 2 : juge indépendant du générateur
USE_GROQ_JUDGE  = os.getenv("USE_GROQ_JUDGE", "true").lower() == "true"

TOP_K_VECTOR   = 5
TOP_K_GRAPH    = 10
CRITIQUE_SEUIL = 0.7
MAX_ITERATIONS = 3

# ══════════════════════════════════════════════════════════════
# INIT LLM PRINCIPAL
# ══════════════════════════════════════════════════════════════
def _build_groq_llm(model="llama-3.3-70b-versatile", max_tokens=800):
    """Construit un wrapper Groq synchrone."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    class _GroqLLM:
        def invoke(self, messages):
            msgs = []
            for m in messages:
                role = "system" if isinstance(m, SystemMessage) else "user"
                msgs.append({"role": role, "content": m.content})
            r = client.chat.completions.create(
                model=model, messages=msgs,
                temperature=0, max_tokens=max_tokens)
            class _R:
                content = r.choices[0].message.content
            return _R()
    return _GroqLLM()

def _get_llm():
    if USE_GROQ and GROQ_API_KEY:
        print("[LLM RESPONSE] Groq llama-3.3-70b-versatile")
        return _build_groq_llm(model="llama-3.3-70b-versatile", max_tokens=800)
    print(f"[LLM RESPONSE] Ollama {MODEL_NAME}")
    return ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL,
                      temperature=0, num_ctx=OLLAMA_NUM_CTX)

# ── LLM juge séparé pour CRITIQUE ────────────────────────────
def _get_judge_llm():
    """
    FIX 2 : LLM juge différent du générateur.
    Groq llama-3.3-70b évalue les réponses de llama3.1:8b.
    Casse le biais d'auto-évaluation (Pearson -0.082 -> cible > 0.40).
    """
    if USE_GROQ_JUDGE and GROQ_API_KEY:
        print("[LLM CRITIQUE] Groq llama-3.3-70b-versatile (juge séparé)")
        return _build_groq_llm(model="llama-3.3-70b-versatile", max_tokens=300)
    print(f"[LLM CRITIQUE] Ollama {MODEL_NAME} (fallback — biais possible)")
    return None  # None = utiliser _llm_call normal

llm       = _get_llm()
judge_llm = _get_judge_llm()

neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
emb_fn       = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectorstore  = Chroma(persist_directory=CHROMA_DIR,
                      embedding_function=emb_fn,
                      collection_name=COLLECTION_NAME)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_VECTOR})

# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    question         : str
    reformulated_q   : str
    graph_results    : list
    vector_results   : list
    fused_context    : str
    response         : str
    critique_score   : float
    critique_reason  : str
    iteration        : int
    final_response   : str
    trace            : list

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
_STOPWORDS = {"what","is","the","a","an","how","does","which","who","why",
              "when","in","for","of","to","and","or","are","was","used","tell","give"}

def _keywords(text: str) -> list:
    words = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS][:6]

def _llm_call(system: str, user: str) -> str:
    """Appel LLM principal (RESPONSE + SELF_CORRECT)."""
    try:
        r = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return r.content
    except Exception as e:
        return f"[LLM ERROR] {e}"

def _judge_call(system: str, user: str) -> str:
    """
    FIX 2 : Appel LLM juge séparé (CRITIQUE uniquement).
    Si judge_llm None -> fallback sur _llm_call.
    """
    if judge_llm is not None:
        try:
            r = judge_llm.invoke([SystemMessage(content=system),
                                   HumanMessage(content=user)])
            return r.content
        except Exception as e:
            print(f"[JUDGE] Erreur Groq : {e} — fallback Ollama")
    return _llm_call(system, user)

# ══════════════════════════════════════════════════════════════
# NOEUDS
# ══════════════════════════════════════════════════════════════

def node_query(state: AgentState) -> AgentState:
    q         = state.get("reformulated_q") or state["question"]
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])
    trace.append(f"[QUERY iter={iteration}] {q}")
    print(f"\n[QUERY] iter={iteration} | {q}")
    return {**state, "reformulated_q": q, "iteration": iteration, "trace": trace}


def node_graph_search(state: AgentState) -> AgentState:
    """Recherche multi-hop Neo4j : seed (hop=0), 1-hop, 2-hop."""
    q     = state["reformulated_q"]
    trace = state.get("trace", [])
    kws   = _keywords(q)
    results = []

    try:
        with neo4j_driver.session() as session:
            seed_names = []
            for kw in kws:
                rows = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS $kw
                       OR toLower(e.description) CONTAINS $kw
                    RETURN e.name AS name, e.description AS description,
                           e.type AS type LIMIT $lim
                """, kw=kw, lim=TOP_K_GRAPH).data()
                for r in rows:
                    results.append({**r, "hop": 0})
                    seed_names.append(r["name"])

            if seed_names:
                hop1 = session.run("""
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE a.name IN $names
                    RETURN a.name AS src, b.name AS tgt,
                           b.description AS description, b.type AS type,
                           r.description AS rel_desc LIMIT 20
                """, names=list(set(seed_names))).data()
                for r in hop1:
                    results.append({
                        "name": f"{r['src']} → {r['tgt']}",
                        "description": f"Relation: {r.get('rel_desc','')} | {r['tgt']}: {r.get('description','')[:200]}",
                        "type": "relation", "hop": 1
                    })

                tgt_names = list({r["tgt"] for r in hop1})[:5]
                if tgt_names:
                    hop2 = session.run("""
                        MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                        WHERE a.name IN $names
                        RETURN a.name AS src, b.name AS tgt,
                               b.description AS description,
                               r.description AS rel_desc LIMIT 10
                    """, names=tgt_names).data()
                    for r in hop2:
                        results.append({
                            "name": f"{r['src']} → {r['tgt']} (2-hop)",
                            "description": f"{r.get('rel_desc','')} | {r.get('description','')}",
                            "type": "2-hop", "hop": 2
                        })
    except Exception as e:
        trace.append(f"[GRAPH_SEARCH] Erreur Neo4j : {e}")

    seen, unique = set(), []
    for r in results:
        key = r.get("name", "")
        if key not in seen:
            seen.add(key)
            unique.append(r)

    trace.append(f"[GRAPH_SEARCH] {len(unique)} résultats (kws={kws})")
    print(f"[GRAPH_SEARCH] {len(unique)} résultats | kws={kws}")
    return {**state, "graph_results": unique, "trace": trace}


def node_vector_search(state: AgentState) -> AgentState:
    """Recherche sémantique ChromaDB."""
    q     = state["reformulated_q"]
    trace = state.get("trace", [])
    try:
        docs  = chroma_retriever.invoke(q)
        texts = [d.page_content for d in docs]
    except Exception as e:
        texts = []
        trace.append(f"[VECTOR_SEARCH] Erreur : {e}")
    trace.append(f"[VECTOR_SEARCH] {len(texts)} chunks")
    print(f"[VECTOR_SEARCH] {len(texts)} chunks")
    return {**state, "vector_results": texts, "trace": trace}


def node_fuse(state: AgentState) -> AgentState:
    """Fusion Neo4j + Chroma : seed > 1-hop > 2-hop > vecteurs."""
    graph   = state.get("graph_results", [])
    vectors = state.get("vector_results", [])
    trace   = state.get("trace", [])
    parts   = []

    seeds = [r for r in graph if r.get("hop", 0) == 0]
    if seeds:
        parts.append("=== Entités du graphe (correspondance directe) ===")
        for r in seeds[:6]:
            parts.append(f"• [{r.get('type','?')}] {r['name']}\n  {r.get('description','')[:250]}")

    rels1 = [r for r in graph if r.get("hop", 0) == 1]
    if rels1:
        parts.append("\n=== Relations 1-hop ===")
        for r in rels1[:8]:
            parts.append(f"• {r['name']}\n  {r.get('description','')[:200]}")

    rels2 = [r for r in graph if r.get("hop", 0) == 2]
    if rels2:
        parts.append("\n=== Relations 2-hop ===")
        for r in rels2[:4]:
            parts.append(f"• {r['name']}\n  {r.get('description','')[:150]}")

    if vectors:
        parts.append("\n=== Passages ChromaDB ===")
        for i, chunk in enumerate(vectors):
            parts.append(f"[Doc {i+1}] {chunk[:500]}")

    fused = "\n".join(parts) if parts else "Aucun contexte disponible."
    hop_counts = {0: len(seeds), 1: len(rels1), 2: len(rels2)}
    trace.append(f"[FUSE] {len(fused)} chars | hops={hop_counts} | vecteurs={len(vectors)}")
    print(f"[FUSE] {len(fused)} chars | graph_hops={hop_counts} | vecteurs={len(vectors)}")
    return {**state, "fused_context": fused, "trace": trace}


def node_response(state: AgentState) -> AgentState:
    """
    FIX 1 : Prompt strict anti-hallucination.
    Interdit toute extrapolation au-delà du contexte fourni.
    Cible : Faithfulness 0.30 -> 0.60-0.75
    """
    q         = state["reformulated_q"]
    context   = state.get("fused_context", "")
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])

    # ── FIX 1 : prompt strict (remplace lignes 282-289 originales) ──
    system = """You are a research assistant grounded EXCLUSIVELY in the
provided context.

STRICT RULES:
1. Use ONLY information explicitly present in the Knowledge Graph or
   Document chunks below.
2. If the answer is not in the context, respond EXACTLY:
   "The corpus does not contain information to answer this question."
3. ALWAYS cite the source after each claim: [Graph: entity_name] or [Doc N].
4. Never use external knowledge or general world facts.
5. If only partial information is available, state what is known AND
   mark what is missing as "[not in corpus]"."""

    user = f"""Context:
{context}

Question: {q}

Answer (use ONLY the context above, cite sources):"""

    answer = _llm_call(system, user)
    trace.append(f"[RESPONSE iter={iteration}] {answer[:120]}...")
    print(f"[RESPONSE iter={iteration}] {answer[:150]}")
    return {**state, "response": answer, "trace": trace}


def node_critique(state: AgentState) -> AgentState:
    """
    FIX 2 : Utilise judge_llm (Groq 70B) au lieu du LLM principal.
    Casse le biais auto-évaluation : Pearson -0.082 -> cible > 0.40.
    SELF_CORRECT devrait se déclencher sur 30-40% des questions.
    """
    q         = state["question"]
    response  = state.get("response", "")
    context   = state.get("fused_context", "")[:800]
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])

    system = """You are a STRICT RAG quality evaluator. You evaluate answers
generated by a different AI model. Be critical and objective.

Score from 0.0 to 1.0:
- Faithfulness (50%): Is EVERY claim in the answer explicitly supported by 
  the context? Penalize heavily any information not in the context.
- Completeness (30%): Does it fully answer the question using available context?
- Clarity (20%): Is it clear and well-structured?

IMPORTANT: If the answer says "I don't know" or "corpus does not contain" 
when the context DOES contain relevant info, score Completeness = 0.2.
If the answer invents facts not in the context, score Faithfulness = 0.0.

Score < 0.7 = needs improvement. Be strict.

Respond ONLY with valid JSON:
{"score": 0.65, "reason": "Brief explanation of what is missing or wrong"}"""

    user = f"""Question: {q}

Context (what the answer should be based on):
{context}

Answer to evaluate:
{response[:600]}

JSON evaluation:"""

    # ── FIX 2 : appel au juge séparé ─────────────────────────
    raw = _judge_call(system, user)

    try:
        m    = re.search(r"\{.*?\}", raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        score  = max(0.0, min(1.0, float(data.get("score", 0.5))))
        reason = data.get("reason", raw[:150])
    except Exception:
        score, reason = 0.5, raw[:150]

    trace.append(f"[CRITIQUE iter={iteration}] score={score:.2f} | {reason[:80]}")
    print(f"[CRITIQUE iter={iteration}] score={score:.2f} | {reason[:80]}")
    print(f"[CRITIQUE] LLM juge : {'Groq 70B' if judge_llm else 'Ollama 8B (fallback)'}")
    return {**state, "critique_score": score, "critique_reason": reason, "trace": trace}


def node_self_correct(state: AgentState) -> AgentState:
    """Reformule la question si score < seuil ET iter < MAX_ITERATIONS."""
    q         = state["question"]
    response  = state.get("response", "")[:300]
    reason    = state.get("critique_reason", "")
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])

    system = """You are a search query optimizer for a RAG system.
Given a question that didn't get a good answer, reformulate it to be:
1. More specific about the key concepts
2. Using different terminology
3. Breaking it into the core concept being asked
Return ONLY the reformulated question, nothing else."""

    user = f"""Original question: {q}
Previous answer (insufficient): {response}
Why it failed: {reason}

Better reformulated question:"""

    new_q    = _llm_call(system, user).strip()
    new_iter = iteration + 1
    trace.append(f"[SELF_CORRECT iter={new_iter}] \"{new_q}\"")
    print(f"[SELF_CORRECT] iter {iteration}->{new_iter}")
    print(f"  Original  : {q}")
    print(f"  Reformulé : {new_q}")
    return {**state, "reformulated_q": new_q, "iteration": new_iter, "trace": trace}


def node_finalize(state: AgentState) -> AgentState:
    score = state.get("critique_score", 0)
    iter_ = state.get("iteration", 0)
    trace = state.get("trace", [])
    trace.append(f"[FINALIZE] score={score:.2f}, iterations={iter_}")
    print(f"[FINALIZE] score={score:.2f} après {iter_} iteration(s)")
    return {**state, "final_response": state.get("response", ""), "trace": trace}


# ══════════════════════════════════════════════════════════════
# ROUTEUR
# ══════════════════════════════════════════════════════════════
def route_after_critique(state: AgentState) -> str:
    score = state.get("critique_score", 0.0)
    iter_ = state.get("iteration", 0)
    if score >= CRITIQUE_SEUIL:
        print(f"[ROUTE] score={score:.2f} >= {CRITIQUE_SEUIL} → FINALIZE")
        return "finalize"
    elif iter_ < MAX_ITERATIONS:
        print(f"[ROUTE] score={score:.2f} < {CRITIQUE_SEUIL}, iter={iter_}/{MAX_ITERATIONS} → SELF_CORRECT")
        return "self_correct"
    else:
        print(f"[ROUTE] MAX_ITERATIONS={MAX_ITERATIONS} atteint → FINALIZE")
        return "finalize"


# ══════════════════════════════════════════════════════════════
# BUILD AGENT
# ══════════════════════════════════════════════════════════════
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("query",         node_query)
    g.add_node("graph_search",  node_graph_search)
    g.add_node("vector_search", node_vector_search)
    g.add_node("fuse",          node_fuse)
    g.add_node("response",      node_response)
    g.add_node("critique",      node_critique)
    g.add_node("self_correct",  node_self_correct)
    g.add_node("finalize",      node_finalize)

    g.set_entry_point("query")
    g.add_edge("query",         "graph_search")
    g.add_edge("graph_search",  "vector_search")
    g.add_edge("vector_search", "fuse")
    g.add_edge("fuse",          "response")
    g.add_edge("response",      "critique")
    g.add_conditional_edges("critique", route_after_critique,
                            {"finalize": "finalize", "self_correct": "self_correct"})
    g.add_edge("self_correct",  "query")
    g.add_edge("finalize",      END)
    return g.compile()


# ══════════════════════════════════════════════════════════════
# PHOENIX
# ══════════════════════════════════════════════════════════════
def setup_phoenix():
    try:
        import phoenix as px
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        session = px.launch_app()
        provider = TracerProvider()
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter("http://localhost:6006/v1/traces")))
        otel_trace.set_tracer_provider(provider)
        LangChainInstrumentor().instrument()
        print(f"[PHOENIX] Dashboard : {session.url}")
        return True
    except ImportError as e:
        print(f"[PHOENIX] Non disponible : {e}")
        return False


# ══════════════════════════════════════════════════════════════
# API PRINCIPALE
# ══════════════════════════════════════════════════════════════
def run_agent(question: str, use_phoenix: bool = False) -> dict:
    if use_phoenix:
        setup_phoenix()
    agent = build_agent()
    init: AgentState = {
        "question": question, "reformulated_q": question,
        "graph_results": [], "vector_results": [],
        "fused_context": "", "response": "",
        "critique_score": 0.0, "critique_reason": "",
        "iteration": 0, "final_response": "", "trace": [],
    }
    print(f"\n{'='*60}\nQUESTION : {question}\n{'='*60}")
    result = agent.invoke(init)
    print(f"\n--- RÉPONSE FINALE ---\n{result['final_response']}")
    print(f"\n--- SCORE / ITERATIONS ---")
    print(f"Score : {result['critique_score']:.2f} | Iterations : {result['iteration']}")
    print(f"LLM juge : {'Groq 70B' if judge_llm else 'Ollama 8B'}")
    return result
