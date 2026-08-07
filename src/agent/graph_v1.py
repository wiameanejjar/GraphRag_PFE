"""
src/agent/graph_v1.py
Agentic GraphRAG - Sprint 3 complet 
Architecture : QUERY → GRAPH_SEARCH + VECTOR_SEARCH → FUSE → RESPONSE → CRITIQUE → SELF_CORRECT (max 3)
Neo4j schéma : (:Entity {name, description, type}) -[:RELATES_TO]-> (:Entity)
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

from src.utils.groq_rotation import groq_chat_completion, AllGroqKeysExhaustedError

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
USE_GROQ        = os.getenv("USE_GROQ", "true").lower() == "true"
USE_GROQ_JUDGE = os.getenv("USE_GROQ_JUDGE", "true").lower() == "true"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

TOP_K_VECTOR   = 5
TOP_K_GRAPH    = 10
CRITIQUE_SEUIL = 0.7   # seuil de qualité (0-1)
MAX_ITERATIONS = 3     # max 3 self-corrections

# ══════════════════════════════════════════════════════════════
# INIT CLIENTS
# ══════════════════════════════════════════════════════════════
def _get_llm():
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
                    messages=msgs, temperature=0, max_tokens=800)
                class _R:
                    content = r.choices[0].message.content
                return _R()
        return _GroqLLM()
    print(f"[LLM] Ollama {MODEL_NAME}")
    return ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL,
                      temperature=0, num_ctx=OLLAMA_NUM_CTX)

llm           = _get_llm()
print("\n========== GENERATION LLM ==========")
print("Type :", type(llm))

if hasattr(llm, "model"):
    print("Model :", llm.model)

if hasattr(llm, "base_url"):
    print("Base URL :", llm.base_url)


def _get_judge_llm():
    if USE_GROQ_JUDGE and GROQ_API_KEY:
        class _GroqJudge:
            def invoke(self, messages):
                msgs = [{"role": "system" if isinstance(m, SystemMessage)
                         else "user", "content": m.content} for m in messages]
                r = groq_chat_completion(
                    model="llama-3.3-70b-versatile",
                    messages=msgs, temperature=0, max_tokens=300)
                class _R:
                    content = r.choices[0].message.content
                return _R()
        return _GroqJudge()
    return llm  # fallback Ollama

judge_llm = _get_judge_llm()
print("\n========== JUDGE LLM ==========")
print("Type :", type(judge_llm))

if hasattr(judge_llm, "model"):
    print("Model :", judge_llm.model)

if hasattr(judge_llm, "base_url"):
    print("Base URL :", judge_llm.base_url)


neo4j_driver  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
emb_fn        = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectorstore   = Chroma(persist_directory=CHROMA_DIR,
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

def _keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS][:6]

def _llm_call(system: str, user: str) -> str:
    try:
        r = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return r.content
    except AllGroqKeysExhaustedError:
        raise
    except Exception as e:
        return f"[LLM ERROR] {e}"


# ══════════════════════════════════════════════════════════════
# NOEUDS
# ══════════════════════════════════════════════════════════════

# ── NOEUD 1 : QUERY ───────────────────────────────────────────
def node_query(state: AgentState) -> AgentState:
    """
    Semaine 1 : Initialise / reçoit la question active.
    En cas de SELF_CORRECT, reformulated_q est déjà mis à jour.
    """
    q         = state.get("reformulated_q") or state["question"]
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])
    trace.append(f"[QUERY iter={iteration}] {q}")
    print(f"\n[QUERY] iter={iteration} | {q}")
    return {**state, "reformulated_q": q, "iteration": iteration, "trace": trace}


# ── NOEUD 2 : GRAPH_SEARCH ────────────────────────
def node_graph_search(state: AgentState) -> AgentState:
    """
    Semaine 2 : Recherche multi-hop dans Neo4j.
    Étape 1 : chercher les entités correspondant aux mots-clés.
    Étape 2 : traverser 1 hop de relations depuis ces entités.
    Schéma : (:Entity {name, description, type}) -[:RELATES_TO]-> (:Entity)
    """
    q     = state["reformulated_q"]
    trace = state.get("trace", [])
    kws   = _keywords(q)
    results = []

    try:
        with neo4j_driver.session() as session:

            # ── Étape 1 : entités seed ────────────────────────
            seed_names = []
            for kw in kws:
                rows = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS $kw
                       OR toLower(e.description) CONTAINS $kw
                    RETURN e.name AS name, e.description AS description,
                           e.type AS type
                    LIMIT $lim
                """, kw=kw, lim=TOP_K_GRAPH).data()
                for r in rows:
                    results.append({**r, "hop": 0})
                    seed_names.append(r["name"])

            # ── Étape 2 : 1-hop depuis les seeds (multi-hop) ──
            if seed_names:
                hop1 = session.run("""
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE a.name IN $names
                    RETURN a.name AS src, b.name AS tgt,
                           b.description AS description,
                           b.type AS type,
                           r.description AS rel_desc
                    LIMIT 20
                """, names=list(set(seed_names))).data()

                for r in hop1:
                    results.append({
                        "name"       : f"{r['src']} → {r['tgt']}",
                        "description": f"Relation: {r.get('rel_desc','')} | {r['tgt']}: {r.get('description','')}",
                        "type"       : "relation",
                        "hop"        : 1
                    })

                # ── Étape 3 : 2-hop (GraphRAG multi-hop) ──────
                tgt_names = list({r["tgt"] for r in hop1})[:5]
                if tgt_names:
                    hop2 = session.run("""
                        MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                        WHERE a.name IN $names
                        RETURN a.name AS src, b.name AS tgt,
                               b.description AS description,
                               r.description AS rel_desc
                        LIMIT 10
                    """, names=tgt_names).data()

                    for r in hop2:
                        results.append({
                            "name"       : f"{r['src']} → {r['tgt']} (2-hop)",
                            "description": f"{r.get('rel_desc','')} | {r.get('description','')}",
                            "type"       : "2-hop",
                            "hop"        : 2
                        })

    except Exception as e:
        trace.append(f"[GRAPH_SEARCH] Erreur Neo4j : {e}")
        print(f"[GRAPH_SEARCH] Erreur : {e}")

    # Dédupliquer
    seen, unique = set(), []
    for r in results:
        key = r.get("name", "")
        if key not in seen:
            seen.add(key)
            unique.append(r)

    trace.append(f"[GRAPH_SEARCH] {len(unique)} résultats (kws={kws})")
    print(f"[GRAPH_SEARCH] {len(unique)} résultats | kws={kws}")
    return {**state, "graph_results": unique, "trace": trace}


# ── NOEUD 3 : VECTOR_SEARCH  ───────────────────────
def node_vector_search(state: AgentState) -> AgentState:
    """Semaine 2 : Recherche sémantique dans ChromaDB."""
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


# ── NOEUD 4 : FUSE  ────────────────────────────────
def node_fuse(state: AgentState) -> AgentState:
    """
    Semaine 2 : Fusion intelligente Neo4j + Chroma.
    Priorité : hop=0 (entités directes) > hop=1 > hop=2 > vecteurs.
    """
    graph   = state.get("graph_results", [])
    vectors = state.get("vector_results", [])
    trace   = state.get("trace", [])
    parts   = []

    # Graph : entités directes
    seeds = [r for r in graph if r.get("hop", 0) == 0]
    if seeds:
        parts.append("=== Entités du graphe (correspondance directe) ===")
        for r in seeds[:6]:
            parts.append(f"• [{r.get('type','?')}: {r['name']}]\n  {r.get('description','')[:250]}")

    # Graph : relations 1-hop
    rels1 = [r for r in graph if r.get("hop", 0) == 1]
    if rels1:
        parts.append("\n=== Relations 1-hop (GraphRAG) ===")
        for r in rels1[:8]:
            parts.append(f"• {r['name']}\n  {r.get('description','')[:200]}")

    # Graph : 2-hop
    rels2 = [r for r in graph if r.get("hop", 0) == 2]
    if rels2:
        parts.append("\n=== Relations 2-hop (multi-hop) ===")
        for r in rels2[:4]:
            parts.append(f"• {r['name']}\n  {r.get('description','')[:150]}")

    # Vecteurs
    if vectors:
        parts.append("\n=== Passages pertinents (ChromaDB) ===")
        for i, chunk in enumerate(vectors):
            parts.append(f"[Doc {i+1}] {chunk[:500]}")

    fused = "\n".join(parts) if parts else "Aucun contexte disponible."
    hop_counts = {0: len(seeds), 1: len(rels1), 2: len(rels2)}
    trace.append(f"[FUSE] {len(fused)} chars | hops={hop_counts} | vecteurs={len(vectors)}")
    print(f"[FUSE] {len(fused)} chars | graph_hops={hop_counts} | vecteurs={len(vectors)}")
    return {**state, "fused_context": fused, "trace": trace}


# ── NOEUD 5 : RESPONSE  ────────────────────────────
def node_response(state: AgentState) -> AgentState:
    """
    Semaine 1 : Génère la réponse depuis le contexte fusionné.
    Le LLM DOIT synthétiser même si l info est partielle.
    """
    q         = state["reformulated_q"]
    context   = state.get("fused_context", "")
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])

    system = """You are a research assistant grounded EXCLUSIVELY in the
        provided context.
        STRICT RULES:
        1. Use ONLY information explicitly present in the Knowledge Graph or
        Document chunks below.
        2. If the answer is not in the context, respond EXACTLY:
        "The corpus does not contain information to answer this question."
        3. ALWAYS cite the source: [Graph: entity_name] or [Doc N].
        4. Never use external knowledge or general world facts.
        5. If only partial information is available, state what is known AND
        mark what is missing as "[not in corpus]"."""
    user = f"""Context:
{context}

Question: {q}

Provide a comprehensive answer based on the context above:"""

    answer = _llm_call(system, user)
    trace.append(f"[RESPONSE iter={iteration}] {answer[:120]}...")
    print(f"[RESPONSE iter={iteration}] {answer[:150]}")
    return {**state, "response": answer, "trace": trace}


# ── NOEUD 6 : CRITIQUE  ───────────────────────
def node_critique(state: AgentState) -> AgentState:
    """
    Semaine 1+3 : Évalue la réponse (0.0 → 1.0).
    Critères : complétude, fidélité au contexte, clarté.
    """
    q         = state["question"]
    response  = state.get("response", "")
    context   = state.get("fused_context", "")[:800]
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])

    system = """You are a strict RAG quality evaluator.
Score the answer from 0.0 to 1.0 based on:
- Completeness (40%): Does it fully answer the question?
- Faithfulness (40%): Is it grounded in the context, no hallucination?
- Clarity (20%): Is it clear and well-structured?

A score of 1.0 = perfect. 0.7+ = acceptable. Below 0.7 = needs improvement.

Respond ONLY with valid JSON, no extra text:
{"score": 0.75, "reason": "Brief explanation..."}"""

    user = f"""Question: {q}

Context excerpt:
{context}

Answer to evaluate:
{response[:600]}

JSON:"""

    raw = judge_llm.invoke([SystemMessage(content=system), 
                        HumanMessage(content=user)]).content
    try:
        m    = re.search(r"\{.*?\}", raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        score  = max(0.0, min(1.0, float(data.get("score", 0.5))))
        reason = data.get("reason", raw[:150])
    except Exception:
        score, reason = 0.5, raw[:150]

    trace.append(f"[CRITIQUE iter={iteration}] score={score:.2f} | {reason[:80]}")
    print(f"[CRITIQUE iter={iteration}] score={score:.2f} | {reason[:80]}")
    return {**state, "critique_score": score, "critique_reason": reason, "trace": trace}


# ── NOEUD 7 : SELF_CORRECT  ────────────────────────
def node_self_correct(state: AgentState) -> AgentState:
    """
    Semaine 3 : Reformule la question si score < seuil ET iter < MAX_ITERATIONS.
    Le routeur garantit qu on n entre ici que si les deux conditions sont vraies.
    """
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

    new_q     = _llm_call(system, user).strip()
    new_iter  = iteration + 1
    trace.append(f"[SELF_CORRECT iter={new_iter}] \"{new_q}\"")
    print(f"[SELF_CORRECT] iter {iteration}→{new_iter}")
    print(f"  Original  : {q}")
    print(f"  Reformulé : {new_q}")
    return {**state, "reformulated_q": new_q, "iteration": new_iter, "trace": trace}


# ── NOEUD 8 : FINALIZE ────────────────────────────────────────
def node_finalize(state: AgentState) -> AgentState:
    """Valide et copie la réponse finale."""
    score = state.get("critique_score", 0)
    iter_ = state.get("iteration", 0)
    trace = state.get("trace", [])
    trace.append(f"[FINALIZE] Accepté — score={score:.2f}, iterations={iter_}")
    print(f"[FINALIZE]  score={score:.2f} après {iter_} iteration(s)")
    return {**state, "final_response": state.get("response", ""), "trace": trace}


# ══════════════════════════════════════════════════════════════
# ROUTEUR (edge conditionnel après CRITIQUE)
# ══════════════════════════════════════════════════════════════
def route_after_critique(state: AgentState) -> str:
    """
    Détermine le prochain nœud en fonction du score de critique.
    - score >= CRITIQUE_SEUIL → finalize
    - score <  CRITIQUE_SEUIL ET iteration < MAX_ITERATIONS → self_correct
    - score <  CRITIQUE_SEUIL ET iteration >= MAX_ITERATIONS → finalize quand même
    """
    score = state.get("critique_score", 0.0)
    iter_ = state.get("iteration", 0)

    if score >= CRITIQUE_SEUIL:
        print(f"[ROUTE]  score={score:.2f} >= {CRITIQUE_SEUIL} → FINALIZE")
        return "finalize"
    elif iter_ < MAX_ITERATIONS:
        print(f"[ROUTE]  score={score:.2f} < {CRITIQUE_SEUIL}, iter={iter_}/{MAX_ITERATIONS} → SELF_CORRECT")
        return "self_correct"
    else:
        print(f"[ROUTE]  score={score:.2f} mais MAX_ITERATIONS={MAX_ITERATIONS} atteint → FINALIZE")
        return "finalize"


# ══════════════════════════════════════════════════════════════
# CONSTRUCTION DU GRAPHE LANGGRAPH
# ══════════════════════════════════════════════════════════════
def build_agent():
    g = StateGraph(AgentState)

    # Noeuds
    g.add_node("query",         node_query)
    g.add_node("graph_search",  node_graph_search)
    g.add_node("vector_search", node_vector_search)
    g.add_node("fuse",          node_fuse)
    g.add_node("response",      node_response)
    g.add_node("critique",      node_critique)
    g.add_node("self_correct",  node_self_correct)
    g.add_node("finalize",      node_finalize)

    # Flux principal
    g.set_entry_point("query")
    g.add_edge("query",         "graph_search")
    g.add_edge("graph_search",  "vector_search")
    g.add_edge("vector_search", "fuse")
    g.add_edge("fuse",          "response")
    g.add_edge("response",      "critique")

    # Branchement conditionnel
    g.add_conditional_edges(
        "critique",
        route_after_critique,
        {"finalize": "finalize", "self_correct": "self_correct"}
    )

    # SELF_CORRECT relance le cycle depuis QUERY
    g.add_edge("self_correct", "query")
    g.add_edge("finalize",     END)

    return g.compile()


# ══════════════════════════════════════════════════════════════
# PHOENIX TRACING 
# ══════════════════════════════════════════════════════════════
def setup_phoenix():
    """Configure Arize Phoenix — localhost:6006."""
    try:
        import phoenix as px
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        session = px.launch_app()
        print(f"[PHOENIX] Dashboard : {session.url}")

        provider = TracerProvider()
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter("http://localhost:6006/v1/traces"))
        )
        otel_trace.set_tracer_provider(provider)
        LangChainInstrumentor().instrument()
        print("[PHOENIX] Tracing actif → http://localhost:6006")
        return True
    except ImportError as e:
        print(f"[PHOENIX] Non disponible : {e}")
        print("  pip install arize-phoenix openinference-instrumentation-langchain")
        return False


# ══════════════════════════════════════════════════════════════
# API PRINCIPALE
# ══════════════════════════════════════════════════════════════
def run_agent(question: str, use_phoenix: bool = False) -> dict:
    """
    Lance l'agent Agentic GraphRAG sur une question.

    Paramètres:
        question    : la question posée
        use_phoenix : True = activer le tracing Arize (Semaine 4)

    Retourne:
        dict avec keys: question, final_response, critique_score,
                        iteration, trace, graph_results, vector_results
    """
    if use_phoenix:
        setup_phoenix()

    agent = build_agent()

    init: AgentState = {
        "question"       : question,
        "reformulated_q" : question,
        "graph_results"  : [],
        "vector_results" : [],
        "fused_context"  : "",
        "response"       : "",
        "critique_score" : 0.0,
        "critique_reason": "",
        "iteration"      : 0,
        "final_response" : "",
        "trace"          : [],
    }

    print(f"\n{'='*60}")
    print(f"QUESTION : {question}")
    print(f"{'='*60}")

    result = agent.invoke(init)

    print(f"\n--- RÉPONSE FINALE ---")
    print(result["final_response"])
    print(f"\n--- SCORE / ITERATIONS ---")
    print(f"Score : {result['critique_score']:.2f} | Iterations : {result['iteration']}")
    print(f"Raison : {result['critique_reason'][:120]}")

    return result


# ══════════════════════════════════════════════════════════════
# 5 QUESTIONS DE VALIDATION 
# ══════════════════════════════════════════════════════════════
TEST_QUESTIONS = [
    # Question dont la réponse est dans le graphe
    "What is the Multi-Track Resolution Strategy used for in federated learning?",
    # Question multi-hop
    "How does GS-Quant address Knowledge Graph Completion?",
    # Question sur une entité spécifique
    "What model uses Segment Anything SAM as a spatial grouping prior?",
    # Question sur une méthode
    "What is DPRM and what type of models does it serve as a plug-in for?",
    # Test anti-hallucination
    "What is the capital of France?",
]

if __name__ == "__main__":
    import json as _json

    print("\n" + "═"*60)
    print("  VALIDATION — 5 QUESTIONS TYPES")
    print("═"*60)

    summary = []
    for q in TEST_QUESTIONS:
        r = run_agent(q, use_phoenix=False)
        summary.append({
            "question"  : r["question"],
            "answer"    : r["final_response"][:200],
            "score"     : r["critique_score"],
            "iterations": r["iteration"],
        })
        print("─"*60)

    print("\n\n══════ RÉSUMÉ FINAL ══════")
    print(f"Seuil acceptation : {CRITIQUE_SEUIL} | Max iter : {MAX_ITERATIONS}\n")
    for r in summary:
        status = "✓" if r["score"] >= CRITIQUE_SEUIL else "✗"
        print(f"{status} score={r['score']:.2f} iter={r['iterations']} | {r['question'][:55]}")
