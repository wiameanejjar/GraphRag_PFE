# src/agent/graph_v2.py
import os, json, re, asyncio
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()
import nest_asyncio
nest_asyncio.apply()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from groq import AsyncGroq, Groq
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
OLLAMA_URL      = os.getenv("OLLAMA_URL",     "http://localhost:11434")
MODEL_NAME      = os.getenv("MODEL_NAME",     "llama3.1:8b")
EMBED_MODEL     = os.getenv("EMBED_MODEL",    "nomic-embed-text")
EMBED_DIM       = int(os.getenv("EMBED_DIM",  "768"))
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
GROQ_API_KEY    = os.getenv("GROQ_API_KEY",   "")
INDEX_DIR       = Path(os.getenv("INDEX_DIR", "indexes/lightrag_500_connected_v2"))
CHROMA_DIR      = os.getenv("CHROMA_DIR",     "indexes/chroma_pfe500_baseline")
COLLECTION_NAME = "pfe_500_baseline"

USE_GROQ        = os.getenv("USE_GROQ",       "false").lower() == "true"
USE_GROQ_JUDGE  = os.getenv("USE_GROQ_JUDGE", "true").lower()  == "true"

TOP_K_LIGHTRAG  = 5
CHUNK_TOP_K     = 3
TOP_K_CHROMA    = 5   # uniquement pour RAGAS
CRITIQUE_SEUIL  = 0.75
MAX_ITERATIONS  = 3

# ══════════════════════════════════════════════════════════════
# INIT LLM
# ══════════════════════════════════════════════════════════════
def _build_groq_sync(model="llama-3.3-70b-versatile", max_tokens=800):
    client = Groq(api_key=GROQ_API_KEY)
    class _G:
        def invoke(self, messages):
            msgs = [{"role": "system" if isinstance(m, SystemMessage)
                     else "user", "content": m.content} for m in messages]
            r = client.chat.completions.create(
                model=model, messages=msgs, temperature=0, max_tokens=max_tokens)
            class _R:
                content = r.choices[0].message.content
            return _R()
    return _G()

def _get_llm():
    if USE_GROQ and GROQ_API_KEY:
        print("[LLM RESPONSE] Groq llama-3.3-70b")
        return _build_groq_sync("llama-3.3-70b-versatile", 800)
    print(f"[LLM RESPONSE] Ollama {MODEL_NAME}")
    return ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL,
                      temperature=0, num_ctx=OLLAMA_NUM_CTX)

def _get_judge_llm():
    if USE_GROQ_JUDGE and GROQ_API_KEY:
        print("[LLM CRITIQUE] Groq llama-3.3-70b (juge séparé)")
        return _build_groq_sync("llama-3.3-70b-versatile", 300)
    print(f"[LLM CRITIQUE] Ollama fallback")
    return None

llm       = _get_llm()
judge_llm = _get_judge_llm()
print(f"======= JUDGE LLM ==========")
print(f"Type : {type(judge_llm)}")

# ══════════════════════════════════════════════════════════════
# INIT LIGHTRAG
# ══════════════════════════════════════════════════════════════
groq_async_client = AsyncGroq(api_key=GROQ_API_KEY)

async def groq_llm_func(prompt, system_prompt=None,
                         history_messages=None, **kwargs):
    history_messages = history_messages or []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in history_messages:
        if isinstance(m, dict):
            messages.append(m)
    messages.append({"role": "user", "content": prompt})
    r = await groq_async_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages, temperature=0.0, max_tokens=800)
    return (r.choices[0].message.content or "").strip()

async def embedding_func(texts):
    import httpx
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for text in texts:
            r = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text})
            results.append(r.json()["embedding"])
    return results

async def _build_lightrag():
    rag = LightRAG(
        working_dir=str(INDEX_DIR),
        llm_model_func=groq_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            model_name=EMBED_MODEL,
            func=embedding_func,
        ),
        llm_model_max_async=1,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag

rag_instance = asyncio.get_event_loop().run_until_complete(_build_lightrag())
print(f"[LIGHTRAG] Initialisé → {INDEX_DIR}")

# Chroma — uniquement pour fournir vector_results à RAGAS
emb_fn = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectorstore = Chroma(persist_directory=CHROMA_DIR,
                     embedding_function=emb_fn,
                     collection_name=COLLECTION_NAME)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHROMA})
print(f"[CHROMA] Chargé")

# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    question         : str
    reformulated_q   : str
    lightrag_context : str   # contexte LightRAG (graphe + chunks natifs)
    vector_results   : list  # chunks Chroma pour RAGAS uniquement
    response         : str
    critique_score   : float
    critique_reason  : str
    iteration        : int
    final_response   : str
    trace            : list

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def _llm_call(system, user):
    try:
        r = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return r.content
    except Exception as e:
        return f"[LLM ERROR] {e}"

def _judge_call(system, user):
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


def node_hybrid_search(state: AgentState) -> AgentState:
    """
    LightRAG mode=hybrid : interroge le graphe ET les chunks vectoriels
    en une seule requête. Remplace GRAPH_SEARCH + VECTOR_SEARCH + FUSE du Sprint 3.
    Retourne le contexte brut (entités + relations + chunks) prêt pour RESPONSE.
    """
    q     = state["reformulated_q"]
    trace = state.get("trace", [])

    async def _query():
        ctx = await rag_instance.aquery(
            q,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=TOP_K_LIGHTRAG,
                chunk_top_k=CHUNK_TOP_K,
                enable_rerank=False,
            ),
        )
        return ctx or ""

    try:
        ctx = asyncio.get_event_loop().run_until_complete(_query())
    except Exception as e:
        ctx = ""
        trace.append(f"[HYBRID_SEARCH] Erreur : {e}")

    trace.append(f"[HYBRID_SEARCH] context={len(ctx)} chars")
    print(f"[HYBRID_SEARCH] context={len(ctx)} chars")
    return {**state, "lightrag_context": ctx, "trace": trace}


def node_chroma_for_ragas(state: AgentState) -> AgentState:
    """
    Récupère les chunks Chroma UNIQUEMENT pour alimenter RAGAS
    (context_precision / context_recall).
    N'est PAS utilisé pour générer la réponse — LightRAG s'en charge.
    """
    q     = state["reformulated_q"]
    trace = state.get("trace", [])
    try:
        docs  = chroma_retriever.invoke(q)
        texts = [d.page_content for d in docs]
    except Exception as e:
        texts = []
        trace.append(f"[CHROMA_RAGAS] Erreur : {e}")

    trace.append(f"[CHROMA_RAGAS] {len(texts)} chunks pour RAGAS")
    print(f"[CHROMA_RAGAS] {len(texts)} chunks (RAGAS only)")
    return {**state, "vector_results": texts, "trace": trace}


def node_response(state: AgentState) -> AgentState:
    """
    Génère la réponse à partir du contexte LightRAG.
    Prompt strict anti-hallucination (Fix Sprint 4).
    """
    q         = state["reformulated_q"]
    context   = state.get("lightrag_context", "")
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])

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
    Juge séparé Groq 70B (Fix Sprint 4).
    Casse le biais auto-évaluation.
    """
    q         = state["question"]
    response  = state.get("response", "")
    context   = state.get("lightrag_context", "")[:800]
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])

    system = """You are a STRICT RAG quality evaluator. You evaluate answers
generated by a different AI model. Be critical and objective.

Score from 0.0 to 1.0:
- Faithfulness (50%): Is EVERY claim explicitly supported by the context?
  Penalize heavily any information not in the context.
- Completeness (30%): Does it fully answer the question using available context?
- Clarity (20%): Is it clear and well-structured?

Rules:
- Answer invents facts not in context → Faithfulness = 0.0
- Answer correctly says "not in corpus" when info absent → score >= 0.8
- Answer says "not in corpus" but context HAS the info → score < 0.4
- Score < 0.75 triggers SELF_CORRECT

Respond ONLY with valid JSON:
{"score": 0.65, "reason": "Brief explanation"}"""

    user = f"""Question: {q}

Context:
{context}

Answer to evaluate:
{response[:600]}

JSON:"""

    raw = _judge_call(system, user)
    try:
        m    = re.search(r"\{.*?\}", raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        score  = max(0.0, min(1.0, float(data.get("score", 0.5))))
        reason = data.get("reason", raw[:150])
    except Exception:
        score, reason = 0.5, raw[:150]

    trace.append(f"[CRITIQUE iter={iteration}] score={score:.2f} | {reason[:80]}")
    print(f"[CRITIQUE iter={iteration}] score={score:.2f} | {'Groq 70B' if judge_llm else 'Ollama'}")
    return {**state, "critique_score": score, "critique_reason": reason, "trace": trace}


def node_self_correct(state: AgentState) -> AgentState:
    """Reformule si score < seuil ET iter < MAX_ITERATIONS."""
    q         = state["question"]
    response  = state.get("response", "")[:300]
    reason    = state.get("critique_reason", "")
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])

    system = """You are a search query optimizer for a RAG system.
Reformulate the question to be more specific and retrieve better information.
Return ONLY the reformulated question."""

    user = f"""Original: {q}
Failed answer: {response}
Why: {reason}
Better question:"""

    new_q    = _llm_call(system, user).strip()
    new_iter = iteration + 1
    trace.append(f"[SELF_CORRECT iter={new_iter}] \"{new_q}\"")
    print(f"[SELF_CORRECT] {iteration}→{new_iter} | {new_q}")
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
        print(f"[ROUTE] MAX_ITERATIONS atteint → FINALIZE")
        return "finalize"


# ══════════════════════════════════════════════════════════════
# BUILD + API
# ══════════════════════════════════════════════════════════════
def build_agent():
    g = StateGraph(AgentState)

    g.add_node("query",             node_query)
    g.add_node("hybrid_search",     node_hybrid_search)     # LightRAG
    g.add_node("chroma_for_ragas",  node_chroma_for_ragas)  # RAGAS only
    g.add_node("response",          node_response)
    g.add_node("critique",          node_critique)
    g.add_node("self_correct",      node_self_correct)
    g.add_node("finalize",          node_finalize)

    g.set_entry_point("query")
    g.add_edge("query",            "hybrid_search")
    g.add_edge("hybrid_search",    "chroma_for_ragas")
    g.add_edge("chroma_for_ragas", "response")
    g.add_edge("response",         "critique")
    g.add_conditional_edges("critique", route_after_critique,
                            {"finalize": "finalize",
                             "self_correct": "self_correct"})
    g.add_edge("self_correct",     "query")
    g.add_edge("finalize",         END)

    return g.compile()


def run_agent(question: str, use_phoenix: bool = False) -> dict:
    """
    Agent Agentic GraphRAG avec LightRAG hybrid search.
    Architecture : QUERY → HYBRID_SEARCH (LightRAG mode=hybrid)
                 → CHROMA_FOR_RAGAS → RESPONSE → CRITIQUE (Groq juge)
                 → SELF_CORRECT (max 3) → FINALIZE
    """
    if use_phoenix:
        _setup_phoenix()

    agent = build_agent()
    init: AgentState = {
        "question":          question,
        "reformulated_q":    question,
        "lightrag_context":  "",
        "vector_results":    [],
        "response":          "",
        "critique_score":    0.0,
        "critique_reason":   "",
        "iteration":         0,
        "final_response":    "",
        "trace":             [],
    }

    print(f"\n{'='*60}\nQUESTION : {question}\n{'='*60}")
    result = agent.invoke(init)
    print(f"\n--- RÉPONSE FINALE ---\n{result['final_response']}")
    print(f"Score : {result['critique_score']:.2f} | Iterations : {result['iteration']}")
    return result


def _setup_phoenix():
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
        print(f"[PHOENIX] {session.url}")
    except ImportError as e:
        print(f"[PHOENIX] {e}")