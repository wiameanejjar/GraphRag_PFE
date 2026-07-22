# src/agent/graph_v3.py
import os, json, re, asyncio
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()
import nest_asyncio
nest_asyncio.apply()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from groq import AsyncGroq, Groq
from openai import AsyncOpenAI, OpenAI
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
OLLAMA_URL      = os.getenv("OLLAMA_URL",     "http://localhost:11434")
MODEL_NAME      = os.getenv("MODEL_NAME",     "llama3.1:8b")          # générateur
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "mistral:7b")        # juge local — DOIT être != MODEL_NAME
EMBED_MODEL     = os.getenv("EMBED_MODEL",    "nomic-embed-text")
EMBED_DIM       = int(os.getenv("EMBED_DIM",  "768"))
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
GROQ_API_KEY    = os.getenv("GROQ_API_KEY",   "")
INDEX_DIR       = Path(os.getenv("INDEX_DIR", "indexes/lightrag_500_connected_v2"))

NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL",   "z-ai/glm-5.2")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
USE_NVIDIA      = os.getenv("USE_NVIDIA",     "false").lower() == "true"

USE_GROQ        = os.getenv("USE_GROQ",       "false").lower() == "true"
USE_GROQ_JUDGE  = os.getenv("USE_GROQ_JUDGE", "true").lower()  == "true"
GROQ_GENERATOR_MODEL = os.getenv("GROQ_GENERATOR_MODEL", "llama-3.1-8b-instant")
GROQ_JUDGE_MODEL     = os.getenv("GROQ_JUDGE_MODEL",     "llama-3.3-70b-versatile")

TOP_K_LIGHTRAG  = int(os.getenv("TOP_K_LIGHTRAG", "40"))
CHUNK_TOP_K     = int(os.getenv("CHUNK_TOP_K", "20"))
CRITIQUE_SEUIL  = float(os.getenv("CRITIQUE_SEUIL", "0.75"))
MAX_ITERATIONS  = 3

if JUDGE_MODEL_NAME == MODEL_NAME:
    raise ValueError(
        f"JUDGE_MODEL_NAME ('{JUDGE_MODEL_NAME}') ne peut PAS être égal à "
        f"MODEL_NAME ('{MODEL_NAME}') — le juge doit toujours être un modèle "
        f"différent du générateur."
    )

if USE_GROQ and USE_GROQ_JUDGE and GROQ_GENERATOR_MODEL == GROQ_JUDGE_MODEL:
    raise ValueError(
        f"GROQ_GENERATOR_MODEL et GROQ_JUDGE_MODEL sont tous les deux "
        f"'{GROQ_GENERATOR_MODEL}' — le juge deviendrait le même modèle que "
        f"le générateur (perte de l'indépendance du juge)."
    )

# ══════════════════════════════════════════════════════════════
# INIT LLM — Générateur
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

def _build_nvidia_sync(model=NVIDIA_MODEL, max_tokens=800):
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    class _N:
        def invoke(self, messages):
            msgs = [{"role": "system" if isinstance(m, SystemMessage)
                     else "user", "content": m.content} for m in messages]
            r = client.chat.completions.create(
                model=model, messages=msgs, temperature=0, max_tokens=max_tokens)
            class _R:
                content = r.choices[0].message.content
            return _R()
    return _N()

def _get_llm():
    if USE_NVIDIA and NVIDIA_API_KEY:
        print(f"[LLM GENERATOR] NVIDIA {NVIDIA_MODEL}")
        return _build_nvidia_sync(NVIDIA_MODEL, 800)
    if USE_GROQ and GROQ_API_KEY:
        print(f"[LLM GENERATOR] Groq {GROQ_GENERATOR_MODEL}")
        return _build_groq_sync(GROQ_GENERATOR_MODEL, 800)
    print(f"[LLM GENERATOR] Ollama {MODEL_NAME}")
    return ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL,
                      temperature=0, num_ctx=OLLAMA_NUM_CTX)

# ══════════════════════════════════════════════════════════════
# INIT LLM — Judge (DOIT être différent du générateur)
# ══════════════════════════════════════════════════════════════
def _get_judge_llm():
    """
    Ordre de préférence, chacun vérifié différent du générateur :
      1) Groq llama-3.3-70b-versatile — différent par provider ET par famille
         de modèle de n'importe quel générateur local.
      2) Un second modèle Ollama local (JUDGE_MODEL_NAME) — vérifié vivant
         par un ping avant d'être adopté. Nécessite 'ollama pull mistral:7b'
         (ou un autre modèle) une seule fois si absent.
      3) Si ni l'un ni l'autre n'est disponible : PAS de repli silencieux sur
         le générateur. On le dit explicitement (judge_independent=False,
         propagé dans le trace et le CSV RAGAS) plutôt que de prétendre à
         une indépendance qui n'existe pas.

    Retourne (judge_object_or_None, is_independent: bool)
    """
    if USE_GROQ_JUDGE and GROQ_API_KEY:
        print(f"[LLM JUDGE] Groq {GROQ_JUDGE_MODEL} (indépendant du générateur)")
        return _build_groq_sync(GROQ_JUDGE_MODEL, 300), True

    try:
        candidate = ChatOllama(model=JUDGE_MODEL_NAME, base_url=OLLAMA_URL,
                                temperature=0, num_ctx=OLLAMA_NUM_CTX)
        candidate.invoke([HumanMessage(content="ping")])  # vérifie que le modèle est bien pullé
        print(f"[LLM JUDGE] Ollama {JUDGE_MODEL_NAME} (indépendant du générateur {MODEL_NAME})")
        return candidate, True
    except Exception as e:
        print(f"[LLM JUDGE] '{JUDGE_MODEL_NAME}' indisponible ({e})")
        print(f"            -> lancez 'ollama pull {JUDGE_MODEL_NAME}' pour un juge local indépendant,")
        print(f"               ou configurez GROQ_API_KEY.")

    print("[LLM JUDGE] AUCUN juge indépendant disponible -> repli sur le générateur. "
          "Chaque score sera marqué judge_independent=False.")
    return None, False

llm = _get_llm()
judge_llm, JUDGE_IS_INDEPENDENT_AT_STARTUP = _get_judge_llm()
print(f"======= JUDGE STATUS ========== independent={JUDGE_IS_INDEPENDENT_AT_STARTUP}")

# ══════════════════════════════════════════════════════════════
# INIT LIGHTRAG
# ══════════════════════════════════════════════════════════════
groq_async_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
nvidia_async_client = AsyncOpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY) if NVIDIA_API_KEY else None

async def groq_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    history_messages = history_messages or []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in history_messages:
        if isinstance(m, dict):
            messages.append(m)
    messages.append({"role": "user", "content": prompt})
    r = await groq_async_client.chat.completions.create(
        model=GROQ_GENERATOR_MODEL,
        messages=messages, temperature=0.0, max_tokens=800)
    return (r.choices[0].message.content or "").strip()


async def nvidia_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    history_messages = history_messages or []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in history_messages:
        if isinstance(m, dict):
            messages.append(m)
    messages.append({"role": "user", "content": prompt})
    r = await nvidia_async_client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages, temperature=0.0, max_tokens=800)
    return (r.choices[0].message.content or "").strip()


async def local_ollama_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """LLM utilisé par LightRAG en interne (extraction de mots-clés
    hybrid/local/global — appelé systématiquement, même en only_need_context)."""
    import httpx
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in (history_messages or []):
        if isinstance(m, dict):
            messages.append(m)
    messages.append({"role": "user", "content": prompt})
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL_NAME, "messages": messages, "stream": False,
                  "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0}},
        )
        r.raise_for_status()
        return (r.json().get("message", {}).get("content") or "").strip()


lightrag_llm_func = (
    nvidia_llm_func if (USE_NVIDIA and NVIDIA_API_KEY) else
    groq_llm_func   if (USE_GROQ and GROQ_API_KEY) else
    local_ollama_llm_func
)


async def embedding_func(texts):
    import httpx
    import numpy as np
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for text in texts:
            r = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text})
            results.append(r.json()["embedding"])
    return np.vstack(results)

async def _build_lightrag():
    rag = LightRAG(
        working_dir=str(INDEX_DIR),
        llm_model_func=lightrag_llm_func,
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
# NOTE : Chroma a été entièrement retiré. Il ne servait qu'à alimenter RAGAS
# (jamais à la génération) et RAGAS reçoit maintenant les contextes réels de
# LightRAG (cf. node_hybrid_search / aquery_data).

# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    question                    : str
    reformulated_q              : str
    lightrag_context             : str    # string fusionnée -> prompt du générateur
    lightrag_retrieved_contexts  : list   # passages itemisés -> RAGAS (remplace vector_results/Chroma)
    response                    : str
    critique_score               : float
    critique_reason              : str
    critique_judge_independent   : bool   # False = ce score est une auto-évaluation
    iteration                   : int
    final_response               : str
    trace                       : list

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
    """Retourne (contenu, was_independent_for_this_call)."""
    if judge_llm is not None:
        try:
            r = judge_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            return r.content, True
        except Exception as e:
            print(f"[JUDGE] Erreur juge indépendant : {e} — repli sur le générateur POUR CET APPEL")
    return _llm_call(system, user), False


def _format_context_from_raw(data: dict) -> str:
    """String fusionnée envoyée au générateur — construite depuis la même
    donnée structurée que celle envoyée à RAGAS (garantit l'alignement)."""
    entities = data.get("entities", []) or []
    relationships = data.get("relationships", []) or []
    chunks = data.get("chunks", []) or []
    parts = []
    if entities:
        parts.append("=== Entités du graphe ===")
        for e in entities:
            parts.append(f"• [{e.get('entity_type', 'entity')}: {e.get('entity_name', '')}]\n"
                         f"  {e.get('description', '')}")
    if relationships:
        parts.append("\n=== Relations (GraphRAG) ===")
        for r in relationships:
            parts.append(f"• {r.get('src_id', '')} → {r.get('tgt_id', '')}\n"
                         f"  {r.get('description', '')}")
    if chunks:
        parts.append("\n=== Passages pertinents ===")
        for i, c in enumerate(chunks, 1):
            parts.append(f"[Doc {i}] {(c.get('content') or '')[:800]}")
    return "\n".join(parts) if parts else "Aucun contexte disponible."


def _contexts_list_from_raw(data: dict) -> list:
    """Passages ITEMISÉS pour RAGAS (context_precision/context_recall comparent
    chaque élément individuellement au ground_truth — un unique blob dégraderait
    ces métriques). Une entrée par entité, par relation, par chunk."""
    entities = data.get("entities", []) or []
    relationships = data.get("relationships", []) or []
    chunks = data.get("chunks", []) or []
    items = []
    for e in entities:
        desc = e.get("description", "")
        if desc:
            items.append(f"[{e.get('entity_type', 'entity')}] {e.get('entity_name', '')}: {desc}")
    for r in relationships:
        desc = r.get("description", "")
        if desc:
            items.append(f"{r.get('src_id', '')} -> {r.get('tgt_id', '')}: {desc}")
    for c in chunks:
        content = c.get("content", "")
        if content:
            items.append(content[:1500])
    return items if items else ["No context retrieved."]


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
    Un seul appel LightRAG (aquery_data, retrieval seul, sans génération)
    qui retourne entités/relations/chunks STRUCTURÉS. On en dérive :
      - lightrag_context             : string pour le prompt du générateur
      - lightrag_retrieved_contexts  : liste de passages pour RAGAS
    Les deux proviennent de LA MÊME requête -> RAGAS évalue exactement ce qui
    a été donné au LLM, jamais une approximation Chroma.
    """
    q     = state["reformulated_q"]
    trace = state.get("trace", [])

    async def _query():
        return await rag_instance.aquery_data(
            q,
            param=QueryParam(
                mode="hybrid",
                top_k=TOP_K_LIGHTRAG,
                chunk_top_k=CHUNK_TOP_K,
                enable_rerank=False,
                max_total_tokens=int(os.getenv("MAX_TOTAL_TOKENS", "12000")),
            ),
        )

    try:
        result = asyncio.get_event_loop().run_until_complete(_query())
    except Exception as e:
        result = {}
        trace.append(f"[HYBRID_SEARCH] Erreur : {e}")
        print(f"[HYBRID_SEARCH] Erreur : {e}")

    data = (result or {}).get("data", {}) if isinstance(result, dict) else {}
    ctx_string = _format_context_from_raw(data)
    ctx_list   = _contexts_list_from_raw(data)

    trace.append(f"[HYBRID_SEARCH] context={len(ctx_string)} chars | {len(ctx_list)} passages RAGAS")
    print(f"[HYBRID_SEARCH] context={len(ctx_string)} chars | {len(ctx_list)} passages RAGAS")
    return {**state, "lightrag_context": ctx_string,
            "lightrag_retrieved_contexts": ctx_list, "trace": trace}


RESPONSE_SYSTEM_PROMPT = """You are a research assistant grounded EXCLUSIVELY in the
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

# Certains backends (ex: Groq llama-3.1-8b-instant) rejettent les requêtes
# trop volumineuses (HTTP 413) bien avant la fenêtre de contexte nominale du
# modèle -> troncature défensive du contexte envoyé au générateur.
MAX_GENERATOR_CONTEXT_CHARS = int(os.getenv("MAX_GENERATOR_CONTEXT_CHARS", "6000"))


def node_response(state: AgentState) -> AgentState:
    q         = state["reformulated_q"]
    context   = state.get("lightrag_context", "")[:MAX_GENERATOR_CONTEXT_CHARS]
    iteration = state.get("iteration", 0)
    trace     = state.get("trace", [])

    user = f"""Context:
{context}

Question: {q}

Answer (use ONLY the context above, cite sources):"""

    answer = _llm_call(RESPONSE_SYSTEM_PROMPT, user)
    trace.append(f"[RESPONSE iter={iteration}] {answer[:120]}...")
    print(f"[RESPONSE iter={iteration}] {answer[:150]}")
    return {**state, "response": answer, "trace": trace}


def node_critique(state: AgentState) -> AgentState:
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

    raw, was_independent = _judge_call(system, user)
    try:
        m    = re.search(r"\{.*?\}", raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        score  = max(0.0, min(1.0, float(data.get("score", 0.5))))
        reason = data.get("reason", raw[:150])
    except Exception:
        score, reason = 0.5, raw[:150]

    trace.append(f"[CRITIQUE iter={iteration}] score={score:.2f} independent={was_independent} | {reason[:80]}")
    print(f"[CRITIQUE iter={iteration}] score={score:.2f} | judge_independent={was_independent}")
    return {**state, "critique_score": score, "critique_reason": reason,
            "critique_judge_independent": was_independent, "trace": trace}


def node_self_correct(state: AgentState) -> AgentState:
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
# BUILD + API — architecture inchangée : Query → Hybrid Search → LLM
# Response → Critique → (Finalize | Self Correct → Query)
# ══════════════════════════════════════════════════════════════
def build_agent():
    g = StateGraph(AgentState)

    g.add_node("query",         node_query)
    g.add_node("hybrid_search", node_hybrid_search)
    g.add_node("response",      node_response)
    g.add_node("critique",      node_critique)
    g.add_node("self_correct",  node_self_correct)
    g.add_node("finalize",      node_finalize)

    g.set_entry_point("query")
    g.add_edge("query",         "hybrid_search")
    g.add_edge("hybrid_search", "response")
    g.add_edge("response",      "critique")
    g.add_conditional_edges("critique", route_after_critique,
                            {"finalize": "finalize",
                             "self_correct": "self_correct"})
    g.add_edge("self_correct",  "query")
    g.add_edge("finalize",      END)

    return g.compile()


def run_agent(question: str, use_phoenix: bool = False) -> dict:
    """
    Architecture : QUERY → HYBRID_SEARCH (LightRAG.aquery_data)
                 → LLM RESPONSE → CRITIQUE (Judge indépendant si possible)
                 → SELF_CORRECT (max 3) → FINALIZE
    RAGAS N'EST PAS APPELÉ ICI — voir eval_lightrag_ragas.py, exécuté
    uniquement après que run_agent() a terminé.
    """
    if use_phoenix:
        _setup_phoenix()

    agent = build_agent()
    init: AgentState = {
        "question":                    question,
        "reformulated_q":              question,
        "lightrag_context":            "",
        "lightrag_retrieved_contexts": [],
        "response":                    "",
        "critique_score":              0.0,
        "critique_reason":             "",
        "critique_judge_independent":  False,
        "iteration":                   0,
        "final_response":              "",
        "trace":                       [],
    }

    print(f"\n{'='*60}\nQUESTION : {question}\n{'='*60}")
    result = agent.invoke(init)
    print(f"\n--- RÉPONSE FINALE ---\n{result['final_response']}")
    print(f"Score : {result['critique_score']:.2f} | judge_independent={result['critique_judge_independent']} "
          f"| Iterations : {result['iteration']}")
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
