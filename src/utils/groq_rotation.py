# src/utils/groq_rotation.py
"""
Rotation automatique de cle Groq sur plusieurs comptes (GROQ_API_KEY,
GROQ_API_KEY_2, GROQ_API_KEY_3), pensee pour l'evaluation Plan C sur 30
questions x 3 systemes (recommandation prof du 30/07 : checkpointer +
etaler/multiplier les comptes pour rester sous le quota Groq gratuit).

Principe :
- chaque appel essaie la cle "active" (persistee sur disque -> survit a un
  redemarrage du notebook/kernel) ;
- des qu'un appel echoue avec une erreur de quota/rate-limit (429), la cle
  est marquee "epuisee" avec un horodatage, et on bascule sur la cle
  suivante non epuisee ;
- une cle marquee epuisee redevient utilisable automatiquement apres
  GROQ_KEY_RESET_HOURS (24h par defaut : le quota gratuit Groq est journalier) ;
- si TOUTES les cles configurees sont epuisees (et qu'aucune n'a depasse le
  delai de reinitialisation), AllGroqKeysExhaustedError est levee -> a
  capturer par l'appelant pour arreter proprement une boucle d'evaluation
  et laisser le checkpoint deja ecrit permettre une reprise ulterieure.
"""
import os
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

from groq import Groq, AsyncGroq, RateLimitError, APIStatusError

STATE_PATH = Path(os.getenv("GROQ_ROTATION_STATE_PATH", "Eval_agentic/groq_key_state.json"))
RESET_AFTER_HOURS = float(os.getenv("GROQ_KEY_RESET_HOURS", "24"))


class AllGroqKeysExhaustedError(RuntimeError):
    """Levee quand toutes les cles Groq configurees ont atteint leur quota
    et qu'aucune n'a encore depasse le delai de reinitialisation."""


def _load_keys() -> list[str]:
    keys = []
    for name in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        v = os.getenv(name, "").strip()
        if v:
            keys.append(v)
    return keys


_KEYS = _load_keys()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"active_index": 0, "exhausted_at": {}}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _is_expired(iso_ts: str | None) -> bool:
    if not iso_ts:
        return True
    try:
        ts = datetime.fromisoformat(iso_ts)
    except Exception:
        return True
    return datetime.now(timezone.utc) - ts > timedelta(hours=RESET_AFTER_HOURS)


def _pick_active_index(state: dict) -> int:
    n = len(_KEYS)
    if n == 0:
        raise AllGroqKeysExhaustedError(
            "Aucune cle GROQ_API_KEY / GROQ_API_KEY_2 / GROQ_API_KEY_3 configuree dans .env"
        )
    exhausted = state.setdefault("exhausted_at", {})
    # Une cle marquee epuisee il y a > RESET_AFTER_HOURS est consideree
    # reinitialisee (quota journalier Groq) -> on la reautorise.
    for idx_str in list(exhausted.keys()):
        if exhausted[idx_str] and _is_expired(exhausted[idx_str]):
            exhausted[idx_str] = None

    start = state.get("active_index", 0) % n
    for offset in range(n):
        idx = (start + offset) % n
        if not exhausted.get(str(idx)):
            if idx != state.get("active_index"):
                state["active_index"] = idx
                _save_state(state)
            return idx

    raise AllGroqKeysExhaustedError(
        f"Les {n} cle(s) Groq configuree(s) sont toutes marquees epuisees "
        f"(aucune reinitialisation detectee depuis moins de {RESET_AFTER_HOURS}h)."
    )


def get_active_key() -> tuple[str, int]:
    """Retourne (cle, index) de la cle actuellement active."""
    state = _load_state()
    idx = _pick_active_index(state)
    return _KEYS[idx], idx


def mark_exhausted(idx: int) -> None:
    """Marque la cle #idx comme epuisee (persiste sur disque) et journalise
    la rotation."""
    state = _load_state()
    exhausted = state.setdefault("exhausted_at", {})
    exhausted[str(idx)] = _now_iso()
    _save_state(state)
    n = len(_KEYS)
    print(f"[GROQ ROTATION] Cle #{idx + 1}/{n} epuisee (quota atteint) -> rotation vers la cle suivante.")


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 429:
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "quota" in msg or "too many requests" in msg


def groq_chat_completion(model: str, messages: list, temperature: float = 0.0, max_tokens: int = 800):
    """Appel Groq synchrone (client Groq officiel) avec rotation automatique
    sur les cles configurees. Leve AllGroqKeysExhaustedError si toutes les
    cles sont epuisees (a capturer par l'appelant)."""
    n = len(_KEYS)
    last_exc: Exception | None = None
    for _ in range(n):
        key, idx = get_active_key()
        client = Groq(api_key=key)
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            if _is_rate_limit_error(e):
                last_exc = e
                mark_exhausted(idx)
                continue
            raise
    raise AllGroqKeysExhaustedError(
        f"Les {n} cle(s) Groq configuree(s) ont toutes atteint leur limite de quota. "
        f"Derniere erreur : {last_exc}"
    )


async def agroq_chat_completion(model: str, messages: list, temperature: float = 0.0, max_tokens: int = 800):
    """Version asynchrone (utilisee par le llm_model_func interne de LightRAG,
    ex. extraction de mots-cles lors du retrieval hybride), meme logique de
    rotation que groq_chat_completion."""
    n = len(_KEYS)
    last_exc: Exception | None = None
    for _ in range(n):
        key, idx = get_active_key()
        client = AsyncGroq(api_key=key)
        try:
            return await client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            if _is_rate_limit_error(e):
                last_exc = e
                mark_exhausted(idx)
                continue
            raise
    raise AllGroqKeysExhaustedError(
        f"Les {n} cle(s) Groq configuree(s) ont toutes atteint leur limite de quota. "
        f"Derniere erreur : {last_exc}"
    )


def rotation_status() -> str:
    """Resume lisible de l'etat de rotation (nombre de cles, laquelle est
    active, lesquelles sont marquees epuisees) -> utile pour un print de
    diagnostic en debut de notebook."""
    n = len(_KEYS)
    if n == 0:
        return "Aucune cle Groq configuree (GROQ_API_KEY absente de .env)."
    state = _load_state()
    try:
        idx = _pick_active_index(state)
    except AllGroqKeysExhaustedError as e:
        return f"{n} cle(s) configuree(s) -> TOUTES EPUISEES : {e}"
    exhausted = state.get("exhausted_at", {})
    n_exhausted = sum(1 for v in exhausted.values() if v and not _is_expired(v))
    return (f"{n} cle(s) Groq configuree(s) | active=#{idx + 1} | "
            f"{n_exhausted} marquee(s) epuisee(s) actuellement")
