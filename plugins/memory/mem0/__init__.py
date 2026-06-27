"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search, and automatic deduplication
via the Mem0 Platform API (cloud) or OSS (self-hosted) via Memory.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Fork-local (Sylva / PRD-029):
  * Chronicle searcher (sylva_chronicle) + orchestrated prefetch
    (IntentGate / QueryModeRouter / ContextBudget / Deduplicator).
  * No per-turn passive fact extraction — sync_turn() is intentionally a
    no-op for writes (it only caches orchestration state). Durable writes
    happen exclusively through explicit agent tool calls (mem0_add) and the
    governed consolidation pipeline. Re-enabling infer=True per-turn would
    reopen the ungoverned-autosave loop PRD-029 exists to close.

Configuration
-------------
Secret (lives in $HERMES_HOME/.env or the environment):
  MEM0_API_KEY       — Mem0 Platform API key (required for platform mode)

Behavioral settings (live in $HERMES_HOME/mem0.json, set via `hermes memory
setup`):
  mode               — Backend mode: "platform" (default) or "oss"
  user_id            — Canonical user identifier. When set, it is applied
                       uniformly across every gateway (CLI, Telegram, Slack,
                       Discord, …) so the same human gets one merged memory
                       store. When unset, the gateway-native id (e.g. Telegram
                       numeric id, Discord snowflake) is used instead.
  agent_id           — Agent identifier (default: hermes)
  oss                — OSS backend config: {vector_store, embedder, llm}
                       (nested). Legacy top-level vector_store/embedder is read
                       as a fallback by the chronicle searcher only.

The matching MEM0_MODE / MEM0_USER_ID / MEM0_AGENT_ID environment variables are
still read as a backward-compatible fallback, but mem0.json is the canonical
home for these non-secret settings.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .chronicle import ChronicleSearcher
from .orchestrator import (
    ContextBudget,
    Deduplicator,
    IntentGate,
    QueryModeRouter,
)

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError")

# Sentinel returned when neither MEM0_USER_ID nor a gateway-native id is
# available. Treated as "no operator-configured user_id" by initialize() so
# that legacy mem0.json files written by the setup wizard (which historically
# wrote this exact placeholder) still allow gateway-native ids to flow
# through instead of silently overriding them with the placeholder.
_DEFAULT_USER_ID = "hermes-user"

# Chronicle config fallbacks — match plugins/memory/mem0/chronicle.py defaults.
_DEFAULT_QDRANT_URL = "http://localhost:6333"
_DEFAULT_TEI_URL = "http://localhost:8085"


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad ID, not found) that should NOT trip circuit breaker."""
    etype = type(exc).__name__
    if etype in _CLIENT_ERROR_TYPES:
        return True
    err_str = str(exc).lower()
    return "404" in err_str or "not found" in err_str or "valid uuid" in err_str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "mode": os.environ.get("MEM0_MODE", "platform"),
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "oss": {},
    }
    # Only carry user_id when the operator explicitly configured one (env or
    # mem0.json). An absent key tells initialize() to fall back to the
    # gateway-native id from kwargs instead of overriding it with a placeholder.
    env_user_id = os.environ.get("MEM0_USER_ID")
    if env_user_id:
        config["user_id"] = env_user_id

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    # Back-compat shim (PRD-029 v3 port): older mem0.json files use a FLAT
    # top-level vector_store/embedder/llm layout with no `mode` / `oss` block.
    # v3 selects the backend off `mode` (default "platform"), so a flat file
    # would resolve to platform mode with no api_key -> backend None -> memory
    # silently dead (the PRD-033/036 "container memory went dark" failure class).
    # A v3 config never carries a top-level vector_store; a legacy OSS config
    # always does. So: top-level vector_store + no oss block + no api_key is an
    # unambiguous legacy-OSS file -> promote it to nested OSS mode in-memory so
    # existing installs keep working without a manual on-disk migration.
    if (config.get("vector_store")
            and not config.get("oss")
            and not config.get("api_key")):
        config["mode"] = "oss"
        config["oss"] = {
            k: config[k]
            for k in ("vector_store", "embedder", "llm", "history_db_path")
            if config.get(k)
        }

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

LIST_SCHEMA = {
    "name": "mem0_list",
    "description": (
        "List all stored memories about the user. "
        "Use at conversation start for full overview."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "Page number (default: 1)."},
            "page_size": {"type": "integer", "description": "Results per page (default: 100, max: 200)."},
        },
        "required": [],
    },
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by relevance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "rerank": {"type": "boolean", "description": "Rerank results for relevance (default: true, platform mode only)."},
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "mem0_add",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": "Update an existing memory's text by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to update."},
            "text": {"type": "string", "description": "New text content."},
        },
        "required": ["memory_id", "text"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": "Delete a memory by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to delete."},
        },
        "required": ["memory_id"],
    },
}

CHRONICLE_SEARCH_SCHEMA = {
    "name": "chronicle_search",
    "description": (
        "Search the conversation chronicle — archived past sessions with Scott. "
        "Use when asked to recall or reference previous conversations, events, "
        "or discussions from specific dates or time periods."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for in past conversations."},
            "speaker": {
                "type": "string",
                "enum": ["scott", "sylva", "any"],
                "description": "Filter by speaker (default: any).",
            },
            "date_from": {"type": "string", "description": "Start date filter (YYYY-MM-DD)."},
            "date_to": {"type": "string", "description": "End date filter (YYYY-MM-DD)."},
            "top_k": {"type": "integer", "description": "Max results (default: 5, max: 15)."},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with server-side extraction and semantic search.

    Supports Platform API (cloud) and OSS (self-hosted) modes via MEM0_MODE.
    """

    def __init__(self):
        self._config = None
        self._backend = None
        self._mode = "platform"
        self._api_key = ""
        self._user_id = _DEFAULT_USER_ID
        self._agent_id = "hermes"
        self._channel = "cli"  # gateway channel name (cli/telegram/discord/...)
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._breaker_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._atexit_registered = False
        # Orchestration state (fork-local) — intent gate + deduplicator inputs.
        self._had_tool_calls = False
        self._last_assistant_content = ""
        self._chronicle = None

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        mode = cfg.get("mode", "platform")
        if mode == "oss":
            return bool(cfg.get("oss", {}).get("vector_store"))
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        cfg = _load_config()
        mode = cfg.get("mode", "platform")
        api_key_required = mode != "oss"
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": api_key_required, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from ._setup import post_setup
        post_setup(hermes_home, config)

    def _create_backend(self):
        try:
            if self._mode == "oss":
                from ._backend import OSSBackend
                return OSSBackend(self._config.get("oss", {}))
            from ._backend import PlatformBackend
            return PlatformBackend(self._api_key)
        except Exception as e:
            logger.error("Mem0 backend failed to initialize (%s mode): %s", self._mode, e)
            self._init_error = str(e)
            return None

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _format_error(self, prefix: str, exc: Exception) -> str:
        msg = f"{prefix}: {exc}"
        if self._mode == "oss":
            err_str = str(exc).lower()
            if "connection" in err_str or "refused" in err_str or "timeout" in err_str:
                vs = (self._config or {}).get("oss", {}).get("vector_store", {})
                msg += f" (check that {vs.get('provider', 'vector store')} is running)"
        return msg

    def _record_success(self):
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self):
        with self._breaker_lock:
            self._consecutive_failures += 1
            count = self._consecutive_failures
            if count >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            else:
                count = 0
        if count >= _BREAKER_THRESHOLD:
            hint = ""
            if self._mode == "oss":
                vs = (self._config or {}).get("oss", {}).get("vector_store", {})
                provider = vs.get("provider", "unknown")
                hint = f" Check that your {provider} vector store is running and reachable."
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.%s",
                count, _BREAKER_COOLDOWN_SECS, hint,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._mode = self._config.get("mode", "platform")
        self._api_key = self._config.get("api_key", "")
        self._rerank = self._config.get("rerank", True)
        # Resolution order for user_id:
        #   1. Operator-configured MEM0_USER_ID (env or $HERMES_HOME/mem0.json) —
        #      the canonical principal, applied across every gateway so the same
        #      human gets one merged memory store.
        #   2. Gateway-native id from kwargs (Telegram numeric id, Discord
        #      snowflake, etc.) — preserves per-platform isolation when no
        #      override is configured.
        #   3. Hardcoded fallback _DEFAULT_USER_ID (CLI with no auth).
        # The literal _DEFAULT_USER_ID string is treated as unset so users who
        # ran the setup wizard with the suggested default still get gateway-
        # native ids instead of being silently bucketed together. This closes
        # the PRD-020 foreign-uid fragmentation bug at source (the upstream v3
        # fix that makes MEM0_USER_ID win over the gateway-native id).
        configured = self._config.get("user_id")
        if configured == _DEFAULT_USER_ID:
            configured = None
        self._user_id = configured or kwargs.get("user_id") or _DEFAULT_USER_ID
        self._agent_id = self._config.get("agent_id", "hermes")
        self._channel = kwargs.get("platform") or "cli"
        self._backend = self._create_backend()
        if self._backend and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

        # Chronicle searcher — direct Qdrant + TEI, bypasses the mem0 backend.
        # Read the nested v3 oss.vector_store/oss.embedder config, falling back
        # to the legacy top-level shape for backward compatibility.
        oss = self._config.get("oss") if isinstance(self._config.get("oss"), dict) else {}
        vs_cfg = (oss.get("vector_store") or self._config.get("vector_store") or {}).get("config", {})
        emb_cfg = (oss.get("embedder") or self._config.get("embedder") or {}).get("config", {})
        qdrant_url = vs_cfg.get("url", _DEFAULT_QDRANT_URL)
        tei_url = emb_cfg.get("openai_base_url", _DEFAULT_TEI_URL)
        searcher = ChronicleSearcher(qdrant_url=qdrant_url, tei_url=tei_url)
        # Cache availability at init — don't do network I/O on every get_tool_schemas().
        self._chronicle = searcher if searcher.is_available() else None
        if self._chronicle:
            logger.info("Chronicle searcher initialized (sylva_chronicle)")
        else:
            logger.info("Chronicle searcher unavailable — tool will degrade gracefully")

    def _read_filters(self) -> Dict[str, Any]:
        # Scoped to user_id only — by design — so recall surfaces memories
        # written from any gateway/agent under this principal. Writes attach
        # agent_id (and metadata.channel) so per-agent / per-channel views are
        # still possible at query time when needed; reads default to the wider
        # cross-agent recall.
        return {"user_id": self._user_id}

    def _write_metadata(self) -> Dict[str, Any]:
        # Tag every write with the gateway channel so the dashboard can offer
        # per-channel filtered views without coupling identity to the channel.
        return {"channel": self._channel} if self._channel else {}

    def system_prompt_block(self) -> str:
        mode_label = "platform (cloud API)" if self._mode == "platform" else "OSS (self-hosted)"
        rerank_note = " Rerank is available on search." if self._mode == "platform" else ""
        lines = [
            "# Mem0 Memory",
            f"Active. Mode: {mode_label}. User: {self._user_id}.",
            "Use mem0_search to find memories, mem0_add to store facts, "
            f"mem0_list for a full overview, mem0_update and mem0_delete to manage by ID.{rerank_note}",
        ]
        if self._chronicle:
            lines.append(
                "Use chronicle_search to recall past conversations by topic, "
                "speaker, or date range."
            )
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        # If the thread still hasn't finished, leave the result for the next call.
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return ""
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        # Deduplicate against current user message + last assistant response.
        dedup_context = f"{query} {self._last_assistant_content}"
        lines = [line for line in result.split("\n") if line.strip()]
        filtered = Deduplicator.deduplicate(lines, dedup_context)
        if not filtered:
            return ""
        return "## Mem0 Memory\n" + "\n".join(filtered)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._backend is None or self._is_breaker_open():
            return

        def _run():
            backend = self._backend
            if backend is None:
                return
            # Intent gate — skip retrieval for social/confirmation messages.
            if not IntentGate.should_retrieve(query, self._had_tool_calls):
                logger.debug("Mem0 prefetch skipped by intent gate: %r", query[:60])
                return

            mode = QueryModeRouter.classify(query)
            facts: list[str] = []
            chronicle_results: list[str] = []

            try:
                if mode == "historical_memory" and self._chronicle:
                    # Route to the chronicle collection (direct Qdrant + TEI).
                    try:
                        results = self._chronicle.search(query, top_k=5)
                        chronicle_results = [
                            f"[{r['date']} {r['speaker']}] {r['data']}"
                            for r in results if r.get("data")
                        ]
                    except Exception as e:
                        logger.debug("Chronicle prefetch failed, falling back: %s", e)
                        # Fall through to stable_knowledge on failure.

                # Always search stable knowledge (curated facts).
                mem_results = backend.search(
                    query, filters=self._read_filters(), top_k=10, rerank=self._rerank,
                )
                if mem_results:
                    facts = [r.get("memory", "") for r in mem_results if r.get("memory")]

                # Assemble within budget.
                assembled = ContextBudget.assemble(
                    facts=facts, chronicle=chronicle_results,
                )
                if assembled:
                    with self._prefetch_lock:
                        self._prefetch_result = assembled
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Cache turn state for orchestration but skip per-turn fact extraction.

        SAFETY-CRITICAL (PRD-029): per-turn server-side extraction is disabled
        on purpose. Upstream v3's sync_turn calls ``backend.add(..., infer=True)``
        every turn, which re-opens the ungoverned-autosave loop that floods the
        store with low-signal, unattributed, ungated confabulations. Durable
        memory writes happen ONLY through explicit agent tool calls (mem0_add)
        and the governed consolidation pipeline. Do NOT reintroduce a per-turn
        ``backend.add`` here — that is the single regression PRD-029 exists to
        prevent. The orchestration state below feeds the intent gate and the
        deduplicator only; it performs no writes.
        """
        self._last_assistant_content = (assistant_content or "")[:2000]
        # Length proxy: substantive responses (>200 chars) likely involved tool use.
        self._had_tool_calls = len(assistant_content) > 200 if assistant_content else False

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Always advertise chronicle_search regardless of self._chronicle state.
        # The memory_manager builds its _tool_to_provider routing map by calling
        # get_tool_schemas() at add_provider() time — which runs BEFORE
        # initialize() populates self._chronicle. Gating schema advertisement on
        # self._chronicle there caused chronicle_search to never enter the
        # routing map, producing "Unknown tool: chronicle_search" at dispatch.
        # The dispatch handler below still checks self._chronicle and returns a
        # graceful "Chronicle not available." when the backend isn't ready.
        return [LIST_SCHEMA, SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA,
                CHRONICLE_SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        # Chronicle uses its own backend (direct Qdrant + TEI) — handled ahead of
        # the mem0 backend/breaker path so it stays available even when the mem0
        # backend is down or the circuit breaker is open.
        if tool_name == "chronicle_search":
            if not self._chronicle:
                return json.dumps({"error": "Chronicle not available."})
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            try:
                results = self._chronicle.search(
                    query,
                    speaker=args.get("speaker", "any"),
                    date_from=args.get("date_from", ""),
                    date_to=args.get("date_to", ""),
                    top_k=min(int(args.get("top_k", 5)), 15),
                )
                if not results:
                    return json.dumps({"result": "No matching chronicle entries found."})
                return json.dumps({"results": results, "count": len(results)})
            except Exception as e:
                return json.dumps({"error": f"Chronicle search failed: {e}"})

        if self._backend is None:
            err = getattr(self, "_init_error", "unknown error")
            hint = ""
            if self._mode == "oss":
                vs = (self._config or {}).get("oss", {}).get("vector_store", {})
                provider = vs.get("provider", "vector store")
                hint = f" Check that {provider} is running and reachable."
            return json.dumps({"error": f"Mem0 backend not initialized: {err}.{hint}"})

        if self._is_breaker_open():
            msg = "Mem0 temporarily unavailable (multiple consecutive failures). Will retry automatically."
            if self._mode == "oss":
                vs = (self._config or {}).get("oss", {}).get("vector_store", {})
                msg += f" Check that your {vs.get('provider', 'vector store')} is running."
            return json.dumps({"error": msg})

        if tool_name == "mem0_list":
            try:
                page = max(1, int(args.get("page", 1)))
                page_size = min(max(1, int(args.get("page_size", 100))), 200)
                response = self._backend.get_all(
                    filters=self._read_filters(), page=page, page_size=page_size,
                )
                self._record_success()
                results = response.get("results", [])
                if not results:
                    return json.dumps({"result": "No memories stored yet."})
                items = [{"id": m.get("id"), "memory": m.get("memory", "")}
                         for m in results]
                return json.dumps({
                    "results": items,
                    "count": response.get("count", len(items)),
                    "page": page, "page_size": page_size,
                })
            except Exception as e:
                if not _is_client_error(e):
                    self._record_failure()
                return tool_error(self._format_error("Failed to list memories", e))

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                top_k = max(1, min(int(args.get("top_k", 10)), 50))
                rerank_raw = args.get("rerank", True)
                if isinstance(rerank_raw, str):
                    rerank = rerank_raw.lower() not in ("false", "0", "no")
                else:
                    rerank = bool(rerank_raw)
                results = self._backend.search(query, filters=self._read_filters(), top_k=top_k, rerank=rerank)
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"id": r.get("id"), "memory": r.get("memory", ""),
                          "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                if not _is_client_error(e):
                    self._record_failure()
                return tool_error(self._format_error("Search failed", e))

        elif tool_name == "mem0_add":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            try:
                result = self._backend.add(
                    [{"role": "user", "content": content}],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=False,
                    metadata=self._write_metadata(),
                )
                self._record_success()
                event_id = result.get("event_id") if isinstance(result, dict) else None
                msg = "Fact stored." if self._mode == "oss" else "Fact queued for storage."
                return json.dumps({"result": msg, "event_id": event_id})
            except Exception as e:
                self._record_failure()
                return tool_error(self._format_error("Failed to store", e))

        elif tool_name == "mem0_update":
            memory_id = args.get("memory_id", "")
            text = args.get("text", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            if not text:
                return tool_error("Missing required parameter: text")
            try:
                result = self._backend.update(memory_id, text)
                self._record_success()
                return json.dumps(result)
            except Exception as e:
                if _is_client_error(e):
                    return tool_error(f"Memory not found: {memory_id}")
                self._record_failure()
                return tool_error(self._format_error("Update failed", e))

        elif tool_name == "mem0_delete":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            try:
                result = self._backend.delete(memory_id)
                self._record_success()
                return json.dumps(result)
            except Exception as e:
                if _is_client_error(e):
                    return tool_error(f"Memory not found: {memory_id}")
                self._record_failure()
                return tool_error(self._format_error("Delete failed", e))

        return tool_error(f"Unknown tool: {tool_name}")

    def _shutdown_backend(self):
        try:
            if self._backend:
                self._backend.close()
                self._backend = None
        except Exception:
            pass

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        self._shutdown_backend()


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
