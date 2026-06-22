"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search with reranking, and
automatic deduplication via the Mem0 Platform API or self-hosted instance.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Config via environment variables:
  MEM0_API_KEY       — Mem0 API key (required for cloud, optional for self-hosted)
  MEM0_HOST          — Self-hosted Mem0 URL (default: https://api.mem0.ai)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0.json.
"""

from __future__ import annotations

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
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "host": os.environ.get("MEM0_HOST", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": True,
        "keyword_search": False,
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
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

    Supports both Mem0 Cloud (api.mem0.ai) and self-hosted instances
    via the ``host`` config key or ``MEM0_HOST`` env var.
    """

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._host = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        # Orchestration state
        self._had_tool_calls = False
        self._last_assistant_content = ""
        self._chronicle = None

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        # OSS mode: available if mem0.json has vector_store config
        if cfg.get("vector_store"):
            return True
        # Upstream self-hosted (host) or Mem0 Platform (api_key) mode
        return bool(cfg.get("host")) or bool(cfg.get("api_key"))

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
        return [
            {"key": "api_key", "description": "Mem0 API key (cloud or self-hosted)", "secret": True, "required": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 URL (e.g. http://localhost:24220)", "default": "", "env_var": "MEM0_HOST"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization.

        Uses OSS Memory class if mem0.json contains vector_store config
        (self-hosted Qdrant + embedder + LLM). Falls back to MemoryClient
        for Mem0 Platform API if api_key is set.
        """
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                if self._config and self._config.get("vector_store"):
                    # OSS self-hosted mode (direct Qdrant + embedder + LLM)
                    from mem0 import Memory
                    oss_cfg = {
                        k: self._config[k]
                        for k in ("vector_store", "embedder", "llm", "history_db_path")
                        if k in self._config
                    }
                    self._client = Memory.from_config({"version": "v1.1", **oss_cfg})
                else:
                    # Mem0 Platform API / upstream self-hosted (host) mode
                    from mem0 import MemoryClient
                    kwargs = {}
                    if self._host:
                        kwargs["host"] = self._host
                    if self._api_key:
                        kwargs["api_key"] = self._api_key
                    elif not self._host:
                        raise ValueError("Mem0: either api_key or host is required")
                    self._client = MemoryClient(**kwargs)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            # Cooldown expired — reset and allow a retry
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        self._host = self._config.get("host", "")
        # Single-persona pin (PRD-020 post-merge fix): an explicit MEM0_USER_ID
        # makes ALL surfaces (CLI, Discord, etc.) share one unified memory
        # namespace. Without it set, prefer the gateway-provided per-user id
        # (upstream multi-user scoping), then the config default. The merge's
        # newer gateway began passing the platform user id (e.g. Discord
        # snowflake), which fragmented Sylva's memory per-platform — this pin
        # restores the single unified `sylva` namespace.
        self._user_id = (
            os.environ.get("MEM0_USER_ID")
            or kwargs.get("user_id")
            or self._config.get("user_id", "hermes-user")
        )
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)
        # Chronicle searcher — direct Qdrant + TEI, bypasses mem0 client
        qdrant_url = (self._config.get("vector_store", {})
                      .get("config", {}).get("url", "http://localhost:6333"))
        tei_url = (self._config.get("embedder", {})
                   .get("config", {}).get("openai_base_url", "http://localhost:8085"))
        searcher = ChronicleSearcher(
            qdrant_url=qdrant_url, tei_url=tei_url,
        )
        # Cache availability at init — don't do network I/O on every get_tool_schemas()
        self._chronicle = searcher if searcher.is_available() else None
        if self._chronicle:
            logger.info("Chronicle searcher initialized (sylva_chronicle)")
        else:
            logger.info("Chronicle searcher unavailable — tool will not be registered")

    @property
    def _is_oss(self) -> bool:
        return bool(self._config and self._config.get("vector_store"))

    def _read_kwargs(self) -> Dict[str, Any]:
        """Kwargs for search/get_all — scoped to user only for cross-session recall.

        OSS Memory uses user_id= kwarg; Platform uses filters= dict.
        """
        if self._is_oss:
            return {"user_id": self._user_id}
        return {"filters": {"user_id": self._user_id}}

    def _write_kwargs(self) -> Dict[str, Any]:
        """Kwargs for add — scoped to user + agent for attribution."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    def _search_limit_key(self) -> str:
        """OSS uses 'limit', Platform uses 'top_k'."""
        return "limit" if self._is_oss else "top_k"

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2 wraps results in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        target = self._host or "cloud"
        lines = [
            f"# Mem0 Memory ({target})",
            f"Active. User: {self._user_id}.",
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview.",
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
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        # Deduplicate against current user message + last assistant response
        dedup_context = f"{query} {self._last_assistant_content}"
        lines = [line for line in result.split("\n") if line.strip()]
        filtered = Deduplicator.deduplicate(lines, dedup_context)
        if not filtered:
            return ""
        return "## Mem0 Memory\n" + "\n".join(filtered)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            # Intent gate — skip retrieval for social/confirmation messages
            if not IntentGate.should_retrieve(query, self._had_tool_calls):
                logger.debug("Mem0 prefetch skipped by intent gate: %r", query[:60])
                return

            mode = QueryModeRouter.classify(query)
            facts: list[str] = []
            chronicle_results: list[str] = []

            try:
                if mode == "historical_memory" and self._chronicle:
                    # Route to chronicle collection
                    try:
                        results = self._chronicle.search(query, top_k=5)
                        chronicle_results = [
                            f"[{r['date']} {r['speaker']}] {r['data']}"
                            for r in results if r.get("data")
                        ]
                    except Exception as e:
                        logger.debug("Chronicle prefetch failed, falling back: %s", e)
                        # Fall through to stable_knowledge on failure

                # Always search stable knowledge (curated facts)
                client = self._get_client()
                mem_results = self._unwrap_results(client.search(
                    query=query,
                    **self._read_kwargs(),
                    rerank=self._rerank,
                    **{self._search_limit_key(): 10},
                ))
                if mem_results:
                    facts = [r.get("memory", "") for r in mem_results if r.get("memory")]

                # Assemble within budget
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

        Per-turn extraction via Qwen3 produced low-signal fragments without
        attribution or quality gating.  Durable memory writes now happen only
        through the nightly reflection cron (mem0_conclude) and explicit agent
        tool calls.  The orchestration state (last assistant content, tool-call
        proxy) is still maintained here for the intent gate and deduplicator.
        """
        # Cache state for orchestration (intent gate + dedup)
        self._last_assistant_content = (assistant_content or "")[:2000]
        # Length proxy: substantive responses (>200 chars) likely involved tool use
        self._had_tool_calls = len(assistant_content) > 200 if assistant_content else False
        # NOTE: per-turn client.add() extraction disabled — nightly reflection
        # is the sole write path for durable memories.  To re-enable, uncomment
        # the _sync block below and restart the gateway.
        #
        # if self._is_breaker_open():
        #     return
        # def _sync():
        #     try:
        #         client = self._get_client()
        #         messages = [
        #             {"role": "user", "content": user_content},
        #             {"role": "assistant", "content": assistant_content},
        #         ]
        #         client.add(messages, **self._write_kwargs())
        #         self._record_success()
        #     except Exception as e:
        #         self._record_failure()
        #         logger.warning("Mem0 sync failed: %s", e)
        # if self._sync_thread and self._sync_thread.is_alive():
        #     self._sync_thread.join(timeout=5.0)
        # self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        # self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Always advertise chronicle_search regardless of self._chronicle state.
        # The memory_manager builds its _tool_to_provider routing map by calling
        # get_tool_schemas() at add_provider() time — which runs BEFORE
        # initialize_all() populates self._chronicle. Gating schema advertisement
        # on self._chronicle there caused chronicle_search to never enter the
        # routing map, producing "Unknown tool: chronicle_search" at dispatch.
        # The dispatch handler at the top of handle_tool_call() still checks
        # self._chronicle and returns "Chronicle not available." when the
        # backend isn't ready, so graceful degradation is preserved.
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA, CHRONICLE_SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        # Chronicle uses its own backend — not gated by mem0 breaker
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

        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._unwrap_results(client.get_all(**self._read_kwargs()))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._unwrap_results(client.search(
                    query=query,
                    **self._read_kwargs(),
                    rerank=rerank,
                    **{self._search_limit_key(): top_k},
                ))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_kwargs(),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
