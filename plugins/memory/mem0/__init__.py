"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search, and automatic deduplication
via the Mem0 Platform API (cloud) or OSS (self-hosted) via Memory.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Fork-local (Sylva / PRD-029):
  * Chronicle searcher (sylva_chronicle) + orchestrated prefetch
    (IntentGate / QueryModeRouter / ContextBudget / Deduplicator).
  * No per-turn passive fact extraction — sync_turn() is intentionally a
    no-op for writes (it only caches orchestration state). Re-enabling
    infer=True per-turn would reopen the ungoverned-autosave loop PRD-029
    exists to close.
  * Decommission (2026-06-28): the mem0_* tools (list/search/add/update/
    delete) are RETIRED and the every-turn "stable_knowledge" prefetch route
    is dropped. Curated identity is the canon self-brief (plugins/memory/
    canon, seed_canon.py governed pipeline — NOT mem0_add); episodic recall
    is chronicle_search. This provider is now chronicle-centric.

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
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

import requests

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
_CHRONICLE_COLLECTION = "sylva_chronicle"

# ── PRD-037 FR-1: session-end episodic ingest governance bounds ──────────────
# Stable namespace so re-running the same session produces identical point ids
# (idempotent upsert; AC-003). Mirrors the migration script's deterministic-id
# pattern (scripts/migrate_sylva_memories_to_chronicle.py).
_INGEST_NS = uuid.UUID("6f29a1c4-0d2e-4e7a-9b13-abcdef012345")
# Skip empty/trivial sessions: need at least this many user+assistant turns and
# this many chars of substantive transcript before a session is worth ingesting.
_INGEST_MIN_TURNS = 2
_INGEST_MIN_SESSION_CHARS = 200
# Bound the write: at most N episodic entries per session, each capped in size.
_INGEST_MAX_ENTRIES = 8
_INGEST_MAX_ENTRY_CHARS = 1000
_INGEST_MIN_ENTRY_CHARS = 25
# Cap how much transcript we feed the summarizer (last N turns) + its output.
_INGEST_MAX_SUMMARY_TURNS = 40
_INGEST_SUMMARY_MAX_TOKENS = 600
_INGEST_SUMMARY_TIMEOUT = 60  # hard cap on the aux summarize call so teardown can't stall
_INGEST_UPSERT_TIMEOUT = 60


def _redact(text: str) -> str:
    """Scrub secret-like patterns from ``text`` before it reaches the aux LLM or
    the chronicle. force=True runs even when display redaction is globally off.
    Best-effort: on import/scan failure, return the input unchanged (we prefer a
    captured-but-unredacted memory over silently losing the session — the aux
    endpoint here is the local container model, not external egress)."""
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text or "", force=True)
    except Exception:
        logger.debug("redact unavailable; passing episodic text through unredacted")
        return text or ""


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
        # Current session id — captured at initialize() and refreshed on
        # on_session_switch so on_session_end (PRD-037 FR-1) attributes the
        # episodic record to the correct session (source=session:<id>).
        self._session_id = ""

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
        self._session_id = session_id or ""
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
        # PRD-029 decommission (2026-06-28): the mem0_* tools are retired. Curated
        # identity is injected separately via the canon self-brief; this provider
        # now exposes only chronicle_search for on-demand episodic recall.
        if not self._chronicle:
            return ""
        return (
            "# Memory Recall\n"
            "Use chronicle_search to recall past conversations by topic, "
            "speaker, or date range."
        )

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
        # PRD-029 decommission (2026-06-28): prefetch is now CHRONICLE-ONLY and no
        # longer depends on the mem0 backend. The old "stable_knowledge" route
        # searched the retired sylva_memories store on EVERY non-trivial turn;
        # adversarial review confirmed that repointing it at the chronicle would
        # inject raw journal fragments into nearly every turn. Curated identity
        # comes from the canon self-brief at session start, so the stable route is
        # intentionally a no-op. The historical_memory route still warms the
        # chronicle; on-demand recall is the chronicle_search tool.
        if self._chronicle is None:
            return

        def _run():
            # Intent gate — skip retrieval for social/confirmation messages.
            if not IntentGate.should_retrieve(query, self._had_tool_calls):
                logger.debug("Mem0 prefetch skipped by intent gate: %r", query[:60])
                return

            mode = QueryModeRouter.classify(query)
            chronicle_results: list[str] = []

            try:
                if mode == "historical_memory":
                    # Route to the chronicle collection (direct Qdrant + TEI).
                    try:
                        results = self._chronicle.search(query, top_k=5)
                        chronicle_results = [
                            f"[{r['date']} {r['speaker']}] {r['data']}"
                            for r in results if r.get("data")
                        ]
                    except Exception as e:
                        logger.debug("Chronicle prefetch failed: %s", e)
                # stable_knowledge route: intentional no-op (see method docstring).

                # Assemble within budget. facts is always empty post-decommission.
                assembled = ContextBudget.assemble(
                    facts=[], chronicle=chronicle_results,
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
        identity writes happen ONLY through the governed canon consolidation
        pipeline (seed_canon.py); the mem0_* write tools were retired in the
        2026-06-28 decommission. Do NOT reintroduce a per-turn ``backend.add``
        here — that is the single regression PRD-029 exists to prevent. The
        orchestration state below feeds the intent gate and the deduplicator
        only; it performs no writes.
        """
        self._last_assistant_content = (assistant_content or "")[:2000]
        # Length proxy: substantive responses (>200 chars) likely involved tool use.
        self._had_tool_calls = len(assistant_content) > 200 if assistant_content else False

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Track the live session id so on_session_end attributes the episodic
        record to the right session (source=session:<id>). The id rotates on
        /resume, /branch, /reset, /new, and compression — this keeps FR-1's
        attribution correct without re-running initialize()."""
        if new_session_id:
            self._session_id = new_session_id

    def on_session_end(self, messages: List[Dict[str, Any]], *, final: bool = True) -> None:
        """Governed session-end episodic ingest (PRD-037 FR-1, CENTERPIECE).

        Summarize the just-ended conversation into 1–N concise episodic entries
        and upsert them into ``sylva_chronicle`` so future sessions can recall
        what happened. This is the *only* live writer to the chronicle — it
        restores the working-memory write path PRD-029 left severed.

        ``final`` distinguishes a TRUE logical-session boundary from a mid-
        conversation rotation (see ``MemoryManager.on_session_end``):

          * ``final=True``  — CLI exit / gateway-or-TUI session close / ``/new``
            / ``/reset`` / ``/clear`` (``new_session``): the conversation is
            ending, so we ingest one episodic record. This is the path that
            guarantees the next session can recall this one.
          * ``final=False`` — automatic context compaction / ``/compress`` /
            ``/compact`` (``commit_memory_session`` from the compressor): the
            SAME conversation continues, just compacted. We DO NOT ingest here —
            (a) it would fire repeatedly mid-session on the turn's hot path, and
            (b) the summarizer is non-deterministic, so each compaction would
            write overlapping near-duplicate entries (the low-signal noise
            PRD-029 removed). The compacted-away turns are NOT lost — they stay
            in ``state.db`` (active=0, FTS-searchable via ``session_search``),
            and the durable episodic summary is written when the session truly
            ends.

        Governance (security-sensitive subsystem — do NOT weaken):
          * **Session-boundary only**, never per-turn — ``sync_turn`` stays a
            write no-op (the PRD-029 confabulation-loop invariant). ``run_agent``
            also suppresses this call entirely when ``_memory_passive_enabled``
            is False (cron), so cron scaffolding is never ingested (AC-002).
          * **Chronicle (episodic) only.** Never canon / SOUL.md / identity.
          * **Redacted.** Transcript is secret-scrubbed before it reaches the
            aux summarizer AND before any entry is persisted/embedded.
          * **Bounded + idempotent.** Content-hash dedup + deterministic ids;
            skip empty/trivial sessions; cap entry count + size (AC-003).
          * **Audited.** Each ingest records a PRD-028 ledger entry.

        Never raises — a memory-ingest failure must not break session teardown.
        """
        if not final:
            # Mid-conversation rotation (compaction) — defer to the real
            # boundary. See the docstring for why this is correct + lossless.
            logger.debug("mem0 on_session_end(final=False): skipping ingest (compaction).")
            return
        try:
            self._ingest_session_episodic(messages)
        except Exception as e:
            logger.warning(
                "Mem0 on_session_end episodic ingest failed: %s", e, exc_info=True
            )

    # -- FR-1 episodic ingest internals -------------------------------------

    @staticmethod
    def _extract_turns(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Flatten the history to plain user/assistant text turns, dropping
        system/tool turns, injected compaction markers, and multimodal/non-text
        parts — the same shape scripts/session-handoff.py feeds its summarizer.

        The compressor injects two marker prefixes into the live transcript —
        ``[CONTEXT COMPACTION — REFERENCE ONLY]`` and ``[CONTEXT SUMMARY]``
        (agent/conversation_compression.py) — plus a todo snapshot. We drop
        anything under the ``[CONTEXT`` prefix so a true-end ingest of an
        already-compacted transcript doesn't re-summarize prior summaries."""
        turns: List[Dict[str, str]] = []
        for msg in messages or []:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            text = content.strip()
            if not text or text.startswith("[CONTEXT"):
                continue
            turns.append({"role": role, "text": text})
        return turns

    def _summarize_session(self, turns: List[Dict[str, str]]) -> str:
        """Summarize the conversation into a concise episodic note via the
        configured auxiliary LLM (the upstream ``compression`` aux client —
        same resolution the context compressor uses). Returns "" if no aux
        provider is available; we never fall back to storing the raw transcript
        (raw fragments are the low-signal noise PRD-029 fought to remove)."""
        try:
            from agent.auxiliary_client import (
                get_text_auxiliary_client,
                is_local_qwen3_endpoint,
            )
        except Exception as e:
            logger.debug("aux client import failed; skipping ingest summary: %s", e)
            return ""

        client, aux_model = get_text_auxiliary_client("compression")
        if client is None or not aux_model:
            logger.info(
                "No auxiliary LLM for session-end summary — skipping episodic ingest."
            )
            return ""

        recent = turns[-_INGEST_MAX_SUMMARY_TURNS:]
        transcript = "\n\n".join(
            f"[{t['role'].upper()}]: {t['text']}" for t in recent
        )
        # Secret-scrub the transcript BEFORE it reaches the aux summarizer.
        # The aux ``compression`` provider can resolve to a remote endpoint, and
        # the summary is persisted + embedded — so credentials in the raw turns
        # must never leave the host or land in the chronicle. force=True scans
        # even when display redaction is globally off (mirrors ask_claude /
        # PRD-024). Best-effort: if the redactor can't load, we proceed on the
        # raw text rather than silently dropping the session's memory.
        transcript = _redact(transcript)
        prompt = (
            "You are Sylva, summarizing a conversation you (the AI assistant) just "
            "had with Scott (the user), for your own future recall.\n\n"
            "Write a concise episodic memory of this session as 3–8 short bullet "
            "points. Each bullet: one concrete thing that happened, was decided, "
            "or is pending — be specific (file paths, tool names, config values, "
            "decisions). No preamble, no headers, no markdown emphasis — just '- ' "
            "bullets. Record what occurred; do NOT invent facts or restate your "
            "identity. Skip trivia and pleasantries.\n\n"
            f"--- TRANSCRIPT (last {len(recent)} turns) ---\n"
            f"{transcript}\n"
            "--- END TRANSCRIPT ---\n\n"
            "Episodic bullets:"
        )

        kwargs: Dict[str, Any] = {
            "model": aux_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": _INGEST_SUMMARY_MAX_TOKENS,
            "temperature": 0.3,
            "stream": False,
            # Hard timeout so a wedged aux endpoint can't stall session teardown.
            "timeout": _INGEST_SUMMARY_TIMEOUT,
        }
        # Local Qwen3 thinking-mode returns empty content / huge latency; disable
        # it the documented way (llama.cpp honors this Jinja kwarg).
        try:
            base_url = str(getattr(client, "base_url", "") or "")
            if is_local_qwen3_endpoint(base_url, aux_model):
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        except Exception:
            pass

        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _summary_to_entries(summary: str) -> List[str]:
        """Split the summary into individual, deduplicated episodic entries —
        bounded count + size, markdown markers stripped, trivial lines dropped."""
        entries: List[str] = []
        seen: set[str] = set()
        for raw in summary.split("\n"):
            line = raw.strip().lstrip("-*•").strip()
            # Drop markdown headers and short/empty fragments (e.g. "Format:").
            if line.startswith("#"):
                continue
            if len(line) < _INGEST_MIN_ENTRY_CHARS:
                continue
            line = line[:_INGEST_MAX_ENTRY_CHARS]
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            entries.append(line)
            if len(entries) >= _INGEST_MAX_ENTRIES:
                break
        return entries

    def _existing_chronicle_hashes(self) -> set[str]:
        """Pull every chronicle point's content hash for cross-session de-dupe
        (idempotent re-runs + no double-insert). Mirrors the migration script."""
        hashes: set[str] = set()
        offset = None
        qdrant = self._chronicle._qdrant_url
        while True:
            body: Dict[str, Any] = {
                "limit": 1000,
                "with_payload": ["data", "hash"],
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            r = requests.post(
                f"{qdrant}/collections/{_CHRONICLE_COLLECTION}/points/scroll",
                json=body, timeout=30,
            )
            r.raise_for_status()
            res = r.json()["result"]
            for p in res.get("points", []):
                pl = p.get("payload", {})
                h = pl.get("hash") or hashlib.md5((pl.get("data") or "").encode()).hexdigest()
                hashes.add(h)
            offset = res.get("next_page_offset")
            if offset is None:
                break
        return hashes

    def _ingest_session_episodic(self, messages: List[Dict[str, Any]]) -> None:
        if self._chronicle is None:
            logger.info("Chronicle unavailable — skipping session-end episodic ingest.")
            return

        turns = self._extract_turns(messages)
        total_chars = sum(len(t["text"]) for t in turns)
        if len(turns) < _INGEST_MIN_TURNS or total_chars < _INGEST_MIN_SESSION_CHARS:
            logger.info(
                "Session too trivial for episodic ingest (turns=%d chars=%d) — skipping.",
                len(turns), total_chars,
            )
            return

        summary = self._summarize_session(turns)
        if not summary:
            return
        entries = self._summary_to_entries(summary)
        if not entries:
            logger.info("Session summary produced no episodic entries — skipping.")
            return

        seen_hashes = self._existing_chronicle_hashes()
        session_id = self._session_id or "unknown"
        date = datetime.now().strftime("%Y-%m-%d")
        source = f"session:{session_id}"

        points: List[Dict[str, Any]] = []
        for entry in entries:
            # Belt-and-suspenders: redact again at persist time in case the
            # summary echoed a secret the model copied from the transcript.
            entry = _redact(entry)
            h = hashlib.md5(entry.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            det_id = str(uuid.uuid5(_INGEST_NS, f"{source}:{h}"))
            payload = {
                "data": entry,
                "speaker": "sylva",
                "date": date,
                "source": source,
                "category": "journal",
                "user_id": self._user_id,
                "role": "user",
                "hash": h,
            }
            try:
                vector = self._chronicle.embed(entry)
            except Exception as e:
                logger.debug("embed failed for episodic entry, skipping: %s", e)
                continue
            points.append({"id": det_id, "vector": vector, "payload": payload})

        if not points:
            logger.info("Episodic ingest: all entries already in chronicle (idempotent no-op).")
            return

        qdrant = self._chronicle._qdrant_url
        r = requests.put(
            f"{qdrant}/collections/{_CHRONICLE_COLLECTION}/points?wait=true",
            json={"points": points}, timeout=_INGEST_UPSERT_TIMEOUT,
        )
        r.raise_for_status()
        logger.info(
            "Episodic ingest: wrote %d entries to sylva_chronicle (%s).",
            len(points), source,
        )
        self._audit_ingest(source, len(points))

    @staticmethod
    def _audit_ingest(source: str, count: int) -> None:
        """Record the episodic write to the PRD-028 audit ledger (best-effort)."""
        try:
            from autonomy import audit
            audit.record(
                tier="T2",
                surface="memory",
                action="chronicle_episodic_ingest",
                rationale=f"{source} +{count} episodic entries",
                outcome="ok",
            )
        except Exception as e:
            logger.debug("audit.record for episodic ingest failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # PRD-029 decommission (2026-06-28): the mem0_* tools (list/search/add/
        # update/delete) are RETIRED. They bound to the old sylva_memories store,
        # and the write tools (add/update/delete) reintroduced a second durable
        # fact store outside the curated canon — the exact anti-pattern PRD-029
        # set out to kill (confirmed by adversarial review). Curated identity now
        # comes from the canon self-brief at session start; on-demand recall is
        # chronicle_search. The LIST_/SEARCH_/ADD_/UPDATE_/DELETE_SCHEMA constants
        # are retained (unadvertised) so re-enable is a one-line change if ever
        # needed.
        #
        # chronicle_search is advertised regardless of self._chronicle state: the
        # memory_manager builds its _tool_to_provider routing map at
        # add_provider() time, BEFORE initialize() populates self._chronicle.
        # Gating here caused "Unknown tool: chronicle_search" at dispatch. The
        # handler below checks self._chronicle and degrades gracefully.
        return [CHRONICLE_SEARCH_SCHEMA]

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

        # PRD-029 decommission (2026-06-28): the mem0_* tool handlers (list/
        # search/add/update/delete) are removed — they bound to the retired
        # sylva_memories store. chronicle_search (handled above) is the sole
        # recall tool. Anything else is unknown.
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
