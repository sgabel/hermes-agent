"""Tests for Mem0 v3 API — new tool names, paginated responses, update/delete tools."""

import json
import pytest

import plugins.memory.mem0 as mem0_module
from plugins.memory.mem0 import Mem0MemoryProvider


class FakeBackend:
    """Fake Mem0Backend for provider-level tests."""

    def __init__(self, search_results=None, all_results=None):
        self._search_results = search_results or []
        self._all_results = all_results or {"results": [], "count": 0}
        self.captured = []

    def search(self, query, *, filters, top_k=10, rerank=True):
        self.captured.append(("search", query, {"filters": filters, "top_k": top_k, "rerank": rerank}))
        return self._search_results

    def get_all(self, *, filters, page=1, page_size=100):
        self.captured.append(("get_all", {"filters": filters, "page": page, "page_size": page_size}))
        return self._all_results

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.captured.append((
            "add",
            messages,
            {"user_id": user_id, "agent_id": agent_id, "infer": infer, "metadata": metadata},
        ))
        return {"status": "PENDING", "event_id": "evt-test-123"}

    def update(self, memory_id, text):
        self.captured.append(("update", memory_id, text))
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id):
        self.captured.append(("delete", memory_id))
        return {"result": "Memory deleted.", "memory_id": memory_id}


# The five mem0_* tools retired in the PRD-029 decommission (2026-06-28).
# They are no longer advertised in get_tool_schemas() and any call to
# handle_tool_call() with these names falls through to "Unknown tool".
RETIRED_MEM0_TOOLS = ["mem0_list", "mem0_search", "mem0_add", "mem0_update", "mem0_delete"]


class TestMem0RetiredTools:
    """PRD-029 decommission (2026-06-28): the five mem0_* tools (list/search/
    add/update/delete) are RETIRED.

    Replaces the old behavioral handler tests (TestMem0V3Tools /
    TestMem0UpdateDelete and the tool-driven circuit-breaker tests in
    TestMem0ErrorHandling). The handlers no longer exist; the contract is now:
      (a) each retired name is ABSENT from get_tool_schemas(), and
      (b) dispatching it via handle_tool_call() returns an "Unknown tool" error.
    """

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    @pytest.mark.parametrize("tool_name", RETIRED_MEM0_TOOLS)
    def test_retired_tool_absent_from_schemas(self, tool_name):
        provider = Mem0MemoryProvider()
        names = [s["name"] for s in provider.get_tool_schemas()]
        assert tool_name not in names

    @pytest.mark.parametrize("tool_name", RETIRED_MEM0_TOOLS)
    def test_retired_tool_dispatch_returns_unknown(self, monkeypatch, tool_name):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(tool_name, {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]
        # The retired handlers are gone — nothing reaches the backend.
        assert backend.captured == []

    def test_retired_dispatch_does_not_trip_circuit_breaker(self, monkeypatch):
        """An unknown (retired) tool name is a no-op dispatch error, not a
        backend failure — it must not move the circuit breaker."""
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        for tool_name in RETIRED_MEM0_TOOLS:
            provider.handle_tool_call(tool_name, {"content": "x", "query": "x",
                                                  "memory_id": "x", "text": "x"})
        assert provider._consecutive_failures == 0
        assert backend.captured == []


class TestMem0V3Internal:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_sync_turn_is_no_op_for_writes(self, monkeypatch):
        """Fork invariant (PRD-029): sync_turn performs NO per-turn write.

        Upstream v3 does backend.add(infer=True) every turn (passive
        extraction). The fork neutralizes that to close the ungoverned-autosave
        loop — durable writes happen only via explicit mem0_add and the governed
        consolidation pipeline. sync_turn must only cache orchestration state.
        """
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.sync_turn("user said", "assistant replied", session_id="s1")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=2)
        # No write of any kind reached the backend.
        assert backend.captured == []
        assert not any(c[0] == "add" for c in backend.captured)
        # Orchestration state IS cached (intent gate + deduplicator inputs).
        assert provider._last_assistant_content == "assistant replied"
        assert provider._had_tool_calls is False  # short reply -> length proxy False

    def test_old_tool_names_return_unknown(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_profile", {}))
        assert "error" in result
        result = json.loads(provider.handle_tool_call("mem0_conclude", {}))
        assert "error" in result


class TestMem0V3Config:

    def test_tool_schemas_v3_tools_present(self):
        """Post-decommission: the five mem0_* tools are ABSENT; chronicle_search
        is the sole advertised tool."""
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "chronicle_search" in names
        for retired in RETIRED_MEM0_TOOLS:
            assert retired not in names
        assert "mem0_profile" not in names
        assert "mem0_conclude" not in names

    def test_system_prompt_new_tool_names(self):
        """The system prompt block advertises chronicle_search only — none of the
        retired mem0_* tools, and no mode/user-id leakage."""
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._chronicle = object()  # non-empty -> block is rendered
        block = provider.system_prompt_block()
        assert "chronicle_search" in block
        for retired in RETIRED_MEM0_TOOLS:
            assert retired not in block
        assert "mem0_profile" not in block
        assert "mem0_conclude" not in block
        assert "Mode:" not in block

    def test_system_prompt_empty_without_chronicle(self):
        """If the chronicle searcher is unavailable, the block is empty."""
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._chronicle = None
        assert provider.system_prompt_block() == ""

    def test_system_prompt_no_mode_leak_platform(self):
        """No 'platform' / 'OSS' / 'Mode:' mode string — mode is no longer surfaced."""
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._mode = "platform"
        provider._chronicle = object()
        block = provider.system_prompt_block()
        assert "chronicle_search" in block
        assert "platform" not in block
        assert "Mode:" not in block

    def test_system_prompt_no_mode_leak_oss(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._mode = "oss"
        provider._chronicle = object()
        block = provider.system_prompt_block()
        assert "chronicle_search" in block
        assert "OSS" not in block
        assert "Mode:" not in block

    def test_search_schema_constant_still_has_rerank(self):
        """The SEARCH_SCHEMA module constant is retained (unadvertised) so the
        tool is a one-line re-enable. It must still carry the rerank property —
        but it is NOT advertised via get_tool_schemas()."""
        from plugins.memory.mem0 import SEARCH_SCHEMA
        assert "rerank" in SEARCH_SCHEMA["parameters"]["properties"]
        assert SEARCH_SCHEMA["parameters"]["properties"]["rerank"]["type"] == "boolean"
        # ...but search is no longer advertised.
        provider = Mem0MemoryProvider()
        names = [s["name"] for s in provider.get_tool_schemas()]
        assert "mem0_search" not in names


class TestMem0ModeSwitch:

    def test_default_mode_is_platform(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        provider.initialize("test")
        assert provider._mode == "platform"

    def test_missing_mode_key_defaults_platform(self, monkeypatch, tmp_path):
        """Backward compat: old mem0.json without mode key works."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "mem0.json"
        config_path.write_text('{"user_id": "old-user"}')
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        provider.initialize("test")
        assert provider._mode == "platform"
        assert provider._user_id == "old-user"

    def test_is_available_platform_needs_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        provider = Mem0MemoryProvider()
        assert provider.is_available() is False

    def test_is_available_oss_needs_vector(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "mem0.json"
        config_path.write_text('{"mode": "oss", "oss": {"vector_store": {"provider": "qdrant"}}}')
        provider = Mem0MemoryProvider()
        assert provider.is_available() is True

    def test_is_available_oss_no_vector(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "mem0.json"
        config_path.write_text('{"mode": "oss", "oss": {}}')
        provider = Mem0MemoryProvider()
        assert provider.is_available() is False

    def test_tool_schemas_stable(self):
        """Post-decommission stable surface: chronicle_search only."""
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert names == ["chronicle_search"]

    def test_system_prompt_independent_of_mode(self):
        """The prompt block no longer surfaces the backend mode at all — the
        same chronicle-only block renders regardless of platform/oss mode."""
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._chronicle = object()
        provider._mode = "oss"
        oss_block = provider.system_prompt_block()
        provider._mode = "platform"
        platform_block = provider.system_prompt_block()
        assert oss_block == platform_block
        assert "chronicle_search" in oss_block
        assert "OSS" not in oss_block
        assert "platform" not in platform_block
        assert "mem0_search" not in oss_block


class TestMem0UserIdResolution:
    """user_id resolution: configured override > gateway-native id > placeholder.

    Same human across CLI / Telegram / Discord / Slack / etc. should map to
    the same memory store when MEM0_USER_ID is set, and only fall back to the
    gateway-native id when it isn't.
    """

    def _provider(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        # Skip backend instantiation — we only care about identity resolution.
        provider._create_backend = lambda: None  # type: ignore[method-assign]
        return provider

    def test_env_override_beats_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_USER_ID", "ryan@example.com")
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "ryan@example.com"

    def test_file_override_beats_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        (tmp_path / "mem0.json").write_text('{"user_id": "ryan@example.com"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "ryan@example.com"

    def test_unset_falls_back_to_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "123456789"

    def test_unset_and_no_kwargs_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test")
        assert provider._user_id == "hermes-user"

    def test_legacy_placeholder_in_config_does_not_override_kwargs(self, monkeypatch, tmp_path):
        # Setup wizard historically wrote {"user_id": "hermes-user"} as the
        # suggested default. Treat that placeholder as unset so users on
        # gateways still get gateway-native ids — not silent collisions.
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        (tmp_path / "mem0.json").write_text('{"user_id": "hermes-user"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "123456789"


class TestMem0WriteMetadata:
    """Writes carry metadata.channel so per-channel filtered views are possible
    without coupling identity to the channel.
    """

    def _make_provider(self, channel: str = "cli"):
        provider = Mem0MemoryProvider()
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._channel = channel
        provider._backend = FakeBackend()
        return provider

    def test_add_tool_retired_writes_nothing(self):
        """mem0_add is retired — dispatching it is an unknown-tool no-op that
        reaches no backend write, regardless of channel. (The channel-metadata
        path it used to exercise is gone with the write tools.)"""
        provider = self._make_provider("telegram")
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "user likes dark mode"}))
        assert "error" in result
        assert "Unknown tool" in result["error"]
        assert provider._backend.captured == []

    def test_sync_turn_writes_nothing(self):
        """Fork invariant (PRD-029): sync_turn never writes (no passive extraction).

        Channel metadata is still attached to EXPLICIT writes (mem0_add), which
        test_add_tool_passes_channel_metadata covers — but sync_turn itself must
        not emit any backend.add, regardless of channel.
        """
        provider = self._make_provider("discord")
        provider.sync_turn("hi", "hello", session_id="s")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        adds = [c for c in provider._backend.captured if c[0] == "add"]
        assert adds == [], "sync_turn must perform no passive write"


class TestMem0Security:
    """Fork-local regression guards (PRD-029 Phase 0)."""

    def test_save_config_sets_owner_only_permissions(self, tmp_path):
        """mem0.json must be 0o600 — it can hold MEM0_API_KEY / pgvector password."""
        import stat
        provider = Mem0MemoryProvider()
        provider.save_config({"user_id": "sylva", "agent_id": "hermes"}, str(tmp_path))
        config_file = tmp_path / "mem0.json"
        assert config_file.exists()
        mode = stat.S_IMODE(config_file.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"

    def test_legacy_flat_config_promoted_to_oss(self, monkeypatch, tmp_path):
        """Back-compat shim: a legacy flat mem0.json (no mode/oss, top-level
        vector_store, no api_key) must resolve to OSS mode so memory does not
        silently go dark after the v3 bump."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        (tmp_path / "mem0.json").write_text(json.dumps({
            "vector_store": {"provider": "qdrant",
                             "config": {"url": "http://qdrant:6333",
                                        "collection_name": "sylva_memories"}},
            "embedder": {"provider": "openai",
                         "config": {"model": "bge-m3",
                                    "openai_base_url": "http://tei-bge-m3:80"}},
            "llm": {"provider": "openai", "config": {"model": "qwen3-4b-instruct"}},
            "history_db_path": "/opt/data/mem0_history.db",
        }))
        from plugins.memory.mem0 import _load_config
        cfg = _load_config()
        assert cfg["mode"] == "oss"
        assert cfg["oss"]["vector_store"]["config"]["collection_name"] == "sylva_memories"
        assert cfg["oss"]["embedder"]["config"]["openai_base_url"] == "http://tei-bge-m3:80"
        assert cfg["oss"]["history_db_path"] == "/opt/data/mem0_history.db"
        provider = Mem0MemoryProvider()
        assert provider.is_available() is True

    def test_v3_config_not_double_promoted(self, monkeypatch, tmp_path):
        """A proper v3 config (mode=oss, nested oss) must pass through untouched."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        (tmp_path / "mem0.json").write_text(json.dumps({
            "mode": "oss",
            "oss": {"vector_store": {"provider": "qdrant", "config": {"collection_name": "mem0"}}},
        }))
        from plugins.memory.mem0 import _load_config
        cfg = _load_config()
        assert cfg["mode"] == "oss"
        assert "vector_store" not in cfg  # no top-level leak
        assert cfg["oss"]["vector_store"]["config"]["collection_name"] == "mem0"


class FakeChronicle:
    """Records chronicle.search() calls for prefetch routing tests."""

    def __init__(self, results=None):
        self._results = results or []
        self.calls = []

    def search(self, query, *, speaker="any", date_from="", date_to="", top_k=5):
        self.calls.append((query, top_k))
        return self._results


class TestMem0PrefetchDecommission:
    """PRD-029 decommission (2026-06-28): prefetch is chronicle-only.

    The old 'stable_knowledge' route searched the retired sylva_memories store
    on every non-trivial turn; it is now an intentional no-op. Only the
    'historical_memory' route warms the chronicle.
    """

    def _make_provider(self, backend, chronicle):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        provider._chronicle = chronicle
        # Previous turn had no tool activity -> intent gate uses pattern check
        # only, and a substantive question still passes through.
        provider._had_tool_calls = False
        return provider

    def test_stable_knowledge_route_is_noop(self):
        """A non-historical query must NOT hit the mem0 backend search, and must
        NOT warm the chronicle — the stable route is a deliberate no-op."""
        backend = FakeBackend(search_results=[{"id": "x", "memory": "y"}])
        chronicle = FakeChronicle(results=[{"date": "2026-01-01", "speaker": "scott", "data": "hi"}])
        provider = self._make_provider(backend, chronicle)

        provider.queue_prefetch("what is my favorite programming language")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)

        # The retired stable route must never call backend.search.
        assert not any(c[0] == "search" for c in backend.captured)
        # Stable knowledge does not route to the chronicle either.
        assert chronicle.calls == []

    def test_historical_route_hits_chronicle_only(self):
        """A recall query routes to the chronicle searcher — never the mem0
        backend — and is bounded to top_k=2 (PRD-041 FR-1)."""
        backend = FakeBackend(search_results=[{"id": "x", "memory": "y"}])
        chronicle = FakeChronicle(
            results=[{"date": "2026-01-01", "speaker": "scott", "data": "we discussed X"}]
        )
        provider = self._make_provider(backend, chronicle)

        provider.queue_prefetch("remember what we talked about last week")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)

        # Chronicle was searched; the mem0 backend was not.
        assert len(chronicle.calls) == 1
        # PRD-041 FR-1 / PRD-052 FR-A1: bounded to the provider's resolved
        # top-k knob, whose DEFAULT stays 2 (never the legacy top-5).
        assert chronicle.calls[0][1] == provider._recall_assist_top_k == 2
        assert not any(c[0] == "search" for c in backend.captured)

    def test_prefetch_noop_without_chronicle(self):
        """No chronicle -> queue_prefetch is an immediate no-op (no thread, no
        backend search)."""
        backend = FakeBackend(search_results=[{"id": "x", "memory": "y"}])
        provider = self._make_provider(backend, None)
        provider.queue_prefetch("remember last week")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        assert not any(c[0] == "search" for c in backend.captured)


class TestRecallAssistOnTurnStart:
    """PRD-041 FR-1 (D4): same-turn warming via on_turn_start, bounded injection."""

    def _make_provider(self, chronicle):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = FakeBackend()
        provider._chronicle = chronicle
        provider._had_tool_calls = False
        return provider

    def test_on_turn_start_warms_same_turn(self):
        """on_turn_start fires queue_prefetch with the CURRENT message so the
        same-turn prefetch() read injects the recall hit (AC-001). Without this
        override the warm would land one turn late."""
        chronicle = FakeChronicle(
            results=[{"date": "2026-06-12", "speaker": "sylva",
                      "data": "I guessed Scott was 37; he is older."}]
        )
        provider = self._make_provider(chronicle)

        provider.on_turn_start(1, "do you remember when you guessed my age?")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)

        # The current-turn recall question warmed the chronicle at the
        # resolved knob depth (default 2 — PRD-052 FR-A1).
        assert len(chronicle.calls) == 1
        assert chronicle.calls[0][1] == provider._recall_assist_top_k == 2
        injected = provider.prefetch("do you remember when you guessed my age?")
        assert "guessed" in injected

    def test_on_turn_start_noop_on_non_recall(self):
        """A non-recall message warms nothing — precision holds (AC-002)."""
        chronicle = FakeChronicle(
            results=[{"date": "2026-06-12", "speaker": "scott", "data": "x"}]
        )
        provider = self._make_provider(chronicle)
        provider.on_turn_start(1, "write me a function to parse CSV")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        assert chronicle.calls == []

    def test_supersede_clears_stale_result(self):
        """A newer queue_prefetch bumps the generation and clears any prior
        result, so a stale recall hit can't survive into a later turn."""
        chronicle = FakeChronicle(
            results=[{"date": "2026-06-01", "speaker": "scott", "data": "old hit"}]
        )
        provider = self._make_provider(chronicle)
        # Turn 1: a recall query populates the result.
        provider.queue_prefetch("remember when we set up the cron jobs?")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        assert provider._prefetch_result  # warmed
        # Turn 2: a non-recall query supersedes — bump+clear, no new write.
        provider.queue_prefetch("write me a function")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        assert provider._prefetch_result == ""  # stale recall cleared

    def test_recall_assist_char_cap(self):
        """The injection is bounded well below the legacy 8000-char budget."""
        big = "x" * 4000
        chronicle = FakeChronicle(results=[
            {"date": "2026-06-01", "speaker": "scott", "data": big},
            {"date": "2026-06-02", "speaker": "sylva", "data": big},
        ])
        provider = self._make_provider(chronicle)
        provider.queue_prefetch("remember when we discussed the budget?")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        injected = provider.prefetch("remember when we discussed the budget?")
        # ~1200-char cap + the PRD-052 fixed frame (header + verify footer,
        # both cap-exempt) — still far below the legacy 8000.
        assert len(injected) < 2200
        assert injected.startswith(mem0_module._RECALL_ASSIST_HEADER)
        assert injected.rstrip().endswith(mem0_module._RECALL_ASSIST_FOOTER)


class TestRecallAssistRouter:
    """PRD-041 FR-1 (D2): precision-first recall classifier."""

    POSITIVE = [
        "do you remember when you first tried to guess my age?",
        "remember when we set up the cron jobs?",
        "earlier you said the pin was 64000, right?",   # weak cue + anchor
        "when did we decide to use Qdrant?",
        "last time we talked about the dashboard",
        "what was in our last conversation?",
        "can you check the chronicle for that?",
        "remind me what we decided about the proxy",
        "have we discussed the egress allowlist before?",
        "back when we first met, what did I say?",
    ]

    NEGATIVE = [
        "write me a function",
        "hey",
        "show me the git history",
        "before you edit the file, run tests",
        "remember to run the formatter",
        "what happened in June for federal grants?",
        "what is my favorite programming language",
        "earlier today the build failed",     # anchor alone, no recall cue
        "previously this was broken",
        "go ahead",
        # review NEEDS-FIX 1: "do you remember TO …" is a task reminder.
        "do you remember to lock the door",
        "did you remember to save the file?",
        # review NEEDS-FIX 2: unanchored second-person refers to THIS session.
        "you wrote this function wrong",
        "you said to use a dict here",
        "you called it with bad args",
    ]

    def test_positives(self):
        from plugins.memory.mem0.orchestrator import RecallAssistRouter
        for m in self.POSITIVE:
            assert RecallAssistRouter.is_recall_query(m), f"should match: {m!r}"

    def test_negatives(self):
        from plugins.memory.mem0.orchestrator import RecallAssistRouter
        for m in self.NEGATIVE:
            assert not RecallAssistRouter.is_recall_query(m), f"should NOT match: {m!r}"

    def test_empty(self):
        from plugins.memory.mem0.orchestrator import RecallAssistRouter
        assert RecallAssistRouter.is_recall_query("") is False


class TestMem0ChronicleSearchAdvertised:
    """chronicle_search remains the sole advertised + dispatchable tool."""

    def test_chronicle_search_advertised(self):
        provider = Mem0MemoryProvider()
        names = [s["name"] for s in provider.get_tool_schemas()]
        assert "chronicle_search" in names

    def test_chronicle_search_dispatchable(self):
        """chronicle_search dispatches to the chronicle searcher (not the
        unknown-tool fallthrough)."""
        provider = Mem0MemoryProvider()
        provider._chronicle = FakeChronicle(
            results=[{"date": "2026-01-01", "speaker": "scott", "data": "we shipped it"}]
        )
        result = json.loads(provider.handle_tool_call("chronicle_search", {"query": "ship"}))
        assert "error" not in result
        assert result["count"] == 1
        assert result["results"][0]["data"] == "we shipped it"
        assert provider._chronicle.calls == [("ship", 5)]


class TestOSSBackendSafety:
    """OSSBackend protected-collection guard (PRD-029 Phase 0)."""

    def test_recreate_refuses_protected_collection(self):
        """A dim mismatch on a configured/curated collection must RAISE, never
        delete — fail loud, preserve data."""
        from plugins.memory.mem0._backend import (
            OSSBackend, ProtectedCollectionError,
        )

        class _FakeVectors:
            size = 1024

        class _FakeParams:
            vectors = _FakeVectors()

        class _FakeConfig:
            params = _FakeParams()

        class _FakeInfo:
            config = _FakeConfig()

        class _FakeQdrant:
            deleted = []

            def __init__(self, *a, **k):
                pass

            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                return _FakeInfo()

            def delete_collection(self, name):
                _FakeQdrant.deleted.append(name)

            def close(self):
                pass

        import qdrant_client
        orig = qdrant_client.QdrantClient
        qdrant_client.QdrantClient = _FakeQdrant
        try:
            with pytest.raises(ProtectedCollectionError):
                OSSBackend._recreate_collection_if_dims_changed(
                    "qdrant",
                    {"url": "http://qdrant:6333", "collection_name": "sylva_memories"},
                    expected_dims=1536,  # != running 1024 -> would delete
                )
            assert _FakeQdrant.deleted == []  # nothing deleted
        finally:
            qdrant_client.QdrantClient = orig

    def test_recreate_allows_default_mem0_collection(self):
        """The throwaway mem0 default IS auto-recreatable on dim change."""
        from plugins.memory.mem0._backend import OSSBackend

        class _FakeVectors:
            size = 768

        class _FakeParams:
            vectors = _FakeVectors()

        class _FakeConfig:
            params = _FakeParams()

        class _FakeInfo:
            config = _FakeConfig()

        class _FakeQdrant:
            deleted = []

            def __init__(self, *a, **k):
                pass

            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                return _FakeInfo()

            def delete_collection(self, name):
                _FakeQdrant.deleted.append(name)

            def close(self):
                pass

        import qdrant_client
        orig = qdrant_client.QdrantClient
        qdrant_client.QdrantClient = _FakeQdrant
        try:
            OSSBackend._recreate_collection_if_dims_changed(
                "qdrant",
                {"url": "http://qdrant:6333", "collection_name": "mem0"},
                expected_dims=1024,  # != running 768 -> allowed delete
            )
            assert _FakeQdrant.deleted == ["mem0"]
        finally:
            qdrant_client.QdrantClient = orig


class TestRecallAssistKnobs:
    """PRD-052 FR-A1 — depth/cap knobs via the initialize() kwargs plumb.
    Defaults UNCHANGED at 2/1200 (adversarial F3 refuted deeper defaults on
    live evidence); overrides clamp to k∈[1,5], chars∈[400,4000]."""

    def _init(self, **kwargs):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session", **kwargs)
        return provider

    def test_defaults_unchanged_without_kwargs(self):
        p = self._init()
        assert p._recall_assist_top_k == 2
        assert p._recall_assist_max_chars == 1200

    def test_overrides_apply(self):
        p = self._init(recall_assist_top_k=4, recall_assist_max_chars=2000)
        assert p._recall_assist_top_k == 4
        assert p._recall_assist_max_chars == 2000

    @pytest.mark.parametrize("raw,expect", [(0, 1), (-3, 1), (99, 5), ("4", 4)])
    def test_top_k_clamps(self, raw, expect):
        assert self._init(recall_assist_top_k=raw)._recall_assist_top_k == expect

    @pytest.mark.parametrize("raw,expect", [(10, 400), (100000, 4000), ("800", 800)])
    def test_max_chars_clamps(self, raw, expect):
        assert self._init(recall_assist_max_chars=raw)._recall_assist_max_chars == expect

    def test_garbage_values_fall_back_to_defaults(self):
        p = self._init(recall_assist_top_k="lots", recall_assist_max_chars=None)
        assert p._recall_assist_top_k == 2
        assert p._recall_assist_max_chars == 1200

    def test_knob_reaches_the_search_call(self):
        chronicle = FakeChronicle(results=[{"date": "2026-06-12", "speaker": "sylva",
                                            "data": "we discussed the budget governor"}])
        p = self._init(recall_assist_top_k=3)
        p._chronicle = chronicle
        p._backend = FakeBackend()
        p._had_tool_calls = False
        p.queue_prefetch("remember when we discussed the budget?")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        assert chronicle.calls[0][1] == 3


class TestRecallAssistFrame:
    """PRD-052 FR-A2 — data-not-instructions header + verify-first footer,
    added AFTER dedup and OUTSIDE the char cap."""

    def _provider(self, results):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._chronicle = FakeChronicle(results=results)
        provider._backend = FakeBackend()
        provider._had_tool_calls = False
        return provider

    def _inject(self, provider, query):
        provider.queue_prefetch(query)
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        return provider.prefetch(query)

    def test_frame_present_exactly_once_with_intact_footer(self):
        p = self._provider([{"date": "2026-06-12", "speaker": "sylva",
                             "data": "we tuned the ranker so eligibility wins"}])
        injected = self._inject(p, "do you remember when we tuned the ranker?")
        assert injected.count(mem0_module._RECALL_ASSIST_HEADER) == 1
        assert injected.count(mem0_module._RECALL_ASSIST_FOOTER) == 1
        assert injected.startswith(mem0_module._RECALL_ASSIST_HEADER)
        assert injected.rstrip().endswith(mem0_module._RECALL_ASSIST_FOOTER)
        assert "we tuned the ranker" in injected

    def test_zero_hits_injects_nothing_no_orphan_frame(self):
        p = self._provider([])
        injected = self._inject(p, "do you remember when we tuned the ranker?")
        assert injected == ""

    def test_all_deduped_injects_nothing_no_orphan_frame(self):
        # the sole hit is (near-)verbatim the user's question → Jaccard-dropped
        q = "do you remember when we tuned the ranker so eligibility wins the day"
        p = self._provider([{"date": "2026-06-12", "speaker": "sylva", "data": q}])
        injected = self._inject(p, q)
        assert injected == ""

    def test_frame_lines_survive_sanitize_context(self):
        """N1: the frame must not match memory_manager's strip patterns, or the
        wrapper would silently delete it from provider output."""
        from agent.memory_manager import sanitize_context

        block = (f"{mem0_module._RECALL_ASSIST_HEADER}\n"
                 "- [2026-06-12 sylva] a recalled line\n"
                 f"{mem0_module._RECALL_ASSIST_FOOTER}")
        assert sanitize_context(block) == block

    def test_sidecar_display_payload_carries_no_frame_text(self):
        """AC-003 parity: the PRD-042 structured payload is presentation-free —
        frame text never reaches state.db."""
        import json as _json

        hit = {"date": "2026-06-12", "speaker": "sylva",
               "data": "we tuned the ranker so eligibility wins"}
        p = self._provider([hit])
        self._inject(p, "do you remember when we tuned the ranker?")
        display = p.take_recall_assist_display()
        assert display and display["hits"] == [hit]
        blob = _json.dumps(display)
        assert mem0_module._RECALL_ASSIST_HEADER not in blob
        assert mem0_module._RECALL_ASSIST_FOOTER not in blob
