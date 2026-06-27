"""Tests for Mem0 v3 API — new tool names, paginated responses, update/delete tools."""

import json
import pytest

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


class TestMem0V3Tools:
    """Test v3 tool names and response handling."""

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_list_returns_paginated_with_ids(self, monkeypatch):
        backend = FakeBackend(all_results={
            "count": 2,
            "results": [
                {"id": "mem-1", "memory": "alpha"},
                {"id": "mem-2", "memory": "beta"},
            ]
        })
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_list", {}))
        assert result["count"] == 2
        assert result["results"][0]["id"] == "mem-1"
        assert result["results"][0]["memory"] == "alpha"

    def test_list_pagination_params(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.handle_tool_call("mem0_list", {"page": 2, "page_size": 50})
        assert backend.captured[0][1]["page"] == 2
        assert backend.captured[0][1]["page_size"] == 50

    def test_list_empty(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_list", {}))
        assert result["result"] == "No memories stored yet."

    def test_search_returns_ids(self, monkeypatch):
        backend = FakeBackend(search_results=[{"id": "mem-1", "memory": "foo", "score": 0.9}])
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_search", {"query": "test"}))
        assert result["results"][0]["id"] == "mem-1"

    def test_search_uses_filters(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.handle_tool_call("mem0_search", {"query": "hello", "top_k": 3})
        assert backend.captured[0][2]["filters"] == {"user_id": "u123"}
        assert backend.captured[0][2]["top_k"] == 3

    def test_search_rerank_default_true(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.handle_tool_call("mem0_search", {"query": "test"})
        assert backend.captured[0][2]["rerank"] is True

    def test_search_rerank_override_false(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.handle_tool_call("mem0_search", {"query": "test", "rerank": False})
        assert backend.captured[0][2]["rerank"] is False

    def test_add_uses_content_param(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "user likes dark mode"}))
        assert len(backend.captured) == 1
        call = backend.captured[0]
        assert call[2]["infer"] is False
        assert call[2]["user_id"] == "u123"
        assert call[2]["agent_id"] == "hermes"
        assert "event_id" in result

    def test_add_returns_event_id(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "test"}))
        assert result["event_id"] == "evt-test-123"

    def test_add_missing_content(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_add", {}))
        assert "error" in result

    def test_old_tool_names_return_unknown(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_profile", {}))
        assert "error" in result
        result = json.loads(provider.handle_tool_call("mem0_conclude", {}))
        assert "error" in result


class TestMem0UpdateDelete:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_update_calls_sdk(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "mem-1", "text": "updated fact"}
        ))
        assert backend.captured[0][1] == "mem-1"
        assert backend.captured[0][2] == "updated fact"
        assert result["result"] == "Memory updated."
        assert result["memory_id"] == "mem-1"

    def test_update_missing_memory_id(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_update", {"text": "no id"}))
        assert "error" in result

    def test_update_missing_text(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_update", {"memory_id": "mem-1"}))
        assert "error" in result

    def test_delete_calls_sdk(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "mem-1"}
        ))
        assert backend.captured[0][1] == "mem-1"
        assert result["result"] == "Memory deleted."

    def test_delete_missing_memory_id(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_delete", {}))
        assert "error" in result


class TestMem0ErrorHandling:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_update_404_no_circuit_breaker(self, monkeypatch):
        backend = FakeBackend()
        backend.update = lambda mid, text: (_ for _ in ()).throw(Exception("404 Not Found"))
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "bad-id", "text": "x"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_delete_404_no_circuit_breaker(self, monkeypatch):
        backend = FakeBackend()
        backend.delete = lambda mid: (_ for _ in ()).throw(Exception("404 not found"))
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "bad-id"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_update_validation_error_no_circuit_breaker(self, monkeypatch):
        """ValidationError (bad UUID format) should not trip circuit breaker."""
        class ValidationError(Exception):
            pass
        backend = FakeBackend()
        backend.update = lambda mid, text: (_ for _ in ()).throw(
            ValidationError('{"error":"memory_id should be a valid UUID"}')
        )
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "not-a-uuid", "text": "x"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_delete_validation_error_no_circuit_breaker(self, monkeypatch):
        class ValidationError(Exception):
            pass
        backend = FakeBackend()
        backend.delete = lambda mid: (_ for _ in ()).throw(
            ValidationError('{"error":"memory_id should be a valid UUID"}')
        )
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "not-a-uuid"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_update_5xx_trips_circuit_breaker(self, monkeypatch):
        backend = FakeBackend()
        backend.update = lambda mid, text: (_ for _ in ()).throw(Exception("500 Internal Server Error"))
        provider = self._make_provider(monkeypatch, backend)
        provider.handle_tool_call("mem0_update", {"memory_id": "mem-1", "text": "x"})
        assert provider._consecutive_failures == 1


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
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        # The five v3 mem0 tools, plus the fork-local chronicle_search.
        assert names[:5] == ["mem0_list", "mem0_search", "mem0_add", "mem0_update", "mem0_delete"]
        assert "chronicle_search" in names
        assert "mem0_profile" not in names
        assert "mem0_conclude" not in names

    def test_system_prompt_new_tool_names(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        block = provider.system_prompt_block()
        assert "mem0_search" in block
        assert "mem0_add" in block
        assert "mem0_list" in block
        assert "mem0_update" in block
        assert "mem0_delete" in block
        assert "mem0_profile" not in block
        assert "mem0_conclude" not in block

    def test_system_prompt_shows_platform_mode(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._mode = "platform"
        block = provider.system_prompt_block()
        assert "platform" in block
        assert "Rerank" in block

    def test_system_prompt_shows_oss_mode(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._mode = "oss"
        block = provider.system_prompt_block()
        assert "OSS" in block
        assert "Rerank" not in block

    def test_search_schema_has_rerank(self):
        """rerank property available in SEARCH_SCHEMA for platform mode."""
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        search = next(s for s in schemas if s["name"] == "mem0_search")
        assert "rerank" in search["parameters"]["properties"]
        assert search["parameters"]["properties"]["rerank"]["type"] == "boolean"


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
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        # Fork ships the five v3 mem0 tools + chronicle_search (always advertised).
        assert names == ["mem0_list", "mem0_search", "mem0_add", "mem0_update",
                         "mem0_delete", "chronicle_search"]

    def test_system_prompt_includes_mode(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._mode = "oss"
        block = provider.system_prompt_block()
        assert "mem0_search" in block
        assert "mem0_list" in block
        assert "OSS" in block


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

    def test_add_tool_passes_channel_metadata(self):
        provider = self._make_provider("telegram")
        provider.handle_tool_call("mem0_add", {"content": "user likes dark mode"})
        call = provider._backend.captured[-1]
        assert call[2]["metadata"] == {"channel": "telegram"}

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
