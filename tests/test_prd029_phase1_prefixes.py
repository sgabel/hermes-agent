"""PRD-029 Phase 1 — separable pre-fixes (cron exclusion + ambient-prefetch dial).

Covers the two structural pre-requisites the governed consolidation pass depends
on, landed ahead of the mem0 v3 merge (they need no merge):

- AC-013 precondition: ``"cron"`` is a member of ``_HIDDEN_SESSION_SOURCES`` so the
  exclusion is structural — ``_list_recent_sessions()`` drops cron rows with NO
  ``extra_exclude_sources`` passed (i.e. the constant change landed, not just a
  prompt passing ``exclude_sources="cron"`` an LLM can omit).
- AC-014 dial: ``memory.ambient_prefetch_enabled: false`` forces the existing
  ``_memory_passive_enabled`` lever OFF even on the interactive path. The gate logic
  lives in ``agent_init`` (``mem_config.get("ambient_prefetch_enabled", True)``); the
  runtime "a turn injects no chronicle" behaviour is verified in-container.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tools.session_search_tool import _HIDDEN_SESSION_SOURCES, _list_recent_sessions


def _build_agent(mem_extra=None):
    """Construct an AIAgent with memory init patched deterministic (no live
    Qdrant/disk), optionally merging ``mem_extra`` into the memory config block.
    Mirrors tests/run_agent/test_skip_memory_split.py::_build_agent."""
    mem = {
        "memory_enabled": True,
        "user_profile_enabled": True,
        "provider": "mem0",
        "nudge_interval": 10,
        "memory_char_limit": 2200,
        "user_char_limit": 1375,
    }
    if mem_extra:
        mem.update(mem_extra)
    cfg = {"memory": mem}
    fake_provider = MagicMock()
    fake_provider.is_available.return_value = True
    fake_provider.get_all_tool_schemas.return_value = []
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("tools.memory_tool.MemoryStore", return_value=MagicMock()),
        patch("plugins.memory.load_memory_provider", return_value=fake_provider),
    ):
        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
        )


def test_ambient_prefetch_default_is_true():
    """AC-014: absent the key, ambient prefetch stays enabled (no behavior change)."""
    a = _build_agent()
    assert a._ambient_prefetch_enabled is True


def test_ambient_prefetch_disabled_sets_dedicated_flag_only():
    """AC-014 (surgical): ambient_prefetch_enabled:false sets _ambient_prefetch_enabled
    False but MUST NOT touch the broader _memory_passive_enabled master switch (which
    also gates per-turn sync_all / session-end extraction / compression hooks — out of
    AC-014's scope). Guards the code-review fix that separated the two levers."""
    a = _build_agent({"ambient_prefetch_enabled": False})
    assert a._ambient_prefetch_enabled is False
    assert a._memory_passive_enabled is True  # write/lifecycle lever untouched


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def test_cron_is_a_hidden_source_constant():
    """AC-013: the exclusion is a constant, not a prompt argument."""
    assert "cron" in _HIDDEN_SESSION_SOURCES


def test_cron_sessions_dropped_without_explicit_exclude(db):
    """AC-013: _list_recent_sessions hides cron rows even when the caller passes
    NO extra_exclude_sources — the structural guarantee the consolidation pass
    relies on so it never re-reads its own reflection runs (echo-chamber)."""
    # A user-facing CLI session (must have a message to survive min_messages).
    db.create_session(session_id="user-cli", source="cli", model="test")
    db.append_message("user-cli", role="user", content="hey Sylva, what's up")
    db.end_session("user-cli", "cli_close")

    # A nightly reflection/cron session — the kind that must be excluded.
    db.create_session(session_id="cron-reflection", source="cron", model="test")
    db.append_message("cron-reflection", role="user", content="nightly reflection")
    db.end_session("cron-reflection", "cli_close")

    raw = _list_recent_sessions(db, limit=10)  # NO extra_exclude_sources
    payload = json.loads(raw)

    blob = json.dumps(payload)
    assert "cron-reflection" not in blob, "cron session leaked into recent-session enumeration"
    assert "user-cli" in blob, "user-facing session was wrongly dropped"


def test_explicit_exclude_still_unions_with_constant(db):
    """Passing extra_exclude_sources must not RE-ENABLE cron — union semantics."""
    db.create_session(session_id="user-cli2", source="cli", model="test")
    db.append_message("user-cli2", role="user", content="hi")
    db.end_session("user-cli2", "cli_close")
    db.create_session(session_id="cron2", source="cron", model="test")
    db.append_message("cron2", role="user", content="reflection")
    db.end_session("cron2", "cli_close")

    raw = _list_recent_sessions(db, limit=10, extra_exclude_sources=["voice"])
    assert "cron2" not in json.dumps(json.loads(raw))
