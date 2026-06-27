"""PRD-022 — cron memory-write split.

Verifies the `skip_memory` flag was split into three independently-controllable
knobs (`skip_file_memory`, `skip_provider_memory`, `memory_passive_enabled`)
while preserving exact behavior for the legacy `skip_memory` alias used by ~12
production call sites + ~58 test sites.

Two layers:
  1. Signature contract (inspect-only) — guards the two silent-breakage traps:
     `skip_memory` default MUST be None (not False), and the new params MUST be
     appended at the END of the signature (so cli.py:823's *args passthrough
     does not shift positional slots).
  2. Flag-resolution behavior — construct AIAgent with config/provider/store
     patched (no live Qdrant / no disk) and assert `_memory_store`,
     `_memory_manager`, and `_memory_passive_enabled` across every mode.

The end-to-end "cron agent actually persists via mem0_add" path is covered
by the live smoke test in the PRD (AC-010/011), not here. (PRD-029 renamed the
mem0 write verb mem0_conclude -> mem0_add and reclassified it T4.)
"""
import inspect
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


# --------------------------------------------------------------------------- #
# Layer 1 — signature contract (no construction)
# --------------------------------------------------------------------------- #

def test_skip_memory_default_is_none_not_false():
    """AC-005 trap: if the default stayed False, the alias stanza
    `if skip_memory is not None:` would fire on every default construction and
    silently force both split flags off, defeating the split."""
    params = inspect.signature(AIAgent.__init__).parameters
    assert params["skip_memory"].default is None, (
        "skip_memory default must be None — a False default re-arms the "
        "silent-breakage trap (alias fires on every default construction)."
    )


def test_split_params_exist_with_expected_defaults():
    params = inspect.signature(AIAgent.__init__).parameters
    assert params["skip_file_memory"].default is False
    assert params["skip_provider_memory"].default is False
    assert params["memory_passive_enabled"].default is True


def test_new_params_appended_at_end_preserves_positional_order():
    """AC-014 (Codex NIT): cli.py:823 does `_AIAgent(*args, **kwargs)`. The new
    params must come AFTER all pre-existing params so positional slots don't
    shift for any caller."""
    names = list(inspect.signature(AIAgent.__init__).parameters)
    for new_param in ("skip_file_memory", "skip_provider_memory", "memory_passive_enabled"):
        assert names.index(new_param) > names.index("pass_session_id"), (
            f"{new_param} must be appended after pre-existing params (pass_session_id)"
        )


# --------------------------------------------------------------------------- #
# Layer 2 — flag-resolution behavior (patched construction)
# --------------------------------------------------------------------------- #

def _build_agent(**flags):
    """Construct an AIAgent with config/provider/store patched so memory init is
    deterministic without live Qdrant or disk access.

    - config: memory_enabled + a configured provider so BOTH blocks are eligible.
    - MemoryStore: MagicMock so `_memory_store` is truthy when the file gate opens
      and `load_from_disk()` is a no-op.
    - load_memory_provider: returns an available fake so `_memory_manager` is
      truthy when the provider gate opens.
    """
    cfg = {
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
            "provider": "mem0",
            "nudge_interval": 10,
            "memory_char_limit": 2200,
            "user_char_limit": 1375,
        }
    }
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
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            **flags,
        )
    return agent


def test_default_construction_enables_both_memories():
    """AC-001: default (skip_memory=None) → both file + provider memory init."""
    a = _build_agent()
    assert a._memory_store is not None
    assert a._memory_manager is not None
    assert a._memory_passive_enabled is True


def test_legacy_alias_disables_both():
    """AC-002: skip_memory=True (the 12 prod + 58 test sites) → both off,
    exactly as before the split."""
    a = _build_agent(skip_memory=True)
    assert a._memory_store is None
    assert a._memory_manager is None


def test_split_file_off_provider_on():
    """AC-003 + AC-006: the cron mode. File memory off, provider on. Provider
    being non-None proves mem_config was hoisted (else NameError → swallowed →
    manager None)."""
    a = _build_agent(skip_file_memory=True, skip_provider_memory=False)
    assert a._memory_store is None
    assert a._memory_manager is not None


def test_inverse_split_file_on_provider_off():
    """AC-004: file memory on, provider off."""
    a = _build_agent(skip_file_memory=False, skip_provider_memory=True)
    assert a._memory_store is not None
    assert a._memory_manager is None


def test_legacy_alias_wins_over_split_flag():
    """AC-005 (P-2): conflicting `skip_memory=True` + `skip_file_memory=False`
    → legacy alias wins (both off). Documented: do not combine the two."""
    a = _build_agent(skip_memory=True, skip_file_memory=False, skip_provider_memory=False)
    assert a._memory_store is None
    assert a._memory_manager is None


def test_passive_flag_resolves_from_param():
    """AC-007: the cron call-site passes memory_passive_enabled=False; with the
    provider on, the manager exists but passive auto-extraction is gated off."""
    a = _build_agent(skip_file_memory=True, skip_provider_memory=False, memory_passive_enabled=False)
    assert a._memory_manager is not None
    assert a._memory_passive_enabled is False


def test_passive_suppression_skips_post_turn_sync():
    """AC-007: with passive off, the post-turn sync/prefetch mirror is a no-op
    even when a memory manager is present (so a cron turn never auto-extracts)."""
    a = _build_agent(skip_file_memory=True, skip_provider_memory=False, memory_passive_enabled=False)
    a._memory_manager = MagicMock()
    a._sync_external_memory_for_turn(
        original_user_message="hello",
        final_response="world",
        interrupted=False,
        messages=[],
    )
    a._memory_manager.sync_all.assert_not_called()
    a._memory_manager.queue_prefetch_all.assert_not_called()


def test_passive_enabled_runs_post_turn_sync():
    """Inverse of the above — with passive on (default), the mirror fires."""
    a = _build_agent()
    a._memory_manager = MagicMock()
    a._sync_external_memory_for_turn(
        original_user_message="hello",
        final_response="world",
        interrupted=False,
        messages=[],
    )
    a._memory_manager.sync_all.assert_called_once()
