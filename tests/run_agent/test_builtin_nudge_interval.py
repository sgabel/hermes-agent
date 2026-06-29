"""PRD-037 FR-3 / AC-006 — built-in MEMORY.md/USER.md upkeep cadence resolves
from the dedicated ``builtin_nudge_interval`` knob, falling back to the legacy
``nudge_interval``.

The turn-gated background memory review fires when ``_memory_nudge_interval > 0``
(agent/turn_context.py). So the upkeep can stay ON (builtin_nudge_interval: 10)
while the legacy broad review stays OFF (nudge_interval: 0) — the exact split the
PRD-029 decommission left missing.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _build_agent(memory_cfg):
    cfg = {"memory": {"memory_enabled": True, "user_profile_enabled": True,
                      "provider": "", **memory_cfg}}
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("tools.memory_tool.MemoryStore", return_value=MagicMock()),
    ):
        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
        )


def test_builtin_nudge_overrides_legacy_zero():
    """AC-006: nudge_interval 0 + builtin_nudge_interval 10 → review fires (>0)."""
    a = _build_agent({"nudge_interval": 0, "builtin_nudge_interval": 10})
    assert a._memory_nudge_interval == 10


def test_both_zero_disables_review():
    """AC-006: both 0 → review never fires."""
    a = _build_agent({"nudge_interval": 0, "builtin_nudge_interval": 0})
    assert a._memory_nudge_interval == 0


def test_falls_back_to_legacy_when_builtin_unset():
    """Back-compat: no builtin_nudge_interval → legacy nudge_interval wins."""
    a = _build_agent({"nudge_interval": 7})
    assert a._memory_nudge_interval == 7


def test_default_when_neither_set():
    a = _build_agent({})
    assert a._memory_nudge_interval == 10
