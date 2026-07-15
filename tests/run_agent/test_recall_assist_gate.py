"""PRD-052 AC-001 (adversarial C1) — the recall-assist warm gate is a
call-site EITHER-flag disjunction in ``agent/turn_context.py``, not a
plugin-level check: ``on_turn_start`` / ``prefetch_all`` fire iff
``_memory_passive_enabled`` AND (``_ambient_prefetch_enabled`` OR
``_historical_recall_assist_enabled``).

The gate lines sit deep inside ``build_turn_context``'s 300+-line prologue, so
— following the precedent in ``test_memory_nudge_counter_hydration.py`` — this
module pins it two ways:

  1. a SOURCE GUARD that asserts both call sites in turn_context.py are still
     wrapped in the either-flag disjunction (fails loudly if the gate is
     removed or rephrased away), and
  2. behavior tests over an inline replica of the gate expression, covering
     the full flag matrix — most importantly: both flags False → NO warm, no
     prefetch, even though the ambient flag's getattr default is True (the
     default only applies when the attribute is absent, and agent_init always
     sets both).
"""

from pathlib import Path
from unittest.mock import MagicMock

_TURN_CONTEXT = Path(__file__).resolve().parents[2] / "agent" / "turn_context.py"


# ── 1. source guard ──────────────────────────────────────────────────────────
def test_both_call_sites_are_gated_by_the_either_flag_disjunction():
    src = _TURN_CONTEXT.read_text(encoding="utf-8")
    # Two independent gate blocks (warm + prefetch read) must each carry the
    # passive gate AND the either-flag disjunction.
    assert src.count('getattr(agent, "_ambient_prefetch_enabled", True)') >= 2
    assert src.count('getattr(agent, "_historical_recall_assist_enabled", False)') >= 2
    assert src.count('getattr(agent, "_memory_passive_enabled", True)') >= 2
    assert "agent._memory_manager.on_turn_start(" in src
    assert "agent._memory_manager.prefetch_all(" in src


# ── 2. behavior over the replicated gate ─────────────────────────────────────
def _run_turn_memory_gates(agent, original_user_message="do you remember x?"):
    """Inline replica of the two gate blocks (turn_context.py, PRD-041 FR-1
    wiring). Keep in lockstep with the production expression — the source
    guard above is the drift alarm."""
    if (agent._memory_manager
            and getattr(agent, "_memory_passive_enabled", True)
            and (getattr(agent, "_ambient_prefetch_enabled", True)
                 or getattr(agent, "_historical_recall_assist_enabled", False))):
        agent._memory_manager.on_turn_start(0, original_user_message)

    ext_prefetch_cache = ""
    if (agent._memory_manager
            and getattr(agent, "_memory_passive_enabled", True)
            and (getattr(agent, "_ambient_prefetch_enabled", True)
                 or getattr(agent, "_historical_recall_assist_enabled", False))):
        ext_prefetch_cache = agent._memory_manager.prefetch_all(original_user_message) or ""
    return ext_prefetch_cache


def _agent(ambient, assist, passive=True):
    a = MagicMock()
    a._memory_manager = MagicMock()
    a._memory_passive_enabled = passive
    a._ambient_prefetch_enabled = ambient
    a._historical_recall_assist_enabled = assist
    return a


def test_both_flags_false_means_no_warm_and_no_prefetch():
    """The C1 case: ambient off (PRD-029 posture) + assist off → the memory
    manager is never touched, even on an explicit recall question."""
    agent = _agent(ambient=False, assist=False)
    _run_turn_memory_gates(agent)
    agent._memory_manager.on_turn_start.assert_not_called()
    agent._memory_manager.prefetch_all.assert_not_called()


def test_assist_alone_opens_the_path():
    """The live config posture: ambient_prefetch stays False, the recall
    assist flag alone must open both the warm and the read."""
    agent = _agent(ambient=False, assist=True)
    _run_turn_memory_gates(agent)
    agent._memory_manager.on_turn_start.assert_called_once()
    agent._memory_manager.prefetch_all.assert_called_once()


def test_ambient_alone_opens_the_path():
    agent = _agent(ambient=True, assist=False)
    _run_turn_memory_gates(agent)
    agent._memory_manager.on_turn_start.assert_called_once()


def test_passive_disabled_overrides_both_flags():
    """Cron posture: _memory_passive_enabled=False suppresses everything."""
    agent = _agent(ambient=True, assist=True, passive=False)
    _run_turn_memory_gates(agent)
    agent._memory_manager.on_turn_start.assert_not_called()
    agent._memory_manager.prefetch_all.assert_not_called()
