"""Tests that the background review agent restricts tools at runtime, not at schema time.

Regression coverage for issue #15204 (the background skill-review agent must
not perform non-skill side effects like terminal, send_message, delegate_task)
combined with issue #25322 / PR #17276 (the review fork must hit the parent's
Anthropic/OpenRouter prefix cache).

Reconciling the two: the fork now inherits the parent's full ``tools`` schema
so the cache-key matches, and enforces the memory+skills restriction at
runtime via a thread-local whitelist on the existing
``get_pre_tool_call_block_message`` gate. Safety is preserved mechanically
(any non-whitelisted dispatch is blocked) without the schema-level narrowing
that caused the prefix-cache miss.
"""

from unittest.mock import patch


def _make_agent_stub(agent_cls):
    """Create a minimal AIAgent-like object with just enough state for _spawn_background_review."""
    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = None
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    # Non-None so the test catches a missing-kwarg regression.
    agent.enabled_toolsets = ["memory", "skills", "terminal"]
    agent.disabled_toolsets = ["spotify", "feishu_doc"]
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def test_background_review_matches_parent_toolset_config():
    """Fork must receive parent's toolset config so ``tools[]`` cache key matches."""
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    captured = {}

    def _capture_init(self, *args, **kwargs):
        captured["enabled_toolsets"] = kwargs.get("enabled_toolsets", "UNSET")
        captured["disabled_toolsets"] = kwargs.get("disabled_toolsets", "UNSET")
        raise RuntimeError("stop after capturing init args")

    with patch.object(run_agent.AIAgent, "__init__", _capture_init), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "enabled_toolsets" in captured, "AIAgent.__init__ was not called"
    assert captured["enabled_toolsets"] == agent.enabled_toolsets, (
        f"enabled_toolsets mismatch: {captured['enabled_toolsets']!r} "
        f"vs expected {agent.enabled_toolsets!r}"
    )
    assert captured["disabled_toolsets"] == agent.disabled_toolsets, (
        f"disabled_toolsets mismatch: {captured['disabled_toolsets']!r} "
        f"vs expected {agent.disabled_toolsets!r}"
    )


def _capture_review_whitelist(agent_cls, *, review_memory, review_skills):
    """Run _spawn_background_review with init stubbed, capturing the installed
    thread-local whitelist (the runtime dispatch gate)."""
    import run_agent
    from hermes_cli import plugins as _plugins

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(agent_cls)

    def _no_init(self, *args, **kwargs):
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=review_memory,
            review_skills=review_skills,
        )
    assert "whitelist" in captured, "set_thread_tool_whitelist was not called"
    return captured["whitelist"]


def test_memory_only_review_whitelist_denies_skills_and_mem0():
    """PRD-037 FR-3 / AC-007: a memory-only upkeep review (builtin_nudge_interval
    trigger) must whitelist ONLY the memory tool — skill_manage/skills and any
    mem0 write/recall tool are denied at dispatch. This is the guarantee that
    re-enabling MEMORY.md/USER.md upkeep does not reopen a skills or external-
    provider write path."""
    import run_agent

    whitelist = _capture_review_whitelist(
        run_agent.AIAgent, review_memory=True, review_skills=False
    )
    assert "memory" in whitelist
    # skills are NOT in a memory-only review.
    assert "skill_manage" not in whitelist
    assert "skill_view" not in whitelist
    assert "skills_list" not in whitelist
    # no mem0 writes (the provider only advertises chronicle_search anyway).
    assert "mem0_add" not in whitelist
    assert "mem0_update" not in whitelist
    assert "mem0_delete" not in whitelist
    # dangerous tools never appear.
    for t in ("terminal", "send_message", "delegate_task", "web_search", "execute_code"):
        assert t not in whitelist


def test_skills_only_review_whitelist_denies_memory():
    """A skills-only review gets the skills toolset, not memory."""
    import run_agent

    whitelist = _capture_review_whitelist(
        run_agent.AIAgent, review_memory=False, review_skills=True
    )
    assert "skill_manage" in whitelist
    assert "memory" not in whitelist


def test_combined_review_whitelist_has_both():
    """A combined trigger keeps both toolsets (back-compat with the old behavior)."""
    import run_agent

    whitelist = _capture_review_whitelist(
        run_agent.AIAgent, review_memory=True, review_skills=True
    )
    assert "memory" in whitelist
    assert "skill_manage" in whitelist
    for t in ("terminal", "send_message", "delegate_task", "execute_code"):
        assert t not in whitelist


def test_background_review_agent_tools_are_limited():
    """Verify the resolved memory+skills toolsets only contain memory and skill tools.

    Sanity check on the source of truth for what the runtime whitelist is
    derived from — if a future PR adds e.g. `terminal` to the `memory`
    toolset, the review-fork safety contract silently breaks.
    """
    from toolsets import resolve_multiple_toolsets

    expected_tools = set(resolve_multiple_toolsets(["memory", "skills"]))

    assert "memory" in expected_tools
    assert "skill_manage" in expected_tools
    assert "skill_view" in expected_tools
    assert "skills_list" in expected_tools

    assert "terminal" not in expected_tools
    assert "send_message" not in expected_tools
    assert "delegate_task" not in expected_tools
    assert "web_search" not in expected_tools
    assert "execute_code" not in expected_tools
