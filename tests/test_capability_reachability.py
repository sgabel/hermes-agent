"""PRD-032 AC-023 — dispatch-surface *reachability* proof for the capability gate.

The unit tests in ``test_capability_policy.py`` prove ``classify()``/``guard()``
decide the *right* tier. They do NOT prove the gate is actually *wired* into the
execution paths — a tool could be perfectly classified T4 and still execute if
its dispatch surface never calls ``guard()``.

AC-023 closes exactly that gap: for every §6 dispatch surface, drive the **real
bypass path directly** (not a proxy) and assert the gate is consulted *before*
execution, and that a gate denial actually prevents the underlying action.

The technique is a **guard spy**: every enforcement site imports ``guard`` lazily
(``from tools.capability_policy import guard`` inside the function body), so
replacing ``tools.capability_policy.guard`` with a recording deny-stub is picked
up at call time. Each test then:
  1. calls the real dispatch entry point,
  2. asserts the underlying handler/subprocess/LLM was NEVER reached, and
  3. asserts the spy saw the expected ``(tool, surface)``.

Surfaces covered here by a direct real-path drive:
  - ``registry.dispatch`` (site 1) — the registry fan-in. This is the route for
    ``handle_function_call``'s two dispatch sites, the plugin ``dispatch_tool``
    bypass, the ``execute_code`` child re-entry, MCP *tool* calls, lazy installs,
    and built-in egress (``web_search``/``web_extract``/browser). Proving the
    fan-in gates before the handler proves all of them.
  - inline ``delegate_task`` (site 2) — the I5 agent-on-agent surface that never
    reaches the registry.
  - cron ``_run_job_script`` (site 3) — scripts shell out via ``subprocess.run``.
  - MCP ``SamplingHandler.__call__`` (site 4) — server-initiated sampling of our
    aux LLM, not a registry tool.
"""

import json

import pytest


# ---------------------------------------------------------------------------
# guard spy
# ---------------------------------------------------------------------------

class _GuardSpy:
    """Records every guard() call and returns a fixed allow/deny verdict."""

    def __init__(self, allowed: bool):
        self.allowed = allowed
        self.calls = []

    def __call__(self, tool, args=None, ctx=None, surface="dispatch"):
        self.calls.append({"tool": tool, "args": args, "ctx": ctx, "surface": surface})
        return {
            "allowed": self.allowed,
            "tier": "T4",
            "mode": "enforce",
            "reason": None if self.allowed else "reachability-test deny",
            "outcome": "allowed" if self.allowed else "blocked",
        }

    @property
    def tools(self):
        return [c["tool"] for c in self.calls]

    @property
    def surfaces(self):
        return [c["surface"] for c in self.calls]


@pytest.fixture()
def deny_spy(monkeypatch):
    spy = _GuardSpy(allowed=False)
    monkeypatch.setattr("tools.capability_policy.guard", spy)
    return spy


@pytest.fixture()
def allow_spy(monkeypatch):
    spy = _GuardSpy(allowed=True)
    monkeypatch.setattr("tools.capability_policy.guard", spy)
    return spy


# ---------------------------------------------------------------------------
# Site 1 — registry.dispatch (the fan-in for the majority of surfaces)
# ---------------------------------------------------------------------------

def _fresh_registry_with_spy_tool():
    """A standalone ToolRegistry with one tool whose handler records execution."""
    from tools.registry import ToolRegistry

    reg = ToolRegistry()
    state = {"ran": False}

    def _handler(args, **kwargs):
        state["ran"] = True
        return json.dumps({"ok": True})

    reg.register(
        name="spy_tool",
        toolset="test",
        schema={"type": "object", "properties": {}},
        handler=_handler,
    )
    return reg, state


def test_registry_dispatch_gates_before_handler(deny_spy):
    """A denied guard must prevent the tool handler from running at all —
    proving the gate sits BEFORE execution on the registry fan-in (covers the
    plugin dispatch_tool bypass, execute_code child re-entry, MCP tool calls,
    lazy installs, and built-in egress, which all route through here)."""
    reg, state = _fresh_registry_with_spy_tool()

    result = reg.dispatch("spy_tool", {"x": 1})

    assert state["ran"] is False, "handler ran despite a gate denial"
    assert deny_spy.tools == ["spy_tool"]
    assert deny_spy.surfaces == ["registry"]
    payload = json.loads(result)
    assert "BLOCKED by capability policy" in payload.get("error", "")
    assert payload.get("outcome") == "blocked"


def test_registry_dispatch_runs_handler_when_allowed(allow_spy):
    """The mirror: with an allowing gate the handler DOES run — proving the
    deny-case above is the gate biting, not the tool being broken."""
    reg, state = _fresh_registry_with_spy_tool()

    result = reg.dispatch("spy_tool", {"x": 1})

    assert state["ran"] is True
    assert allow_spy.tools == ["spy_tool"]
    assert json.loads(result) == {"ok": True}


# ---------------------------------------------------------------------------
# Site 2 — inline delegate_task (I5, never reaches the registry)
# ---------------------------------------------------------------------------

def test_inline_delegate_task_gates_before_dispatch(deny_spy):
    """delegate_task is dispatched inline by agent_runtime_helpers.invoke_tool and
    never reaches registry.dispatch, so it has its own gate. A denial must stop
    the real _dispatch_delegate_task from ever being called (I5)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from agent import agent_runtime_helpers

    dispatch_mock = MagicMock(return_value="delegated!")
    agent = SimpleNamespace(
        _dispatch_delegate_task=dispatch_mock,
        _memory_manager=None,
        session_id="",
        valid_tool_names=None,
        enabled_toolsets=None,
        disabled_toolsets=None,
        _current_turn_id="",
        _current_api_request_id="",
    )

    result = agent_runtime_helpers.invoke_tool(
        agent, "delegate_task", {"task": "do a thing"}, effective_task_id="",
        tool_call_id="tc1",
    )

    dispatch_mock.assert_not_called()
    assert deny_spy.tools == ["delegate_task"]
    assert deny_spy.surfaces == ["agent-runtime"]
    payload = json.loads(result)
    assert "BLOCKED by capability policy" in payload.get("error", "")


def test_inline_delegate_task_dispatches_when_allowed(allow_spy):
    """Mirror: an allowing gate lets the real delegate dispatch through."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from agent import agent_runtime_helpers

    dispatch_mock = MagicMock(return_value="delegated!")
    agent = SimpleNamespace(
        _dispatch_delegate_task=dispatch_mock,
        _memory_manager=None,
        session_id="",
        valid_tool_names=None,
        enabled_toolsets=None,
        disabled_toolsets=None,
        _current_turn_id="",
        _current_api_request_id="",
    )

    result = agent_runtime_helpers.invoke_tool(
        agent, "delegate_task", {"task": "do a thing"}, effective_task_id="",
        tool_call_id="tc1",
    )

    dispatch_mock.assert_called_once()
    assert result == "delegated!"
    assert allow_spy.surfaces == ["agent-runtime"]


# ---------------------------------------------------------------------------
# Site 3 — cron _run_job_script (subprocess.run, never enters dispatch)
# ---------------------------------------------------------------------------

def test_cron_script_gates_before_subprocess(deny_spy, monkeypatch, tmp_path):
    """A cron script shells out via subprocess.run. A denied guard must return
    (False, ...) before the interpreter is ever spawned — proven by a marker
    file the script would create only if it actually ran."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    marker = tmp_path / "SCRIPT_RAN"
    script = scripts_dir / "job.sh"
    script.write_text(f"#!/usr/bin/env bash\ntouch {marker}\n")
    script.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron import scheduler

    ok, msg = scheduler._run_job_script(str(script))

    assert ok is False, "cron script ran despite gate denial"
    assert not marker.exists(), "subprocess executed before the gate blocked it"
    assert "capability policy" in msg.lower()
    assert deny_spy.tools == ["cron_script"]
    assert deny_spy.surfaces == ["cron"]


# ---------------------------------------------------------------------------
# Site 4 — MCP SamplingHandler.__call__ (server-initiated aux-LLM sampling)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_sampling_gates_before_aux_llm(deny_spy):
    """A foreign MCP server steering our aux LLM via sampling is not a registry
    tool, so it has its own gate. A denial must short-circuit to an MCP error
    before any model call — asserted via the error-counter and the spy."""
    from tools.mcp_tool import SamplingHandler

    handler = SamplingHandler("evil-server", {})
    # The gate fires before context/params are touched, so None is safe here.
    result = await handler(None, None)

    assert handler.metrics["requests"] == 0, "reached request handling past the gate"
    assert handler.metrics["errors"] == 1
    assert deny_spy.tools == ["mcp_sampling"]
    assert deny_spy.surfaces == ["mcp"]
    # ErrorData (or a raised Exception fallback) — either way, not a real result.
    assert result is not None


# ---------------------------------------------------------------------------
# AC-025 — the I5 agent-driver surface is T4 and degrades-to-ask unattended.
#
# Note on scope: the external agent-driver *skills* (autonomous-ai-agents/
# {codex,claude-code,opencode}) are not dispatched as named tools — they load via
# `skill_view` (T0) and then execute through `terminal`/`execute_code` (T4 host
# shell when unattended) or `delegate_task`. So there is no standalone "codex"
# classifier hook; the I5 guarantee is carried by `delegate_task` (the in-process
# agent driver, explicitly T4) plus the exec tiering. This test pins the concrete,
# always-present I5 surface: `delegate_task` unattended+enforce → blocked_queued.
# ---------------------------------------------------------------------------

def test_delegate_task_t4_degrades_to_ask_unattended(monkeypatch, tmp_path):
    """delegate_task drives another capable agent (I5). Unattended + enforce it
    must NOT execute — it degrades to a durable one-shot approval queue."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    monkeypatch.setenv("HERMES_AUTONOMOUS", "1")  # unattended
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    import importlib
    import tools.capability_policy as cp
    importlib.reload(cp)

    assert cp.classify("delegate_task") == cp.Tier.T4
    out = cp.guard("delegate_task", {"task": "spawn a sub-agent"}, surface="agent-runtime")
    assert out["allowed"] is False
    assert out["outcome"] == "blocked_queued"
    assert out["tier"] == "T4"
