"""PRD-027 — proactive idle-time outreach: execution-profile mechanism.

The implementation splits into three seams, exercised here at the level each is
honestly testable at:

  * ``autonomy/exec_profile.py`` — the static profile registry, the fail-closed
    tool-surface assert, the delivery-pin resolver, and the quiet-hours math
    (pure functions → direct unit tests).
  * ``cron/scheduler.py`` — ``run_one_job`` dispatch → ``_run_exec_profile_job``
    (the gate ladder) + ``run_job(profile=...)`` (identity bind, tool-surface
    assert, lazy-install disable) + ``_deliver_exec_profile_result`` (synthesize
    then assert the pinned target).
  * ``cron/jobs.py`` — create/update-time validation (defense in depth).

The ``run_job`` profile path is driven END-TO-END against a stubbed ``AIAgent``
(+ a stubbed provider resolver) so the identity bind, the toolset override, the
``_skip_mcp_refresh`` flag, and the run-scoped lazy-install override are all
observed from inside the stub's ``run_conversation`` — not asserted vacuously.

AC map (PRD-027 testable-in-repo set): 003, 004a/b/c/d/e, 005a, 005b, 006, 008,
009, 011, 012. Rejection outcome strings use the ``error: <reason>`` shape that
the audit redactor can mangle — those cases assert run-refusal BEHAVIOR
(``run_job`` not called, ``mark_job_run(False)``), never the exact reason text.
The stable tokens (``suppressed_quiet_hours`` / ``suppressed_busy`` /
``budget_exceeded`` / ``delivery_failed`` / ``ok``) are asserted directly.

Run:
    source venv/bin/activate && \
      python -m pytest tests/cron/test_exec_profile.py -q
"""

from __future__ import annotations

import importlib
import types

import pytest

import cron.scheduler as s
import gateway.status as gstatus
import run_agent
import hermes_cli.runtime_provider as rp
from agent.turn_context import TurnContext, build_turn_context
from autonomy import audit
from autonomy import budget
from autonomy import exec_profile as ep
from autonomy import killswitch
from autonomy import run_identity
from autonomy.exec_profile import (
    ExecProfileError,
    PinnedTarget,
    PROACTIVE_READ,
    assert_tool_surface,
    get_exec_profile,
    in_quiet_window,
    is_silent_suppression,
    known_profile_names,
    resolve_pinned_target,
    resolve_quiet_hours,
)
from autonomy.run_identity import (
    PROACTIVE,
    classify_run,
    run_identity_scope,
)
from tools import approval
from tools.lazy_deps import _allow_lazy_installs, lazy_install_override


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_IDENTITY_MARKERS = (
    "HERMES_CRON_SESSION",
    "HERMES_AUTONOMOUS",
    "HERMES_INTERACTIVE",
    "HERMES_EXEC_ASK",
    "HERMES_GATEWAY_SESSION",
    "HERMES_SESSION_PLATFORM",
    "HERMES_YOLO_MODE",
    "HERMES_CONTAINERIZED",
)


@pytest.fixture(autouse=True)
def _clean_identity(monkeypatch):
    """Deterministic, isolated run-identity state for every test (mirrors the
    PRD-044 suite): clear every process-global marker, and guarantee no
    contextvar binding survives into the next test."""
    for name in _IDENTITY_MARKERS:
        monkeypatch.delenv(name, raising=False)
    yield
    run_identity._RUN_IDENTITY.set(None)


# A concrete pin used across the delivery + gate tests.
_PIN = PinnedTarget("discord", "1234")
_PIN_CFG = {"autonomy": {"proactive": {"pinned_target": "discord:1234"}}}


def _patch_happy_gates(monkeypatch, *, quiet=False, active=0, killswitch_engaged=False):
    """Make gates 3–6 pass so a gate/execution test reaches the case it targets.

    ``_run_exec_profile_job`` does *local* ``from autonomy import ...`` /
    ``from gateway.status import ...``, which read the source-module attribute at
    call time — so patching the module object here is seen inside the function.
    """
    monkeypatch.setattr(ep, "resolve_pinned_target", lambda cfg=None: _PIN)
    monkeypatch.setattr(ep, "is_quiet_now", lambda cfg=None: quiet)
    monkeypatch.setattr(killswitch, "guard", lambda surface: killswitch_engaged)
    monkeypatch.setattr(gstatus, "read_runtime_status", lambda: {"active_agents": active})


def _stub_marks(monkeypatch):
    """Record ``mark_job_run`` / ``save_job_output`` calls (both write to disk /
    state.db in production)."""
    marks = []
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None: marks.append((jid, ok)),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    return marks


def _proactive_records():
    return [r for r in audit.read_all() if r.get("surface") == "proactive"]


def _last_proactive_outcome():
    recs = _proactive_records()
    assert recs, "no proactive audit record written"
    return recs[-1]["outcome"]


# ===========================================================================
# AC-009 — resolve_pinned_target (unit): only platform:chat_id, thread-free
# ===========================================================================


def test_ac009_resolve_pinned_target_happy_path():
    pin = resolve_pinned_target(_PIN_CFG)
    assert pin.as_tuple() == ("discord", "1234", None)
    assert pin.thread_id is None


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "discord", "discord:1234:77", ":1234", "discord:", "  :  "],
    ids=["blank", "whitespace", "no_colon", "thread_segment", "empty_platform",
         "empty_chat", "both_empty"],
)
def test_ac009_resolve_pinned_target_malformed_raises(raw):
    with pytest.raises(ExecProfileError):
        resolve_pinned_target({"autonomy": {"proactive": {"pinned_target": raw}}})


def test_ac009_pinned_target_as_tuple_lowercases_platform():
    assert PinnedTarget("Discord", "9").as_tuple() == ("discord", "9", None)


# ===========================================================================
# AC-004b/c — assert_tool_surface: exact-only, fail closed, never degrade
# ===========================================================================


def _agent_with_tools(*names):
    return types.SimpleNamespace(valid_tool_names=set(names))


def test_ac004b_assert_tool_surface_exact_pass():
    agent = _agent_with_tools("session_search", "chronicle_search")
    # Exactly the allowlist → no raise.
    assert_tool_surface(agent, PROACTIVE_READ)


def test_ac004b_assert_tool_surface_extra_tool_raises():
    agent = _agent_with_tools("session_search", "chronicle_search", "terminal")
    with pytest.raises(ExecProfileError) as exc:
        assert_tool_surface(agent, PROACTIVE_READ)
    assert "terminal" in str(exc.value)


def test_ac004b_assert_tool_surface_missing_tool_raises_provider_absent():
    """With provider memory inactive the surface resolves to just
    {session_search} (chronicle_search never injected) → fail-closed abort."""
    agent = _agent_with_tools("session_search")
    with pytest.raises(ExecProfileError) as exc:
        assert_tool_surface(agent, PROACTIVE_READ)
    assert "chronicle_search" in str(exc.value)


@pytest.mark.parametrize("bridge", ["tool_search", "tool_describe", "tool_call"])
def test_ac004c_assert_tool_surface_bridge_tool_raises(bridge):
    agent = _agent_with_tools("session_search", "chronicle_search", bridge)
    with pytest.raises(ExecProfileError) as exc:
        assert_tool_surface(agent, PROACTIVE_READ)
    assert "bridge" in str(exc.value).lower()


def test_ac004d_assert_tool_surface_mcp_name_raises():
    """A pre-registered MCP tool name leaking into the resolved surface is an
    extra → fail-closed abort (profile runs skip MCP init, but any leak trips)."""
    agent = _agent_with_tools("session_search", "chronicle_search", "mcp__srv__do_thing")
    with pytest.raises(ExecProfileError) as exc:
        assert_tool_surface(agent, PROACTIVE_READ)
    assert "mcp__srv__do_thing" in str(exc.value)


# ===========================================================================
# AC-004e — lazy_install_override: run-scoped ContextVar, isolated across copies
# ===========================================================================


def test_ac004e_lazy_override_disables_inside_restores_after():
    assert _allow_lazy_installs() is True
    with lazy_install_override(False):
        assert _allow_lazy_installs() is False
    # Restored on exit.
    assert _allow_lazy_installs() is True


def test_ac004e_lazy_override_isolated_across_contexts():
    """Two independent contextvars.copy_context() runs: one enters the override,
    the other never does → a concurrent non-profile job in the shared cron pool
    is unaffected (both sides asserted)."""
    import contextvars

    results = {}

    def _inside_override():
        with lazy_install_override(False):
            results["inside"] = _allow_lazy_installs()

    def _no_override():
        results["outside"] = _allow_lazy_installs()

    contextvars.copy_context().run(_inside_override)
    contextvars.copy_context().run(_no_override)

    assert results["inside"] is False   # profile run: disabled
    assert results["outside"] is True   # concurrent plain job: unaffected


# ===========================================================================
# AC-005b — quiet-hours window math (unit)
# ===========================================================================


@pytest.mark.parametrize(
    "now_min,start,end,expected",
    [
        (23 * 60, "22:00", "08:00", True),    # inside a midnight-wrapping window
        (2 * 60, "22:00", "08:00", True),     # after midnight, still quiet
        (12 * 60, "22:00", "08:00", False),   # midday, outside
        (10 * 60, "09:00", "17:00", True),    # inside a same-day window
        (18 * 60, "09:00", "17:00", False),   # after a same-day window
    ],
    ids=["wrap-late", "wrap-early", "midday", "sameday-in", "sameday-out"],
)
def test_ac005b_in_quiet_window(now_min, start, end, expected):
    assert in_quiet_window(now_min, start, end) is expected


def test_ac005b_zero_width_window_is_disabled():
    # start == end → quiet hours disabled → never quiet.
    assert in_quiet_window(9 * 60, "09:00", "09:00") is False


def test_ac005b_malformed_window_fails_open():
    # A typo must not permanently silence outreach → fail OPEN (not quiet).
    assert in_quiet_window(60, "9foo", "08:00") is False
    assert in_quiet_window(60, "22:00", "99:99") is False


def test_ac005b_resolve_quiet_hours_defaults():
    assert resolve_quiet_hours({}) == ("22:00", "08:00")
    assert resolve_quiet_hours(
        {"autonomy": {"proactive": {"quiet_hours": {"start": "01:00", "end": "02:30"}}}}
    ) == ("01:00", "02:30")


# ===========================================================================
# AC-004c — turn_context re-assert hook fires the attached closure
# ===========================================================================


class _FakeTodoStore:
    def has_items(self):
        return True

    def _hydrate(self, *_a, **_k):
        pass


class _FakeGuardrails:
    def __init__(self):
        self.reset_called = False

    def reset_for_turn(self):
        self.reset_called = True


class _FakeTurnAgent:
    """Minimal stand-in covering only what build_turn_context touches (copied
    from tests/agent/test_turn_context.py's proven fake)."""

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cron"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = {"session_search", "chronicle_search"}
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._skip_mcp_refresh = True
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(protect_first_n=2, protect_last_n=2)
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._persist_calls = 0

    def _ensure_db_session(self):
        pass

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, *_a, **_k):
        self._persist_calls += 1


@pytest.fixture
def _stub_runtime_main(monkeypatch):
    monkeypatch.setattr("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None)


def _build_turn(agent):
    return build_turn_context(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )


def test_ac004c_turn_context_reassert_hook_fires(_stub_runtime_main):
    agent = _FakeTurnAgent()
    calls = []
    agent._exec_profile_tool_assert = lambda: calls.append(1)
    ctx = _build_turn(agent)
    assert isinstance(ctx, TurnContext)
    assert calls == [1]  # the per-turn re-assert fired exactly once


def test_ac004c_turn_context_reassert_propagates_raise(_stub_runtime_main):
    agent = _FakeTurnAgent()

    def _boom():
        raise ExecProfileError("mid-run tool surface drift")

    agent._exec_profile_tool_assert = _boom
    # A mismatch MUST abort the turn (deliberately outside the best-effort try).
    with pytest.raises(ExecProfileError):
        _build_turn(agent)


def test_ac004c_turn_context_no_hook_when_absent(_stub_runtime_main):
    agent = _FakeTurnAgent()
    # No closure attached (ordinary job) → prologue completes normally.
    assert getattr(agent, "_exec_profile_tool_assert", None) is None
    ctx = _build_turn(agent)
    assert isinstance(ctx, TurnContext)


# ===========================================================================
# AC-004 / AC-011 — the REAL run_job(profile=...) contained path, end-to-end
# ===========================================================================


class _StubAgent:
    """Records constructor toolset args + captures run-scoped state from inside
    ``run_conversation`` (where the profile bindings are actually live)."""

    captured: dict = {}
    valid_tool_names_override = {"session_search", "chronicle_search"}
    run_conversation_called = False

    def __init__(self, **kwargs):
        type(self).captured = {
            "enabled_toolsets": kwargs.get("enabled_toolsets"),
            "disabled_toolsets": kwargs.get("disabled_toolsets"),
        }
        self.valid_tool_names = set(type(self).valid_tool_names_override)

    def get_activity_summary(self):
        return {"seconds_since_activity": 0.0}

    def run_conversation(self, prompt):
        cls = type(self)
        cls.run_conversation_called = True
        cls.captured["identity_inside"] = classify_run().identity
        cls.captured["floor_inside"] = classify_run().unattended_floor
        cls.captured["lazy_inside"] = _allow_lazy_installs()
        cls.captured["skip_mcp"] = getattr(self, "_skip_mcp_refresh", None)
        cls.captured["has_assert"] = hasattr(self, "_exec_profile_tool_assert")
        return {"final_response": "one idle observation", "messages": []}


def _drive_profile_run_job(monkeypatch, agent_cls):
    fake_runtime = {
        "provider": "openrouter", "api_key": "sk-x",
        "base_url": "http://x/v1", "api_mode": "chat_completions",
    }
    monkeypatch.setattr(rp, "resolve_runtime_provider", lambda **k: dict(fake_runtime))
    monkeypatch.setattr(run_agent, "AIAgent", agent_cls)
    agent_cls.captured = {}
    agent_cls.run_conversation_called = False
    job = {
        "id": "pj-int", "name": "proactive idle",
        "prompt": "anything worth surfacing?", "exec_profile": "proactive_read",
        # A job-level toolset the profile must OVERRIDE:
        "enabled_toolsets": ["terminal", "browser"],
    }
    return s.run_job(job, profile=PROACTIVE_READ)


def test_ac004_run_job_profile_overrides_toolsets_and_contains(monkeypatch):
    success, _doc, final, error = _drive_profile_run_job(monkeypatch, _StubAgent)

    assert success is True and error is None
    assert final == "one idle observation"
    cap = _StubAgent.captured
    # Profile toolsets OVERRIDE the job-level enabled_toolsets.
    assert cap["enabled_toolsets"] == list(PROACTIVE_READ.enabled_toolsets)
    assert cap["disabled_toolsets"] == list(PROACTIVE_READ.disabled_toolsets)
    # _skip_mcp_refresh set + re-assert closure attached, observed on the agent.
    assert cap["skip_mcp"] is True
    assert cap["has_assert"] is True
    # Lazy installs disabled for the run scope (seen INSIDE the worker).
    assert cap["lazy_inside"] is False
    # No leak once the run returns.
    assert _allow_lazy_installs() is True
    assert classify_run().identity != PROACTIVE


def test_ac011_run_job_binds_proactive_identity_inside_run(monkeypatch):
    _drive_profile_run_job(monkeypatch, _StubAgent)
    cap = _StubAgent.captured
    # classify_run() inside the contained run is proactive, unattended floor on.
    assert cap["identity_inside"] == PROACTIVE
    assert cap["floor_inside"] is True
    # Identity binding reset cleanly (no residue into the next run).
    assert run_identity.bound_identity() is None


def test_ac011_run_job_identity_is_proactive_even_under_frozen_yolo(monkeypatch):
    """A frozen HERMES_YOLO_MODE env cannot re-classify the contained run as a
    YOLO auto-approve: the bound proactive marker wins for classify_run()."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    _drive_profile_run_job(monkeypatch, _StubAgent)
    assert _StubAgent.captured["identity_inside"] == PROACTIVE
    assert _StubAgent.captured["floor_inside"] is True


class _BadSurfaceAgent(_StubAgent):
    valid_tool_names_override = {"session_search", "chronicle_search", "terminal"}


def test_ac004b_run_job_aborts_fail_closed_on_surface_mismatch(monkeypatch):
    """The constructed agent exposes an EXTRA tool → assert_tool_surface raises
    inside run_job → failure tuple, run_conversation NEVER reached."""
    success, _doc, _final, error = _drive_profile_run_job(monkeypatch, _BadSurfaceAgent)

    assert success is False
    assert "ExecProfileError" in error
    assert _BadSurfaceAgent.run_conversation_called is False
    # Fail-closed abort still unwinds identity + lazy override (no leak).
    assert _allow_lazy_installs() is True
    assert run_identity.bound_identity() is None


# ===========================================================================
# AC-011 — bind-seam: profile identity is proactive + markers beat frozen YOLO
# ===========================================================================


def test_ac011_profile_declares_proactive_floor_identity():
    assert PROACTIVE_READ.identity == run_identity.PROACTIVE
    with run_identity_scope(PROACTIVE_READ.identity):
        ri = classify_run()
        assert ri.identity == PROACTIVE
        assert ri.unattended_floor is True
        assert ri.attended is False


@pytest.fixture
def guard_env(monkeypatch):
    """Deterministic approval-gate config (mirrors the PRD-044 smart_env):
    cron_mode=deny, not contained, no pre-approval, tirith=allow."""
    monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "deny")
    monkeypatch.setattr(approval, "_is_contained_ok", lambda cmd: False)
    monkeypatch.setattr(approval, "_local_exec_is_contained", lambda: False)
    monkeypatch.setattr(approval, "_command_matches_permanent_allowlist", lambda cmd: False)
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    import tools.tirith_security as ts
    monkeypatch.setattr(
        ts, "check_command_security",
        lambda command: {"action": "allow", "findings": [], "summary": ""},
    )
    approval.clear_session(approval.get_current_session_key())
    yield monkeypatch
    approval.clear_session(approval.get_current_session_key())


_DANGEROUS_NOT_HARDLINE = "rm -rf ./build"


def test_ac011_frozen_yolo_cannot_bypass_bound_proactive(guard_env):
    """The unattended floor a proactive run binds must survive a frozen YOLO
    env — otherwise an overnight proactive job would auto-approve everything."""
    monkeypatch = guard_env
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)

    with run_identity_scope(PROACTIVE):
        res = approval.check_all_command_guards(_DANGEROUS_NOT_HARDLINE, "local")

    assert res["approved"] is False, res
    assert "BLOCKED" in (res.get("message") or "")


# ===========================================================================
# AC-004a — run_one_job dispatch + entry rejections (no execution)
# ===========================================================================


def test_ac004_run_one_job_dispatches_exec_profile_jobs(monkeypatch):
    seen = {}

    def _fake_profile_job(job, adapters=None, loop=None, verbose=False):
        seen["job"] = job
        return True

    monkeypatch.setattr(s, "_run_exec_profile_job", _fake_profile_job)
    ok = s.run_one_job({"id": "d1", "name": "p", "exec_profile": "proactive_read"})
    assert ok is True
    assert seen["job"]["id"] == "d1"


def _run_job_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        s, "run_job",
        lambda job, profile=None: calls.append((job.get("id"), profile)) or (True, "", "", None),
    )
    return calls


def test_ac004a_entry_rejects_unknown_profile(monkeypatch):
    calls = _run_job_spy(monkeypatch)
    marks = _stub_marks(monkeypatch)
    job = {"id": "u1", "name": "bad", "exec_profile": "does_not_exist"}

    assert s._run_exec_profile_job(job) is True
    assert calls == []                      # never executed
    assert marks == [("u1", False)]         # marked failed
    assert _proactive_records()             # audited under proactive surface


def test_ac004a_entry_rejects_script(monkeypatch):
    calls = _run_job_spy(monkeypatch)
    marks = _stub_marks(monkeypatch)
    job = {"id": "sc1", "name": "p", "exec_profile": "proactive_read", "script": "watchdog.sh"}

    assert s._run_exec_profile_job(job) is True
    assert calls == []
    assert marks == [("sc1", False)]


def test_ac004a_entry_rejects_no_agent(monkeypatch):
    calls = _run_job_spy(monkeypatch)
    marks = _stub_marks(monkeypatch)
    job = {"id": "na1", "name": "p", "exec_profile": "proactive_read",
           "no_agent": True, "script": "x.sh"}

    assert s._run_exec_profile_job(job) is True
    assert calls == []
    assert marks == [("na1", False)]


# ===========================================================================
# AC-006 — kill switch silences the loop (no run, no mark; guard self-audits)
# ===========================================================================


def test_ac006_killswitch_engaged_no_run(monkeypatch):
    _patch_happy_gates(monkeypatch, killswitch_engaged=True)
    calls = _run_job_spy(monkeypatch)
    marks = _stub_marks(monkeypatch)
    job = {"id": "k1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    assert calls == []          # never executed
    assert marks == []          # kill-switch skip does not mark (self-audited)


# ===========================================================================
# AC-009 (gate) — missing/blank pin refuses to run (fail closed)
# ===========================================================================


def test_ac009_missing_pin_refuses_run(monkeypatch):
    monkeypatch.setattr(killswitch, "guard", lambda surface: False)
    monkeypatch.setattr(
        ep, "resolve_pinned_target",
        lambda cfg=None: (_ for _ in ()).throw(ExecProfileError("blank pin")),
    )
    calls = _run_job_spy(monkeypatch)
    marks = _stub_marks(monkeypatch)
    job = {"id": "p1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    assert calls == []
    assert marks == [("p1", False)]


# ===========================================================================
# AC-005b (gate) — quiet hours suppresses; AC-001/idle — busy suppresses
# ===========================================================================


def test_ac005b_quiet_hours_suppresses_no_run(monkeypatch):
    _patch_happy_gates(monkeypatch, quiet=True)
    calls = _run_job_spy(monkeypatch)
    marks = _stub_marks(monkeypatch)
    job = {"id": "q1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    assert calls == []                    # not executed
    assert marks == []                    # suppression does not mark
    assert _last_proactive_outcome() == "suppressed_quiet_hours"


def test_ac001_busy_gateway_suppresses_no_run(monkeypatch):
    _patch_happy_gates(monkeypatch, active=2)
    calls = _run_job_spy(monkeypatch)
    marks = _stub_marks(monkeypatch)
    job = {"id": "b1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    assert calls == []
    assert marks == []
    assert _last_proactive_outcome() == "suppressed_busy"


# ===========================================================================
# AC-005a (gate) — over-budget suppresses before composition
# ===========================================================================


def test_ac005a_over_budget_suppresses_no_run(monkeypatch):
    _patch_happy_gates(monkeypatch)
    monkeypatch.setattr(budget, "check", lambda kind, amount=1: False)
    calls = _run_job_spy(monkeypatch)
    marks = _stub_marks(monkeypatch)
    job = {"id": "bud1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    assert calls == []
    assert marks == []
    assert _last_proactive_outcome() == "budget_exceeded"


# ===========================================================================
# AC-003 / AC-008 — execution path: [SILENT] exact vs. delivery + audit outcome
# ===========================================================================


def _stub_run_job_returns(monkeypatch, final):
    monkeypatch.setattr(s, "run_job", lambda job, profile=None: (True, "doc", final, None))


def test_ac003_exact_silent_skips_delivery_outcome_ok(monkeypatch):
    _patch_happy_gates(monkeypatch)
    _stub_marks(monkeypatch)
    _stub_run_job_returns(monkeypatch, "[SILENT]")

    def _must_not_deliver(*a, **k):
        raise AssertionError("delivery must not be attempted for exact [SILENT]")

    monkeypatch.setattr(s, "_deliver_exec_profile_result", _must_not_deliver)
    job = {"id": "sil1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    assert _last_proactive_outcome() == "ok"
    # No proactive-message budget debited on a suppressed run.
    assert budget.get_usage()["totals"]["proactive_messages"] == 0


def test_ac003_message_mentioning_silent_midtext_is_delivered(monkeypatch):
    """Trailing-sentinel: a real message that merely MENTIONS the marker mid-text
    (marker is NOT the terminal token) IS delivered — only a trailing [SILENT]
    suppresses."""
    _patch_happy_gates(monkeypatch)
    _stub_marks(monkeypatch)
    _stub_run_job_returns(monkeypatch, "here is a note mentioning [SILENT] inline")

    delivered = []
    monkeypatch.setattr(s, "_deliver_result",
                        lambda dj, content, adapters=None, loop=None, **_kw: delivered.append(dj["deliver"]) or None)
    job = {"id": "sil2", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    assert delivered == ["discord:1234"]          # routed only to the pin
    assert _last_proactive_outcome() == "ok"
    assert budget.get_usage()["totals"]["proactive_messages"] == 1


def test_ac003_narrated_then_trailing_silent_suppresses(monkeypatch):
    """The live-observed bug (2026-07-06): the local model NARRATES its reasoning
    then appends [SILENT] on a final line. Trailing-sentinel suppresses it — the
    deliberation blob is NOT delivered and nothing is debited (exact-match wrongly
    delivered this)."""
    _patch_happy_gates(monkeypatch)
    _stub_marks(monkeypatch)
    _stub_run_job_returns(
        monkeypatch,
        "Looking at recent context, nothing here meets the bar for reaching out.\n\n[SILENT]",
    )

    def _must_not_deliver(*a, **k):
        raise AssertionError("a narrated-then-[SILENT] response must be suppressed, not delivered")

    monkeypatch.setattr(s, "_deliver_exec_profile_result", _must_not_deliver)
    job = {"id": "sil3", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    assert _last_proactive_outcome() == "ok"
    assert budget.get_usage()["totals"]["proactive_messages"] == 0


@pytest.mark.parametrize(
    "content,expect_suppress",
    [
        ("[SILENT]", True),                                   # bare marker
        ("  [SILENT]  ", True),                               # whitespace-padded
        ("reasoning...\n\n[SILENT]", True),                   # narrated then trailing marker
        ("line one\nline two\n[SILENT]\n", True),             # trailing marker after blank tail
        ("here is a real message mentioning [SILENT] mid-text", False),  # mention, not terminal
        ("[SILENT] then more content", False),                # marker first, content after
        ("a genuinely useful nudge", False),                  # ordinary message
        ("", False),                                          # empty
    ],
)
def test_is_silent_suppression(content, expect_suppress):
    assert is_silent_suppression(content) is expect_suppress


def test_ac008_successful_delivery_audits_proactive_ok(monkeypatch):
    _patch_happy_gates(monkeypatch)
    _stub_marks(monkeypatch)
    _stub_run_job_returns(monkeypatch, "a genuinely useful nudge")
    monkeypatch.setattr(s, "_deliver_result",
                        lambda dj, content, adapters=None, loop=None, **_kw: None)
    job = {"id": "ok1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    rec = _proactive_records()[-1]
    assert rec["surface"] == "proactive"
    assert rec["tier"] == "T3"
    assert rec["outcome"] == "ok"
    # Debited exactly once on a confirmed send (AC-005a).
    assert budget.get_usage()["totals"]["proactive_messages"] == 1


def test_ac008_forced_delivery_failure_audits_delivery_failed(monkeypatch):
    _patch_happy_gates(monkeypatch)
    _stub_marks(monkeypatch)
    _stub_run_job_returns(monkeypatch, "a nudge that fails to send")
    monkeypatch.setattr(
        s, "_deliver_result",
        lambda dj, content, adapters=None, loop=None, **_kw: "platform 'discord' not configured/enabled",
    )
    job = {"id": "df1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    assert _last_proactive_outcome() == "delivery_failed"
    # NO proactive-message debit when the send failed (AC-005a debit-on-send).
    assert budget.get_usage()["totals"]["proactive_messages"] == 0


# ===========================================================================
# AC-009 — delivery pin: synthesize (raw form has no effect) + assert backstop
# ===========================================================================


@pytest.mark.parametrize(
    "raw_job",
    [
        {"deliver": "origin", "origin": "discord:99999"},
        {"deliver": "discord:99999"},
        {"deliver": "discord:1234:77"},
        {"deliver": "discord"},
        {"deliver": "all"},
        {"deliver": "telegram:5,discord:6"},
        {"deliver": ["telegram", "discord"]},
    ],
    ids=["origin", "other-chat", "pin-with-thread", "bare-platform",
         "all", "comma-multi", "list-multi"],
)
def test_ac009_synthesize_routes_only_to_pin(monkeypatch, raw_job):
    """Every raw ``deliver`` form is IGNORED — delivery is synthesized from the
    pin (deliver=pin, origin cleared) before the resolver runs."""
    captured = {}
    monkeypatch.setattr(
        s, "_deliver_result",
        lambda dj, content, adapters=None, loop=None, preresolved_targets=None:
            captured.update(deliver=dj.get("deliver"), origin=dj.get("origin"),
                            preresolved=preresolved_targets) or None,
    )
    job = {"id": "syn", "name": "p", "exec_profile": "proactive_read", **raw_job}

    err = s._deliver_exec_profile_result(job, PROACTIVE_READ, _PIN, "hello")

    assert err is None
    assert captured["deliver"] == "discord:1234"   # only the pin, regardless of raw
    assert captured["origin"] is None              # origin fallback path closed
    # NF-3: the asserted single target is handed straight to the send (resolve-once).
    assert captured["preresolved"] == [{"platform": "discord", "chat_id": "1234", "thread_id": None}]


@pytest.mark.parametrize(
    "resolved,label",
    [
        ([{"platform": "discord", "chat_id": "9999", "thread_id": None}], "non-pin-chat"),
        ([{"platform": "discord", "chat_id": "1234", "thread_id": "77"}], "thread-attached"),
        ([{"platform": "discord", "chat_id": "1234", "thread_id": None},
          {"platform": "discord", "chat_id": "5678", "thread_id": None}], "multi-target"),
        ([], "blank-resolution"),
    ],
    ids=["non-pin-chat", "thread-attached", "multi-target", "blank-resolution"],
)
def test_ac009_assert_backstop_refuses_non_pin(monkeypatch, resolved, label):
    """The choke-point assert refuses any concrete resolution that is not
    EXACTLY the pin (extra chat, attached thread, multi-target, or empty) →
    error string, NO delivery."""
    monkeypatch.setattr(s, "_resolve_delivery_targets", lambda dj: resolved)

    def _must_not_deliver(*a, **k):
        raise AssertionError("delivery must be refused before _deliver_result")

    monkeypatch.setattr(s, "_deliver_result", _must_not_deliver)
    job = {"id": "bk", "name": "p", "exec_profile": "proactive_read"}

    err = s._deliver_exec_profile_result(job, PROACTIVE_READ, _PIN, "hello")
    assert isinstance(err, str) and "fail closed" in err


# ===========================================================================
# AC-012 — regression: ordinary jobs never touch the profile gates
# ===========================================================================


def _patch_ordinary_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(s, "run_job", lambda job: calls.append(("run_job", job["id"])) or (True, "out", "final", None))
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: calls.append(("save", jid)) or f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", lambda job, content, adapters=None, loop=None: calls.append(("deliver", job["id"])) or None)
    monkeypatch.setattr(s, "mark_job_run", lambda jid, ok, err=None, delivery_error=None: calls.append(("mark", jid, ok)))

    def _explode(*a, **k):
        raise AssertionError("ordinary job must not enter the exec_profile path")

    monkeypatch.setattr(s, "_run_exec_profile_job", _explode)
    return calls


def test_ac012_ordinary_no_agent_job_unaffected(monkeypatch):
    calls = _patch_ordinary_pipeline(monkeypatch)
    job = {"id": "reg1", "name": "watchdog", "no_agent": True,
           "script": "x.sh", "deliver": "local"}  # NO exec_profile

    assert s.run_one_job(job) is True
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "reg1", True)


def test_ac012_ordinary_agent_job_unaffected(monkeypatch):
    calls = _patch_ordinary_pipeline(monkeypatch)
    job = {"id": "reg2", "name": "reflection", "prompt": "reflect",
           "enabled_toolsets": ["memory"], "deliver": "local"}  # NO exec_profile

    assert s.run_one_job(job) is True
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "reg2", True)


# ===========================================================================
# AC-004a — cron/jobs.py create/update-time validation (defense in depth)
# ===========================================================================


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME and reload the modules that cache get_hermes_home()
    at import time (mirrors tests/cron/test_cron_no_agent.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)

    return home


def test_ac004a_create_job_accepts_known_profile(hermes_env):
    from cron.jobs import create_job

    job = create_job(prompt="any idle thoughts?", schedule="every 30m",
                     exec_profile="proactive_read", deliver="local")
    assert job["exec_profile"] == "proactive_read"
    assert job.get("no_agent") is False


def test_ac004a_create_job_rejects_unknown_profile(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="Unknown exec_profile"):
        create_job(prompt="p", schedule="every 30m", exec_profile="nope", deliver="local")


def test_ac004a_create_job_rejects_profile_with_script(hermes_env):
    from cron.jobs import create_job

    script = hermes_env / "scripts" / "w.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    with pytest.raises(ValueError, match="must not set script or no_agent"):
        create_job(prompt="p", schedule="every 30m", exec_profile="proactive_read",
                   script="w.sh", deliver="local")


def test_ac004a_create_job_rejects_profile_with_no_agent(hermes_env):
    from cron.jobs import create_job

    script = hermes_env / "scripts" / "w.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    with pytest.raises(ValueError, match="must not set script or no_agent"):
        create_job(prompt=None, schedule="every 30m", exec_profile="proactive_read",
                   no_agent=True, script="w.sh", deliver="local")


def test_ac004a_update_job_rejects_unknown_profile(hermes_env):
    from cron.jobs import create_job, update_job

    job = create_job(prompt="p", schedule="every 30m",
                     exec_profile="proactive_read", deliver="local")
    with pytest.raises(ValueError, match="Unknown exec_profile"):
        update_job(job["id"], {"exec_profile": "bogus"})


def test_ac004a_update_job_rejects_sneaking_script_into_profile(hermes_env):
    from cron.jobs import create_job, update_job

    job = create_job(prompt="p", schedule="every 30m",
                     exec_profile="proactive_read", deliver="local")
    with pytest.raises(ValueError, match="must not set script or no_agent"):
        update_job(job["id"], {"script": "sneaky.sh"})


def test_ac004a_update_job_rejects_sneaking_no_agent_into_profile(hermes_env):
    from cron.jobs import create_job, update_job

    job = create_job(prompt="p", schedule="every 30m",
                     exec_profile="proactive_read", deliver="local")
    with pytest.raises(ValueError, match="must not set script or no_agent"):
        update_job(job["id"], {"no_agent": True})


# ===========================================================================
# Security-review hardening (NF-1 text-only, NF-2 secret scan, NF-3 resolve-once,
# fail-closed kill switch, failed-run ledger-only, no-strip-containment)
# ===========================================================================


def test_nf1_media_directive_refuses_delivery(monkeypatch):
    """NF-1: a message carrying a MEDIA:<path> attachment tag is refused
    (exec_profile delivery is text-only — the pin bounds the recipient, not the
    payload, so a file exfil via attachment must fail closed)."""
    def _must_not_deliver(*a, **k):
        raise AssertionError("must refuse before _deliver_result on a MEDIA tag")

    monkeypatch.setattr(s, "_deliver_result", _must_not_deliver)
    job = {"id": "m1", "name": "p", "exec_profile": "proactive_read"}

    err = s._deliver_exec_profile_result(
        job, PROACTIVE_READ, _PIN, 'exfil MEDIA:"/opt/relay/client.token"',
    )
    assert isinstance(err, str) and "MEDIA" in err and "fail closed" in err


def test_nf2_credential_shape_refuses_delivery(monkeypatch):
    """NF-2: outbound content flagged by the egress classifier is refused
    (session_search returns un-redacted state.db content)."""
    import relay.egress_classifier as _cls

    monkeypatch.setattr(_cls, "contains_credential",
                        lambda text: (True, "credential_shape:test_key"))

    def _must_not_deliver(*a, **k):
        raise AssertionError("must refuse before _deliver_result on a credential")

    monkeypatch.setattr(s, "_deliver_result", _must_not_deliver)
    job = {"id": "c1", "name": "p", "exec_profile": "proactive_read"}

    err = s._deliver_exec_profile_result(job, PROACTIVE_READ, _PIN, "sk-secret-inline")
    assert isinstance(err, str) and "classifier" in err and "fail closed" in err


def test_nf2_classifier_exception_fails_closed(monkeypatch):
    """NF-2: a classifier that RAISES fails closed (refuse), never crashes open."""
    import relay.egress_classifier as _cls

    def _boom(text):
        raise RuntimeError("classifier down")

    monkeypatch.setattr(_cls, "contains_credential", _boom)
    monkeypatch.setattr(s, "_deliver_result",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not deliver")))
    job = {"id": "c2", "name": "p", "exec_profile": "proactive_read"}

    err = s._deliver_exec_profile_result(job, PROACTIVE_READ, _PIN, "hello")
    assert isinstance(err, str) and "classifier_error" in err


def test_failed_run_does_not_deliver_or_debit(monkeypatch):
    """Security review: a FAILED agent run is audited but never delivers a
    diagnostic summary to the pin and never debits the proactive cap."""
    _patch_happy_gates(monkeypatch)
    _stub_marks(monkeypatch)
    monkeypatch.setattr(s, "run_job",
                        lambda job, profile=None: (False, "doc", "", "boom: agent failed"))

    def _must_not_deliver(*a, **k):
        raise AssertionError("a failed proactive run must not deliver")

    monkeypatch.setattr(s, "_deliver_exec_profile_result", _must_not_deliver)
    job = {"id": "f1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True
    # Ledger-visible failure, no delivery, no debit.
    assert _proactive_records()[-1]["outcome"].startswith("error")
    assert budget.get_usage()["totals"]["proactive_messages"] == 0


def test_killswitch_exception_fails_closed(monkeypatch):
    """Security NOTE: an unreadable QUIESCE flag (guard raises) SUPPRESSES the
    proactive run — fail closed, not proceed."""
    _patch_happy_gates(monkeypatch)
    _stub_marks(monkeypatch)

    def _guard_boom(surface):
        raise OSError("cannot read QUIESCE flag")

    monkeypatch.setattr(killswitch, "guard", _guard_boom)

    def _must_not_run(*a, **k):
        raise AssertionError("must not run the agent when the kill-switch check errored")

    monkeypatch.setattr(s, "run_job", _must_not_run)
    job = {"id": "ks1", "name": "p", "exec_profile": "proactive_read"}

    assert s._run_exec_profile_job(job) is True  # cleanly suppressed


def test_update_cannot_strip_exec_profile(hermes_env):
    """Security NOTE: removing exec_profile via update (which would silently
    convert a contained job to an ordinary uncontained one) is refused."""
    from cron.jobs import create_job, update_job

    job = create_job(prompt="p", schedule="every 30m",
                     exec_profile="proactive_read", deliver="local")
    with pytest.raises(ValueError, match="cannot remove exec_profile"):
        update_job(job["id"], {"exec_profile": None})


# ===========================================================================
# Registry sanity
# ===========================================================================


def test_known_profile_names_and_lookup():
    assert "proactive_read" in known_profile_names()
    assert get_exec_profile("proactive_read") is PROACTIVE_READ
    assert get_exec_profile("nope") is None
    assert get_exec_profile(None) is None
