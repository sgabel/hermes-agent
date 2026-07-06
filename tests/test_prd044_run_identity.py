"""PRD-044 FR-5 identity regression suite.

Pins the ONE canonical run-identity classifier (``autonomy/run_identity.py``)
and its four consumers so a future edit that re-scatters the attended/unattended
env reads — or silently stops calling ``classify_run()`` at a gate site —
fails loudly here.

FR-5 matrix covered (one clearly-named test / parametrized group each):
  1. classifier precedence (incl. leaked-flag combos + contextvar override)
  2. YOLO x unattended-marker (STOP-3: markers beat YOLO; unmarked keeps -z)
  3. execute_code x HERMES_AUTONOMOUS (the named accepted delta)
  4. sticky-env regression (AC-010: run_identity_scope leaves no residue)
  5. reachability spies (MINOR-10: each consumer really calls classify_run)
  6. unmarked_legacy audit (labeled + WARNING-audited, still approved)
  7. ask_claude surface label + governed-unattended local-gate skip

Run:
    source venv/bin/activate && \
      python -m pytest tests/test_prd044_run_identity.py -q
"""

import os
from unittest import mock

import pytest

from autonomy import run_identity
from autonomy.run_identity import (
    classify_run,
    bind_run_identity,
    reset_run_identity,
    run_identity_scope,
    bound_identity,
    INTERACTIVE_CLI,
    GATEWAY_ATTENDED,
    CRON,
    ORCHESTRATED_HEADLESS,
    PROACTIVE,
    DELEGATED_CHILD,
    UNMARKED_LEGACY,
)
from tools import approval
from tools import capability_policy


# All process-global identity markers. The conftest hermetic fixture already
# clears most of these, but NOT HERMES_AUTONOMOUS / HERMES_CONTAINERIZED — and
# env markers are process-global, so we clear the full set here regardless.
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
    """Deterministic, isolated identity state for every test.

    - delenv every HERMES_* identity marker (process-global; must not leak in)
    - reset the ``_RUN_IDENTITY`` contextvar to unbound on teardown (belt-and-
      braces; the tests themselves use ``run_identity_scope`` / try-finally).
    """
    for name in _IDENTITY_MARKERS:
        monkeypatch.delenv(name, raising=False)
    yield
    # Defensive: guarantee no contextvar binding survives into the next test.
    run_identity._RUN_IDENTITY.set(None)


# =========================================================================
# Family 1 — classifier precedence
# =========================================================================

# (label, env-to-set, expected identity, attended, unattended_floor)
_PRECEDENCE_CASES = [
    ("no_markers", {}, UNMARKED_LEGACY, False, False),
    ("interactive", {"HERMES_INTERACTIVE": "1"}, INTERACTIVE_CLI, True, False),
    ("exec_ask", {"HERMES_EXEC_ASK": "1"}, INTERACTIVE_CLI, True, False),
    ("cron", {"HERMES_CRON_SESSION": "1"}, CRON, False, True),
    ("autonomous", {"HERMES_AUTONOMOUS": "1"}, ORCHESTRATED_HEADLESS, False, True),
    ("gateway_session", {"HERMES_GATEWAY_SESSION": "1"}, GATEWAY_ATTENDED, True, False),
    ("gateway_platform", {"HERMES_SESSION_PLATFORM": "discord"}, GATEWAY_ATTENDED, True, False),
    # Unattended beats leaked attended flags (the in-process cron/autonomous
    # thread carries the gateway HERMES_EXEC_ASK + cli HERMES_INTERACTIVE leak).
    (
        "cron_beats_leaked_attended",
        {"HERMES_CRON_SESSION": "1", "HERMES_INTERACTIVE": "1", "HERMES_EXEC_ASK": "1"},
        CRON,
        False,
        True,
    ),
    (
        "autonomous_beats_exec_ask",
        {"HERMES_AUTONOMOUS": "1", "HERMES_EXEC_ASK": "1"},
        ORCHESTRATED_HEADLESS,
        False,
        True,
    ),
    # Gateway signal is authoritative over the co-present interactive flag.
    (
        "gateway_beats_interactive",
        {"HERMES_GATEWAY_SESSION": "1", "HERMES_INTERACTIVE": "1"},
        GATEWAY_ATTENDED,
        True,
        False,
    ),
    # Unattended env marker still wins over an attended gateway platform.
    (
        "cron_beats_gateway_platform",
        {"HERMES_CRON_SESSION": "1", "HERMES_SESSION_PLATFORM": "discord"},
        CRON,
        False,
        True,
    ),
]


@pytest.mark.parametrize(
    "label,env,identity,attended,floor",
    _PRECEDENCE_CASES,
    ids=[c[0] for c in _PRECEDENCE_CASES],
)
def test_classify_precedence(monkeypatch, label, env, identity, attended, floor):
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    ri = classify_run()

    assert ri.identity == identity, label
    assert ri.attended is attended, label
    assert ri.unattended_floor is floor, label
    # Axis invariants: is_unattended is exactly "no human present".
    assert ri.is_unattended is (not attended), label
    assert ri.is_legacy is (identity == UNMARKED_LEGACY), label
    # unmarked_legacy is the distinct third state — BOTH axes False.
    if identity == UNMARKED_LEGACY:
        assert ri.attended is False and ri.unattended_floor is False


def test_source_field_provenance(monkeypatch):
    """``source`` is audit-only metadata: default / env / context."""
    assert classify_run().source == "default"
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    assert classify_run().source == "env"
    with run_identity_scope(GATEWAY_ATTENDED):
        assert classify_run().source == "context"


def test_contextvar_binding_overrides_env(monkeypatch):
    """Explicit run-context binding is highest precedence — it overrides even an
    unattended env marker."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    assert classify_run().identity == CRON  # env, no binding yet
    with run_identity_scope(GATEWAY_ATTENDED):
        ri = classify_run()
        assert ri.identity == GATEWAY_ATTENDED  # binding wins over env cron
        assert ri.attended is True
        assert ri.source == "context"
    # Binding released → back to the ambient env marker.
    assert classify_run().identity == CRON


@pytest.mark.parametrize("identity", [PROACTIVE, DELEGATED_CHILD])
def test_context_only_floor_identities(identity):
    """proactive / delegated_child are representable ONLY via a context bind
    (children share the parent's env byte-for-byte)."""
    with run_identity_scope(identity):
        ri = classify_run()
        assert ri.identity == identity
        assert ri.unattended_floor is True
        assert ri.attended is False
        assert ri.source == "context"


def test_bind_rejects_unknown_identity():
    """A typo'd bind must fail loud, never fall through to a laxer env verdict."""
    with pytest.raises(ValueError):
        bind_run_identity("root")


# =========================================================================
# Family 2 — YOLO x unattended-marker (STOP-3 regression guard)
# =========================================================================


@pytest.fixture
def guard_env(monkeypatch):
    """Deterministic approval-gate config: cron_mode=deny, not contained,
    no permanent/session pre-approval, tirith=allow. Mirrors the PRD-015
    smart_env fixture. Also clears session state before/after."""
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


# A recursive delete that is DANGEROUS but NOT hardline (no /, ~, $HOME target),
# so the YOLO-vs-floor precedence — not the unconditional hardline floor — is
# what the test exercises.
_DANGEROUS_NOT_HARDLINE = "rm -rf ./build"


def test_yolo_frozen_is_bypassed_by_unattended_env_marker(guard_env):
    """YOLO frozen ON + an unattended env marker → the floor still blocks.

    ``hermes -z`` sets HERMES_YOLO_MODE=1 unconditionally; without the
    markers-beat-YOLO rule an overnight run would auto-approve every dangerous
    command and the unattended floor would be dead code."""
    monkeypatch = guard_env
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    res = approval.check_all_command_guards(_DANGEROUS_NOT_HARDLINE, "local")

    assert res["approved"] is False, res
    assert "BLOCKED" in (res.get("message") or "")


def test_yolo_frozen_is_bypassed_by_bound_unattended_identity(guard_env):
    """Same guarantee via the authoritative wire format — a bound
    orchestrated_headless identity (env alone can't re-freeze YOLO)."""
    monkeypatch = guard_env
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)

    with run_identity_scope(ORCHESTRATED_HEADLESS):
        res = approval.check_all_command_guards(_DANGEROUS_NOT_HARDLINE, "local")

    assert res["approved"] is False, res


def test_yolo_frozen_still_bypasses_unmarked_legacy(guard_env):
    """YOLO ON + NO markers (unmarked_legacy) → YOLO bypass is UNCHANGED.

    unmarked_legacy has unattended_floor=False, so ``not _unattended`` is True and
    the plain -z bypass fires — this is the 'plain -z unchanged' half of STOP-3."""
    monkeypatch = guard_env
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)

    res = approval.check_all_command_guards(_DANGEROUS_NOT_HARDLINE, "local")

    assert res["approved"] is True, res


def test_check_dangerous_command_markers_beat_yolo(guard_env):
    """The same precedence holds in ``check_dangerous_command`` (the terminal
    per-call gate), not just the combined guard."""
    monkeypatch = guard_env
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    res = approval.check_dangerous_command(_DANGEROUS_NOT_HARDLINE, "local")

    assert res["approved"] is False, res


# =========================================================================
# Family 3 — execute_code x HERMES_AUTONOMOUS (the named accepted delta)
# =========================================================================


@pytest.fixture
def exec_guard_env(monkeypatch):
    """execute_code guard config: cron_mode=deny + not contained → the BLOCK
    path for governed-unattended runs."""
    monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "deny")
    monkeypatch.setattr(approval, "_local_exec_is_contained", lambda: False)
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    approval.clear_session(approval.get_current_session_key())
    yield monkeypatch
    approval.clear_session(approval.get_current_session_key())


_SCRIPT = "import os; os.system('id')"


def test_execute_code_autonomous_takes_unattended_floor(exec_guard_env):
    """THE ACCEPTED DELTA: pre-044 this guard keyed the floor off
    HERMES_CRON_SESSION only, so HERMES_AUTONOMOUS fell through to auto-approve.
    Post-044 orchestrated_headless now hits the same contained floor cron had."""
    monkeypatch = exec_guard_env
    monkeypatch.setenv("HERMES_AUTONOMOUS", "1")

    res = approval.check_execute_code_guard(_SCRIPT, "local")

    assert res["approved"] is False, res
    assert "BLOCKED" in (res.get("message") or "")


def test_execute_code_cron_takes_unattended_floor(exec_guard_env):
    """Control: cron already blocked here pre-044 — assert it still does, so the
    delta test above is measured against the identical floor, not a fluke."""
    monkeypatch = exec_guard_env
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    res = approval.check_execute_code_guard(_SCRIPT, "local")

    assert res["approved"] is False, res


def test_execute_code_autonomous_contained_allows(exec_guard_env):
    """Proves the block is the FLOOR (containment-gated), not a blanket deny:
    when the host is declared contained, the same AUTONOMOUS run is allowed."""
    monkeypatch = exec_guard_env
    monkeypatch.setattr(approval, "_local_exec_is_contained", lambda: True)
    monkeypatch.setenv("HERMES_AUTONOMOUS", "1")

    res = approval.check_execute_code_guard(_SCRIPT, "local")

    assert res["approved"] is True, res


def test_execute_code_unmarked_legacy_allows(exec_guard_env):
    """Contrast + no-breakage: with NO markers, local execute_code auto-approves
    (the attended/legacy contract is unchanged)."""
    res = approval.check_execute_code_guard(_SCRIPT, "local")
    assert res["approved"] is True, res


# =========================================================================
# Family 4 — sticky-env regression (AC-010)
# =========================================================================


def test_scope_leaves_no_residue_in_clean_env():
    """The scheduler pattern at the unit level: bind CRON, exit, classification
    returns to the ambient env (unmarked_legacy) and NO env marker leaks."""
    assert classify_run().identity == UNMARKED_LEGACY
    with run_identity_scope(CRON):
        assert classify_run().identity == CRON
        # The wire format is a contextvar, NOT process env — the old bug set
        # os.environ["HERMES_CRON_SESSION"] permanently.
        assert "HERMES_CRON_SESSION" not in os.environ
    assert classify_run().identity == UNMARKED_LEGACY
    assert bound_identity() is None


def test_scope_restores_ambient_env_marker(monkeypatch):
    """Exiting a CRON scope restores whatever the ambient env says — here an
    attended interactive session — not a wiped/blank state."""
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    assert classify_run().identity == INTERACTIVE_CLI
    with run_identity_scope(CRON):
        assert classify_run().identity == CRON
    assert classify_run().identity == INTERACTIVE_CLI


def test_bind_reset_nest_correctly():
    """Manual bind/reset must nest LIFO and unwind cleanly to unbound."""
    assert bound_identity() is None
    t1 = bind_run_identity(CRON)
    assert classify_run().identity == CRON
    t2 = bind_run_identity(INTERACTIVE_CLI)
    assert classify_run().identity == INTERACTIVE_CLI
    reset_run_identity(t2)
    assert classify_run().identity == CRON  # inner released → outer restored
    reset_run_identity(t1)
    assert classify_run().identity == UNMARKED_LEGACY
    assert bound_identity() is None


# =========================================================================
# Family 5 — reachability spies (MINOR-10: non-vacuous consumption)
# =========================================================================


def test_check_dangerous_command_calls_classify_run(monkeypatch):
    spy = mock.Mock(side_effect=run_identity.classify_run)
    # approval.py does `from autonomy.run_identity import classify_run` at module
    # top → the live reference is `tools.approval.classify_run`.
    monkeypatch.setattr(approval, "classify_run", spy)

    approval.check_dangerous_command("ls -la", "local")

    assert spy.call_count >= 1


def test_check_all_command_guards_calls_classify_run(monkeypatch):
    spy = mock.Mock(side_effect=run_identity.classify_run)
    monkeypatch.setattr(approval, "classify_run", spy)

    approval.check_all_command_guards("ls -la", "local")

    assert spy.call_count >= 1


def test_check_execute_code_guard_calls_classify_run(monkeypatch):
    spy = mock.Mock(side_effect=run_identity.classify_run)
    monkeypatch.setattr(approval, "classify_run", spy)

    approval.check_execute_code_guard("print('hi')", "local")

    assert spy.call_count >= 1


def test_capability_policy_is_unattended_calls_classify_run(monkeypatch):
    # capability_policy.is_unattended does a LAZY `from autonomy.run_identity
    # import classify_run` → patch the source module, not tools.approval.
    spy = mock.Mock(side_effect=run_identity.classify_run)
    monkeypatch.setattr(run_identity, "classify_run", spy)

    # ctx=None → no explicit override, so it must defer to the classifier.
    result = capability_policy.is_unattended(None)

    assert spy.call_count >= 1
    assert result is False  # unmarked_legacy → stays on the attended path


def test_capability_policy_ctx_override_short_circuits(monkeypatch):
    """The explicit ctx override stays HIGHEST precedence and does NOT consult
    the classifier (the cron_script gate fires before any binding exists)."""
    spy = mock.Mock(side_effect=run_identity.classify_run)
    monkeypatch.setattr(run_identity, "classify_run", spy)

    assert capability_policy.is_unattended({"unattended": True}) is True
    assert capability_policy.is_unattended({"unattended": False}) is False
    spy.assert_not_called()


# =========================================================================
# Family 6 — unmarked_legacy audit (labeled + audited, still approved)
# =========================================================================


def test_unmarked_legacy_dangerous_is_audited_and_approved(guard_env):
    """No markers, no human: today's auto-approve is preserved (no breakage) but
    each dangerous auto-approval writes a WARNING audit row so the hole is
    measured for the eventual fail-closed flip."""
    monkeypatch = guard_env
    audit_spy = mock.Mock()
    monkeypatch.setattr(approval, "_audit_unmarked_legacy_autoapprove", audit_spy)

    res = approval.check_dangerous_command(_DANGEROUS_NOT_HARDLINE, "local")

    assert res["approved"] is True, res
    assert audit_spy.call_count == 1
    # The audited command is the one that was auto-approved.
    assert audit_spy.call_args.args[0] == _DANGEROUS_NOT_HARDLINE


def test_unmarked_legacy_benign_not_audited(guard_env):
    """The audit fires only for DANGEROUS auto-approvals, not benign commands."""
    monkeypatch = guard_env
    audit_spy = mock.Mock()
    monkeypatch.setattr(approval, "_audit_unmarked_legacy_autoapprove", audit_spy)

    res = approval.check_all_command_guards("ls -la", "local")

    assert res["approved"] is True, res
    audit_spy.assert_not_called()


# =========================================================================
# Family 7 — ask_claude surface label + governed-unattended gate skip
# =========================================================================


@pytest.fixture
def relay_env(monkeypatch):
    """Make the ask_claude relay path exercisable in the unit env: relay
    'available', no secret in prompt, and a captured relay call."""
    from tools import claude_review_tool as crt
    monkeypatch.setattr(crt, "_relay_available", lambda: True)
    monkeypatch.setattr(crt, "_contains_secret", lambda text: False)
    relay = mock.Mock(return_value=(200, {"advisory_text": "ok"}))
    monkeypatch.setattr(crt, "_call_relay", relay)
    return crt, relay, monkeypatch


def test_ask_claude_attended_labels_surface_and_runs_local_gate(relay_env):
    crt, relay, monkeypatch = relay_env
    gate = mock.Mock(return_value={"approved": True})
    # ask_claude does `from tools.approval import check_all_command_guards`
    # at call time → patch the source attribute.
    monkeypatch.setattr(approval, "check_all_command_guards", gate)

    with run_identity_scope(INTERACTIVE_CLI):
        out = crt.ask_claude(prompt="is this plan sound?")

    # Attended → the interactive-only local approval/tirith gate runs.
    gate.assert_called_once()
    # Surface label is the classifier identity name.
    assert relay.call_count == 1
    assert relay.call_args.args[1] == INTERACTIVE_CLI
    assert isinstance(out, str)


def test_ask_claude_governed_unattended_skips_local_gate(relay_env):
    crt, relay, monkeypatch = relay_env
    gate = mock.Mock(return_value={"approved": True})
    monkeypatch.setattr(approval, "check_all_command_guards", gate)

    with run_identity_scope(CRON):
        crt.ask_claude(prompt="overnight review please")

    # Governed-unattended → the local gate is skipped (relay checks authoritative).
    gate.assert_not_called()
    assert relay.call_count == 1
    assert relay.call_args.args[1] == CRON


def test_ask_claude_orchestrated_headless_surface_and_skip(relay_env):
    crt, relay, monkeypatch = relay_env
    gate = mock.Mock(return_value={"approved": True})
    monkeypatch.setattr(approval, "check_all_command_guards", gate)

    with run_identity_scope(ORCHESTRATED_HEADLESS):
        crt.ask_claude(prompt="autonomous batch check")

    gate.assert_not_called()
    assert relay.call_args.args[1] == ORCHESTRATED_HEADLESS


# ---------------------------------------------------------------------------
# Review-fold: launcher wiring (N1 gateway bind) + scheduler restore (T4)
# ---------------------------------------------------------------------------


def test_gateway_runner_binds_and_resets_attended_identity(monkeypatch):
    """N1/MEDIUM-2 fix: the gateway per-turn wrapper binds GATEWAY_ATTENDED via
    the classifier contextvar so a concurrent in-process cron job's process-global
    HERMES_CRON_SESSION can't misclassify an attended gateway turn as cron.

    Simulates the concurrent-cron condition (env marker set) and asserts the
    bound identity wins DURING the turn and is cleanly reset AFTER.
    """
    from types import SimpleNamespace
    from gateway.run import GatewayRunner

    # A hashable platform stand-in (the real Platform is an enum used as a dict
    # key; SimpleNamespace defines __eq__ so it is unhashable — can't be a key).
    class _FakePlatform:
        value = "discord"

    # A concurrent cron job has set the process-global marker.
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    assert classify_run().identity == CRON  # ambient, no gateway binding yet

    runner = object.__new__(GatewayRunner)  # bare runner (no __init__), review pattern
    runner.adapters = {}
    ctx = SimpleNamespace(
        source=SimpleNamespace(
            platform=_FakePlatform(),
            chat_id="c1", chat_name="general", thread_id=None,
            user_id="u1", user_name="scott", message_id="m1",
        ),
        session_key="sess-1",
    )

    tokens = runner._set_session_env(ctx)
    try:
        # DURING the turn: the attended binding wins over the leaked cron env.
        ri = classify_run()
        assert ri.identity == GATEWAY_ATTENDED
        assert ri.attended is True
        assert ri.source == "context"
    finally:
        runner._clear_session_env(tokens)

    # AFTER the turn: binding reset; ambient cron env is visible again (would be
    # cleared by the cron job's own finally in production).
    assert bound_identity() is None
    assert classify_run().identity == CRON


def test_scheduler_env_restore_contract(monkeypatch):
    """T4: pin the scheduler's HERMES_CRON_SESSION save/restore invariant
    (cron/scheduler.py) — prior=None → popped; prior='1' → restored to '1'.

    Reproduces the exact bind→set→(work)→reset→restore sequence the scheduler's
    run_job try/finally performs, exercising the real run_identity bind/reset.
    """
    # Case A: no prior marker → restored to absent.
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    prior = os.environ.get("HERMES_CRON_SESSION")
    tok = run_identity.bind_run_identity(CRON)
    os.environ["HERMES_CRON_SESSION"] = "1"
    try:
        assert classify_run().identity == CRON
    finally:
        run_identity.reset_run_identity(tok)
        if prior is None:
            os.environ.pop("HERMES_CRON_SESSION", None)
        else:
            os.environ["HERMES_CRON_SESSION"] = prior
    assert "HERMES_CRON_SESSION" not in os.environ
    assert bound_identity() is None

    # Case B: a prior marker ('1') → restored to '1', never left stuck absent.
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    prior = os.environ.get("HERMES_CRON_SESSION")
    tok = run_identity.bind_run_identity(CRON)
    os.environ["HERMES_CRON_SESSION"] = "1"
    try:
        pass
    finally:
        run_identity.reset_run_identity(tok)
        if prior is None:
            os.environ.pop("HERMES_CRON_SESSION", None)
        else:
            os.environ["HERMES_CRON_SESSION"] = prior
    assert os.environ.get("HERMES_CRON_SESSION") == "1"
