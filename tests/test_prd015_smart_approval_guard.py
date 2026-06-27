"""PRD-015 FR-1a/FR-1b regression tests for the smart-approval guard.

Covers the adversarial-review findings that gate the manual->smart flip:
  * STOP-1  — cron must never reach smart auto-approval even when the in-process
              gateway/CLI attended flags leak into the scheduler thread.
  * SERIOUS-2 — an aux `approve` must be one-shot (no session-wide pattern grant).
  * FR-1b   — privileged/persistence/cloud/credential-egress commands must be
              force-escalated (the aux LLM may never silently auto-approve them),
              and must be flagged by detection in the first place.

Run: source venv/bin/activate && python -m pytest tests/test_prd015_smart_approval_guard.py -q
"""
import os
from unittest import mock

import pytest

from tools import approval


# --- FR-1b: detection + force-escalate set --------------------------------

MUST_ESCALATE = [
    "sudo useradd backdoor",
    "sudo usermod -aG sudo eve",
    "chmod u+s /bin/bash",
    "echo '* * * * * sh' | crontab -",
    "crontab /tmp/evil",
    "gh auth token",
    "gh pr merge 5 --admin",
    "kubectl delete namespace prod",
    "aws s3 rm s3://bucket --recursive",
    "cat ../.env",
    "base64 ../../opt/data/.env",
]

# Legit commands that must NOT be force-escalated (false-positive guard).
NOT_ESCALATE = [
    "chmod 755 build.sh",
    "crontab -l",
    "cat ../README.md",
    "kubectl get pods",
    "aws s3 ls",
    "gh pr view 5",
    "git status",
    "ls -la",
    "sudo apt-get install ripgrep",   # privileged but intentionally not in the set
]


@pytest.mark.parametrize("cmd", MUST_ESCALATE)
def test_must_escalate_true(cmd):
    assert approval._must_escalate(cmd) is True, cmd


@pytest.mark.parametrize("cmd", NOT_ESCALATE)
def test_must_escalate_false(cmd):
    assert approval._must_escalate(cmd) is False, cmd


@pytest.mark.parametrize("cmd", MUST_ESCALATE)
def test_must_escalate_commands_are_flagged_dangerous(cmd):
    # They must trip detection, else "clean" smart mode auto-runs them.
    is_dangerous, _key, _desc = approval.detect_dangerous_command(cmd)
    assert is_dangerous is True, cmd


# --- Shared harness for the full-flow tests --------------------------------

@pytest.fixture
def smart_env(monkeypatch):
    """approvals.mode=smart, cron_mode=deny, no real gateway, tirith=allow."""
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "deny")
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_is_contained_ok", lambda cmd: False)
    monkeypatch.setattr(approval, "_command_matches_permanent_allowlist", lambda cmd: False)
    # Neutralize tirith so the test exercises only the dangerous-pattern path.
    import tools.tirith_security as ts
    monkeypatch.setattr(
        ts, "check_command_security",
        lambda command: {"action": "allow", "findings": [], "summary": ""},
    )
    # Clean slate for session approvals.
    approval.clear_session(approval.get_current_session_key())
    yield monkeypatch
    approval.clear_session(approval.get_current_session_key())


def _set_cron_leak(monkeypatch):
    """Simulate the in-process cron thread: cron flag set AND both attended
    flags leaked (gateway HERMES_EXEC_ASK + cli HERMES_INTERACTIVE)."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")


def _set_attended_cli(monkeypatch):
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)


# --- STOP-1: cron never reaches smart auto-approval ------------------------

@pytest.mark.parametrize("unattended_flag", ["HERMES_CRON_SESSION", "HERMES_AUTONOMOUS"])
def test_unattended_blocks_dangerous_despite_leaked_attended_flags(smart_env, unattended_flag):
    monkeypatch = smart_env
    # The unattended flag plus BOTH leaked attended flags (the real in-process
    # cron/autonomous thread state): gateway HERMES_EXEC_ASK + cli HERMES_INTERACTIVE.
    monkeypatch.setenv(unattended_flag, "1")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    aux = mock.Mock(return_value="approve")
    monkeypatch.setattr(approval, "_smart_approve", aux)

    res = approval.check_all_command_guards("rm -rf ./build", "local")

    assert res["approved"] is False, res          # blocked by cron_mode floor
    assert not res.get("smart_approved")
    aux.assert_not_called()                        # guardian never reached unattended


# --- SERIOUS-2: aux approve is one-shot, not a session-wide grant -----------

def test_aux_approve_is_one_shot(smart_env):
    monkeypatch = smart_env
    _set_attended_cli(monkeypatch)
    aux = mock.Mock(return_value="approve")
    monkeypatch.setattr(approval, "_smart_approve", aux)

    # "rm -rf ./build" is dangerous but NOT in the must-escalate set, so it
    # routes to the aux LLM. Two successive invocations must BOTH consult it.
    r1 = approval.check_all_command_guards("rm -rf ./build", "local")
    r2 = approval.check_all_command_guards("rm -rf ./build", "local")

    assert r1.get("smart_approved") and r2.get("smart_approved")
    assert aux.call_count == 2, "session-grant short-circuited the 2nd call"


# --- FR-1b: must-escalate commands skip the aux and reach a human ----------

def test_must_escalate_skips_aux_and_reaches_human(smart_env):
    monkeypatch = smart_env
    _set_attended_cli(monkeypatch)
    aux = mock.Mock(return_value="approve")     # would auto-approve if consulted
    monkeypatch.setattr(approval, "_smart_approve", aux)
    human = mock.Mock(return_value="deny")       # the escalation target
    monkeypatch.setattr(approval, "prompt_dangerous_approval",
                        lambda *a, **k: human())

    res = approval.check_all_command_guards("sudo useradd backdoor", "local")

    aux.assert_not_called()                      # forced escalate, aux skipped
    human.assert_called_once()                   # human prompt reached
    assert res["approved"] is False              # human denied
