"""Unit tests for the PRD-032 R1 central capability gate."""

import os
import importlib
import pytest


@pytest.fixture()
def cp(monkeypatch):
    # Default to observe unless a test sets enforce; clear cron/autonomous markers.
    monkeypatch.delenv("HERMES_CAPABILITY_POLICY_MODE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_AUTONOMOUS", raising=False)
    import tools.capability_policy as m
    importlib.reload(m)
    return m


# --- classify: fail-closed + tier map ---

def test_classify_unknown_is_t4(cp):
    assert cp.classify("totally_unknown_tool") == cp.Tier.T4
    assert cp.classify("") == cp.Tier.T4
    assert cp.classify("some_mcp_server.do_thing") == cp.Tier.T4


def test_classify_reads_are_t0(cp):
    for t in ("read_file", "grep", "list_directory", "session_search", "todo"):
        assert cp.classify(t) == cp.Tier.T0, t


def test_classify_egress_is_t1(cp):
    for t in ("web_search", "web_extract", "browser_navigate", "ask_claude"):
        assert cp.classify(t) == cp.Tier.T1, t


def test_classify_messages_are_t3(cp):
    assert cp.classify("send_message") == cp.Tier.T3


def test_classify_delegate_task_is_t4(cp):
    # I5 — the agent-on-agent surface MUST be T4.
    assert cp.classify("delegate_task") == cp.Tier.T4


def test_classify_host_writes_are_t4(cp):
    for t in ("write_file", "patch", "delete_file"):
        assert cp.classify(t) == cp.Tier.T4, t


def test_classify_exec_tier_depends_on_backend(cp):
    assert cp.classify("execute_code", ctx={"backend": "sylva-sandbox"}) == cp.Tier.T2
    assert cp.classify("terminal", ctx={"backend": "local"}) == cp.Tier.T4
    assert cp.classify("terminal", ctx={"backend": "docker"}) == cp.Tier.T4


# --- guard: observe mode never blocks ---

def test_observe_mode_never_blocks(cp, monkeypatch):
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "observe")
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")  # unattended
    for t in ("write_file", "delegate_task", "totally_unknown", "terminal"):
        g = cp.guard(t, ctx={"backend": "local"})
        assert g["allowed"] is True, t
        assert g["outcome"] == "observed"


# --- guard: enforce mode ---

def test_enforce_attended_allows_t4(cp, monkeypatch):
    # Attended (no cron/autonomous marker): existing approval gate handles T4.
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    g = cp.guard("write_file", ctx={"unattended": False})
    assert g["allowed"] is True
    assert g["outcome"] == "allowed"


def test_enforce_unattended_denies_t4(cp, monkeypatch):
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    for t in ("write_file", "delegate_task", "unknown_tool"):
        g = cp.guard(t, ctx={"unattended": True, "backend": "local"})
        assert g["allowed"] is False, t
        assert g["outcome"] == "blocked"
        assert g["tier"] == "T4"


def test_enforce_unattended_allows_t0(cp, monkeypatch):
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    g = cp.guard("read_file", ctx={"unattended": True})
    assert g["allowed"] is True


def test_enforce_unattended_sandbox_exec_allowed(cp, monkeypatch):
    # T2 contained exec is allowed unattended (within budget).
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    g = cp.guard("execute_code", ctx={"unattended": True, "backend": "sylva-sandbox"})
    assert g["allowed"] is True
    assert g["tier"] == "T2"


def test_enforce_unattended_killswitch_halts_t1(cp, monkeypatch):
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    import tools.capability_policy as m
    monkeypatch.setattr("autonomy.killswitch.is_quiesced", lambda: True)
    g = m.guard("web_search", ctx={"unattended": True})
    assert g["allowed"] is False
    assert "kill-switch" in (g["reason"] or "")


def test_deny_result_is_json_error(cp):
    g = {"allowed": False, "tier": "T4", "reason": "nope", "outcome": "blocked"}
    import json
    out = json.loads(cp.deny_result("write_file", g))
    assert "BLOCKED by capability policy" in out["error"]
    assert out["capability_tier"] == "T4"
