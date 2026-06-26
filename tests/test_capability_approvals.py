"""Tests for the PRD-032 R4 durable per-action T4 approval store."""

import importlib
import os
import pytest


@pytest.fixture()
def approvals(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.capability_approvals as m
    importlib.reload(m)
    return m


def test_submit_then_check_is_false_until_approved(approvals):
    approvals.submit("write_file", {"file_path": "/etc/x"}, "/etc/x")
    # Not approved yet → no one-shot grant.
    assert approvals.check_and_consume("write_file", {"file_path": "/etc/x"}, "/etc/x") is False


def test_approve_then_consume_once(approvals):
    sub = approvals.submit("write_file", {"file_path": "/etc/x"}, "/etc/x")
    res = approvals.approve(sub["hash"])
    assert res["ok"] is True
    # One-shot: first consume succeeds, second fails.
    assert approvals.check_and_consume("write_file", {"file_path": "/etc/x"}, "/etc/x") is True
    assert approvals.check_and_consume("write_file", {"file_path": "/etc/x"}, "/etc/x") is False


def test_hash_binds_to_resolved_target(approvals):
    sub = approvals.submit("write_file", {"file_path": "/etc/x"}, "/etc/x")
    approvals.approve(sub["hash"])
    # A different resolved target must NOT reuse the approval (I9).
    assert approvals.check_and_consume("write_file", {"file_path": "/etc/y"}, "/etc/y") is False
    # The exact action still works once.
    assert approvals.check_and_consume("write_file", {"file_path": "/etc/x"}, "/etc/x") is True


def test_approve_by_prefix(approvals):
    sub = approvals.submit("delegate_task", {"goal": "x"}, "")
    res = approvals.approve(sub["hash"][:10])
    assert res["ok"] is True


def test_no_approve_all(approvals):
    # There is no bulk approve; approving requires a specific (unique) id.
    approvals.submit("write_file", {"file_path": "/a"}, "/a")
    approvals.submit("write_file", {"file_path": "/b"}, "/b")
    # An empty/ambiguous prefix must not approve everything.
    res = approvals.approve("")
    assert res["ok"] is False
    assert "ambiguous" in res["error"] or "no active" in res["error"]
    assert len(approvals.pending()) == 2


def test_expired_approval_not_consumable(approvals, monkeypatch):
    sub = approvals.submit("write_file", {"file_path": "/etc/x"}, "/etc/x")
    approvals.approve(sub["hash"])
    # Force expiry by rewriting the store with a past expires_ts.
    import json
    p = approvals._store_path()
    data = json.loads(p.read_text())
    data[sub["hash"]]["expires_ts"] = 1.0
    p.write_text(json.dumps(data))
    assert approvals.check_and_consume("write_file", {"file_path": "/etc/x"}, "/etc/x") is False


def test_pending_lists_queued(approvals):
    approvals.submit("write_file", {"file_path": "/a"}, "/a")
    items = approvals.pending()
    assert len(items) == 1
    assert items[0]["tool"] == "write_file"


def test_guard_t4_unattended_queues_then_approves(monkeypatch, tmp_path):
    """End-to-end: enforce+unattended T4 -> blocked_queued -> approve -> allowed once."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    import tools.capability_approvals as approvals
    import tools.capability_policy as cp
    importlib.reload(approvals)
    importlib.reload(cp)

    ctx = {"unattended": True, "backend": "local"}
    g1 = cp.guard("delegate_task", {"goal": "do a thing"}, ctx=ctx)
    assert g1["allowed"] is False
    assert g1["outcome"] == "blocked_queued"
    h = g1["approval_hash"]
    assert h

    assert approvals.approve(h)["ok"] is True
    g2 = cp.guard("delegate_task", {"goal": "do a thing"}, ctx=ctx)
    assert g2["allowed"] is True
    assert g2["outcome"] == "approved-once"
    # One-shot consumed — a third attempt re-queues.
    g3 = cp.guard("delegate_task", {"goal": "do a thing"}, ctx=ctx)
    assert g3["allowed"] is False
