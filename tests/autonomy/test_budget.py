"""PRD-028 R-3 / AC-006 — daily budget governor (durable, degrade-to-ask).

PRD-043 FR-1 (AC-001/AC-003): unknown kinds fail CLOSED (deny + WARNING +
best-effort ``denied_unknown_kind`` audit); a KINDS member with no cap mapping
raises the defined ``BudgetKindError`` (never bare ``KeyError``) on all three
of ``check``/``debit``/``get_usage``.
"""

import logging

import pytest

from autonomy import audit, budget


def _set_caps(monkeypatch, **caps):
    base = dict(budget._DEFAULT_CAPS)
    base.update(caps)
    monkeypatch.setattr(budget, "_load_caps", lambda: base)


def test_debit_increments_and_attributes(monkeypatch):
    _set_caps(monkeypatch, max_autonomous_actions=10)
    budget.debit("cron", "actions", 1)
    budget.debit("proactive", "actions", 2)
    usage = budget.get_usage()
    assert usage["totals"]["actions"] == 3
    assert usage["by_surface"]["cron"]["actions"] == 1
    assert usage["by_surface"]["proactive"]["actions"] == 2
    assert usage["remaining"]["actions"] == 7


def test_cap_breach_degrades(monkeypatch):
    _set_caps(monkeypatch, max_second_opinion_calls=2)
    assert budget.debit("proactive", "second_opinion_calls")["degrade"] is False
    assert budget.debit("proactive", "second_opinion_calls")["degrade"] is False
    r = budget.debit("proactive", "second_opinion_calls")
    assert r["degrade"] is True
    assert r["allowed"] is False


def test_check_does_not_consume(monkeypatch):
    _set_caps(monkeypatch, max_autonomous_actions=5)
    assert budget.check("actions", 1) is True
    assert budget.get_usage()["totals"]["actions"] == 0
    budget.debit("cron", "actions", 5)
    assert budget.check("actions", 1) is False


def test_durable_across_reload(monkeypatch):
    _set_caps(monkeypatch, max_autonomous_tokens=1_000)
    budget.debit("cron", "tokens", 400)
    # Simulate a fresh process: counters re-read from disk.
    assert budget.get_usage()["totals"]["tokens"] == 400
    budget.debit("cron", "tokens", 400)
    assert budget.get_usage()["totals"]["tokens"] == 800


# ---------------------------------------------------------------------------
# PRD-043 FR-1 / AC-001 — unknown kinds fail closed
# (replaces the pre-043 ``test_unknown_kind_is_noop``, which asserted the
# fail-open behavior this PRD removes)
# ---------------------------------------------------------------------------

def test_ac001_unknown_kind_check_fails_closed(monkeypatch, caplog):
    _set_caps(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="autonomy.budget"):
        assert budget.check("no_such_kind", 1) is False
    assert any("no_such_kind" in r.getMessage() for r in caplog.records)
    recs = audit.read_all()
    assert recs, "denial audit record missing"
    assert recs[-1]["outcome"] == "denied_unknown_kind"
    assert recs[-1]["surface"] == "budget-check"
    assert recs[-1]["tier"] == "T3"


def test_ac001_unknown_kind_debit_refuses_full_dict(monkeypatch, caplog):
    _set_caps(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="autonomy.budget"):
        r = budget.debit("testsurf", "no_such_kind", 3)
    # full denied dict — shape-identical to the allowed return (MINOR-2)
    assert r["allowed"] is False
    assert r["degrade"] is False
    assert r["kind"] == "no_such_kind"
    assert "usage" in r
    assert any("no_such_kind" in rec.getMessage() for rec in caplog.records)
    # nothing recorded against caps
    usage = budget.get_usage()
    assert all(v == 0 for v in usage["totals"].values())
    assert "testsurf" not in usage["by_surface"]
    recs = audit.read_all()
    assert recs[-1]["outcome"] == "denied_unknown_kind"
    assert recs[-1]["surface"] == "testsurf"


def test_ac001_denial_audit_fires_even_with_audit_false(monkeypatch):
    """The capability-policy call shape (audit=False) must NOT mute the
    unknown-kind denial audit — it's a governance event (SERIOUS-1)."""
    _set_caps(monkeypatch)
    r = budget.debit("capgate", "no_such_kind", 1, audit=False)
    assert r["allowed"] is False
    recs = audit.read_all()
    assert recs, "denial audit was muted by audit=False"
    assert recs[-1]["outcome"] == "denied_unknown_kind"


def test_ac001_denial_audit_raise_does_not_propagate(monkeypatch):
    """check/debit run inside capability_policy.guard — a broken audit ledger
    must never raise into the dispatch gate (SERIOUS-1)."""
    _set_caps(monkeypatch)

    def boom(**kw):
        raise RuntimeError("ledger down")

    import autonomy.audit as audit_mod
    monkeypatch.setattr(audit_mod, "record", boom)
    assert budget.check("no_such_kind") is False
    r = budget.debit("s", "no_such_kind")
    assert r["allowed"] is False and r["degrade"] is False


# ---------------------------------------------------------------------------
# PRD-043 FR-1 / AC-003 — partial registration raises BudgetKindError
# ---------------------------------------------------------------------------

def test_ac003_partial_registration_raises_defined_error_all_paths(monkeypatch):
    """A kind in KINDS with no _DEFAULT_CAPS/_KIND_TO_CAP mapping fails closed
    with BudgetKindError — not bare KeyError — on check, debit AND get_usage
    (get_usage backs `hermes autonomy status`; SERIOUS-2)."""
    _set_caps(monkeypatch)
    monkeypatch.setattr(budget, "KINDS", (*budget.KINDS, "halfkind"))

    with pytest.raises(budget.BudgetKindError):
        budget.check("halfkind", 1)
    with pytest.raises(budget.BudgetKindError):
        budget.debit("cron", "halfkind")
    with pytest.raises(budget.BudgetKindError):
        budget.get_usage()
    # the debit raise happens BEFORE any counter mutation
    monkeypatch.setattr(budget, "KINDS", ("actions", "second_opinion_calls", "tokens"))
    assert budget.get_usage()["totals"]["actions"] == 0


def test_ac003_budget_kind_error_is_not_keyerror():
    assert not issubclass(budget.BudgetKindError, KeyError)


def test_ac003_partial_registration_of_other_kind_does_not_poison_valid_debit(monkeypatch):
    """Implemented-gate STOP (Codex 2026-07-05): a partial registration of a
    NEW kind must not make check/debit of a VALID kind raise — debit's
    return-path usage snapshot iterates all KINDS, and the raise (after the
    counter mutation) would escape into capability_policy.guard's broad
    ``except Exception: pass`` and fail OPEN at the dispatch gate."""
    _set_caps(monkeypatch)
    monkeypatch.setattr(budget, "KINDS", (*budget.KINDS, "halfkind"))

    assert budget.check("actions", 1) is True          # own-kind check unaffected
    r = budget.debit("cron", "actions", 1)             # must NOT raise
    assert r["allowed"] is True and r["degrade"] is False
    assert r["usage"]["totals"]["actions"] == 1
    assert "usage_error" in r["usage"]                 # degraded snapshot stays loud
    # unknown-kind denial path must also survive the cross-kind misconfiguration
    d = budget.debit("cron", "no_such_kind", 1)
    assert d["allowed"] is False

    # the valid debit was recorded exactly once
    monkeypatch.setattr(budget, "KINDS", ("actions", "second_opinion_calls", "tokens"))
    assert budget.get_usage()["totals"]["actions"] == 1


def test_parallel_debits_do_not_lose_updates(monkeypatch):
    """Code-review NEEDS-FIX: cron runs jobs in a ThreadPoolExecutor, so the
    debit read-modify-write must not lose updates (was 54/400 without a lock)."""
    import threading

    _set_caps(monkeypatch, max_autonomous_actions=10_000)
    threads = [
        threading.Thread(target=lambda: [budget.debit("cron", "actions", 1) for _ in range(50)])
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert budget.get_usage()["totals"]["actions"] == 8 * 50
