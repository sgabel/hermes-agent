"""PRD-028 R-3 / AC-006 — daily budget governor (durable, degrade-to-ask)."""

from autonomy import budget


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


def test_unknown_kind_is_noop(monkeypatch):
    _set_caps(monkeypatch)
    r = budget.debit("cron", "bogus", 1)
    assert r["allowed"] is True and r["degrade"] is False


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
