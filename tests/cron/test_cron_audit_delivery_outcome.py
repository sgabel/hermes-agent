"""PRD-043 FR-2 — cron audit outcome derived NEGATIVELY from delivery.

AC-004 + AC-006, driven through the REAL ``run_one_job →
_record_autonomous_cron_run → audit.read_all`` path. ``audit.record`` and
``budget.debit`` are deliberately NOT stubbed (the stubbed-record shape in
``test_autonomy_governance.py`` would miss this bug class); the ledger lands
in the per-test hermetic HERMES_HOME. Only the job pipeline primitives
(``run_job``/``save_job_output``/``mark_job_run``) are stubbed, and
``_deliver_result`` is monkeypatched only where a forced delivery failure is
needed — the AC-006 local/no-origin cases run the real ``_deliver_result``.

The invariant (STOP-1): ``_deliver_result`` returns ``None`` on success AND on
every legitimate non-delivery, so ``outcome`` must be ``delivery_failed`` iff
``delivery_error is not None`` — absence of delivery is a legitimate ``ok``.
The nightly ``deliver=local`` crons (consolidation, reflection) depend on it.
"""
import cron.scheduler as s
from autonomy import audit


def _patch_pipeline(monkeypatch, *, success=True, final="final response", error=None):
    monkeypatch.setattr(s, "run_job", lambda job: (success, "out", final, error))
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None: None,
    )


def _last_cron_outcome():
    recs = [r for r in audit.read_all() if r["surface"] == "cron"]
    assert recs, "no cron audit record written"
    return recs[-1]["outcome"]


# ---------------------------------------------------------------------------
# AC-004 — forced delivery failure → delivery_failed; agent failure stays error
# ---------------------------------------------------------------------------

def test_ac004_forced_delivery_failure_records_delivery_failed(monkeypatch):
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(
        s, "_deliver_result",
        lambda job, content, adapters=None, loop=None:
            "platform 'discord' not configured/enabled",
    )
    assert s.run_one_job({"id": "d1", "name": "delivery-fails", "deliver": "discord"}) is True
    assert _last_cron_outcome() == "delivery_failed"


def test_ac004_deliver_result_raise_records_delivery_failed(monkeypatch):
    """A raise inside _deliver_result is stringified into delivery_error at the
    call site — it must also read as delivery_failed."""
    _patch_pipeline(monkeypatch)

    def boom(job, content, adapters=None, loop=None):
        raise RuntimeError("socket down")

    monkeypatch.setattr(s, "_deliver_result", boom)
    s.run_one_job({"id": "d2", "name": "t", "deliver": "discord"})
    assert _last_cron_outcome() == "delivery_failed"


def test_ac004_agent_failure_outcome_independent_of_delivery(monkeypatch):
    """An agent-run failure keeps its ``error: …`` outcome even when delivery
    of the failure notice ALSO fails (STOP-2: the two are threaded separately)."""
    _patch_pipeline(monkeypatch, success=False, final="", error="boom")
    monkeypatch.setattr(
        s, "_deliver_result",
        lambda job, content, adapters=None, loop=None:
            "platform 'discord' not configured/enabled",
    )
    s.run_one_job({"id": "d3", "name": "t", "deliver": "discord"})
    out = _last_cron_outcome()
    assert out.startswith("error:")
    assert "boom" in out


def test_ac004_empty_response_soft_failure_is_agent_side(monkeypatch):
    """The empty-response soft failure (scheduler flips success→False with its
    own error string) is agent-side — it must NOT read as a delivery failure."""
    _patch_pipeline(monkeypatch, final="   ")
    s.run_one_job({"id": "d4", "name": "t", "deliver": "local"})
    out = _last_cron_outcome()
    assert out.startswith("error:")
    assert "empty response" in out


# ---------------------------------------------------------------------------
# AC-006 — legitimately-silent / local runs stay "ok" (STOP-1 regression guard)
# ---------------------------------------------------------------------------

def test_ac006_deliver_local_stays_ok(monkeypatch):
    """deliver=local through the REAL _deliver_result (returns None without
    attempting delivery) — the nightly consolidation/reflection cron shape."""
    _patch_pipeline(monkeypatch)
    s.run_one_job({"id": "l1", "name": "local job", "deliver": "local"})
    assert _last_cron_outcome() == "ok"


def test_ac006_silent_run_stays_ok(monkeypatch):
    """[SILENT] → should_deliver=False → delivery never attempted → ok."""
    _patch_pipeline(monkeypatch, final="[SILENT] nothing to report")

    def must_not_be_called(job, content, adapters=None, loop=None):
        raise AssertionError("_deliver_result must not be called for [SILENT]")

    monkeypatch.setattr(s, "_deliver_result", must_not_be_called)
    s.run_one_job({"id": "s1", "name": "silent job", "deliver": "discord"})
    assert _last_cron_outcome() == "ok"


def test_ac006_no_origin_stays_ok(monkeypatch):
    """deliver=origin with no captured origin and no configured home channels
    (the CLI-created job shape) through the REAL _deliver_result → ok."""
    _patch_pipeline(monkeypatch)
    s.run_one_job({"id": "o1", "name": "origin job", "deliver": "origin"})
    assert _last_cron_outcome() == "ok"
