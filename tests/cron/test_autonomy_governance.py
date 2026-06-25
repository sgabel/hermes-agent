"""PRD-028 — kill switch + audit/budget wiring into the cron run path.

Verifies the integration points (not the autonomy package internals, which are
covered in tests/autonomy/): the kill switch is honored on the directly-invoked
``run_one_job`` path (AC-009, the gateway-wedged bad path Codex named), and a
successful autonomous run writes an audit record + debits the budget (AC-003/006).
"""
import cron.scheduler as s
from autonomy import killswitch


def _patch_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(s, "run_job", lambda job: calls.append(("run_job", job["id"])) or (True, "out", "final", None))
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", lambda job, content, adapters=None, loop=None: None)
    monkeypatch.setattr(s, "mark_job_run", lambda jid, ok, err=None, delivery_error=None: calls.append(("mark", jid, ok)))
    return calls


def test_killswitch_blocks_run_one_job(monkeypatch, tmp_path):
    """AC-009: with the flag set, a directly-invoked run_one_job refuses to run
    the job (the bad path that bypasses tick())."""
    flag = tmp_path / "QUIESCE"
    monkeypatch.setenv("HERMES_AUTONOMY_QUIESCE_FLAG", str(flag))
    flag.write_text("halt\n")
    calls = _patch_pipeline(monkeypatch)

    result = s.run_one_job({"id": "blocked", "name": "t"})

    assert result is True                      # cleanly skipped, not a failure
    assert ("run_job", "blocked") not in calls  # the job body never ran
    assert killswitch.is_quiesced() is True


def test_run_one_job_runs_and_records_when_armed(monkeypatch, tmp_path):
    """AC-003/006: armed → job runs, an audit record is written, budget debited."""
    flag = tmp_path / "QUIESCE"
    monkeypatch.setenv("HERMES_AUTONOMY_QUIESCE_FLAG", str(flag))  # absent = armed
    calls = _patch_pipeline(monkeypatch)

    recorded = {}

    def fake_record(**kw):
        recorded.update(kw)
        return kw

    debited = {}

    def fake_debit(surface, kind, amount=1, **_):
        debited["surface"], debited["kind"] = surface, kind
        return {"allowed": True, "degrade": False}

    import autonomy.audit as audit_mod
    import autonomy.budget as budget_mod
    monkeypatch.setattr(audit_mod, "record", fake_record)
    monkeypatch.setattr(budget_mod, "debit", fake_debit)

    result = s.run_one_job({"id": "ok1", "name": "morning briefing"})

    assert result is True
    assert ("run_job", "ok1") in calls
    assert recorded.get("surface") == "cron"
    assert "morning briefing" in recorded.get("action", "")
    assert debited == {"surface": "cron", "kind": "actions"}
