"""PRD-028 R-2 / AC-003, AC-004 — append-only JSONL ledger + hash chain."""

import json
import os
import stat

from autonomy import audit


def test_record_appends_and_sets_mode():
    audit.record(tier="T0", surface="cron", action="read sessions",
                 rationale="prefetch", authority="auto-by-tier", outcome="ok")
    audit.record(tier="T3", surface="proactive", action="send discord nudge",
                 rationale="idle", authority="second-opinion", outcome="ok")
    recs = audit.read_all()
    assert len(recs) == 2
    assert recs[0]["surface"] == "cron"
    assert recs[1]["authority"] == "second-opinion"
    path = audit._audit_path()
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_hash_chain_verifies():
    for i in range(5):
        audit.record(tier="T1", surface="sandbox", action=f"job {i}")
    ok, bad = audit.verify_chain()
    assert ok is True
    assert bad is None


def test_genesis_prev_hash():
    audit.record(tier="T0", surface="cli", action="first")
    recs = audit.read_all()
    assert recs[0]["prev_hash"] == audit.GENESIS_HASH


def test_tamper_detection():
    for i in range(4):
        audit.record(tier="T1", surface="cron", action=f"action {i}")
    path = audit._audit_path()
    lines = path.read_text(encoding="utf-8").splitlines()
    # Tamper with the rationale of line 2 without recomputing its hash.
    rec = json.loads(lines[1])
    rec["action"] = "TAMPERED"
    lines[1] = json.dumps(rec, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, bad = audit.verify_chain()
    assert ok is False
    assert bad == 2


def test_secret_redacted_in_record():
    audit.record(tier="T3", surface="proactive",
                 action="emit MY_THING=s3cr3tValue9x8y7z6w5v4u to peer",
                 rationale="test")
    recs = audit.read_all()
    assert "s3cr3tValue9x8y7z6w5v4u" not in json.dumps(recs)


def test_audit_enabled_flag_honored(monkeypatch):
    monkeypatch.setattr(audit, "_audit_enabled", lambda: False)
    rec = audit.record(tier="T0", surface="cron", action="should not persist")
    assert rec.get("persisted") is False
    assert audit.read_all() == []


def test_query_window():
    audit.record(tier="T0", surface="cron", action="recent")
    recent = audit.query(hours=1)
    assert any(r["action"] == "recent" for r in recent)
    assert audit.query(hours=0) == []
