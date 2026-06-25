"""Append-only JSONL audit ledger for autonomous actions (PRD-028 R-2 / FR-2).

The missing *ledger*: gates (tirith, approvals) decide whether an action may run;
this records what actually ran autonomously, why, and what authorised it.

Format (one JSON object per line, ``~/.hermes/autonomy/audit.jsonl``):
    ts        ISO-8601 UTC
    tier      T0..T4
    surface   cron | gateway | sandbox | proactive | cli
    action    short action description (redacted)
    rationale why it ran (redacted)
    authority auto-by-tier | human | second-opinion | <free text> (redacted)
    outcome   ok | degraded | blocked | error | <free text> (redacted)
    prev_hash sha256 of the previous line's record (genesis = 64×"0")
    hash      sha256(prev_hash + canonical-record-without-hash)

Write discipline: ``O_APPEND|O_CREAT|O_WRONLY``, advisory ``flock`` (serialises
concurrent writers under one user), fsync, file mode 0600, dir 0700. The hash
chain gives tamper-*evidence* — a rewritten/inserted line breaks verification.

Honest non-goal: true append-only needs root ``chattr +a``; under a single Unix
user this is convention + tamper-evidence, not tamper-proofing. Documented as such.
Specifically, ``verify_chain`` detects a rewritten / inserted / reordered / deleted
*interior* line, but it CANNOT detect **tail truncation** — an attacker who can write
the file may drop trailing records and leave an internally-valid chain (there is no
external high-water mark). This is inherent to a bare hash chain and in-scope of the
"tamper-evidence, not tamper-proofing" non-goal.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:  # POSIX advisory locking
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore

from autonomy import autonomy_dir

GENESIS_HASH = "0" * 64
_AUDIT_FIELDS = ("ts", "tier", "surface", "action", "rationale", "authority", "outcome")


def _audit_path() -> Path:
    return autonomy_dir() / "audit.jsonl"


def _audit_enabled() -> bool:
    """Honor ``autonomy.audit_enabled`` in config.yaml (default True)."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        val = cfg_get(read_raw_config(), "autonomy", "audit_enabled", default=True)
        return bool(val)
    except Exception:
        return True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(record: dict[str, Any]) -> str:
    """Stable serialisation of the record EXCLUDING ``hash`` (incl. prev_hash)."""
    payload = {k: record.get(k) for k in (*_AUDIT_FIELDS, "prev_hash")}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_record(record: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


def _last_hash(path: Path) -> str:
    """Return the ``hash`` of the final record, or GENESIS if the log is empty."""
    if not path.exists():
        return GENESIS_HASH
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return GENESIS_HASH
    try:
        return json.loads(last).get("hash", GENESIS_HASH)
    except (json.JSONDecodeError, AttributeError):
        return GENESIS_HASH


def _redact(text: Any) -> Any:
    """Run free-text fields through the hardened autonomy secret-screen.

    Fail-closed: if redaction cannot run, replace with a sentinel rather than
    writing a potentially-secret-bearing string.
    """
    if not isinstance(text, str) or not text:
        return text
    try:
        from autonomy.redact import redact_for_autonomy

        return redact_for_autonomy(text)
    except Exception:
        return "[REDACTED:redaction-failed]"


def record(
    *,
    tier: str,
    surface: str,
    action: str,
    rationale: str = "",
    authority: str = "auto-by-tier",
    outcome: str = "ok",
) -> dict[str, Any]:
    """Append one audit record. Returns the written record (with hash).

    No-op (returns the record un-persisted) when ``autonomy.audit_enabled`` is
    false in config.
    """
    if not _audit_enabled():
        return {"tier": tier, "surface": surface, "action": action, "persisted": False}
    path = _audit_path()
    rec: dict[str, Any] = {
        "ts": _now_iso(),
        "tier": tier,
        "surface": surface,
        "action": _redact(action),
        "rationale": _redact(rationale),
        "authority": _redact(authority),
        "outcome": _redact(outcome),
    }

    # Serialise the read-prev-hash → append region with an advisory lock so
    # concurrent writers (cron threads, gateway) can't interleave the chain.
    lock_path = path.with_suffix(".jsonl.lock")
    lock_fd = open(lock_path, "w", encoding="utf-8")
    try:
        try:
            os.chmod(lock_path, 0o600)
        except OSError:  # pragma: no cover
            pass
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        rec["prev_hash"] = _last_hash(path)
        rec["hash"] = _hash_record(rec)
        line = json.dumps(rec, ensure_ascii=False) + "\n"

        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        fd = os.open(path, flags, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        # Ensure mode is 0600 even if the file pre-existed with looser perms.
        try:
            os.chmod(path, 0o600)
        except OSError:  # pragma: no cover
            pass
    finally:
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
    return rec


def read_all() -> list[dict[str, Any]]:
    path = _audit_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def query(hours: float = 24.0) -> list[dict[str, Any]]:
    """Records written in the last ``hours`` hours (AC-005 'what ran + why')."""
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = []
    for rec in read_all():
        try:
            ts = datetime.fromisoformat(rec["ts"])
        except (KeyError, ValueError):
            continue
        if ts >= cutoff:
            out.append(rec)
    return out


def verify_chain() -> tuple[bool, Optional[int]]:
    """Verify the hash chain. Returns (ok, first_bad_line_number or None).

    Detects a tampered/rewritten/inserted/reordered line (AC-004).
    """
    prev = GENESIS_HASH
    for i, rec in enumerate(read_all(), start=1):
        if rec.get("prev_hash") != prev:
            return False, i
        expected = _hash_record(rec)
        if rec.get("hash") != expected:
            return False, i
        prev = rec["hash"]
    return True, None


__all__ = ["record", "query", "read_all", "verify_chain", "GENESIS_HASH"]
