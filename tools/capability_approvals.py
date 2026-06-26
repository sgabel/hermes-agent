"""Durable per-action T4 approval store (PRD-032 R4).

Unattended T4 actions degrade-to-ask: instead of a hard deny, the action is
recorded as ``queued`` in a durable store keyed by a **canonical hash of
(tool, args, resolved-target)** (I9). A human approves it **one-shot**; the next
attempt of the *same* action (same hash) is allowed exactly once and then marked
``consumed``. Approvals **expire**. There is deliberately **no "approve all", no
session-wide or permanent approval, and no stale execution** (Codex STOP-D).

State: ``~/.hermes/autonomy/pending_approvals.json`` (0600), flock-serialised
(mirrors ``autonomy/budget.py``). Distinct from the in-memory attended approval
queue in ``tools/approval.py`` — that governs attended prompts; this governs the
unattended capability ladder.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore

DEFAULT_TTL_SECONDS = 24 * 3600  # one-shot approvals expire after a day


def _store_dir() -> Path:
    try:
        from autonomy import autonomy_dir
        return autonomy_dir()
    except Exception:
        d = Path(os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "autonomy"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _store_path() -> Path:
    return _store_dir() / "pending_approvals.json"


def canonical_hash(tool: str, args: Optional[Dict[str, Any]],
                   resolved_target: str = "") -> str:
    """Stable hash of (tool, canonical args, resolved target). Binds approval to
    the exact action (I9) so a swapped target can't reuse an approval."""
    try:
        canon_args = json.dumps(args or {}, sort_keys=True, ensure_ascii=False,
                                default=str)
    except Exception:
        canon_args = str(args)
    blob = f"{tool}\x1f{canon_args}\x1f{resolved_target}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _now() -> float:
    return time.time()


def _read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8")) or {}
    except Exception:
        return {}


def _write(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:  # pragma: no cover
        pass
    os.replace(tmp, path)


class _Lock:
    """flock context (mirrors budget.py). No-op where fcntl is unavailable."""

    def __init__(self):
        self._fd = None

    def __enter__(self):
        lock_path = _store_path().with_suffix(".json.lock")
        self._fd = open(lock_path, "w", encoding="utf-8")
        if fcntl:
            try:
                os.chmod(lock_path, 0o600)
            except OSError:  # pragma: no cover
                pass
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if fcntl and self._fd:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        if self._fd:
            self._fd.close()


def _is_active(entry: Dict[str, Any], now: float) -> bool:
    return entry.get("status") == "queued" and entry.get("expires_ts", 0) > now


def submit(tool: str, args: Optional[Dict[str, Any]], resolved_target: str = "",
           rationale: str = "", tier: str = "T4",
           ttl_seconds: int = DEFAULT_TTL_SECONDS) -> Dict[str, Any]:
    """Queue (or refresh) a T4 action for one-shot human approval. Idempotent on
    the canonical hash — re-submitting an active request does not duplicate it."""
    h = canonical_hash(tool, args, resolved_target)
    now = _now()
    path = _store_path()
    with _Lock():
        data = _read(path)
        existing = data.get(h)
        if existing and existing.get("status") == "approved":
            # Already approved + waiting to be consumed — leave as is.
            return {"hash": h, "status": "approved"}
        try:
            arg_summary = json.dumps(args or {}, ensure_ascii=False, default=str)[:300]
        except Exception:
            arg_summary = str(args)[:300]
        data[h] = {
            "hash": h,
            "tool": tool,
            "args_summary": arg_summary,
            "resolved_target": resolved_target,
            "tier": tier,
            "rationale": rationale,
            "status": "queued",
            "created_ts": existing.get("created_ts", now) if existing else now,
            "expires_ts": now + ttl_seconds,
        }
        _write(path, data)
    return {"hash": h, "status": "queued"}


def check_and_consume(tool: str, args: Optional[Dict[str, Any]],
                      resolved_target: str = "") -> bool:
    """If this exact action has an approved, unexpired entry, consume it (one-shot)
    and return True. Otherwise False. Consumption is atomic under the lock."""
    h = canonical_hash(tool, args, resolved_target)
    now = _now()
    path = _store_path()
    with _Lock():
        data = _read(path)
        entry = data.get(h)
        if not entry or entry.get("status") != "approved":
            return False
        if entry.get("expires_ts", 0) <= now:
            entry["status"] = "expired"
            _write(path, data)
            return False
        entry["status"] = "consumed"
        entry["consumed_ts"] = now
        _write(path, data)
    return True


def approve(hash_prefix: str) -> Dict[str, Any]:
    """Human approves a queued action (one-shot). Matches a full hash or a unique
    prefix. No 'approve all'. Returns {'ok', 'hash'|'error'}."""
    now = _now()
    path = _store_path()
    with _Lock():
        data = _read(path)
        matches = [h for h, e in data.items()
                   if h.startswith(hash_prefix) and _is_active(e, now)]
        if not matches:
            return {"ok": False, "error": "no active queued approval matches that id"}
        if len(matches) > 1:
            return {"ok": False, "error": f"ambiguous prefix ({len(matches)} matches)"}
        h = matches[0]
        data[h]["status"] = "approved"
        data[h]["approved_ts"] = now
        # Re-arm the TTL from approval time so an old queued item gets a fresh
        # one-shot window (still expires — no stale execution).
        data[h]["expires_ts"] = now + DEFAULT_TTL_SECONDS
        _write(path, data)
    return {"ok": True, "hash": h}


def pending() -> List[Dict[str, Any]]:
    """Active queued (awaiting approval) entries, newest first."""
    now = _now()
    data = _read(_store_path())
    out = [e for e in data.values() if _is_active(e, now)]
    return sorted(out, key=lambda e: e.get("created_ts", 0), reverse=True)


def purge_expired() -> int:
    """Drop expired/consumed entries. Returns count removed."""
    now = _now()
    path = _store_path()
    with _Lock():
        data = _read(path)
        keep = {h: e for h, e in data.items()
                if e.get("status") in {"queued", "approved"}
                and e.get("expires_ts", 0) > now}
        removed = len(data) - len(keep)
        if removed:
            _write(path, keep)
    return removed
