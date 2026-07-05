"""Daily budget governor for autonomous work (PRD-028 R-3 / FR-3).

Global daily ceilings with per-surface attribution (Codex P-4):
    max_autonomous_actions       — count of autonomous actions/day
    max_second_opinion_calls     — ask_claude (Max-quota) calls/day
    max_autonomous_tokens        — autonomous LLM token spend/day

On breach the governor returns a **degrade-to-ask** signal (NOT a hard stop —
nothing in flight is lost, new autonomous initiative just pauses to ask) and the
breach is recorded to the audit ledger. Counters are durable across process
restart in ``~/.hermes/autonomy/budget-YYYYMMDD.json`` (0600, atomic replace).

VRAM/model-load is intentionally NOT budgeted — it is unmetered and stays
T4/manual (PRD-025 'no cgroup for VRAM').
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # POSIX advisory locking — serialise the debit read-modify-write
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore

from autonomy import autonomy_dir

logger = logging.getLogger(__name__)

KINDS = ("actions", "second_opinion_calls", "tokens")


class BudgetKindError(Exception):
    """A kind is in ``KINDS`` but missing its ``_DEFAULT_CAPS``/``_KIND_TO_CAP``
    registration (PRD-043 FR-1). Registering a kind requires all three; a
    partial registration fails closed with this defined error — never a bare
    ``KeyError`` — identically on ``check``/``debit``/``get_usage``."""

# Conservative defaults; overridden by config.yaml ``autonomy.budget``.
_DEFAULT_CAPS = {
    "max_autonomous_actions": 200,
    "max_second_opinion_calls": 25,
    "max_autonomous_tokens": 2_000_000,
}
_KIND_TO_CAP = {
    "actions": "max_autonomous_actions",
    "second_opinion_calls": "max_second_opinion_calls",
    "tokens": "max_autonomous_tokens",
}


def _cap_for(kind: str, caps: dict[str, int]) -> int:
    """Single guarded cap lookup (PRD-043 FR-1, SERIOUS-2).

    Used by ``check()``, ``debit()`` AND ``get_usage()`` so a ``KINDS`` member
    with no cap mapping fails closed with the defined ``BudgetKindError`` on
    all three call paths (``get_usage`` backs ``hermes autonomy status``, which
    must not crash with a bare ``KeyError`` on a partial registration).
    """
    cap_key = _KIND_TO_CAP.get(kind)
    if cap_key is None or cap_key not in caps:
        raise BudgetKindError(
            f"budget kind '{kind}' is registered in KINDS but has no "
            f"_KIND_TO_CAP/_DEFAULT_CAPS mapping — register all three "
            f"(KINDS, _DEFAULT_CAPS, _KIND_TO_CAP) in the same change"
        )
    return caps[cap_key]


def _record_unknown_kind_denial(surface: str, kind: str, op: str) -> None:
    """Best-effort ``denied_unknown_kind`` audit (PRD-043 FR-1, SERIOUS-1).

    An unknown/unregistered kind is a governance event, not a routine debit, so
    this fires regardless of the caller's ``audit=False`` flag. Wrapped so it
    can NEVER raise into a caller — ``check``/``debit`` run inside
    ``capability_policy.guard`` (the dispatch gate), which is not inside the
    cron path's try/except.
    """
    try:
        from autonomy import audit as _audit

        _audit.record(
            tier="T3",
            surface=surface,
            # em dash, not a colon — "denied: <word>" pattern-matches the audit
            # redactor's key:value opaque-env heuristic and gets mangled in the
            # permanent ledger row
            action=f"budget {op} denied — unregistered kind '{str(kind)[:80]}'",
            rationale="unknown budget kind — fail-closed (PRD-043 FR-1)",
            authority="auto-by-tier",
            outcome="denied_unknown_kind",
        )
    except Exception:
        pass


def _usage_best_effort() -> dict[str, Any]:
    """``get_usage()`` that degrades instead of raising (PRD-043 FR-1 fix-up).

    Used ONLY for the informational ``usage`` snapshot in ``debit()``'s return
    value. A ``BudgetKindError`` here means some OTHER kind is partially
    registered; the current operation is on a valid (or already-denied) kind
    and must not be poisoned by it — the raise would escape ``debit()`` into
    ``capability_policy.guard``'s broad ``except`` and fail OPEN at the
    dispatch gate, the exact inversion this PRD removes. ``get_usage()``
    itself still raises the defined error (AC-003) so ``hermes autonomy
    status`` surfaces the misconfiguration loudly.
    """
    try:
        return get_usage()
    except BudgetKindError as exc:
        logger.warning("budget usage snapshot degraded (partial kind registration): %s", exc)
        data = _read_counters()
        return {"day": data["day"], "totals": data["totals"], "caps": _load_caps(),
                "remaining": {}, "by_surface": data.get("by_surface", {}),
                "usage_error": str(exc)}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _counter_path(day: str | None = None) -> Path:
    return autonomy_dir() / f"budget-{day or _today()}.json"


def _load_caps() -> dict[str, int]:
    caps = dict(_DEFAULT_CAPS)
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        cfg = read_raw_config()
        for cap_key in _DEFAULT_CAPS:
            val = cfg_get(cfg, "autonomy", "budget", cap_key)
            if isinstance(val, (int, float)) and val >= 0:
                caps[cap_key] = int(val)
    except Exception:
        pass
    return caps


def _read_counters(day: str | None = None) -> dict[str, Any]:
    path = _counter_path(day)
    if not path.exists():
        return {"day": day or _today(), "totals": {k: 0 for k in KINDS}, "by_surface": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in KINDS:
            data.setdefault("totals", {}).setdefault(k, 0)
        data.setdefault("by_surface", {})
        return data
    except (json.JSONDecodeError, OSError):
        return {"day": day or _today(), "totals": {k: 0 for k in KINDS}, "by_surface": {}}


def _write_counters(data: dict[str, Any], day: str | None = None) -> None:
    path = _counter_path(day)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".budget-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        try:
            from utils import atomic_replace

            atomic_replace(tmp, path)
        except Exception:
            os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:  # pragma: no cover
            pass
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def get_usage(day: str | None = None) -> dict[str, Any]:
    """Return today's counters + caps + remaining (read-only)."""
    data = _read_counters(day)
    caps = _load_caps()
    remaining = {
        kind: max(0, _cap_for(kind, caps) - data["totals"].get(kind, 0))
        for kind in KINDS
    }
    return {"day": data["day"], "totals": data["totals"], "caps": caps,
            "remaining": remaining, "by_surface": data.get("by_surface", {})}


def check(kind: str, amount: int = 1) -> bool:
    """True if debiting ``amount`` of ``kind`` would stay within the daily cap.

    Unknown kinds fail CLOSED (PRD-043 FR-1): deny, WARN, and write a
    best-effort ``denied_unknown_kind`` audit record.
    """
    if kind not in KINDS:
        logger.warning(
            "budget.check denied: unknown kind %r (registered kinds: %s) — "
            "fail-closed per PRD-043 FR-1", kind, ", ".join(KINDS)
        )
        _record_unknown_kind_denial("budget-check", kind, "check")
        return False
    caps = _load_caps()
    data = _read_counters()
    return data["totals"].get(kind, 0) + amount <= _cap_for(kind, caps)


def debit(surface: str, kind: str, amount: int = 1, *, audit: bool = True) -> dict[str, Any]:
    """Record consumption. Returns {'allowed', 'degrade', 'kind', 'usage'}.

    ``degrade`` True means the cap is now exceeded → caller should degrade to
    ask-only. The debit is still recorded (the action happened / was counted);
    degrade governs *future* autonomous initiative, per FR-3 'degrade-to-ask'.
    """
    if kind not in KINDS:
        logger.warning(
            "budget.debit refused: unknown kind %r from surface %r (registered "
            "kinds: %s) — fail-closed per PRD-043 FR-1", kind, surface, ", ".join(KINDS)
        )
        # Governance event: fires regardless of ``audit=False`` (that flag mutes
        # only the routine degrade audit below).
        _record_unknown_kind_denial(surface, kind, "debit")
        return {"allowed": False, "degrade": False, "kind": kind, "usage": _usage_best_effort()}

    caps = _load_caps()
    cap = _cap_for(kind, caps)  # fail closed BEFORE recording anything

    # Serialise the read-modify-write so parallel cron threads
    # (HERMES_CRON_MAX_PARALLEL > 1) can't lose an update and silently
    # undercount the daily cap. Mirrors audit.py's flock discipline.
    lock_path = _counter_path().with_suffix(".json.lock")
    lock_fd = open(lock_path, "w", encoding="utf-8")
    try:
        if fcntl:
            try:
                os.chmod(lock_path, 0o600)
            except OSError:  # pragma: no cover
                pass
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data = _read_counters()
        data["totals"][kind] = data["totals"].get(kind, 0) + amount
        surf = data["by_surface"].setdefault(surface, {k: 0 for k in KINDS})
        surf[kind] = surf.get(kind, 0) + amount
        _write_counters(data)
    finally:
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    degrade = data["totals"][kind] > cap

    if degrade and audit:
        try:
            from autonomy import audit as _audit

            _audit.record(
                tier="T3",
                surface=surface,
                action=f"budget cap reached: {kind}={data['totals'][kind]}/{cap}",
                rationale="daily autonomous budget exceeded",
                authority="auto-by-tier",
                outcome="degraded",
            )
        except Exception:
            pass

    return {"allowed": not degrade, "degrade": degrade, "kind": kind, "usage": _usage_best_effort()}


__all__ = ["check", "debit", "get_usage", "KINDS", "BudgetKindError"]
