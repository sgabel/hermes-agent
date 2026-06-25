"""Flag-file kill switch for autonomous loops (PRD-028 R-6 / FR-4).

A single global flag file ``~/.hermes/autonomy/QUIESCE``. Its *presence* means
"do no new autonomous work." ``is_quiesced()`` is a cheap ``stat`` polled at
every autonomous entry site (the cron scheduler ``start``/``tick``/``run_one_job``
and provider ``fire_due``). Because it is a file — not an RPC — it works even
when the gateway is wedged: a directly-invoked ``run_one_job`` still sees it
on its next poll (AC-009).

Granularity: one global T1–T3 off (Codex P-3). Interactive Sylva and the LLM
servers are untouched — they never consult the flag. Re-arm = delete the flag
(explicit). The hard fallback for already-spawned children is
``systemctl --user stop sylva-autonomy.slice`` (PRD-025); this flag stops *new*
work, not work already mid-flight.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from autonomy import autonomy_dir

_FLAG_NAME = "QUIESCE"
# Allow an env override (tests / alternate deployments) without touching config.
_ENV_OVERRIDE = "HERMES_AUTONOMY_QUIESCE_FLAG"


def flag_path() -> Path:
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override).expanduser()
    # config.yaml may relocate the flag; default to the autonomy dir.
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        configured = cfg_get(read_raw_config(), "autonomy", "killswitch_flag")
        if isinstance(configured, str) and configured.strip():
            return Path(configured).expanduser()
    except Exception:
        pass
    return autonomy_dir() / _FLAG_NAME


def is_quiesced() -> bool:
    """True if autonomous work is currently halted (flag present)."""
    try:
        return flag_path().exists()
    except OSError:  # pragma: no cover
        # Fail safe: if we cannot even stat the flag, assume quiesced.
        return True


def quiesce(reason: str = "") -> Path:
    """Engage the kill switch (create the flag). Idempotent."""
    path = flag_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    payload = f"quiesced_at={stamp}\nreason={reason}\n"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    _audit("autonomy kill switch ENGAGED", reason or "manual", "blocked")
    return path


def rearm() -> bool:
    """Disengage the kill switch (delete the flag). Returns True if it existed."""
    path = flag_path()
    existed = path.exists()
    if existed:
        try:
            path.unlink()
        except OSError:  # pragma: no cover
            pass
    _audit("autonomy kill switch RE-ARMED", "manual re-arm", "ok")
    return existed


def status() -> dict:
    path = flag_path()
    info: dict = {"quiesced": path.exists(), "flag": str(path)}
    if info["quiesced"]:
        try:
            info["detail"] = path.read_text(encoding="utf-8").strip()
        except OSError:  # pragma: no cover
            info["detail"] = ""
    return info


def guard(surface: str) -> bool:
    """Convenience for loop sites: if quiesced, audit the skip and return True.

    Caller pattern:  ``if killswitch.guard("cron"): return``  (skip the work).
    """
    if is_quiesced():
        _audit(f"autonomous work skipped on {surface}", "kill switch engaged", "blocked", surface=surface)
        return True
    return False


def _audit(action: str, rationale: str, outcome: str, surface: str = "cli") -> None:
    try:
        from autonomy import audit

        audit.record(tier="T3", surface=surface, action=action,
                     rationale=rationale, authority="human", outcome=outcome)
    except Exception:
        pass


__all__ = ["is_quiesced", "quiesce", "rearm", "status", "guard", "flag_path"]
