"""Autonomy governance & audit spine (PRD-028).

The *observe-and-quiesce* half of Sylva's autonomy governance:

- ``audit``      — append-only JSONL ledger of autonomous actions (+ hash chain).
- ``budget``     — daily ceilings on autonomous actions / second-opinion calls / tokens.
- ``killswitch`` — flag-file quiesce that stops new autonomous work on next poll.
- ``redact``     — hardened secret-screen for audit/egress text (PRD-024 handoff).

Dispatch-level *fail-closed enforcement* of the T0–T4 ladder (classifying a host
write / model load as T4 before it runs) is deliberately NOT here — it needs a
sandbox root and a tool-dispatch policy hook, and is handed to PRD-026. This
package can only observe, cap, and quiesce; it can never expand capability.

All state lives under ``~/.hermes/autonomy/`` (dir mode 0700, files 0600).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - fallback if constants module unavailable
    def get_hermes_home() -> Path:  # type: ignore
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def autonomy_dir() -> Path:
    """Return ``~/.hermes/autonomy/`` (created, mode 0700) at call time.

    Resolved dynamically so profile/env changes and test monkeypatching of
    ``HERMES_HOME`` are honored, mirroring ``cron/scheduler.py``.
    """
    d = get_hermes_home() / "autonomy"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:  # pragma: no cover - non-POSIX
        pass
    return d


__all__ = ["autonomy_dir", "get_hermes_home"]
