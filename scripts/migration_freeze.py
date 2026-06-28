#!/usr/bin/env python3
"""PRD-029 Phase 5 — write-freeze for the sylva_memories migration (AC-015 NF-5).

Before the final re-snapshot + cutover, freeze every writer into sylva_memories so
no point arrives between snapshot and retirement: disable the reflection +
memory-cleanup crons, and confirm ``nudge_interval: 0`` (already set in Phase 1).

Default is REPORT-ONLY (owner reviews what would be frozen). Pass ``--freeze`` to
actually pause the crons (reversible: ``--thaw`` re-enables them). This touches
live autonomous scheduling, so it is deliberately a separate, explicit step in the
owner-gated cutover sequence — never auto-run.

Usage:
    python3 scripts/migration_freeze.py            # report writers + freeze state
    python3 scripts/migration_freeze.py --freeze   # pause reflection/cleanup crons
    python3 scripts/migration_freeze.py --thaw      # re-enable them (rollback)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# crons whose prompts/skills write into memory (reflection + hygiene/cleanup)
_FREEZE_PATTERNS = re.compile(r"(reflect|memory|hygiene|cleanup|consolidat)", re.I)


def _nudge_interval() -> object:
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        return cfg_get(read_raw_config(), "memory", "nudge_interval", default=None)
    except Exception as e:
        return f"<unreadable: {e}>"


def _matching_jobs():
    from cron import jobs
    out = []
    for j in jobs.list_jobs(include_disabled=True):
        blob = " ".join(str(j.get(k, "")) for k in ("name", "prompt", "skills"))
        if _FREEZE_PATTERNS.search(blob):
            out.append(j)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="pause the matching crons")
    ap.add_argument("--thaw", action="store_true", help="re-enable the matching crons (rollback)")
    args = ap.parse_args()

    nudge = _nudge_interval()
    print(f"nudge_interval = {nudge}  ({'OK (frozen)' if nudge == 0 else 'WARNING: should be 0'})")

    try:
        from cron import jobs
        matches = _matching_jobs()
    except Exception as e:
        print(f"cron jobs unavailable (host context?): {e}")
        matches = []

    if not matches:
        print("no reflection/memory-cleanup crons found (already removed, or running in a context "
              "without the jobs store). Confirm via `docker exec hermes hermes cron list`.")
    for j in matches:
        state = "enabled" if j.get("enabled", True) else "PAUSED"
        print(f"  [{state:8s}] {j.get('id','?')}  {j.get('name','')[:50]}")

    if args.freeze:
        from cron import jobs
        for j in matches:
            if j.get("enabled", True):
                jobs.update_job(j["id"], {"enabled": False, "state": "paused"})
                print(f"  FROZEN: {j['id']}")
        print("freeze applied. Re-snapshot now, then proceed to migration. Rollback: --thaw")
    elif args.thaw:
        from cron import jobs
        for j in matches:
            if not j.get("enabled", True):
                jobs.update_job(j["id"], {"enabled": True})
                print(f"  THAWED: {j['id']}")
    else:
        print("\n(report-only — pass --freeze to pause these in the cutover sequence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
