#!/usr/bin/env python3
"""Run the PRD-029 Phase 4 ratification gate (adversary → route → canon).

For each ``status:candidate`` row in ``sylva_candidates`` (written by the Phase-3
consolidation pass), runs the six-check adversary ("skeptic-Sylva"), then routes
the verdict: affirm→canon, refute→rejected, tension→surfaced, demote→re-tier,
merge→fold. The SOLE writer into ``sylva_canon``; never writes SOUL.md. Every
mutation is logged to the PRD-028 ledger; nothing is hard-deleted.

Usage (host, venv active):

    python3 scripts/ratification_run.py            # ratify the pending queue
    python3 scripts/ratification_run.py --dry-run  # adversary + route plan, no write

The adversary model is configurable via ``auxiliary.canon_adversary.*`` and
defaults to the neutral main model. Cross-model doctrine favours a different
model from the Phase-3 proposer; point it at ``model.second_opinion_model``
(Sonnet) once the PRD-026 egress allowlist makes it reachable in-container.

⚠️ NOT armed as a cron. Like Phase 3, arming the autonomous gate is owner-gated.
The default ratify hook auto-stamps Sylva (sovereign meaning); Phase 5 seeding
injects a Scott-batch-QA gate before any legacy entry reaches canon.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.memory.canon.ratification import run_ratification  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="PRD-029 canon ratification gate")
    ap.add_argument("--dry-run", action="store_true", help="adversary + route plan, no write")
    ap.add_argument("--limit", type=int, default=200, help="max candidates to process")
    args = ap.parse_args()

    result = run_ratification(dry_run=args.dry_run, limit=args.limit)
    print(result.summary())
    for m in result.mutations:
        line = f"  {m.verdict:8s} → {m.action:10s} {m.candidate_id}"
        if m.note:
            line += f"  ({m.note})"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
