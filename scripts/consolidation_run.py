#!/usr/bin/env python3
"""Run the PRD-029 Phase 3 identity-consolidation pass (governed candidate proposer).

The consolidation engine reads recent user-facing sessions (cron excluded,
deterministic) + the structured agency layer (PRD-028 ledger / kanban / work-block
records), asks the neutral derivation model to PROPOSE identity candidate-deltas,
and writes them as ``status: candidate`` into ``sylva_candidates`` via the bespoke
direct-Qdrant CanonStore (the SOLE writer — AC-004/AC-017). It never writes
``sylva_canon`` and never touches SOUL.md; promotion is Phase 4's ratification gate.

Usage (host, venv active):

    python3 scripts/consolidation_run.py            # live run → sylva_candidates
    python3 scripts/consolidation_run.py --dry-run  # derive + report, no write
    python3 scripts/consolidation_run.py --sandbox sylva_lab   # AC-010 validation

Cron job spec (NOT armed — owner-gated until the Phase 4 ratification gate exists;
arming an autonomous memory writer before the gate resumes the ungoverned-autosave
loop PRD-029 exists to kill). When Phase 4 lands, register via the cron infra:

    name:    "Nightly identity consolidation"
    script:  scripts/consolidation_run.py        # resides in HERMES_HOME/scripts/
    prompt:  "Report the consolidation summary above to Scott."
    model:   <model.second_opinion_model>        # neutral; or inherit main 35B
    deliver: local
    schedule: nightly

The derivation model is configurable via ``auxiliary.canon_consolidation.model``
and defaults to the neutral main model (local 35B in the isolated container).
``model.second_opinion_model`` (Sonnet) needs the PRD-026 egress allowlist before
it is reachable from the network-isolated hermes container.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.memory.canon.consolidation import run_consolidation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="PRD-029 identity consolidation pass")
    ap.add_argument("--dry-run", action="store_true", help="derive + report, no write")
    ap.add_argument(
        "--sandbox",
        metavar="COLLECTION",
        default=None,
        help="write to a sandbox collection (e.g. sylva_lab) instead of sylva_candidates",
    )
    ap.add_argument(
        "--limit-sessions",
        type=int,
        default=12,
        help="max recent user-facing sessions to read (default 12)",
    )
    ap.add_argument(
        "--cron",
        action="store_true",
        help="tag the audit entry surface as 'cron' (set when run by the scheduler)",
    )
    ap.add_argument(
        "--include-chronicle",
        action="store_true",
        help="PRD-051: also source the episodic chronicle (omitted -> the "
        "memory.consolidation_chronicle_source config knob decides, default off)",
    )
    ap.add_argument(
        "--chronicle-days",
        type=int,
        default=7,
        help="chronicle lookback window in days (default 7; only ~21 points are "
        "<=7 days old today — widen for early knob-on trials)",
    )
    args = ap.parse_args()

    kwargs = {
        "dry_run": args.dry_run,
        "limit_sessions": args.limit_sessions,
        "surface": "cron" if args.cron else "cli",
        "chronicle_days": args.chronicle_days,
    }
    if args.include_chronicle:
        kwargs["include_chronicle"] = True
    if args.sandbox:
        kwargs["target_collection"] = args.sandbox

    result = run_consolidation(**kwargs)
    print(result.summary())
    if result.candidate_ids:
        print(f"  candidate ids: {', '.join(result.candidate_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
