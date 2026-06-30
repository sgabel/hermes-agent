#!/usr/bin/env python3
"""Hermes-cron entrypoint for the PRD-038 identity-consolidation pass.

WHY THIS WRAPPER (vs. pointing cron straight at ``consolidation_run.py``):
Hermes cron only runs scripts that resolve within ``$HERMES_HOME/scripts/``
(``~/.hermes/scripts/`` on the host; ``/opt/data/scripts/`` in the container —
``tools/cronjob_tools.py:_validate_cron_script_path``), and the scheduler invokes
them as ``[sys.executable, <path>]`` with **no argv** and ``cwd`` = that scripts dir
(``cron/scheduler.py:_run_job_script``). So the cron job cannot pass ``--cron`` to
``consolidation_run.py``; this wrapper hard-codes the cron surface instead. It lives
in the repo as the canonical source and is COPIED to ``$HERMES_HOME/scripts/`` at
arm time (see the PRD-038 "Arming the cadence" runbook) — it is intentionally NOT
auto-installed.

GOVERNANCE: running this through Hermes cron is the point (PRD-038 M4) — the
scheduler gates it via ``guard("cron_script", ...)`` (capability policy) and honours
the QUIESCE kill switch at tick + dispatch, so the consolidation writer inherits the
PRD-028 containment for free. It is **queue-only**: it proposes into
``sylva_candidates`` (never ``sylva_canon``); promotion stays the human cockpit-ratify
step. ``run_consolidation`` records its own PRD-028 audit-ledger entry (surface=cron).
"""

from __future__ import annotations

import sys


def _import_run_consolidation():
    """Import the consolidation engine.

    In the container the hermes-agent package is venv-installed, so the plain
    import resolves regardless of cwd. The fallback covers a source/host layout
    where the package root must be put on the path explicitly.
    """
    try:
        from plugins.memory.canon.consolidation import run_consolidation
        return run_consolidation
    except ModuleNotFoundError:
        from pathlib import Path

        # Common roots: the baked container install, then a repo checkout relative
        # to this file (…/hermes-agent/scripts/consolidation_cron.py → …/hermes-agent).
        for root in ("/opt/hermes", str(Path(__file__).resolve().parent.parent)):
            if root not in sys.path:
                sys.path.insert(0, root)
        from plugins.memory.canon.consolidation import run_consolidation
        return run_consolidation


def main() -> int:
    run_consolidation = _import_run_consolidation()
    # surface="cron" tags the audit-ledger entry with the real invocation context.
    result = run_consolidation(surface="cron")
    print(result.summary())
    if result.candidate_ids:
        print(f"  candidate ids: {', '.join(result.candidate_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
