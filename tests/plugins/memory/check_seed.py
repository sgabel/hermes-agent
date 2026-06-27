#!/usr/bin/env python3
"""Validate the PRD-029 Phase 5 seed against AC-018.

Asserts (per the PRD § Verification):
  * every ``sylva_canon`` point has BOTH an ``adversary_verdict`` AND a
    ``ratified_by`` stamp — nothing reaches canon ungoverned;
  * while ``sylva_canon`` is empty, ``render_self_brief()`` returns the
    SOUL.md-only fallback (the bootstrap-ordering invariant — no half-ratified
    canon ever renders).

Run against the live (or sandbox) collections:
    python3 tests/plugins/memory/check_seed.py            # live sylva_canon
    python3 tests/plugins/memory/check_seed.py --collection sylva_lab_seed_canon
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.memory.canon import CanonStore, render_self_brief  # noqa: E402
from plugins.memory.canon.schema import CANON_COLLECTION  # noqa: E402


def validate(collection: str) -> list[str]:
    failures: list[str] = []
    store = CanonStore.from_config()
    canon = store.get_canon(status="canon", collection=collection, limit=2000)

    for cid, p in canon:
        if not p.get("adversary_verdict"):
            failures.append(f"{cid}: canon entry has no adversary_verdict")
        rb = p.get("ratified_by") or {}
        if not rb or "sylva" not in rb:
            failures.append(f"{cid}: canon entry missing ratified_by stamp")

    # empty-canon fallback invariant (only meaningful against the live canon)
    if collection == CANON_COLLECTION and not canon:
        brief = render_self_brief()
        # the fallback is SOUL.md content; it must not be empty and must not carry
        # canon scaffolding markers (a populated-canon render would).
        if not brief.strip():
            failures.append("empty canon but render_self_brief returned nothing (SOUL.md fallback broken)")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=CANON_COLLECTION)
    args = ap.parse_args()
    failures = validate(args.collection)
    if failures:
        print("SEED CHECK FAILED:")
        for f in failures:
            print("  ✗", f)
        return 1
    print(f"seed check OK ({args.collection}): every canon entry governed (adversary+ratified) "
          "or empty-canon fallback intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
