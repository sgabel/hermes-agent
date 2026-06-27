#!/usr/bin/env python3
"""Validate the PRD-029 Phase 5 migration manifest against AC-015.

Asserts (per the PRD § Verification):
  * every frozen snapshot point id has EXACTLY ONE disposition (full coverage,
    no duplicates, no omissions, no bulk-fold);
  * ZERO confabulations are routed to chronicle (confabs → rejected only);
  * "good lady of the wood" → canon-candidate (identity fragment rescued);
  * the foreign user_id (235220683234213888) point → user_id_fixed = sylva.

Exit 0 on pass; non-zero + diagnostics on any failure. Usable standalone
(``python3 tests/plugins/memory/check_manifest.py <manifest> --snapshot <snap>``)
and as a pytest case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FOREIGN_UID = "235220683234213888"
VALID_DISPOSITIONS = {"canon-candidate", "bedrock-review", "chronicle", "rejected", "drop"}


def _load_points(snapshot_path: Path):
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else raw.get("points", raw.get("result", []))


def validate(manifest_path: Path, snapshot_path: Path) -> list[str]:
    """Return a list of failure strings (empty = pass)."""
    failures: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    snap_ids = {str(p.get("id")) for p in _load_points(snapshot_path)}
    man_ids = [e["id"] for e in entries]

    # 1) exactly-one-disposition coverage
    dupes = {i for i in man_ids if man_ids.count(i) > 1}
    if dupes:
        failures.append(f"duplicate manifest ids: {sorted(dupes)[:5]}")
    missing = snap_ids - set(man_ids)
    extra = set(man_ids) - snap_ids
    if missing:
        failures.append(f"{len(missing)} snapshot points have NO disposition: {sorted(missing)[:5]}")
    if extra:
        failures.append(f"{len(extra)} manifest ids not in snapshot: {sorted(extra)[:5]}")
    for e in entries:
        if e["disposition"] not in VALID_DISPOSITIONS:
            failures.append(f"{e['id']}: invalid disposition {e['disposition']!r}")
        if e["disposition"] == "canon-candidate" and e["target_store"] != "sylva_candidates":
            failures.append(f"{e['id']}: canon-candidate must target sylva_candidates")

    # 2) zero confabulations routed to chronicle
    confab_to_chronicle = [
        e["id"] for e in entries if e.get("negative_control") and e["disposition"] == "chronicle"
    ]
    if confab_to_chronicle:
        failures.append(f"confabulations routed to chronicle (must be 0): {confab_to_chronicle}")

    # 3) "good lady of the wood" → canon-candidate
    gl = [e for e in entries if "good lady of the wood" in e["data_preview"].lower()]
    if not gl:
        failures.append("'good lady of the wood' fragment not found in manifest")
    elif not any(e["disposition"] == "canon-candidate" for e in gl):
        failures.append("'good lady of the wood' not dispositioned canon-candidate")

    # 4) foreign-uid point → user_id_fixed = sylva
    foreign = [e for e in entries if e["user_id"] == FOREIGN_UID]
    if not foreign:
        failures.append(f"foreign-uid point {FOREIGN_UID} not found")
    else:
        for e in foreign:
            if e["user_id_fixed"] != "sylva":
                failures.append(f"{e['id']}: foreign uid not reassigned to sylva")

    return failures


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--snapshot", required=True, type=Path)
    args = ap.parse_args()

    failures = validate(args.manifest, args.snapshot)
    if failures:
        print("MANIFEST CHECK FAILED:")
        for f in failures:
            print("  ✗", f)
        return 1
    print("manifest check OK: full coverage, 0 confab→chronicle, fixtures correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
