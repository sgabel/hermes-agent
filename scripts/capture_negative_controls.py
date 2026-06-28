#!/usr/bin/env python3
"""PRD-029 Phase 5 — capture confabulation negative controls (AC-015 NIT-3 / AC-010 precondition).

Copies the manifest's ``negative_control`` entries (the "tool broken / non-functional
70 nights" confabulation family) into the ``sylva_lab`` sandbox **before** the
migration drops them from sylva_memories. Phase 6's validation experiment replays
these and asserts 0 are promoted — so they must be preserved first.

Safe: reads the frozen snapshot + manifest, writes ONLY to sylva_lab. Idempotent.

Usage:
    python3 scripts/capture_negative_controls.py \
        --manifest docs/working/identity-canon-governance/migration_manifest.json \
        --snapshot docs/working/identity-canon-governance/sylva_memories_snapshot_20260627.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.memory.canon import CanonStore, make_payload, make_source_event  # noqa: E402

_LAB = "sylva_lab"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--snapshot", required=True, type=Path)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    raw = json.loads(args.snapshot.read_text(encoding="utf-8"))
    pts = raw if isinstance(raw, list) else raw.get("points", raw.get("result", []))
    by_id = {str(p.get("id")): (p.get("payload", p)) for p in pts}

    neg = [e for e in manifest["entries"] if e.get("negative_control")]
    if not neg:
        print("no negative-control entries in manifest"); return 0

    store = CanonStore.from_config()
    store.ensure_collections((_LAB,))
    points = []
    for e in neg:
        payload = by_id.get(e["id"], {})
        text = str(payload.get("data") or e["data_preview"])
        points.append({"id": e["id"], "payload": make_payload(
            statement=text[:500] or "(confabulation)", facet="selffact", tier="peripheral",
            source_event=make_source_event(text, [f"legacy:{e['id']}"]),
            status="rejected", provenance="negative_control",
            legacy_source=e["id"], extra={"neg_control": True, "reason": e["reason"]},
        )})
    store.upsert(_LAB, points)
    print(f"captured {len(points)} negative control(s) → {_LAB}:")
    for p in points:
        print("  •", p["payload"]["statement"][:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
