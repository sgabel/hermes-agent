#!/usr/bin/env python3
"""PRD-029 decommission — migrate the chronicle-disposition points from the
frozen ``sylva_memories`` snapshot into ``sylva_chronicle``.

This is the *only* sanctioned migration path. It does NOT use ``mem0_add`` (which
is retired) — it re-embeds via TEI and upserts directly into Qdrant with the
chronicle journal schema, exactly like ``ChronicleSearcher`` expects.

Safety / correctness (per adversarial review 2026-06-28):
  * Joins each manifest entry back to the FULL snapshot text (the manifest only
    stores ``data_preview[:200]``).
  * Deterministic point IDs (uuid5 of the original id) → idempotent re-runs.
  * Content-hash de-dupe against the live chronicle → never double-inserts.
  * Robust date: parse a leading "Month DD, YYYY" or any "YYYY-MM-DD" in the
    text, else fall back to the point's ``created_at`` date. A point with no
    derivable date is still inserted (semantic search finds it) but is FLAGGED —
    date-filtered chronicle_search would miss it.
  * Honors a per-id disposition OVERRIDE map (the "please delete" test marker is
    forced to drop even though the classifier defaulted it to chronicle).
  * Dry-run by default. ``--apply`` writes. Prints before/after counts and a
    per-row report; ``--verify`` runs a chronicle search per migrated row.

Usage:
    python3 scripts/migrate_sylva_memories_to_chronicle.py \
        --snapshot docs/working/identity-canon-governance/sylva_memories_snapshot_20260628.json \
        --manifest docs/working/identity-canon-governance/migration_manifest_20260628.json \
        [--qdrant http://localhost:6333] [--tei http://localhost:8085] \
        [--apply] [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

COLLECTION = "sylva_chronicle"
SOURCE_TAG = "journal:migrated_sylva_memories_20260628"
FOREIGN_UID = "235220683234213888"
# Stable namespace so re-runs produce identical point ids (idempotent upsert).
_NS = uuid.UUID("6f29a1c4-0d2e-4e7a-9b13-prd029mem0".replace("prd029mem0", "abcdef012345"))

# Per-id disposition overrides applied ON TOP of the manifest classifier.
# 3d38a41c… is the literal "MEMORY RESTORATION TEST … Please delete this entry"
# marker — the classifier defaulted it to chronicle; it must drop.
DISPOSITION_OVERRIDES: Dict[str, str] = {
    # "MEMORY RESTORATION TEST … Please delete this entry" — disposable marker.
    "3d38a41c-7da9-4229-8dcd-5e8686af3da7": "drop",
    # Two "REMOVED: toddler with loaded gun analogy …" tombstones — deletion
    # markers from the old store, not episodic memories. Don't carry into chronicle.
    "2796e3e6-e1c2-4745-b957-c6436fc76969": "drop",
    "828014f8-9a07-44b9-bdc9-ce2a7f27b878": "drop",
}

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
# No \b boundaries: an ISO created_at like "2026-06-27T14:..." has no word
# boundary between the day digits and the "T", which would defeat a trailing \b.
_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})", re.I)


def derive_date(text: str, created_at: Optional[str]) -> tuple[str, str]:
    """Return (YYYY-MM-DD, how) — how ∈ {iso, month, created_at, none}."""
    m = _ISO_RE.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "iso"
    m = _MONTH_RE.search(text)
    if m:
        mon = _MONTHS.get(m.group(1).lower()[:3] if len(m.group(1)) >= 3 else m.group(1).lower())
        # map 3-letter back
        mon = _MONTHS.get(m.group(1).lower()) or {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }.get(m.group(1).lower()[:3])
        if mon:
            return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}", "month"
    if created_at:
        # created_at may be ISO ("2026-06-27T...") or epoch-ish; handle ISO prefix.
        cm = _ISO_RE.search(str(created_at))
        if cm:
            return f"{cm.group(1)}-{cm.group(2)}-{cm.group(3)}", "created_at"
        try:
            dt = datetime.fromtimestamp(float(created_at))
            return dt.strftime("%Y-%m-%d"), "created_at"
        except Exception:
            pass
    return "", "none"


def embed(tei_url: str, text: str) -> List[float]:
    resp = requests.post(f"{tei_url.rstrip('/')}/embed",
                         json={"inputs": text, "truncate": True}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) and data and isinstance(data[0], list) else data


def collection_count(qdrant: str, coll: str) -> int:
    r = requests.get(f"{qdrant.rstrip('/')}/collections/{coll}", timeout=10)
    r.raise_for_status()
    return r.json()["result"]["points_count"]


def existing_hashes(qdrant: str, coll: str) -> set[str]:
    """Pull every chronicle point's content hash (md5 of data) for de-dupe."""
    hashes: set[str] = set()
    offset = None
    while True:
        body: Dict[str, Any] = {"limit": 1000, "with_payload": ["data", "hash"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{qdrant.rstrip('/')}/collections/{coll}/points/scroll",
                          json=body, timeout=30)
        r.raise_for_status()
        res = r.json()["result"]
        for p in res["points"]:
            pl = p.get("payload", {})
            h = pl.get("hash") or hashlib.md5((pl.get("data") or "").encode()).hexdigest()
            hashes.add(h)
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--tei", default="http://localhost:8085")
    ap.add_argument("--apply", action="store_true", help="write to Qdrant (default: dry-run)")
    ap.add_argument("--verify", action="store_true", help="chronicle_search each migrated row after apply")
    args = ap.parse_args()

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    by_id = {str(p["id"]): p for p in snapshot}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    # Select chronicle-disposition entries, applying overrides.
    selected = []
    overridden_out = []
    for e in manifest["entries"]:
        pid = e["id"]
        disp = DISPOSITION_OVERRIDES.get(pid, e["disposition"])
        if e["disposition"] == "chronicle" and disp != "chronicle":
            overridden_out.append((pid, disp, e["data_preview"][:60]))
            continue
        if disp == "chronicle":
            selected.append(e)

    print(f"manifest chronicle rows: {sum(1 for e in manifest['entries'] if e['disposition']=='chronicle')}")
    print(f"overridden OUT of chronicle: {len(overridden_out)}")
    for pid, disp, prev in overridden_out:
        print(f"  - {pid} -> {disp}: {prev!r}")
    print(f"to migrate: {len(selected)}\n")

    seen_hashes = existing_hashes(args.qdrant, COLLECTION)
    before = collection_count(args.qdrant, COLLECTION)
    print(f"sylva_chronicle count BEFORE: {before}\n")

    points = []
    dateless = []
    skipped_dup = []
    for e in selected:
        pid = e["id"]
        snap = by_id.get(pid)
        if not snap:
            print(f"  !! {pid} not in snapshot — SKIP", file=sys.stderr)
            continue
        payload = snap.get("payload", {})
        text = str(payload.get("data") or payload.get("memory") or "").strip()
        if not text:
            print(f"  !! {pid} empty data — SKIP", file=sys.stderr)
            continue
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_hashes:
            skipped_dup.append(pid)
            continue
        date, how = derive_date(text, payload.get("created_at"))
        if not date:
            dateless.append(pid)
        det_id = str(uuid.uuid5(_NS, f"migrated:{pid}"))
        new_payload = {
            "data": text,
            "speaker": "sylva",
            "date": date,
            "source": SOURCE_TAG,
            "category": "journal",
            "user_id": "sylva",
            "role": "user",
            "hash": h,
            "migrated_from": pid,
            "migrated_date_source": how,
        }
        points.append({"id": det_id, "payload": new_payload, "_text": text, "_date": date, "_how": how})

    # Report
    print("=== migration plan ===")
    for p in points:
        print(f"  {p['id'][:8]} date={p['_date'] or '(none)':<12} via={p['_how']:<10} "
              f"len={len(p['_text']):>4}  {p['_text'][:70]!r}")
    print(f"\nplanned inserts: {len(points)}  | skipped (content dup): {len(skipped_dup)} "
          f"| dateless (semantic-only): {len(dateless)}")
    if dateless:
        print(f"  dateless ids: {[d[:8] for d in dateless]}")
    if skipped_dup:
        print(f"  dup ids: {[d[:8] for d in skipped_dup]}")

    if not args.apply:
        print("\nDRY RUN — no writes. Re-run with --apply to migrate.")
        return 0

    # Embed + upsert
    print("\n=== applying (embed via TEI + upsert) ===")
    upsert_points = []
    for p in points:
        vec = embed(args.tei, p["_text"])
        upsert_points.append({"id": p["id"], "vector": vec, "payload": p["payload"]})
    r = requests.put(
        f"{args.qdrant.rstrip('/')}/collections/{COLLECTION}/points?wait=true",
        json={"points": upsert_points}, timeout=120)
    r.raise_for_status()
    print(f"upsert status: {r.json().get('result', {}).get('status')}")

    after = collection_count(args.qdrant, COLLECTION)
    print(f"sylva_chronicle count AFTER: {after}  (delta +{after - before}, expected +{len(points)})")

    if args.verify:
        print("\n=== verify (chronicle search per migrated row) ===")
        ok = 0
        for p in points:
            vec = embed(args.tei, p["_text"][:200])
            sr = requests.post(
                f"{args.qdrant.rstrip('/')}/collections/{COLLECTION}/points/search",
                json={"vector": vec, "limit": 3, "with_payload": ["migrated_from"]}, timeout=20)
            ids = [hit.get("payload", {}).get("migrated_from") for hit in sr.json().get("result", [])]
            hit = p["payload"]["migrated_from"] in ids
            ok += hit
            print(f"  {p['id'][:8]} findable={hit}")
        print(f"\nverified findable: {ok}/{len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
