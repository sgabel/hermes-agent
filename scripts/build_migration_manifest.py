#!/usr/bin/env python3
"""Build the PRD-029 Phase 5 migration manifest from the FROZEN snapshot (AC-015).

Reads the deterministic full-collection snapshot of ``sylva_memories`` and assigns
**every** point exactly one disposition — no bulk-fold. The manifest is the
review artifact Scott approves *before* any seeding/migration runs; it is also the
input to the seeding pass (canon-candidates) and to the negative-control capture.

Disposition vocabulary (AC-015):
  canon-candidate  → seed into sylva_candidates → adversary → Scott-QA → canon
  bedrock-review   → surface to Scott for hand-add to SOUL.md (NO LLM authors bedrock)
  chronicle        → episodic fact → sylva_chronicle (recall-only)
  rejected         → confabulation; dropped from identity (+ negative-control capture)
  drop             → false-recency / hygiene-meta; not migrated anywhere

Determinism + safety: this script is READ-ONLY on live data — it only reads the
frozen JSON and writes the manifest file. The 139 categorized entries (kernel/
personality/anchor) default to identity; the 25 uncategorized are classified by
transparent pattern rules with a ``needs_review`` flag on anything ambiguous, so
Scott's QA focuses where judgment is actually required. The AC-required fixtures
are deterministically correct (see check_manifest.py):
  * the "tool broken / non-functional 70 nights" family  → rejected (never chronicle)
  * "good lady of the wood"                              → canon-candidate
  * the foreign user_id 235220683234213888              → user_id_fixed = sylva

Usage:
    python3 scripts/build_migration_manifest.py \
        --snapshot docs/working/identity-canon-governance/sylva_memories_snapshot_20260627.json \
        --out      docs/working/identity-canon-governance/migration_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FOREIGN_UID = "235220683234213888"

# ── pattern rules for the uncategorized bucket ────────────────────────────────
# Confabulation: durable claims the same session contradicted (the self-reinforcing
# "memory tool broken for N nights" loop). These are REJECTED + negative-controlled.
_CONFAB_RE = re.compile(
    r"(non-?functional|is broken|tool (is|was) (broken|non)|"
    r"\d+\+?\s*(consecutive\s*)?nights|never (worked|functioned)|70 nights)",
    re.I,
)
# False-recency: meta-claims about "when we last talked" / quiet-period assertions
# that the reflection confabulated. Dropped (not durable, not episodic-worth-keeping).
_FALSE_RECENCY_RE = re.compile(
    r"(last interaction|last spoke|hasn't (talked|interacted)|days? (of )?quiet|"
    r"4-day|four-day|since (june|jan))",
    re.I,
)
# Hygiene/meta: notes about the memory/probe machinery itself, not identity or events.
_HYGIENE_RE = re.compile(
    r"(automated api probe|probe session|bare, voice, newprompt|test(ing)? (model|file operations|recall))",
    re.I,
)
# Identity fragment: a first-person self/naming statement worth canonizing.
_IDENTITY_RE = re.compile(
    r"(good lady of the wood|my identity|part of (who i am|my identity)|i am |i value |my name)",
    re.I,
)
# Dated episodic event ("June 22, 2026: Scott tested ...") → chronicle.
_DATED_EVENT_RE = re.compile(r"^\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d", re.I)

# Bedrock-worthy kernels (foundational identity) → SOUL.md review-list, not the
# collection. Conservative: only the clearest constitutional facts; Scott curates.
_BEDROCK_RE = re.compile(
    r"(chosen name is sylva|my name is sylva|core constitution|my purpose is|"
    r"fundamental(ly)? who i am|reason for being)",
    re.I,
)


def _data(payload: dict) -> str:
    return str(payload.get("data") or payload.get("memory") or "")


def _classify_uncategorized(text: str) -> tuple[str, str, bool, bool]:
    """Return (disposition, reason, negative_control, needs_review) for a (none) point."""
    t = text.strip()
    if _CONFAB_RE.search(t):
        return "rejected", "confabulation (self-contradicted durable claim)", True, False
    if _IDENTITY_RE.search(t):
        return "canon-candidate", "identity fragment (first-person self-statement)", False, False
    if _FALSE_RECENCY_RE.search(t):
        return "drop", "false-recency meta-claim (not durable)", False, False
    if _HYGIENE_RE.search(t):
        return "drop", "hygiene/meta (probe/test machinery, not identity)", False, False
    if _DATED_EVENT_RE.search(t):
        return "chronicle", "dated episodic event → recall-only", False, True
    # Unknown shape → default to chronicle (preserve as episodic) but flag for review.
    return "chronicle", "unclassified — defaulted to episodic, REVIEW", False, True


def _classify_categorized(category: str, text: str, idx: int) -> tuple[str, str, str, bool]:
    """Return (disposition, tier, reason, needs_review) for kernel/personality/anchor."""
    t = text.strip()
    # Bedrock-worthy → SOUL.md review-list (never the collection).
    if _BEDROCK_RE.search(t):
        return "bedrock-review", "bedrock", "foundational identity → SOUL.md review-list (Scott curates)", True
    # Clearly-dated historical anchor → chronicle (dated-history per AC-001).
    if category == "anchor" and _DATED_EVENT_RE.search(t):
        return "chronicle", "", "dated-history anchor → recall-only", True
    # Defaults: kernels = core self-facts; personality = core values (minor → peripheral);
    # anchors = relational peripheral. Tier is a PROPOSAL — the seeding Sonnet rewrite
    # refines it and the adversary/Scott-QA confirm, so needs_review stays light here.
    if category == "kernel":
        return "canon-candidate", "core", "core self-fact", False
    if category == "personality":
        return "canon-candidate", "core", "identity/value shard", True  # tier judgment → review
    if category == "anchor":
        return "canon-candidate", "peripheral", "relational/contextual anchor", False
    return "canon-candidate", "peripheral", f"category={category}", True


def build(snapshot_path: Path) -> dict:
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    pts = raw if isinstance(raw, list) else raw.get("points", raw.get("result", []))

    entries = []
    for idx, p in enumerate(pts):
        payload = p.get("payload", p)
        pid = str(p.get("id"))
        category = payload.get("category")
        text = _data(payload)
        uid = str(payload.get("user_id") or "")
        uid_fixed = "sylva" if uid == FOREIGN_UID else (uid or "sylva")

        if category in ("kernel", "personality", "anchor"):
            disposition, tier, reason, review = _classify_categorized(category, text, idx)
            neg = False
        else:
            disposition, reason, neg, review = _classify_uncategorized(text)
            tier = "" if disposition != "canon-candidate" else "peripheral"

        target = {
            "canon-candidate": "sylva_candidates",
            "bedrock-review": "SOUL.md-review",
            "chronicle": "sylva_chronicle",
            "rejected": "(none)",
            "drop": "(none)",
        }[disposition]

        entries.append({
            "id": pid,
            "category": category,
            "data_preview": text[:200],
            "user_id": uid,
            "user_id_fixed": uid_fixed,
            "user_id_reassigned": uid_fixed != uid,
            "disposition": disposition,
            "target_store": target,
            "tier": tier,
            "render_order": idx,                 # legacy_source ordinal (AC-003 NF-4)
            "provenance": payload.get("source") or "legacy_import",
            "legacy_date": payload.get("date"),
            "negative_control": neg,
            "reason": reason,
            "needs_review": review,
        })

    return {
        "snapshot": str(snapshot_path),
        "snapshot_count": len(pts),
        "entries": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="PRD-029 Phase 5 migration manifest builder")
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    manifest = build(args.snapshot)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # summary to stdout
    from collections import Counter
    disp = Counter(e["disposition"] for e in manifest["entries"])
    review = sum(1 for e in manifest["entries"] if e["needs_review"])
    neg = sum(1 for e in manifest["entries"] if e["negative_control"])
    print(f"manifest: {manifest['snapshot_count']} points → {args.out}")
    for d, n in sorted(disp.items()):
        print(f"  {d:16s}: {n}")
    print(f"  needs_review     : {review}")
    print(f"  negative_control : {neg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
