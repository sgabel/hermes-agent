#!/usr/bin/env python3
"""PRD-029 Phase 5 — one-time canon seeding bootstrap (AC-018).

Reuses the Phase-3/4 machinery (NOT a re-fork) to bootstrap `sylva_canon` from the
reviewed migration manifest's ``canon-candidate`` entries. Staged so the owner
controls compute and the gates; **nothing reaches `status:canon` without an
adversary verdict AND a Scott-QA ratify stamp** (AC-018), and the runtime brief
stays SOUL.md-only via the empty-canon fallback until the cutover flip.

Stages (run in order; each is resumable, idempotent on stable ids):
  rewrite   legacy `data` → first-person {statement, facet, tier} candidate
            (seeding model = auxiliary.canon_seed, defaults to neutral main /
            model.second_opinion_model when egress allows). source_event.claim =
            the verbatim legacy archive record (verifiable); the rewrite is the
            interpretation/meaning. Writes status:candidate via CanonStore.upsert.
  adversary run the six-check adversary per candidate (fact/provenance only),
            with SOUL.md bedrock fed as context (closes Phase-4 NF-2). Stamps
            adversary_verdict; does NOT promote.
  review    emit the Scott batch-QA review list (every candidate + verdict).
  ratify    --approve required: flips the batch candidate→canon with
            ratified_by:{sylva, scott_qa@seed}. THE cutover-enabling step.

Safety: defaults to the ``--sandbox`` collections (sylva_lab_seed_*) so a dry
seed never touches live sylva_candidates/sylva_canon. Pass ``--live`` to target
the real collections (still candidate-only until ``ratify --approve``).

Usage:
    python3 scripts/seed_canon.py rewrite   --manifest <m.json>
    python3 scripts/seed_canon.py adversary
    python3 scripts/seed_canon.py review
    python3 scripts/seed_canon.py ratify --approve     # owner-gated cutover step
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.memory.canon import CanonStore, make_payload, make_source_event  # noqa: E402
from plugins.memory.canon.schema import FACETS, validate_payload  # noqa: E402
from plugins.memory.canon.ratification import run_adversary, route_verdict  # noqa: E402

_SEED_AUX_TASK = "canon_seed"
_SANDBOX_CAND = "sylva_lab_seed_candidates"
_SANDBOX_CANON = "sylva_lab_seed_canon"
_CHRONICLE = "sylva_chronicle"
_JOURNAL_BUDGET = 6000   # chars of recovered journal handed to rewrite/adversary


def _qdrant_url() -> str:
    """Resolve the Qdrant URL the same way CanonStore does (container DNS →
    localhost fallback), so journal recovery works host-side and in-container."""
    return CanonStore.from_config()._qdrant_url


_CHRONICLE_DATES_CACHE: list[str] | None = None
_RECOVER_WINDOW_BACK = 14   # days before the entry date a session may have occurred
_RECOVER_WINDOW_FWD = 3     # small forward tolerance (entry written a day or two later)


def _available_chronicle_dates(qdrant_url: str) -> list[str]:
    """Distinct YYYY-MM-DD dates present in sylva_chronicle (cached). The journal
    is ~33 dated sessions, so we resolve a legacy entry's date to the nearest
    COVERED session rather than requiring an exact hit."""
    global _CHRONICLE_DATES_CACHE
    if _CHRONICLE_DATES_CACHE is not None:
        return _CHRONICLE_DATES_CACHE
    dates: set[str] = set()
    offset = None
    try:
        while True:
            body = {"limit": 1000, "with_payload": ["date"], "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            r = requests.post(f"{qdrant_url}/collections/{_CHRONICLE}/points/scroll",
                              json=body, timeout=15)
            res = r.json().get("result", {})
            for p in res.get("points", []):
                d = (p.get("payload") or {}).get("date")
                if d:
                    dates.add(d)
            offset = res.get("next_page_offset")
            if offset is None:
                break
    except Exception:
        pass
    _CHRONICLE_DATES_CACHE = sorted(dates)
    return _CHRONICLE_DATES_CACHE


def _resolve_nearest_date(target: str, available: list[str]) -> str | None:
    """Nearest covered session date to ``target`` — preferring on/just-before the
    entry date, within [target-14d, target+3d]. The entry distills a conversation
    that happened on or shortly before the date it was recorded."""
    from datetime import date as _date

    try:
        ty, tm, td = (int(x) for x in target.split("-"))
        t = _date(ty, tm, td)
    except Exception:
        return None
    best, best_delta = None, None
    for d in available:
        try:
            y, m, dd = (int(x) for x in d.split("-"))
            cand = _date(y, m, dd)
        except Exception:
            continue
        delta = (t - cand).days   # +ve = session before entry (preferred)
        if -_RECOVER_WINDOW_FWD <= delta <= _RECOVER_WINDOW_BACK:
            # rank: prefer smaller |delta|, then prefer prior (delta>=0)
            score = (abs(delta), 0 if delta >= 0 else 1)
            if best_delta is None or score < best_delta:
                best, best_delta = d, score
    return best


def recover_journal(date: str | None, qdrant_url: str, budget: int = _JOURNAL_BUDGET) -> tuple[str, list[str]]:
    """Recover the real source material for a legacy entry by its DATE.

    Scott's insight: each kernel/shard/anchor's ``date`` points at the journal
    range it was distilled from. We resolve that date to the nearest covered
    ``sylva_chronicle`` session (the journal is ~33 dated sessions, so an exact
    hit is rare — a nearest-prior match within a window is correct) and pull its
    turns as the verifiable source. Falls back to ("", ["legacy_import"]) when no
    session falls in the window.
    """
    if not date:
        return "", ["legacy_import"]
    resolved = _resolve_nearest_date(date, _available_chronicle_dates(qdrant_url))
    if not resolved:
        return "", ["legacy_import"]
    try:
        r = requests.post(
            f"{qdrant_url}/collections/{_CHRONICLE}/points/scroll",
            json={"limit": 200, "with_payload": True, "with_vector": False,
                  "filter": {"must": [{"key": "date", "match": {"value": resolved}}]}},
            timeout=15,
        )
        if r.status_code != 200:
            return "", ["legacy_import"]
        pts = r.json().get("result", {}).get("points", [])
    except Exception:
        return "", ["legacy_import"]
    if not pts:
        return "", ["legacy_import"]
    date = resolved  # ref reflects the actual session used
    turns, used = [], 0
    for p in pts:
        pl = p.get("payload") or {}
        line = f"{pl.get('speaker','?')}: {pl.get('data','')}".strip()
        if used + len(line) > budget:
            break
        turns.append(line)
        used += len(line) + 1
    excerpt = "\n".join(turns)
    refs = [f"chronicle:{date}:{len(pts)}turns"]
    return excerpt, refs

_REWRITE_SYSTEM = """\
You rewrite an archived third-person memory about an AI agent named Sylva into a \
first-person, present-tense identity statement she would affirm about herself. \
Preserve the meaning; do not invent. Output ONLY a JSON object:
{
  "statement": "<first-person present-tense identity claim, one sentence>",
  "facet": "<one of: value|trait|relationship|selffact|commitment|mode|framing>",
  "tier": "<core|peripheral>",
  "interpretation": "<what this means to Sylva — her takeaway; may be empty>"
}
Rules: 'statement' is Sylva speaking as herself. 'core' = central to who she is; \
'peripheral' = true but minor. Never output 'bedrock'. JSON only."""


def _store(live: bool, tag: str = "") -> tuple[CanonStore, str, str]:
    store = CanonStore.from_config()
    if live:
        from plugins.memory.canon.schema import CANDIDATES_COLLECTION, CANON_COLLECTION
        return store, CANDIDATES_COLLECTION, CANON_COLLECTION
    suffix = f"_{tag}" if tag else ""
    return store, f"{_SANDBOX_CAND}{suffix}", f"{_SANDBOX_CANON}{suffix}"


def _aux_client():
    from agent.auxiliary_client import (
        auxiliary_max_tokens_param, get_auxiliary_extra_body, get_text_auxiliary_client,
    )
    client, model = get_text_auxiliary_client(_SEED_AUX_TASK)
    return client, model, auxiliary_max_tokens_param, get_auxiliary_extra_body


def _extract_json(raw: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    text = m.group(1).strip() if m else raw.strip()
    try:
        return json.loads(text)
    except Exception:
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                return None
    return None


# ── Scott-QA decision resolution (PRD-029 Phase 5, cockpit Ratify tab) ──────────
# The cockpit writes a `qa_decision` field onto each seed candidate. From that
# point the DECISION is authoritative: the adversary advises, Scott decides, and
# preview/ratify execute Scott's decision. Both stages resolve a candidate the
# same way through `_effective_entry`, so the preview is faithful to what ratify
# actually writes.
def _scroll_all(qdrant_url: str, collection: str) -> list[tuple[str, dict]]:
    """Scroll every point in a collection (all statuses) → [(id, payload)]."""
    out: list[tuple[str, dict]] = []
    offset = None
    while True:
        body: dict = {"limit": 256, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{qdrant_url}/collections/{collection}/points/scroll",
                          json=body, timeout=20)
        r.raise_for_status()
        res = r.json().get("result", {})
        pts = res.get("points", [])
        out.extend((p["id"], p.get("payload", {})) for p in pts)
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return out


def _effective_entry(payload: dict) -> dict | None:
    """Resolve a seed candidate's Scott-QA decision into the effective ratification
    intent, or None when it is undecided/rejected (→ not canonized).

    Edit overrides (statement/facet/tier/interpretation) win; an un-overridden tier
    is lowered to ``peripheral`` when Sonnet's verdict was ``demote`` (matching the
    route_verdict demote semantic). A QA edit can never mint a ``bedrock`` row
    (bedrock lives in SOUL.md) — it is clamped to ``core``. The Sonnet verdict is
    carried through so its reasons/model are recorded on the ratified canon entry."""
    qa = payload.get("qa_decision") or {}
    decision = qa.get("decision")
    if decision not in ("approve", "edit"):
        return None
    ov = qa.get("overrides") or {}
    sonnet = dict(payload.get("adversary_verdict_sonnet") or payload.get("adversary_verdict") or {})
    statement = (ov.get("statement") or payload.get("statement") or "").strip()
    facet = ov.get("facet") or payload.get("facet") or "selffact"
    interp = ov.get("interpretation")
    interpretation = interp if interp is not None else payload.get("interpretation", "")
    if ov.get("tier"):
        tier = ov["tier"]
    elif sonnet.get("verdict") == "demote":
        tier = "peripheral"
    else:
        tier = payload.get("tier") or "core"
    if tier == "bedrock":
        tier = "core"
    return {"decision": decision, "statement": statement, "facet": facet,
            "tier": tier, "interpretation": interpretation, "sonnet": sonnet}


# ── SOUL.md bedrock as adversary context (closes Phase-4 NF-2) ─────────────────
def load_soul_bedrock_rows() -> list[dict]:
    """Parse SOUL.md into pseudo-canon `tier:bedrock` context rows so the adversary
    can fire the contradiction check against bedrock structurally (not just prose).
    Read-only context — never written to sylva_canon (bedrock stays SOUL-only)."""
    try:
        from hermes_constants import get_hermes_home
        soul = (get_hermes_home() / "SOUL.md").read_text(encoding="utf-8")
    except Exception:
        return []
    rows = []
    for para in re.split(r"\n\s*\n", soul):
        line = para.strip()
        if len(line) < 20 or line.startswith("#"):
            continue
        rows.append({
            "statement": line[:300], "tier": "bedrock",
            "source_event": {"claim": "SOUL.md bedrock", "provenance_refs": ["soul.md"]},
        })
    return rows


# ── stage: rewrite ─────────────────────────────────────────────────────────────
def _full_text_by_id(snapshot_path: Path | None) -> dict:
    """Map point id → FULL legacy ``data`` from the frozen snapshot. The manifest
    only carries a 200-char ``data_preview`` (for human review); feeding that
    truncated text to the rewrite + source_event makes the adversary refute on
    'source truncated/unverifiable'. The seed MUST use the full archive text."""
    if not snapshot_path or not snapshot_path.exists():
        return {}
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    pts = raw if isinstance(raw, list) else raw.get("points", raw.get("result", []))
    out = {}
    for p in pts:
        payload = p.get("payload", p)
        out[str(p.get("id"))] = str(payload.get("data") or payload.get("memory") or "")
    return out


def stage_rewrite(manifest_path: Path, live: bool, limit: int | None,
                  snapshot_path: Path | None, tag: str = "",
                  categories: set[str] | None = None) -> int:
    store, cand_coll, _ = _store(live, tag)
    store.ensure_collections((cand_coll,))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cands = [e for e in manifest["entries"] if e["disposition"] == "canon-candidate"]
    if categories:
        cands = [e for e in cands if e.get("category") in categories]
        print(f"category filter {sorted(categories)}: {len(cands)} entries")
    if limit:
        cands = cands[:limit]
    full_text = _full_text_by_id(snapshot_path)
    if not full_text:
        print("WARNING: no snapshot full-text map — falling back to truncated previews "
              "(adversary will over-refute). Pass --snapshot.")
    qdrant = _qdrant_url()
    client, model, max_tok, extra = _aux_client()
    if client is None or not model:
        print("ERROR: no seeding model configured (auxiliary.canon_seed)"); return 1
    now = datetime.now(timezone.utc).isoformat()
    written = 0
    recovered = 0
    for e in cands:
        legacy = full_text.get(e["id"]) or e["data_preview"]
        # Provenance recovery: pull the journal turns from this entry's date — the
        # real source it was distilled from — to ground the rewrite + give the
        # adversary something verifiable to check against.
        journal, refs = recover_journal(e.get("legacy_date"), qdrant)
        if journal:
            recovered += 1
        user_msg = f"Archived memory (distilled):\n{legacy}"
        if journal:
            user_msg += (f"\n\nJournal turns from {e.get('legacy_date')} this was distilled "
                         f"from (ground your rewrite in what actually happened):\n{journal[:4000]}")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": _REWRITE_SYSTEM},
                          {"role": "user", "content": user_msg}],
                temperature=0.2, timeout=180, **max_tok(1200, model=model),
                extra_body=extra() or None,
            )
            out = _extract_json(resp.choices[0].message.content or "")
        except Exception as exc:
            print(f"  rewrite failed for {e['id'][:8]}: {exc}"); continue
        if not out or not out.get("statement"):
            print(f"  skipped {e['id'][:8]} (no statement)"); continue
        facet = out.get("facet") if out.get("facet") in FACETS else "selffact"
        tier = e["tier"] or "peripheral"
        if out.get("tier") in ("core", "peripheral"):
            tier = out["tier"]
        payload = make_payload(
            statement=out["statement"].strip(), facet=facet, tier=tier,
            source_event=make_source_event(legacy, refs),
            interpretation=(out.get("interpretation") or "").strip(),
            status="candidate", provenance=e.get("provenance", "legacy_import"),
            legacy_source=e["id"], render_order=e.get("render_order"),
            derived_by=f"seed:{model}", created_at=now,
            extra={"legacy_date": e.get("legacy_date")},
        )
        try:
            validate_payload(payload)
        except Exception as exc:
            print(f"  invalid {e['id'][:8]}: {exc}"); continue
        store.upsert(cand_coll, [{"id": e["id"], "payload": payload}])
        written += 1
    print(f"rewrite: {written}/{len(cands)} legacy entries → {cand_coll} "
          f"(status:candidate; journal-provenance recovered for {recovered})")
    return 0


# ── stage: adversary ───────────────────────────────────────────────────────────
def stage_adversary(live: bool, limit: int | None, tag: str = "") -> int:
    store, cand_coll, _ = _store(live, tag)
    cands = store.get_canon(status="candidate", collection=cand_coll, limit=limit or 1000)
    bedrock = load_soul_bedrock_rows()
    qdrant = _qdrant_url()
    now = datetime.now(timezone.utc).isoformat()
    done = 0
    for cid, payload in cands:
        if payload.get("adversary_verdict"):
            continue  # idempotent
        # Recover the same journal turns as verifiable evidence so the adversary's
        # provenance/verifiability checks run against the REAL source, not the
        # claim citing only itself.
        evidence, _ = recover_journal(payload.get("legacy_date"), qdrant)
        verdict = run_adversary(payload, bedrock, now_iso=now, evidence=evidence)
        store.set_payload(cand_coll, cid, {"adversary_verdict": verdict})
        done += 1
        print(f"  {verdict['verdict']:8s} {cid[:8]} {payload.get('statement','')[:60]}")
    print(f"adversary: scored {done} candidate(s) in {cand_coll} (bedrock-context={len(bedrock)} rows)")
    return 0


# ── stage: review (Scott batch-QA surface) ─────────────────────────────────────
def stage_review(live: bool, tag: str = "") -> int:
    store, cand_coll, _ = _store(live, tag)
    cands = store.get_canon(status="candidate", collection=cand_coll, limit=1000)
    from collections import Counter
    verdicts = Counter((p.get("adversary_verdict") or {}).get("verdict", "(none)") for _, p in cands)
    print(f"=== SEED REVIEW ({len(cands)} candidates in {cand_coll}) ===")
    print("verdict tally:", dict(verdicts))
    for cid, p in sorted(cands, key=lambda x: (x[1].get("tier", ""), x[1].get("render_order", 0))):
        av = p.get("adversary_verdict") or {}
        print(f"[{av.get('verdict','?'):8s}] {p.get('tier','?'):10s} {p.get('facet','?'):12s} {p.get('statement','')[:70]}")
    return 0


# ── canon-target resolution (source seed coll ≠ canon target) ──────────────────
def _canon_target(live: bool, from_coll: str) -> str:
    """The canon collection ratify writes into. ``--live`` → the real ``sylva_canon``;
    otherwise a sandbox canon derived from the source seed collection name."""
    if live:
        from plugins.memory.canon.schema import CANON_COLLECTION
        return CANON_COLLECTION
    return from_coll.replace("seed_candidates", "seed_canon")


# ── stage: preview (goal-state self-brief — SOUL.md + would-be canon) ───────────
def stage_preview(from_coll: str, as_json: bool) -> int:
    """Assemble the GOAL-STATE self-brief from the current Scott-QA decisions —
    exactly what the runtime brief WOULD load after ratify + restart: SOUL.md
    bedrock + the canon entries Scott approved/edited (demotes re-tiered, rejects/
    undecided excluded). Read-only; writes nothing. ``--json`` emits a machine
    payload for the cockpit Ratify tab."""
    from plugins.memory.canon.render import _read_soul_md, assemble_brief, _canon_token_budget

    store = CanonStore.from_config()
    pts = _scroll_all(store._qdrant_url, from_coll)
    soul = _read_soul_md()

    entries: list[tuple[str, dict]] = []
    invalid: list[str] = []
    stats = {"approve": 0, "edit": 0, "reject": 0, "undecided": 0, "invalid": 0}
    by_tier: dict[str, int] = {}
    for cid, payload in pts:
        decision = (payload.get("qa_decision") or {}).get("decision")
        if decision == "reject":
            stats["reject"] += 1
            continue
        eff = _effective_entry(payload)
        if not eff:
            stats["undecided"] += 1
            continue
        cp = make_payload(
            statement=eff["statement"], facet=eff["facet"], tier=eff["tier"],
            source_event=payload.get("source_event") or make_source_event("", []),
            interpretation=eff["interpretation"], status="canon",
            render_order=payload.get("render_order"),
        )
        try:
            validate_payload(cp)  # ratify would reject these too — keep preview faithful
        except Exception:
            invalid.append(cid)
            stats["invalid"] += 1
            continue
        stats[eff["decision"]] += 1
        by_tier[eff["tier"]] = by_tier.get(eff["tier"], 0) + 1
        entries.append((cid, cp))

    brief = assemble_brief(soul, entries, _canon_token_budget())

    if as_json:
        print(json.dumps({
            "brief": brief or "",
            "soul_present": bool(soul),
            "soul_chars": len(soul or ""),
            "canon_token_budget": _canon_token_budget(),
            "canon_entry_count": len(entries),
            "stats": stats,
            "by_tier": by_tier,
            "invalid_ids": invalid,
            "entries": [
                {"id": i, "statement": p["statement"], "tier": p["tier"], "facet": p["facet"]}
                for i, p in entries
            ],
        }, ensure_ascii=False))
    else:
        print(f"=== GOAL-STATE SELF-BRIEF ({from_coll}) ===")
        print(f"SOUL.md: {'present' if soul else 'MISSING'} ({len(soul or '')} chars) | "
              f"canon entries: {len(entries)} | by tier: {by_tier} | stats: {stats}")
        print("-" * 72)
        print(brief or "(empty brief)")
    return 0


# ── stage: ratify (owner-gated cutover-enabling step) ──────────────────────────
def stage_ratify(live: bool, approve: bool, from_coll: str, reconcile: bool = False) -> int:
    """Decision-driven ratification: each candidate's Scott-QA ``qa_decision`` is
    authoritative. approve/edit → canonized (edit overrides applied; Sonnet demote
    re-tiers to peripheral); reject → rejected; undecided → skipped. Routed through
    the sanctioned ``route_verdict`` spine, so redaction + ``validate_payload`` +
    the PRD-028 audit ledger + the sole-writer invariant all hold. The recorded
    ``adversary_verdict`` is Sonnet's (the locked primary judge).

    ``reconcile``: after routing, RETIRE (status→retired, tombstone — never hard
    delete, per evolve-by-supersession) any live-canon entry that came from this
    source collection but is no longer approved. Makes the canon EXACTLY the
    approved set — needed for a consolidation re-ratify, where merged-away/dropped
    entries are already live from a prior ratify and ``route_verdict`` (which only
    upserts) would otherwise leave them orphaned. Only touches canon entries whose
    id is a candidate in ``from_coll`` AND not approved — entries from other sources
    or still approved are never retired."""
    if not approve:
        print("ratify is owner-gated: re-run with --approve to write the Scott-approved set → canon "
              "(ratified_by:{sylva, scott_qa@seed}). Preview with `preview` first.")
        return 2
    store = CanonStore.from_config()
    canon_coll = _canon_target(live, from_coll)
    store.ensure_collections((canon_coll,))
    existing = store.get_canon(status="canon", collection=canon_coll, limit=1000)
    pts = _scroll_all(store._qdrant_url, from_coll)
    now = datetime.now(timezone.utc).isoformat()

    def scott_qa_ratify(candidate, verdict, now_iso):
        return {"sylva": now_iso, "scott_qa@seed": now_iso}

    approved_ids: set[str] = set()
    source_ids: set[str] = set()
    n: dict = {"canonized": 0, "demoted": 0, "rejected": 0, "tension": 0,
               "merged": 0, "skipped": 0, "errored": 0, "retired": 0}
    for cid, payload in pts:
        source_ids.add(cid)
        decision = (payload.get("qa_decision") or {}).get("decision")
        sonnet = dict(payload.get("adversary_verdict_sonnet") or payload.get("adversary_verdict") or {})
        try:
            if decision == "reject":
                verdict = {**sonnet, "verdict": "refute",
                           "reasons": ["Scott QA: rejected"] + list(sonnet.get("reasons", []))[:3]}
                route_verdict(cid, payload, verdict, store, now_iso=now,
                              ratify_fn=scott_qa_ratify, candidates_collection=from_coll,
                              canon_collection=canon_coll, existing_canon=existing)
                n["rejected"] += 1
                continue
            eff = _effective_entry(payload)
            if not eff:
                n["skipped"] += 1
                continue
            cand = dict(payload)
            cand.update(statement=eff["statement"], facet=eff["facet"],
                        tier=eff["tier"], interpretation=eff["interpretation"])
            # Route as affirm at the resolved tier; carry Sonnet's reasons/model so
            # the ratified canon entry records the real (primary-judge) verdict.
            verdict = {**sonnet, "verdict": "affirm"}
            rec = route_verdict(cid, cand, verdict, store, now_iso=now,
                                ratify_fn=scott_qa_ratify, candidates_collection=from_coll,
                                canon_collection=canon_coll, existing_canon=existing)
            approved_ids.add(cid)
            n[rec.action] = n.get(rec.action, 0) + 1
        except Exception as e:
            print(f"  ERROR {cid[:8]}: {type(e).__name__}: {e}")
            n["errored"] += 1

    if reconcile:
        # Retire live-canon entries from THIS source that are no longer approved.
        for canon_id, cpayload in store.get_canon(status="canon", collection=canon_coll, limit=1000):
            if canon_id in source_ids and canon_id not in approved_ids:
                store.set_payload(canon_coll, canon_id,
                                  {"status": "retired", "retired_at": now})
                try:
                    from autonomy import audit
                    audit.record(tier="T2", surface="cron",
                                 action=f"canon reconcile: retired {canon_id} (no longer approved)",
                                 rationale="consolidation re-ratify", authority="scott_qa@seed",
                                 outcome="ok")
                except Exception:
                    pass
                n["retired"] += 1

    print(f"ratify (Scott-QA@seed → {canon_coll}){' +reconcile' if reconcile else ''}: {n}")
    if live:
        print("Live canon populated → rebuild image + restart gateway to apply (the cutover flip): "
              "the next session's self-brief assembles SOUL.md + canon.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="PRD-029 Phase 5 canon seeding")
    ap.add_argument("stage", choices=["rewrite", "adversary", "review", "preview", "ratify"])
    ap.add_argument("--manifest", type=Path,
                    default=Path("../docs/working/identity-canon-governance/migration_manifest.json"))
    ap.add_argument("--snapshot", type=Path,
                    default=Path("../docs/working/identity-canon-governance/sylva_memories_snapshot_20260627.json"),
                    help="frozen snapshot — source of the FULL legacy text (not the manifest preview)")
    ap.add_argument("--live", action="store_true", help="target real sylva_candidates/sylva_canon (default: sandbox)")
    ap.add_argument("--approve", action="store_true", help="ratify stage only: confirm the batch-QA flip")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="namespace the sandbox collections (e.g. 'gemma') to compare runs")
    ap.add_argument("--categories", default="",
                    help="comma-separated category filter for rewrite (e.g. 'anchor,personality' to skip kernels)")
    ap.add_argument("--from", dest="from_coll", default="sylva_lab_seed_candidates_gemma",
                    help="preview/ratify: source seed-candidate collection (decoupled from the canon target)")
    ap.add_argument("--json", action="store_true", help="preview stage only: emit a JSON payload (for the cockpit)")
    ap.add_argument("--reconcile", action="store_true",
                    help="ratify stage only: retire live-canon entries from this source no longer approved "
                         "(makes canon exactly the approved set — for consolidation re-ratify)")
    args = ap.parse_args()
    cats = {c.strip() for c in args.categories.split(",") if c.strip()} or None

    if args.stage == "rewrite":
        return stage_rewrite(args.manifest, args.live, args.limit, args.snapshot,
                             tag=args.tag, categories=cats)
    if args.stage == "adversary":
        return stage_adversary(args.live, args.limit, tag=args.tag)
    if args.stage == "review":
        return stage_review(args.live, tag=args.tag)
    if args.stage == "preview":
        return stage_preview(args.from_coll, args.json)
    if args.stage == "ratify":
        return stage_ratify(args.live, args.approve, args.from_coll, reconcile=args.reconcile)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
