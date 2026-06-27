"""PRD-029 Phase 4 — verdict routing + ledger + bedrock-writer scope
(AC-007 / AC-008 / AC-009).

Covers:
  * AC-007 — every verdict routes correctly: affirm→canon, refute→rejected
    (not surfaced), tension→surfaced, demote→re-tier+canon, merge→fold.
  * AC-008 — every mutation writes exactly one PRD-028 ledger entry; nothing is
    hard-deleted (status tombstones survive for rollback).
  * AC-009 — the ratification writer touches ONLY sylva_canon / sylva_candidates,
    never SOUL.md / the filesystem.
"""

import json

import pytest

from plugins.memory.canon import (
    CANDIDATES_COLLECTION,
    CANON_COLLECTION,
    make_payload,
    make_source_event,
)
from plugins.memory.canon import ratification as R
from plugins.memory.canon.ratification import route_verdict, run_ratification


# ── fake store: records every collection it is asked to mutate ─────────────────
class _FakeStore:
    def __init__(self, candidates=None, canon=None):
        self._candidates = candidates or []   # [(id, payload)]
        self._canon = canon or []             # [(id, payload)]
        self.upserts = []        # [(collection, [points])]
        self.payload_sets = []   # [(collection, id, updates)]

    def get_canon(self, *, layer, status, collection, limit=1000):
        src = self._candidates if collection == CANDIDATES_COLLECTION else self._canon
        return [(i, p) for (i, p) in src if p.get("status") == status]

    def upsert(self, collection, points):
        self.upserts.append((collection, points))

    def set_payload(self, collection, point_id, updates):
        self.payload_sets.append((collection, point_id, updates))

    def get_point(self, collection, point_id):
        for i, p in (self._canon if collection == CANON_COLLECTION else self._candidates):
            if i == point_id:
                return p
        return None

    # every collection this store was asked to write
    def written_collections(self):
        cols = {c for c, _ in self.upserts}
        cols |= {c for c, _, _ in self.payload_sets}
        return cols


@pytest.fixture
def ledger_spy(monkeypatch):
    calls = []
    import autonomy.audit as audit
    monkeypatch.setattr(audit, "record", lambda **kw: calls.append(kw) or kw)
    # ratification imports audit lazily inside _record, so patch the module fn
    return calls


def _cand(cid="c1", statement="I value care.", tier="core", interp="meaning"):
    p = make_payload(
        statement=statement, facet="value", tier=tier,
        source_event=make_source_event("Scott asked me to harden.", ["session:1"]),
        interpretation=interp, status="candidate",
    )
    return cid, p


def _verdict(v, reasons=None, refs=None, checks=None):
    return {
        "verdict": v, "reasons": reasons or ["r"], "evidence_refs": refs or [],
        "model": "fake", "checked_at": "t", "checks": checks or {},
    }


# ── AC-007 routing ─────────────────────────────────────────────────────────────
def test_affirm_writes_canon_and_tombstones_candidate(ledger_spy):
    store = _FakeStore()
    cid, cand = _cand()
    rec = route_verdict(cid, cand, _verdict("affirm"), store, now_iso="2026-06-27T00:00:00+00:00")
    assert rec.action == "canonized"
    # canon write went to sylva_canon, status canon, ratified stamp present
    assert len(store.upserts) == 1
    coll, points = store.upserts[0]
    assert coll == CANON_COLLECTION
    payload = points[0]["payload"]
    assert payload["status"] == "canon"
    assert "sylva" in payload["ratified_by"]
    assert payload["adversary_verdict"]["verdict"] == "affirm"
    # candidate tombstoned (not deleted)
    assert any(c == CANDIDATES_COLLECTION and u.get("status") == "canon"
               for c, _, u in store.payload_sets)
    assert len(ledger_spy) == 1


def test_refute_rejects_candidate_no_canon_write_not_surfaced(ledger_spy):
    store = _FakeStore()
    cid, cand = _cand()
    rec = route_verdict(cid, cand, _verdict("refute"), store, now_iso="t")
    assert rec.action == "rejected"
    assert store.upserts == []                       # no canon write
    assert store.payload_sets[0][2]["status"] == "rejected"
    assert len(ledger_spy) == 1                       # ledgered (but not surfaced to Scott)


def test_tension_surfaces_without_canon_write(ledger_spy):
    store = _FakeStore()
    cid, cand = _cand()
    rec = route_verdict(cid, cand, _verdict("tension", refs=["canon:x"]), store, now_iso="t")
    assert rec.action == "tension"
    assert store.upserts == []
    upd = store.payload_sets[0][2]
    assert upd["status"] == "tension"
    assert upd["tension_with"] == ["canon:x"]
    assert "Sylva" in rec.note


def test_tension_touching_bedrock_routes_to_scott(ledger_spy):
    store = _FakeStore()
    cid, cand = _cand()
    v = _verdict("tension", checks={"contradiction": "conflicts with a BEDROCK value"})
    rec = route_verdict(cid, cand, v, store, now_iso="t")
    assert "Scott" in rec.note
    assert ledger_spy[0]["action"].endswith("scott") or "scott" in ledger_spy[0]["action"].lower()


def test_demote_canonizes_at_peripheral(ledger_spy):
    store = _FakeStore()
    cid, cand = _cand(tier="core")
    rec = route_verdict(cid, cand, _verdict("demote"), store, now_iso="t")
    assert rec.action == "demoted"
    assert store.upserts[0][1][0]["payload"]["tier"] == "peripheral"


def test_merge_folds_and_bumps_existing(ledger_spy):
    existing = [("canon:1", make_payload(
        statement="I value care.", facet="value", tier="core",
        source_event=make_source_event("x", ["r"]), status="canon"))]
    store = _FakeStore(canon=existing)
    cid, cand = _cand()
    rec = route_verdict(cid, cand, _verdict("merge", refs=["canon:1"]), store,
                        now_iso="2026-06-27T00:00:00+00:00",
                        existing_canon=existing)
    assert rec.action == "merged"
    assert rec.canon_id == "canon:1"
    # candidate superseded; existing canon last_affirmed bumped
    assert any(u.get("status") == "superseded" for _, _, u in store.payload_sets)
    assert any(c == CANON_COLLECTION and "last_affirmed_at" in u
               for c, _, u in store.payload_sets)


def test_ratify_withheld_leaves_candidate_queued(ledger_spy):
    """Scott-QA hook (ratify_fn returns None) → no canon write, candidate stays."""
    store = _FakeStore()
    cid, cand = _cand()
    rec = route_verdict(cid, cand, _verdict("affirm"), store, now_iso="t",
                        ratify_fn=lambda c, v, n: None)
    assert rec.action == "noop"
    assert store.upserts == []


# ── AC-008 ledger-per-mutation across a full pipeline run ──────────────────────
def test_pipeline_ledgers_every_mutation(ledger_spy):
    cands = [_cand("a", statement="A"), _cand("b", statement="B"), _cand("c", statement="C")]
    store = _FakeStore(candidates=cands)
    by_statement = {"A": "affirm", "B": "refute", "C": "tension"}
    res = run_ratification(
        store=store,
        adversary_fn=lambda cand, canon: _verdict(by_statement[cand["statement"]]),
        now_iso="2026-06-27T00:00:00+00:00",
    )
    assert res.canonized == 1 and res.rejected == 1 and res.tension == 1
    assert len(ledger_spy) == 3   # one per mutation
    # only canon collection got a canon upsert
    assert all(c == CANON_COLLECTION for c, _ in store.upserts)


def test_merge_with_no_resolvable_target_falls_back_to_tension(ledger_spy):
    """STOP-3: a merge whose evidence_refs match no canon id must NOT silently
    retire the candidate — it downgrades to tension (surfaced)."""
    store = _FakeStore(canon=[])
    cid, cand = _cand()
    rec = route_verdict(cid, cand, _verdict("merge", refs=["canon:does-not-exist"]),
                        store, now_iso="t", existing_canon=[])
    assert rec.action == "tension"      # not "merged"
    assert store.upserts == []          # no canon write
    assert any(u.get("status") == "tension" for _, _, u in store.payload_sets)


def test_cross_model_flag_false_when_adversary_equals_proposer(ledger_spy):
    """STOP-1: same model for proposer + adversary is flagged, not silently
    claimed as cross-model."""
    cid, cand = _cand()
    cand["derived_by"] = "qwen3.6-35b-a3b"
    store = _FakeStore(candidates=[(cid, cand)])

    def adv(c, k):
        v = _verdict("affirm")
        v["model"] = "qwen3.6-35b-a3b"   # same as proposer
        return v

    res = run_ratification(store=store, adversary_fn=adv, now_iso="t")
    assert res.cross_model is False
    assert "SAME-MODEL" in res.summary()


def test_cross_model_flag_true_when_models_differ(ledger_spy):
    cid, cand = _cand()
    cand["derived_by"] = "qwen3.6-35b-a3b"
    store = _FakeStore(candidates=[(cid, cand)])

    def adv(c, k):
        v = _verdict("refute")
        v["model"] = "claude-sonnet-4-6"
        return v

    res = run_ratification(store=store, adversary_fn=adv, now_iso="t")
    assert res.cross_model is True


def test_reaffirm_bumps_version_and_preserves_created_at(ledger_spy):
    prior = make_payload(
        statement="I value care.", facet="value", tier="core",
        source_event=make_source_event("x", ["r"]), status="canon",
        created_at="2026-01-01T00:00:00+00:00")
    prior["version"] = 2
    store = _FakeStore(canon=[("c1", prior)])
    cand = make_payload(
        statement="I value care.", facet="value", tier="core",
        source_event=make_source_event("x", ["r"]), status="candidate",
        created_at="2026-06-27T00:00:00+00:00")
    rec = route_verdict("c1", cand, _verdict("affirm"), store, now_iso="t")
    written = store.upserts[0][1][0]["payload"]
    assert written["version"] == 3                         # bumped from prior
    assert written["created_at"] == "2026-01-01T00:00:00+00:00"  # preserved


def test_dry_run_applies_nothing(ledger_spy):
    store = _FakeStore(candidates=[_cand()])
    res = run_ratification(store=store, adversary_fn=lambda c, k: _verdict("affirm"),
                           now_iso="t", dry_run=True)
    assert res.dry_run is True
    assert store.upserts == [] and store.payload_sets == []
    assert ledger_spy == []


# ── AC-009 bedrock-writer scope ────────────────────────────────────────────────
def test_writer_touches_only_canon_collections(ledger_spy):
    cands = [_cand("a"), _cand("b", statement="B"), _cand("c", statement="C"),
             _cand("d", statement="D")]
    store = _FakeStore(candidates=cands)
    vmap = {"a": "affirm", "b": "refute", "c": "tension", "d": "demote"}
    order = [cid for cid, _ in cands]
    idx = {"i": 0}

    def adv(cand, canon):
        cid = order[idx["i"]]; idx["i"] += 1
        return _verdict(vmap[cid])

    run_ratification(store=store, adversary_fn=adv, now_iso="t")
    assert store.written_collections() <= {CANON_COLLECTION, CANDIDATES_COLLECTION}


def test_ratification_module_never_writes_files():
    """Structural AC-009(b): the canon writer has no filesystem write of any kind
    (SOUL.md/bedrock is unreachable). Checks executable code, not the docstring —
    which legitimately *documents* that SOUL.md is out of reach."""
    import ast
    import inspect
    import plugins.memory.canon.ratification as mod

    tree = ast.parse(inspect.getsource(mod))
    # assert no file-write primitives in executable code (docstrings may mention
    # SOUL.md — that's documentation, not a write path)
    calls = [
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    attrs = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert "open" not in calls                  # no open(...) anywhere in code
    assert "write_text" not in attrs            # no Path.write_text
    assert "write_bytes" not in attrs


def test_no_ratified_row_can_be_bedrock(ledger_spy):
    """validate_payload (called inside ratify) rejects tier:bedrock — proven by a
    candidate that somehow claims bedrock: it errors, never canonizes."""
    store = _FakeStore()
    cid, cand = _cand(tier="core")
    cand["tier"] = "bedrock"   # force an illegal tier past the proposer
    with pytest.raises(Exception):
        route_verdict(cid, cand, _verdict("affirm"), store, now_iso="t")
    assert store.upserts == []   # nothing written
