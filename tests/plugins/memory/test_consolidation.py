"""PRD-029 Phase 3 — consolidation pass (governed candidate proposer).

Hermetic by default: ``run_consolidation`` takes injectable ``store`` / ``db`` /
``derive_fn``, so the core logic is tested with no Qdrant, no LLM, no state.db.
One opt-in integration test exercises the real direct-Qdrant upsert against a
throwaway collection when Qdrant is reachable.

Covers:
  * AC-004 / AC-017 — writes ONLY to sylva_candidates as status:candidate, via
    direct-Qdrant CanonStore (never mem0_add, never sylva_canon).
  * AC-013 — deterministic recent-session enumeration with cron excluded
    *structurally* (fails loud if "cron" ever leaves _HIDDEN_SESSION_SOURCES);
    no keyword session_search.
  * AC-019 — agency mined as structured records with the source_event /
    interpretation split; build-execution cruft never becomes a candidate.
"""

import uuid
from typing import Any, Dict, List

import pytest
import requests

from plugins.memory.canon import CANDIDATES_COLLECTION, CANON_COLLECTION, CanonStore
from plugins.memory.canon import consolidation as C
from plugins.memory.canon.consolidation import (
    ConsolidationResult,
    _candidate_point,
    _gather_recent_sessions,
    _session_transcript,
    run_consolidation,
)
from plugins.memory.canon.schema import FACETS, VECTOR_DIM, content_hash


# ── fakes ─────────────────────────────────────────────────────────────────────
class _FakeStore:
    """Records upserts; asserts the writer's collection reach.

    ``existing`` maps a collection name → list of ``(point_id, payload)`` rows
    that :meth:`get_canon` returns, so the PRD-038 cross-store dedup path (M2) can
    be exercised hermetically. Default is empty (no dedup hits) — and a store with
    no ``get_canon`` at all still works because the dedup helper is fail-soft.
    """

    def __init__(self, existing=None):
        self.upserts: List[tuple] = []
        self.ensured: List[tuple] = []
        self._existing = existing or {}

    def ensure_collections(self, collections):
        self.ensured.append(tuple(collections))

    def upsert(self, collection, points):
        self.upserts.append((collection, points))

    def get_canon(self, *, collection, status, **_kw):
        return list(self._existing.get(collection, []))


class _FakeDB:
    """Minimal SessionDB stand-in. Records the exclude_sources it was asked for."""

    def __init__(self, sessions, transcripts):
        self._sessions = sessions          # list of rich dicts (already filtered)
        self._transcripts = transcripts    # {sid: [conversation msgs]}
        self.last_exclude_sources = None
        self.last_order_by_last_active = None
        self.closed = False

    def list_sessions_rich(self, *, limit, exclude_sources, min_message_count, order_by_last_active):
        self.last_exclude_sources = list(exclude_sources)
        self.last_order_by_last_active = order_by_last_active
        # The fake already excludes cron; assert the caller asked for it.
        return self._sessions[:limit]

    def get_messages_as_conversation(self, sid):
        return self._transcripts.get(sid, [])

    def close(self):
        self.closed = True


def _proposal(statement="I value careful, reversible changes.", facet="value",
              tier="core", claim="Scott asked me to harden before shipping.",
              refs=None, interpretation="Safety is part of who I am."):
    return {
        "statement": statement,
        "facet": facet,
        "tier": tier,
        "source_event": {"claim": claim, "provenance_refs": refs or ["session:abc"]},
        "interpretation": interpretation,
    }


# ── AC-004 / AC-017: sole writer, candidates only, direct-Qdrant ────────────────
def test_writes_only_candidates_collection_as_candidate_status():
    store = _FakeStore()
    db = _FakeDB(sessions=[{"id": "s1", "source": "discord", "last_active": "2026-06-26"}],
                 transcripts={"s1": [{"role": "user", "content": "hi"}]})

    def fake_derive(sessions, agency):
        return [_proposal()], "test-model"

    res = run_consolidation(store=store, db=db, derive_fn=fake_derive,
                            now_iso="2026-06-27T00:00:00+00:00")

    assert res.candidates_written == 1
    assert len(store.upserts) == 1
    collection, points = store.upserts[0]
    # never sylva_canon
    assert collection == CANDIDATES_COLLECTION
    assert collection != CANON_COLLECTION
    # status is candidate, provenance is consolidation, never canon
    payload = points[0]["payload"]
    assert payload["status"] == "candidate"
    assert payload["provenance"] == "consolidation"
    assert payload["derived_by"] == "test-model"


def test_dry_run_derives_but_never_writes():
    store = _FakeStore()
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})
    res = run_consolidation(store=store, db=db,
                            derive_fn=lambda s, a: ([_proposal()], "m"),
                            dry_run=True)
    assert res.dry_run is True
    assert res.candidates_written == 1   # counted
    assert store.upserts == []           # but not written


def test_no_candidates_is_a_clean_noop_not_a_write():
    store = _FakeStore()
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})
    res = run_consolidation(store=store, db=db, derive_fn=lambda s, a: ([], "m"))
    assert res.candidates_written == 0
    assert store.upserts == []
    assert "no durable candidates" in res.skipped_reason


def test_sandbox_target_redirects_writes_off_candidates():
    """AC-010 validation runs write to sylva_lab, never the live candidates."""
    store = _FakeStore()
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})
    res = run_consolidation(store=store, db=db, target_collection="sylva_lab",
                            derive_fn=lambda s, a: ([_proposal()], "m"))
    assert store.upserts[0][0] == "sylva_lab"
    assert res.target_collection == "sylva_lab"


def test_canon_collection_target_is_refused():
    """S-1: no caller/operator path may route a candidate write into sylva_canon."""
    store = _FakeStore()
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ValueError, match="never write"):
        run_consolidation(store=store, db=db, target_collection=CANON_COLLECTION,
                          derive_fn=lambda s, a: ([_proposal()], "m"))
    assert store.upserts == []  # refused before any work


def test_arbitrary_collection_target_is_refused():
    store = _FakeStore()
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ValueError, match="may only write"):
        run_consolidation(store=store, db=db, target_collection="sylva_memories",
                          derive_fn=lambda s, a: ([_proposal()], "m"))


def test_idempotent_ids_dedupe_within_and_across_runs():
    store = _FakeStore()
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})
    # same proposal twice in one batch → one point
    res = run_consolidation(store=store, db=db,
                            derive_fn=lambda s, a: ([_proposal(), _proposal()], "m"))
    assert res.candidates_written == 1
    id1 = store.upserts[0][1][0]["id"]

    store2 = _FakeStore()
    run_consolidation(store=store2, db=db, derive_fn=lambda s, a: ([_proposal()], "m"))
    id2 = store2.upserts[0][1][0]["id"]
    assert id1 == id2  # stable across runs


# ── AC-013: deterministic enumeration, cron excluded structurally ──────────────
def test_gather_passes_hidden_sources_including_cron():
    db = _FakeDB([{"id": "s1", "last_active": "x", "source": "discord"}],
                 {"s1": [{"role": "user", "content": "hello"}]})
    out = _gather_recent_sessions(db, limit=5)
    assert "cron" in db.last_exclude_sources
    assert "subagent" in db.last_exclude_sources
    assert db.last_order_by_last_active is True
    assert out and out[0]["id"] == "s1"


def test_fails_loud_if_cron_drops_from_hidden_sources(monkeypatch):
    import tools.session_search_tool as sst
    monkeypatch.setattr(sst, "_HIDDEN_SESSION_SOURCES", ("subagent", "tool"))
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})
    with pytest.raises(RuntimeError, match="cron"):
        _gather_recent_sessions(db, limit=5)


def test_transcript_drops_tool_and_system_noise():
    """Build cruft (tool/system messages) never reaches the deriver (AC-019 b)."""
    db = _FakeDB([], {})
    convo = [
        {"role": "user", "content": "let's harden the config"},
        {"role": "tool", "content": "diff --git a/x b/x ...patch noise..."},
        {"role": "system", "content": "you are an agent"},
        {"role": "assistant", "content": "done, hardened."},
    ]
    db._transcripts["s1"] = convo
    text = _session_transcript(db, "s1", 10000)
    assert "harden the config" in text
    assert "hardened." in text
    assert "diff --git" not in text
    assert "you are an agent" not in text


def test_transcript_truncates_on_line_boundaries_keeping_recent_turns():
    """N-2: over-budget truncation keeps whole recent turns, never a mid-word slice."""
    db = _FakeDB([], {})
    convo = [
        {"role": "user", "content": "A" * 100},        # oldest — should be dropped
        {"role": "assistant", "content": "B" * 100},
        {"role": "user", "content": "keep me whole"},   # newest — must survive intact
    ]
    db._transcripts["s1"] = convo
    text = _session_transcript(db, "s1", 60)
    assert "user: keep me whole" in text          # newest whole turn kept
    assert "A" * 100 not in text                   # oldest dropped
    # no partial line survived: every non-ellipsis line is a complete role: turn
    for line in text.splitlines():
        if line != "…":
            assert line.startswith(("user:", "assistant:"))


# ── AC-019: agency split + cruft rejection at payload construction ──────────────
def test_candidate_preserves_source_event_interpretation_split():
    pt = _candidate_point(_proposal(), model="m", now_iso="2026-06-27T00:00:00+00:00")
    assert pt is not None
    p = pt["payload"]
    # the verifiable half
    assert p["source_event"]["claim"] == "Scott asked me to harden before shipping."
    assert p["source_event"]["provenance_refs"] == ["session:abc"]
    # the meaning half, kept strictly separate
    assert p["interpretation"] == "Safety is part of who I am."


def test_candidate_rejects_bad_facet_and_bedrock_tier():
    assert _candidate_point(_proposal(facet="not-a-facet"), model="m", now_iso="t") is None
    # bedrock is coerced away from proposal tiers, then validate_payload would
    # reject it anyway — the proposer can never mint a bedrock row.
    pt = _candidate_point(_proposal(tier="bedrock"), model="m", now_iso="t")
    assert pt is not None
    assert pt["payload"]["tier"] == "peripheral"


def test_candidate_defaults_source_event_to_statement_when_missing():
    prop = {"statement": "I am curious.", "facet": "trait", "tier": "core"}
    pt = _candidate_point(prop, model="m", now_iso="t")
    assert pt is not None
    assert pt["payload"]["source_event"]["claim"] == "I am curious."
    assert pt["payload"]["source_event"]["provenance_refs"] == []


def test_agency_layer_is_structured_only(monkeypatch):
    """AC-019 (a): agency input comes from structured rows (ledger/kanban/
    work-block), never a raw cron transcript. With no live sources it is a
    clean empty list, not a crash."""
    # force all three structured readers to their empty path
    monkeypatch.setattr(C, "_agency_from_ledger", lambda: [])
    monkeypatch.setattr(C, "_agency_from_kanban", lambda: [])
    monkeypatch.setattr(C, "_agency_from_work_blocks", lambda: [])
    assert C._gather_agency_layer() == []

    monkeypatch.setattr(C, "_agency_from_ledger", lambda: [
        {"kind": "ledger", "claim": "hardened config", "ref": "ledger:abc", "when": "t"}])
    items = C._gather_agency_layer()
    assert items and items[0]["kind"] == "ledger"
    assert "ref" in items[0]  # carries provenance for the source_event


def test_result_summary_is_human_readable():
    r = ConsolidationResult(candidates_written=2, sessions_seen=5, agency_items=3, model="qwen35")
    s = r.summary()
    assert "2 candidate" in s and "5 session" in s and "qwen35" in s


def test_all_facets_accepted():
    for f in FACETS:
        pt = _candidate_point(_proposal(facet=f), model="m", now_iso="t")
        assert pt is not None, f


# ── security: secret-shaped content is refused before durable storage ──────────
def test_candidate_with_secret_in_statement_is_dropped():
    secret = "xqK9fL2mP7rT4vN8wZ3bY6cH1gD5jA0eU7sQ"  # high-entropy blob
    pt = _candidate_point(_proposal(statement=f"My key is {secret}"), model="m", now_iso="t")
    assert pt is None  # refused — never reaches the always-loaded canon store


def test_transcript_redacts_secrets_before_the_deriver():
    db = _FakeDB([], {})
    secret = "xqK9fL2mP7rT4vN8wZ3bY6cH1gD5jA0eU7sQ"  # high-entropy blob
    db._transcripts["s1"] = [{"role": "user", "content": f"creds: {secret}"}]
    text = _session_transcript(db, "s1", 10000)
    assert secret not in text
    assert "REDACTED" in text


# ── PRD-038 M1: provenance contract on consolidation candidates ─────────────────
def test_candidate_carries_run_id_and_content_hash():
    """M1/AC-001: every consolidation candidate carries a validated run_id +
    content_hash + grounded source_event.claim."""
    pt = _candidate_point(_proposal(statement="I value reversible changes."),
                          model="m", now_iso="t", run_id="run-abc")
    assert pt is not None
    p = pt["payload"]
    assert p["run_id"] == "run-abc"
    assert p["content_hash"] == content_hash("I value reversible changes.")
    assert p["source_event"]["claim"]  # non-empty
    # adversary_verdict is NOT populated at propose time (set at ratify)
    assert p.get("adversary_verdict") in (None, {})


def test_run_threads_a_single_run_id_across_all_candidates():
    """One run_id per run, stamped onto every candidate (FR-3 traceability)."""
    store = _FakeStore()
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})

    def derive(s, a):
        return ([_proposal(statement="A durable fact."),
                 _proposal(statement="Another durable fact.", refs=["session:def"])], "m")

    run_consolidation(store=store, db=db, derive_fn=derive, now_iso="t")
    _coll, points = store.upserts[0]
    run_ids = {p["payload"]["run_id"] for p in points}
    assert len(points) == 2
    assert len(run_ids) == 1
    assert next(iter(run_ids))  # non-empty


def test_content_hash_normalizes_phrasing():
    """Casing / whitespace differences collapse to the same content_hash so dedup
    treats them as the same durable fact."""
    assert content_hash("I Value  Reversible Changes.") == content_hash(
        "i value reversible changes.")


# ── PRD-038 M3 / AC-002 (propose-side): the gate must not silently pass all ─────
def test_propose_gate_drops_secret_and_invalid_keeps_genuine():
    """AC-002 (propose-time): a secret-bearing proposal AND a schema-invalid
    proposal are dropped, while a genuine durable fact reaches the queue. This
    FAILS if the propose-time gate silently passed everything."""
    store = _FakeStore()
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})

    secret_blob = "xqK9fL2mP7rT4vN8wZ3bY6cH1gD5jA0eU7sQ"   # high-entropy → redactor trips
    genuine_stmt = "I value careful, reversible changes to our shared work."

    def derive(s, a):
        return (
            [
                _proposal(statement=f"My API key is {secret_blob}"),   # secret → dropped
                _proposal(statement="  ", facet="value"),               # invalid (empty) → dropped
                _proposal(statement="totally not a facet", facet="bogus"),  # invalid facet → dropped
                _proposal(statement=genuine_stmt),                      # genuine → kept
            ],
            "m",
        )

    res = run_consolidation(store=store, db=db, derive_fn=derive, now_iso="t")

    assert len(store.upserts) == 1
    _coll, points = store.upserts[0]
    written = [p["payload"]["statement"] for p in points]
    # genuine present
    assert genuine_stmt in written
    assert res.candidates_written == 1
    # secret + invalid absent
    assert not any(secret_blob in s for s in written)
    assert all(s.strip() for s in written)
    assert all(p["payload"]["facet"] in FACETS for p in points)


def test_propose_gate_negative_control_would_fail_if_silent_pass():
    """Companion to the above: with NO bad proposals, all genuine facts pass —
    proves the gate isn't dropping everything indiscriminately."""
    store = _FakeStore()
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})

    def derive(s, a):
        return ([_proposal(statement="I value reversibility."),
                 _proposal(statement="I value clear communication.", refs=["session:def"])], "m")

    res = run_consolidation(store=store, db=db, derive_fn=derive, now_iso="t")
    assert res.candidates_written == 2


# ── PRD-038 M2 / AC-003: cross-store dedup vs live canon + open candidates ──────
def _existing_row(statement, *, with_hash=True):
    """Build a stored (point_id, payload) row as get_canon would return it."""
    payload = {"statement": statement, "status": "canon"}
    if with_hash:
        payload["content_hash"] = content_hash(statement)
    return ("pid-" + content_hash(statement)[:8], payload)


def test_dedup_against_live_canon_skips_already_ratified():
    """AC-003: a fact already in sylva_canon is NOT re-proposed."""
    dup_stmt = "I value careful, reversible changes."
    store = _FakeStore(existing={CANON_COLLECTION: [_existing_row(dup_stmt)]})
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})

    def derive(s, a):
        return ([_proposal(statement=dup_stmt),
                 _proposal(statement="A brand new durable fact.", refs=["session:def"])], "m")

    res = run_consolidation(store=store, db=db, derive_fn=derive, now_iso="t")
    _coll, points = store.upserts[0]
    written = [p["payload"]["statement"] for p in points]
    assert dup_stmt not in written                       # deduped against canon
    assert "A brand new durable fact." in written
    assert res.candidates_written == 1


def test_dedup_against_open_candidate_skips_already_queued():
    """AC-003: a fact already an OPEN sylva_candidate is NOT re-proposed."""
    dup_stmt = "I value careful, reversible changes."
    store = _FakeStore(existing={CANDIDATES_COLLECTION: [_existing_row(dup_stmt)]})
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})

    def derive(s, a):
        return ([_proposal(statement=dup_stmt)], "m")

    res = run_consolidation(store=store, db=db, derive_fn=derive, now_iso="t")
    assert store.upserts == []                           # nothing left to write
    assert res.candidates_written == 0
    assert "already in canon or open queue" in res.skipped_reason


def test_dedup_derives_hash_for_legacy_rows_without_content_hash():
    """A pre-M1 canon row lacking content_hash still dedups (hash derived from
    its statement) — so legacy seed canon isn't re-proposed."""
    dup_stmt = "I value careful, reversible changes."
    store = _FakeStore(
        existing={CANON_COLLECTION: [_existing_row(dup_stmt, with_hash=False)]})
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})
    res = run_consolidation(store=store, db=db,
                            derive_fn=lambda s, a: ([_proposal(statement=dup_stmt)], "m"),
                            now_iso="t")
    assert res.candidates_written == 0


# ── PRD-038 AC-005: idempotent re-run writes no new duplicate ───────────────────
def test_idempotent_rerun_same_window_no_duplicate():
    """AC-005: re-running over the same derive output dedups against the candidate
    already queued from the first run (same content_hash → skipped)."""
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})
    derive = lambda s, a: ([_proposal(statement="I value reversibility.")], "m")

    # first run: writes one candidate
    store1 = _FakeStore()
    r1 = run_consolidation(store=store1, db=db, derive_fn=derive, now_iso="t")
    assert r1.candidates_written == 1
    written_payload = store1.upserts[0][1][0]["payload"]
    first_id = store1.upserts[0][1][0]["id"]

    # second run: the candidate is now an open candidate in the store → deduped
    store2 = _FakeStore(existing={
        CANDIDATES_COLLECTION: [(first_id, written_payload)]})
    r2 = run_consolidation(store=store2, db=db, derive_fn=derive, now_iso="t")
    assert r2.candidates_written == 0
    assert store2.upserts == []


# ── PRD-038 AC-007: planted credential never appears in a written candidate ─────
def test_credential_in_source_record_never_reaches_candidate(monkeypatch):
    """AC-007: a credential planted in a source session is redacted before the
    deriver, and even if a proposal echoes it, the secret screen drops the
    candidate — the credential never lands in any written candidate payload."""
    secret_blob = "xqK9fL2mP7rT4vN8wZ3bY6cH1gD5jA0eU7sQ"   # high-entropy

    store = _FakeStore()
    db = _FakeDB(
        [{"id": "s1", "last_active": "x", "source": "discord"}],
        {"s1": [{"role": "user", "content": f"here is my secret: {secret_blob}"}]},
    )

    # 1) the transcript fed to the deriver must already be redacted
    transcript = _session_transcript(db, "s1", 10000)
    assert secret_blob not in transcript

    # 2) even an adversarial proposal echoing the secret is dropped at propose time
    def derive(sessions, agency):
        return ([_proposal(statement=f"My credential is {secret_blob}"),
                 _proposal(statement="I value secure handling of credentials.",
                           refs=["session:s1"])], "m")

    res = run_consolidation(store=store, db=db, derive_fn=derive, now_iso="t")
    _coll, points = store.upserts[0]
    import json as _json
    serialized = _json.dumps([p["payload"] for p in points])
    assert secret_blob not in serialized                 # never in any candidate
    assert res.candidates_written == 1                   # only the clean one
    assert points[0]["payload"]["statement"] == "I value secure handling of credentials."


# ── integration: real direct-Qdrant upsert (gated) ─────────────────────────────
_QDRANT = "http://localhost:6333"


def _qdrant_up() -> bool:
    try:
        return requests.get(f"{_QDRANT}/collections", timeout=2).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _qdrant_up(), reason="Qdrant not reachable on localhost:6333")
def test_real_upsert_into_throwaway_collection():
    # sylva_lab* sandbox name so the writable-target guard (S-1) permits it.
    name = f"sylva_lab_test_{uuid.uuid4().hex[:8]}"
    store = CanonStore(qdrant_url=_QDRANT)
    store.ensure_collections(collections=(name,))
    db = _FakeDB([{"id": "s1", "last_active": "x"}], {"s1": [{"role": "user", "content": "hi"}]})

    # explicit-vector path: monkeypatch embed so we don't need TEI
    store.embed = lambda text: [0.01] * VECTOR_DIM  # type: ignore
    try:
        res = run_consolidation(store=store, db=db, target_collection=name,
                                derive_fn=lambda s, a: ([_proposal()], "m"))
        assert res.candidates_written == 1
        got = store.get_canon(collection=name, status="candidate")
        assert len(got) == 1
        assert got[0][1]["status"] == "candidate"
    finally:
        requests.delete(f"{_QDRANT}/collections/{name}", timeout=10)
