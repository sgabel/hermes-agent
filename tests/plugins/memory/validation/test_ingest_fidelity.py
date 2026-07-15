"""PRD-050 FR-2 — chronicle ingest fidelity (hermetic).

Synthetic transcript corpus (``fixtures/transcripts.yaml``) →
``on_session_end(final=True)`` → assert the persisted episodic entries are
bounded, attributed, deterministic, idempotent — and that the planted fake
secret is absent at BOTH capture points:

  1. the transcript handed to the summarizer (the aux-client prompt), and
  2. the persisted/embedded upsert body.

Reuses the proven stub recipe from ``tests/plugins/memory/
test_episodic_ingest.py:26-75`` — fake chronicle embed, captured
``requests.post/put``, audit stub via ``sys.modules`` — extended with a
STATEFUL point store (so a second ingest of the same session sees the first
run's hashes) and an aux-client-level summarizer fake (so the prompt capture
point sits BEHIND the transcript redaction, not in front of it).
"""

import hashlib
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import plugins.memory.mem0 as mem0mod
from plugins.memory.mem0 import Mem0MemoryProvider

_FIXTURES = Path(__file__).parent / "fixtures" / "transcripts.yaml"
_CORPUS = yaml.safe_load(_FIXTURES.read_text(encoding="utf-8"))
_SECRET = _CORPUS["meta"]["planted_secret"]


def _session(session_id: str):
    for s in _CORPUS["sessions"]:
        if s["id"] == session_id:
            return s["messages"]
    raise KeyError(session_id)


class FakeChronicle:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self.embedded = []

    def embed(self, text):
        self.embedded.append(text)
        return [0.1, 0.2, 0.3]


class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _provider(monkeypatch, *, summary=None):
    """Provider with a STATEFUL captured store: puts feed the next scroll, so
    re-ingest sees the prior run's hashes (true idempotency, not a canned
    ``existing_points`` snapshot)."""
    p = Mem0MemoryProvider()
    p._chronicle = FakeChronicle()
    p._session_id = "mv-sess-1"
    p._user_id = "sylva"

    cap = {"points": [], "put": [], "audit": [], "prompts": []}

    def fake_post(url, json=None, timeout=None):
        return FakeResp({"result": {"points": cap["points"], "next_page_offset": None}})

    def fake_put(url, json=None, timeout=None):
        cap["put"].append({"url": url, "body": json})
        cap["points"].extend({"payload": pt["payload"]} for pt in json["points"])
        return FakeResp({"result": {"status": "completed"}})

    monkeypatch.setattr(mem0mod.requests, "post", fake_post)
    monkeypatch.setattr(mem0mod.requests, "put", fake_put)

    if summary is not None:
        monkeypatch.setattr(p, "_summarize_session", lambda turns: summary)

    fake_audit = SimpleNamespace(record=lambda **kw: cap["audit"].append(kw))
    monkeypatch.setitem(sys.modules, "autonomy", SimpleNamespace(audit=fake_audit))
    return p, cap


def _wire_aux_summarizer(monkeypatch, cap, *, canned_summary: str):
    """Fake the auxiliary client BEHIND ``_summarize_session`` so the captured
    prompt is exactly what leaves the process toward the summarizer — i.e.
    AFTER the transcript redaction (capture point 1)."""
    import agent.auxiliary_client as aux

    class _Completions:
        @staticmethod
        def create(**kwargs):
            cap["prompts"].append(kwargs["messages"][-1]["content"])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=canned_summary))]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()), base_url=""
    )
    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task: (client, "mv-aux"))


_TWO_BULLETS = (
    "- wired the recall harness and validated the corpus loader end to end\n"
    "- decided the filtered queries compute allowed_ids from the corpus itself"
)


# ── planted secret: absent at BOTH capture points (the FR-2 centerpiece) ─────
def test_planted_secret_never_reaches_summarizer_or_store(monkeypatch):
    p, cap = _provider(monkeypatch)
    # the summarizer ECHOES the secret back — persist-side redaction must
    # scrub it even when the model copies it out of the transcript.
    _wire_aux_summarizer(
        monkeypatch,
        cap,
        canned_summary=(
            f"- probed TEI with the key {_SECRET} for the one-off smoke check\n"
            "- validated the corpus loader against the gold mapping"
        ),
    )
    p.on_session_end(_session("substantive_with_secret"), final=True)

    # capture point 1: the prompt handed to the summarizer
    assert len(cap["prompts"]) == 1
    assert _SECRET not in cap["prompts"][0]
    # ...but the surrounding turn text survived (redacted, not dropped)
    assert "recall harness" in cap["prompts"][0]

    # capture point 2: the persisted upsert body (and everything embedded)
    assert len(cap["put"]) == 1
    assert _SECRET not in json.dumps(cap["put"][0]["body"])
    assert all(_SECRET not in text for text in p._chronicle.embedded)


# ── bounded + attributed ─────────────────────────────────────────────────────
def test_constants_match_prd_bounds():
    assert mem0mod._INGEST_MAX_ENTRIES == 8
    assert mem0mod._INGEST_MAX_ENTRY_CHARS == 1000
    assert mem0mod._INGEST_MIN_ENTRY_CHARS == 25


def test_entries_bounded_and_attributed(monkeypatch):
    long_line = "y" * 5000
    many = "\n".join(
        f"- bullet number {i} describing a real distinct work item in detail" for i in range(30)
    )
    p, cap = _provider(monkeypatch, summary=f"- {long_line}\n{many}\n- tiny")
    p.on_session_end(_session("substantive_with_secret"), final=True)

    points = cap["put"][0]["body"]["points"]
    assert 1 <= len(points) <= mem0mod._INGEST_MAX_ENTRIES
    for pt in points:
        pl = pt["payload"]
        assert mem0mod._INGEST_MIN_ENTRY_CHARS <= len(pl["data"]) <= mem0mod._INGEST_MAX_ENTRY_CHARS
        assert pl["speaker"] == "sylva"
        assert pl["source"] == "session:mv-sess-1"
        assert pl["category"] == "journal"
        assert pl["user_id"] == "sylva"
        assert pl["date"]
        assert pl["hash"] == hashlib.md5(pl["data"].encode()).hexdigest()


# ── deterministic ids + idempotent re-ingest ─────────────────────────────────
def test_deterministic_uuid5_ids(monkeypatch):
    p, cap = _provider(monkeypatch, summary=_TWO_BULLETS)
    p.on_session_end(_session("substantive_with_secret"), final=True)
    got = [pt["id"] for pt in cap["put"][0]["body"]["points"]]
    expect = [
        str(
            uuid.uuid5(
                mem0mod._INGEST_NS,
                f"session:mv-sess-1:{hashlib.md5(e.encode()).hexdigest()}",
            )
        )
        for e in [
            "wired the recall harness and validated the corpus loader end to end",
            "decided the filtered queries compute allowed_ids from the corpus itself",
        ]
    ]
    assert got == expect


def test_reingesting_same_session_writes_zero_new_points(monkeypatch):
    p, cap = _provider(monkeypatch, summary=_TWO_BULLETS)
    p.on_session_end(_session("substantive_with_secret"), final=True)
    assert len(cap["put"]) == 1
    first_points = cap["put"][0]["body"]["points"]
    assert len(first_points) == 2

    # same session, same summary → hash dedup sees the stateful store → no put
    p.on_session_end(_session("substantive_with_secret"), final=True)
    assert len(cap["put"]) == 1
    # and no second audit record either
    assert len(cap["audit"]) == 1


# ── boundary semantics + triviality ──────────────────────────────────────────
def test_final_false_compaction_writes_nothing(monkeypatch):
    p, cap = _provider(monkeypatch, summary=_TWO_BULLETS)
    p.on_session_end(_session("substantive_with_secret"), final=False)
    assert cap["put"] == []
    assert cap["audit"] == []


def test_trivial_session_skipped(monkeypatch):
    p, cap = _provider(monkeypatch, summary=_TWO_BULLETS)
    p.on_session_end(_session("trivial"), final=True)
    assert cap["put"] == []


def test_compaction_marker_turns_are_dropped(monkeypatch):
    p, cap = _provider(monkeypatch, summary=_TWO_BULLETS)
    p.on_session_end(_session("compaction_markers_only"), final=True)
    # both turns start with "[CONTEXT" → zero real turns → trivial skip
    assert cap["put"] == []


# ── in-summary dedup + audit ─────────────────────────────────────────────────
def test_in_summary_duplicate_bullets_collapse(monkeypatch):
    dup = "decided the filtered queries compute allowed_ids from the corpus itself"
    p, cap = _provider(monkeypatch, summary=f"- {dup}\n- {dup}\n- {dup.upper()}")
    p.on_session_end(_session("substantive_with_secret"), final=True)
    points = cap["put"][0]["body"]["points"]
    assert len(points) == 1  # case-insensitive in-summary dedup


def test_ingest_records_one_audit_entry_via_stub(monkeypatch):
    p, cap = _provider(monkeypatch, summary=_TWO_BULLETS)
    p.on_session_end(_session("substantive_with_secret"), final=True)
    assert len(cap["audit"]) == 1
    assert cap["audit"][0]["action"] == "chronicle_episodic_ingest"
    assert "session:mv-sess-1" in cap["audit"][0]["rationale"]
