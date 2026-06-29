"""PRD-037 FR-1 — governed session-end episodic ingest (Mem0 provider).

Covers the centerpiece write path that restores Sylva's working memory:
on_session_end summarizes the conversation and upserts attributed, deduped,
bounded episodic entries into sylva_chronicle.

  * AC-001 — interactive session → a findable chronicle entry with the correct
             speaker/date/source=session:<id>/category=journal.
  * AC-002 — cron session → NO write (the run_agent passive gate is a no-op).
  * AC-003 — idempotent (hash-dedup, deterministic ids) + bounded (skip trivial,
             cap entry count + size).
  * AC-004 — only a session-boundary write (sync_turn stays a no-op) + each
             ingest records a PRD-028 audit-ledger entry.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import plugins.memory.mem0 as mem0mod
from plugins.memory.mem0 import Mem0MemoryProvider


class FakeChronicle:
    """Stand-in for ChronicleSearcher — deterministic embed, fixed url."""

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


def _provider(monkeypatch, *, existing_points=None, summary="- did a real thing with the foo.py module\n- decided to use bar for the new config knob"):
    """Build a provider wired with a fake chronicle + captured Qdrant I/O."""
    p = Mem0MemoryProvider()
    p._chronicle = FakeChronicle()
    p._session_id = "sess-abc"
    p._user_id = "sylva"

    captured = {"put": [], "audit": []}

    def fake_post(url, json=None, timeout=None):
        # scroll for existing hashes
        return FakeResp({"result": {"points": existing_points or [], "next_page_offset": None}})

    def fake_put(url, json=None, timeout=None):
        captured["put"].append({"url": url, "body": json})
        return FakeResp({"result": {"status": "completed"}})

    monkeypatch.setattr(mem0mod.requests, "post", fake_post)
    monkeypatch.setattr(mem0mod.requests, "put", fake_put)
    monkeypatch.setattr(p, "_summarize_session", lambda turns: summary)

    # Capture audit.record without importing the real ledger.
    fake_audit = SimpleNamespace(record=lambda **kw: captured["audit"].append(kw))
    import sys
    monkeypatch.setitem(sys.modules, "autonomy", SimpleNamespace(audit=fake_audit))

    return p, captured


_GOOD_MESSAGES = [
    {"role": "system", "content": "you are sylva"},
    {"role": "user", "content": (
        "Let's refactor the foo.py module today and decide on the right config "
        "value for the new chronicle ingest path. I want it bounded and idempotent."
    )},
    {"role": "assistant", "content": (
        "Sure — I refactored foo.py, wired the session-end hook, and we decided to "
        "use bar for the setting after weighing the trade-offs against baz."
    )},
    {"role": "user", "content": "Great, ship it when the tests are green and the build passes."},
]


# --- AC-001 / AC-004 : interactive write + attribution + audit ----------------

def test_ingest_writes_attributed_chronicle_entries(monkeypatch):
    p, cap = _provider(monkeypatch)
    p.on_session_end(_GOOD_MESSAGES)

    assert len(cap["put"]) == 1
    points = cap["put"][0]["body"]["points"]
    assert len(points) == 2  # two bullets → two entries
    for pt in points:
        pl = pt["payload"]
        assert pl["speaker"] == "sylva"
        assert pl["source"] == "session:sess-abc"
        assert pl["category"] == "journal"
        assert pl["user_id"] == "sylva"
        assert pl["date"]  # YYYY-MM-DD stamped
        assert pl["hash"]
        assert pt["vector"] == [0.1, 0.2, 0.3]
    # AC-004: audit ledger entry recorded for the write.
    assert len(cap["audit"]) == 1
    assert cap["audit"][0]["action"] == "chronicle_episodic_ingest"
    assert "session:sess-abc" in cap["audit"][0]["rationale"]


def test_ingest_strips_markdown_and_drops_headers(monkeypatch):
    p, cap = _provider(
        monkeypatch,
        summary="### Summary\n- **Bold** point about deploying the gateway container\n- x\n* another bullet that is long enough to keep",
    )
    p.on_session_end(_GOOD_MESSAGES)
    points = cap["put"][0]["body"]["points"]
    datas = [pt["payload"]["data"] for pt in points]
    # header dropped, "- x" too short dropped, markdown markers stripped.
    assert any(d.startswith("Bold point") or d.startswith("**Bold**") is False for d in datas)
    assert all(not d.startswith("#") for d in datas)
    assert all(not d.startswith("- ") for d in datas)
    assert "x" not in datas


# --- AC-003 : idempotent + bounded -------------------------------------------

def test_ingest_is_idempotent_via_hash_dedup(monkeypatch):
    import hashlib
    # Pre-seed the chronicle with the exact hashes the summary will produce.
    entries = ["did a real thing with the foo.py module", "decided to use bar for the new config knob"]
    existing = [{"payload": {"hash": hashlib.md5(e.encode()).hexdigest()}} for e in entries]
    p, cap = _provider(monkeypatch, existing_points=existing)
    p.on_session_end(_GOOD_MESSAGES)
    # All entries already present → no upsert, no audit.
    assert cap["put"] == []
    assert cap["audit"] == []


def test_ingest_deterministic_point_ids(monkeypatch):
    p, cap = _provider(monkeypatch)
    p.on_session_end(_GOOD_MESSAGES)
    ids1 = [pt["id"] for pt in cap["put"][0]["body"]["points"]]
    # Re-derive independently — ids are uuid5(NS, "session:<id>:<hash>").
    import hashlib
    expect = [
        str(uuid.uuid5(mem0mod._INGEST_NS, f"session:sess-abc:{hashlib.md5(e.encode()).hexdigest()}"))
        for e in ["did a real thing with the foo.py module", "decided to use bar for the new config knob"]
    ]
    assert ids1 == expect


def test_trivial_session_skipped(monkeypatch):
    p, cap = _provider(monkeypatch)
    p.on_session_end([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert cap["put"] == []
    assert cap["audit"] == []


def test_entry_count_and_size_bounded(monkeypatch):
    long_line = "x" * 5000
    many = "\n".join(f"- bullet number {i} describing some real work item here" for i in range(30))
    p, cap = _provider(monkeypatch, summary=f"- {long_line}\n{many}")
    p.on_session_end(_GOOD_MESSAGES)
    points = cap["put"][0]["body"]["points"]
    assert len(points) <= mem0mod._INGEST_MAX_ENTRIES
    assert all(len(pt["payload"]["data"]) <= mem0mod._INGEST_MAX_ENTRY_CHARS for pt in points)


def test_empty_summary_writes_nothing(monkeypatch):
    p, cap = _provider(monkeypatch, summary="")
    p.on_session_end(_GOOD_MESSAGES)
    assert cap["put"] == []
    assert cap["audit"] == []


# --- robustness ---------------------------------------------------------------

def test_on_session_end_never_raises(monkeypatch):
    p, cap = _provider(monkeypatch)
    def boom(text):
        raise RuntimeError("embed down")
    p._chronicle.embed = boom
    # Embedding fails for every entry → no points → no write, but NO exception.
    p.on_session_end(_GOOD_MESSAGES)
    assert cap["put"] == []


def test_no_chronicle_is_noop(monkeypatch):
    p, cap = _provider(monkeypatch)
    p._chronicle = None
    p.on_session_end(_GOOD_MESSAGES)
    assert cap["put"] == []


def test_sync_turn_remains_write_noop(monkeypatch):
    """AC-004: sync_turn must never write — it only caches orchestration state."""
    p, cap = _provider(monkeypatch)
    p.sync_turn("hello", "a long assistant reply " * 20, session_id="sess-abc")
    assert cap["put"] == []
    assert cap["audit"] == []


def test_extract_turns_drops_system_tool_and_compaction():
    turns = Mem0MemoryProvider._extract_turns([
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": "tool output"},
        {"role": "user", "content": "[CONTEXT COMPACTION] old stuff"},
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": [{"type": "image"}]},  # multimodal non-str
        {"role": "assistant", "content": "real answer"},
    ])
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["text"] == "real question"


def test_on_session_switch_updates_session_id():
    p = Mem0MemoryProvider()
    p._session_id = "old"
    p.on_session_switch("new-sess")
    assert p._session_id == "new-sess"
    p.on_session_switch("")  # empty → keep prior
    assert p._session_id == "new-sess"


# --- S1 fix: compaction (final=False) must NOT ingest -------------------------

def test_compaction_final_false_does_not_ingest(monkeypatch):
    """PRD-037 S1: a mid-conversation compaction calls on_session_end(final=False)
    — the chronicle writer must defer to the true boundary (no overlapping
    near-dup writes on the hot path). Compacted turns stay in state.db FTS."""
    p, cap = _provider(monkeypatch)
    p.on_session_end(_GOOD_MESSAGES, final=False)
    assert cap["put"] == []
    assert cap["audit"] == []


def test_true_boundary_final_true_ingests(monkeypatch):
    """Inverse: a true boundary (/new, /reset, exit) ingests."""
    p, cap = _provider(monkeypatch)
    p.on_session_end(_GOOD_MESSAGES, final=True)
    assert len(cap["put"]) == 1


def test_default_final_is_true(monkeypatch):
    """Back-compat: bare on_session_end(messages) defaults to a true boundary."""
    p, cap = _provider(monkeypatch)
    p.on_session_end(_GOOD_MESSAGES)
    assert len(cap["put"]) == 1


# --- H2 fix: secret redaction before persist ----------------------------------

def test_secrets_are_redacted_before_persist(monkeypatch):
    """PRD-037 H2: a credential in the summary must be scrubbed before the entry
    is embedded/persisted to the chronicle."""
    secret = "sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKKLLLL"
    p, cap = _provider(
        monkeypatch,
        summary=f"- wired the deploy using the key {secret} for the gateway today",
    )
    p.on_session_end(_GOOD_MESSAGES)
    datas = [pt["payload"]["data"] for pt in cap["put"][0]["body"]["points"]]
    assert all(secret not in d for d in datas), f"secret leaked into chronicle: {datas}"


# --- manager forwarding of `final` --------------------------------------------

def test_manager_detects_final_capable_provider():
    from agent.memory_manager import _on_session_end_accepts_final

    p = Mem0MemoryProvider()
    assert _on_session_end_accepts_final(p.on_session_end) is True

    # Legacy-style provider (no final kwarg, no **kwargs) → not forwarded.
    class Legacy:
        def on_session_end(self, messages):
            pass
    assert _on_session_end_accepts_final(Legacy().on_session_end) is False

    # **kwargs-style provider → forwarded.
    class Kwargy:
        def on_session_end(self, messages, **kw):
            pass
    assert _on_session_end_accepts_final(Kwargy().on_session_end) is True


# --- AC-002 : cron exclusion (run_agent passive gate) -------------------------

def _bare_agent_with_mm():
    from run_agent import AIAgent
    a = AIAgent.__new__(AIAgent)
    a._memory_manager = MagicMock()
    a.session_id = "sess-cron"
    a.context_compressor = None
    return a


def test_cron_session_end_does_not_ingest():
    """AC-002: with passive memory off (the cron call-site), the close path must
    NOT call on_session_end — no cron scaffolding reaches the chronicle."""
    a = _bare_agent_with_mm()
    a._memory_passive_enabled = False
    a.shutdown_memory_provider(messages=_GOOD_MESSAGES)
    a._memory_manager.on_session_end.assert_not_called()
    # resource cleanup still runs.
    a._memory_manager.shutdown_all.assert_called_once()


def test_interactive_session_end_does_ingest():
    """Inverse of AC-002 — interactive (passive on) DOES fire on_session_end."""
    a = _bare_agent_with_mm()
    a._memory_passive_enabled = True
    a.shutdown_memory_provider(messages=_GOOD_MESSAGES)
    a._memory_manager.on_session_end.assert_called_once()
