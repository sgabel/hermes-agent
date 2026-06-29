"""Tests for the PRD-042 recall-assist cockpit-visibility sidecar.

The PRD-041 FR-1 ambient recall assist injects a chronicle hit at API-call time
but never persists it. PRD-042 adds a DISPLAY-ONLY sidecar record so the glass
cockpit can show what auto-recall retrieved. These tests pin the invariants that
make the design safe: it is a separate table (never `messages`), the per-turn
write is idempotent, and a non-recall turn persists nothing.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    session_db.create_session(session_id="s1", source="cli")
    yield session_db
    session_db.close()


_HITS = [
    {"date": "2026-05-01", "speaker": "scott", "data": "want to wager a guess at my age"},
    {"date": "2026-05-02", "speaker": "sylva", "data": "I guessed 42 and you said warmer"},
]


class TestRecallAssistSidecar:
    def test_append_and_get_roundtrip(self, db):
        """AC-001: query + structured hits persist and read back faithfully."""
        db.append_recall_assist("s1", "do you remember my age?", _HITS, anchor_turn=3)
        rows = db.get_recall_assist("s1")
        assert len(rows) == 1
        row = rows[0]
        assert row["query"] == "do you remember my age?"
        assert row["anchor_turn"] == 3
        assert isinstance(row["hits"], list)
        assert row["hits"][0] == _HITS[0]
        assert row["hits"][1]["speaker"] == "sylva"

    def test_idempotent_within_turn(self, db):
        """AC-006: a retry / tool loop / fallback re-send on the same turn writes
        exactly one record (same dedup key -> same row id)."""
        id1 = db.append_recall_assist("s1", "my age?", _HITS, anchor_turn=3)
        id2 = db.append_recall_assist("s1", "my age?", _HITS, anchor_turn=3)
        assert id1 == id2
        assert len(db.get_recall_assist("s1")) == 1

    def test_same_question_distinct_turns_not_collapsed(self, db):
        """The same recall question on two different turns persists two records
        (anchor_turn discriminates the dedup key)."""
        id1 = db.append_recall_assist("s1", "my age?", _HITS, anchor_turn=3)
        id2 = db.append_recall_assist("s1", "my age?", _HITS, anchor_turn=7)
        assert id1 != id2
        assert len(db.get_recall_assist("s1")) == 2

    def test_quiet_default_empty_session(self, db):
        """AC-003: a session with no recall turn has no records."""
        assert db.get_recall_assist("s1") == []

    def test_not_written_to_messages_table(self, db):
        """AC-005: the recall record lives in the sidecar, NEVER in `messages`,
        so it is structurally invisible to every agent-facing consumer
        (compression, session_search, replay, on_session_end)."""
        db.append_recall_assist("s1", "my age?", _HITS, anchor_turn=3)
        assert db.get_messages("s1") == []
        assert len(db.get_recall_assist("s1")) == 1
