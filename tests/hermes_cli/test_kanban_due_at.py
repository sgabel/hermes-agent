"""PRD-049 AC-001/AC-005: due_at additive migration, round-trip, sort."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# AC-001: legacy DB back-compat
# ---------------------------------------------------------------------------

def test_legacy_db_without_due_at_migrates_in_place(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Simulate a genuine pre-049 DB: initialize the FULL modern schema, then
    # drop the due_at column (SQLite >= 3.35) and seed a legacy row. Hand-
    # building a minimal table is wrong — the module expects the whole
    # pre-existing baseline (claim_lock, run bookkeeping, ...).
    kb.init_db()
    db_path = home / "kanban.db"
    conn = sqlite3.connect(db_path)
    # ALTER ... DROP COLUMN trips on the schema's inline SQL comments, so
    # rebuild the table without due_at (column presence is what the
    # migration keys on — constraints are irrelevant to this test).
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
    legacy_cols = ", ".join(c for c in cols if c != "due_at")
    conn.executescript(
        f"""
        CREATE TABLE tasks_legacy AS SELECT {legacy_cols} FROM tasks;
        DROP TABLE tasks;
        ALTER TABLE tasks_legacy RENAME TO tasks;
        """
    )
    conn.execute(
        "INSERT INTO tasks (id, title, status, priority, created_at) "
        "VALUES ('t_legacy1', 'legacy card', 'todo', 0, 1700000000)"
    )
    conn.commit()
    cols2 = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "due_at" not in cols2  # precondition: genuinely legacy now
    conn.close()

    # The additive migration runs on module init (the real startup path).
    kb.init_db()
    with kb.connect_closing() as conn2:
        cols = {r[1] for r in conn2.execute("PRAGMA table_info(tasks)").fetchall()}
        assert "due_at" in cols
        row = conn2.execute(
            "SELECT due_at FROM tasks WHERE id = 't_legacy1'"
        ).fetchone()
        assert row[0] is None


# ---------------------------------------------------------------------------
# Round-trip + clear (AC-005 engine half)
# ---------------------------------------------------------------------------

def test_create_task_due_at_round_trip(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="dated", due_at=1784591999)
        task = kb.get_task(conn, tid)
    assert task.due_at == 1784591999


def test_set_due_set_and_clear(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="x")
        assert kb.get_task(conn, tid).due_at is None
        assert kb.set_due(conn, tid, 1784591999) is True
        assert kb.get_task(conn, tid).due_at == 1784591999
        assert kb.set_due(conn, tid, None) is True
        assert kb.get_task(conn, tid).due_at is None
        # unknown id
        assert kb.set_due(conn, "t_nope", 1) is False


def test_set_due_appends_event_not_status_change(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", initial_status="scheduled")
        kb.set_due(conn, tid, 1784591999)
        events = kb.list_events(conn, tid)
        kinds = [e.kind for e in events]
        assert "due_set" in kinds
        assert kb.get_task(conn, tid).status == "scheduled"


# ---------------------------------------------------------------------------
# Sort: 'due' NULLS LAST
# ---------------------------------------------------------------------------

def test_due_sort_nulls_last(kanban_home):
    with kb.connect_closing() as conn:
        kb.create_task(conn, title="undated")
        kb.create_task(conn, title="later", due_at=2000000000)
        kb.create_task(conn, title="sooner", due_at=1900000000)
        assert "due" in kb.VALID_SORT_ORDERS
        tasks = kb.list_tasks(conn, order_by="due")
    titles = [t.title for t in tasks]
    assert titles.index("sooner") < titles.index("later") < titles.index("undated")
