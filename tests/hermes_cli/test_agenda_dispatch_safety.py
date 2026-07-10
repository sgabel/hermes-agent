"""PRD-049 AC-002: agenda cards are structurally non-dispatchable.

The invariant: agenda cards rest ONLY in ``scheduled``; the claim path
(``claim_task``) transitions ``ready -> running`` exclusively, and the
creation path lands in ``scheduled`` with no dispatchable intermediate state
— proven deterministically via the task_events log (the durable witness),
not a flaky concurrent reader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def agenda_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.create_board("agenda")
    return home


def _agenda_card(**kw):
    with kb.connect_closing(board="agenda") as conn:
        tid = kb.create_task(conn, title=kw.pop("title", "agenda item"),
                             initial_status="scheduled", **kw)
    return tid


def test_scheduled_card_never_claimed_even_with_assignee(agenda_home):
    tid = _agenda_card(assignee="sylva")
    with kb.connect_closing(board="agenda") as conn:
        claimed = kb.claim_task(conn, tid)
        assert claimed is None
        assert kb.get_task(conn, tid).status == "scheduled"


def test_recompute_ready_ignores_scheduled(agenda_home):
    tid = _agenda_card(assignee="sylva")
    with kb.connect_closing(board="agenda") as conn:
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "scheduled"


def test_created_event_records_scheduled_with_no_prior_transition(agenda_home):
    tid = _agenda_card()
    with kb.connect_closing(board="agenda") as conn:
        events = kb.list_events(conn, tid)
    assert events, "expected at least the created event"
    first = events[0]
    assert first.kind == "created"
    assert (first.payload or {}).get("status") == "scheduled"
    # No status-transition event of any kind precedes or accompanies creation.
    transition_kinds = {"promoted", "unblocked", "status", "ready", "claimed"}
    assert not [e for e in events if e.kind in transition_kinds]


def test_unblock_task_is_the_documented_unpark_path(agenda_home):
    # Documented residual (FR-1): unblock_task CAN un-park a scheduled card.
    # This test pins the residual so a future change that silently widens or
    # closes it shows up here.
    tid = _agenda_card(assignee="sylva")
    with kb.connect_closing(board="agenda") as conn:
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status in {"ready", "todo"}
