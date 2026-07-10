"""PRD-049 AC-004: agenda_view tool — registered, read-only, bounded."""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.agenda_tool as at
from hermes_cli import agenda as agenda_mod
from hermes_cli import kanban_db as kb
from tools.registry import registry


@pytest.fixture
def agenda_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.create_board("agenda")
    with kb.connect_closing(board="agenda") as conn:
        kb.create_task(conn, title="tracked topic", initial_status="scheduled",
                       priority=2, body="some body text")
    return home


def test_registered_under_agenda_toolset():
    assert registry.get_toolset_for_tool("agenda_view") == "agenda"
    assert "agenda_view" in registry.get_tool_names_for_toolset("agenda")


def test_check_fn_true_without_config():
    assert at.check_agenda_available() is True


def test_view_returns_bounded_text(agenda_home):
    out = at.agenda_view()
    assert "tracked topic" in out
    assert len(out) <= 4000


def test_empty_board_returns_explicit_string(tmp_path, monkeypatch):
    home = tmp_path / ".hermes2"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    out = at.agenda_view()
    assert isinstance(out, str)
    assert "empty" in out.lower() or "no agenda" in out.lower()


def test_read_only_no_write_helpers_invoked(agenda_home, monkeypatch):
    def _boom(*a, **kw):  # pragma: no cover - should never fire
        raise AssertionError("agenda_view invoked a kanban write helper")

    for name in ("create_task", "set_due", "schedule_task", "complete_task",
                 "block_task", "unblock_task", "assign_task", "archive_task",
                 "delete_task"):
        monkeypatch.setattr(kb, name, _boom)
    out = at.agenda_view()
    assert "tracked topic" in out


def test_handler_dispatch_include_undated(agenda_home):
    entry = registry._tools["agenda_view"]
    out = entry.handler({"include_undated": False})
    assert isinstance(out, str)
