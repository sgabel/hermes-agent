"""PRD-049 AC-005 (REST half): due_at round-trip + agenda-board policy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_due_test", plugin_file,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.create_board("agenda")
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


B = "/api/plugins/kanban"


def test_due_at_round_trip_on_default_board(client):
    r = client.post(f"{B}/tasks", json={
        "title": "dated", "assignee": "sylva", "due_at": 1784591999,
    })
    assert r.status_code == 200, r.text
    tid = r.json()["task"]["id"]
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).due_at == 1784591999


def test_patch_due_at_set_and_null_clears(client):
    tid = client.post(f"{B}/tasks", json={
        "title": "x", "assignee": "sylva",
    }).json()["task"]["id"]
    r = client.patch(f"{B}/tasks/{tid}", json={"due_at": 1784591999})
    assert r.status_code == 200, r.text
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).due_at == 1784591999
    # Explicit null PRESENT in the body clears; absent field = unchanged.
    r = client.patch(f"{B}/tasks/{tid}", json={"title": "renamed"})
    assert r.status_code == 200
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).due_at == 1784591999  # unchanged
    r = client.patch(f"{B}/tasks/{tid}", json={"due_at": None})
    assert r.status_code == 200, r.text
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).due_at is None


def test_agenda_board_create_coerces_scheduled_and_rejects_triage(client):
    r = client.post(f"{B}/tasks?board=agenda", json={"title": "agenda item"})
    assert r.status_code == 200, r.text
    tid = r.json()["task"]["id"]
    with kb.connect_closing(board="agenda") as conn:
        assert kb.get_task(conn, tid).status == "scheduled"
    r = client.post(f"{B}/tasks?board=agenda", json={
        "title": "bad", "triage": True,
    })
    assert r.status_code == 422, r.text


def test_agenda_board_bulk_rejects_dispatchable_statuses(client):
    # Found at code review: POST /tasks/bulk was a status-write surface that
    # bypassed the PATCH-route guard. Pin the closure.
    tid = client.post(f"{B}/tasks?board=agenda", json={"title": "b"}).json()["task"]["id"]
    for status in ("triage", "todo", "ready", "running"):
        r = client.post(f"{B}/tasks/bulk?board=agenda",
                        json={"ids": [tid], "status": status})
        assert r.status_code == 422, f"{status}: {r.status_code} {r.text}"
    with kb.connect_closing(board="agenda") as conn:
        assert kb.get_task(conn, tid).status == "scheduled"


def test_agenda_board_patch_rejects_dispatchable_statuses(client):
    tid = client.post(f"{B}/tasks?board=agenda", json={"title": "a"}).json()["task"]["id"]
    for status in ("triage", "todo", "ready", "running"):
        r = client.patch(f"{B}/tasks/{tid}?board=agenda", json={"status": status})
        assert r.status_code == 422, f"{status}: {r.status_code} {r.text}"
    with kb.connect_closing(board="agenda") as conn:
        assert kb.get_task(conn, tid).status == "scheduled"
