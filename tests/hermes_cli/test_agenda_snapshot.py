"""PRD-049 AC-003/AC-004: agenda snapshot date math, ordering, bounds.

All timezone resolution goes through ``hermes_time`` (the codebase's one
canonical resolver). Tests inject ``America/New_York`` via the env seam
(highest precedence, ``hermes_time._resolve_timezone_name``) and reset the
module cache around each use.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import hermes_time
from hermes_cli import agenda
from hermes_cli import kanban_db as kb


@pytest.fixture
def ny_tz(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/New_York")
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


@pytest.fixture
def agenda_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialized agenda board."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.create_board("agenda")
    return home


def _add(title, *, due=None, priority=0, body=""):
    with kb.connect_closing(board="agenda") as conn:
        return kb.create_task(
            conn,
            title=title,
            body=body,
            initial_status="scheduled",
            due_at=due,
            priority=priority,
        )


def _epoch(y, m, d, hh=12, tz="America/New_York"):
    from zoneinfo import ZoneInfo

    return int(datetime(y, m, d, hh, tzinfo=ZoneInfo(tz)).timestamp())


# ---------------------------------------------------------------------------
# Calendar-day math (AC-003)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("utc_hour", [2, 23])
def test_days_until_two_days_ahead_regardless_of_utc_hour(ny_tz, agenda_home, utc_hour):
    # 'now' = 2026-07-10 in ET for BOTH utc hours: 02:00Z Jul 10 = 22:00 ET
    # Jul 9... careful: pick now values that are the SAME ET calendar day.
    # 2026-07-10T12:00Z and 2026-07-10T23:00Z are both 2026-07-10 in ET.
    now = datetime(2026, 7, 10, utc_hour, tzinfo=timezone.utc)
    # 02:00Z Jul 10 is 22:00 ET Jul 9 -> ET date Jul 9; due Jul 11 -> 2 days.
    # 23:00Z Jul 10 is 19:00 ET Jul 10 -> ET date Jul 10; due Jul 12 -> 2 days.
    et_today = 9 if utc_hour == 2 else 10
    _add("trip", due=agenda.parse_due(f"2026-07-{et_today + 2:02d}"))
    items, text = agenda.build_agenda_snapshot(now=now)
    assert len(items) == 1
    assert items[0]["days_until"] == 2
    assert items[0]["overdue"] is False
    assert "trip" in text


def test_same_day_is_zero(ny_tz, agenda_home):
    now = datetime(2026, 7, 10, 15, tzinfo=timezone.utc)  # 11:00 ET Jul 10
    _add("today-item", due=agenda.parse_due("2026-07-10"))
    items, _ = agenda.build_agenda_snapshot(now=now)
    assert items[0]["days_until"] == 0
    assert items[0]["overdue"] is False


def test_past_due_negative_and_overdue(ny_tz, agenda_home):
    now = datetime(2026, 7, 10, 15, tzinfo=timezone.utc)
    _add("late", due=agenda.parse_due("2026-07-07"))
    items, text = agenda.build_agenda_snapshot(now=now)
    assert items[0]["days_until"] == -3
    assert items[0]["overdue"] is True
    assert "OVERDUE" in text


def test_dst_boundary_spring_forward(ny_tz, agenda_home):
    # US DST 2026: spring forward Mar 8. Now = Mar 6 ET; due Mar 9 ET.
    # Calendar-day diff must be 3 despite the 23-hour day in between.
    now = datetime(2026, 3, 6, 17, tzinfo=timezone.utc)  # 12:00 ET Mar 6
    _add("post-dst", due=agenda.parse_due("2026-03-09"))
    items, _ = agenda.build_agenda_snapshot(now=now)
    assert items[0]["days_until"] == 3


# ---------------------------------------------------------------------------
# Ordering + bounds (AC-003/AC-004)
# ---------------------------------------------------------------------------

def test_ordering_overdue_then_soonest_then_priority(ny_tz, agenda_home):
    now = datetime(2026, 7, 10, 15, tzinfo=timezone.utc)
    _add("undated-low", priority=1)
    _add("undated-high", priority=5)
    _add("due-far", due=agenda.parse_due("2026-07-20"))
    _add("due-soon", due=agenda.parse_due("2026-07-11"))
    _add("overdue", due=agenda.parse_due("2026-07-01"))
    items, _ = agenda.build_agenda_snapshot(now=now)
    titles = [i["title"] for i in items]
    assert titles[:3] == ["overdue", "due-soon", "due-far"]
    # undated after dated, priority DESC
    assert titles[3:] == ["undated-high", "undated-low"]


def test_include_undated_false_filters(ny_tz, agenda_home):
    now = datetime(2026, 7, 10, 15, tzinfo=timezone.utc)
    _add("undated")
    _add("dated", due=agenda.parse_due("2026-07-12"))
    items, _ = agenda.build_agenda_snapshot(now=now, include_undated=False)
    assert [i["title"] for i in items] == ["dated"]


def test_bounded_at_100_items(ny_tz, agenda_home):
    now = datetime(2026, 7, 10, 15, tzinfo=timezone.utc)
    for i in range(100):
        _add(f"item-{i:03d} " + "x" * 80, due=agenda.parse_due("2026-07-15"), body="b" * 300)
    items, text = agenda.build_agenda_snapshot(now=now)
    assert len(text) <= 4000
    assert len(items) <= 30  # default limit


def test_missing_board_returns_empty_no_raise(ny_tz, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    items, text = agenda.build_agenda_snapshot()
    assert items == []
    assert isinstance(text, str)
    assert "empty" in text.lower() or "no agenda" in text.lower()


# ---------------------------------------------------------------------------
# parse_due semantics
# ---------------------------------------------------------------------------

def test_parse_due_end_of_day_in_configured_tz(ny_tz):
    from zoneinfo import ZoneInfo

    epoch = agenda.parse_due("2026-07-20")
    dt = datetime.fromtimestamp(epoch, tz=ZoneInfo("America/New_York"))
    assert (dt.year, dt.month, dt.day) == (2026, 7, 20)
    assert (dt.hour, dt.minute, dt.second) == (23, 59, 59)


def test_parse_due_iso_offset_verbatim(ny_tz):
    epoch = agenda.parse_due("2026-07-20T10:00:00+02:00")
    assert epoch == int(datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc).timestamp())


def test_parse_due_invalid_raises(ny_tz):
    with pytest.raises(ValueError):
        agenda.parse_due("not-a-date")
