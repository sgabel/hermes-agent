"""PRD-049: Standing-agenda snapshot API.

Single home for the date math, ordering, and rendering of the standing agenda
— the dedicated ``agenda`` kanban board of cards that rest in ``scheduled``
status (the one status the dispatcher never claims). Consumed by the
``agenda_view`` read-only tool (``tools/agenda_tool.py``) and — later — by
PRD-034's orchestrator, so the date math lives in exactly one place.

Timezone handling: ALL tz resolution routes through ``hermes_time`` (env →
``config.yaml timezone`` → server-local, cached) — never a raw ``HERMES_TIMEZONE``
read. The tool/snapshot computes ``days_until`` (the local 35B is unreliable at
date arithmetic); the model never does the calendar math.

Fail-soft: a missing board / unreadable DB returns ``([], <empty string>)`` and
never raises, so the proactive loop degrades to today's behaviour.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

import hermes_time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BOARD = "agenda"
DEFAULT_LIMIT = 30
DEFAULT_MAX_CHARS = 4000
_SNIPPET_CHARS = 120

# Explicit empty-agenda strings (never an error — fail-soft reads).
_EMPTY_NO_BOARD = "standing agenda: (empty — no agenda board)"
_EMPTY_NO_ITEMS = "standing agenda: (empty — no items)"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Date parsing (curation write path)
# ---------------------------------------------------------------------------

def parse_due(s: str) -> int:
    """Parse a due-date string into an epoch-seconds integer (UTC).

    Accepted forms:
      * ``YYYY-MM-DD`` — interpreted as **end-of-day (23:59:59)** in the
        configured timezone, then converted to an epoch. A bare calendar date
        is "due by the end of that day, Scott-local".
      * a full ISO 8601 datetime **with an offset** (e.g.
        ``2026-07-20T09:00:00-04:00`` or ``...Z``) — accepted verbatim.

    Anything else — an empty string, a naive ISO datetime with no offset, or a
    malformed value — raises :class:`ValueError` (the caller surfaces it as a
    loud CLI/tool error; a due date is never silently dropped).
    """
    if not isinstance(s, str) or not s.strip():
        raise ValueError("due date is required (YYYY-MM-DD or ISO8601 with offset)")
    raw = s.strip()

    if _DATE_RE.match(raw):
        try:
            d = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"invalid due date {raw!r}: {exc}") from exc
        tz = hermes_time.get_timezone()
        if tz is not None:
            eod = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=tz)
        else:
            # No tz configured → end-of-day server-local (astimezone attaches
            # the server's zone to the naive wall-clock time).
            eod = datetime(d.year, d.month, d.day, 23, 59, 59).astimezone()
        return int(eod.timestamp())

    # Full ISO 8601 with offset. Accept a trailing ``Z`` as UTC.
    iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise ValueError(
            f"invalid due date {raw!r}: expected YYYY-MM-DD or ISO8601 with offset"
        ) from exc
    if dt.tzinfo is None:
        raise ValueError(
            f"invalid due date {raw!r}: an ISO8601 datetime must include a "
            "timezone offset (e.g. 2026-07-20T09:00-04:00 or ...Z)"
        )
    return int(dt.timestamp())


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def _local_date(epoch: int, tz) -> Any:
    """Return the calendar date of ``epoch`` (UTC seconds) in tz ``tz``."""
    dt = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
    dt = dt.astimezone(tz) if tz is not None else dt.astimezone()
    return dt.date()


def _snippet(body: Optional[str]) -> str:
    if not body:
        return ""
    flat = " ".join(str(body).split())
    if len(flat) <= _SNIPPET_CHARS:
        return flat
    return flat[: _SNIPPET_CHARS - 1].rstrip() + "…"


def _days_label(item: dict) -> str:
    if item["due_at"] is None:
        return "no date"
    d = item["days_until"]
    if item["overdue"]:
        return f"OVERDUE {abs(d)}d"
    if d == 0:
        return "due today"
    return f"in {d}d"


def _render(items: list[dict], max_chars: int) -> str:
    """Render items to bounded text (≤ ``max_chars``). Truncates with a marker
    when the full list would overflow."""
    if not items:
        return _EMPTY_NO_ITEMS
    header = f"Standing agenda ({len(items)} item{'s' if len(items) != 1 else ''}):"
    lines = [header]
    used = len(header)
    shown = 0
    for it in items:
        prio = f"p{it['priority']}"
        label = _days_label(it)
        snip = f" — {it['snippet']}" if it["snippet"] else ""
        line = f"- [{label}] ({prio}) {it['title']}{snip}"
        # +1 for the newline join.
        if used + 1 + len(line) > max_chars and shown > 0:
            remaining = len(items) - shown
            marker = f"… (+{remaining} more)"
            if used + 1 + len(marker) <= max_chars:
                lines.append(marker)
            break
        lines.append(line)
        used += 1 + len(line)
        shown += 1
    text = "\n".join(lines)
    # Hard belt-and-suspenders cap (a single pathological title can't blow the
    # budget — PRD-023 lesson: bound tool output at source).
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def build_agenda_snapshot(
    now: Optional[datetime] = None,
    board: str = DEFAULT_BOARD,
    limit: int = DEFAULT_LIMIT,
    max_chars: int = DEFAULT_MAX_CHARS,
    *,
    include_undated: bool = True,
) -> tuple[list[dict], str]:
    """Build the standing-agenda snapshot: ``(items, rendered_text)``.

    ``items`` is a list of dicts, each with ``id``, ``title``, ``snippet``
    (~120 chars of body), ``priority``, ``due_at`` (epoch or None),
    ``days_until`` (calendar-day difference in the configured tz; None when
    undated), and ``overdue`` (bool).

    Ordering: dated items first, most-overdue → soonest (by ``days_until``
    ascending), then ``priority`` DESC, then ``created_at`` ASC; undated items
    last, by ``priority`` DESC then ``created_at`` ASC.

    Reads only ``scheduled`` cards on ``board`` via a read-only connection.
    A missing board or any read failure returns ``([], <empty string>)`` and
    NEVER raises.
    """
    tz = hermes_time.get_timezone()
    now_dt = now if now is not None else hermes_time.now()
    if now_dt.tzinfo is None:
        now_dt = now_dt.astimezone()
    today = now_dt.astimezone(tz).date() if tz is not None else now_dt.astimezone().date()

    rows = _read_scheduled(board)
    if rows is None:
        return [], _EMPTY_NO_BOARD

    items: list[dict] = []
    for r in rows:
        due_at = r["due_at"]
        if due_at is None:
            if not include_undated:
                continue
            days_until = None
            overdue = False
        else:
            due_date = _local_date(due_at, tz)
            days_until = (due_date - today).days
            overdue = days_until < 0
        items.append(
            {
                "id": r["id"],
                "title": r["title"],
                "snippet": _snippet(r["body"]),
                "priority": int(r["priority"]) if r["priority"] is not None else 0,
                "due_at": int(due_at) if due_at is not None else None,
                "days_until": days_until,
                "overdue": overdue,
                "created_at": int(r["created_at"]) if r["created_at"] is not None else 0,
            }
        )

    items.sort(key=_sort_key)
    if limit and limit > 0:
        items = items[:limit]

    return items, _render(items, max_chars)


def _sort_key(item: dict):
    undated = item["due_at"] is None
    return (
        1 if undated else 0,
        item["days_until"] if not undated else 0,
        -item["priority"],
        item["created_at"],
    )


def _read_scheduled(board: str) -> Optional[list[sqlite3.Row]]:
    """Return the ``scheduled`` rows on ``board`` (read-only), or ``None`` when
    the board is absent / unreadable. Never raises."""
    try:
        from hermes_cli import kanban_db
    except Exception:  # pragma: no cover - import guard
        logger.warning("agenda snapshot: kanban_db import failed", exc_info=True)
        return None

    try:
        if not kanban_db.board_exists(board):
            return None
        path = kanban_db.kanban_db_path(board)
    except Exception:
        logger.warning("agenda snapshot: board resolution failed", exc_info=True)
        return None

    if not path.exists():
        return None

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        # Defensive column probe: a pre-due_at board (shouldn't happen — the
        # agenda board is created post-migration — but a hand-rolled board
        # might lack it) is treated as all-undated rather than crashing.
        has_due = any(
            c["name"] == "due_at" for c in conn.execute("PRAGMA table_info(tasks)")
        )
        due_expr = "due_at" if has_due else "NULL AS due_at"
        rows = conn.execute(
            f"SELECT id, title, body, priority, created_at, {due_expr} "
            "FROM tasks WHERE status = 'scheduled'"
        ).fetchall()
        return list(rows)
    except Exception:
        logger.warning("agenda snapshot: read failed for board %r", board, exc_info=True)
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
