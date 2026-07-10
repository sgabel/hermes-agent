"""PRD-049: ``agenda_view`` — read-only standing-agenda tool.

Surfaces the standing agenda (the dedicated ``agenda`` kanban board — items
Scott tracks deliberately: topics, events, projects, priorities) as bounded
text for the proactive (PRD-027) and nightly (PRD-034) reasoning surfaces.

Read-only by construction: it delegates to
``hermes_cli.agenda.build_agenda_snapshot``, which opens the board read-only and
issues zero writes. The tool computes ``days_until`` and flags overdue items —
the model never does the date arithmetic (the local 35B is unreliable at it).

``check_fn`` is AVAILABILITY-BASED (effectively always-true — the
``session_search`` pattern), deliberately NOT a config-``toolsets`` read: a
config-gated check_fn would return ``False`` inside the PRD-027 proactive run
(whose exec_profile asserts an exact tool-name allowlist) and fail-closed-abort
every fire.
"""

from __future__ import annotations

import logging

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


def check_agenda_available() -> bool:
    """Availability check: the agenda snapshot module must be importable.

    Deliberately NOT a config-``toolsets`` read (see module docstring).
    Effectively always-true on a healthy install.
    """
    try:
        from hermes_cli import agenda  # noqa: F401
        return True
    except Exception:
        return False


def agenda_view(include_undated: bool = True) -> str:
    """Return the standing agenda as bounded text (≤ 4,000 chars).

    Never writes. An empty or missing board returns an explicit
    "standing agenda: (empty …)" string, never an error.
    """
    try:
        from hermes_cli.agenda import build_agenda_snapshot
    except Exception as exc:  # pragma: no cover - import guard
        return tool_error(f"agenda_view: unavailable ({exc})")
    try:
        _items, text = build_agenda_snapshot(include_undated=bool(include_undated))
        return text
    except Exception as exc:  # pragma: no cover - build_agenda_snapshot is fail-soft
        logger.exception("agenda_view failed")
        return tool_error(f"agenda_view: {exc}")


AGENDA_VIEW_SCHEMA = {
    "name": "agenda_view",
    "description": (
        "Read the standing agenda: items Scott tracks deliberately — topics, "
        "events, projects, and priorities, each with a standing priority and an "
        "optional due date. Use this to reason anticipatorily (an item due in a "
        "few days is a strong reason to reach out; an overdue item even more so; "
        "a standing priority with no date is a reason for an occasional check-in, "
        "not a daily one). Read-only: it never adds, edits, or completes agenda "
        "items. Each line already carries a pre-computed days-until / OVERDUE "
        "label — do NOT recompute dates yourself. Returns a short explicit "
        "'empty' string when there is nothing on the agenda."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "include_undated": {
                "type": "boolean",
                "description": (
                    "Include standing items that have no due date (default "
                    "true). Set false to see only time-sensitive dated items."
                ),
            },
        },
        "required": [],
    },
}


registry.register(
    name="agenda_view",
    toolset="agenda",
    schema=AGENDA_VIEW_SCHEMA,
    handler=lambda args, **kw: agenda_view(
        include_undated=args.get("include_undated", True),
    ),
    check_fn=check_agenda_available,
    emoji="🗓️",
)
