"""Contained execution profiles for autonomous cron runs (PRD-027 FR-6/7/8).

Deliberately named **execution profile** — job field ``exec_profile``, module
``autonomy.exec_profile`` — NOT plain "profile", which already means the
user/workspace identity in ``hermes_cli.profiles`` (passed to the memory
provider as ``agent_identity`` at ``agent/agent_init.py:1270``). Conflating the
two names is a review CONCERN this PRD folded.

An execution profile is the allowlist job-type that PRD-044's FR-3 split handed
here (parking ``prd044-profile-mechanism``). It closes every non-tool-catalog
execution path the cron surface exposes (``script`` / ``no_agent`` / pre-run
script / MCP init / lazy installs), pins an EXACT-NAME read-only tool surface
asserted after post-build provider injection, binds the ``proactive`` run
identity (unattended floor, markers-beat-YOLO), and fail-closed-delivers to a
config-pinned home channel.

Cron is **in-process and thread-based**, not a fresh process — so every closure
this profile drives is **run-scoped (contextvar), never process-env-scoped**
(jobs share thread pools with copied contextvars). See ``cron/scheduler.py`` for
the wiring and ``tools/lazy_deps.py::lazy_install_override`` for the ContextVar
that disables lazy installs for the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from autonomy import run_identity

logger = logging.getLogger(__name__)


class ExecProfileError(Exception):
    """A profile run cannot proceed and MUST fail closed.

    Raised for an unresolvable/blank delivery pin, an unknown profile name, or a
    resolved tool surface that does not match the declared allowlist EXACTLY.
    Never degrade to a laxer surface — the whole safety case (AC-004/AC-009) is
    that a profile that cannot enforce its own contract aborts the run.
    """


@dataclass(frozen=True)
class PinnedTarget:
    """A fail-closed delivery target as a canonical 3-tuple (FR-7).

    ``thread_id`` is REQUIRED to be ``None`` for a proactive profile — delivery
    targets are 3-tuples and a stray ``DISCORD_HOME_CHANNEL_THREAD_ID`` must not
    attach a thread. The choke-point assert (``cron/scheduler.py``) compares the
    concrete resolved tuple against ``as_tuple()``.
    """

    platform: str
    chat_id: str
    thread_id: Optional[str] = None

    def as_tuple(self) -> tuple[str, str, Optional[str]]:
        return (self.platform.lower(), str(self.chat_id), self.thread_id)


@dataclass(frozen=True)
class ExecProfile:
    """A named contained execution profile.

    Holds the STATIC declaration only. Config-derived values (the delivery pin,
    quiet-hours window, budget cap) are resolved from ``autonomy.proactive.*`` /
    ``autonomy.budget.*`` at run time via the module functions below — mirroring
    ``autonomy/budget.py`` which reads caps from config at call time so a config
    edit takes effect without an image rebuild.
    """

    #: Profile name, matched against a job's ``exec_profile`` field.
    name: str
    #: EXACT-NAME tool allowlist. The constructed agent's resolved
    #: ``valid_tool_names`` must equal this set EXACTLY (after provider
    #: injection) or the run aborts fail-closed. See the module docstring in
    #: PRD-027 FR-6 for the three backing inputs.
    tool_allowlist: frozenset[str]
    #: Construction ``enabled_toolsets`` — OVERRIDES any job-level
    #: ``enabled_toolsets``. ``memory`` is present ONLY to make
    #: ``memory_provider_tools_enabled`` return True so the provider injects
    #: ``chronicle_search``; the built-in ``memory`` tool is stripped by
    #: ``disabled_toolsets`` below (verified: registry then resolves to just
    #: ``session_search``, provider adds ``chronicle_search``).
    enabled_toolsets: tuple[str, ...]
    #: Construction ``disabled_toolsets`` — a job/PROFILE-scoped denylist, NOT
    #: the global ``agent.disabled_toolsets`` (wrong scope, review STOP-2).
    disabled_toolsets: tuple[str, ...]
    #: Skip ``discover_mcp_tools()`` and set ``agent._skip_mcp_refresh`` so the
    #: per-turn MCP re-injection prologue is closed.
    skip_mcp: bool = True
    #: Disable lazy dependency installs for the run scope (ContextVar-backed).
    disable_lazy_installs: bool = True
    #: Reject ``script`` / ``no_agent`` / pre-run script at run-entry.
    reject_script_no_agent: bool = True
    #: Run identity to bind (unattended floor, markers-beat-YOLO).
    identity: str = run_identity.PROACTIVE
    #: Budget kind debited on a confirmed successful send.
    budget_kind: str = "proactive_messages"
    #: Audit surface + tier the run is recorded under.
    audit_surface: str = "proactive"
    tier: str = "T3"
    #: Kill-switch surface label checked at run-entry.
    killswitch_surface: str = "proactive"


# ---------------------------------------------------------------------------
# The registry. First (and only, for the MVP) instance: proactive_read.
# ---------------------------------------------------------------------------

#: D-3 (owner default): exactly ``session_search`` (a static toolset) +
#: ``chronicle_search`` (provider-injected) — deliberately exercising both
#: backing inputs. No Discord read (the ``discord`` toolset writes);
#: repeat-avoidance is ``session_search`` over prior proactive sessions.
#: EMPIRICALLY VERIFIED (HEAD 7c829e32e): the registry resolves this
#: enabled/disabled pair to exactly {session_search}, and the mem0 provider's
#: get_tool_schemas() returns exactly [chronicle_search]; no core/finish/think
#: tool is auto-added (get_tool_definitions is toolset-scoped, and the
#: context-engine tools are gated on "context_engine" in enabled_toolsets).
PROACTIVE_READ = ExecProfile(
    name="proactive_read",
    tool_allowlist=frozenset({"session_search", "chronicle_search"}),
    enabled_toolsets=("session_search", "memory"),
    disabled_toolsets=("cronjob", "messaging", "clarify", "memory"),
)

_PROFILES: dict[str, ExecProfile] = {
    PROACTIVE_READ.name: PROACTIVE_READ,
}

#: Tool Search bridge names — must NEVER appear in a profile tool surface (they
#: collapse/recurse into the deferred catalog past a visible-names assertion).
_TOOL_SEARCH_BRIDGE = frozenset({"tool_search", "tool_describe", "tool_call"})


def get_exec_profile(name: Optional[str]) -> Optional[ExecProfile]:
    """Return the profile for ``name``, or ``None`` if unknown.

    An unknown name is a fail-closed condition at the call site (no run), never
    a silent fall-through to the ordinary cron path.
    """
    if not name:
        return None
    return _PROFILES.get(str(name))


def known_profile_names() -> frozenset[str]:
    """Names accepted by the ``exec_profile`` job field (creation-time check)."""
    return frozenset(_PROFILES)


# ---------------------------------------------------------------------------
# Tool-surface assertion (FR-6) — fail closed, never degrade.
# ---------------------------------------------------------------------------


def assert_tool_surface(agent, profile: ExecProfile) -> None:
    """Assert the agent's resolved tool surface == the profile allowlist EXACTLY.

    Called AFTER construction + post-build provider injection, AND re-run in the
    per-turn prologue (``agent/turn_context.py``) before each model call. On any
    mismatch — a missing tool, an extra tool, an MCP tool name, or a Tool Search
    bridge name — raises :class:`ExecProfileError` so the caller aborts the run
    fail-closed. Never silently narrows or accepts a superset.
    """
    resolved = set(getattr(agent, "valid_tool_names", None) or set())
    allowed = set(profile.tool_allowlist)

    bridge = resolved & _TOOL_SEARCH_BRIDGE
    if bridge:
        raise ExecProfileError(
            f"exec_profile {profile.name!r}: Tool Search bridge tool(s) present "
            f"in resolved surface {sorted(bridge)} — refusing to run (fail closed)"
        )

    if resolved != allowed:
        extra = sorted(resolved - allowed)
        missing = sorted(allowed - resolved)
        raise ExecProfileError(
            f"exec_profile {profile.name!r}: resolved tool surface "
            f"{sorted(resolved)} != declared allowlist {sorted(allowed)} "
            f"(extra={extra}, missing={missing}) — refusing to run (fail closed)"
        )


# ---------------------------------------------------------------------------
# Config-derived resolution (read at run time, like autonomy/budget.py).
# ---------------------------------------------------------------------------


def _read_cfg() -> dict:
    try:
        from hermes_cli.config import read_raw_config

        return read_raw_config() or {}
    except Exception:
        return {}


def resolve_pinned_target(cfg: Optional[dict] = None) -> PinnedTarget:
    """Resolve the fail-closed delivery pin from ``autonomy.proactive.pinned_target``.

    Format ``platform:chat_id`` (e.g. ``discord:1490111277819756564``).
    Missing / blank / malformed / carrying a thread segment → raise
    :class:`ExecProfileError` (the profile run REFUSES to run — fail closed).
    """
    if cfg is None:
        cfg = _read_cfg()
    try:
        from hermes_cli.config import cfg_get

        raw = cfg_get(cfg, "autonomy", "proactive", "pinned_target", default="")
    except Exception:
        raw = ""
    text = str(raw or "").strip()
    if not text:
        raise ExecProfileError(
            "autonomy.proactive.pinned_target is missing/blank — proactive run "
            "refuses to run (fail closed, FR-7/D-2)"
        )
    parts = text.split(":")
    if len(parts) != 2:
        raise ExecProfileError(
            f"autonomy.proactive.pinned_target must be 'platform:chat_id' with "
            f"thread_id None (got {text!r}) — refusing to run (fail closed)"
        )
    platform = parts[0].strip().lower()
    chat_id = parts[1].strip()
    if not platform or not chat_id:
        raise ExecProfileError(
            f"autonomy.proactive.pinned_target has an empty platform/chat_id "
            f"(got {text!r}) — refusing to run (fail closed)"
        )
    return PinnedTarget(platform=platform, chat_id=chat_id, thread_id=None)


# Default quiet-hours window (Scott-local wall clock via HERMES_TIMEZONE).
_DEFAULT_QUIET_START = "22:00"
_DEFAULT_QUIET_END = "08:00"


def resolve_quiet_hours(cfg: Optional[dict] = None) -> tuple[str, str]:
    """Return the ``(start, end)`` quiet-hours window as ``"HH:MM"`` strings.

    Reads ``autonomy.proactive.quiet_hours.{start,end}``; defaults 22:00/08:00.
    """
    if cfg is None:
        cfg = _read_cfg()
    try:
        from hermes_cli.config import cfg_get

        qh = cfg_get(cfg, "autonomy", "proactive", "quiet_hours", default=None)
    except Exception:
        qh = None
    if not isinstance(qh, dict):
        return (_DEFAULT_QUIET_START, _DEFAULT_QUIET_END)
    start = str(qh.get("start") or _DEFAULT_QUIET_START).strip() or _DEFAULT_QUIET_START
    end = str(qh.get("end") or _DEFAULT_QUIET_END).strip() or _DEFAULT_QUIET_END
    return (start, end)


def _hhmm_to_minutes(value: str) -> int:
    """Parse ``"HH:MM"`` to minutes-of-day; raise ValueError on garbage."""
    h_str, _, m_str = value.partition(":")
    h, m = int(h_str), int(m_str)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"time out of range: {value!r}")
    return h * 60 + m


def in_quiet_window(now_minutes: int, start: str, end: str) -> bool:
    """Whether ``now_minutes`` (minutes-of-day) falls in the quiet window.

    Handles a window that wraps midnight (``start > end``, e.g. 22:00→08:00).
    A zero-width window (``start == end``) means quiet hours are DISABLED.
    A malformed window fails SAFE toward suppression? No — fails OPEN (not
    quiet) so a typo can't permanently silence outreach; the misconfig is
    surfaced by the caller's log.
    """
    try:
        start_m = _hhmm_to_minutes(start)
        end_m = _hhmm_to_minutes(end)
    except (ValueError, TypeError):
        logger.warning(
            "proactive quiet_hours malformed (start=%r end=%r) — treating as "
            "OUTSIDE the quiet window", start, end,
        )
        return False
    if start_m == end_m:
        return False
    if start_m < end_m:
        return start_m <= now_minutes < end_m
    return now_minutes >= start_m or now_minutes < end_m


def is_quiet_now(cfg: Optional[dict] = None) -> bool:
    """True if the current HERMES_TIMEZONE wall-clock time is in quiet hours."""
    from hermes_time import now as _now

    start, end = resolve_quiet_hours(cfg)
    dt = _now()
    return in_quiet_window(dt.hour * 60 + dt.minute, start, end)


__all__ = [
    "ExecProfile",
    "ExecProfileError",
    "PinnedTarget",
    "PROACTIVE_READ",
    "get_exec_profile",
    "known_profile_names",
    "assert_tool_surface",
    "resolve_pinned_target",
    "resolve_quiet_hours",
    "in_quiet_window",
    "is_quiet_now",
]
