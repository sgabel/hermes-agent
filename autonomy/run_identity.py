"""Canonical run-identity classification (PRD-044).

ONE place that answers "what kind of run is this, and is a human present to
approve?" — replacing the scattered per-site readings of the attended/unattended
env pile (``HERMES_EXEC_ASK`` / ``HERMES_INTERACTIVE`` / ``HERMES_CRON_SESSION`` /
``HERMES_AUTONOMOUS``) that each gate site interpreted independently.

Why this module exists
----------------------
A misclassification at any single gate site is a silent privilege gain (the
PRD-015 STOP-1 class of bug). Three PRDs independently re-discovered the same
missing primitive; 044 centralises it here.

The wire format (the load-bearing correction, Opus/Codex STOP-2)
---------------------------------------------------------------
Identity binds **per-run via a contextvar** (``_RUN_IDENTITY``), NOT process env.
``cron/scheduler.py`` used to set ``os.environ["HERMES_CRON_SESSION"]="1"``
*process-globally and permanently*: after the first agent-mode cron job, every
subsequent attended gateway session in that process would classify unattended.
Env markers are now **launcher-boundary inputs only** — read where a fresh
process starts (``-z`` oneshot, ``docker exec``, ``no_agent`` scripts) or where a
launcher explicitly binds identity. In-process launchers (the cron scheduler,
delegated children) bind the contextvar for the exact scope of the run instead.

Precedence (Opus/Codex, owner-decided 2026-07-06)
-------------------------------------------------
``classify_run()`` resolves identity in this order; the FULL approval precedence
stack (hardline > sudo-stdin > **run-identity unattended floor, which beats
YOLO** > YOLO/off > attended flags > unmarked_legacy) is applied by the approval
gate functions using this verdict — see ``tools/approval.py``.

1. Explicit run-context (the ``_RUN_IDENTITY`` contextvar) — highest. This is how
   ``delegated_child`` / ``orchestrated_headless`` / ``proactive`` and the cron
   scheduler bind themselves. ``delegated_child`` is representable ONLY this way
   (children share the parent's env byte-for-byte).
2. Unattended env markers — ``HERMES_CRON_SESSION`` -> ``cron``;
   ``HERMES_AUTONOMOUS`` -> ``orchestrated_headless``. Both unattended.
3. Attended flags — a gateway context (``HERMES_GATEWAY_SESSION`` or a bound
   session platform) -> ``gateway_attended``; ``HERMES_INTERACTIVE`` /
   ``HERMES_EXEC_ASK`` -> ``interactive_cli``.
4. Nothing -> ``unmarked_legacy`` (owner decision: keep today's non-interactive
   auto-approve behavior, but LABEL it and WARNING-audit every dangerous
   auto-approval; the fail-closed flip is a parked follow-up gated on a caller
   census).

Ambiguity between declared identities resolves toward LESS privilege
(unattended wins). ``unmarked_legacy`` is a distinct third state — it is NOT
attended (no human) but it does NOT get the unattended cron floor either; it
reproduces today's legacy auto-approve so no existing marker-less caller breaks.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Identity taxonomy
# ---------------------------------------------------------------------------

INTERACTIVE_CLI = "interactive_cli"
GATEWAY_ATTENDED = "gateway_attended"
CRON = "cron"
ORCHESTRATED_HEADLESS = "orchestrated_headless"
PROACTIVE = "proactive"
DELEGATED_CHILD = "delegated_child"
UNMARKED_LEGACY = "unmarked_legacy"

#: Identities where a human is present to approve.
_ATTENDED = frozenset({INTERACTIVE_CLI, GATEWAY_ATTENDED})

#: Governed unattended identities — no human present; the cron/capability
#: unattended floor applies (and, per the owner decision, beats YOLO).
_UNATTENDED_FLOOR = frozenset(
    {CRON, ORCHESTRATED_HEADLESS, PROACTIVE, DELEGATED_CHILD}
)

#: All identities a launcher/context may legitimately bind.
VALID_IDENTITIES = _ATTENDED | _UNATTENDED_FLOOR | {UNMARKED_LEGACY}


@dataclass(frozen=True)
class RunIdentity:
    """The classified identity of the current run.

    ``attended``        — a human is present to approve (interactive/gateway).
    ``unattended_floor``— the governed-unattended cron/capability floor applies
                          (cron/orchestrated_headless/proactive/delegated_child).
                          This is the flag the approval gates use to know they
                          must apply the floor BEFORE the YOLO bypass.
    ``source``          — where the verdict came from (``context`` / ``env`` /
                          ``default``), for audit/debug only — never a gate input.

    ``unmarked_legacy`` is the third state: ``attended`` and ``unattended_floor``
    are BOTH False — it reproduces today's legacy non-interactive auto-approve.
    """

    identity: str
    attended: bool
    unattended_floor: bool
    source: str

    @property
    def is_unattended(self) -> bool:
        """True when no human is present — governed unattended OR unmarked legacy.

        Callers that only need the attended/unattended axis (budget/audit
        surface attribution, ``ask_claude`` label) use this. Gate sites that
        must decide whether to apply the cron floor use ``unattended_floor``.
        """
        return not self.attended

    @property
    def is_legacy(self) -> bool:
        return self.identity == UNMARKED_LEGACY


def _make(identity: str, source: str) -> RunIdentity:
    return RunIdentity(
        identity=identity,
        attended=identity in _ATTENDED,
        unattended_floor=identity in _UNATTENDED_FLOOR,
        source=source,
    )


# ---------------------------------------------------------------------------
# Per-run binding (the authoritative wire format)
# ---------------------------------------------------------------------------

_RUN_IDENTITY: ContextVar[Optional[str]] = ContextVar("hermes_run_identity", default=None)


def bind_run_identity(identity: str) -> Token:
    """Bind the run identity for the current context (task/thread-local).

    Returns the reset ``Token`` — callers MUST pair with ``reset_run_identity``
    in a ``finally`` (or use ``run_identity_scope``) so the binding does not leak
    to the next run sharing this process. Raises on an unknown identity
    (fail-loud: a typo'd bind must never silently fall through to a laxer env
    classification).
    """
    if identity not in VALID_IDENTITIES:
        raise ValueError(
            f"unknown run identity {identity!r}; expected one of {sorted(VALID_IDENTITIES)}"
        )
    return _RUN_IDENTITY.set(identity)


def reset_run_identity(token: Token) -> None:
    _RUN_IDENTITY.reset(token)


class run_identity_scope:
    """Context manager binding an identity for the enclosed block.

    Usage::

        with run_identity_scope(run_identity.CRON):
            ...   # every classify_run() inside sees `cron`
    """

    def __init__(self, identity: str) -> None:
        self._identity = identity
        self._token: Optional[Token] = None

    def __enter__(self) -> "run_identity_scope":
        self._token = bind_run_identity(self._identity)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            reset_run_identity(self._token)
            self._token = None


def bound_identity() -> Optional[str]:
    """The explicitly-bound identity for this context, or None."""
    return _RUN_IDENTITY.get()


# ---------------------------------------------------------------------------
# Env-marker reading (launcher boundary only)
# ---------------------------------------------------------------------------


def _env_truthy(name: str) -> bool:
    """Match the gate sites' ``env_var_enabled`` truthiness (1/true/yes/on)."""
    val = os.getenv(name, "")
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _gateway_context() -> bool:
    """True inside a gateway/API session (env flag or a bound session platform).

    Mirrors ``tools.approval._is_gateway_approval_context`` minus the cron
    guard (cron is resolved earlier here). Kept import-light: read the session
    contextvar directly with an env fallback.
    """
    if _env_truthy("HERMES_GATEWAY_SESSION"):
        return True
    try:
        from gateway.session_context import get_session_env

        return bool(get_session_env("HERMES_SESSION_PLATFORM", "") or "")
    except Exception:
        return bool(os.getenv("HERMES_SESSION_PLATFORM", "") or "")


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


def classify_run() -> RunIdentity:
    """Classify the current run. THE single precedence definition (PRD-044 FR-1).

    Named ``classify_run`` (not ``classify``) to avoid collision with
    ``capability_policy.classify()`` (the tool-tier function).
    """
    # 1. Explicit run-context binding — highest precedence, the authoritative
    #    wire format for in-process launchers and delegated children.
    bound = _RUN_IDENTITY.get()
    if bound is not None:
        return _make(bound, "context")

    # 2. Unattended env markers (launcher boundary). Unattended beats attended:
    #    even if an attended flag leaked into this process (the in-process
    #    scheduler / cli.py legacy set), an unattended marker forces the floor.
    if _env_truthy("HERMES_CRON_SESSION"):
        return _make(CRON, "env")
    if _env_truthy("HERMES_AUTONOMOUS"):
        return _make(ORCHESTRATED_HEADLESS, "env")

    # 3. Attended flags. Gateway first (a cli.py-launched gateway carries both
    #    HERMES_INTERACTIVE and HERMES_EXEC_ASK — the gateway signal is the
    #    authoritative one for that identity).
    if _gateway_context():
        return _make(GATEWAY_ATTENDED, "env")
    if _env_truthy("HERMES_INTERACTIVE") or _env_truthy("HERMES_EXEC_ASK"):
        return _make(INTERACTIVE_CLI, "env")

    # 4. Nothing declared -> legacy. Reproduces today's non-interactive
    #    auto-approve; the gate sites LABEL it + WARNING-audit dangerous
    #    approvals. Fail-closed flip is a parked follow-up (owner decision).
    return _make(UNMARKED_LEGACY, "default")
