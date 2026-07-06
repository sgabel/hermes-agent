"""Dashboard security-floor denylist (PRD-045 FR-6).

Fork-local guard that keeps the **security floor** immutable through the web
dashboard's config/env write surface, even for an authenticated client.

Background (PRD-045, Codex STOP-4 + re-review): the dashboard's
``PUT /api/config`` / ``PUT /api/config/raw`` persist the whole submitted
config verbatim (``save_config`` overwrites), and ``PUT /api/env`` /
``POST /api/env/reveal`` write/disclose arbitrary ``.env`` values. FR-0 auth
alone does NOT satisfy AC-005 — an authenticated (or credential-stealing)
client could still flip ``approvals.mode``, disable tirith, repoint the
proactive ``pinned_target``, or rewrite/exfiltrate ``API_SERVER_KEY`` and the
dashboard's own auth credentials. This module is the server-side enforcement.

Two enforcement points, deliberately different semantics:

* **Config write path (`pin_config_floor`)** — the SPA round-trips the *whole*
  config dict on every save, so a "reject if a floor key is present" rule would
  reject every save, and a "strip the floor keys" rule would *delete* them from
  ``config.yaml``. The correct behaviour is to **pin**: force each floor path in
  the incoming payload back to its current on-disk value (and drop it if it is
  absent on disk). Benign full-config saves pass through; the floor is
  immutable-through-dashboard.

* **Env write/reveal path (`is_floor_env_key`)** — ``PUT /api/env`` is
  single-key, so a straight reject of floor env keys is correct. Enforced in the
  *handlers*, NOT by adding to the shared ``_ENV_VAR_NAME_DENYLIST`` in
  ``config.py`` (that would also block the trusted host CLI + setup wizard, which
  legitimately prompt for ``API_SERVER_KEY``). The floor env set is the exact
  credentials that bypass the security/auth stack — NOT provider API keys, which
  the dashboard Keys tab legitimately manages.

The floor keys stay host-only-editable (``config.yaml`` / ``.env`` on the host).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import yaml

from hermes_cli.config import get_config_path

_log = logging.getLogger(__name__)

# --- The security floor ------------------------------------------------------

# Nested config paths (dotted) that must never change via a dashboard config
# write. Each entry may be a scalar leaf OR a subtree node (e.g. the whole
# ``dashboard.basic_auth`` mapping) — pinning replaces the whole node.
CONFIG_FLOOR: tuple[str, ...] = (
    # Approval perimeter (PRD-015) — must not be weakened from the web.
    "approvals.mode",
    "approvals.cron_mode",
    "approvals.manual_whitelist",       # auto-approve list (consumed in the `manual` rollback mode)
    # tirith security stack (PRD-020 hardening).
    "security.tirith_enabled",
    "security.tirith_fail_open",
    "security.tirith_path",
    "security.tirith_timeout",
    # Broader security toggles (SSRF gate + secret redaction).
    "security.allow_private_urls",
    "security.redact_secrets",
    "browser.allow_private_urls",
    # Capability ladder + governor (PRD-032/028) — the load-bearing switches. Note
    # the container dashboard IS agent-reachable (sibling on the internal net) and
    # the agent can read the basic-auth SECRET from the ro .env to forge a session
    # cookie — so this floor, not FR-0 auth, is what stops a code-exec agent
    # disabling its own governor.
    "autonomy.capability_policy_mode",  # THE enforce/observe master switch (default observe when absent)
    "autonomy.budget",                  # governor caps subtree (owner tunes via the host, not the web)
    "autonomy.unattended_write_roots",  # unattended write blast radius
    # Proactive recipient lock + suppression window (PRD-027 FR-7).
    "autonomy.proactive.pinned_target",
    "autonomy.proactive.quiet_hours",
    # Command-execution surfaces — RCE / approval-gate bypass if writable.
    "quick_commands",                   # named /cmd exec (shell=True, ungated) — arbitrary RCE
    "command_allowlist",                # commands allowed to skip approval
    "terminal.env_passthrough",         # env-var names leaked into sandboxed exec
    # The dashboard's own auth config — pinning these prevents an authenticated
    # client from weakening/locking the very gate that protects this surface.
    "dashboard.basic_auth",
    "dashboard.oauth",
)

# Exact ``.env`` keys that cannot be WRITTEN or REVEALED via the dashboard.
# These are the credentials that bypass the security/auth stack, NOT provider
# API keys (OPENAI_API_KEY, …) which the Keys tab legitimately manages — so this
# is an explicit set, never a broad ``*_KEY`` pattern.
ENV_FLOOR: frozenset[str] = frozenset({
    # Gateway API key — reading it lets the agent/LAN drive :8642 directly
    # (the non-web self-reconfig path; also the crown jewel for reveal).
    "API_SERVER_KEY",
    # The dashboard's own basic-auth credentials — rewriting them = lockout or
    # auth-bypass; revealing the hash/secret weakens the gate.
    "HERMES_DASHBOARD_BASIC_AUTH_USERNAME",
    "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
    "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH",
    "HERMES_DASHBOARD_BASIC_AUTH_SECRET",
})


# --- Dotted-path helpers -----------------------------------------------------

def _dotted_get(node: Any, dotted: str) -> tuple[bool, Any]:
    """Return ``(present, value)`` for a dotted path in a nested mapping."""
    parts = dotted.split(".")
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return False, None
        node = node[p]
    return True, node


def _dotted_set(root: dict, dotted: str, value: Any) -> None:
    """Set a dotted path in ``root``, creating intermediate dicts as needed."""
    parts = dotted.split(".")
    node = root
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value


def _dotted_del(root: dict, dotted: str) -> bool:
    """Delete a dotted path from ``root``. Returns True if something was removed."""
    parts = dotted.split(".")
    node = root
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            return False
        node = node[p]
    if isinstance(node, dict) and parts[-1] in node:
        del node[parts[-1]]
        return True
    return False


def _read_on_disk_config() -> dict:
    """Read the raw, unexpanded config.yaml (what ``save_config`` would overwrite).

    Deliberately reads the raw file rather than ``load_config()`` so pinned
    values are the true persisted floor (no ``${VAR}`` expansion / managed-scope
    merge). Returns ``{}`` if the file is missing or unreadable — which makes the
    floor keys get *dropped* from the incoming write (fail-safe: a dashboard
    write can never introduce a floor key that is not already host-provisioned).
    """
    try:
        path = get_config_path()
        if not path.exists():
            return {}
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        _log.warning("floor: could not read on-disk config; treating floor as absent", exc_info=True)
        return {}


# --- Public API --------------------------------------------------------------

def pin_config_floor(incoming: dict, *, on_disk: Optional[dict] = None) -> dict:
    """Force every :data:`CONFIG_FLOOR` path in ``incoming`` to its on-disk value.

    Mutates and returns ``incoming`` (so it drops straight into
    ``save_config(pin_config_floor(...))``). For each floor path: if present on
    disk, overwrite the incoming value with the on-disk one; if absent on disk,
    delete it from ``incoming``. Any attempted change to a floor key is logged.

    ``on_disk`` may be injected for tests; otherwise the raw config.yaml is read.
    """
    if not isinstance(incoming, dict):
        return incoming
    disk = _read_on_disk_config() if on_disk is None else on_disk
    changed: list[str] = []
    for dotted in CONFIG_FLOOR:
        disk_present, disk_val = _dotted_get(disk, dotted)
        inc_present, inc_val = _dotted_get(incoming, dotted)
        if disk_present:
            if not inc_present or inc_val != disk_val:
                _dotted_set(incoming, dotted, disk_val)
                if inc_present:
                    changed.append(dotted)
        else:
            if _dotted_del(incoming, dotted) and inc_val is not None:
                changed.append(dotted)
    if changed:
        _log.warning(
            "floor: dashboard config write attempted to change security-floor keys "
            "%s — pinned to host values (PRD-045 FR-6)", sorted(set(changed)),
        )
    return incoming


def is_floor_env_key(key: str) -> bool:
    """True if ``key`` is a security-floor ``.env`` credential (write/reveal blocked)."""
    if not isinstance(key, str):
        return False
    return key in ENV_FLOOR or key.upper() in ENV_FLOOR


def floor_env_write_message(key: str) -> str:
    return (
        f"{key!r} is a security-floor credential and cannot be set from the "
        f"dashboard. Change it on the host (~/.hermes/.env). (PRD-045 FR-6)"
    )


def floor_env_reveal_message(key: str) -> str:
    return (
        f"{key} is a security-floor credential; reveal is disabled on the web "
        f"surface. Read it on the host if needed. (PRD-045 FR-6)"
    )


def floor_paths() -> Iterable[str]:
    """The config floor paths (for docs/introspection)."""
    return CONFIG_FLOOR
