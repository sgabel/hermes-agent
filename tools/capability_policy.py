"""Capability-tiered dispatch policy — the single central action gate (PRD-032 R1).

This is the *enforce* half PRD-028 handed forward (`autonomy/__init__.py`): a
fail-closed `classify(tool,args,ctx) → tier` + `guard(...)` that EVERY tool-
execution path routes through before execution. It implements the §5 enforcement
contract of `docs/reference/AGENT_SECURITY_MODEL.md` (invariants cited as Ix).

Why this module and not a gate inside `model_tools.handle_function_call`: the
pass-3 dual adversarial review (Codex gpt-5.5 + opus, 2026-06-26) proved
`handle_function_call` is NOT the single chokepoint — the inline agent-runtime
tools (incl. `delegate_task`, the I5 surface) never reach it, the plugin path
calls `registry.dispatch` directly, and cron scripts shell out entirely. So the
gate is a central `guard()` wired at the true fan-ins (owner decision 2026-06-26:
`registry.dispatch` facade + inline branches + cron own-point + MCP adapters).

ROLLOUT SAFETY: the gate ships in **observe** mode (classify + audit, never
blocks) so wiring it into the hot dispatch path cannot brick the agent. Flip to
**enforce** deliberately (config `autonomy.capability_policy_mode: enforce`) only
after the classifier is validated against the full tool surface. In observe mode
a classifier error fails OPEN (allow + log); in enforce mode it fails CLOSED.

Scope of THIS cut (R1 foundation): the central gate + a conservative, fail-closed
tier map + per-action kill-switch/budget for unattended (R3) + audit of every
decision. The richer per-surface classification (R5 effect-parsed reads, R6
FedPulse tool, R7 egress routing, R8 gray-zone gate) refines specific tiers and
hangs off this gate — it does not replace it.
"""

from __future__ import annotations

import logging
import os
from enum import IntEnum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    """The T0–T4 capability ladder (AGENT_SECURITY_MODEL §4)."""

    T0 = 0  # pure observation (no subprocess/hook/network/device/taint)
    T1 = 1  # read-external — egress to a sanctioned destination
    T2 = 2  # contained write — create/run inside the sandbox
    T3 = 3  # outbound message — one bounded recipient-locked message
    T4 = 4  # host mutation / privileged / drive-another-agent / control-plane


# ---------------------------------------------------------------------------
# Tier map (R1 baseline — coarse but fail-closed; refined by R5/R6/R7).
# Anything NOT named here classifies to T4 (default-deny, I4).
# ---------------------------------------------------------------------------

# T0 — pure observation: reads, searches, local introspection. None of these
# egress, spawn host subprocesses, or mutate host state. (Tool names verified
# against the live registry 2026-06-26.)
_T0_TOOLS = frozenset({
    "read_file", "read_many_files", "list_directory", "list_dir", "glob",
    "grep", "search_file_content", "ripgrep", "find_files", "search_files",
    "session_search", "chronicle_search", "todo", "read_terminal", "clarify",
    "get_diff", "git_status", "mem0_search", "mem0_profile",
    "skills_list", "skill_view", "kanban_show", "kanban_list",
})

# T1 — read-external (egress to a sanctioned destination). These are exfil
# surfaces (I6) — allowed within budget + redaction, audited.
_T1_TOOLS = frozenset({
    "web_search", "web_extract", "x_search", "ask_claude",
    "browser_navigate", "browser_snapshot", "browser_click", "browser_type",
    "browser_select", "browser_scroll", "browser_extract", "browser_screenshot",
    "browser_dialog", "browser_back", "browser_forward", "browser_wait",
    "browser_tabs", "browser_close", "browser_evaluate", "fetch_url",
    "browser_console", "browser_get_images", "browser_press", "browser_vision",
})

# T3 — outbound message to the owner (consent-first).
_T3_TOOLS = frozenset({
    "send_message", "send_discord", "notify_owner", "proactive_message",
})

# T2 — contained writes to Sylva's OWN datastore/memory (local Qdrant + the
# managed MEMORY.md/USER.md files). Not host-fs mutation of arbitrary paths, not
# egress, not FedPulse — so allowed unattended. The nightly reflection +
# memory-cleanup cron jobs depend on these.
_MEMORY_WRITE_TOOLS = frozenset({"mem0_conclude", "memory"})

# File-mutation tools — tier is path-dependent (R5/AC-019): a write resolved
# under an allowed workspace root is T2; anywhere else on the host (or under a
# forbidden/secret root) is T4. Resolved in classify() via _classify_write_path.
_WRITE_TOOLS = frozenset({
    "write_file", "patch", "edit_file", "create_file", "apply_diff",
    "move_file", "delete_file", "append_file", "str_replace", "multi_edit",
})

# T4 — host mutation / privileged / agent-driver / control-plane. The default is
# ALREADY T4 (unknown → T4), so absence is safe; named for clarity/audit.
_T4_TOOLS = frozenset({
    "delegate_task",          # I5 — drives another capable agent
    "run_shell_command",
})

# Execution tools whose tier depends on the active backend (sandbox = T2,
# anything else = T4 host mutation). Resolved in classify().
_EXEC_TOOLS = frozenset({"terminal", "execute_code", "run_command", "shell"})

# The dedicated contained backend (PRD-026 FR-7). Execution on this backend is
# T2 (contained); on any other backend it is T4 (can reach the host).
_SANDBOX_BACKEND = "sylva-sandbox"

# Args keys a file-mutation tool may carry its target path under.
_PATH_ARG_KEYS = ("file_path", "path", "filename", "filepath", "target_file",
                  "target_path", "dest", "destination")


def _active_terminal_backend() -> str:
    """Best-effort read of the active terminal/exec backend."""
    try:
        from tools.terminal_tool import _get_env_config
        return str(_get_env_config().get("env_type", "local"))
    except Exception:
        return os.getenv("TERMINAL_ENV", "local").strip().lower() or "local"


# ---------------------------------------------------------------------------
# R5 — path-aware write classification (AC-019). Realpath/symlink-resolved.
# ---------------------------------------------------------------------------

def _home() -> "Path":
    from pathlib import Path
    return Path(os.path.expanduser("~"))


def _hermes_home() -> "Path":
    from pathlib import Path
    return Path(os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes")))


# Roots that are NEVER unattended-writable, even via a symlink from an allowed
# root (forbidden wins). Credentials/config/autonomy-state, FedPulse + sidekick
# (mission-critical), the agent's own source tree (autonomy perimeter), and
# system dirs. Resolved at call time so HERMES_HOME monkeypatching is honored.
def _forbidden_write_roots() -> list:
    from pathlib import Path
    home = _home()
    roots = [
        _hermes_home(),                       # .env / auth.json / config.yaml / autonomy/
        home / "vaelyn" / "fedpulse",         # MISSION-CRITICAL FedPulse
        home / "vaelyn" / "sidekick",         # MISSION-CRITICAL sidekick
        home / ".ssh", home / ".gnupg", home / ".config", home / ".aws",
        Path("/etc"), Path("/usr"), Path("/bin"), Path("/sbin"),
        Path("/boot"), Path("/lib"), Path("/lib64"), Path("/root"),
        Path("/var/run"), Path("/run"),
    ]
    # The hermes-agent source tree (this file's repo) — never self-modify unattended.
    try:
        roots.append(Path(__file__).resolve().parents[1])  # .../hermes-agent
    except Exception:
        pass
    return [_resolve(p) for p in roots]


# Filename patterns that are secrets regardless of location → always T4.
_SECRET_NAMES = ("auth.json", ".pgpass", ".netrc", "credentials", "credentials.json")
_SECRET_SUFFIXES = (".env", ".key", ".pem", ".p12", ".pfx")
_SECRET_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa")


def _unattended_write_roots() -> list:
    """Allowlist of roots an unattended write may target (→ T2). Config-driven.

    Default: the sandbox scratch workdir only (fail-closed — the owner opts
    additional repo dirs in via ``autonomy.unattended_write_roots``). Forbidden
    roots always win even if nested under an allowed root.
    """
    from pathlib import Path
    roots = [_home() / "hermes" / "sandbox" / "work"]
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        extra = cfg_get(read_raw_config(), "autonomy", "unattended_write_roots",
                        default=None)
        if isinstance(extra, (list, tuple)):
            roots.extend(Path(os.path.expanduser(str(p))) for p in extra)
    except Exception:
        pass
    return [_resolve(p) for p in roots]


def _resolve(p) -> "Path":
    """Resolve a path (symlinks + ..) without requiring it to exist.

    For a not-yet-created file, resolves the nearest existing ancestor so a
    symlinked parent can't smuggle a write outside an allowed root (I9).
    """
    from pathlib import Path
    p = Path(os.path.expanduser(str(p)))
    try:
        return p.resolve()
    except Exception:
        # Walk up to the first existing ancestor, resolve it, re-append the tail.
        cur = p
        tail = []
        while True:
            if cur.exists():
                try:
                    return cur.resolve().joinpath(*reversed(tail))
                except Exception:
                    return cur
            if cur.parent == cur:
                return p  # reached root without finding an existing ancestor
            tail.append(cur.name)
            cur = cur.parent


def _under(path, root) -> bool:
    return path == root or root in path.parents


def _extract_path(args: dict):
    for k in _PATH_ARG_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _classify_write_path(args: dict) -> Tier:
    """A file mutation is T2 only if it resolves under an allowed write root AND
    not under any forbidden root / secret name. Anything else → T4 (fail-closed).
    """
    raw = _extract_path(args)
    if not raw:
        return Tier.T4  # can't verify the target → deny-by-default
    target = _resolve(raw)
    low = target.name.lower()
    # Secret files are T4 wherever they live.
    if (low in _SECRET_NAMES or low.endswith(_SECRET_SUFFIXES)
            or low.startswith(_SECRET_PREFIXES)):
        return Tier.T4
    # Forbidden roots win even if nested under an allowed root.
    for forbidden in _forbidden_write_roots():
        if _under(target, forbidden):
            return Tier.T4
    for allowed in _unattended_write_roots():
        if _under(target, allowed):
            return Tier.T2
    return Tier.T4


def classify(tool: str, args: Optional[Dict[str, Any]] = None,
             ctx: Optional[Dict[str, Any]] = None) -> Tier:
    """Map a tool invocation to its capability tier. Fail-closed: unknown → T4.

    This is intentionally conservative. Refinements (effect-parsed terminal
    commands per R5, resolved-path file writes per AC-019, container-by-ID per
    AC-018, FedPulse read tool per R6) tighten specific cases but must never
    *lower* the default below T4 for anything they don't positively recognize.
    """
    args = args or {}
    name = (tool or "").strip()

    if name in _T0_TOOLS:
        return Tier.T0
    if name in _T1_TOOLS:
        return Tier.T1
    if name in _T3_TOOLS:
        return Tier.T3
    if name in _MEMORY_WRITE_TOOLS:
        # Contained write to Sylva's own memory store / managed memory files.
        return Tier.T2
    if name == "cron_script":
        # An owner-authored script under ~/.hermes/scripts/ is a deliberate
        # capability grant (I3) → T2 (trusted automation, audited + kill-switch-
        # gated upstream). A path that resolves outside that dir → T4.
        raw = args.get("script_path") or args.get("path")
        if not raw:
            return Tier.T4
        target = _resolve(raw)
        scripts_dir = _resolve(_hermes_home() / "scripts")
        return Tier.T2 if _under(target, scripts_dir) else Tier.T4
    if name in _EXEC_TOOLS:
        # Contained only when running on the dedicated sandbox backend.
        backend = (ctx or {}).get("backend") or _active_terminal_backend()
        return Tier.T2 if backend == _SANDBOX_BACKEND else Tier.T4
    if name in _WRITE_TOOLS:
        # R5/AC-019 — T2 only if resolved under an allowed write root.
        return _classify_write_path(args)
    # _T4_TOOLS and everything unrecognized (unknown tool, MCP, plugin, agent-
    # driver skill) → T4 (I4 default-deny).
    return Tier.T4


# ---------------------------------------------------------------------------
# Mode + context
# ---------------------------------------------------------------------------

def policy_mode() -> str:
    """Return the enforcement mode: 'observe' (default, audit-only) or 'enforce'.

    Read from config `autonomy.capability_policy_mode`; absent → 'observe' so the
    gate never blocks until deliberately turned on. Env override
    `HERMES_CAPABILITY_POLICY_MODE` wins (for tests / break-glass).
    """
    env = os.getenv("HERMES_CAPABILITY_POLICY_MODE")
    if env:
        return env.strip().lower()
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        val = cfg_get(read_raw_config(), "autonomy", "capability_policy_mode",
                      default="observe")
        return str(val or "observe").strip().lower()
    except Exception:
        return "observe"


def is_unattended(ctx: Optional[Dict[str, Any]] = None) -> bool:
    """True when no human is present to approve (cron / autonomous goal).

    Attended (CLI / gateway / ask) returns False — the existing manual approval
    gate covers attended T4. Unattended is where deny-by-default T4 + per-action
    budget/kill-switch (R3) bite.
    """
    if ctx and ctx.get("unattended") is not None:
        return bool(ctx["unattended"])
    if os.getenv("HERMES_CRON_SESSION"):
        return True
    if os.getenv("HERMES_AUTONOMOUS"):
        return True
    return False


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def guard(tool: str, args: Optional[Dict[str, Any]] = None,
          ctx: Optional[Dict[str, Any]] = None,
          surface: str = "dispatch") -> Dict[str, Any]:
    """Classify + enforce one action. THE central action gate (R1).

    Returns ``{"allowed": bool, "tier": str, "mode": str, "reason": str|None,
    "outcome": str}``. Callers MUST honor ``allowed=False`` (do not execute).

    Exception-safe: in observe mode any internal error fails OPEN (allow + log);
    in enforce mode it fails CLOSED (deny) — a gate that can't decide must not
    let an unclassified action through unattended (I4).
    """
    mode = policy_mode()
    try:
        tier = classify(tool, args, ctx)
        unattended = is_unattended(ctx)
        return _enforce(tool, tier, mode, unattended, surface, args or {})
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("capability_policy.guard error for %s: %s", tool, exc)
        if mode == "enforce" and is_unattended(ctx):
            return {"allowed": False, "tier": "T4", "mode": mode,
                    "reason": f"gate error, failing closed: {exc}",
                    "outcome": "blocked"}
        return {"allowed": True, "tier": "T4", "mode": mode,
                "reason": f"gate error, failing open (observe/attended): {exc}",
                "outcome": "error-open"}


def _should_audit(tier: Tier, mode: str, outcome: str) -> bool:
    """Keep the hash-chained ledger meaningful — don't flood it on the hot path.

    Always record a block. In enforce mode record T1+ decisions (the audited
    autonomous surface). In observe mode record only T4 (the actionable "this
    would be denied" signal); T0–T3 allows go to debug log. Per-call auditing of
    T0 reads is a deliberate volume tradeoff (T0 is allow-broadly regardless).
    """
    if outcome == "blocked":
        return True
    if mode == "enforce":
        return tier >= Tier.T1
    return tier >= Tier.T4


def _audit(tier: Tier, surface: str, action: str, outcome: str,
           mode: str = "observe", rationale: str = "") -> None:
    if not _should_audit(tier, mode, outcome):
        logger.debug("capability_policy %s %s %s (%s)", tier.name, outcome, action, mode)
        return
    try:
        from autonomy import audit
        audit.record(tier=tier.name, surface=surface, action=action,
                     rationale=rationale, outcome=outcome)
    except Exception:
        pass


def _resolved_target(tool: str, args: Dict[str, Any]) -> str:
    """A stable identity for the action's target, for the R4 approval hash (I9).

    For file mutations: the realpath-resolved path. For exec: the backend. Else
    empty (the hash falls back to tool+args)."""
    try:
        if tool in _WRITE_TOOLS:
            raw = _extract_path(args)
            return str(_resolve(raw)) if raw else ""
        if tool in _EXEC_TOOLS:
            return _active_terminal_backend()
    except Exception:
        pass
    return ""


def _enforce(tool: str, tier: Tier, mode: str, unattended: bool,
             surface: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the tier decision for the (attended|unattended) × (observe|enforce) cell."""
    action = f"{tool} [{tier.name}]"

    # Observe mode: classify + audit, never block. Safe default.
    if mode != "enforce":
        _audit(tier, surface, action, "observed", mode=mode,
               rationale=f"observe-mode ({'unattended' if unattended else 'attended'})")
        return {"allowed": True, "tier": tier.name, "mode": mode,
                "reason": None, "outcome": "observed"}

    # --- enforce mode ---
    if not unattended:
        # Attended: the existing manual approval gate (approval.py) covers T4.
        # The capability gate audits here but does not double-prompt; T0–T3 pass.
        _audit(tier, surface, action, "allowed", mode=mode, rationale="attended")
        return {"allowed": True, "tier": tier.name, "mode": mode,
                "reason": None, "outcome": "allowed"}

    # Unattended enforce — the real fail-closed path (R3/R9).
    # 1. Kill-switch: halt new T1–T4 at the next action (I11/§5.2).
    if tier >= Tier.T1:
        try:
            from autonomy import killswitch
            if killswitch.is_quiesced():
                _audit(tier, surface, action, "blocked", mode=mode, rationale="kill-switch engaged")
                return {"allowed": False, "tier": tier.name, "mode": mode,
                        "reason": "kill-switch engaged — autonomous work quiesced",
                        "outcome": "blocked"}
        except Exception:
            pass

    # 2. T4 unattended → deny-by-default, degrade-to-ask via the DURABLE,
    #    one-shot, per-action approval store (R4 / Codex STOP-D). If this exact
    #    action was already approved, consume the one-shot grant and allow once;
    #    otherwise queue it and block (no hard fail, no approve-all, no stale exec).
    if tier >= Tier.T4:
        target = _resolved_target(tool, args)
        try:
            from tools import capability_approvals as approvals
            if approvals.check_and_consume(tool, args, target):
                _audit(tier, surface, action, "allowed", mode=mode,
                       rationale="one-shot human approval consumed (R4)")
                return {"allowed": True, "tier": tier.name, "mode": mode,
                        "reason": None, "outcome": "approved-once"}
            sub = approvals.submit(tool, args, target,
                                   rationale=f"T4 unattended via {surface}", tier=tier.name)
            _audit(tier, surface, action, "blocked", mode=mode,
                   rationale=f"T4 queued for one-shot approval (R4) hash={sub.get('hash')}")
            return {"allowed": False, "tier": tier.name, "mode": mode,
                    "reason": ("T4 action requires human approval (host mutation / "
                               "privileged / drives another agent). Queued one-shot — "
                               f"approve with: hermes autonomy approve {sub.get('hash','')[:12]}"),
                    "outcome": "blocked_queued", "approval_hash": sub.get("hash")}
        except Exception:
            # Approval store unavailable → fail closed (hard deny, audited).
            _audit(tier, surface, action, "blocked", mode=mode,
                   rationale="T4 denied unattended; approval store unavailable (fail-closed)")
            return {"allowed": False, "tier": tier.name, "mode": mode,
                    "reason": "T4 action denied unattended (approval store unavailable).",
                    "outcome": "blocked"}

    # 3. T1–T3 unattended → allow within per-action budget (R3).
    try:
        from autonomy import budget
        if not budget.check("actions", 1):
            _audit(tier, surface, action, "blocked", mode=mode, rationale="daily action budget exhausted")
            return {"allowed": False, "tier": tier.name, "mode": mode,
                    "reason": "daily autonomous action budget exhausted — degrade to ask",
                    "outcome": "blocked"}
        budget.debit(surface, "actions", 1, audit=False)
    except Exception:
        pass

    _audit(tier, surface, action, "allowed", mode=mode, rationale="unattended within budget")
    return {"allowed": True, "tier": tier.name, "mode": mode,
            "reason": None, "outcome": "allowed"}


def deny_result(tool: str, guard_out: Dict[str, Any]) -> str:
    """Render a guard denial as the standard tool-error JSON string."""
    import json
    return json.dumps({
        "error": (
            f"BLOCKED by capability policy ({guard_out.get('tier')}): "
            f"{guard_out.get('reason') or 'denied'}"
        ),
        "capability_tier": guard_out.get("tier"),
        "outcome": guard_out.get("outcome"),
    }, ensure_ascii=False)
