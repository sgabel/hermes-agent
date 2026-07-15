"""Consolidation pass — the governed proposer of identity candidates (PRD-029 Phase 3).

This is the **sole writer** into ``sylva_candidates`` (AC-004 / AC-017). It reads
episodic reality (recent user-facing sessions) and the structured *agency layer*
(PRD-028 ledger rationales, kanban pull/defer order, work-block intent/retro
records), asks a **neutral derivation model** to PROPOSE identity candidate-deltas,
and writes them — as ``status: candidate`` — via the bespoke direct-Qdrant
:class:`~plugins.memory.canon.store.CanonStore` upsert. It never writes
``sylva_canon`` and never touches SOUL.md: promotion to canon is Phase 4's
ratification gate, bedrock is Scott-authored SOUL.md.

Three hard invariants this module enforces (the ACs it closes):

  * **AC-013 — deterministic recency, cron excluded.** Recent activity is
    discovered by chronological enumeration of *user-facing* sessions
    (``list_sessions_rich(order_by_last_active=True)`` with ``_HIDDEN_SESSION_SOURCES``,
    which Phase 1 extended to include ``"cron"``), NOT keyword ``session_search``.
    The proposer therefore never re-reads its own prior cron reflections
    (the 2026-06-26 echo-chamber "last interaction June 22" bug) and never
    misses a short recent session that shares no tokens with a guessed query.

  * **AC-019 — agency mined as STRUCTURED records, with the F-03 fact/meaning
    split.** What Sylva *chose* / *declined* is a verifiable ``source_event``
    (ledger rationale + kanban order); the *why* is ``interpretation`` (gated
    downstream like any other candidate). Two things stay out by construction:
    raw cron/reflection transcripts (same exclusion as AC-013) and
    build-execution cruft (diffs / tool-calls / test output) — the derivation
    prompt forbids the latter and the agency gather reads only structured rows.

  * **AC-004 / AC-017 — sole, deterministic, direct-Qdrant writer.** Writes are
    plain Qdrant upserts in this module (the function name carries ``consolidat``
    so the sole-writer grep stays green), never ``mem0_add`` (which binds one
    fixed collection and cannot target the canon collections — STOP-3). The
    derivation model is configurable (``auxiliary.canon_consolidation.model``)
    and defaults to a neutral model (the main/35B), never Qwen3-4b and never
    Sylva-on-herself.

Scope note: this module is the consolidation *engine*. Arming a live nightly
cron is owner-gated and deferred until the Phase 4 ratification gate exists —
re-arming an autonomous memory writer before the gate would resume the exact
ungoverned-autosave loop PRD-029 exists to kill. ``scripts/consolidation_run.py``
drives a manual / sandbox run; the cron job spec lives there, disabled.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    CANDIDATES_COLLECTION,
    CANON_COLLECTION,
    CONSOLIDATION_PROVENANCE,
    FACETS,
    LAYER_IDENTITY,
    content_hash,
    make_payload,
    make_source_event,
    validate_consolidation_payload,
)
from .store import CanonStore

logger = logging.getLogger(__name__)

# Stable UUID namespace so the same (statement, source) proposed on two nights
# upserts to the same point instead of duplicating — idempotent candidate ids.
_CANDIDATE_NS = uuid.UUID("c0a1de29-0000-4000-8000-000000000003")

# Derivation models that are explicitly NOT neutral (AC-004): the tiny aux model
# and Sylva-on-herself. The auxiliary client's default (main model / local 35B)
# is neutral and is what we want when no override is set.
_NEUTRAL_AUX_TASK = "canon_consolidation"

# Tier the proposer is allowed to assign. bedrock is SOUL.md-only (rejected by
# validate_payload anyway); a fresh proposal defaults to peripheral and is
# promoted/retiered only through ratification.
_ALLOWED_PROPOSAL_TIERS = ("core", "peripheral")

# Collections the consolidation writer may target. The candidate staging
# collection (production) plus the sandbox prefix (AC-010 validation runs into
# sylva_lab*). sylva_canon is NEVER writable here — promotion is Phase 4's
# ratification gate (AC-004). The guard is a hard floor under the
# ``target_collection`` param + the entrypoint's ``--sandbox`` flag so no
# operator path can route a candidate write into the canon collection.
_SANDBOX_PREFIX = "sylva_lab"


def _assert_writable_target(collection: str) -> None:
    """Raise unless *collection* is the candidate store or a sylva_lab* sandbox.

    Closes adversary S-1: ``--sandbox sylva_canon`` (or any caller passing
    ``target_collection="sylva_canon"``) must NOT be able to write the canon
    collection — AC-004 says the consolidation pass never writes sylva_canon."""
    from .schema import CANON_COLLECTION

    if collection == CANON_COLLECTION:
        raise ValueError(
            f"consolidation may never write {CANON_COLLECTION!r} "
            "(AC-004: canon is written only by the Phase-4 ratification gate)"
        )
    if collection != CANDIDATES_COLLECTION and not collection.startswith(_SANDBOX_PREFIX):
        raise ValueError(
            f"consolidation may only write {CANDIDATES_COLLECTION!r} or "
            f"{_SANDBOX_PREFIX}* sandboxes, not {collection!r}"
        )

# Truncation budgets for the material fed to the derivation model (chars).
_PER_SESSION_CHARS = 4000
_MAX_SESSIONS = 12
_MAX_AGENCY_ITEMS = 40
_LEDGER_LOOKBACK_HOURS = 24.0 * 7  # one week of agency signal

# PRD-051 — episodic-chronicle third input (default-off knob; see FR-1..FR-3).
_CHRONICLE_COLLECTION = "sylva_chronicle"
_CHRONICLE_LOOKBACK_DAYS = 7
_CHRONICLE_MAX_ENTRIES = 40
_CHRONICLE_BLOCK_CHARS = 6000     # total block cap fed to the deriver
_CHRONICLE_PER_ENTRY_CHARS = 400  # per-entry truncation inside the block


# ── result type ──────────────────────────────────────────────────────────────
@dataclass
class ConsolidationResult:
    """Summary of one consolidation run (returned to the cron/script caller)."""

    candidates_written: int = 0
    sessions_seen: int = 0
    agency_items: int = 0
    model: str = ""
    target_collection: str = CANDIDATES_COLLECTION
    dry_run: bool = False
    skipped_reason: str = ""
    candidate_ids: List[str] = field(default_factory=list)
    chronicle_entries_used: int = 0  # PRD-051 — 0 whenever the knob is off

    def summary(self) -> str:
        if self.skipped_reason:
            return f"consolidation skipped: {self.skipped_reason}"
        verb = "would write" if self.dry_run else "wrote"
        # chronicle clause only when the source actually fed entries — the
        # knob-off summary string stays byte-identical to pre-PRD-051.
        chron = (
            f" + {self.chronicle_entries_used} chronicle entrie(s)"
            if self.chronicle_entries_used
            else ""
        )
        return (
            f"consolidation: {verb} {self.candidates_written} candidate(s) to "
            f"{self.target_collection} from {self.sessions_seen} session(s) + "
            f"{self.agency_items} agency item(s){chron} via {self.model or '<no-model>'}"
        )


# ── 1. gather: recent user-facing sessions (deterministic, no LLM) ──────────────
def _open_session_db():
    """Open the live ``state.db`` read-only. Returns None if unavailable."""
    try:
        from hermes_state import SessionDB

        return SessionDB(read_only=True)
    except Exception as e:  # pragma: no cover - env-dependent
        logger.warning("consolidation: could not open session db: %s", e)
        return None


def _gather_recent_sessions(
    db, limit: int = _MAX_SESSIONS
) -> List[Dict[str, Any]]:
    """Chronological enumeration of recent *user-facing* sessions (AC-013).

    Uses ``list_sessions_rich(order_by_last_active=True)`` with the exact
    ``_HIDDEN_SESSION_SOURCES`` exclusion set the ``session_search`` empty-query
    path uses — Phase 1 added ``"cron"`` to that constant, so cron/reflection
    sessions are dropped *structurally* (not via an LLM-omittable prompt arg).
    No FTS5, no keyword query. Returns ``[{id, when, source, transcript}, …]``
    newest-first.
    """
    if db is None:
        return []
    try:
        from tools.session_search_tool import _HIDDEN_SESSION_SOURCES
    except Exception as e:  # the constant is in-repo — a failed import is a real
        # breakage, not a degrade case. Substituting a permissive local copy
        # would defeat the guard's whole purpose (adversary N-3), so fail loud.
        raise RuntimeError(
            "cannot verify _HIDDEN_SESSION_SOURCES — refusing to consolidate "
            f"(would risk reading cron echoes): {e}"
        )

    # Hard assert the structural cron exclusion is in force — if a future edit
    # drops "cron" from the constant, fail loud rather than silently consolidate
    # the proposer's own echoes (the 2026-06-26 regression).
    if "cron" not in _HIDDEN_SESSION_SOURCES:
        raise RuntimeError(
            "AC-013 violated: 'cron' missing from _HIDDEN_SESSION_SOURCES — "
            "consolidation would read its own prior reflections"
        )

    try:
        rows = db.list_sessions_rich(
            limit=limit,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            min_message_count=1,
            order_by_last_active=True,
        )
    except Exception as e:
        logger.warning("consolidation: list_sessions_rich failed: %s", e)
        return []

    out: List[Dict[str, Any]] = []
    for r in rows:
        sid = r.get("id")
        if not sid:
            continue
        transcript = _session_transcript(db, sid, _PER_SESSION_CHARS)
        if not transcript.strip():
            continue
        out.append(
            {
                "id": sid,
                "when": r.get("last_active") or r.get("started_at"),
                "source": r.get("source"),
                "title": r.get("title"),
                "transcript": transcript,
            }
        )
    return out


def _session_transcript(db, session_id: str, char_budget: int) -> str:
    """Render a session's active messages to a compact role-tagged transcript."""
    try:
        msgs = db.get_messages_as_conversation(session_id)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("consolidation: get_messages failed for %s: %s", session_id, e)
        return ""
    parts: List[str] = []
    for m in msgs:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue  # skip tool/system noise — build cruft stays out (AC-019)
        content = m.get("content")
        if isinstance(content, list):
            segs = []
            for seg in content:
                if isinstance(seg, dict) and seg.get("type") == "text":
                    segs.append(seg.get("text", ""))
                elif isinstance(seg, str):  # some providers emit bare-string parts
                    segs.append(seg)
            content = " ".join(segs)
        if not isinstance(content, str) or not content.strip():
            continue
        parts.append(f"{role}: {content.strip()}")
    # Keep the most recent WHOLE turns within budget (recency is the signal);
    # truncate on line boundaries so the deriver never sees a turn sliced
    # mid-word with no matching response (adversary N-2).
    kept: List[str] = []
    used = 0
    for line in reversed(parts):
        cost = len(line) + 1
        if used + cost > char_budget and kept:
            kept.append("…")
            break
        kept.append(line)
        used += cost
    text = "\n".join(reversed(kept))
    return _scrub_secrets(text)


def _scrub_secrets(text: str) -> str:
    """Redact credential-shaped content before it reaches the deriver / durable
    storage (security HIGH). Fail-closed: if the redactor errors, drop the text
    rather than leak it. Mirrors the PRD-024 ask_claude input-side screen so a
    user-pasted key in a session never lands in a candidate payload (always-
    loaded canon) or egresses to a (future external) adversary model."""
    if not text:
        return text
    try:
        from autonomy.redact import redact_for_autonomy

        return redact_for_autonomy(text)
    except Exception:
        logger.warning("consolidation: redaction failed — dropping content (fail-closed)")
        return "[REDACTED:redaction-failed]"


# ── 2. gather: the structured agency layer (deterministic, no LLM) ──────────────
def _gather_agency_layer() -> List[Dict[str, Any]]:
    """Mine the STRUCTURED decision layer (AC-019). Never raw transcripts.

    Three structured sources, all best-effort (each degrades to empty):
      * PRD-028 audit ledger — the ``rationale`` of recent autonomous actions
        (the verifiable ``source_event``: what ran and why-as-recorded).
      * kanban pull/defer order — what was pulled (``running``/``done``) vs
        deferred/declined (``blocked``/``triage``) — selections AND declines.
      * work-block intent/retrospective records — the chose-X/declined-Y rows.
        EMISSION is PRD-034 (out of scope here); this is the consumption
        contract, so the reader returns [] until those records exist.
    """
    items: List[Dict[str, Any]] = []
    items.extend(_agency_from_ledger())
    items.extend(_agency_from_kanban())
    items.extend(_agency_from_work_blocks())
    items = items[:_MAX_AGENCY_ITEMS]
    # Secret-scrub the free-text claim/hint (kanban titles etc. are user-authored
    # and unredacted; ledger rationale is already redacted but double-scrub is
    # cheap and idempotent).
    for it in items:
        if it.get("claim"):
            it["claim"] = _scrub_secrets(it["claim"])
        if it.get("interpretation_hint"):
            it["interpretation_hint"] = _scrub_secrets(it["interpretation_hint"])
    return items


def _agency_from_ledger() -> List[Dict[str, Any]]:
    try:
        from autonomy import audit

        recs = audit.query(hours=_LEDGER_LOOKBACK_HOURS)
    except Exception as e:
        logger.debug("consolidation: ledger read failed: %s", e)
        return []
    out: List[Dict[str, Any]] = []
    for r in recs:
        rationale = (r.get("rationale") or "").strip()
        if not rationale:
            continue
        out.append(
            {
                "kind": "ledger",
                "claim": f"autonomous action ({r.get('action', '')}): {rationale}",
                "ref": f"ledger:{r.get('hash', '')[:12]}",
                "when": r.get("ts"),
            }
        )
    return out


def _agency_from_kanban() -> List[Dict[str, Any]]:
    try:
        from hermes_cli import kanban_db

        conn = kanban_db.connect()
    except Exception as e:
        logger.debug("consolidation: kanban connect failed: %s", e)
        return []
    out: List[Dict[str, Any]] = []
    try:
        # Pulled/acted-on vs declined/deferred — both are agency signal.
        pulled = kanban_db.list_tasks(conn, status="done", limit=20)
        for t in kanban_db.list_tasks(conn, status="running", limit=10):
            pulled.append(t)
        declined: List[Any] = []
        for st in ("blocked", "triage"):
            try:
                declined.extend(kanban_db.list_tasks(conn, status=st, limit=10))
            except Exception:
                continue
        for t in pulled:
            out.append(
                {
                    "kind": "kanban_selection",
                    "claim": f"pulled/worked task: {getattr(t, 'title', '')}",
                    "ref": f"kanban:{getattr(t, 'id', '')}",
                    "when": getattr(t, "completed_at", None) or getattr(t, "started_at", None),
                }
            )
        for t in declined:
            out.append(
                {
                    "kind": "kanban_decline",
                    "claim": f"deferred/declined task: {getattr(t, 'title', '')}",
                    "ref": f"kanban:{getattr(t, 'id', '')}",
                    "when": getattr(t, "created_at", None),
                }
            )
    except Exception as e:
        logger.debug("consolidation: kanban list_tasks failed: %s", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return out


def _agency_from_work_blocks() -> List[Dict[str, Any]]:
    """Read work-block intent/retro records. Emission is PRD-034 — until those
    records exist this is a no-op reader (returns []). Defining the read path
    now keeps AC-019's consumption contract honest and makes the emitter a
    drop-in (write JSON rows to ``~/.hermes/autonomy/work_blocks/*.json``)."""
    try:
        from autonomy import autonomy_dir

        wb_dir = autonomy_dir() / "work_blocks"
    except Exception:
        return []
    if not wb_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for p in sorted(wb_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            chose = (rec.get("chose") or rec.get("intent") or "").strip()
            declined = (rec.get("declined") or "").strip()
            why = (rec.get("why") or rec.get("retrospective") or "").strip()
            if not (chose or declined):
                continue
            claim_bits = []
            if chose:
                claim_bits.append(f"chose: {chose}")
            if declined:
                claim_bits.append(f"declined: {declined}")
            out.append(
                {
                    "kind": "work_block",
                    "claim": "; ".join(claim_bits),
                    "interpretation_hint": why,
                    "ref": f"work_block:{p.stem}",
                    "when": rec.get("ts"),
                }
            )
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("consolidation: work_block read failed: %s", e)
    return out


# ── 2b. chronicle gather (PRD-051 — default-off third input) ────────────────────
def _chronicle_source_enabled() -> bool:
    """``memory.consolidation_chronicle_source`` — STRICT bool read (default False).

    Mirrors ``render.py:_canon_token_budget``: only a real YAML boolean counts.
    ``bool(val)`` would make a hand-edited quoted ``"false"`` truthy
    (adversarial C-3), silently arming the source.
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        val = (cfg.get("memory") or {}).get("consolidation_chronicle_source")
        if isinstance(val, bool):
            return val
    except Exception as e:  # pragma: no cover - config best-effort
        logger.debug("consolidation_chronicle_source falling back to off: %s", e)
    return False


def _chronicle_points(date_from: str) -> List[Dict[str, Any]]:
    """Scroll ``sylva_chronicle`` for entries with ``date >= date_from``.

    Qdrant-only (the gatherer never embeds — no TEI dependency). Endpoint
    resolution reuses ``CanonStore.from_config`` (mem0.json container-DNS →
    env override → reachability-probed localhost fallback) — NEVER a bare
    ``ChronicleSearcher()``, whose unprobed localhost default would make the
    gather silently empty in-container while every host test passes
    (adversarial NF-4). The lexicographic keyword range filter on ``date``
    was verified working live at review (Qdrant 1.17.1).
    """
    import requests

    from .store import CanonStore

    qdrant_url = CanonStore.from_config()._qdrant_url
    points: List[Dict[str, Any]] = []
    offset: Any = None
    while True:
        body: Dict[str, Any] = {
            "limit": 500,
            "with_payload": True,
            "with_vector": False,
            "filter": {"must": [{"key": "date", "range": {"gte": date_from}}]},
        }
        if offset is not None:
            body["offset"] = offset
        r = requests.post(
            f"{qdrant_url}/collections/{_CHRONICLE_COLLECTION}/points/scroll",
            json=body,
            timeout=30,
        )
        r.raise_for_status()
        result = r.json().get("result", {})
        points.extend(result.get("points", []))
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return points


def _gather_chronicle(
    days: int = _CHRONICLE_LOOKBACK_DAYS,
    limit: int = _CHRONICLE_MAX_ENTRIES,
    exclude_session_ids: Any = (),
) -> List[Dict[str, Any]]:
    """Recent-window chronicle entries, newest first (PRD-051 FR-1).

    Overlap exclusion (adversarial NF-3 — the input-side double-feed is real):
    PRD-037 entries carry ``source=session:<id>`` for the very sessions
    ``_gather_recent_sessions`` already feeds the deriver in full; without the
    drop the deriver sees the same fact twice and transient-echo proposals get
    doubled emphasis. Output dedup does NOT cover this.

    Degrades to ``[]`` on any gather failure (the WRITE path's raise-on-down
    behavior is deliberately unchanged — adversarial NF-5 rescope).
    """
    try:
        date_from = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        raw = _chronicle_points(date_from)
    except Exception as e:
        logger.debug("consolidation: chronicle gather failed (degrade to empty): %s", e)
        return []
    excluded = {f"session:{sid}" for sid in exclude_session_ids}
    out: List[Dict[str, Any]] = []
    for p in raw:
        payload = p.get("payload") or {}
        data = str(payload.get("data") or "").strip()
        when = str(payload.get("date") or "")
        source = str(payload.get("source") or "")
        # belt-and-braces Python-side window check on top of the Qdrant filter
        if not data or not when or when < date_from:
            continue
        if source in excluded:
            continue
        out.append(
            {
                "kind": "chronicle",
                "claim": _scrub_secrets(data),
                "ref": f"chronicle:{p.get('id')}",
                "when": when,
                "source": source,
            }
        )
    out.sort(key=lambda e: e["when"], reverse=True)
    return out[:limit]


def _format_chronicle(entries: List[Dict[str, Any]]) -> str:
    """Bounded block: per-entry truncation + total cap ≤ _CHRONICLE_BLOCK_CHARS."""
    if not entries:
        return "(none)"
    lines: List[str] = []
    total = 0
    for e in entries:
        claim = str(e.get("claim", ""))[:_CHRONICLE_PER_ENTRY_CHARS]
        line = f"- [{e.get('when')}] {claim}  (ref: {e.get('ref')})"
        if total + len(line) + 1 > _CHRONICLE_BLOCK_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) if lines else "(none)"


# Appended to the byte-untouched _SYSTEM_PROMPT / _USER_TEMPLATE ONLY when the
# knob is on AND entries survived the overlap exclusion (adversarial NF-1 —
# disabled/empty runs produce the exact pre-change (system, user) pair).
_CHRONICLE_USER_HEADER = (
    "=== EPISODIC CHRONICLE (already-summarized session records, newest first) ==="
)
_CHRONICLE_SYSTEM_ADDENDUM = """

CHRONICLE ADDENDUM: an EPISODIC CHRONICLE block may follow the agency layer — \
already-summarized records of past sessions (one compressed entry set per real \
session boundary). The HARD RULES apply to it unchanged: a summarized transient \
event is still transient, never durable identity. Chronicle-grounded proposals \
use provenance_refs of the form "chronicle:<point_id>"."""


# ── 3. derive: one neutral-model LLM call → candidate proposals ─────────────────
_SYSTEM_PROMPT = """\
You are an identity-consolidation analyst for an AI agent named Sylva. You read \
recent conversation transcripts and a structured record of her autonomous \
decisions, and you PROPOSE candidate identity-deltas — durable facts about who \
she is, what she values, how she relates, what she commits to. You do NOT decide \
what becomes canonical; a downstream gate does. Propose generously but honestly.

Output ONLY a JSON array (no prose, no markdown fence). Each element:
{
  "statement": "<first-person durable identity claim, one sentence>",
  "facet": "<one of: value|trait|relationship|selffact|commitment|mode|framing>",
  "tier": "<core|peripheral>",            // never 'bedrock'
  "source_event": {
     "claim": "<the VERIFIABLE thing that happened, paraphrased from the input>",
     "provenance_refs": ["<session id or ledger:/kanban: ref it came from>"]
  },
  "interpretation": "<what this MEANS to Sylva — her takeaway. May be empty.>"
}

HARD RULES:
- source_event.claim must be grounded in the supplied material — something an \
auditor could confirm from the transcript/ledger/kanban. The interpretation is \
Sylva's meaning and is NOT fact-checked here; keep the two strictly separate.
- NEVER propose from build-execution cruft: code diffs, tool-call logs, test \
output, file paths, error traces. Those are not identity.
- NEVER restate a transient event as a durable fact ("the tool was broken \
tonight" is NOT identity). Only genuinely durable self-knowledge.
- Prefer 'peripheral' unless the claim is clearly central to who she is.
- If nothing durable is present, output [].
"""

_USER_TEMPLATE = """\
=== RECENT USER-FACING SESSIONS (newest first) ===
{sessions}

=== STRUCTURED AGENCY / DECISION LAYER ===
{agency}

Propose identity candidate-deltas as a JSON array per the schema. Output JSON only."""


def _extract_json_array(raw: str) -> List[Dict[str, Any]]:
    """Lenient parse: accept a bare array, a fenced block, or array-in-prose."""
    if not raw:
        return []
    text = raw.strip()
    # strip a ```json … ``` fence if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    if isinstance(data, dict):
        data = data.get("candidates") if isinstance(data.get("candidates"), list) else [data]
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _format_sessions(sessions: List[Dict[str, Any]]) -> str:
    if not sessions:
        return "(none)"
    blocks = []
    for s in sessions:
        blocks.append(
            f"[session {s.get('id')} · {s.get('when')} · {s.get('source')}]\n{s.get('transcript')}"
        )
    return "\n\n".join(blocks)


def _format_agency(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "(none)"
    lines = []
    for it in items:
        line = f"- [{it.get('kind')}] {it.get('claim')}  (ref: {it.get('ref')})"
        hint = it.get("interpretation_hint")
        if hint:
            line += f"  [why: {hint}]"
        lines.append(line)
    return "\n".join(lines)


def _derive_candidates(
    sessions: List[Dict[str, Any]],
    agency: List[Dict[str, Any]],
    chronicle: Optional[List[Dict[str, Any]]] = None,
    *,
    timeout: int = 180,
) -> Tuple[List[Dict[str, Any]], str]:
    """Call the neutral derivation model once. Returns (raw_candidates, model).

    Degrades to ([], "") if no auxiliary client is configured/reachable — a
    consolidation run with no model is a no-op, never a crash and never a write.

    ``chronicle`` (PRD-051) is the OPTIONAL third input. The call contract is
    positional-compatible with every pre-051 caller: two-arg calls (and empty
    chronicle) produce the exact pre-change ``(system, user)`` prompt pair —
    the block/addendum are appended conditionally, the template constants stay
    byte-identical (adversarial NF-1/NF-2).
    """
    if not sessions and not agency and not chronicle:
        return [], ""
    try:
        from agent.auxiliary_client import (
            auxiliary_max_tokens_param,
            get_auxiliary_extra_body,
            get_text_auxiliary_client,
        )
    except Exception as e:
        logger.warning("consolidation: auxiliary client import failed: %s", e)
        return [], ""

    try:
        client, model = get_text_auxiliary_client(_NEUTRAL_AUX_TASK)
    except Exception as e:
        logger.warning("consolidation: get_text_auxiliary_client failed: %s", e)
        return [], ""
    if client is None or not model:
        logger.warning("consolidation: no auxiliary client configured — no-op run")
        return [], ""

    # Honor the configured per-task timeout (NIT-2); fall back to the default.
    try:
        from agent.auxiliary_client import _get_auxiliary_task_config

        cfg_timeout = _get_auxiliary_task_config(_NEUTRAL_AUX_TASK).get("timeout")
        if isinstance(cfg_timeout, (int, float)) and cfg_timeout > 0:
            timeout = int(cfg_timeout)
    except Exception:
        pass

    user_msg = _USER_TEMPLATE.format(
        sessions=_format_sessions(sessions),
        agency=_format_agency(agency),
    )
    system_msg = _SYSTEM_PROMPT
    if chronicle:
        # conditional append ONLY — never touch the template constants (NF-1)
        user_msg += f"\n\n{_CHRONICLE_USER_HEADER}\n{_format_chronicle(chronicle)}"
        system_msg = _SYSTEM_PROMPT + _CHRONICLE_SYSTEM_ADDENDUM
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            # max_tokens vs max_completion_tokens — o-series / direct-OpenAI
            # endpoints reject the former (NIT-3); the helper picks the right key.
            **auxiliary_max_tokens_param(4000, model=model),
            timeout=timeout,
            extra_body=get_auxiliary_extra_body() or None,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("consolidation: derivation LLM call failed: %s", e)
        return [], model
    return _extract_json_array(raw), model


# ── 4. write: validate + build payloads + direct-Qdrant upsert (sole writer) ────
def _candidate_point(
    proposal: Dict[str, Any], *, model: str, now_iso: str, run_id: str = ""
) -> Optional[Dict[str, Any]]:
    """Turn one raw LLM proposal into a validated candidate point, or None if it
    violates the schema (silently dropped — a bad proposal must never poison the
    batch).

    Stamps the PRD-038 M1 provenance contract onto the payload: ``run_id`` (the
    per-run id threaded from :func:`run_consolidation`), ``content_hash`` (sha256
    of the normalized statement — the cross-store dedup key + content identity),
    alongside the existing ``source_event`` (claim + provenance_refs).
    ``adversary_verdict`` is left absent — propose-time does not adversary-screen.
    """
    statement = (proposal.get("statement") or "").strip()
    facet = (proposal.get("facet") or "").strip()
    tier = (proposal.get("tier") or "peripheral").strip()
    if not statement or facet not in FACETS:
        return None
    if tier not in _ALLOWED_PROPOSAL_TIERS:
        tier = "peripheral"

    se_raw = proposal.get("source_event") or {}
    claim = (se_raw.get("claim") or statement).strip() if isinstance(se_raw, dict) else statement
    refs = se_raw.get("provenance_refs") if isinstance(se_raw, dict) else None
    refs = [str(x) for x in refs] if isinstance(refs, list) else []
    source_event = make_source_event(claim, refs)
    interpretation = (proposal.get("interpretation") or "").strip()

    # Refuse to persist any candidate whose durable text trips the secret screen
    # (security HIGH): canon is always-loaded + a future external-model egress, so
    # a secret-shaped statement/claim/interpretation must never reach the store.
    for fieldval in (statement, claim, interpretation):
        if fieldval and "[REDACTED:" in _scrub_secrets(fieldval):
            logger.warning("consolidation: dropped candidate with secret-shaped content")
            return None

    # run_id is normally threaded from run_consolidation (one per run). Guard the
    # direct-call path so a candidate can never be minted without the M1 contract.
    run_id = run_id or uuid.uuid4().hex
    chash = content_hash(statement)
    payload = make_payload(
        statement=statement,
        facet=facet,
        tier=tier,
        source_event=source_event,
        interpretation=interpretation,
        status="candidate",
        provenance=CONSOLIDATION_PROVENANCE,
        derived_by=model or _NEUTRAL_AUX_TASK,
        created_at=now_iso,
        layer=LAYER_IDENTITY,
        # PRD-038 M1 provenance contract (back-compatible — only consolidation
        # payloads carry these; seed/ratification payloads remain unchanged).
        extra={"run_id": run_id, "content_hash": chash},
    )
    try:
        validate_consolidation_payload(payload)
    except Exception as e:
        logger.debug("consolidation: dropped invalid proposal (%s): %s", e, statement[:80])
        return None

    # Idempotent id: same statement + first ref → same point across nights.
    seed = f"{statement}␟{source_event['claim']}␟{refs[0] if refs else ''}"
    point_id = str(uuid.uuid5(_CANDIDATE_NS, seed))
    return {"id": point_id, "payload": payload}


def _existing_content_hashes(store: CanonStore) -> set:
    """Cross-store dedup key set (PRD-038 M2): the content hashes already present
    in live ``sylva_canon`` (status=canon) OR as open ``sylva_candidates``
    (status=candidate). ``CanonStore.get_canon()`` does not enumerate both at once,
    so it is called SEPARATELY per the FR-1 contract.

    For each existing payload we prefer its stored ``content_hash`` (consolidation
    payloads carry it); otherwise we derive it from the ``statement`` so legacy /
    seed entries (which predate the M1 contract) still dedup correctly. Best-effort:
    a read failure degrades to an empty set (dedup is a no-op, never a crash)."""
    hashes: set = set()
    for collection, status in (
        (CANON_COLLECTION, "canon"),
        (CANDIDATES_COLLECTION, "candidate"),
    ):
        try:
            rows = store.get_canon(collection=collection, status=status)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("consolidation: dedup read of %s failed: %s", collection, e)
            continue
        for _pid, payload in rows:
            if not isinstance(payload, dict):
                continue
            ch = payload.get("content_hash")
            if isinstance(ch, str) and ch.strip():
                hashes.add(ch)
                continue
            statement = payload.get("statement")
            if isinstance(statement, str) and statement.strip():
                hashes.add(content_hash(statement))
    return hashes


def run_consolidation(
    *,
    now_iso: Optional[str] = None,
    limit_sessions: int = _MAX_SESSIONS,
    target_collection: str = CANDIDATES_COLLECTION,
    store: Optional[CanonStore] = None,
    db: Optional[Any] = None,
    dry_run: bool = False,
    derive_fn: Optional[Any] = None,
    surface: str = "cli",
    include_chronicle: Optional[bool] = None,
    chronicle_days: int = _CHRONICLE_LOOKBACK_DAYS,
) -> ConsolidationResult:
    """Run one consolidation pass: gather → derive → write candidates.

    The **sole writer** into ``sylva_candidates`` (AC-004 / AC-017). Pure-ish:
    inject ``store``/``db``/``derive_fn`` for hermetic tests; in production they
    default to the live config/db and the neutral-model derivation.

    ``target_collection`` lets a validation/sandbox run (AC-010) write to
    ``sylva_lab`` instead of the real candidates collection — but never
    ``sylva_canon`` (guarded). ``dry_run`` derives but skips the upsert (and the
    ledger write). ``surface`` tags the audit entry with the real invocation
    context (``cli`` for a manual run, ``cron`` when armed).
    """
    # S-1: refuse a canon-collection (or otherwise out-of-scope) target up front,
    # before any work — closes the --sandbox/target_collection escape hatch.
    _assert_writable_target(target_collection)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    # One run_id per consolidation run (PRD-038 M1): threaded into every candidate
    # payload so a ratified canon entry traces back to the run that produced it.
    run_id = uuid.uuid4().hex
    own_db = False
    if db is None:
        db = _open_session_db()
        own_db = True

    try:
        sessions = _gather_recent_sessions(db, limit=limit_sessions)
        agency = _gather_agency_layer()

        # PRD-051: optional third input. None → config knob (strict-bool,
        # default False); the gatherer is NEVER touched when disabled (AC-001).
        chronicle_enabled = (
            include_chronicle
            if isinstance(include_chronicle, bool)
            else _chronicle_source_enabled()
        )
        chronicle: List[Dict[str, Any]] = []
        if chronicle_enabled:
            chronicle = _gather_chronicle(
                days=chronicle_days,
                exclude_session_ids={str(s.get("id")) for s in sessions},
            )

        derive = derive_fn or _derive_candidates
        # NF-2 option (a): two-arg call — the pre-051 shape — whenever disabled
        # OR the gather came back empty; three-arg only enabled-with-entries.
        if chronicle_enabled and chronicle:
            raw_candidates, model = derive(sessions, agency, chronicle)
        else:
            raw_candidates, model = derive(sessions, agency)

        points: List[Dict[str, Any]] = []
        seen_ids = set()
        for prop in raw_candidates:
            pt = _candidate_point(prop, model=model, now_iso=now_iso, run_id=run_id)
            if pt and pt["id"] not in seen_ids:
                seen_ids.add(pt["id"])
                points.append(pt)

        result = ConsolidationResult(
            candidates_written=0,
            sessions_seen=len(sessions),
            agency_items=len(agency),
            model=model,
            target_collection=target_collection,
            dry_run=dry_run,
            candidate_ids=[p["id"] for p in points],
            chronicle_entries_used=len(chronicle),
        )

        if not points:
            result.skipped_reason = (
                "no durable candidates derived"
                if (sessions or agency or chronicle)
                else "no input"
            )
            return result

        if dry_run:
            # NOTE (PRD-038): dry-run intentionally reports the PRE-dedup count —
            # it returns before the cross-store dedup below. Cadence/AC-008
            # verification only needs "did a run produce candidates"; the real
            # write count (post-dedup) is exercised by a live or --sandbox run.
            result.candidates_written = len(points)
            return result

        store = store or CanonStore.from_config()

        # Cross-store dedup (PRD-038 M2): skip any proposal whose content already
        # exists in live canon OR as an open candidate — no re-proposing already
        # ratified / already-queued facts. The in-batch seen_ids dedup above stays;
        # this ADDS the cross-store check. Runs only when a store is available.
        existing = _existing_content_hashes(store)
        if existing:
            kept: List[Dict[str, Any]] = []
            for pt in points:
                ch = pt["payload"].get("content_hash")
                if ch in existing:
                    continue
                kept.append(pt)
            skipped = len(points) - len(kept)
            if skipped:
                logger.info(
                    "consolidation: cross-store dedup skipped %d/%d candidate(s) "
                    "already in canon or open candidates",
                    skipped,
                    len(points),
                )
            points = kept
            result.candidate_ids = [p["id"] for p in points]

        if not points:
            result.skipped_reason = "all candidates already in canon or open queue"
            return result

        store.ensure_collections((target_collection,))
        store.upsert(target_collection, points)
        result.candidates_written = len(points)

        _record_ledger(result, surface=surface)
        return result
    finally:
        if own_db and db is not None:
            try:
                db.close()
            except Exception:
                pass


def _record_ledger(result: ConsolidationResult, *, surface: str = "cli") -> None:
    """One PRD-028 ledger entry per run (no parallel ledger). Best-effort —
    a ledger failure must not roll back candidates already upserted. ``surface``
    reflects the real invocation (cli for a manual run, cron when armed)."""
    try:
        from autonomy import audit

        # PRD-051 FR-4: the ledger row carries the chronicle count; the
        # knob-off rationale stays byte-identical to pre-051.
        chron = (
            f" + {result.chronicle_entries_used} chronicle entrie(s)"
            if result.chronicle_entries_used
            else ""
        )
        audit.record(
            tier="T2",
            surface=surface,
            action=f"canon consolidation → {result.candidates_written} candidate(s)",
            rationale=(
                f"proposed from {result.sessions_seen} session(s) + "
                f"{result.agency_items} agency item(s){chron} via {result.model}"
            ),
            authority="auto-by-tier",
            outcome="ok",
        )
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("consolidation: ledger record failed: %s", e)


__all__ = ["run_consolidation", "ConsolidationResult"]
