"""Canon schema — the typed shape of identity entries (PRD-029 Phase 2).

This is the load-bearing contract Phases 3 (consolidation), 4 (ratification /
adversary), and 5 (migration / seeding) build on. It defines:

  * the **content** that renders into the self-brief (`statement`, `facet`, …),
  * the **structural fact/meaning split** (`source_event` vs `interpretation`)
    that the adversary boundary (AC-006) is enforced by — the adversary verdict
    schema may reference `source_event` only; there is no field it can cite to
    refute `interpretation`,
  * the **tier** (prompt-budget knob + ratification bar),
  * **provenance** (receipts) and **lifecycle/governance** (status, versioning,
    supersession, ratification stamps, adversary verdict).

Nothing here talks to Qdrant — `store.py` owns I/O. Keeping the schema pure makes
it cheap to validate and impossible to couple to a backend.

Invariants (PRD-029):
  * Bedrock identity lives in SOUL.md, NOT as a ``tier: bedrock`` row — the
    collection holds ``core``/``peripheral`` only (AC-001/AC-009). ``bedrock``
    stays in the enum for completeness and tier-rank ordering, but a populated
    ``sylva_canon`` should never contain a bedrock row.
  * Evolution is supersession, never overwrite (``supersedes``/``superseded_by``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── Collections ────────────────────────────────────────────────────────────────
CANON_COLLECTION = "sylva_canon"
CANDIDATES_COLLECTION = "sylva_candidates"

# bge-m3 via TEI — must match sylva_chronicle / sylva_memories for vector parity.
VECTOR_DIM = 1024
VECTOR_DISTANCE = "Cosine"

# The render layer tag — identity entries only. (Leaves room for other layers
# to share the store later without colliding in the brief query.)
LAYER_IDENTITY = "identity"

# ── Enumerations ────────────────────────────────────────────────────────────────
# `framing` = ratified connective prose for the self-brief (so the brief is 100%
# ratified content, no generative step at render time).
FACETS = (
    "value",
    "trait",
    "relationship",
    "selffact",
    "commitment",
    "mode",
    "framing",
)

# Triple duty: prompt-budget knob, adversary escalation dial, ratification bar.
TIERS = ("bedrock", "core", "peripheral")

# Lower rank renders first. bedrock(0) is SOUL.md-sourced and prepended ahead of
# any canon row; core(1) fills the budget; peripheral(2) overflows.
TIER_RANK: Dict[str, int] = {"bedrock": 0, "core": 1, "peripheral": 2}

STATUSES = (
    "candidate",
    "canon",
    "tension",
    "rejected",
    "superseded",
    "retired",
)

# Sentinel for entries lacking an explicit render_order (NF-4): they sort AFTER
# every ordered entry, ties broken by stable_id — the sort stays total either way.
RENDER_ORDER_SENTINEL = 1 << 30


class CanonSchemaError(ValueError):
    """Raised when a canon payload violates the schema contract."""


def make_source_event(claim: str, provenance_refs: Optional[List[str]] = None) -> Dict[str, Any]:
    """The verifiable half — *what happened*. The ONLY field the adversary's
    provenance/verifiability checks may touch (AC-006)."""
    return {"claim": claim, "provenance_refs": list(provenance_refs or [])}


def make_payload(
    *,
    statement: str,
    facet: str,
    tier: str,
    source_event: Dict[str, Any],
    interpretation: str = "",
    status: str = "candidate",
    render_order: Optional[int] = None,
    legacy_source: Optional[str] = None,
    chosen_quote: Optional[str] = None,
    provenance: Any = "legacy_import",
    derived_by: str = "",
    ratified_by: Optional[Dict[str, Any]] = None,
    created_at: str = "",
    ratified_at: str = "",
    last_affirmed_at: str = "",
    version: int = 1,
    supersedes: Optional[str] = None,
    superseded_by: Optional[str] = None,
    adversary_verdict: Optional[Dict[str, Any]] = None,
    tension_with: Optional[List[str]] = None,
    layer: str = LAYER_IDENTITY,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a fully-formed canon/candidate payload dict.

    Sensible defaults so Phase 3/5 callers fill only what they have; Phase 2
    only *reads* ``statement``/``facet``/``tier``/``status``/``layer``/
    ``render_order`` for the render, but the full shape is materialised now so
    later phases write against a stable contract. Call :func:`validate_payload`
    on the result before upserting.
    """
    payload: Dict[str, Any] = {
        # content
        "statement": statement,
        "facet": facet,
        "legacy_source": legacy_source,
        "chosen_quote": chosen_quote,
        # fact / meaning split (F-03)
        "source_event": source_event,
        "interpretation": interpretation,
        # tier
        "tier": tier,
        # provenance
        "provenance": provenance,
        "derived_by": derived_by,
        "ratified_by": ratified_by if ratified_by is not None else {},
        # lifecycle / governance
        "status": status,
        "version": version,
        "supersedes": supersedes,
        "superseded_by": superseded_by,
        "created_at": created_at,
        "ratified_at": ratified_at,
        "last_affirmed_at": last_affirmed_at,
        "adversary_verdict": adversary_verdict,
        "tension_with": list(tension_with or []),
        # render
        "render_order": render_order,
        "layer": layer,
    }
    if extra:
        payload.update(extra)
    return payload


def validate_payload(payload: Dict[str, Any]) -> None:
    """Raise :class:`CanonSchemaError` if *payload* violates the contract.

    Cheap structural validation — enum membership, the fact/meaning split shape,
    and the bedrock-not-in-collection invariant. Not a full type-check.
    """
    if not isinstance(payload, dict):
        raise CanonSchemaError("payload must be a dict")

    statement = payload.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        raise CanonSchemaError("statement must be a non-empty string")

    facet = payload.get("facet")
    if facet not in FACETS:
        raise CanonSchemaError(f"facet {facet!r} not in {FACETS}")

    tier = payload.get("tier")
    if tier not in TIERS:
        raise CanonSchemaError(f"tier {tier!r} not in {TIERS}")
    if tier == "bedrock":
        # AC-001/AC-009: bedrock lives in SOUL.md, never as a collection row.
        raise CanonSchemaError(
            "bedrock entries must live in SOUL.md, not the canon collection"
        )

    status = payload.get("status")
    if status not in STATUSES:
        raise CanonSchemaError(f"status {status!r} not in {STATUSES}")

    se = payload.get("source_event")
    if not isinstance(se, dict) or "claim" not in se or "provenance_refs" not in se:
        raise CanonSchemaError(
            "source_event must be {claim, provenance_refs[]}"
        )
    if not isinstance(se.get("provenance_refs"), list):
        raise CanonSchemaError("source_event.provenance_refs must be a list")

    if not isinstance(payload.get("interpretation", ""), str):
        raise CanonSchemaError("interpretation must be a string")

    ro = payload.get("render_order")
    if ro is not None and not isinstance(ro, int):
        raise CanonSchemaError("render_order must be an int or None")


def sort_key(point_id: str, payload: Dict[str, Any]) -> tuple:
    """Total-order key for deterministic render (AC-003 / F-02 / NF-4).

    ``(tier_rank, render_order, stable_id)`` — Python-sorted BEFORE budget
    truncation, never Qdrant ``order_by`` (which 400s on these non-indexed
    fields). Missing ``render_order`` falls back to a sentinel so the entry
    sorts after ordered ones; the stable Qdrant point id breaks every remaining
    tie, so the key is total and the brief is byte-identical across restarts.
    """
    tier = payload.get("tier")
    tier_rank = TIER_RANK.get(tier, len(TIERS))  # unknown tiers sort last
    ro = payload.get("render_order")
    ro = ro if isinstance(ro, int) else RENDER_ORDER_SENTINEL
    return (tier_rank, ro, str(point_id))
