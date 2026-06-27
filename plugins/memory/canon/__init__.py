"""Canon — Sylva's identity store (PRD-029).

The four-layer memory architecture's identity layer: ratified canon that is
**always loaded into the prompt as a deterministic first-person self-brief and
never semantically retrieved at runtime**. Bedrock lives in SOUL.md;
core/peripheral live in the ``sylva_canon`` Qdrant collection; the governed
consolidation pass proposes into ``sylva_candidates`` (Phase 3+).

Phase 2 surface: typed schema, the direct-Qdrant store, and the deterministic
``render_self_brief`` assembler wired into the SOUL.md prompt slot.
"""

from __future__ import annotations

from .render import assemble_brief, render_self_brief
from .schema import (
    CANDIDATES_COLLECTION,
    CANON_COLLECTION,
    FACETS,
    LAYER_IDENTITY,
    STATUSES,
    TIER_RANK,
    TIERS,
    CanonSchemaError,
    make_payload,
    make_source_event,
    sort_key,
    validate_payload,
)
from .store import CanonStore

__all__ = [
    "render_self_brief",
    "assemble_brief",
    "CanonStore",
    "CANON_COLLECTION",
    "CANDIDATES_COLLECTION",
    "LAYER_IDENTITY",
    "FACETS",
    "TIERS",
    "TIER_RANK",
    "STATUSES",
    "CanonSchemaError",
    "make_payload",
    "make_source_event",
    "validate_payload",
    "sort_key",
]
