"""Deterministic self-brief render (PRD-029 Phase 2, AC-002/003).

The self-brief is **assembled, never generated** — a pure function of
(SOUL.md bytes, ratified canon). No LLM/generative step sits between canon and
the prompt (Constraint: "No generative step at render time").

Determinism mechanism (F-02 / AC-003):
  1. an *unordered* filtered scroll of ``layer:identity,status:canon`` (Qdrant
     ``order_by`` 400s on the non-range-indexed payload fields),
  2. a Python sort on the total-order key ``(tier_rank, render_order, stable_id)``
     (see :func:`schema.sort_key`),
  3. greedy truncation at ``canon_token_budget`` AFTER the sort — so the cap is
     not itself a selection lottery.

Same canon in → byte-identical brief out, across process restarts.

Identity is **never** semantically retrieved at runtime: this module talks to
Qdrant only via a filtered scroll (no vector, no ``search``). With ``sylva_canon``
empty (Phases 2–4, pre-seeding) the brief is SOUL.md-only — zero runtime change.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .schema import LAYER_IDENTITY, sort_key
from .store import CanonStore

logger = logging.getLogger(__name__)

# Codebase convention (hermes_cli/config.py memory char-limit comments): ~2.75
# chars per token. The budget is in tokens; we bound the canon-assembled block by
# the equivalent char count. A deterministic estimate — no tokenizer, so the
# render stays byte-stable and model-independent.
_CHARS_PER_TOKEN = 2.75

_DEFAULT_CANON_TOKEN_BUDGET = 4096

# SOUL.md (bedrock) and the assembled canon block are separated by a blank line.
# `framing`-facet canon entries supply any connective prose; nothing synthetic is
# inserted, keeping the brief 100% ratified content.
_BLOCK_SEPARATOR = "\n\n"


def _canon_token_budget() -> int:
    """Read ``memory.canon_token_budget`` (default 4096) from live config."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        val = (cfg.get("memory") or {}).get("canon_token_budget")
        if isinstance(val, int) and val > 0:
            return val
    except Exception as e:  # pragma: no cover - config best-effort
        logger.debug("canon_token_budget falling back to default: %s", e)
    return _DEFAULT_CANON_TOKEN_BUDGET


def _read_soul_md() -> Optional[str]:
    """Raw SOUL.md bedrock text, stripped — or None if absent/empty.

    Returns raw bytes; the security scan + context-length truncation are applied
    by the caller (``load_soul_md``), so this stays a pure file read.
    """
    try:
        from hermes_constants import get_hermes_home

        soul_path = get_hermes_home() / "SOUL.md"
        if not soul_path.exists():
            return None
        content = soul_path.read_text(encoding="utf-8").strip()
        return content or None
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("could not read SOUL.md: %s", e)
        return None


def assemble_brief(
    soul_text: Optional[str],
    entries: List[Tuple[str, Dict[str, Any]]],
    canon_token_budget: int = _DEFAULT_CANON_TOKEN_BUDGET,
) -> Optional[str]:
    """Pure assembler — the AC-003 deterministic core.

    Given fixed inputs (SOUL.md bedrock text, the canon ``(id, payload)`` list,
    and the budget) returns a byte-identical brief. Sorts by the total-order key,
    then greedily fills the budget in sorted order (bedrock is SOUL.md, so canon
    rows are core→peripheral by tier_rank; peripheral overflows as designed).
    Returns None only when there is no SOUL.md AND no canon content.
    """
    # 1. total-order sort BEFORE truncation
    ordered = sorted(entries, key=lambda e: sort_key(e[0], e[1]))

    # 2. greedy budget fill (canon block only; bedrock/SOUL.md is separate)
    char_budget = int(canon_token_budget * _CHARS_PER_TOKEN)
    selected: List[str] = []
    used = 0
    for _pid, payload in ordered:
        statement = (payload.get("statement") or "").strip()
        if not statement:
            continue
        # +1 for the joining newline between statements
        cost = len(statement) + (1 if selected else 0)
        if used + cost > char_budget:
            break
        selected.append(statement)
        used += cost

    canon_block = "\n".join(selected)

    # 3. combine bedrock + canon
    parts: List[str] = []
    if soul_text:
        parts.append(soul_text)
    if canon_block:
        parts.append(canon_block)
    if not parts:
        return None
    return _BLOCK_SEPARATOR.join(parts)


def render_self_brief() -> Optional[str]:
    """Assemble the full self-brief (SOUL.md bedrock + ratified canon).

    Zero-arg entry point used by ``load_soul_md`` and the AC-003 verification.
    Resolves SOUL.md, the canon store (live config URLs), and the token budget
    itself, so it is deterministic across process restarts. Any canon-store
    failure degrades to SOUL.md-only — never raises.
    """
    soul_text = _read_soul_md()
    entries: List[Tuple[str, Dict[str, Any]]] = []
    try:
        store = CanonStore.from_config()
        entries = store.get_canon(layer=LAYER_IDENTITY, status="canon")
    except Exception as e:
        logger.debug("canon read failed; SOUL.md-only brief: %s", e)
        entries = []
    return assemble_brief(soul_text, entries, _canon_token_budget())
