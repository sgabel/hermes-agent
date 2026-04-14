"""Memory orchestration components for the mem0 plugin.

Intent gating, query mode routing, context budgeting, and deduplication.
All heuristic — no LLM calls.
"""

from __future__ import annotations

import re
import string
from typing import List, Optional


# ---------------------------------------------------------------------------
# Intent Gate
# ---------------------------------------------------------------------------

class IntentGate:
    """Decide whether memory retrieval is warranted for a given message."""

    SKIP_PATTERNS = [
        # Greetings / social
        r"^(hi|hey|hello|morning|evening|thanks|thank you|ok|okay|sure|yep"
        r"|yes|no|nah|lol|haha|nice|cool|great|good|got it|sounds good)\s*[.!]?$",
        # Action confirmations
        r"^(go ahead|do it|proceed|looks good|perfect|approved?)\s*[.!]?$",
    ]

    _compiled = [re.compile(p, re.IGNORECASE) for p in SKIP_PATTERNS]

    @classmethod
    def should_retrieve(cls, message: str, had_tool_calls: bool = False) -> bool:
        """Return True if retrieval is warranted."""
        text = message.strip().lower()
        if not text:
            return False
        # Never skip when the previous turn had active tool use —
        # terse follow-ups like "continue" or "that one" are context-dependent.
        if had_tool_calls:
            return True
        for pattern in cls._compiled:
            if pattern.match(text):
                return False
        return True


# ---------------------------------------------------------------------------
# Query Mode Router
# ---------------------------------------------------------------------------

class QueryModeRouter:
    """Classify a user message into a memory search mode."""

    HISTORICAL_SIGNALS = [
        re.compile(
            r"\b(remember|recall|earlier|before|last time|previously|history"
            r"|back when|used to|chronicle|conversation from)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b\d+\s+(days?|weeks?|months?)\s+ago\b", re.IGNORECASE),
        re.compile(
            r"\blast\s+(week|month|year|session|conversation|time we)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bin\s+(january|february|march|april|may|june|july|august"
            r"|september|october|november|december)\b",
            re.IGNORECASE,
        ),
    ]

    @classmethod
    def classify(cls, message: str) -> str:
        """Return 'historical_memory' or 'stable_knowledge'."""
        text = message.lower()
        for signal in cls.HISTORICAL_SIGNALS:
            if signal.search(text):
                return "historical_memory"
        return "stable_knowledge"


# ---------------------------------------------------------------------------
# Context Budget
# ---------------------------------------------------------------------------

class ContextBudget:
    """Assemble memory results within a character budget."""

    MAX_CHARS = 8000

    @classmethod
    def assemble(
        cls,
        *,
        facts: Optional[List[str]] = None,
        chronicle: Optional[List[str]] = None,
    ) -> str:
        """Greedy fill-down: facts first, then chronicle."""
        budget = cls.MAX_CHARS
        parts: list[str] = []

        if facts:
            block = "\n".join(f"- {f}" for f in facts)
            if len(block) > budget:
                block = block[:budget]
            parts.append(block)
            budget -= len(block)

        if chronicle and budget > 200:
            header = "### Chronicle\n"
            budget -= len(header)
            block = "\n".join(f"- {c}" for c in chronicle)
            if len(block) > budget:
                block = block[:budget]
            parts.append(header + block)

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

class Deduplicator:
    """Remove memory candidates that overlap heavily with recent context."""

    THRESHOLD = 0.65  # Jaccard word overlap

    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)

    @classmethod
    def _normalize(cls, text: str) -> str:
        return text.lower().translate(cls._PUNCT_TABLE)

    @classmethod
    def _jaccard(cls, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0

    @classmethod
    def deduplicate(cls, candidates: List[str], context: str) -> List[str]:
        """Filter candidates that overlap too heavily with context."""
        context_words = set(cls._normalize(context).split())
        if not context_words:
            return candidates
        return [
            c for c in candidates
            if cls._jaccard(set(cls._normalize(c).split()), context_words) < cls.THRESHOLD
        ]
