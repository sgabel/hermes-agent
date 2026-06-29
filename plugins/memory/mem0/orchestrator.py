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
# Recall-Assist Router (PRD-041 FR-1)
# ---------------------------------------------------------------------------

class RecallAssistRouter:
    """Precision-first classifier for the bounded historical-recall assist.

    PRD-041 D2: the broad ``QueryModeRouter.historical_memory`` signal matches
    bare ``history|before|earlier|previously|in <month>`` — it false-positives on
    "git **history**", "**before** you edit", "**remember to** run the formatter",
    and "**in June**". That is acceptable for the on-demand ``chronicle_search``
    routing it was built for, but too loose for a *default-ON ambient injector*
    (the noise PRD-029 removed for cause). This router requires a **recall cue tied
    to the shared past** — a recall verb plus a relational pronoun (you / we / our)
    or an explicit chronicle reference — and deliberately does NOT fire on the bare
    temporal/sequencing words. When unsure, it returns False (no injection).
    """

    # Strong cues — an explicit appeal to the shared past; each fires on its own.
    STRONG_SIGNALS = [
        # "do you remember", "did you recall" — but NOT "do you remember TO …"
        # (a task reminder, not a recall question — review NEEDS-FIX 1).
        re.compile(
            r"\b(do|did|don'?t)\s+you\s+(remember|recall)\b(?!\s+to\b)",
            re.IGNORECASE,
        ),
        # "remember when/that time/how/what/we/our…" — structurally excludes
        # "remember to" (the verb is followed by a recall complement, not "to").
        re.compile(
            r"\bremember\s+(when|that\s+time|the\s+time|how|what|why|where|who"
            r"|us|we|our|you\s+and\s+i)\b",
            re.IGNORECASE,
        ),
        # "have we / did we / when did we … <past activity>"
        re.compile(
            r"\b(have|did|when\s+did|where\s+did|why\s+did|how\s+did)\s+we"
            r"\s+(ever\s+)?(talk|talked|discuss|discussed|spoke|speak|meet|met"
            r"|do|done|try|tried|decide|decided|agree|agreed|cover|covered|go\s+over)\b",
            re.IGNORECASE,
        ),
        # "last time we/you/i", "the first/last time"
        re.compile(r"\blast\s+time\s+(we|you|i)\b", re.IGNORECASE),
        re.compile(r"\bthe\s+(first|last)\s+time\b", re.IGNORECASE),
        # "we talked/discussed/agreed about", "we were talking about"
        re.compile(
            r"\bwe\s+(talked|spoke|discussed|chatted|agreed|decided)\s+about\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bwe\s+were\s+(talking|discussing)\s+about\b", re.IGNORECASE),
        # "our/your/that last/previous/earlier/first chat/conversation/session"
        re.compile(
            r"\b(our|your|that)\s+(last|previous|earlier|first|recent)\s+"
            r"(chat|conversation|session|talk|call|discussion)\b",
            re.IGNORECASE,
        ),
        # explicit chronicle reference
        re.compile(r"\b(the|your|my|our)\s+chronicle\b", re.IGNORECASE),
        # "back when we/you/i"
        re.compile(r"\bback\s+when\s+(we|you|i)\b", re.IGNORECASE),
        # "remind me what/when/how we decided …"
        re.compile(
            r"\bremind\s+me\s+(what|when|how|why|who|where)\s+(we|you|i)\b",
            re.IGNORECASE,
        ),
    ]

    # Weak cue — a second-person statement ("you said/told me/…"). Ambiguous
    # between THIS conversation (code review: "you said to use a dict", "you
    # wrote this wrong") and the shared past, so it fires ONLY when paired with a
    # temporal/past anchor (review NEEDS-FIX 2). Precision over recall: an
    # unanchored "you said X" mid-session must NOT inject.
    WEAK_YOU_VERB = re.compile(
        r"\byou\s+(said|told\s+me|mentioned|asked|called|guessed|wrote"
        r"|promised|explained)\b",
        re.IGNORECASE,
    )
    TEMPORAL_ANCHOR = re.compile(
        r"\b(earlier|before|previously|last\s+time|a\s+while\s+ago"
        r"|the\s+other\s+(day|time)|back\s+then|once|yesterday"
        r"|last\s+(week|month|year)|\d+\s+(days?|weeks?|months?)\s+ago|when\s+we)\b",
        re.IGNORECASE,
    )

    @classmethod
    def is_recall_query(cls, message: str) -> bool:
        """True only for an explicit recall question about the shared past.

        Strong cues fire alone; the ambiguous "you <verb>" cue fires only with a
        temporal anchor. When neither holds, return False (do NOT inject).
        """
        if not message:
            return False
        text = message.lower()
        for signal in cls.STRONG_SIGNALS:
            if signal.search(text):
                return True
        if cls.WEAK_YOU_VERB.search(text) and cls.TEMPORAL_ANCHOR.search(text):
            return True
        return False


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
        max_chars: Optional[int] = None,
    ) -> str:
        """Greedy fill-down: facts first, then chronicle.

        ``max_chars`` overrides the default budget — PRD-041 FR-1 passes a small
        cap for the bounded recall assist (the legacy 8000 is for the retired
        facts+chronicle path, far too large for a top-2 same-turn injection).
        """
        budget = max_chars if max_chars is not None else cls.MAX_CHARS
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
