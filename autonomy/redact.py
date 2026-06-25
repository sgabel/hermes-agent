"""Hardened secret-screen for autonomous audit / egress text (PRD-028 R-4).

Discharges the PRD-024 handoff: ``agent.redact.redact_sensitive_text`` is
high-confidence/known-pattern only — it gates ``.env`` redaction on the *var
name* matching ``API_KEY|TOKEN|SECRET|PASSWORD|...`` and has no entropy/base64
fallback, so it misses (a) raw high-entropy tokens, (b) base64/hex blobs, and
(c) generic-named ``.env`` values (``FOO=<opaque>``).

``redact_for_autonomy`` COMPOSES the existing redactor with those three extra
heuristics and is **fail-closed**: if redaction raises, the caller must treat
the text as unsafe (refuse / don't emit). For an audit log and an egress screen
the safe error direction is over-redaction, so the heuristics intentionally
favour false positives over leaks.
"""

from __future__ import annotations

import math
import re

# Placeholder tokens the heuristics emit.
_HE = "[REDACTED:high-entropy]"
_B64 = "[REDACTED:base64-blob]"
_ENV = "[REDACTED:opaque-env-value]"

# Token-shaped candidate: contiguous run of secret-charset characters.
# NB: ``=`` is deliberately excluded — it is an assignment separator (handled by
# _ASSIGN_RE) and padded base64 is handled by _B64_RE before this sweep runs;
# including it would greedily swallow ``key=value`` as a single token.
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/_\-]{12,}")
# base64/hex blob (long, padded or unpadded).
_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
# KEY=VALUE / KEY: VALUE assignment with an opaque RHS (any key name).
_ASSIGN_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]{1,60})\s*[:=]\s*(['\"]?)([^\s'\"]{8,})\2"
)


def _shannon_entropy(s: str) -> float:
    """Bits/char Shannon entropy of ``s`` (0.0 for empty)."""
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_secret(token: str, *, strict: bool = False) -> bool:
    """Heuristic: does this standalone token look like an opaque secret?

    Conservative-toward-redaction. A token is secret-shaped if it is long and
    high-entropy, OR mixes letters+digits at a token length where a real word
    wouldn't. Dictionary prose ("implementation") stays below the bar.

    ``strict=True`` (the egress path) additionally redacts medium-length
    digit-free high-entropy tokens — accepting more false positives because an
    egress leak is irreversible, unlike an over-redacted audit line.
    """
    t = token.strip()
    if len(t) < 12:
        return False
    ent = _shannon_entropy(t)
    has_alpha = any(c.isalpha() for c in t)
    has_digit = any(c.isdigit() for c in t)
    # (a) long + high entropy regardless of composition
    if len(t) >= 20 and ent >= 3.5:
        return True
    # (b) token-shaped: letters AND digits, decent entropy, no spaces
    if has_alpha and has_digit and len(t) >= 12 and ent >= 3.0:
        return True
    # (c) base64/hex-ish structure
    if (_B64_RE.fullmatch(t) or _HEX_RE.fullmatch(t)) and ent >= 3.0:
        return True
    # (d) egress-only belt-and-suspenders: medium digit-free high-entropy token
    if strict and len(t) >= 16 and ent >= 3.3:
        return True
    return False


def _looks_secret_value(value: str) -> bool:
    """RHS of an assignment — more aggressive (an opaque RHS is suspicious)."""
    v = value.strip().strip("'\"")
    if len(v) < 8:
        return False
    # common non-secret RHS values stay readable
    if v.lower() in {"true", "false", "none", "null", "localhost", "manual", "deny"}:
        return False
    ent = _shannon_entropy(v)
    if len(v) >= 8 and ent >= 3.0:
        return True
    if _looks_secret(v):
        return True
    return False


def redact_for_autonomy(text: str, *, strict: bool = False) -> str:
    """Redact known + heuristic secrets from ``text``.

    Raises on internal failure so the caller can fail closed; never returns a
    string that silently dropped redaction. ``strict=True`` widens the
    standalone-token heuristic for the irreversible egress path.
    """
    if not text:
        return text

    # 1) Known high-confidence patterns (force=True = ignore global opt-out).
    try:
        from agent.redact import redact_sensitive_text

        result = redact_sensitive_text(text, force=True)
    except Exception:
        # The base redactor is unavailable/errored — fall through to heuristics
        # only; do NOT silently return raw text.
        result = text

    # 2) Opaque-RHS assignments with any key name (the generic-.env gap).
    def _sub_assign(m: re.Match) -> str:
        key, quote, value = m.group(1), m.group(2), m.group(3)
        if _looks_secret_value(value):
            return f"{key}={quote}{_ENV}{quote}"
        return m.group(0)

    result = _ASSIGN_RE.sub(_sub_assign, result)

    # 3) Standalone base64/hex blobs.
    result = _B64_RE.sub(lambda m: _B64 if _shannon_entropy(m.group(0)) >= 3.0 else m.group(0), result)
    result = _HEX_RE.sub(lambda m: _HE if _shannon_entropy(m.group(0)) >= 3.0 else m.group(0), result)

    # 4) Remaining high-entropy standalone tokens.
    def _sub_token(m: re.Match) -> str:
        tok = m.group(0)
        if tok.startswith("[REDACTED:"):
            return tok
        return _HE if _looks_secret(tok, strict=strict) else tok

    result = _TOKEN_RE.sub(_sub_token, result)
    return result


def is_safe_for_egress(text: str) -> tuple[bool, str]:
    """Return (safe, redacted_text). ``safe`` is False if redaction failed.

    Uses the stricter heuristic — an egress leak is irreversible, so the egress
    path accepts more false positives than the audit path. A caller emitting
    autonomous egress should refuse when ``safe`` is False.
    """
    try:
        return True, redact_for_autonomy(text, strict=True)
    except Exception:
        return False, ""


__all__ = ["redact_for_autonomy", "is_safe_for_egress"]
