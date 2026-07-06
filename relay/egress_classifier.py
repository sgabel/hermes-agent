"""Fail-closed egress secret classifier for the advisory relay (PRD-035 FR-6/FR-6a).

This is a **second line**, not the confidentiality boundary (AGENT_SECURITY_MODEL
I1: a blocklist is incomplete by construction and bypassable by obfuscation). The
authoritative boundary is that Anthropic is an accepted T1 destination for
non-credential content. This module's job is narrower and concrete: refuse a
consult whose payload contains a **named credential shape**, and — on the return
channel — redact any such shape that comes back.

Design contract (PRD-035 FR-6):
  * Scan the **concatenated** ``prompt + context`` blob (a secret split across the
    two request fields must still be caught). The caller assembles the blob.
  * **Fail closed**: ``contains_credential`` returns True (refuse) on a positive
    match AND on any internal error, invalid input type, or scan timeout. The
    relay refuses the consult in every one of those cases.
  * **No blanket high-entropy refusal.** Real code-review payloads carry benign
    high-entropy strings (git SHAs, UUIDs, ``sha512-`` lockfile integrity, base64
    fixtures). We match *named* credential shapes only, never a generic entropy
    score, so the tool does not self-DoS on the payloads it exists to serve.

The three peer OAuth credential-file formats are named exactly (verified against
the live fork, PRD-035 FR-6):
  (a) Claude Code   ``claudeAiOauth.{accessToken,refreshToken,...}``  (camelCase)
  (b) Codex/Hermes  ``tokens.{access_token,refresh_token}``           (snake_case)
  (c) Gemini/Antigravity managed OAuth ``{access,refresh,expires,email}`` — bare
      ``access``/``refresh`` JSON keys, NOT ``*_token`` (google_oauth.py:13,430).
"""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass
from typing import Optional

# A scan of a bounded payload is near-instant; the timeout is a fail-closed
# backstop against a pathological regex-vs-input blowup, never the normal path.
_SCAN_TIMEOUT_SECONDS = 5
# Hard ceiling mirrored from the tool's _MAX_PROMPT_CHARS; oversize => refuse.
_MAX_SCAN_CHARS = 48_000


class ClassifierError(Exception):
    """Internal classifier failure. Callers treat this as *refuse* (fail-closed)."""


@dataclass(frozen=True)
class Finding:
    """One matched credential shape. ``label`` is safe to log; ``span`` is not
    the secret itself (we never log the matched value)."""

    label: str
    span: tuple[int, int]


# ---------------------------------------------------------------------------
# Named credential-shape patterns. Each entry is (label, compiled regex).
# Patterns match the *shape*, never a specific tenant's value. We deliberately
# avoid a generic entropy heuristic (see module docstring).
# ---------------------------------------------------------------------------

def _c(pattern: str, flags: int = 0) -> re.Pattern:
    return re.compile(pattern, flags)


# JSON-key shapes for the three peer credential files. We match the *key*
# adjacent to a quoted/bare value so a bare English word ("access denied") does
# not trip the gate. The value is captured GREEDILY (`+`) so that redaction masks
# the WHOLE token, not just its first character (NF-1) — detection only needs
# presence, but redact() reuses these patterns via .sub().
_JSON_VALUE = r'\s*:\s*["\']?[^"\',}\s]+'

_PATTERNS: list[tuple[str, re.Pattern]] = [
    # (a) Claude Code OAuth file — the container-envelope key + camelCase tokens.
    ("claude_oauth_container", _c(r'claudeAiOauth', re.IGNORECASE)),
    ("camel_access_token", _c(r'["\']?accessToken["\']?' + _JSON_VALUE)),
    ("camel_refresh_token", _c(r'["\']?refreshToken["\']?' + _JSON_VALUE)),
    # (b) Codex / Hermes OAuth file — snake_case tokens (also generic OAuth JSON).
    ("snake_access_token", _c(r'["\']?access_token["\']?' + _JSON_VALUE)),
    ("snake_refresh_token", _c(r'["\']?refresh_token["\']?' + _JSON_VALUE)),
    # (c) Gemini / Antigravity managed OAuth — bare access/refresh keys, matched
    # only when a co-occurring managed-OAuth field is present so plain prose is
    # not swept up (handled in _gemini_shape below, not a single regex).
    # OAuth token *values* (Google).
    ("google_access_value", _c(r'\bya29\.[A-Za-z0-9._\-]{20,}')),
    ("google_refresh_value", _c(r'\b1//[A-Za-z0-9._\-]{20,}')),
    # JWT (three base64url segments).
    ("jwt", _c(r'\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}')),
    # PEM private-key blocks.
    ("pem_private_key", _c(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----')),
    # Provider API-key prefixes.
    ("anthropic_key", _c(r'\bsk-ant-[A-Za-z0-9_\-]{16,}')),
    ("openai_key", _c(r'\bsk-(?:proj-)?[A-Za-z0-9]{20,}')),
    ("github_pat", _c(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}')),
    ("github_fine_pat", _c(r'\bgithub_pat_[A-Za-z0-9_]{20,}')),
    ("aws_access_key_id", _c(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ("slack_token", _c(r'\bxox[baprs]-[A-Za-z0-9\-]{10,}')),
    ("fireworks_key", _c(r'\bfw_[A-Za-z0-9]{20,}')),
    # URL userinfo credentials  scheme://user:secret@host. Quantifiers are BOUNDED
    # (N1) so a long run of matching chars with no "://" is O(n), not O(n^2).
    ("url_userinfo", _c(r'[a-zA-Z][a-zA-Z0-9+.\-]{1,40}://[^/\s:@]{1,256}:[^/\s:@]{1,256}@')),
    # Token-bearing URL query / form params (the display redactor's blind spot).
    ("url_or_form_token", _c(
        r'(?:access_token|refresh_token|id_token|api[_\-]?key|auth[_\-]?token|'
        r'client[_\-]?secret|password|passwd|pwd|session|token)=[^&\s"\']{6,}',
        re.IGNORECASE,
    )),
    # Common secret JSON-key shapes in config/code payloads (the primary review
    # use case): "api_key"/"apiKey"/"client_secret"/"secret"/"password".
    ("json_api_key", _c(r'["\']?api[_\-]?key["\']?' + _JSON_VALUE, re.IGNORECASE)),
    ("json_client_secret", _c(r'["\']?client[_\-]?secret["\']?' + _JSON_VALUE, re.IGNORECASE)),
    ("json_secret", _c(r'["\'](?:secret|password|passwd)["\']' + _JSON_VALUE, re.IGNORECASE)),
]

# Benign high-entropy shapes we must NOT treat as secrets (avoid self-DoS on
# ordinary code-review payloads). These are checked only to document intent;
# because we match *named* shapes above rather than raw entropy, they already
# pass — the allowlist is asserted by the test corpus.
_BENIGN_EXAMPLES = ("git_sha", "uuid", "sha512_integrity", "base64_fixture")

# Gemini/Antigravity managed-OAuth co-occurrence shape (bare access + refresh
# alongside expires/email). Matched as a pair so plain English does not trip it.
_GEMINI_ACCESS = _c(r'["\']access["\']' + _JSON_VALUE)
_GEMINI_REFRESH = _c(r'["\']refresh["\']' + _JSON_VALUE)
_GEMINI_COMPANION = _c(r'["\'](?:expires|email)["\']\s*:')


def _gemini_shape(text: str) -> Optional[Finding]:
    """Gemini/Antigravity file shape: bare ``access`` + ``refresh`` keys with a
    co-occurring ``expires``/``email`` key. Requires the pairing so a lone
    ``"access": ...`` config field elsewhere does not false-positive."""
    a = _GEMINI_ACCESS.search(text)
    if not a:
        return None
    if _GEMINI_REFRESH.search(text) and _GEMINI_COMPANION.search(text):
        return Finding("gemini_antigravity_oauth", a.span())
    return None


class _ScanTimeout(Exception):
    pass


def _scan(text: str) -> Optional[Finding]:
    """Return the first credential Finding, or None. Raises ClassifierError on a
    bad input type or oversize payload."""
    if not isinstance(text, str):
        raise ClassifierError(f"payload must be str, got {type(text).__name__}")
    if len(text) > _MAX_SCAN_CHARS:
        # Oversize is itself a refuse condition (FR-6 fail-closed list).
        raise ClassifierError(f"payload too large to scan ({len(text)} > {_MAX_SCAN_CHARS})")

    for label, pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return Finding(label, m.span())
    return _gemini_shape(text)


# One shared single-thread executor for the scan timeout. `signal.alarm` only
# fires on the main thread, but the relay runs consults on worker threads
# (ThreadingMixIn) — a futures timeout is thread-safe (NF-4). The regexes are
# all linear (bounded input, negated char classes, no nested quantifiers), so
# the timeout is a backstop, not the primary defense.
_SCAN_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="egress-scan")


def _run_with_timeout(text: str) -> Optional[Finding]:
    """Run _scan under a real wall-clock timeout that works on any thread. On
    timeout raise _ScanTimeout => the caller refuses (fail-closed)."""
    fut = _SCAN_POOL.submit(_scan, text)
    try:
        return fut.result(timeout=_SCAN_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        raise _ScanTimeout()


def contains_credential(text: str) -> tuple[bool, Optional[str]]:
    """Fail-closed gate. Returns ``(refuse, reason)``.

    ``refuse`` is True when the payload contains a named credential shape OR when
    the scan cannot be completed for ANY reason (bad type, oversize, timeout,
    internal error). ``reason`` is a short, secret-free label for the audit line.
    The matched credential value is NEVER returned or logged.
    """
    try:
        finding = _run_with_timeout(text)
    except _ScanTimeout:
        return True, "classifier_timeout"
    except ClassifierError as exc:
        return True, f"classifier_refuse:{exc.args[0].split(' ')[0] if exc.args else 'error'}"
    except Exception:  # pragma: no cover - any unexpected error => refuse
        return True, "classifier_error"

    if finding is not None:
        return True, f"credential_shape:{finding.label}"
    return False, None


# ---------------------------------------------------------------------------
# Return-channel redaction (FR-6a). Advisory text coming back from Anthropic is
# scrubbed with THIS classifier's shapes (a redacting variant), not the weak
# display redactor. On any error we return a fully-masked string (fail-closed).
# ---------------------------------------------------------------------------

_REDACT_MASK = "[REDACTED-CREDENTIAL]"


def redact(text: str) -> str:
    """Mask every named credential shape in ``text``. Fail-closed: if scanning
    raises, return a single mask token rather than risk leaking the original."""
    try:
        if not isinstance(text, str):
            return _REDACT_MASK
        out = text
        for _label, pat in _PATTERNS:
            out = pat.sub(_REDACT_MASK, out)
        # Gemini pair: mask both bare keys' values if the companion shape is present.
        if _GEMINI_COMPANION.search(out) and _GEMINI_REFRESH.search(out):
            out = _GEMINI_ACCESS.sub(_REDACT_MASK, out)
            out = _GEMINI_REFRESH.sub(_REDACT_MASK, out)
        return out
    except Exception:  # pragma: no cover
        return _REDACT_MASK
