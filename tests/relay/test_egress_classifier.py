"""Corpus tests for the fail-closed egress classifier (PRD-035 AC-006 / AC-013).

Two invariants:
  * Every enumerated credential shape is REFUSED (contains_credential -> True).
  * Benign high-entropy shapes that appear in real code-review payloads are
    ALLOWED (contains_credential -> False) — the classifier must not self-DoS.
Plus the fail-closed contract: bad type / oversize / (simulated) error => refuse.
"""

from __future__ import annotations

import pytest

from relay import egress_classifier as ec


# --- Credential shapes that MUST be refused ---------------------------------

REFUSE_CASES = {
    "claude_oauth_file": '{"claudeAiOauth": {"accessToken": "abc123def456ghi", "refreshToken": "zzz999"}}',
    "camel_access_token": '"accessToken": "sk-xyz-01234567890"',
    "camel_refresh_token": '{"refreshToken":"1a2b3c4d5e6f7g"}',
    "snake_access_token": '"access_token": "ya29placeholderxxxx"',
    "snake_refresh_token": 'refresh_token=ABCDEF123456ghijkl',
    "codex_tokens_block": '{"tokens": {"access_token": "eyJhd), "refresh_token": "r-9988"}}',
    "gemini_antigravity": '{"access": "tok-aaaa1111", "refresh": "tok-bbbb2222", "expires": 1730, "email": "x@y.z"}',
    "google_access_value": 'here is a token ya29.a0AfB_byC3d4e5f6g7h8i9j0k1l2m3n4o5p6',
    "google_refresh_value": 'refresh: 1//0 abc omitted -> 1//04aBcDeFgHiJkLmNoPqRsTuV',
    "jwt": 'Authorization: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJ',
    "pem_key": '-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEA\n-----END',
    "anthropic_key": 'export ANTHROPIC_API_KEY=sk-ant-api03-AbCdEf0123456789xyz',
    "openai_key": 'key sk-proj-ABCDEFGHIJKLMNOPQRST0123',
    "github_pat": 'token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    "github_fine_pat": 'github_pat_11ABCDEFG0123456789_abcdefghijklmnop',
    "aws_akid": 'AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE',
    "slack_token": 'xoxb-2401-2402-AbCdEfGhIjKlMnOp',
    "url_userinfo": 'clone https://scott:s3cr3tPASSWORD@github.com/org/repo.git',
    "url_query_token": 'GET https://api.example.com/v1/thing?api_key=abcdef123456&x=1',
    "form_password": 'username=scott&password=hunter2hunter2&submit=1',
}


@pytest.mark.parametrize("name,payload", sorted(REFUSE_CASES.items()))
def test_credential_shapes_are_refused(name, payload):
    refuse, reason = ec.contains_credential(payload)
    assert refuse is True, f"{name}: expected refuse, got allow"
    assert reason and reason.startswith(("credential_shape:", "classifier_")), reason


# --- Benign high-entropy shapes that MUST be allowed ------------------------

ALLOW_CASES = {
    "git_sha": "Fixed in commit 9f2a1c4b7e8d3f60a1b2c3d4e5f60718293a4b5c on main.",
    "uuid": 'id = "550e8400-e29b-41d4-a716-446655440000"  # request id',
    "sha512_integrity": 'sha512-3+2Kt9O5X0wYcQ0m1n2o3p4q5r6s7t8u9vAbCdEfGhIjKlMnOpQrStUvWxYz==',
    "base64_fixture": 'const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC";',
    "plain_prose": "Please review this plan: we access the refresh button, then email the team about expires-header caching.",
    "code_no_secret": "def add(a, b):\n    return a + b  # simple helper, no tokens here",
    "hex_hash_line": "integrity checksum: d41d8cd98f00b204e9800998ecf8427e",
}


@pytest.mark.parametrize("name,payload", sorted(ALLOW_CASES.items()))
def test_benign_payloads_are_allowed(name, payload):
    refuse, reason = ec.contains_credential(payload)
    assert refuse is False, f"{name}: expected allow, got refuse ({reason})"
    assert reason is None


# --- Split-across-fields: the caller concatenates, so a secret spanning the
#     prompt/context boundary is caught once assembled -------------------------

def test_concatenated_scan_catches_secret_in_context_field():
    # The relay scans the assembled prompt+context ONCE. A credential living
    # entirely in the context field (clean prompt) must still be caught — which
    # a naive per-field scan of only `prompt` would miss.
    prompt = 'Please review this deployment script for correctness.'
    context = 'ANTHROPIC_API_KEY=sk-ant-api03-AbCdEf0123456789xyz'
    assembled = f"{prompt}\n\n--- context ---\n{context}"
    refuse, reason = ec.contains_credential(assembled)
    assert refuse is True
    assert reason and reason.startswith("credential_shape:")


# --- Fail-closed contract ---------------------------------------------------

def test_non_string_input_refuses():
    refuse, reason = ec.contains_credential(b"bytes-not-str")  # type: ignore[arg-type]
    assert refuse is True
    assert reason and reason.startswith("classifier_refuse")


def test_none_input_refuses():
    refuse, reason = ec.contains_credential(None)  # type: ignore[arg-type]
    assert refuse is True


def test_oversize_payload_refuses():
    huge = "a" * (ec._MAX_SCAN_CHARS + 1)
    refuse, reason = ec.contains_credential(huge)
    assert refuse is True
    assert reason and reason.startswith("classifier_refuse")


def test_internal_error_fails_closed(monkeypatch):
    def _boom(_text):
        raise RuntimeError("simulated scanner crash")

    monkeypatch.setattr(ec, "_run_with_timeout", _boom)
    refuse, reason = ec.contains_credential("anything")
    assert refuse is True
    assert reason == "classifier_error"


# --- Return-channel redaction (FR-6a) ---------------------------------------

def test_redact_masks_named_shapes():
    text = "Here is your key sk-ant-api03-AbCdEf0123456789xyz and a jwt eyJabc.def123.ghi456 done."
    out = ec.redact(text)
    assert "sk-ant-api03-AbCdEf0123456789xyz" not in out
    assert ec._REDACT_MASK in out


def test_redact_masks_gemini_pair():
    text = '{"access": "tok-aaaa1111", "refresh": "tok-bbbb2222", "email": "x@y.z"}'
    out = ec.redact(text)
    assert "tok-aaaa1111" not in out
    assert "tok-bbbb2222" not in out


def test_redact_non_string_fails_closed():
    assert ec.redact(12345) == ec._REDACT_MASK  # type: ignore[arg-type]


def test_redact_preserves_benign_text():
    text = "Fixed in commit 9f2a1c4b7e8d3f60a1b2c3d4e5f60718293a4b5c."
    assert ec.redact(text) == text
