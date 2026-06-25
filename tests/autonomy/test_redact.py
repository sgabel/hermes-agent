"""PRD-028 R-4 / AC-007 — hardened secret-screen.

The screen must catch what ``agent.redact.redact_sensitive_text`` misses:
a raw high-entropy token, a base64 blob, and a generic-named .env value —
and fail closed if redaction errors.
"""

from autonomy.redact import is_safe_for_egress, redact_for_autonomy


def test_redacts_raw_high_entropy_token():
    secret = "Zx9Qr7Lp2Wv4Nb8Kc1Hd5Tg3Yf6Ms0Aa"  # 33-char opaque, mixed case+digits
    out = redact_for_autonomy(f"the credential is {secret} use it")
    assert secret not in out
    assert "REDACTED" in out


def test_redacts_base64_blob():
    blob = "QUtJQUlPU0ZPRE5ON0VYQU1QTEVBV1NTRUNSRVRLRVk="  # base64-ish, 44 chars
    out = redact_for_autonomy(f"payload={blob}")
    assert blob not in out
    assert "REDACTED" in out


def test_redacts_generic_named_env_value():
    # var name does NOT match API_KEY|TOKEN|SECRET|... — the documented gap
    out = redact_for_autonomy("MY_THING=s3cr3tValue9x8y7z6w5v4u")
    assert "s3cr3tValue9x8y7z6w5v4u" not in out
    assert "REDACTED" in out


def test_keeps_ordinary_prose_readable():
    text = "the cron job ran and delivered the morning briefing to discord"
    out = redact_for_autonomy(text)
    assert out == text


def test_keeps_benign_config_values():
    out = redact_for_autonomy("approvals_mode=manual cron_mode=deny enabled=true")
    assert "manual" in out and "deny" in out and "true" in out


def test_still_redacts_known_anthropic_key():
    out = redact_for_autonomy("key sk-ant-" + "a" * 40)
    assert "sk-ant-aaaa" not in out


def test_is_safe_for_egress_returns_tuple():
    safe, redacted = is_safe_for_egress("hello world")
    assert safe is True
    assert redacted == "hello world"


def test_egress_strict_redacts_medium_digit_free_token():
    # NIT fix: the egress path is stricter — a medium-length digit-free
    # high-entropy token is redacted there even though the audit path keeps it.
    tok = "xkqjvbmwzhdfnrptls"  # 18 chars, no digit, high entropy
    safe, redacted = is_safe_for_egress(f"emit {tok} now")
    assert safe is True
    assert tok not in redacted


def test_fail_closed_on_redactor_error(monkeypatch):
    # If the inner redactor explodes in an unexpected way, is_safe_for_egress
    # must report unsafe rather than leak. Force redact_for_autonomy to raise.
    import autonomy.redact as r

    monkeypatch.setattr(r, "_ASSIGN_RE", None)  # .sub on None -> AttributeError
    safe, redacted = r.is_safe_for_egress("MY_THING=opaquevalue123456")
    assert safe is False
    assert redacted == ""
