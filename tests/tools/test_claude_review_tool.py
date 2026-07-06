"""Tests for ask_claude routed through the host advisory relay (PRD-035).

The tool no longer spawns `claude`; it forwards to the relay over a unix socket.
These tests stub the relay call + the interactive gate and assert the tool's
fail-closed behavior, the untrusted-provenance marker, and cron routing.
"""

import json
from unittest.mock import patch

import pytest

import tools.claude_review_tool as t


def _approved():
    return {"approved": True}


def _relay_ok(text="here is my advisory"):
    return (200, {"advisory_text": text, "truncated": False})


@pytest.fixture(autouse=True)
def _relay_present(monkeypatch):
    # Default: relay reachable, bearer readable, not cron.
    monkeypatch.setattr(t, "_relay_available", lambda: True)
    monkeypatch.setattr(t, "_read_bearer", lambda: "test-token")
    monkeypatch.setattr(t, "_in_cron_context", lambda: False)


class TestGate:
    def test_gate_block_prevents_relay_call(self):
        with patch("tools.approval.check_all_command_guards",
                   return_value={"approved": False, "description": "blocked by tirith"}), \
             patch("tools.claude_review_tool._call_relay") as call:
            out = json.loads(t.ask_claude(prompt="review my plan"))
        assert out["success"] is False
        call.assert_not_called()

    def test_pending_approval_surfaced_no_call(self):
        with patch("tools.approval.check_all_command_guards",
                   return_value={"approved": False, "status": "pending_approval",
                                 "description": "needs approval"}), \
             patch("tools.claude_review_tool._call_relay") as call:
            out = json.loads(t.ask_claude(prompt="review my plan"))
        assert out.get("approval_pending") is True
        call.assert_not_called()

    def test_gate_error_fails_closed(self):
        with patch("tools.approval.check_all_command_guards", side_effect=RuntimeError("boom")), \
             patch("tools.claude_review_tool._call_relay") as call:
            out = json.loads(t.ask_claude(prompt="review my plan"))
        assert out["success"] is False
        call.assert_not_called()


class TestRelayDownFailsClosed:
    def test_relay_unavailable_refuses_no_fallback(self, monkeypatch):
        monkeypatch.setattr(t, "_relay_available", lambda: False)
        out = json.loads(t.ask_claude(prompt="review my plan"))
        assert out["success"] is False
        assert out.get("blocked") == "relay_down"

    def test_relay_transport_error_fails_closed(self):
        with patch("tools.approval.check_all_command_guards", return_value=_approved()), \
             patch("tools.claude_review_tool._call_relay", side_effect=OSError("connrefused")):
            out = json.loads(t.ask_claude(prompt="review my plan"))
        assert out["success"] is False
        assert out.get("blocked") == "relay_error"


class TestSecretRefusal:
    @pytest.mark.parametrize("secret", [
        "sk-ant-api03-AbCdEf0123456789xyzTOKEN",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ])
    def test_secret_in_prompt_refused_no_call(self, secret):
        with patch("tools.claude_review_tool._call_relay") as call:
            out = json.loads(t.ask_claude(prompt=f"review this: {secret}"))
        assert out["success"] is False
        assert out.get("blocked") == "secret_in_prompt"
        call.assert_not_called()

    def test_clean_prompt_reaches_relay(self):
        with patch("tools.approval.check_all_command_guards", return_value=_approved()), \
             patch("tools.claude_review_tool._call_relay", return_value=_relay_ok()):
            out = json.loads(t.ask_claude(prompt="refactor the parser for clarity"))
        assert out["success"] is True
        assert "UNTRUSTED ADVISORY" in out["response"]


class TestCronRouting:
    def test_cron_skips_discord_gate_and_calls_relay(self, monkeypatch):
        monkeypatch.setattr(t, "_in_cron_context", lambda: True)
        with patch("tools.approval.check_all_command_guards") as gate, \
             patch("tools.claude_review_tool._call_relay", return_value=_relay_ok()) as call:
            out = json.loads(t.ask_claude(prompt="pre-action sanity check"))
        assert out["success"] is True
        gate.assert_not_called()            # FR-7: relay is authoritative in cron
        assert call.call_args.args[1] == "cron"   # surface forwarded


class TestRelayErrors:
    @pytest.mark.parametrize("status,blocked", [
        (401, "relay_auth"), (422, "secret_in_prompt"), (429, "budget"), (503, "quiesced_or_busy"),
    ])
    def test_relay_error_statuses_mapped(self, status, blocked):
        with patch("tools.approval.check_all_command_guards", return_value=_approved()), \
             patch("tools.claude_review_tool._call_relay",
                   return_value=(status, {"error": "nope"})):
            out = json.loads(t.ask_claude(prompt="review"))
        assert out["success"] is False
        assert out.get("blocked") == blocked


class TestUntrustedMarkerAndCap:
    def test_untrusted_prefix_present(self):
        with patch("tools.approval.check_all_command_guards", return_value=_approved()), \
             patch("tools.claude_review_tool._call_relay", return_value=_relay_ok("advice body")):
            out = json.loads(t.ask_claude(prompt="x"))
        assert out["response"].startswith("[UNTRUSTED ADVISORY")
        assert "advice body" in out["response"]

    def test_output_capped(self):
        big = "z" * (t._MAX_RESPONSE_CHARS + 500)
        with patch("tools.approval.check_all_command_guards", return_value=_approved()), \
             patch("tools.claude_review_tool._call_relay", return_value=(200, {"advisory_text": big})):
            out = json.loads(t.ask_claude(prompt="summarize"))
        assert out.get("truncated") is True
        assert "truncated" in out["response"]


class TestInputValidation:
    def test_empty_prompt_rejected(self):
        out = json.loads(t.ask_claude(prompt="   "))
        assert out["success"] is False

    def test_oversized_prompt_rejected(self):
        out = json.loads(t.ask_claude(prompt="a" * (t._MAX_PROMPT_CHARS + 1)))
        assert out["success"] is False


class TestAvailability:
    def test_check_fn_true_when_relay_present(self, monkeypatch):
        monkeypatch.setattr(t, "_relay_available", lambda: True)
        assert t.check_ask_claude_requirements() is True

    def test_check_fn_false_when_relay_absent(self, monkeypatch):
        monkeypatch.setattr(t, "_relay_available", lambda: False)
        assert t.check_ask_claude_requirements() is False
