"""PRD-024 — ask_claude (Claude Sonnet second-opinion tool) security envelope.

Covers the three STOP fixes from adversarial review:
  - AC-002 (STOP-1): the handler calls check_all_command_guards; a blocked gate
    decision returns a blocked result and does NOT spawn the subprocess.
  - AC-003: the child env is built by the sanitizer (provider creds stripped,
    $HOME preserved for OAuth-via-file).
  - AC-006/EXFIL (STOP-2): a prompt containing a secret is refused before spawn.
  - AC-007/CRON (STOP-3): unavailable + hard-refused in cron context.
  - AC-004: output is capped/truncated with a marker.
  - Model resolution, non-JSON fallback, empty-prompt guard.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

import tools.claude_review_tool as t


def _fake_proc(stdout="", returncode=0, stderr=""):
    p = MagicMock()
    p.stdout = stdout
    p.stderr = stderr
    p.returncode = returncode
    return p


# --------------------------------------------------------------------------
# AC-002 (STOP-1) — explicit gate; block => no spawn
# --------------------------------------------------------------------------
class TestGateWiring:
    def test_gate_is_called_and_block_prevents_spawn(self):
        with patch("tools.approval.check_all_command_guards",
                   return_value={"approved": False, "description": "denied"}) as gate, \
             patch("tools.claude_review_tool.subprocess.run") as run:
            out = json.loads(t.ask_claude(prompt="review my plan"))
        assert gate.called
        assert not run.called
        assert out.get("blocked") == "gate"
        assert out.get("success") is False

    def test_pending_approval_surfaced_and_no_spawn(self):
        with patch("tools.approval.check_all_command_guards",
                   return_value={"approved": False, "status": "pending_approval",
                                 "description": "needs approval"}), \
             patch("tools.claude_review_tool.subprocess.run") as run:
            out = json.loads(t.ask_claude(prompt="review my plan"))
        assert not run.called
        assert out.get("status") == "pending_approval"
        assert out.get("approval_pending") is True

    def test_gate_error_fails_closed(self):
        with patch("tools.approval.check_all_command_guards",
                   side_effect=RuntimeError("boom")), \
             patch("tools.claude_review_tool.subprocess.run") as run:
            out = json.loads(t.ask_claude(prompt="review my plan"))
        assert not run.called
        assert out.get("success") is False
        assert "gate failed" in out.get("error", "")

    def test_approved_path_spawns_with_sanitized_env(self):
        with patch("tools.approval.check_all_command_guards",
                   return_value={"approved": True}), \
             patch("tools.environments.local._make_run_env",
                   return_value={"HOME": "/home/sgabel", "PATH": "/usr/bin"}) as mkenv, \
             patch("tools.claude_review_tool.subprocess.run",
                   return_value=_fake_proc(stdout=json.dumps({"result": "LGTM",
                                                              "total_cost_usd": 0.01}))) as run:
            out = json.loads(t.ask_claude(prompt="review my plan"))
        assert run.called
        # spawned with the sanitizer-built env, not raw os.environ
        assert run.call_args.kwargs["env"] == {"HOME": "/home/sgabel", "PATH": "/usr/bin"}
        assert mkenv.called
        assert out["response"] == "LGTM"
        assert out["structured"] is True


# --------------------------------------------------------------------------
# AC-003 — real sanitizer strips provider creds, keeps $HOME
# --------------------------------------------------------------------------
def test_sanitizer_strips_provider_creds_keeps_home(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-HOSTILE")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-HOSTILE")
    from tools.environments.local import _make_run_env
    child = _make_run_env({})
    assert "ANTHROPIC_API_KEY" not in child
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in child
    assert child.get("HOME")          # preserved for OAuth-via-file


# --------------------------------------------------------------------------
# AC-006 / EXFIL (STOP-2) — secret in prompt is refused before spawn
# --------------------------------------------------------------------------
class TestExfilRefusal:
    @pytest.mark.parametrize("secret", [
        "here is the key sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFF",
        "db at postgres://admin:hunter2@10.0.0.1:5432/prod",
        "Authorization: Bearer ya29.SECRETTOKENVALUE12345",
        "aws key AKIAIOSFODNN7EXAMPLE in the config",   # AC-006 names AWS keys
    ])
    def test_secret_in_prompt_refused_no_spawn(self, secret):
        with patch("tools.approval.check_all_command_guards") as gate, \
             patch("tools.claude_review_tool.subprocess.run") as run:
            out = json.loads(t.ask_claude(prompt=f"review this: {secret}"))
        assert out.get("blocked") == "secret_in_prompt"
        assert not run.called          # refused BEFORE the gate/spawn
        assert not gate.called

    def test_secret_in_context_also_refused(self):
        with patch("tools.claude_review_tool.subprocess.run") as run:
            out = json.loads(t.ask_claude(
                prompt="review my config",
                context="API_KEY=sk-ant-api03-ZZZZYYYYXXXXWWWW"))
        assert out.get("blocked") == "secret_in_prompt"
        assert not run.called

    def test_clean_prompt_not_refused(self):
        with patch("tools.approval.check_all_command_guards",
                   return_value={"approved": True}), \
             patch("tools.environments.local._make_run_env", return_value={"HOME": "/h"}), \
             patch("tools.claude_review_tool.subprocess.run",
                   return_value=_fake_proc(stdout=json.dumps({"result": "ok"}))):
            out = json.loads(t.ask_claude(prompt="refactor the parser for clarity"))
        assert out.get("blocked") is None
        assert out["response"] == "ok"


# --------------------------------------------------------------------------
# AC-007 / CRON (STOP-3) — disabled in cron
# --------------------------------------------------------------------------
class TestCronDisabled:
    def test_check_fn_false_in_cron(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        assert t.check_ask_claude_requirements() is False

    def test_check_fn_true_outside_cron_when_cli_present(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        with patch("tools.claude_review_tool.shutil.which", return_value="/usr/bin/claude"):
            assert t.check_ask_claude_requirements() is True

    def test_handler_hard_refuses_in_cron(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        with patch("tools.claude_review_tool.subprocess.run") as run:
            out = json.loads(t.ask_claude(prompt="review my plan"))
        assert out.get("blocked") == "cron"
        assert not run.called


# --------------------------------------------------------------------------
# AC-004 — output cap / truncation
# --------------------------------------------------------------------------
def test_output_capped_with_marker():
    huge = "z" * (t._MAX_RESPONSE_CHARS + 5000)
    with patch("tools.approval.check_all_command_guards", return_value={"approved": True}), \
         patch("tools.environments.local._make_run_env", return_value={"HOME": "/h"}), \
         patch("tools.claude_review_tool.subprocess.run",
               return_value=_fake_proc(stdout=json.dumps({"result": huge}))):
        out = json.loads(t.ask_claude(prompt="summarize"))
    assert out["truncated"] is True
    assert "[truncated" in out["response"]
    assert len(out["response"]) <= t._MAX_RESPONSE_CHARS + 80


# --------------------------------------------------------------------------
# Misc: model resolution, non-JSON fallback, guards
# --------------------------------------------------------------------------
class TestMisc:
    def test_empty_prompt_rejected(self):
        with patch("tools.claude_review_tool.subprocess.run") as run:
            out = json.loads(t.ask_claude(prompt="   "))
        assert out.get("success") is False
        assert not run.called

    def test_oversized_prompt_rejected(self):
        out = json.loads(t.ask_claude(prompt="x" * (t._MAX_PROMPT_CHARS + 1)))
        assert out.get("success") is False
        assert "too large" in out.get("error", "")

    def test_non_json_output_marked_unstructured(self):
        with patch("tools.approval.check_all_command_guards", return_value={"approved": True}), \
             patch("tools.environments.local._make_run_env", return_value={"HOME": "/h"}), \
             patch("tools.claude_review_tool.subprocess.run",
                   return_value=_fake_proc(stdout="plain text not json")):
            out = json.loads(t.ask_claude(prompt="review"))
        assert out["structured"] is False
        assert out["response"] == "plain text not json"

    def test_model_resolution_returns_nonempty(self):
        # Resolves the pinned id from config, or the default — never empty.
        assert isinstance(t._resolve_model(), str) and t._resolve_model().strip()

    def test_nonzero_exit_reported(self):
        with patch("tools.approval.check_all_command_guards", return_value={"approved": True}), \
             patch("tools.environments.local._make_run_env", return_value={"HOME": "/h"}), \
             patch("tools.claude_review_tool.subprocess.run",
                   return_value=_fake_proc(returncode=1, stderr="auth error")):
            out = json.loads(t.ask_claude(prompt="review"))
        assert out.get("success") is False
        assert "exited 1" in out.get("error", "")

