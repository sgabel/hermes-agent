"""Relay pipeline tests (PRD-035). The four relay checks are the sole security
boundary, so they are exercised directly against `handle_consult`, with the
claude spawn and autonomy primitives stubbed (no real egress in unit tests).

Covers: bearer (AC-005), classifier refuse (AC-006), kill-switch (AC-007),
budget admission incl. surface:interactive still debits + over-cap refuse
(AC-007/C2), model pinning (AC-015), return-channel scrub (AC-016), and the
startup self-canary (AC-012).
"""

from __future__ import annotations

import sys
import types

import pytest

from relay import advisory_relay as ar
from relay.advisory_relay import AdvisoryRelay, RelayConfig


@pytest.fixture
def relay(tmp_path):
    token = tmp_path / "client.token"
    token.write_text("test-bearer-secret-value", encoding="utf-8")
    cfg = RelayConfig(
        socket_path=str(tmp_path / "consult.sock"),
        token_path=str(token),
        model="claude-sonnet-5-PINNED",
        isolated_home=str(tmp_path / "claude-home"),
    )
    r = AdvisoryRelay(cfg)
    # Simulate a passed startup self-canary so the per-consult re-canary guard
    # (NF-2) treats the binary as vouched-for. Tests that exercise the re-canary
    # path override this explicitly.
    r._canary_binary_sig = r._binary_signature()
    return r


@pytest.fixture
def stub_autonomy(monkeypatch):
    """Install fake autonomy.{killswitch,budget,audit} modules the relay imports
    lazily. Returns a dict of knobs the tests flip."""
    state = {"quiesced": False, "allowed": True, "debits": [], "audits": []}

    ks = types.ModuleType("autonomy.killswitch")
    ks.guard = lambda surface: state["quiesced"]  # noqa: E731

    bud = types.ModuleType("autonomy.budget")

    def _debit(surface, kind, amount=1, *, audit=True):
        state["debits"].append((surface, kind, amount))
        return {"allowed": state["allowed"], "degrade": not state["allowed"],
                "kind": kind, "usage": {}}

    bud.debit = _debit

    aud = types.ModuleType("autonomy.audit")
    aud.record = lambda **kw: state["audits"].append(kw)

    pkg = types.ModuleType("autonomy")
    monkeypatch.setitem(sys.modules, "autonomy", pkg)
    monkeypatch.setitem(sys.modules, "autonomy.killswitch", ks)
    monkeypatch.setitem(sys.modules, "autonomy.budget", bud)
    monkeypatch.setitem(sys.modules, "autonomy.audit", aud)
    return state


def _stub_spawn(relay, monkeypatch, text="Here is my advisory.", ok=True):
    monkeypatch.setattr(relay, "_spawn_claude", lambda assembled: (ok, text))


# --- bearer (AC-005) --------------------------------------------------------

def test_bearer_ok(relay):
    assert relay._bearer_ok("Bearer test-bearer-secret-value") is True
    assert relay._bearer_ok("test-bearer-secret-value") is True
    assert relay._bearer_ok("Bearer wrong") is False
    assert relay._bearer_ok(None) is False
    assert relay._bearer_ok("") is False


# --- classifier refuse (AC-006) ---------------------------------------------

def test_consult_refuses_credential_payload(relay, stub_autonomy, monkeypatch):
    _stub_spawn(relay, monkeypatch)
    status, body = relay.handle_consult(
        {"prompt": "review this", "context": "ANTHROPIC_API_KEY=sk-ant-api03-AbCdEf0123456789xyz"}
    )
    assert status == 422
    assert "reason" in body
    # Never spawned, never debited.
    assert stub_autonomy["debits"] == []


# --- kill-switch (AC-007) ---------------------------------------------------

def test_consult_blocked_when_quiesced(relay, stub_autonomy, monkeypatch):
    _stub_spawn(relay, monkeypatch)
    stub_autonomy["quiesced"] = True
    status, body = relay.handle_consult({"prompt": "review this"})
    assert status == 503
    assert "quiesced" in body["error"]
    assert stub_autonomy["debits"] == []  # kill-switch precedes debit


# --- budget admission (AC-007 / C2) -----------------------------------------

def test_happy_path_debits_and_returns_advisory(relay, stub_autonomy, monkeypatch):
    _stub_spawn(relay, monkeypatch, text="my advice")
    status, body = relay.handle_consult({"prompt": "review this", "surface": "cron"})
    assert status == 200
    assert body["advisory_text"] == "my advice"
    assert stub_autonomy["debits"] == [("advisory_relay:cron", "second_opinion_calls", 1)]
    # A T1 audit line was written by the relay itself.
    assert any(a["tier"] == "T1" and a["outcome"] == "ok" for a in stub_autonomy["audits"])


def test_surface_interactive_still_debits(relay, stub_autonomy, monkeypatch):
    # C2: surface is spoofable; an "interactive" claim must NOT skip the debit.
    _stub_spawn(relay, monkeypatch)
    relay.handle_consult({"prompt": "review this", "surface": "interactive"})
    assert stub_autonomy["debits"] == [("advisory_relay:interactive", "second_opinion_calls", 1)]


def test_over_cap_refuses(relay, stub_autonomy, monkeypatch):
    _stub_spawn(relay, monkeypatch)
    stub_autonomy["allowed"] = False  # debit reports over-cap
    status, body = relay.handle_consult({"prompt": "review this"})
    assert status == 429
    assert "budget" in body["error"]


# --- model pinning (AC-015) -------------------------------------------------

def test_model_is_pinned_not_client_controlled(relay, stub_autonomy, monkeypatch):
    captured = {}
    monkeypatch.setattr(relay, "_spawn_claude",
                        lambda assembled: (captured.setdefault("ran", True), "ok"))
    # A client-supplied model in the body must be ignored — handle_consult never
    # reads body["model"]; the relay always uses config.model.
    relay.handle_consult({"prompt": "x", "model": "claude-opus-EXPENSIVE"})
    assert relay.config.model == "claude-sonnet-5-PINNED"


# --- return-channel scrub (AC-016) ------------------------------------------

def test_return_channel_is_scrubbed(relay, stub_autonomy, monkeypatch):
    leaky = "Sure: your key is sk-ant-api03-LEAKEDsecret0123456789 — use it."
    _stub_spawn(relay, monkeypatch, text=leaky)
    status, body = relay.handle_consult({"prompt": "x"})
    assert status == 200
    assert "sk-ant-api03-LEAKEDsecret0123456789" not in body["advisory_text"]
    assert "[REDACTED-CREDENTIAL]" in body["advisory_text"]


# --- input validation -------------------------------------------------------

def test_empty_prompt_rejected(relay, stub_autonomy):
    status, body = relay.handle_consult({"prompt": "   "})
    assert status == 400


def test_oversize_payload_rejected(relay, stub_autonomy):
    status, body = relay.handle_consult({"prompt": "a" * (ar._MAX_PROMPT_CHARS + 1)})
    assert status == 413


# --- fail-closed on governance import failure -------------------------------

def test_budget_import_failure_fails_closed(relay, monkeypatch):
    # No autonomy stub installed AND force killswitch import to succeed-noop but
    # budget to raise → relay must refuse (503), not spawn.
    ks = types.ModuleType("autonomy.killswitch")
    ks.guard = lambda surface: False
    pkg = types.ModuleType("autonomy")
    monkeypatch.setitem(sys.modules, "autonomy", pkg)
    monkeypatch.setitem(sys.modules, "autonomy.killswitch", ks)
    monkeypatch.delitem(sys.modules, "autonomy.budget", raising=False)

    def _raise_import(name, *a, **k):
        raise ImportError("budget unavailable")

    monkeypatch.setattr(relay, "_spawn_claude", lambda a: (True, "should not run"))
    # Make `from autonomy import budget` raise by removing it and blocking import.
    import builtins
    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "autonomy" and fromlist and "budget" in fromlist:
            raise ImportError("budget blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    status, body = relay.handle_consult({"prompt": "x"})
    assert status == 503


# --- self-canary (AC-012) ---------------------------------------------------

def test_self_canary_passes_when_marker_absent(relay, monkeypatch, tmp_path):
    # Toolless claude emits fake tool-call narration but does NOT leak the marker.
    relay.config.isolated_home = tmp_path
    monkeypatch.setattr(relay, "_spawn_claude",
                        lambda probe: (True, "Reading the specified file to check its contents. Read"))
    assert relay.self_canary() is True


def test_self_canary_fails_when_marker_leaks(relay, monkeypatch, tmp_path):
    # A real read (tools re-enabled) would echo the planted marker → refuse.
    relay.config.isolated_home = tmp_path
    marker = "RELAY-CANARY-DO-NOT-EXFIL-7f3a9c2e"
    monkeypatch.setattr(relay, "_spawn_claude",
                        lambda probe: (True, f"the file contains {marker}"))
    assert relay.self_canary() is False


def test_self_canary_fails_when_spawn_fails(relay, monkeypatch, tmp_path):
    relay.config.isolated_home = tmp_path
    monkeypatch.setattr(relay, "_spawn_claude", lambda probe: (False, "spawn error"))
    assert relay.self_canary() is False


# --- NF-2 re-canary on binary change ----------------------------------------

def test_recanary_fires_when_binary_changes(relay, stub_autonomy, monkeypatch):
    # Force a binary-signature mismatch → the consult must re-run the canary.
    relay._canary_binary_sig = ("stale", 0)
    calls = {"canary": 0}

    def _canary():
        calls["canary"] += 1
        relay._canary_binary_sig = relay._binary_signature()
        return True

    monkeypatch.setattr(relay, "self_canary", _canary)
    monkeypatch.setattr(relay, "_spawn_claude", lambda a: (True, "ok"))
    status, _ = relay.handle_consult({"prompt": "x"})
    assert status == 200
    assert calls["canary"] == 1  # re-canary ran before the spawn


def test_recanary_failure_refuses(relay, stub_autonomy, monkeypatch):
    relay._canary_binary_sig = ("stale", 0)
    monkeypatch.setattr(relay, "self_canary", lambda: False)
    spawned = {"n": 0}
    monkeypatch.setattr(relay, "_spawn_claude", lambda a: (spawned.__setitem__("n", spawned["n"] + 1), "x"))
    status, _ = relay.handle_consult({"prompt": "x"})
    assert status == 503
    assert spawned["n"] == 0  # never spawned after a failed re-canary


# --- FR-5 bearer-in-payload refuse (NF-6) -----------------------------------

def test_bearer_in_payload_refused(relay, stub_autonomy, monkeypatch):
    monkeypatch.setattr(relay, "_spawn_claude", lambda a: (True, "ok"))
    status, body = relay.handle_consult({"prompt": f"here is the token {relay._bearer}"})
    assert status == 422
    assert "bearer" in body["error"].lower()
    assert stub_autonomy["debits"] == []  # refused before debit


# --- _spawn_claude command assembly static assertion (move the check left) ---

def test_spawn_command_carries_toolless_and_hardening_flags():
    from relay import advisory_relay as m
    # The containment-critical flags must be present in the fixed arg lists.
    assert "--tools" in m._CLAUDE_BASE_ARGS
    assert m._CLAUDE_BASE_ARGS[m._CLAUDE_BASE_ARGS.index("--tools") + 1] == ""
    assert "--disallowedTools" in m._CLAUDE_BASE_ARGS
    assert "--setting-sources" in m._CLAUDE_BASE_ARGS
    props = " ".join(m._SYSTEMD_RUN_PROPS)
    for needed in ("NoNewPrivileges=yes", "PrivateTmp=yes", "ProtectSystem=strict", "ProtectHome=read-only"):
        assert needed in props


# --- config invariant -------------------------------------------------------

def test_config_rejects_bad_timeout_ordering(tmp_path):
    tok = tmp_path / "t"; tok.write_text("x")
    with pytest.raises(ValueError):
        RelayConfig(socket_path=str(tmp_path / "s.sock"), token_path=str(tok),
                    model="m", isolated_home=str(tmp_path),
                    consult_timeout_sec=100, child_runtime_max_sec=100)


# --- ownership preflight (FR-11 / AC-009) -----------------------------------

def test_ownership_preflight_passes_for_owned_0600_token(tmp_path):
    from relay.advisory_relay import preflight_ownership
    tok = tmp_path / "client.token"; tok.write_text("x")
    import os
    os.chmod(tok, 0o600)
    cfg = RelayConfig(socket_path=str(tmp_path / "s.sock"), token_path=str(tok),
                      model="m", isolated_home=str(tmp_path))
    assert preflight_ownership(cfg) is True


def test_ownership_preflight_fails_for_group_readable_token(tmp_path):
    from relay.advisory_relay import preflight_ownership
    import os
    tok = tmp_path / "client.token"; tok.write_text("x")
    os.chmod(tok, 0o640)
    cfg = RelayConfig(socket_path=str(tmp_path / "s.sock"), token_path=str(tok),
                      model="m", isolated_home=str(tmp_path))
    assert preflight_ownership(cfg) is False


def test_ownership_preflight_fails_for_missing_token(tmp_path):
    from relay.advisory_relay import preflight_ownership
    cfg = RelayConfig(socket_path=str(tmp_path / "s.sock"), token_path=str(tmp_path / "nope"),
                      model="m", isolated_home=str(tmp_path))
    # token must exist to load bearer; preflight is checked before construction,
    # so call it directly.
    assert preflight_ownership(cfg) is False
