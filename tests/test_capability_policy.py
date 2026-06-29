"""Unit tests for the PRD-032 R1 central capability gate."""

import os
import importlib
import pytest


@pytest.fixture()
def cp(monkeypatch, tmp_path):
    # Default to observe unless a test sets enforce; clear cron/autonomous markers.
    monkeypatch.delenv("HERMES_CAPABILITY_POLICY_MODE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_AUTONOMOUS", raising=False)
    # Isolate the durable approval store so tests never touch real ~/.hermes.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.capability_policy as m
    importlib.reload(m)
    return m


# --- classify: fail-closed + tier map ---

def test_classify_unknown_is_t4(cp):
    assert cp.classify("totally_unknown_tool") == cp.Tier.T4
    assert cp.classify("") == cp.Tier.T4
    assert cp.classify("some_mcp_server.do_thing") == cp.Tier.T4


def test_classify_reads_are_t0(cp):
    for t in ("read_file", "grep", "list_directory", "session_search", "todo"):
        assert cp.classify(t) == cp.Tier.T0, t


def test_classify_egress_is_t1(cp):
    for t in ("web_search", "web_extract", "browser_navigate", "ask_claude"):
        assert cp.classify(t) == cp.Tier.T1, t


def test_classify_messages_are_t3(cp):
    assert cp.classify("send_message") == cp.Tier.T3


def test_classify_delegate_task_is_t4(cp):
    # I5 — the agent-on-agent surface MUST be T4.
    assert cp.classify("delegate_task") == cp.Tier.T4


def test_classify_host_writes_are_t4(cp):
    for t in ("write_file", "patch", "delete_file"):
        assert cp.classify(t) == cp.Tier.T4, t


def test_classify_exec_tier_depends_on_backend(cp):
    assert cp.classify("execute_code", ctx={"backend": "sylva-sandbox"}) == cp.Tier.T2
    assert cp.classify("terminal", ctx={"backend": "local"}) == cp.Tier.T4
    assert cp.classify("terminal", ctx={"backend": "docker"}) == cp.Tier.T4


# --- guard: observe mode never blocks ---

def test_observe_mode_never_blocks(cp, monkeypatch):
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "observe")
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")  # unattended
    for t in ("write_file", "delegate_task", "totally_unknown", "terminal"):
        g = cp.guard(t, ctx={"backend": "local"})
        assert g["allowed"] is True, t
        assert g["outcome"] == "observed"


# --- guard: enforce mode ---

def test_enforce_attended_allows_t4(cp, monkeypatch):
    # Attended (no cron/autonomous marker): existing approval gate handles T4.
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    g = cp.guard("write_file", ctx={"unattended": False})
    assert g["allowed"] is True
    assert g["outcome"] == "allowed"


def test_enforce_unattended_denies_t4(cp, monkeypatch):
    # T4 unattended degrades to ask (R4): not executed, queued for one-shot approval.
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    for t in ("write_file", "delegate_task", "unknown_tool"):
        g = cp.guard(t, ctx={"unattended": True, "backend": "local"})
        assert g["allowed"] is False, t
        assert g["outcome"] == "blocked_queued", t
        assert g["tier"] == "T4"


def test_enforce_unattended_allows_t0(cp, monkeypatch):
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    g = cp.guard("read_file", ctx={"unattended": True})
    assert g["allowed"] is True


def test_enforce_unattended_sandbox_exec_allowed(cp, monkeypatch):
    # T2 contained exec is allowed unattended (within budget).
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    g = cp.guard("execute_code", ctx={"unattended": True, "backend": "sylva-sandbox"})
    assert g["allowed"] is True
    assert g["tier"] == "T2"


def test_enforce_unattended_killswitch_halts_t1(cp, monkeypatch):
    monkeypatch.setenv("HERMES_CAPABILITY_POLICY_MODE", "enforce")
    import tools.capability_policy as m
    monkeypatch.setattr("autonomy.killswitch.is_quiesced", lambda: True)
    g = m.guard("web_search", ctx={"unattended": True})
    assert g["allowed"] is False
    assert "kill-switch" in (g["reason"] or "")


def test_deny_result_is_json_error(cp):
    g = {"allowed": False, "tier": "T4", "reason": "nope", "outcome": "blocked"}
    import json
    out = json.loads(cp.deny_result("write_file", g))
    assert "BLOCKED by capability policy" in out["error"]
    assert out["capability_tier"] == "T4"


# --- R5: path-aware write classification ---

def test_memory_writes_are_t2(cp):
    # Managed memory files (MEMORY.md/USER.md) stay T2 (contained own-store write).
    assert cp.classify("memory") == cp.Tier.T2


def test_mem0_reads_removed_from_t0(cp):
    # PRD-037 FR-4 / AC-008: the mem0_* read tools (mem0_search/mem0_list) were
    # retired in the PRD-029 decommission — the provider advertises only
    # chronicle_search now. Their stale T0 entries were removed, so they fall
    # through to default-deny T4 (a removed tool must never stay allow-broadly).
    assert cp.classify("mem0_search") == cp.Tier.T4
    assert cp.classify("mem0_list") == cp.Tier.T4
    # chronicle_search — the live recall tool — remains the T0 read.
    assert cp.classify("chronicle_search") == cp.Tier.T0


def test_mem0_writes_are_t4(cp):
    # PRD-029: mem0_conclude is gone. Its successor mem0_add and the new
    # destructive mem0_update/mem0_delete verbs are NOT in _MEMORY_WRITE_TOOLS,
    # so they fall through to default-deny T4 — governed pipeline only, no
    # unattended autosave. (mem0_conclude itself, now an unknown tool, is T4.)
    assert cp.classify("mem0_add") == cp.Tier.T4
    assert cp.classify("mem0_update") == cp.Tier.T4
    assert cp.classify("mem0_delete") == cp.Tier.T4
    assert cp.classify("mem0_conclude") == cp.Tier.T4


def test_write_under_allowed_root_is_t2(cp, monkeypatch, tmp_path):
    allowed = tmp_path / "work"
    allowed.mkdir()
    monkeypatch.setattr(cp, "_unattended_write_roots", lambda: [cp._resolve(allowed)])
    monkeypatch.setattr(cp, "_forbidden_write_roots", lambda: [])
    t = cp.classify("write_file", {"file_path": str(allowed / "out.md")})
    assert t == cp.Tier.T2


def test_write_outside_allowed_root_is_t4(cp, monkeypatch, tmp_path):
    allowed = tmp_path / "work"
    allowed.mkdir()
    monkeypatch.setattr(cp, "_unattended_write_roots", lambda: [cp._resolve(allowed)])
    monkeypatch.setattr(cp, "_forbidden_write_roots", lambda: [])
    t = cp.classify("write_file", {"file_path": str(tmp_path / "elsewhere.txt")})
    assert t == cp.Tier.T4


def test_write_to_forbidden_root_is_t4_even_if_nested_in_allowed(cp, monkeypatch, tmp_path):
    allowed = tmp_path / "work"
    secrets = allowed / "secrets"
    secrets.mkdir(parents=True)
    monkeypatch.setattr(cp, "_unattended_write_roots", lambda: [cp._resolve(allowed)])
    monkeypatch.setattr(cp, "_forbidden_write_roots", lambda: [cp._resolve(secrets)])
    t = cp.classify("write_file", {"file_path": str(secrets / "x")})
    assert t == cp.Tier.T4  # forbidden wins


def test_write_secret_filename_is_t4(cp, monkeypatch, tmp_path):
    allowed = tmp_path / "work"
    allowed.mkdir()
    monkeypatch.setattr(cp, "_unattended_write_roots", lambda: [cp._resolve(allowed)])
    monkeypatch.setattr(cp, "_forbidden_write_roots", lambda: [])
    for fn in (".env", "auth.json", "id_ed25519", "server.key"):
        assert cp.classify("write_file", {"file_path": str(allowed / fn)}) == cp.Tier.T4, fn


def test_write_no_path_is_t4(cp):
    assert cp.classify("write_file", {}) == cp.Tier.T4


def test_symlink_escape_is_t4(cp, monkeypatch, tmp_path):
    allowed = tmp_path / "work"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = allowed / "escape"
    link.symlink_to(outside)  # allowed/escape -> outside
    monkeypatch.setattr(cp, "_unattended_write_roots", lambda: [cp._resolve(allowed)])
    monkeypatch.setattr(cp, "_forbidden_write_roots", lambda: [])
    # A write "inside" allowed via the symlink resolves outside -> T4.
    t = cp.classify("write_file", {"file_path": str(link / "pwned.txt")})
    assert t == cp.Tier.T4


def test_validated_read_tools_are_t0(cp):
    # Registry-name reads that were mis-defaulting to T4 before the 2026-06-26
    # tier-map validation.
    for t in ("search_files", "skills_list", "skill_view", "kanban_show",
              "kanban_list", "clarify"):
        assert cp.classify(t) == cp.Tier.T0, t


def test_missing_browser_ops_are_t1(cp):
    for t in ("browser_console", "browser_get_images", "browser_press", "browser_vision"):
        assert cp.classify(t) == cp.Tier.T1, t


# --- Site 3: cron script classification ---

def test_cron_script_in_trusted_dir_is_t2(cp, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    s = scripts / "watchdog.sh"
    s.write_text("echo hi")
    assert cp.classify("cron_script", {"script_path": str(s)}) == cp.Tier.T2


def test_cron_script_outside_trusted_dir_is_t4(cp, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "scripts").mkdir()
    outside = tmp_path / "evil.sh"
    outside.write_text("echo pwned")
    assert cp.classify("cron_script", {"script_path": str(outside)}) == cp.Tier.T4
    assert cp.classify("cron_script", {}) == cp.Tier.T4  # no path → fail-closed


# --- Site 4: MCP sampling defaults to T4 (fail-closed) ---

def test_mcp_sampling_is_t4(cp):
    assert cp.classify("mcp_sampling", {"server": "x"}) == cp.Tier.T4


def test_hermes_home_is_forbidden_by_default(cp, monkeypatch, tmp_path):
    # ~/.hermes is always forbidden even when set as an allowed root.
    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    monkeypatch.setattr(cp, "_unattended_write_roots", lambda: [cp._resolve(fake_home)])
    t = cp.classify("write_file", {"file_path": str(fake_home / "config.yaml")})
    assert t == cp.Tier.T4


# --- Most-specific-match carve-out: ~/.hermes/cron/output allowed, rest forbidden ---

def test_cron_output_carveout_is_t2(cp, monkeypatch, tmp_path):
    hh = tmp_path / "dot-hermes"
    (hh / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hh))
    # Use the REAL forbidden/allowed roots (carve-out logic), not monkeypatched.
    t = cp.classify("write_file", {"file_path": str(hh / "cron" / "output" / "skill-audit.md")})
    assert t == cp.Tier.T2  # deeper allowed root wins over forbidden ~/.hermes


def test_hermes_secrets_still_forbidden_despite_carveout(cp, monkeypatch, tmp_path):
    hh = tmp_path / "dot-hermes"
    (hh / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hh))
    # Sibling files in ~/.hermes remain T4 (only cron/output is carved out).
    assert cp.classify("write_file", {"file_path": str(hh / "config.yaml")}) == cp.Tier.T4
    assert cp.classify("write_file", {"file_path": str(hh / "auth.json")}) == cp.Tier.T4
    assert cp.classify("write_file", {"file_path": str(hh / ".env")}) == cp.Tier.T4
    # A secret filename even inside the carved-out dir stays T4 (secret-name wins).
    assert cp.classify("write_file", {"file_path": str(hh / "cron" / "output" / ".env")}) == cp.Tier.T4
