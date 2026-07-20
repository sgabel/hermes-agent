"""PRD-047 increment 2 — Hermes governance/identity/credential write guard.

The exec-plane jail (increment 1) closed the EXEC route to Hermes secrets and
governance state; file tools run gateway-side, so a prompt-injected agent could
still rewrite SOUL.md / autonomy/ / cron/jobs.json via write_file. These tests
verify BOTH write guards now block those paths while leaving reads and normal
workspace writes untouched.

Hermetic: HERMES_HOME points at a throwaway tmp dir — no real files touched.
"""

import importlib
import os

import pytest


@pytest.fixture
def fake_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "cron").mkdir(parents=True)
    (home / "autonomy").mkdir(parents=True)
    (home / "SOUL.md").write_text("identity")
    (home / "auth.json").write_text("{}")
    (home / ".env").write_text("SECRET=x")
    (home / "mem0.json").write_text("{}")
    (home / "cron" / "jobs.json").write_text("[]")
    (home / "autonomy" / "QUIESCE").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Reset caches that resolve HERMES_HOME so the env var takes effect.
    import tools.file_tools as ft
    ft._hermes_protected_write = None
    ft._hermes_config_resolved = None
    ft._hermes_config_resolved_loaded = False
    return home


# ── file_tools._check_sensitive_path (first-hit write guard) ─────────────────

@pytest.mark.parametrize("rel", [
    "SOUL.md",
    ".env",
    "mem0.json",
    "cron/jobs.json",
    "autonomy/QUIESCE",
    "autonomy/budget-20260720.json",
    "autonomy/audit.jsonl",
])
def test_check_sensitive_path_blocks_governance_identity(fake_hermes_home, rel):
    import tools.file_tools as ft
    target = str(fake_hermes_home / rel)
    err = ft._check_sensitive_path(target)
    assert err is not None, f"{rel} should be write-blocked"
    assert "protected Hermes state" in err or "config" in err


def test_check_sensitive_path_allows_normal_workspace_write(fake_hermes_home, tmp_path):
    import tools.file_tools as ft
    ok = tmp_path / "work" / "notes.md"
    ok.parent.mkdir(parents=True)
    assert ft._check_sensitive_path(str(ok)) is None


def test_check_sensitive_path_allows_workspace_under_hermes(fake_hermes_home):
    # A workspace file under HERMES_HOME (not a protected name) stays writable.
    import tools.file_tools as ft
    assert ft._check_sensitive_path(str(fake_hermes_home / "work" / "scratch.txt")) is None


def test_soul_md_reads_are_not_blocked(fake_hermes_home):
    # The guard is WRITE-only — it must never appear on the read path. The read
    # block list (get_read_block_error) does not contain SOUL.md, so the
    # self-brief can still read identity.
    from agent.file_safety import get_read_block_error
    assert get_read_block_error(str(fake_hermes_home / "SOUL.md")) is None


def test_traversal_to_protected_path_is_caught(fake_hermes_home):
    import tools.file_tools as ft
    sneaky = str(fake_hermes_home / "work" / ".." / "SOUL.md")
    assert ft._check_sensitive_path(sneaky) is not None


# ── file_safety.is_write_denied (second write path) ─────────────────────────

@pytest.mark.parametrize("rel", [
    "SOUL.md", "mem0.json", "cron/jobs.json",
    "autonomy/QUIESCE", "autonomy/audit.jsonl",
])
def test_is_write_denied_blocks_governance_identity(fake_hermes_home, rel):
    from agent import file_safety as fs
    assert fs.is_write_denied(str(fake_hermes_home / rel)) is True


def test_auth_json_stays_writable_control_file(fake_hermes_home):
    # auth.json is a deliberately-writable T4-gated control file (see
    # TestIsWriteDenied.test_control_files_requested_writable). The governance
    # guard must NOT block it — write authority is gated at the capability layer,
    # not here.
    from agent import file_safety as fs
    assert fs.is_write_denied(str(fake_hermes_home / "auth.json")) is False


def test_is_write_denied_allows_normal_file(fake_hermes_home, tmp_path):
    from agent import file_safety as fs
    # Not under any denied prefix/path.
    assert fs.is_write_denied(str(tmp_path / "elsewhere" / "x.txt")) is False
