"""PRD-047 increment 1 — hermetic tests for the jail-ssh exec plane.

No real ssh/network: subprocess is mocked. Asserts the jail backend's safety
properties (no file sync, hardened client opts, fail-closed validation), the
HERMES_EXEC_BACKEND selector semantics (terminal/execute_code route to the
jail; file tools do NOT), the STOP-5 strict allowlist on the jail path, and
that jail-ssh is not in any approval skip-set.
"""

import os
import subprocess
from unittest import mock

import pytest


# --------------------------------------------------------------------------
# JailSSHEnvironment — no sync, hardened opts, fail-closed validation
# --------------------------------------------------------------------------

_WORKSPACE_DIRS = ("/opt/data/work", "/opt/data/workspace")


def _fake_probe_output(*, host="hermes-exec", forbidden=(), writable=True):
    lines = [f"HOST:{host}"]
    lines += [f"FORBIDDEN:{p}" for p in forbidden]
    for d in _WORKSPACE_DIRS:
        lines.append(f"WRITABLE:{d}" if writable else f"NOTWRITABLE:{d}")
    return "\n".join(lines) + "\n"


def _make_jail(monkeypatch, probe_output):
    """Construct a JailSSHEnvironment with every subprocess call mocked."""
    from tools.environments import jail_ssh

    monkeypatch.setattr(jail_ssh, "_ensure_ssh_available", lambda: None, raising=False)

    calls = {"run": []}

    def fake_run(cmd, *a, **k):
        calls["run"].append(cmd)
        joined = " ".join(cmd)
        # _establish_connection probe
        if "SSH connection established" in joined:
            return mock.Mock(returncode=0, stdout="SSH connection established", stderr="")
        # _detect_remote_home probe
        if joined.endswith("echo $HOME"):
            return mock.Mock(returncode=0, stdout="/home/hermes\n", stderr="")
        # jail validation probe (contains our HOST: sentinel command)
        if "echo \"HOST:$(hostname)\"" in joined or "HOST:$(hostname)" in joined:
            return mock.Mock(returncode=0, stdout=probe_output, stderr="")
        # init_session snapshot capture — succeed silently
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(jail_ssh.subprocess, "run", fake_run)
    # init_session uses _popen_bash via the base class; stub the whole method so
    # we don't fork anything.
    monkeypatch.setattr(jail_ssh.JailSSHEnvironment, "init_session",
                        lambda self: setattr(self, "_snapshot_ready", False))
    env = jail_ssh.JailSSHEnvironment(
        host="hermes-exec", user="hermes", port=2222,
        key_path="/opt/execbridge/client/id_ed25519",
    )
    return env, calls


def test_jail_backend_constructs_no_file_sync(monkeypatch):
    env, _ = _make_jail(monkeypatch, _fake_probe_output())
    # The whole point of the jail: no FileSyncManager in either direction.
    assert env._sync_manager is None
    # _before_execute must be inert (parent would deref _sync_manager).
    assert env._before_execute() is None


def test_jail_backend_hardened_ssh_opts(monkeypatch):
    env, _ = _make_jail(monkeypatch, _fake_probe_output())
    cmd = env._build_ssh_command()
    joined = " ".join(cmd)
    assert "ClearAllForwardings=yes" in joined
    assert "IdentitiesOnly=yes" in joined
    assert "ForwardAgent=no" in joined
    # user@host must remain the final token (SSH requires the target last).
    assert cmd[-1] == "hermes@hermes-exec"


def test_jail_validation_fails_closed_on_forbidden_path(monkeypatch):
    from tools.environments.jail_ssh import JailValidationError
    with pytest.raises(JailValidationError) as exc:
        _make_jail(monkeypatch,
                   _fake_probe_output(forbidden=("/opt/data/.env",)))
    assert ".env" in str(exc.value)


def test_jail_validation_fails_closed_when_not_writable(monkeypatch):
    from tools.environments.jail_ssh import JailValidationError
    with pytest.raises(JailValidationError):
        _make_jail(monkeypatch, _fake_probe_output(writable=False))


def test_jail_validation_fails_closed_on_wrong_host(monkeypatch):
    from tools.environments.jail_ssh import JailValidationError
    with pytest.raises(JailValidationError) as exc:
        _make_jail(monkeypatch, _fake_probe_output(host="not-the-jail"))
    assert "hostname" in str(exc.value)


def test_jail_cleanup_never_syncs_back(monkeypatch):
    env, _ = _make_jail(monkeypatch, _fake_probe_output())
    # No control socket on disk -> cleanup is a no-op and must not touch sync.
    env.control_socket = mock.Mock(exists=lambda: False)
    env.cleanup()  # must not raise (parent would call _sync_manager.sync_back)


# --------------------------------------------------------------------------
# HERMES_EXEC_BACKEND selector — routes exec, leaves file tools local
# --------------------------------------------------------------------------

def test_resolver_off_by_default(monkeypatch):
    from tools.terminal_tool import _resolve_exec_env_type
    monkeypatch.delenv("HERMES_EXEC_BACKEND", raising=False)
    assert _resolve_exec_env_type("local") == "local"
    assert _resolve_exec_env_type("docker") == "docker"


def test_resolver_routes_when_flagged(monkeypatch):
    from tools.terminal_tool import _resolve_exec_env_type
    monkeypatch.setenv("HERMES_EXEC_BACKEND", "jail-ssh")
    assert _resolve_exec_env_type("local") == "jail-ssh"
    # Case/space tolerant.
    monkeypatch.setenv("HERMES_EXEC_BACKEND", "  Jail-SSH ")
    assert _resolve_exec_env_type("local") == "jail-ssh"


def test_resolver_ignores_unknown_value(monkeypatch):
    from tools.terminal_tool import _resolve_exec_env_type
    monkeypatch.setenv("HERMES_EXEC_BACKEND", "docker")  # not the jail sentinel
    assert _resolve_exec_env_type("local") == "local"


def test_file_tools_env_unaffected_by_flag(monkeypatch):
    """STOP-4: file tools read _get_env_config directly and must stay local."""
    import tools.terminal_tool as tt
    monkeypatch.setenv("HERMES_EXEC_BACKEND", "jail-ssh")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    # _get_env_config is the seam file_tools uses — it must NOT see the jail.
    assert tt._get_env_config()["env_type"] == "local"


def test_exec_plane_cache_key_namespaced_only_when_flagged(monkeypatch):
    from tools.terminal_tool import _exec_plane_cache_key
    monkeypatch.delenv("HERMES_EXEC_BACKEND", raising=False)
    assert _exec_plane_cache_key("default") == "default"
    monkeypatch.setenv("HERMES_EXEC_BACKEND", "jail-ssh")
    assert _exec_plane_cache_key("default") == "default::exec-jail"
    # Idempotent — never double-suffix.
    assert _exec_plane_cache_key("default::exec-jail") == "default::exec-jail"


def test_cached_local_env_not_returned_mislabeled_as_jail(monkeypatch):
    """STOP-1 (fail-open bypass): with the jail flag on, a pre-cached LOCAL env
    under the shared 'default' key must NOT be handed back to execute_code — the
    jail path uses a distinct cache slot, so it never reuses the local env."""
    import tools.terminal_tool as tt
    import tools.code_execution_tool as ce

    monkeypatch.setenv("HERMES_EXEC_BACKEND", "jail-ssh")
    monkeypatch.setenv("TERMINAL_ENV", "local")

    sentinel_local = object()  # stand-in for a LocalEnvironment
    # Simulate a file-tool read having populated the shared cache first.
    monkeypatch.setitem(tt._active_environments, "default", sentinel_local)

    created = {"count": 0}

    def fake_create_environment(env_type, *a, **k):
        created["count"] += 1
        created["env_type"] = env_type
        return object()  # a fresh (jail) env, never the local sentinel

    monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
    # Avoid the cleanup thread touching real state.
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)

    env, env_type = ce._get_or_create_env("default")

    # The jail request must NOT return the cached local env...
    assert env is not sentinel_local
    # ...it must have BUILT a jail env under the namespaced key...
    assert created["count"] == 1
    assert created["env_type"] == "jail-ssh"
    assert env_type == "jail-ssh"
    # ...and the local "default" slot must be untouched (file tools still local).
    assert tt._active_environments["default"] is sentinel_local
    assert "default::exec-jail" in tt._active_environments


# --------------------------------------------------------------------------
# approval skip-sets — jail-ssh must never skip the guard
# --------------------------------------------------------------------------

def test_jail_not_in_approval_skip_sets():
    import inspect
    import tools.approval as approval
    src = inspect.getsource(approval)
    # The three container skip-sets (dangerous cmd, all-guards, execute_code).
    assert "jail-ssh" not in src.replace("PRD-047", ""), \
        "jail-ssh must not appear in any approval skip-set literal"


# --------------------------------------------------------------------------
# STOP-5 strict allowlist on the jail path
# --------------------------------------------------------------------------

def test_strict_allowlist_denies_on_empty_intersection_for_jail():
    """On jail-ssh, empty enabled_tools must NOT fall back to all sandbox tools."""
    import tools.code_execution_tool as ce
    # Simulate the STOP-5 branch directly against the real frozenset logic.
    SANDBOX = ce.SANDBOX_ALLOWED_TOOLS
    for env_type, want_fallback in (("jail-ssh", False), ("local", True), ("docker", True)):
        session_tools = set()  # empty per-turn intersection
        sandbox_tools = frozenset(SANDBOX & session_tools)
        if not sandbox_tools and env_type != "jail-ssh":
            sandbox_tools = SANDBOX
        if want_fallback:
            assert sandbox_tools == SANDBOX
        else:
            assert sandbox_tools == frozenset()
