"""Jail SSH execution backend (PRD-047 FR-1/FR-2/FR-5).

The exec-plane transport for the ``hermes-exec`` jail. A hardened subclass of
:class:`SSHEnvironment` that is SAFE to point at the isolated exec container —
where the stock backend is NOT.

Why not stock ``SSHEnvironment`` (jail-design adversarial pass, 2026-07-20,
STOP-1): the stock backend always wires a :class:`FileSyncManager`
(``ssh.py`` ``__init__``) that **uploads credentials, skills, and cache** into
the remote (``file_sync.iter_sync_files`` -> ``credential_files``) and **syncs
remote changes back to the host** on ``cleanup()``. Both directions defeat the
whole point of the exec plane: the split exists so the jail holds NO secrets and
cannot write host state. This subclass therefore:

  * constructs **no** ``FileSyncManager`` — ``self._sync_manager is None``;
  * overrides ``_before_execute`` (no per-command sync) and ``cleanup`` (no
    ``sync_back``) so the parent's sync hooks are inert;
  * never calls ``_ensure_remote_dirs`` (which would seed ``~/.hermes`` creds
    dirs on the remote);
  * hardens the client invocation (``ClearAllForwardings=yes``,
    ``IdentitiesOnly=yes``) on top of the parent's ``BatchMode``/host-key opts;
  * runs a **fail-closed startup jail validation** (sylva_sandbox precedent):
    the forbidden secret/governance paths MUST be absent and the workspace
    writable, or ``__init__`` raises :class:`JailValidationError` before any
    command can run.

The jail's mounts (workspace rw, skills ro, relay ro) provide everything the
plane legitimately needs — nothing is synced over SSH. File movement between
planes happens through the shared workspace bind-mount, never this channel
(sshd also has the sftp subsystem disabled).
"""

import logging
import subprocess

from tools.environments.ssh import SSHEnvironment

logger = logging.getLogger(__name__)

# Paths that MUST NOT be visible inside the jail namespace. Presence of any
# means the mount contract (PRD-047 FR-2) was violated — fail closed rather
# than run agent code in a plane that can read secrets or write governance.
_FORBIDDEN_PATHS = (
    "/opt/data/.env",
    "/opt/data/auth.json",
    "/opt/data/config.yaml",
    "/opt/data/mem0.json",
    "/opt/data/SOUL.md",
    "/opt/data/state.db",
    "/opt/data/autonomy",
    "/opt/data/cron/jobs.json",
    "/opt/data/sessions",
)
# Both workspace dirs must exist and be writable (positive invariant, FR-2).
_REQUIRED_WRITABLE = ("/opt/data/work", "/opt/data/workspace")
_EXPECTED_HOSTNAME = "hermes-exec"


class JailValidationError(RuntimeError):
    """Raised when the exec-plane target fails its jail invariants.

    Fail-closed: when raised, no agent command has run — the backend refused to
    bind to a plane that does not satisfy the PRD-047 FR-2 mount contract.
    """


class JailSSHEnvironment(SSHEnvironment):
    """Hardened, no-sync SSH backend for the hermes-exec jail plane."""

    def __init__(self, host: str, user: str, cwd: str = "/opt/data/work",
                 timeout: int = 60, port: int = 2222, key_path: str = ""):
        # Deliberately DO NOT call super().__init__ — it constructs the
        # FileSyncManager and seeds remote creds dirs. Re-implement the safe
        # subset: connection + session snapshot, no sync of any kind.
        from pathlib import Path
        import hashlib
        import tempfile

        # BaseEnvironment.__init__ (via the grandparent) sets cwd/timeout and
        # the execute() plumbing; call it directly, skipping SSHEnvironment.
        from tools.environments.base import BaseEnvironment
        BaseEnvironment.__init__(self, cwd=cwd, timeout=timeout)

        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        _socket_id = hashlib.sha256(
            f"{user}@{host}:{port}".encode()
        ).hexdigest()[:16]
        self.control_socket = self.control_dir / f"{_socket_id}.sock"

        # No FileSyncManager — the jail holds no secrets and syncs nothing.
        self._sync_manager = None

        from tools.environments.ssh import _ensure_ssh_available
        _ensure_ssh_available()
        self._establish_connection()
        self._remote_home = self._detect_remote_home()

        # Fail-closed jail validation BEFORE the first agent command.
        self._validate_jail()

        # No _ensure_remote_dirs (that seeds ~/.hermes/{credentials,cache} on
        # the remote); the workspace mounts already exist.
        self.init_session()

    def _build_ssh_command(self, extra_args: list | None = None) -> list:
        """Parent opts + jail hardening (no forwarding, pinned host key)."""
        import os

        cmd = super()._build_ssh_command(extra_args=None)
        # Insert hardening before the user@host terminal token (last element).
        target = cmd.pop()
        cmd.extend(["-o", "ClearAllForwardings=yes"])
        cmd.extend(["-o", "IdentitiesOnly=yes"])
        cmd.extend(["-o", "ForwardAgent=no"])
        # FR-5 host-key pinning: if provisioning wrote a known_hosts alongside
        # the client key, pin it (StrictHostKeyChecking=yes overrides the
        # parent's accept-new TOFU). If absent (pre-provisioning / tests), fall
        # back to the parent's accept-new so first-arming is not bricked.
        known_hosts = os.path.join(
            os.path.dirname(self.key_path or "/opt/execbridge/client/id_ed25519"),
            "known_hosts",
        )
        if os.path.isfile(known_hosts):
            cmd.extend(["-o", f"UserKnownHostsFile={known_hosts}"])
            cmd.extend(["-o", "StrictHostKeyChecking=yes"])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(target)
        return cmd

    def _validate_jail(self) -> None:
        """Assert the FR-2 mount contract on the live target; fail closed."""
        # One remote probe: emit hostname, then a token per present-forbidden
        # path, then WRITABLE/NOTWRITABLE for the workspace.
        checks = " ; ".join(
            f'[ -e {p} ] && echo "FORBIDDEN:{p}"' for p in _FORBIDDEN_PATHS
        )
        writable_checks = " ; ".join(
            f'( touch {d}/.hermes-jail-probe 2>/dev/null '
            f'&& rm -f {d}/.hermes-jail-probe '
            f'&& echo "WRITABLE:{d}" ) || echo "NOTWRITABLE:{d}"'
            for d in _REQUIRED_WRITABLE
        )
        probe = (
            'echo "HOST:$(hostname)"; '
            f'{checks} ; '
            f'{writable_checks}'
        )
        cmd = self._build_ssh_command()
        cmd.append(probe)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.SubprocessError as exc:
            raise JailValidationError(
                f"jail validation probe failed to run: {exc}"
            ) from exc

        # Fail closed if the probe itself errored (defence in depth on top of
        # the positive-sentinel requirement below).
        if result.returncode != 0:
            raise JailValidationError(
                f"jail validation probe exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:200]}"
            )
        out = result.stdout or ""
        out_lines = out.splitlines()
        violations = [
            line.split("FORBIDDEN:", 1)[1]
            for line in out_lines if line.startswith("FORBIDDEN:")
        ]
        if violations:
            raise JailValidationError(
                "exec-plane exposes forbidden paths (mount contract violated): "
                + ", ".join(violations)
            )
        # Require an explicit WRITABLE:<dir> line for every required dir. Exact
        # per-dir match — "NOTWRITABLE:x" must not satisfy "WRITABLE:x".
        for d in _REQUIRED_WRITABLE:
            if f"WRITABLE:{d}" not in out_lines:
                raise JailValidationError(
                    f"exec-plane workspace {d} is not writable (jail misconfigured)"
                )
        host_line = next(
            (l for l in out.splitlines() if l.startswith("HOST:")), ""
        )
        remote_host = host_line.split("HOST:", 1)[1].strip() if host_line else ""
        if remote_host != _EXPECTED_HOSTNAME:
            raise JailValidationError(
                f"exec-plane hostname {remote_host!r} != {_EXPECTED_HOSTNAME!r} "
                "(bound to the wrong target)"
            )
        logger.info("jail-ssh: validated exec plane %s@%s:%s",
                    self.user, self.host, self.port)

    def _before_execute(self) -> None:
        """No-op: the jail syncs nothing (parent would call _sync_manager)."""
        return None

    def cleanup(self):
        """Tear down the control socket only — never sync_back from the jail."""
        if self.control_socket.exists():
            try:
                cmd = ["ssh", "-o", f"ControlPath={self.control_socket}",
                       "-O", "exit", f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5,
                               stdin=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                self.control_socket.unlink()
            except OSError:
                pass
