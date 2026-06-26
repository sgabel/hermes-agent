"""Sylva-sandbox execution backend (PRD-026 FR-7).

A dedicated, *locked* execution backend that runs Sylva's code/commands inside
the verified ``sylva-sandbox`` compose stack (``~/hermes/sandbox/``), NOT via the
generic ``docker`` backend.

Why a separate backend (Codex STOP-B, AGENT_SECURITY_MODEL.md §7a gap 3):
the generic ``tools/environments/docker.py`` backend is **not** a safe T2
substrate — it defaults network-on, mounts host credentials/cache/skills
read-only, appends arbitrary ``docker_extra_args`` last, and the approval gate
*skips* container backends entirely (``approval.py`` container skip-set). This
backend instead:

  * ``docker exec``s into the long-lived ``sylva-sandbox-worker`` container that
    already runs on the ``internal: true`` bridge with the tinyproxy egress
    allowlist — so routine execution physically cannot reach the host, the LAN,
    FedPulse, or the docker socket;
  * mounts NOTHING new — no creds, no cache, no skills, no ``docker_extra_args``,
    no ``docker.sock``;
  * is **deliberately absent from the approval container skip-set** so the
    tirith / manual-approval gate still fires (AC-012);
  * **validates the target container's identity at startup** — resolved
    immutable ID + compose project/service labels + sole-network ==
    ``sylva-sandbox-internal`` (and that network is ``internal: true``) + no
    docker.sock / host-secret mounts + ``no-new-privileges`` + non-privileged —
    and **refuses to start (fail-closed) if any invariant fails** (I9: classify
    and bind to the resolved target, never a spoofable name).

The compose stack is long-lived and owned by ``docker compose -p sylva-sandbox``;
this backend NEVER creates, starts, stops, or removes the worker — ``cleanup()``
is a no-op so a per-task teardown can never reach the shared container.
"""

import json
import logging
import os
import subprocess
from pathlib import Path

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.docker import find_docker

logger = logging.getLogger(__name__)

# Compose-stack invariants the worker MUST satisfy (PRD-026 FR-7 / AC-020).
_PROJECT_LABEL = "com.docker.compose.project"
_SERVICE_LABEL = "com.docker.compose.service"
_EXPECTED_PROJECT = "sylva-sandbox"
_EXPECTED_SERVICE = "worker"
_EXPECTED_CONTAINER = "sylva-sandbox-worker"
_EXPECTED_NETWORK = "sylva-sandbox-internal"
_WORKDIR = "/work"

# Host paths that must NEVER be bind-mounted into the worker (defence in depth on
# top of the compose definition — a manual edit / out-of-band container with the
# same name must not slip a credential or the docker socket in).
_DOCKER_SOCK = "/var/run/docker.sock"


class SylvaSandboxBackendError(RuntimeError):
    """Raised when the sylva-sandbox worker is missing or fails validation.

    Fail-closed: when this is raised, no command has run — the backend refused
    to attach to an unverified target.
    """


class SylvaSandboxEnvironment(BaseEnvironment):
    """Run commands inside the verified ``sylva-sandbox-worker`` container.

    Resolves + validates the worker once at construction, binds to its immutable
    container ID, and ``docker exec``s every command into it. No new mounts, no
    creds, no docker.sock, no extra args.
    """

    def __init__(self, cwd: str = _WORKDIR, timeout: int = 180, **_ignored):
        # Always run in the scratch workdir (the only writable bind). Ignore any
        # host/`/root` cwd the factory passed — host paths are meaningless here.
        super().__init__(cwd=_WORKDIR, timeout=timeout)
        self._docker_exe = find_docker() or "docker"
        # Resolve + validate the target. Raises (fail-closed) on any violation.
        self._container_id = self._resolve_and_validate()
        self.init_session()

    # ------------------------------------------------------------------
    # Target resolution + invariant validation (I9 — bind to resolved target)
    # ------------------------------------------------------------------

    def _inspect(self, kind: str, name: str) -> dict:
        """``docker inspect`` *name* (a container or network); return parsed JSON."""
        try:
            result = subprocess.run(
                [self._docker_exe, kind, "inspect", name],
                capture_output=True, text=True, timeout=20,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise SylvaSandboxBackendError(
                f"docker {kind} inspect {name} failed: {exc}"
            ) from exc
        if result.returncode != 0:
            raise SylvaSandboxBackendError(
                f"docker {kind} inspect {name} failed (rc={result.returncode}): "
                f"{result.stderr.strip() or 'not found'}. Is the sandbox up? "
                f"Run: docker compose -p {_EXPECTED_PROJECT} up -d"
            )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise SylvaSandboxBackendError(
                f"could not parse docker {kind} inspect {name}: {exc}"
            ) from exc
        if not data:
            raise SylvaSandboxBackendError(
                f"docker {kind} inspect {name} returned no object"
            )
        return data[0]

    def _resolve_and_validate(self) -> str:
        """Resolve the worker to an immutable ID and assert every invariant.

        Returns the container ID on success; raises SylvaSandboxBackendError
        (fail-closed) on any violation so an unverified target never executes.
        """
        info = self._inspect("container", _EXPECTED_CONTAINER)
        cid = info.get("Id") or ""
        if not cid:
            raise SylvaSandboxBackendError("worker container has no Id")

        state = info.get("State") or {}
        if not state.get("Running"):
            raise SylvaSandboxBackendError(
                f"{_EXPECTED_CONTAINER} is not running (status="
                f"{state.get('Status')!r}). Run: "
                f"docker compose -p {_EXPECTED_PROJECT} up -d"
            )

        config = info.get("Config") or {}
        labels = config.get("Labels") or {}
        if labels.get(_PROJECT_LABEL) != _EXPECTED_PROJECT:
            raise SylvaSandboxBackendError(
                f"worker compose project label is {labels.get(_PROJECT_LABEL)!r}, "
                f"expected {_EXPECTED_PROJECT!r} — refusing (possible name-spoof)"
            )
        if labels.get(_SERVICE_LABEL) != _EXPECTED_SERVICE:
            raise SylvaSandboxBackendError(
                f"worker compose service label is {labels.get(_SERVICE_LABEL)!r}, "
                f"expected {_EXPECTED_SERVICE!r} — refusing"
            )

        host_config = info.get("HostConfig") or {}
        if host_config.get("Privileged"):
            raise SylvaSandboxBackendError("worker is privileged — refusing")
        sec_opt = host_config.get("SecurityOpt") or []
        if not any("no-new-privileges:true" in s for s in sec_opt):
            raise SylvaSandboxBackendError(
                "worker lacks no-new-privileges:true — refusing"
            )

        # Network: the worker must be attached ONLY to the internal bridge, and
        # that bridge must be internal:true (no route off the box / to FedPulse).
        networks = (info.get("NetworkSettings") or {}).get("Networks") or {}
        net_names = set(networks)
        if net_names != {_EXPECTED_NETWORK}:
            raise SylvaSandboxBackendError(
                f"worker networks are {sorted(net_names)}, expected exactly "
                f"[{_EXPECTED_NETWORK!r}] — refusing (extra network = egress hole)"
            )
        net_info = self._inspect("network", _EXPECTED_NETWORK)
        if not net_info.get("Internal"):
            raise SylvaSandboxBackendError(
                f"network {_EXPECTED_NETWORK} is not internal:true — refusing "
                "(workload could reach the internet / LAN directly)"
            )

        # Mounts: no docker.sock, no host secrets. Resolve real paths (I9 —
        # defeat symlink games) before comparing.
        self._assert_safe_mounts(info.get("Mounts") or [])

        logger.info(
            "sylva-sandbox backend bound to worker %s (project=%s service=%s "
            "net=%s internal=true)",
            cid[:12], _EXPECTED_PROJECT, _EXPECTED_SERVICE, _EXPECTED_NETWORK,
        )
        return cid

    @staticmethod
    def _assert_safe_mounts(mounts: list) -> None:
        """Reject docker.sock and any mount of a host-secret directory."""
        hermes_home = Path(
            os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
        ).resolve()
        forbidden_files = {Path(_DOCKER_SOCK)}
        for m in mounts:
            src = (m.get("Source") or "").strip()
            if not src:
                continue
            try:
                rsrc = Path(src).resolve()
            except (OSError, RuntimeError):
                rsrc = Path(src)
            # The docker socket = full host control. Never.
            if rsrc in forbidden_files or rsrc.name == "docker.sock":
                raise SylvaSandboxBackendError(
                    f"worker mounts the docker socket ({src}) — refusing"
                )
            # ~/.hermes holds .env / auth.json / config.yaml — the credential
            # store. A sandbox that can read it defeats the point.
            try:
                if rsrc == hermes_home or hermes_home in rsrc.parents:
                    raise SylvaSandboxBackendError(
                        f"worker mounts a host-secret path ({src}) under "
                        f"{hermes_home} — refusing"
                    )
            except SylvaSandboxBackendError:
                raise
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def get_temp_dir(self) -> str:
        return "/tmp"

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn a bash process inside the worker container via ``docker exec``.

        Bound to the resolved immutable container ID (not the name) so the target
        cannot be swapped between validation and use (I9 / TOCTOU). No ``-e`` env
        injection: the worker carries only what the compose file gave it.
        """
        assert self._container_id, "worker container not resolved"
        cmd = [self._docker_exe, "exec"]
        if stdin_data is not None:
            cmd.append("-i")
        cmd.append(self._container_id)
        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])
        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
        """No-op: the worker is a long-lived, compose-owned shared container.

        Per-task teardown must NEVER ``docker rm``/``stop`` the shared worker
        (PRD-026 R1 / AC-008/009 — only ``docker compose -p sylva-sandbox down``
        may tear the stack down). We just drop our reference.
        """
        self._container_id = ""
