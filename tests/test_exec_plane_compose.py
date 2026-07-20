"""PRD-047 increment 1 — compose contract test for the hermes-exec jail.

Parses docker-compose.migration.yml and asserts the exec-plane service's
mount/network/profile invariants (the jail-design pass FR-2/FR-4 contract):
it ships dark, mounts only the allowed paths, and joins only the p2p bridge.
"""

import os

import pytest

yaml = pytest.importorskip("yaml")

_COMPOSE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docker-compose.migration.yml",
)


@pytest.fixture(scope="module")
def compose():
    with open(_COMPOSE) as fh:
        return yaml.safe_load(fh)


def _svc(compose, name):
    assert name in compose["services"], f"service {name} missing"
    return compose["services"][name]


def test_exec_plane_ships_dark(compose):
    svc = _svc(compose, "exec-plane")
    # profiles:["exec-plane"] => a plain `up -d` never starts it.
    assert svc.get("profiles") == ["exec-plane"]
    assert svc.get("container_name") == "hermes-exec"
    # Custom entrypoint bypasses /init (stage2 self-seeding — STOP-3).
    assert svc.get("entrypoint") == ["/opt/hermes/docker/exec-plane/run.sh"]


def test_exec_plane_mounts_only_allowed(compose):
    svc = _svc(compose, "exec-plane")
    sources = {v.split(":")[0] for v in svc["volumes"]}
    targets = {v.split(":")[1] for v in svc["volumes"]}

    # FR-2: exactly the allowed mounts, nothing else.
    assert targets == {
        "/opt/data/work",
        "/opt/data/workspace",
        "/opt/data/skills",
        "/opt/relay",
        "/opt/execbridge/server",
    }
    # The forbidden state MUST NOT be mounted (no /opt/data root, no secrets).
    assert "~/.hermes" not in sources
    for bad in ("/opt/data/.env", "/opt/data/auth.json", "/opt/data/config.yaml",
                "/opt/data/autonomy", "/opt/data/cron", "/opt/data"):
        assert bad not in targets

    # Read-only where it must be; workspace rw.
    def mode(target):
        for v in svc["volumes"]:
            parts = v.split(":")
            if parts[1] == target:
                return parts[2] if len(parts) > 2 else "rw"
        return None
    assert mode("/opt/data/skills") == "ro"
    assert mode("/opt/relay") == "ro"
    assert mode("/opt/execbridge/server") == "ro"
    assert mode("/opt/data/work") == "rw"
    assert mode("/opt/data/workspace") == "rw"


def test_exec_plane_network_is_p2p_only(compose):
    svc = _svc(compose, "exec-plane")
    # ONLY the p2p bridge — not deps, not egress.
    assert svc["networks"] == ["hermes-exec-bridge"]
    assert "hermes-agent-deps" not in svc["networks"]
    assert "hermes-egress" not in svc["networks"]
    # No published ports (no host/LAN exposure).
    assert "ports" not in svc


def test_exec_bridge_is_internal(compose):
    nets = compose["networks"]
    assert "hermes-exec-bridge" in nets
    assert nets["hermes-exec-bridge"].get("internal") is True


def test_gateway_joins_bridge_and_holds_client_key(compose):
    gw = _svc(compose, "gateway")
    assert "hermes-exec-bridge" in gw["networks"]
    # The gateway holds the SSH CLIENT key (a credential INTO the jail),
    # mounted ro and outside /opt/data (chown-lock-safe).
    client_mounts = [v for v in gw["volumes"] if "/opt/execbridge/client" in v]
    assert client_mounts, "gateway must mount the exec-bridge client key"
    assert client_mounts[0].endswith(":ro")
    # The gateway must NOT mount the jail's server key/authorized_keys.
    assert not any("/opt/execbridge/server" in v for v in gw["volumes"])


def test_exec_plane_has_resource_caps(compose):
    svc = _svc(compose, "exec-plane")
    assert "mem_limit" in svc
    assert "pids_limit" in svc
