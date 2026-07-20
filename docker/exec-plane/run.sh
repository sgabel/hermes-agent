#!/bin/sh
# PRD-047 FR-5 — hermes-exec entrypoint. DELIBERATELY bypasses /init (s6) and
# stage2-hook.sh: the stock boot self-seeds /opt/data/.env, config.yaml,
# SOUL.md, and skills into the container namespace (jail-design STOP-3), which
# would violate the jail's forbidden-state invariant (AC-007). This entrypoint
# does the ONE thing stage2 does that the jail still needs — remap the `hermes`
# account to the host UID/GID so it matches the shared workspace ownership —
# and starts hardened sshd. It seeds NO files under /opt/data.
#
# Runs as ROOT (container starts as root): root is needed to remap the account
# and for sshd privilege separation (the login session itself drops to
# `hermes`; PermitRootLogin is no). The jail mounts NO secrets/governance and
# has NO network route out, so isolation is enforced by the mount + network +
# PID namespaces, NOT by the process uid.
set -eu

CONF=/opt/hermes/docker/exec-plane/sshd_config

# --- UID/GID remap (the stage2 subset the jail needs; NO secret seeding) ------
# The image bakes `hermes` at a fixed build uid; the shared workspace mounts are
# owned by the host user. Align them so `hermes` can read/write the workspace
# and gateway-side reads see the same owner. Idempotent + guarded.
TARGET_UID="${HERMES_UID:-10000}"
TARGET_GID="${HERMES_GID:-10000}"
CUR_UID="$(id -u hermes 2>/dev/null || echo '')"
CUR_GID="$(id -g hermes 2>/dev/null || echo '')"
if [ -n "$CUR_GID" ] && [ "$CUR_GID" != "$TARGET_GID" ]; then
    groupmod -g "$TARGET_GID" hermes 2>/dev/null || true
fi
if [ -n "$CUR_UID" ] && [ "$CUR_UID" != "$TARGET_UID" ]; then
    usermod -u "$TARGET_UID" -g "$TARGET_GID" hermes 2>/dev/null || true
fi

# The image bakes `hermes` as a LOCKED service account (no password). With
# `UsePAM no`, sshd refuses ALL auth — including pubkey — to a locked account
# ("User hermes not allowed because account is locked"). Clear the password so
# pubkey login is permitted. Password login stays impossible: sshd_config sets
# `PasswordAuthentication no` + `PermitEmptyPasswords no`, and the jail has no
# other login surface. (Pubkey-only isolated service-account pattern.)
passwd -d hermes >/dev/null 2>&1 || true

# sshd privilege-separation dir (root-owned, mode 0755). Standard requirement.
mkdir -p /run/sshd
chmod 0755 /run/sshd

# Fail fast (container restart-loops visibly) if provisioning is missing —
# never generate key material here: scripts/prd047-provision-execbridge.sh
# owns it host-side, mounted ro at /opt/execbridge/server.
for f in /opt/execbridge/server/ssh_host_ed25519_key \
         /opt/execbridge/server/authorized_keys; do
    if [ ! -r "$f" ]; then
        echo "hermes-exec: missing or unreadable $f — run scripts/prd047-provision-execbridge.sh on the host" >&2
        exit 1
    fi
done

exec /usr/sbin/sshd -D -e -f "$CONF"
