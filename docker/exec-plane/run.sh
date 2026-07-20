#!/bin/sh
# PRD-047 FR-5 — hermes-exec entrypoint. DELIBERATELY bypasses /init (s6) and
# stage2-hook.sh: the stock boot self-seeds /opt/data/.env, config.yaml,
# SOUL.md, and skills into the container namespace (jail-pass STOP-3), which
# would violate the jail's forbidden-state invariant (AC-007). This entrypoint
# starts hardened sshd and NOTHING else — it creates no files under /opt/data.
#
# Runs as the compose-pinned non-root user; sshd binds 2222 (unprivileged) and
# only same-user logins are possible (AllowUsers hermes, pubkey-only).
set -eu

CONF=/opt/hermes/docker/exec-plane/sshd_config

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
