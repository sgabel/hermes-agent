#!/usr/bin/env bash
# PRD-033 FR-12 — one-shot, REVERSIBLE cutover to the containerized agent.
# Does NOT run FR-14/FR-15 destructive teardown (deferred to post-soak per Codex ruling).
# Keeps hermes-cockpit and the LLM/embed/Qdrant containers RUNNING. Rollback: see ROLLBACK below.
set -euo pipefail
cd "$(dirname "$0")/../.."                 # hermes-agent/
COMPOSE="docker-compose.migration.yml"
PROJECT="hermes-agent"
DEPS="llama-qwen36-35b llama-qwen3-4b tei-bge-m3 qdrant"
# Only the services that MOVE into containers. hermes-sylva-voice + kokoro-llama-bridge
# STAY host-side (FR-7 + Kokoro-deferral): voice reaches the agent via the host-ingress
# forwarder :8642; kokoro proxies llama on host :8081. hermes-cockpit also stays host-side.
HOST_SERVICES="hermes-gateway hermes-dashboard"

echo "############ PRD-033 CUTOVER ############"
echo "ROLLBACK if anything fails:"
echo "  bash docker/migration/rollback-host-config.sh    # restore localhost URLs"
echo "  bash docker/migration/rollback-network.sh        # compose down + un-multi-home deps + rm nets"
echo "  systemctl --user start hermes-gateway hermes-dashboard"
echo "  (cockpit, voice, kokoro, and LLM containers are never stopped by this script)"
echo "#########################################"

# --- Guard: never touch FedPulse / sidekick ---
for forbidden in supabase_db_fedpulse fedpulse-api-1 sidekick-postgres; do
  echo "guard: $forbidden stays up -> $(docker inspect -f '{{.State.Running}}' "$forbidden" 2>/dev/null || echo missing)"
done

echo "=== 1. quiesce autonomous work (kill switch) ==="
hermes autonomy off "PRD-033 docker cutover" || echo "WARN: 'hermes autonomy off' failed (continuing; gateway about to stop anyway)"

echo "=== 2. stop host agent services (NOT cockpit, NOT LLM containers) ==="
systemctl --user stop $HOST_SERVICES || true
systemctl --user status hermes-gateway --no-pager 2>/dev/null | grep -E 'Active:' || true

echo "=== 3. backup ~/.hermes (agent now stopped -> consistent state.db) ==="
bash docker/migration/backup-hermes.sh

echo "=== 4. create contained networks (idempotent) ==="
docker network inspect hermes-agent-deps >/dev/null 2>&1 || docker network create --internal hermes-agent-deps
docker network inspect hermes-egress    >/dev/null 2>&1 || docker network create hermes-egress

echo "=== 5. multi-home the dependency containers onto the internal net (no recreate) ==="
for d in $DEPS; do
  if docker inspect -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$d" 2>/dev/null | grep -qw hermes-agent-deps; then
    echo "  $d already on hermes-agent-deps"
  else
    docker network connect hermes-agent-deps "$d" && echo "  connected $d"
  fi
done

echo "=== 6. rewrite host-local config -> container DNS (FR-4) ==="
bash docker/migration/apply-container-config.sh

echo "=== 7. compose up (build proxy, start gateway+dashboard+discord-proxy) ==="
HERMES_UID="$(id -u)" HERMES_GID="$(id -g)" docker compose -p "$PROJECT" -f "$COMPOSE" up -d --build

echo "=== 8. wait for the agent container to be running ==="
for i in $(seq 1 30); do
  st=$(docker inspect -f '{{.State.Status}}' hermes 2>/dev/null || echo none)
  echo "  hermes: $st ($i/30)"; [ "$st" = "running" ] && break; sleep 2
done

echo "=== 9. restart cockpit so it picks up the Bearer-auth patch ==="
systemctl --user restart hermes-cockpit || echo "WARN: cockpit restart failed"

echo "=== CUTOVER STEPS DONE. Now run: bash docker/migration/ac-suite.sh ==="
