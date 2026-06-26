#!/usr/bin/env bash
# PRD-033 FR-12 — full network teardown for a clean rollback. Reverses the deps
# multi-home (docker network connect) and removes the migration networks so the host
# returns to its pre-cutover docker topology. Run AFTER `compose down` (network rm
# fails while the proxy/ingress/agent containers are still attached). Never touches
# FedPulse/sidekick networks.
set -uo pipefail
COMPOSE="$(dirname "$0")/../../docker-compose.migration.yml"
PROJECT="hermes-agent"
DEPS="llama-qwen36-35b llama-qwen3-4b tei-bge-m3 qdrant"

echo "=== 1. compose down (detach proxy/ingress/agent/dashboard) ==="
docker compose -p "$PROJECT" -f "$COMPOSE" down 2>/dev/null || true

echo "=== 2. disconnect deps from the internal net (reverse the multi-home) ==="
for d in $DEPS; do
  docker network disconnect hermes-agent-deps "$d" 2>/dev/null && echo "  disconnected $d" || echo "  $d not attached"
done

echo "=== 3. remove the migration networks ==="
docker network rm hermes-agent-deps hermes-egress 2>/dev/null && echo "  networks removed" || echo "  (some networks still in use or already gone)"

echo "=== deps remain on their original hermes_default network + host publishes (untouched) ==="
docker network inspect hermes_default -f '{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || true
