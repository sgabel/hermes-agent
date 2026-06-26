#!/usr/bin/env bash
# PRD-033 robustness — ensure the dependency containers are attached to the internal
# `hermes-agent-deps` network so the containerized agent can reach them by DNS.
#
# The multi-home is applied imperatively (`docker network connect`) and SURVIVES reboots
# (containers restart, not recreate). It is only lost if the LLM/Qdrant stack is explicitly
# recreated (`docker compose up`/`down`+`up`/`--force-recreate` on ~/hermes/docker-compose.yml).
# Run this after any such recreate. Idempotent + name-scoped (never touches FedPulse/sidekick).
#
# PERMANENT fix (so this is automatic on every recreate): add to each dep service in
# ~/hermes/docker-compose.yml:
#     networks:
#       - default
#       - hermes-agent-deps
#   and a top-level:
#     networks:
#       hermes-agent-deps:
#         external: true
set -uo pipefail
DEPS="llama-qwen36-35b llama-qwen3-4b tei-bge-m3 qdrant"
docker network inspect hermes-agent-deps >/dev/null 2>&1 || docker network create --internal hermes-agent-deps
for d in $DEPS; do
  if docker inspect -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$d" 2>/dev/null | grep -qw hermes-agent-deps; then
    echo "  $d already attached"
  else
    docker network connect hermes-agent-deps "$d" 2>/dev/null && echo "  re-attached $d" || echo "  WARN: could not attach $d (running?)"
  fi
done
echo "done. (if you toggled the :8081 model to a gemma variant, also attach that container"
echo " and update model.base_url in config.yaml to the new service DNS name.)"
