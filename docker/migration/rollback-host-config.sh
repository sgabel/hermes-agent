#!/usr/bin/env bash
# PRD-033 FR-12 — restore the host-local config so host systemd services work again.
# Restores config.yaml, mem0.json, .env from the snapshot taken by apply-container-config.sh.
# Does NOT restart services (operator does that) and does NOT touch the data (state.db etc.).
set -euo pipefail
HERMES_HOME="${HERMES_HOME_HOST:-$HOME/.hermes}"
SNAP="$HERMES_HOME/.migration/host-originals"
if [ ! -d "$SNAP" ]; then
  echo "ERROR: no snapshot at $SNAP — nothing to roll back (was apply-container-config.sh run?)" >&2
  exit 1
fi
for base in config.yaml mem0.json .env; do
  if [ -e "$SNAP/$base" ]; then
    cp -p "$SNAP/$base" "$HERMES_HOME/$base"
    echo "restored: $base"
  else
    echo "WARN: no snapshot for $base — left as-is" >&2
  fi
done
echo "=== host-local URLs restored. verify: ==="
grep -nE 'base_url' "$HERMES_HOME/config.yaml" | grep localhost || true
grep -nE 'localhost' "$HERMES_HOME/mem0.json" || true
echo "Now: docker compose -p hermes-agent -f docker-compose.migration.yml down; systemctl --user start hermes-gateway hermes-dashboard"
echo "(voice + kokoro stay host-side and were never stopped; cockpit + LLM containers untouched)"
