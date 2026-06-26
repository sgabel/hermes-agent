#!/usr/bin/env bash
# PRD-033 FR-4/FR-13 — rewrite host-local URLs in ~/.hermes config to container service
# DNS, and tighten mem0.json perms. IDEMPOTENT and REVERSIBLE: snapshots the original
# four files to ~/.hermes/.migration/host-originals/ before the first edit (only if a
# snapshot does not already exist), so rollback-host-config.sh can restore them verbatim.
#
# Run AFTER backup-hermes.sh and BEFORE `docker compose up`. Does not touch services.
set -euo pipefail

HERMES_HOME="${HERMES_HOME_HOST:-$HOME/.hermes}"
CFG="$HERMES_HOME/config.yaml"
MEM0="$HERMES_HOME/mem0.json"
ENVF="$HERMES_HOME/.env"
SNAP="$HERMES_HOME/.migration/host-originals"

mkdir -p "$SNAP"
# Snapshot originals once (never overwrite an existing snapshot — that is the rollback source).
for f in "$CFG" "$MEM0" "$ENVF"; do
  base="$(basename "$f")"
  if [ ! -e "$SNAP/$base" ]; then
    cp -p "$f" "$SNAP/$base"
    echo "snapshot: $base -> $SNAP/$base"
  else
    echo "snapshot exists (kept): $SNAP/$base"
  fi
done

echo "--- config.yaml: localhost:8081 -> llama-qwen36-35b:8080 (4 expected) ---"
sed -i 's#http://localhost:8081/v1#http://llama-qwen36-35b:8080/v1#g' "$CFG"

echo "--- mem0.json: qdrant / tei / qwen3-4b / history_db_path ---"
sed -i \
  -e 's#http://localhost:6333#http://qdrant:6333#g' \
  -e 's#http://localhost:8085#http://tei-bge-m3:80#g' \
  -e 's#http://localhost:8082/v1#http://llama-qwen3-4b:8080/v1#g' \
  -e 's#/home/sgabel/.hermes/mem0_history.db#/opt/data/mem0_history.db#g' \
  "$MEM0"

echo "--- .env: QDRANT_URL / TEI_BASE_URL / MEM0_LLM_BASE_URL / API_SERVER_HOST ---"
sed -i -E \
  -e 's#^QDRANT_URL=.*$#QDRANT_URL=http://qdrant:6333#' \
  -e 's#^TEI_BASE_URL=.*$#TEI_BASE_URL=http://tei-bge-m3:80#' \
  -e 's#^MEM0_LLM_BASE_URL=.*$#MEM0_LLM_BASE_URL=http://llama-qwen3-4b:8080/v1#' \
  "$ENVF"
# API_SERVER_HOST MUST be 0.0.0.0 inside the container so the host-ingress forwarder
# (a separate container) can reach the API. Append-if-missing: a substitute-only sed
# would silently no-op if the key were ever absent, leaving the API bound to the
# DEFAULT_HOST 127.0.0.1 and breaking cockpit/voice ingress.
if grep -qE '^API_SERVER_HOST=' "$ENVF"; then
  sed -i -E 's#^API_SERVER_HOST=.*$#API_SERVER_HOST=0.0.0.0#' "$ENVF"
else
  printf 'API_SERVER_HOST=0.0.0.0\n' >> "$ENVF"
fi

echo "--- FR-13: enforce owner-only perms (sed -i above can reset config.yaml/.env from 600) ---"
chmod 600 "$MEM0" "$CFG" "$ENVF"

echo "=== applied. verification (all diagnostics guarded; never abort under set -e): ==="
{ grep -nE 'base_url' "$CFG" | grep -E 'llama-qwen36-35b|localhost'; } || true
grep -nE 'qdrant:6333|tei-bge-m3:80|llama-qwen3-4b:8080|/opt/data/mem0_history.db' "$MEM0" || true
grep -nE '^(QDRANT_URL|TEI_BASE_URL|MEM0_LLM_BASE_URL|API_SERVER_HOST)=' "$ENVF" || true
echo "remaining localhost:8081 in config (should be 0):"; grep -c 'localhost:8081' "$CFG" || true
ls -l "$MEM0"
