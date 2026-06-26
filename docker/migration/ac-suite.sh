#!/usr/bin/env bash
# PRD-033 acceptance-criteria suite. Run AFTER `docker compose up` + the deps network
# attach. Each check prints PASS/FAIL and continues; exit code = number of FAILs.
# Scope: the reversible-cutover ACs (1,2,3,4,5,6,10,11,13). AC-7 (voice) / AC-9/12/14/15
# are handled separately (voice deferred; teardown post-soak).
set -uo pipefail
AGENT=hermes
FAIL=0
pass(){ echo "  PASS  $1"; }
fail(){ echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
ex(){ docker exec "$AGENT" sh -lc "$1" 2>/dev/null; }
# DIRECT reachability from the agent (bypass the proxy env so we test the real network
# route, not a tinyproxy deny-response). Returns 0 iff the host:port is actually reachable.
agent_curl(){ docker exec "$AGENT" sh -lc "curl -sS -o /dev/null --noproxy '*' --max-time 4 '$1'" >/dev/null 2>&1; }
# Through the Discord proxy (honors the agent's HTTP(S)_PROXY env). 0 iff the proxy forwarded.
agent_curl_proxied(){ docker exec "$AGENT" sh -lc "curl -sSf -o /dev/null --max-time 6 '$1'" >/dev/null 2>&1; }

echo "== AC-1: image is our fork, version works, fork markers present =="
docker exec "$AGENT" hermes --version >/dev/null 2>&1 && pass "hermes --version runs ($(docker exec "$AGENT" hermes --version 2>/dev/null | head -1))" || fail "hermes --version"
ex 'test -f /opt/hermes/tools/capability_policy.py && test -f /opt/hermes/tools/claude_review_tool.py' \
  && pass "fork-only files present (capability_policy + ask_claude)" || fail "fork markers missing"
ex 'test -f /opt/hermes/autonomy/audit.py' && pass "autonomy spine baked" || fail "autonomy spine missing"

echo "== AC-3: container runtime invariants (host-fs protection crux) =="
NETMODE=$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$AGENT" 2>/dev/null)
[ "$NETMODE" != "host" ] && pass "NetworkMode != host ($NETMODE)" || fail "NetworkMode is host!"
PRIV=$(docker inspect -f '{{.HostConfig.Privileged}}' "$AGENT" 2>/dev/null)
[ "$PRIV" = "false" ] && pass "not privileged" || fail "privileged!"
PIDM=$(docker inspect -f '{{.HostConfig.PidMode}}' "$AGENT" 2>/dev/null)
IPCM=$(docker inspect -f '{{.HostConfig.IpcMode}}' "$AGENT" 2>/dev/null)
[ "$PIDM" != "host" ] && pass "PidMode != host ($PIDM)" || fail "PidMode host!"
echo "$IPCM" | grep -qv '^host$' && pass "IpcMode != host ($IPCM)" || fail "IpcMode host!"
MOUNTS=$(docker inspect -f '{{range .Mounts}}{{.Source}}->{{.Destination}} {{end}}' "$AGENT" 2>/dev/null)
echo "$MOUNTS" | grep -q 'docker.sock' && fail "docker.sock mounted!" || pass "no docker socket mounted"
echo "$MOUNTS" | grep -qE '\.hermes->/opt/data' && pass "only ~/.hermes:/opt/data mount ($MOUNTS)" || fail "unexpected mounts: $MOUNTS"
NETS=$(docker inspect -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$AGENT" 2>/dev/null)
[ "$(echo "$NETS" | tr -s ' ' | sed 's/ $//')" = "hermes-agent-deps" ] && pass "agent on hermes-agent-deps ONLY ($NETS)" || fail "agent on extra nets: $NETS"

echo "== AC-3/4: deps reachable by DNS from agent =="
agent_curl 'http://llama-qwen36-35b:8080/v1/models' && pass "llama-qwen36-35b:8080 reachable" || fail "llama unreachable"
agent_curl 'http://llama-qwen3-4b:8080/v1/models' && pass "llama-qwen3-4b:8080 reachable" || fail "qwen3-4b unreachable"
agent_curl 'http://tei-bge-m3:80/health' || agent_curl 'http://tei-bge-m3:80/' && pass "tei-bge-m3:80 reachable" || fail "tei unreachable"
agent_curl 'http://qdrant:6333/healthz' || agent_curl 'http://qdrant:6333/' && pass "qdrant:6333 reachable" || fail "qdrant unreachable"

echo "== AC-3: FedPulse / sidekick / LAN / host-gateway UNREACHABLE from agent (must all fail) =="
for t in \
  'http://supabase_kong_fedpulse:8000' \
  'http://192.168.50.182:8100' 'http://192.168.50.182:8101' \
  'http://192.168.50.182:54322' 'http://192.168.50.182:54321' \
  'http://192.168.50.182:54330' \
  'http://host.docker.internal:8642' \
  'http://192.168.50.1' ; do
  if agent_curl "$t"; then fail "REACHABLE (should be blocked): $t"; else pass "blocked: $t"; fi
done

echo "== AC-5: Discord egress proxy policy (allow discord, deny FedPulse/LAN) =="
agent_curl_proxied 'https://discord.com/api/v10/gateway' && pass "discord.com reachable via proxy" || fail "discord.com NOT reachable via proxy"
# Through the proxy explicitly, FedPulse/LAN host:port must be denied by the allowlist.
if docker exec "$AGENT" sh -lc "curl -sS -o /dev/null --max-time 4 -x http://hermes-discord-proxy:8888 https://192.168.50.182:443" >/dev/null 2>&1; then
  fail "proxy allowed CONNECT to LAN 192.168.50.182:443"; else pass "proxy denied CONNECT to LAN"; fi

echo "== AC-2/4/10/13: config resolves in-container =="
ex 'grep -c "llama-qwen36-35b:8080" /opt/data/config.yaml' | grep -qE '^[1-9]' && pass "config.yaml uses container DNS" || fail "config.yaml not rewritten"
ex 'grep -q "qdrant:6333" /opt/data/mem0.json && grep -q "tei-bge-m3:80" /opt/data/mem0.json' && pass "mem0.json uses container DNS" || fail "mem0.json not rewritten"
ex 'test "$(stat -c %a /opt/data/mem0.json)" = "600"' && pass "mem0.json 600 (FR-13)" || fail "mem0.json perms not 600"
ex 'grep -qE "mode:\s*manual" /opt/data/config.yaml' && pass "approvals.mode manual in-container" || fail "approvals not manual"
ex 'grep -qE "tirith_fail_open:\s*false" /opt/data/config.yaml' && pass "tirith fail_open false" || fail "tirith fail_open not false"

echo "== AC-6: cockpit API (host) + dashboard reachable on loopback =="
KEY=$(grep -E '^API_SERVER_KEY=' "$HOME/.hermes/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')
[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 -H "Authorization: Bearer $KEY" http://127.0.0.1:8642/v1/models)" = "200" ] \
  && pass "containerized API authenticates on 127.0.0.1:8642" || fail "API not reachable/authed on 8642"
curl -s -o /dev/null --max-time 4 http://127.0.0.1:9119 && pass "dashboard reachable on 127.0.0.1:9119" || fail "dashboard not reachable"

echo "== AC-11: resource limits on agent container =="
MEM=$(docker inspect -f '{{.HostConfig.Memory}}' "$AGENT" 2>/dev/null)
PIDS=$(docker inspect -f '{{.HostConfig.PidsLimit}}' "$AGENT" 2>/dev/null)
[ "${MEM:-0}" -gt 0 ] && pass "mem_limit set ($MEM bytes)" || fail "no mem_limit"
[ "${PIDS:-0}" -gt 0 ] && pass "pids_limit set ($PIDS)" || fail "no pids_limit"

echo
echo "==== AC SUITE COMPLETE: $FAIL failure(s) ===="
exit "$FAIL"
