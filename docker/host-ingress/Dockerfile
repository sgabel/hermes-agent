# PRD-033 FR-6 — host-ingress forwarder. Docker disables published-port DNAT for a
# container attached ONLY to an `internal: true` network, so the agent (internal-only,
# for FedPulse isolation) cannot itself publish 127.0.0.1:8642 / :9119 to the host.
# This dumb socat forwarder is dual-homed (internal deps net + a normal egress net) so
# its publish works, and it forwards FIXED targets back to the agent/dashboard over the
# internal net. It is NOT a general proxy (two fixed listeners), so it gives the agent
# no new egress/pivot — the agent stays internal-only. Same trust shape as the alpine
# Discord proxy.
FROM alpine:3.20
RUN apk add --no-cache socat
# 8642 -> gateway API, 9119 -> dashboard. Both reached by service DNS on the internal net.
CMD sh -c 'socat TCP-LISTEN:8642,fork,reuseaddr TCP:hermes:8642 & \
           socat TCP-LISTEN:9119,fork,reuseaddr TCP:hermes-dashboard:9119 & \
           wait'
