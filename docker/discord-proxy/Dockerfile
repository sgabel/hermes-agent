# PRD-033 FR-3: Discord egress proxy. The ONLY route off the agent's internal
# `hermes-agent-deps` bridge to the public internet, and it reaches Discord ONLY.
# Built from alpine (auditable trust anchor), mirrors sandbox/proxy/Dockerfile.
FROM alpine:3.20

RUN apk add --no-cache tinyproxy curl \
    && mkdir -p /var/log/tinyproxy \
    && chown -R tinyproxy:tinyproxy /var/log/tinyproxy

COPY tinyproxy.conf /etc/tinyproxy/tinyproxy.conf
COPY filter /etc/tinyproxy/filter

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -s -o /dev/null --max-time 4 http://127.0.0.1:8888 || exit 1

CMD ["tinyproxy", "-d", "-c", "/etc/tinyproxy/tinyproxy.conf"]
