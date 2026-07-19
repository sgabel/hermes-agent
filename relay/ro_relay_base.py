"""Shared substrate for host-side capability relays (PRD-048).

This is the repeatable half of the RO-access pattern (canonical recipe:
``docs/reference/RO_ACCESS_PATTERN.md``): a host-side process that grants the
containerized agent one narrow, governed capability over HTTP-over-UDS, with
the credential for that capability held host-side only — it never enters any
container env, config, or mount.

Extracted from the PRD-035 advisory relay (``relay/advisory_relay.py``), which
still runs on its original standalone code (retrofit parked — it is live and
working; touching it was ruled out at the PRD-048 build). The base carries the
same security semantics for NEW relays:

  * HTTP-over-UDS on a dedicated socket under the hermes-only relay dir
    (0700; bind-mounted read-only into the gateway container ONLY — the
    dashboard has no ``/opt/relay``).
  * A per-capability bearer token file (0600) — capabilities are separately
    revocable by deleting one token, and the token preflight refuses startup
    on wrong owner/mode.
  * A fixed request pipeline, every stage fail-closed:

      1. bearer auth             Authorization header, constant-time compare
      2. request validation      subclass (shape, size, capability parse)
      3. bearer-not-in-payload   the bearer rides the header, never the body
      4. request screen          egress classifier on the request text
      5. kill switch             QUIESCE refuses (fail-closed on check error)
      6. budget admission-debit  per-capability kind, debited BEFORE execution
                                 (a failed execution is never a free retry;
                                 over-count-never-under-count, the PRD-035
                                 accepted trade)
      7. concurrency semaphore   bounded in-flight executions (bounded wait,
                                 then 503 — a burst cannot fan out host work)
      8. execute                 subclass
      9. response scrub          DLP on the way back to the agent
     10. audit                   one T-tier ledger row per request

  * A startup self-canary (subclass-defined) that must pass or the relay
    REFUSES TO BIND the socket.

Subclasses implement: ``validate_request`` / ``request_text`` / ``execute`` /
``scrub_response`` / ``self_canary``. Everything else — the pipeline order,
the UDS server, the preflights — is deliberately not overridable per relay:
the pattern's value is that every relay enforces the same spine.

This module is host-only. The autonomy primitives (budget / killswitch /
audit) are imported lazily and resolve their state under ``~/.hermes/autonomy``
— the same ledger the container writes via ``/opt/data``, so host and
in-container actions share one budget counter.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional

from relay import egress_classifier

logger = logging.getLogger("hermes.ro_relay_base")

_SURFACE_LABEL_MAX = 32

# Hard cap on concurrent connection threads (not executions — those are bounded
# by the per-relay semaphore). Bounds the fd/thread cost of a client burst.
_MAX_CONNECTIONS = 16


class RelayRequestError(Exception):
    """A request-level refusal with an HTTP status. Raised by subclass
    ``validate_request``/``execute``; never carries secret material."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class RORelayConfig:
    """Resolved settings for one relay instance."""

    def __init__(
        self,
        *,
        socket_path: str,
        token_path: str,
        route: str,
        budget_kind: str,
        audit_surface_prefix: str,
        audit_tier: str = "T1",
        max_body_bytes: int = 64 * 1024,
        concurrency: int = 1,
        queue_wait_sec: float = 2.0,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.token_path = Path(token_path)
        self.route = route
        self.budget_kind = budget_kind
        self.audit_surface_prefix = audit_surface_prefix
        self.audit_tier = audit_tier
        self.max_body_bytes = max_body_bytes
        self.concurrency = concurrency
        self.queue_wait_sec = queue_wait_sec


class RORelayBase:
    """The shared relay spine. Owns the bearer, the pipeline, and the audit."""

    def __init__(self, config: RORelayConfig) -> None:
        self.config = config
        self._bearer = self._load_bearer()
        self._exec_sem = threading.BoundedSemaphore(config.concurrency)

    # -- subclass surface -----------------------------------------------------

    def validate_request(self, body: dict[str, Any]) -> Any:
        """Parse/validate the request body into a capability-specific object.
        Raise ``RelayRequestError`` to refuse."""
        raise NotImplementedError

    def request_text(self, validated: Any) -> str:
        """The request's textual payload, for the bearer-in-payload check and
        the request-side classifier screen."""
        raise NotImplementedError

    def execute(self, validated: Any) -> tuple[bool, dict[str, Any]]:
        """Perform the capability. Returns ``(ok, payload)``; on ``ok=False``
        the payload's ``error`` string is returned with a 502."""
        raise NotImplementedError

    def scrub_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """DLP pass over the response payload. Must be fail-closed: raise (or
        return a fully-masked payload) rather than risk leaking. The base
        treats an exception here as refuse-with-502."""
        raise NotImplementedError

    def self_canary(self) -> bool:
        """Prove the relay's enforcement invariants at startup. The relay
        refuses to bind unless this returns True."""
        raise NotImplementedError

    # -- bearer ---------------------------------------------------------------

    def _load_bearer(self) -> str:
        token = self.config.token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise RuntimeError(f"relay bearer token empty at {self.config.token_path}")
        return token

    def _bearer_ok(self, header_value: Optional[str]) -> bool:
        if not header_value:
            return False
        prefix = "Bearer "
        presented = header_value[len(prefix):] if header_value.startswith(prefix) else header_value
        return hmac.compare_digest(presented.strip(), self._bearer)

    # -- the pipeline ---------------------------------------------------------

    def handle_request(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Run one request through the full pipeline (stages 2–10; stage 1,
        bearer auth, runs at the HTTP layer before the body is parsed)."""
        # surface is an UNTRUSTED analytics label only. Never a gating input.
        surface = str(body.get("surface") or "relay")[:_SURFACE_LABEL_MAX]

        # (2) capability-specific validation.
        try:
            validated = self.validate_request(body)
        except RelayRequestError as exc:
            return exc.status, {"error": exc.message}
        except Exception as exc:  # subclass bug => refuse, never crash the server
            logger.error("validate_request failed — refusing: %s", exc)
            return 400, {"error": "request validation failed"}

        try:
            text = self.request_text(validated)
        except Exception:
            return 400, {"error": "request text extraction failed"}

        # (3) the bearer must never appear in the payload (it would otherwise
        # transit logs/audit or the executed capability itself).
        if self._bearer and self._bearer in text:
            self._audit(surface, action="request refused: bearer in payload",
                        rationale="bearer token present in request body", outcome="refused_secret")
            return 422, {"error": "refused: relay bearer must not appear in the payload"}

        # (4) request screen — fail-closed classifier on the request text.
        refuse, reason = egress_classifier.contains_credential(text)
        if refuse:
            self._audit(surface, action="request refused: credential screen",
                        rationale=reason or "classifier", outcome="refused_secret")
            return 422, {"error": "refused: request contains a credential shape", "reason": reason}

        # (5) kill switch — QUIESCE halts the capability (fail-closed on error).
        try:
            from autonomy import killswitch
            if killswitch.guard(f"{self.config.audit_surface_prefix}:{surface}"):
                return 503, {"error": "relay quiesced (autonomy kill switch engaged)"}
        except Exception as exc:
            logger.error("kill-switch check failed — failing closed: %s", exc)
            return 503, {"error": "kill-switch unavailable; refusing (fail-closed)"}

        # (6) budget admission-debit — BEFORE execution, so timeouts/errors are
        # not free retries (over-count, never under-count).
        try:
            from autonomy import budget
            result = budget.debit(f"{self.config.audit_surface_prefix}:{surface}",
                                  self.config.budget_kind, 1, audit=True)
        except Exception as exc:
            logger.error("budget debit failed — failing closed: %s", exc)
            return 503, {"error": "budget unavailable; refusing (fail-closed)"}
        if not result.get("allowed", False):
            self._audit(surface, action="request refused: budget cap",
                        rationale=f"{self.config.budget_kind} daily cap reached", outcome="degraded")
            return 429, {"error": f"{self.config.budget_kind} daily budget exhausted (degrade-to-ask)"}

        # (7) concurrency — bounded wait, then 503.
        acquired = self._exec_sem.acquire(timeout=self.config.queue_wait_sec)
        if not acquired:
            return 503, {"error": "relay busy; retry shortly"}
        try:
            # (8) the capability itself.
            ok, payload = self.execute(validated)
        except RelayRequestError as exc:
            self._audit(surface, action="request error", rationale=exc.message[:200], outcome="error")
            return exc.status, {"error": exc.message}
        except Exception as exc:
            logger.error("execute failed: %s", exc, exc_info=True)
            self._audit(surface, action="request error", rationale="executor exception", outcome="error")
            return 502, {"error": "execution failed"}
        finally:
            self._exec_sem.release()

        if not ok:
            self._audit(surface, action="request error", rationale="capability error", outcome="error")
            return 502, payload if "error" in payload else {"error": "execution failed"}

        # (9) response scrub — fail-closed DLP.
        try:
            scrubbed = self.scrub_response(payload)
        except Exception as exc:
            logger.error("response scrub failed — refusing (fail-closed): %s", exc)
            self._audit(surface, action="request refused: response scrub",
                        rationale="scrub error (fail-closed)", outcome="refused_secret")
            return 502, {"error": "response scrub failed (fail-closed)"}

        # (10) audit.
        self._audit(surface, action="request served", rationale=self.config.budget_kind, outcome="ok")
        return 200, scrubbed

    def _audit(self, surface: str, *, action: str, rationale: str, outcome: str) -> None:
        """One ledger row per request. budget.debit only audits on the degrade
        branch, so the relay records its own line unconditionally."""
        try:
            from autonomy import audit
            audit.record(tier=self.config.audit_tier,
                         surface=f"{self.config.audit_surface_prefix}:{surface}",
                         action=action, rationale=rationale,
                         authority="relay", outcome=outcome)
        except Exception as exc:  # pragma: no cover
            logger.debug("relay audit record failed (non-fatal): %s", exc)


# --- HTTP-over-UDS server ----------------------------------------------------

class _RelayHTTPHandler(BaseHTTPRequestHandler):
    # HTTP/1.0 → the connection closes after each response (no keep-alive), so
    # a slow client cannot hold a worker thread across requests.
    protocol_version = "HTTP/1.0"
    # Per-request socket timeout bounds slow-loris.
    timeout = 15

    def log_message(self, fmt, *args):  # keep BaseHTTPRequestHandler off stderr
        logger.debug("relay http: " + fmt, *args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            # Liveness only; no state leak, no bearer required.
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        relay: RORelayBase = self.server.relay  # type: ignore[attr-defined]
        # (1) bearer auth — the bearer rides the header, never the body, so the
        # request classifier never sees it.
        if not relay._bearer_ok(self.headers.get("Authorization")):
            self._send(401, {"error": "unauthorized"})
            return
        if self.path != relay.config.route:
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "bad content-length"})
            return
        if length <= 0 or length > relay.config.max_body_bytes:
            self._send(413, {"error": "empty or oversize body"})
            return
        try:
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except Exception:
            self._send(400, {"error": "invalid JSON body"})
            return

        status, payload = relay.handle_request(body)
        self._send(status, payload)


class _ThreadingUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """UnixStreamServer speaking HTTP. Threading is only for connection
    handling; executions are bounded by the relay's semaphore, and concurrent
    connections are capped."""

    daemon_threads = True
    allow_reuse_address = True
    relay: RORelayBase
    _conn_sem = threading.BoundedSemaphore(_MAX_CONNECTIONS)

    def get_request(self):
        conn, _ = super().get_request()
        return conn, ("unix", 0)

    def process_request(self, request, client_address):
        if not self._conn_sem.acquire(blocking=False):
            try:
                request.close()
            finally:
                return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._conn_sem.release()


def create_server(relay: RORelayBase) -> _ThreadingUnixHTTPServer:
    """Bind the relay's unix socket (0600) after unlinking any stale one."""
    sock_path = relay.config.socket_path
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(sock_path.parent, 0o700)
    except OSError:
        pass
    if sock_path.exists() or sock_path.is_socket():
        try:
            sock_path.unlink()
        except FileNotFoundError:
            pass
    old_umask = os.umask(0o177)
    try:
        server = _ThreadingUnixHTTPServer(str(sock_path), _RelayHTTPHandler)
    finally:
        os.umask(old_umask)
    try:
        os.chmod(sock_path, 0o600)
    except OSError:
        pass
    server.relay = relay
    return server


def preflight_secret_file(path: Path, *, label: str) -> bool:
    """Fail-closed preflight for any relay secret file (bearer token, DB
    password): must exist, be owned by THIS uid, and be exactly private
    (no group/other bits). A wrong owner/mode refuses startup."""
    try:
        st = os.stat(path)
    except OSError as exc:
        logger.error("preflight: cannot stat %s %s: %s", label, path, exc)
        return False
    if st.st_uid != os.getuid():
        logger.error("preflight: %s %s owner uid=%s != relay uid=%s — refusing (fail-closed)",
                     label, path, st.st_uid, os.getuid())
        return False
    mode = st.st_mode & 0o777
    if mode & 0o077:
        logger.error("preflight: %s %s mode %o is group/other-accessible — refusing", label, path, mode)
        return False
    return True


def run_relay(relay: RORelayBase, *, extra_secret_files: Optional[list[tuple[Path, str]]] = None) -> int:
    """Standard startup: secret-file preflights → self-canary → bind → serve.
    Any preflight or canary failure refuses to bind (fail-closed)."""
    if not preflight_secret_file(relay.config.token_path, label="bearer token"):
        logger.error("ownership preflight failed — NOT starting the relay (fail-closed)")
        return 4
    for path, label in (extra_secret_files or []):
        if not preflight_secret_file(path, label=label):
            logger.error("ownership preflight failed (%s) — NOT starting the relay (fail-closed)", label)
            return 4

    if not relay.self_canary():
        logger.error("startup self-canary failed — NOT binding the socket (fail-closed)")
        return 3

    server = create_server(relay)
    logger.info("relay listening on %s (route=%s, kind=%s)",
                relay.config.socket_path, relay.config.route, relay.config.budget_kind)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            relay.config.socket_path.unlink()
        except FileNotFoundError:
            pass
    return 0
