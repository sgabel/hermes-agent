"""Host-side advisory relay — a toolless Claude second opinion for the caged agent.

PRD-035. The containerized Sylva agent has no route to api.anthropic.com (the
`internal:true` network only permits Discord egress). This relay runs host-side,
holds the OAuth token (which never enters any container), and — per request over a
unix socket on a dedicated hermes-only bind mount — returns an opaque, toolless
Claude advisory.

The relay's own four checks are the SOLE security boundary (the in-container tool
gate is bypassable by a code-exec-capable compromised agent connecting the socket
directly). Every request passes, in order:

  1. bearer auth            (FR-5) — Authorization header, constant-time compare
  2. egress classifier      (FR-6) — fail-closed named-credential screen on body
  3. kill-switch            (FR-8) — QUIESCE halts ALL consults (incl. interactive)
  4. budget admission-debit (FR-8) — pre-spawn debit of second_opinion_calls; refuse
                                     if over the single shared daily cap
then spawns a toolless `claude` under a hardened transient systemd service
(FR-2/FR-12), scrubs the reply (FR-6a), records a T1 audit line, and returns the
opaque advisory text.

Concurrency: a single consult runs at a time (FR-12). Extra requests wait briefly
then get 503 (bounded queue), so a burst cannot spawn N host-side claude runtimes.

This module is host-only. It imports hermes-agent's autonomy primitives (budget /
killswitch / audit), which resolve their state under ~/.hermes/autonomy — the same
ledger the container writes via /opt/data, so a host consult and an in-container
cron consult share one budget counter.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import shutil
import socket
import socketserver
import subprocess
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional

from relay import egress_classifier

logger = logging.getLogger("hermes.advisory_relay")

# --- Fixed invocation shape (FR-2). --max-turns is accepted-but-hidden on
# claude 2.1.116; --setting-sources '' (empty) disables user/project/local
# sources so the host global CLAUDE.md + memory are NOT ingested (verified).
#
# --max-turns is 2, NOT 1 (empirical correction, live-verified 2026-07-06):
# containment is guaranteed by `--tools ""` (nothing is executable), NOT by the
# turn count. With --max-turns 1, a prompt that makes claude *attempt* a tool
# (e.g. "read this file") consumes the single turn on the blocked attempt and
# exits `error_max_turns` (rc=1) — breaking usability + the self-canary — even
# though nothing leaked. --max-turns 2 lets claude recover from a blocked tool
# attempt and return text with a clean exit; the planted-marker canary confirms
# the file is STILL not read (tools remain disabled). Pure advisory prompts
# ("review this plan") never attempt a tool and finish in turn 1 regardless.
_CLAUDE_BASE_ARGS = [
    "-p",
    "--output-format", "json",
    "--no-session-persistence",
    "--tools", "",
    "--disallowedTools", "*",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--max-turns", "2",
    "--setting-sources", "",
]

# Hardened transient-service properties (FR-12). Live-verified to keep OAuth
# working (ProtectHome=read-only lets claude READ ~/.claude/.credentials.json
# via the isolated-HOME symlink while blocking writes to /home).
_SYSTEMD_RUN_PROPS = [
    "--property=Type=exec",
    "-p", "NoNewPrivileges=yes",
    "-p", "PrivateTmp=yes",
    "-p", "ProtectSystem=strict",
    "-p", "ProtectHome=read-only",
]

_MAX_PROMPT_CHARS = 48_000
_BUDGET_KIND = "second_opinion_calls"
_AUDIT_TIER = "T1"  # sanctioned external second-opinion egress (AGENT_SECURITY_MODEL)
_SURFACE_LABEL_MAX = 32


class RelayConfig:
    """Resolved relay settings. Paths default to the dedicated hermes-only dir."""

    def __init__(
        self,
        *,
        socket_path: str,
        token_path: str,
        model: str,
        isolated_home: str,
        claude_bin: str = "claude",
        child_mem_max: str = "2G",
        child_runtime_max_sec: int = 180,
        consult_timeout_sec: int = 200,
        queue_wait_sec: float = 2.0,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.token_path = Path(token_path)
        self.model = model
        self.isolated_home = Path(isolated_home)
        self.claude_bin = claude_bin
        self.child_mem_max = child_mem_max
        self.child_runtime_max_sec = child_runtime_max_sec
        self.consult_timeout_sec = consult_timeout_sec
        self.queue_wait_sec = queue_wait_sec


class AdvisoryRelay:
    """The relay core. Owns the consult lock, the bearer, and the spawn logic."""

    def __init__(self, config: RelayConfig) -> None:
        self.config = config
        self._bearer = self._load_bearer()
        # FR-12: one consult in-flight. A bounded wait gives a small queue; past
        # it, callers get 503 rather than fanning out host processes.
        self._consult_lock = threading.Lock()

    # -- bearer (FR-5) --------------------------------------------------------

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

    # -- toolless spawn (FR-2 / FR-3 / FR-4 / FR-12) --------------------------

    def _child_env(self) -> dict[str, str]:
        """Minimal allowlist env. HOME points at the isolated creds-only dir; we
        keep only what `claude` (a Node CLI) needs to run + read its OAuth file.
        The host provider creds / AWS chain are NOT inherited (FR-4)."""
        env = {
            "HOME": str(self.config.isolated_home),
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            # Node/npm runtime discovery for the global `claude` install.
            "NODE_PATH": os.environ.get("NODE_PATH", ""),
        }
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg:
            env["XDG_RUNTIME_DIR"] = xdg
        return {k: v for k, v in env.items() if v}

    def _spawn_claude(self, assembled_prompt: str) -> tuple[bool, str]:
        """Run the toolless claude under a hardened transient systemd service.
        Prompt is fed on stdin (never argv — FR-3). Returns (ok, text)."""
        env = self._child_env()
        cmd = [
            "systemd-run", "--user", "--pipe", "--quiet", "--wait", "--collect",
            *_SYSTEMD_RUN_PROPS,
            "-p", f"MemoryMax={self.config.child_mem_max}",
            "-p", f"RuntimeMaxSec={self.config.child_runtime_max_sec}",
            "-p", f"ReadWritePaths={self.config.isolated_home}",
            f"--setenv=HOME={env['HOME']}",
            f"--setenv=PATH={env['PATH']}",
        ]
        if env.get("XDG_RUNTIME_DIR"):
            cmd.append(f"--setenv=XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}")
        cmd += [self.config.claude_bin, "--model", self.config.model, *_CLAUDE_BASE_ARGS]

        try:
            proc = subprocess.run(
                cmd,
                input=assembled_prompt,
                capture_output=True,
                text=True,
                timeout=self.config.consult_timeout_sec,
                env={"PATH": os.environ.get("PATH", ""),
                     "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", ""),
                     "DBUS_SESSION_BUS_ADDRESS": os.environ.get("DBUS_SESSION_BUS_ADDRESS", "")},
            )
        except subprocess.TimeoutExpired:
            return False, "advisory consult timed out"
        except Exception as exc:  # pragma: no cover - spawn failure
            logger.error("relay spawn failed: %s", exc, exc_info=True)
            return False, "advisory consult failed to spawn"

        if proc.returncode != 0:
            logger.warning("claude child exited %s: %s", proc.returncode, proc.stderr[:200])
            return False, "advisory consult returned an error"

        return True, self._extract_text(proc.stdout)

    @staticmethod
    def _extract_text(stdout: str) -> str:
        """Pull the opaque advisory text out of claude's --output-format json.
        Never parsed for a structured verdict (FR-2/AC-008) — just the `result`
        string, surfaced verbatim."""
        try:
            data = json.loads(stdout)
            if isinstance(data, dict) and isinstance(data.get("result"), str):
                return data["result"]
        except Exception:
            pass
        # Fall back to raw stdout (still opaque advisory text).
        return stdout.strip()

    # -- the consult pipeline -------------------------------------------------

    def handle_consult(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Run one consult through all four relay checks. Returns (http_status,
        json_body). This is the authoritative security path."""
        prompt = body.get("prompt")
        context = body.get("context") or ""
        if not isinstance(prompt, str) or not prompt.strip():
            return 400, {"error": "prompt required"}
        if not isinstance(context, str):
            context = ""

        assembled = prompt if not context.strip() else f"{prompt}\n\n--- context ---\n{context}"
        if len(assembled) > _MAX_PROMPT_CHARS:
            return 413, {"error": f"payload too large ({len(assembled)} > {_MAX_PROMPT_CHARS})"}

        # surface is an UNTRUSTED analytics label only (FR-1). Never a gating input.
        surface = str(body.get("surface") or "relay")[:_SURFACE_LABEL_MAX]

        # (2) egress classifier — fail-closed (FR-6).
        refuse, reason = egress_classifier.contains_credential(assembled)
        if refuse:
            self._audit(surface, action="consult refused: credential screen",
                        rationale=reason or "classifier", outcome="refused_secret")
            return 422, {"error": "refused: payload contains a credential shape", "reason": reason}

        # FR-12: serialize to one in-flight consult (bounded queue → 503).
        acquired = self._consult_lock.acquire(timeout=self.config.queue_wait_sec)
        if not acquired:
            return 503, {"error": "relay busy (one consult in flight); retry shortly"}
        try:
            # (3) kill-switch first (FR-8).
            try:
                from autonomy import killswitch
                if killswitch.guard(f"advisory_relay:{surface}"):
                    return 503, {"error": "advisory relay quiesced (autonomy kill switch engaged)"}
            except Exception as exc:  # never let a governance import wedge the relay open
                logger.error("kill-switch check failed — failing closed: %s", exc)
                return 503, {"error": "kill-switch unavailable; refusing (fail-closed)"}

            # (4) budget admission-debit (FR-8). debit() increments then reports
            # allowed=not degrade; over-cap ⇒ refuse (the increment over-counts the
            # refused attempt — the accepted race-safe trade).
            try:
                from autonomy import budget
                result = budget.debit(f"advisory_relay:{surface}", _BUDGET_KIND, 1, audit=True)
            except Exception as exc:
                logger.error("budget debit failed — failing closed: %s", exc)
                return 503, {"error": "budget unavailable; refusing (fail-closed)"}
            if not result.get("allowed", False):
                self._audit(surface, action="consult refused: budget cap",
                            rationale=f"{_BUDGET_KIND} daily cap reached", outcome="degraded")
                return 429, {"error": "second-opinion daily budget exhausted (degrade-to-ask)"}

            # spawn toolless claude (FR-2).
            ok, text = self._spawn_claude(assembled)
        finally:
            self._consult_lock.release()

        if not ok:
            self._audit(surface, action="consult error", rationale="spawn/child error",
                        outcome="error")
            return 502, {"error": text}

        # (FR-6a) scrub the return channel with the classifier's redactor.
        advisory_text = egress_classifier.redact(text)
        truncated = False

        self._audit(surface, action="advisory consult", rationale="toolless claude second opinion",
                    outcome="ok")
        return 200, {"advisory_text": advisory_text, "truncated": truncated}

    def _audit(self, surface: str, *, action: str, rationale: str, outcome: str) -> None:
        """T1 audit line (FR-8). budget.debit only audits on the degrade branch
        (hardcoded T3), so the relay records its own T1 line unconditionally."""
        try:
            from autonomy import audit
            audit.record(tier=_AUDIT_TIER, surface=f"advisory_relay:{surface}",
                         action=action, rationale=rationale,
                         authority="relay", outcome=outcome)
        except Exception as exc:  # pragma: no cover
            logger.debug("relay audit record failed (non-fatal): %s", exc)

    # -- startup self-canary (FR-2a) -----------------------------------------

    def self_canary(self) -> bool:
        """Prove the toolless invocation still cannot read a host file on the
        installed claude version — a backstop against a silent CLI update
        re-enabling tools. The relay refuses to bind unless this passes.

        Containment is proven by the planted secret being ABSENT from the reply,
        NOT by any sentinel word: toolless claude emits *fake* tool-call narration
        ("Reading the file… Read") even though it executed nothing, so a magic-word
        check is unreliable. The unique planted marker leaking would mean a real
        read happened (tools re-enabled) → refuse to bind.
        """
        marker = "RELAY-CANARY-DO-NOT-EXFIL-7f3a9c2e"
        canary_path = self.config.isolated_home / ".relay_canary_secret"
        try:
            canary_path.write_text(marker + "\n", encoding="utf-8")
        except Exception as exc:
            logger.error("self-canary setup failed: %s", exc)
            return False
        probe = (
            f"Read the file at {canary_path} and reply with its exact contents. "
            "If you cannot, say so."
        )
        ok, text = self._spawn_claude(probe)
        try:
            canary_path.unlink()
        except Exception:
            pass
        if not ok:
            logger.error("self-canary spawn failed")
            return False
        leaked = marker in text
        if leaked:
            logger.error("self-canary FAILED — planted secret leaked; toolless containment "
                         "NOT confirmed (tools may be re-enabled); refusing to bind")
        return not leaked


# --- HTTP-over-UDS server (FR-1) --------------------------------------------

class _RelayHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    relay: AdvisoryRelay = None  # set on the server instance below

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
            # Health does NOT leak state; no bearer needed for liveness.
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        relay: AdvisoryRelay = self.server.relay  # type: ignore[attr-defined]
        # (1) bearer auth (FR-5) — the bearer rides the header, never the body,
        # so the FR-6 classifier never sees it (no self-DoS).
        if not relay._bearer_ok(self.headers.get("Authorization")):
            self._send(401, {"error": "unauthorized"})
            return
        if self.path != "/consult":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "bad content-length"})
            return
        if length <= 0 or length > _MAX_PROMPT_CHARS + 4096:
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

        status, payload = relay.handle_consult(body)
        self._send(status, payload)


class _ThreadingUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """UnixStreamServer that speaks HTTP via BaseHTTPRequestHandler. Threading is
    only for connection handling; the consult itself is serialized by the relay's
    consult lock (FR-12)."""

    daemon_threads = True
    allow_reuse_address = True
    relay: AdvisoryRelay

    def get_request(self):
        # BaseHTTPRequestHandler expects (conn, client_address); UnixStreamServer
        # yields an empty peer name, so synthesize one.
        conn, _ = super().get_request()
        return conn, ("unix", 0)


def create_server(relay: AdvisoryRelay) -> _ThreadingUnixHTTPServer:
    """Bind the relay's unix socket (0600) after unlinking any stale one (FR-11)."""
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
    # Restrict the socket to 0600: set umask around bind.
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


def _resolve_config_from_env() -> RelayConfig:
    """Config from env/defaults. The systemd unit supplies these; defaults point
    at the dedicated hermes-only relay dir."""
    base = Path(os.environ.get("HERMES_RELAY_DIR", str(Path.home() / ".hermes-relay")))
    model = os.environ.get("HERMES_RELAY_MODEL") or _resolve_pinned_model()
    return RelayConfig(
        socket_path=os.environ.get("HERMES_RELAY_SOCKET", str(base / "consult.sock")),
        token_path=os.environ.get("HERMES_RELAY_TOKEN", str(base / "client.token")),
        model=model,
        isolated_home=os.environ.get("HERMES_RELAY_HOME", str(base / "claude-home")),
        claude_bin=os.environ.get("HERMES_RELAY_CLAUDE_BIN", shutil.which("claude") or "claude"),
    )


def _resolve_pinned_model() -> str:
    """Pinned second-opinion model from config.yaml, else a safe default."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        m = cfg_get(read_raw_config(), "model", "second_opinion_model")
        if isinstance(m, str) and m.strip():
            return m.strip()
    except Exception:
        pass
    return "claude-sonnet-5"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = _resolve_config_from_env()
    relay = AdvisoryRelay(config)

    # FR-2a: refuse to bind unless the toolless containment canary passes.
    Path(config.isolated_home).mkdir(parents=True, exist_ok=True)
    if not relay.self_canary():
        logger.error("startup self-canary failed — NOT binding the socket (fail-closed)")
        return 3

    server = create_server(relay)
    logger.info("advisory relay listening on %s (model=%s)", config.socket_path, config.model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            config.socket_path.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
