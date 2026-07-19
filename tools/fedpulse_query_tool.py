"""fedpulse_query — read-only FedPulse database query, routed through the host relay (PRD-048).

Sylva (primary model: local qwen3.6-35b) can run a single read-only SQL statement
against the live FedPulse Postgres. The containerized agent has **no route** to
that database (PRD-033 `internal: true`), so this tool does NOT connect to
Postgres itself — it forwards the query over a unix socket to the **host-side
FedPulse RO relay** (PRD-048), which holds the `sylva_ro` credential (never in any
container), validates the SQL against a real PostgreSQL AST, executes it in an
explicit read-only transaction, scrubs the result rows, meters a daily budget, and
returns rows as JSON.

Security model (PRD-048, mirrors the PRD-035 advisory tool):
* The **relay's own checks are the authoritative boundary** (bearer, fail-closed
  AST validation with a schema allowlist + customer-table denylist + function
  denylist, read-only execution invariant, DB-side row/byte caps, result DLP,
  QUIESCE, `fedpulse_ro_queries` budget). A code-exec-capable compromised agent
  can reach the socket directly, so the in-container checks below are UX +
  defense-in-depth, NOT the security boundary.
* There is **no direct-DB fallback** — no pg client is baked into the image and
  the network is `internal: true`. If the relay is unreachable the tool REFUSES
  (fail-closed).
* This is a **T1 read** (capability_policy `_T1_TOOLS`): read-external, no host
  mutation. Writes are impossible — the relay rejects any non-SELECT/WITH/EXPLAIN
  and runs every query read-only.
* Result rows are live product/reference data; the relay's egress classifier
  redacts credential-shaped cells before they cross back. Treat rows as data.
"""

import http.client
import json
import logging
import os
import socket

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# In-container mount paths (the ~/.hermes-relay host dir is bind-mounted read-only
# to /opt/relay in the gateway container ONLY — never the dashboard). A dedicated
# socket + bearer, separate from the advisory relay, so the DB-read capability is
# independently revocable. Overridable for host-side testing.
_RELAY_SOCKET = os.environ.get("FEDPULSE_RO_SOCKET", "/opt/relay/fedpulse-ro.sock")
_RELAY_TOKEN_PATH = os.environ.get("FEDPULSE_RO_TOKEN", "/opt/relay/fedpulse.token")
_RELAY_TIMEOUT_SECONDS = 45   # > relay wall-timeout (30s) so the relay's error wins
_MAX_SQL_CHARS = 16_000       # mirrors the relay's SQL size cap
_MAX_RESPONSE_CHARS = 24_000  # cap the rendered result chip
_DEFAULT_MAX_ROWS = 200
_MAX_ROWS_CEILING = 1000


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that dials a unix socket instead of TCP."""

    def __init__(self, socket_path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def _relay_available() -> bool:
    """Availability predicate — the relay socket + token must both be present."""
    return os.path.exists(_RELAY_SOCKET) and os.path.exists(_RELAY_TOKEN_PATH)


def check_fedpulse_query_requirements() -> bool:
    """Advertise the tool only when the FedPulse RO relay is reachable."""
    return _relay_available()


def _read_bearer() -> str:
    with open(_RELAY_TOKEN_PATH, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def _contains_secret(text: str) -> bool:
    """Defense-in-depth input scan of the SQL (the relay's fail-closed classifier
    is the authoritative screen — this catches an obvious credential literal
    before it egresses to the relay/DB). Fail closed if the redactor can't run."""
    if not text:
        return False
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True) != text
    except Exception:
        logger.error("input secret scan failed; refusing fedpulse_query", exc_info=True)
        return True


def _in_cron_context() -> bool:
    return bool(os.environ.get("HERMES_CRON_SESSION"))


def _run_surface() -> str:
    """Analytics-only surface label (honest attended/unattended identity). Never a
    gating input at the relay — the relay's checks are authoritative there."""
    try:
        from autonomy.run_identity import classify_run

        ident = classify_run()
        if ident is not None:
            return ident.identity
    except Exception:
        pass
    return "cron" if _in_cron_context() else "interactive"


def _call_relay(sql: str, max_rows: int, surface: str) -> tuple[int, dict]:
    """POST the query to the relay over the unix socket. Returns (status, body).
    Any transport failure raises — the caller turns it into a fail-closed refuse."""
    payload = json.dumps({"sql": sql, "max_rows": max_rows, "surface": surface}).encode("utf-8")
    conn = _UnixHTTPConnection(_RELAY_SOCKET, timeout=_RELAY_TIMEOUT_SECONDS)
    try:
        conn.request(
            "POST", "/query", body=payload,
            headers={
                "Authorization": f"Bearer {_read_bearer()}",
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"error": raw[:300]}
        return resp.status, body
    finally:
        conn.close()


def fedpulse_query(sql: str = "", max_rows: int = _DEFAULT_MAX_ROWS, **_) -> str:
    """Run one read-only SQL statement against FedPulse via the host RO relay."""
    if not isinstance(sql, str) or not sql.strip():
        return tool_error("fedpulse_query requires a non-empty 'sql' statement", success=False)

    # Clamp max_rows to the relay's contract [1, 1000]; coerce loosely-typed input.
    try:
        max_rows = int(max_rows)
    except (TypeError, ValueError):
        max_rows = _DEFAULT_MAX_ROWS
    max_rows = max(1, min(max_rows, _MAX_ROWS_CEILING))

    if len(sql) > _MAX_SQL_CHARS:
        return tool_error(
            f"sql too large ({len(sql)} chars > {_MAX_SQL_CHARS}); narrow the query.",
            success=False,
        )

    # No direct-DB fallback — refuse if the relay is unreachable (fail-closed).
    if not _relay_available():
        return tool_error(
            "fedpulse_query is unavailable: the FedPulse RO relay is not reachable "
            f"(socket {_RELAY_SOCKET}). The tool never connects to the database directly — "
            "refusing (fail-closed).",
            success=False, blocked="relay_down",
        )

    # Defense-in-depth: refuse an obvious credential literal in the SQL before it
    # egresses (the relay's classifier is authoritative, but this fails fast).
    if _contains_secret(sql):
        return tool_error(
            "Refused: the SQL contains a secret-like pattern. Remove it before querying.",
            success=False, blocked="secret_in_sql",
        )

    try:
        status, body = _call_relay(sql, max_rows, _run_surface())
    except Exception as e:
        logger.error("fedpulse_query relay call failed: %s", e, exc_info=True)
        return tool_error(
            f"fedpulse_query failed to reach the FedPulse RO relay: {e} (fail-closed, no fallback)",
            success=False, blocked="relay_error",
        )

    if status != 200:
        msg = body.get("error", f"relay returned HTTP {status}")
        blocked = {400: "query_rejected", 401: "relay_auth", 422: "query_rejected",
                   429: "budget", 503: "quiesced_or_busy"}.get(status)
        return tool_error(f"fedpulse_query: {msg}", success=False, blocked=blocked)

    columns = body.get("columns", []) or []
    rows = body.get("rows", []) or []
    row_count = body.get("row_count", len(rows))
    truncated = bool(body.get("truncated"))
    dlp_redacted = bool(body.get("dlp_redacted"))

    # Render a bounded, readable result. Rows are arrays of strings (psql text).
    rendered = json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False)
    if len(rendered) > _MAX_RESPONSE_CHARS:
        rendered = rendered[:_MAX_RESPONSE_CHARS] + f"… [truncated > {_MAX_RESPONSE_CHARS} chars]"
        truncated = True

    return tool_result(
        success=True,
        columns=columns,
        rows=rows,
        row_count=row_count,
        truncated=truncated,
        dlp_redacted=dlp_redacted,
        result=rendered,
    )


FEDPULSE_QUERY_SCHEMA = {
    "name": "fedpulse_query",
    "description": (
        "Run ONE read-only SQL statement against the live FedPulse Postgres database "
        "(schemas `public` and `score_lab`) via the host RO relay, and get the rows back. "
        "Use this to see the real product data you are reasoning about — grant records, "
        "scoring tables, federal/IRS reference data.\n\n"
        "RULES (enforced by the relay — a violation is rejected, not executed):\n"
        "  • exactly ONE statement, and it must be a SELECT / WITH / EXPLAIN (no writes, "
        "no DDL, no SET, no multi-statement, no data-modifying CTEs, no FOR UPDATE).\n"
        "  • only schemas public and score_lab are readable; auth/vault/storage/net and the "
        "customer tables (profiles, organizations, subscriptions, …) are rejected.\n"
        "  • results are capped (use max_rows; default 200, max 1000) and credential-shaped "
        "values are redacted before they reach you.\n"
        "Do NOT put secrets or credential literals in the SQL. The database is READ-ONLY — "
        "you cannot change anything, by design."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "A single read-only SQL statement (SELECT / WITH / EXPLAIN). Example: "
                    "`SELECT count(*) FROM foundation_grants`. Reference tables live in the "
                    "`public` schema; scoring tables in `score_lab`."
                ),
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum rows to return (default 200, max 1000). Excess is truncated with a flag.",
            },
        },
        "required": ["sql"],
    },
}


registry.register(
    name="fedpulse_query",
    toolset="fedpulse_query",
    schema=FEDPULSE_QUERY_SCHEMA,
    handler=lambda args, **kw: fedpulse_query(
        sql=args.get("sql") or "",
        max_rows=args.get("max_rows") or _DEFAULT_MAX_ROWS,
    ),
    check_fn=check_fedpulse_query_requirements,
    max_result_size_chars=_MAX_RESPONSE_CHARS,
    emoji="🗄️",
)
