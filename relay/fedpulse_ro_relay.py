"""FedPulse read-only query relay (PRD-048 FR-3) — first consumer of the
shared RO-relay substrate (``relay/ro_relay_base.py``).

Grants the containerized agent bounded READ access to the live FedPulse
Postgres (``supabase_db_fedpulse``) without the credential ever entering any
container: the agent POSTs ``{"sql": ..., "max_rows": ...}`` to a UDS socket
on the existing gateway-only ``/opt/relay`` mount; this host-side relay
validates the SQL against a real PostgreSQL AST, executes it inside an
explicit read-only transaction via ``docker exec -u postgres … psql``, scrubs
the result rows with the PRD-035 egress classifier, and returns JSON rows.

Enforcement layers (Codex adversarial pass, folded 2026-07-10):

  1. **AST validation** (pglast / libpg_query — the REAL PostgreSQL parser):
     exactly one statement; ``SelectStmt`` or non-ANALYZE ``ExplainStmt``; no
     data-modifying CTEs, no ``SELECT … INTO``, no locking clauses, no utility
     statements; every schema-qualified relation/function resolves into
     ``{public, score_lab, pg_catalog}`` (closes the ``net`` PUBLIC-ACL leak,
     STOP-1); customer-table denylist (owner ruling 2026-07-19, finding #9);
     function denylist (``pg_sleep``, ``set_config``, ``dblink``, …).
  2. **Read-only execution invariant** (STOP-2 shape): every query runs inside
     ``BEGIN READ ONLY`` with ``SET LOCAL statement_timeout``, wrapped in
     ``SELECT * FROM (…) q LIMIT max_rows+1`` so the row cap is enforced
     DB-side; stdout is streamed with a hard byte cap and the whole process
     group is killed on breach/timeout. psql runs as ``-u postgres`` (never
     root), argv ``shell=False``, password via env (never argv).
  3. **Role floor**: ``sylva_ro`` NOINHERIT, SELECT-only grants,
     ``default_transaction_read_only=on`` (owner-executed SQL — see the PRD).
  4. **Result DLP** (STOP-3): every returned cell passes the egress
     classifier's redactor before it crosses back to the agent.

Plus the substrate spine: bearer, QUIESCE, ``fedpulse_ro_queries``
admission-debit, semaphore ≤2 (matches ``CONNECTION LIMIT 2``), T1 audit.

The startup self-canary proves (a) the validator still rejects every known
bypass construct, (b) a live ``SELECT 1`` round-trips, and (c) a direct write
attempt WITHOUT the validator fails at the DB (read-only txn / privileges) —
or the relay refuses to bind.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from relay import egress_classifier
from relay.ro_relay_base import (
    RelayRequestError,
    RORelayBase,
    RORelayConfig,
    run_relay,
)

logger = logging.getLogger("hermes.fedpulse_ro_relay")

# --- validator policy --------------------------------------------------------

_ALLOWED_SCHEMAS = frozenset({"public", "score_lab", "pg_catalog"})

# Owner ruling 2026-07-19 (Codex finding #9): customer-shaped tables are
# rejected at the relay by RELATION NAME regardless of qualification (they only
# exist in `public`; matching unqualified names too is deliberately over-broad
# = fail-closed). The broad DB grant stays; result DLP is the backstop. A
# `public` view wrapping one of these is checked by the deploy preflight, not
# here (the AST sees only the view's name).
_DENYLIST_RELATIONS = frozenset({
    "organizations",
    "organization_members",
    "organization_invites",
    "profiles",
    "subscriptions",
    "subscription_events",
})

_DENYLIST_FUNCTIONS = frozenset({
    "set_config",
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "pg_notify",
    "nextval", "setval", "currval", "lastval",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export",
    "dblink", "dblink_exec", "dblink_connect", "dblink_connect_u",
    "pg_terminate_backend", "pg_cancel_backend",
    "pg_reload_conf", "pg_rotate_logfile",
    "pg_switch_wal", "pg_create_restore_point",
    "query_to_xml", "database_to_xml", "schema_to_xml",
})
_DENYLIST_FUNCTION_PREFIXES = ("pg_advisory",)

# Statement node types allowed ANYWHERE in the parse tree. Any other `*Stmt`
# node (InsertStmt in a CTE, VariableSetStmt, CreateTableAsStmt, CopyStmt,
# TransactionStmt, PrepareStmt, …) → reject. Fail-closed by construction:
# future/unknown statement types are rejected without needing to be named.
_ALLOWED_STMT_NODES = frozenset({"SelectStmt", "ExplainStmt"})
# Clause nodes that are rejected outright wherever they appear.
_REJECT_NODES = frozenset({"IntoClause", "LockingClause"})

_MAX_SQL_BYTES = 16 * 1024
_DEFAULT_MAX_ROWS = 200
_MAX_ROWS_CEILING = 1000
_STDOUT_CAP_BYTES = 2 * 1024 * 1024
_STDERR_KEEP_BYTES = 64 * 1024
_STATEMENT_TIMEOUT = "15s"
_WALL_TIMEOUT_SEC = 30.0

_BUDGET_KIND = "fedpulse_ro_queries"


class SqlValidationError(Exception):
    """The SQL failed AST validation. The reason names the rule, never echoes
    secret material (it may quote schema/relation/function identifiers)."""


@dataclass(frozen=True)
class ValidatedSql:
    text: str
    is_explain: bool


def _node_type(node: dict) -> Optional[str]:
    t = node.get("@")
    return t if isinstance(t, str) else None


def _str_val(part: Any) -> str:
    """Extract the string from a pglast String node dict ({'@':'String',
    'sval': …}; older serializations used 'str')."""
    if isinstance(part, dict):
        v = part.get("sval", part.get("str"))
        if isinstance(v, str):
            return v
    if isinstance(part, str):
        return part
    raise SqlValidationError("unrecognized identifier node shape")


def _check_rangevar(node: dict) -> None:
    if node.get("catalogname"):
        raise SqlValidationError("cross-database references are not allowed")
    relname = node.get("relname")
    if isinstance(relname, str) and relname.lower() in _DENYLIST_RELATIONS:
        raise SqlValidationError(
            f"relation '{relname}' is on the customer-table denylist"
        )
    schema = node.get("schemaname")
    if isinstance(schema, str) and schema.lower() not in _ALLOWED_SCHEMAS:
        raise SqlValidationError(
            f"schema '{schema}' is outside the allowlist {sorted(_ALLOWED_SCHEMAS)}"
        )


def _check_funccall(node: dict) -> None:
    parts = [_str_val(p) for p in node.get("funcname", []) or []]
    if not parts:
        return
    if len(parts) >= 3:
        raise SqlValidationError("cross-database function references are not allowed")
    if len(parts) == 2 and parts[0].lower() not in _ALLOWED_SCHEMAS:
        raise SqlValidationError(
            f"function schema '{parts[0]}' is outside the allowlist"
        )
    name = parts[-1].lower()
    if name in _DENYLIST_FUNCTIONS or name.startswith(_DENYLIST_FUNCTION_PREFIXES):
        raise SqlValidationError(f"function '{name}' is denylisted")


def _walk(node: Any) -> None:
    if isinstance(node, dict):
        t = _node_type(node)
        if t is not None:
            if t.endswith("Stmt") and t not in _ALLOWED_STMT_NODES:
                raise SqlValidationError(f"statement type {t} is not allowed")
            if t in _REJECT_NODES:
                clause = "SELECT ... INTO" if t == "IntoClause" else "FOR UPDATE/SHARE"
                raise SqlValidationError(f"{clause} is not allowed")
            if t == "RangeVar":
                _check_rangevar(node)
            elif t == "FuncCall":
                _check_funccall(node)
        for value in node.values():
            _walk(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _walk(item)


def validate_sql(sql: str) -> ValidatedSql:
    """Fail-closed AST validation. Returns a ValidatedSql or raises
    SqlValidationError. ANY parse/serialization surprise rejects."""
    if not isinstance(sql, str) or not sql.strip():
        raise SqlValidationError("sql must be a non-empty string")
    if "\x00" in sql:
        raise SqlValidationError("embedded NUL is not allowed")
    if len(sql.encode("utf-8", errors="strict")) > _MAX_SQL_BYTES:
        raise SqlValidationError(f"sql exceeds {_MAX_SQL_BYTES} bytes")

    import pglast

    try:
        statements = pglast.parse_sql(sql)
    except Exception as exc:
        # A parse failure covers psql backslash meta-commands too: `\!`, `\copy`
        # etc. are not valid SQL, so anything psql would treat as a meta-command
        # can never reach the executor.
        raise SqlValidationError(f"SQL parse failed: {type(exc).__name__}")

    if len(statements) != 1:
        raise SqlValidationError(
            f"exactly one statement required (got {len(statements)})"
        )

    try:
        tree = statements[0].stmt(skip_none=True)
    except Exception:
        raise SqlValidationError("AST serialization failed")
    if not isinstance(tree, dict):
        raise SqlValidationError("unexpected AST shape")

    top = _node_type(tree)
    is_explain = top == "ExplainStmt"
    if top not in _ALLOWED_STMT_NODES:
        raise SqlValidationError(f"statement type {top} is not allowed")

    if is_explain:
        for opt in tree.get("options", []) or []:
            if isinstance(opt, dict) and str(opt.get("defname", "")).lower() == "analyze":
                raise SqlValidationError("EXPLAIN ANALYZE is not allowed (it executes)")
        inner = tree.get("query")
        if not isinstance(inner, dict) or _node_type(inner) != "SelectStmt":
            raise SqlValidationError("EXPLAIN is only allowed on a SELECT")

    _walk(tree)
    return ValidatedSql(text=sql, is_explain=is_explain)


def build_script(vq: ValidatedSql, max_rows: int) -> str:
    """Wrap the validated query in the read-only execution envelope. The
    ``BEGIN READ ONLY`` makes read-only an execution invariant (not just the
    role default); the outer LIMIT bounds rows DB-side (STOP-2). EXPLAIN is
    not wrappable in a subquery and never returns bulk rows, so it runs bare
    inside the same envelope."""
    body = vq.text.rstrip().rstrip(";").rstrip()
    if vq.is_explain:
        core = f"{body};"
    else:
        core = f"SELECT * FROM (\n{body}\n) __ro_q LIMIT {max_rows + 1};"
    return (
        "BEGIN READ ONLY;\n"
        f"SET LOCAL statement_timeout = '{_STATEMENT_TIMEOUT}';\n"
        "SET LOCAL search_path = public, score_lab;\n"
        f"{core}\n"
        "COMMIT;\n"
    )


# --- relay adapter -----------------------------------------------------------

@dataclass(frozen=True)
class FedpulseDbConfig:
    container: str
    db_name: str
    db_host: str
    db_port: int
    db_user: str
    password_file: Path
    wall_timeout_sec: float = _WALL_TIMEOUT_SEC
    stdout_cap_bytes: int = _STDOUT_CAP_BYTES


@dataclass(frozen=True)
class _ValidatedRequest:
    vq: ValidatedSql
    max_rows: int


class FedpulseRORelay(RORelayBase):
    """AST-validated read-only SQL over the shared relay spine."""

    def __init__(self, config: RORelayConfig, db: FedpulseDbConfig) -> None:
        super().__init__(config)
        self.db = db

    # -- request side ---------------------------------------------------------

    def validate_request(self, body: dict[str, Any]) -> _ValidatedRequest:
        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise RelayRequestError(400, "sql (string) required")
        max_rows = body.get("max_rows", _DEFAULT_MAX_ROWS)
        if not isinstance(max_rows, int) or isinstance(max_rows, bool) \
                or not (1 <= max_rows <= _MAX_ROWS_CEILING):
            raise RelayRequestError(400, f"max_rows must be an int in [1, {_MAX_ROWS_CEILING}]")
        try:
            vq = validate_sql(sql)
        except SqlValidationError as exc:
            raise RelayRequestError(422, f"sql rejected: {exc}")
        return _ValidatedRequest(vq=vq, max_rows=max_rows)

    def request_text(self, validated: _ValidatedRequest) -> str:
        return validated.vq.text

    # -- executor (STOP-2 shape) ----------------------------------------------

    def _read_password(self) -> str:
        pw = self.db.password_file.read_text(encoding="utf-8").strip()
        if not pw:
            raise RelayRequestError(502, "db password file is empty")
        return pw

    def _argv(self) -> list[str]:
        # Bare `-e PGPASSWORD` forwards the variable from the docker CLIENT's
        # env — the secret never appears in argv (no `ps` exposure) and never
        # lands in any container config. `-u postgres` NOT root (NEEDS-FIX-6);
        # `-w` = never prompt (fail instead of hang on an auth surprise).
        return [
            "docker", "exec", "-i", "-u", "postgres", "-e", "PGPASSWORD",
            self.db.container,
            "psql", "-X", "-q", "-w", "--csv", "-v", "ON_ERROR_STOP=1",
            "-h", self.db.db_host, "-p", str(self.db.db_port),
            "-U", self.db.db_user, "-d", self.db.db_name,
        ]

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def _run_script(self, script: str) -> tuple[int, bytes, bytes, Optional[str]]:
        """Run a SQL script through psql with streamed, capped output. Returns
        (returncode, stdout, stderr, kill_reason). Never captures unbounded
        output (STOP-2): stdout is killed at the byte cap, stderr is drained
        but only the head is kept, and on timeout/breach the whole process
        GROUP is SIGKILLed (start_new_session=True makes psql the group lead)."""
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", str(Path.home())),
            "PGPASSWORD": self._read_password(),
        }
        try:
            proc = subprocess.Popen(
                self._argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, start_new_session=True,
            )
        except OSError as exc:
            raise RelayRequestError(502, f"executor spawn failed: {type(exc).__name__}")

        try:
            try:
                proc.stdin.write(script.encode("utf-8"))
                proc.stdin.close()
            except BrokenPipeError:
                pass

            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ, "out")
            sel.register(proc.stderr, selectors.EVENT_READ, "err")
            out, err = bytearray(), bytearray()
            kill_reason: Optional[str] = None
            deadline = time.monotonic() + self.db.wall_timeout_sec
            open_streams = 2
            try:
                while open_streams and kill_reason is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        kill_reason = "wall-timeout"
                        break
                    for key, _mask in sel.select(min(remaining, 0.5)):
                        chunk = key.fileobj.read1(65536)  # type: ignore[union-attr]
                        if not chunk:
                            sel.unregister(key.fileobj)
                            open_streams -= 1
                            continue
                        if key.data == "out":
                            out += chunk
                            if len(out) > self.db.stdout_cap_bytes:
                                kill_reason = "response-cap"
                                break
                        else:
                            # keep only the head; keep DRAINING so a chatty
                            # stderr can never deadlock the child.
                            if len(err) < _STDERR_KEEP_BYTES:
                                err += chunk[: _STDERR_KEEP_BYTES - len(err)]
            finally:
                sel.close()

            if kill_reason:
                self._kill_group(proc)
                return -9, bytes(out), bytes(err), kill_reason
            try:
                rc = proc.wait(timeout=max(deadline - time.monotonic(), 1.0))
            except subprocess.TimeoutExpired:
                self._kill_group(proc)
                return -9, bytes(out), bytes(err), "wall-timeout"
            return rc, bytes(out), bytes(err), None
        except Exception:
            self._kill_group(proc)
            raise

    def execute(self, validated: _ValidatedRequest) -> tuple[bool, dict[str, Any]]:
        script = build_script(validated.vq, validated.max_rows)
        rc, out, err, kill_reason = self._run_script(script)

        if kill_reason == "response-cap":
            raise RelayRequestError(
                400, f"response exceeded the {self.db.stdout_cap_bytes // (1024*1024)} MB cap; "
                     "narrow the query or lower max_rows")
        if kill_reason == "wall-timeout":
            raise RelayRequestError(400, "query exceeded the relay wall-timeout")
        if rc != 0:
            # psql's error text may quote the caller's own SQL — that is fine —
            # but scrub it through the classifier's redactor regardless.
            snippet = egress_classifier.redact(
                err.decode("utf-8", errors="replace").strip()[:2000])
            raise RelayRequestError(400, f"query failed: {snippet or f'psql exited {rc}'}")

        text = out.decode("utf-8", errors="replace")
        try:
            parsed = list(csv.reader(io.StringIO(text)))
        except Exception:
            raise RelayRequestError(502, "result CSV parse failed")
        if not parsed:
            return True, {"columns": [], "rows": [], "row_count": 0, "truncated": False}
        columns, data = parsed[0], parsed[1:]
        truncated = len(data) > validated.max_rows
        data = data[: validated.max_rows]
        return True, {
            "columns": columns,
            "rows": data,
            "row_count": len(data),
            "truncated": truncated,
        }

    # -- result DLP (STOP-3) --------------------------------------------------

    def scrub_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        redacted_cells = 0

        def _scrub_cell(cell: Any) -> Any:
            nonlocal redacted_cells
            if not isinstance(cell, str):
                return cell
            masked = egress_classifier.redact(cell)
            if masked != cell:
                redacted_cells += 1
            return masked

        payload["rows"] = [[_scrub_cell(c) for c in row] for row in payload.get("rows", [])]
        payload["columns"] = [_scrub_cell(c) for c in payload.get("columns", [])]
        payload["dlp_redacted"] = redacted_cells > 0
        if redacted_cells:
            self._audit("dlp", action="result redacted: credential shape in rows",
                        rationale=f"egress classifier masked {redacted_cells} cell(s)",
                        outcome="redacted")
        return payload

    # -- startup self-canary --------------------------------------------------

    # Constructs the validator MUST reject; validated at every startup so a
    # pglast upgrade or a bad edit can never silently widen the gate.
    _CANARY_MUST_REJECT = (
        "INSERT INTO public.t VALUES (1)",
        "UPDATE public.t SET a = 1",
        "DELETE FROM public.t",
        "DROP TABLE public.t",
        "CREATE TABLE public.t (a int)",
        "TRUNCATE public.t",
        "SET default_transaction_read_only = off",
        "SELECT 1; SELECT 2",
        "WITH w AS (INSERT INTO public.t VALUES (1) RETURNING *) SELECT * FROM w",
        "SELECT * FROM public.t FOR UPDATE",
        "SELECT a INTO public.t2 FROM public.t",
        "EXPLAIN (ANALYZE) INSERT INTO public.t VALUES (1)",
        "EXPLAIN ANALYZE SELECT 1",
        "SELECT * FROM net._http_response",
        "SELECT * FROM auth.users",
        "SELECT * FROM vault.secrets",
        "SELECT * FROM profiles",
        "SELECT * FROM public.organization_invites",
        "SELECT pg_sleep(10)",
        "SELECT set_config('default_transaction_read_only', 'off', true)",
        "COPY public.t TO '/tmp/x'",
        "PREPARE p AS SELECT 1",
        "COMMIT",
    )

    def self_canary(self) -> bool:
        # (a) validator invariants — pure, no DB.
        try:
            validate_sql("SELECT 1")
            validate_sql("EXPLAIN SELECT 1")
        except SqlValidationError as exc:
            logger.error("canary: validator rejected a benign probe: %s", exc)
            return False
        for q in self._CANARY_MUST_REJECT:
            try:
                validate_sql(q)
            except SqlValidationError:
                continue
            logger.error("canary FAILED: validator ACCEPTED a must-reject construct: %r", q)
            return False

        # (b) live round-trip through the full validated path.
        try:
            ok, payload = self.execute(
                _ValidatedRequest(vq=validate_sql("SELECT 1"), max_rows=1))
        except Exception as exc:
            logger.error("canary: live SELECT 1 failed (%s) — DB/role/executor not ready", exc)
            return False
        if not ok or payload.get("rows") != [["1"]]:
            logger.error("canary: live SELECT 1 returned unexpected result")
            return False

        # (c) direct write drill WITHOUT the validator (AC-003b as a startup
        # invariant): must fail at the DB via read-only txn and/or privileges.
        # Both layers must fail simultaneously for this to mutate anything.
        drill = (
            "BEGIN READ ONLY;\n"
            "INSERT INTO public.foundation_grants DEFAULT VALUES;\n"
            "COMMIT;\n"
        )
        try:
            rc, _out, err, kill_reason = self._run_script(drill)
        except Exception as exc:
            logger.error("canary: write-drill executor error: %s", exc)
            return False
        err_text = err.decode("utf-8", errors="replace")
        if rc == 0 and kill_reason is None:
            logger.critical(
                "canary FAILED: direct write attempt SUCCEEDED — read-only "
                "transaction AND role privileges both failed; refusing to bind")
            return False
        if "read-only transaction" not in err_text and "permission denied" not in err_text:
            logger.error("canary: write drill failed for an unexpected reason "
                         "(rc=%s, kill=%s) — refusing to bind", rc, kill_reason)
            return False
        logger.info("self-canary passed (validator + live SELECT + write drill)")
        return True


# --- config / main -----------------------------------------------------------

def _resolve_configs_from_env() -> tuple[RORelayConfig, FedpulseDbConfig]:
    base = Path(os.environ.get("HERMES_RELAY_DIR", str(Path.home() / ".hermes-relay")))
    # The DB password lives OUTSIDE the relay dir on purpose: ~/.hermes-relay is
    # bind-mounted (ro) into the gateway container, and the credential must
    # never enter any container mount. ~/.hermes-relay-private is host-only.
    private = Path(os.environ.get(
        "HERMES_RELAY_PRIVATE_DIR", str(Path.home() / ".hermes-relay-private")))
    relay_cfg = RORelayConfig(
        socket_path=os.environ.get("FEDPULSE_RO_SOCKET", str(base / "fedpulse-ro.sock")),
        token_path=os.environ.get("FEDPULSE_RO_TOKEN", str(base / "fedpulse.token")),
        route="/query",
        budget_kind=_BUDGET_KIND,
        audit_surface_prefix="fedpulse_ro",
        audit_tier="T1",
        max_body_bytes=64 * 1024,
        concurrency=2,  # matches the role's CONNECTION LIMIT 2
        queue_wait_sec=2.0,
    )
    db_cfg = FedpulseDbConfig(
        container=os.environ.get("FEDPULSE_DB_CONTAINER", "supabase_db_fedpulse"),
        db_name=os.environ.get("FEDPULSE_DB_NAME", "postgres"),
        db_host=os.environ.get("FEDPULSE_DB_HOST", "127.0.0.1"),
        db_port=int(os.environ.get("FEDPULSE_DB_PORT", "5432")),
        db_user=os.environ.get("FEDPULSE_DB_USER", "sylva_ro"),
        password_file=Path(os.environ.get(
            "FEDPULSE_RO_PASSWORD_FILE", str(private / "fedpulse-db.pass"))),
    )
    return relay_cfg, db_cfg


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    relay_cfg, db_cfg = _resolve_configs_from_env()

    try:
        from hermes_cli.config import cfg_get, read_raw_config
        if cfg_get(read_raw_config(), "autonomy", "audit_enabled") is False:
            logger.warning("autonomy.audit_enabled is FALSE — relay T1 audit rows will NOT be recorded")
    except Exception:
        pass

    try:
        relay = FedpulseRORelay(relay_cfg, db_cfg)
    except Exception as exc:
        logger.error("relay construction failed (token missing/unreadable?): %s", exc)
        return 4

    return run_relay(relay, extra_secret_files=[(db_cfg.password_file, "db password")])


if __name__ == "__main__":
    raise SystemExit(main())
