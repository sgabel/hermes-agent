"""FedPulse RO relay tests (PRD-048 FR-3).

Two layers, both hermetic (no live DB, no docker):

  * ``validate_sql`` — the AST gate. Every AC-003(a)/AC-004b bypass construct is
    asserted-rejected; benign SELECT/EXPLAIN accepted. This is the security core
    (STOP-1 net leak, data-modifying CTE, FOR UPDATE, INTO, EXPLAIN ANALYZE,
    excluded/denylisted schemas, customer-table denylist, function denylist).

  * The relay pipeline — the substrate spine (bearer / classifier / kill-switch /
    budget admission-debit / concurrency / result DLP / audit), with ``execute``
    stubbed so no docker/psql runs.
"""

from __future__ import annotations

import pytest

from relay import fedpulse_ro_relay as fr
from relay.fedpulse_ro_relay import (
    FedpulseDbConfig,
    FedpulseRORelay,
    SqlValidationError,
    build_script,
    validate_sql,
)
from relay.ro_relay_base import RORelayConfig


# ---------------------------------------------------------------------------
# AST validator — the security core.
# ---------------------------------------------------------------------------

ACCEPT = [
    "SELECT 1",
    "SELECT count(*) FROM foundation_grants",
    "SELECT count(*) FROM public.foundation_grants",
    "SELECT * FROM score_lab.some_table LIMIT 5",
    "WITH t AS (SELECT 1 AS a) SELECT a FROM t",
    "SELECT g.id FROM foundation_grants g JOIN public.other o ON o.id = g.id",
    "EXPLAIN SELECT 1",
    "EXPLAIN (FORMAT JSON) SELECT * FROM foundation_grants",
    "SELECT now()",
    "SELECT lower(name) FROM foundation_grants",
]

REJECT = [
    # multi-statement
    ("SELECT 1; SELECT 2", "one statement"),
    # DML / DDL
    ("INSERT INTO public.t VALUES (1)", "not allowed"),
    ("UPDATE public.t SET a=1", "not allowed"),
    ("DELETE FROM public.t", "not allowed"),
    ("DROP TABLE public.t", "not allowed"),
    ("CREATE TABLE public.t (a int)", "not allowed"),
    ("TRUNCATE public.t", "not allowed"),
    ("ALTER TABLE public.t ADD COLUMN b int", "not allowed"),
    # SET / session mutation
    ("SET default_transaction_read_only = off", "not allowed"),
    # data-modifying CTE (NEEDS-FIX-4)
    ("WITH w AS (INSERT INTO public.t VALUES (1) RETURNING *) SELECT * FROM w", "not allowed"),
    ("WITH w AS (UPDATE public.t SET a=1 RETURNING *) SELECT * FROM w", "not allowed"),
    ("WITH w AS (DELETE FROM public.t RETURNING *) SELECT * FROM w", "not allowed"),
    # locking clause
    ("SELECT * FROM public.t FOR UPDATE", "FOR UPDATE"),
    ("SELECT * FROM public.t FOR SHARE", "FOR UPDATE"),
    # SELECT INTO
    ("SELECT a INTO public.t2 FROM public.t", "INTO"),
    # EXPLAIN ANALYZE (executes)
    ("EXPLAIN ANALYZE SELECT 1", "ANALYZE"),
    ("EXPLAIN (ANALYZE) INSERT INTO public.t VALUES (1)", "ANALYZE"),
    ("EXPLAIN INSERT INTO public.t VALUES (1)", "only allowed on a SELECT"),
    # excluded schema (STOP-1 net PUBLIC-ACL leak — AC-004b)
    ("SELECT * FROM net._http_response", "outside the allowlist"),
    ("SELECT * FROM net.http_request_queue", "outside the allowlist"),
    ("SELECT * FROM auth.users", "outside the allowlist"),
    ("SELECT * FROM vault.secrets", "outside the allowlist"),
    ("SELECT * FROM storage.objects", "outside the allowlist"),
    # customer-table denylist (finding #9) — qualified and bare
    ("SELECT * FROM profiles", "denylist"),
    ("SELECT * FROM public.profiles", "denylist"),
    ("SELECT * FROM organization_invites", "denylist"),
    ("SELECT * FROM public.subscriptions", "denylist"),
    ("SELECT email FROM public.organization_members", "denylist"),
    ("SELECT * FROM subscription_events", "denylist"),
    ("SELECT * FROM organizations", "denylist"),
    # function denylist
    ("SELECT pg_sleep(10)", "denylisted"),
    ("SELECT set_config('x','y',true)", "denylisted"),
    ("SELECT pg_read_file('/etc/passwd')", "denylisted"),
    ("SELECT dblink('', '')", "denylisted"),
    ("SELECT pg_advisory_lock(1)", "denylisted"),
    ("SELECT nextval('s')", "denylisted"),
    ("SELECT pg_terminate_backend(1)", "denylisted"),
    # utility statements
    ("COPY public.t TO '/tmp/x'", "not allowed"),
    ("PREPARE p AS SELECT 1", "not allowed"),
    ("COMMIT", "not allowed"),
    ("BEGIN", "not allowed"),
    ("VACUUM", "not allowed"),
    # cross-db
    ("SELECT * FROM otherdb.public.t", "cross-database"),
    # meta / junk / NUL
    ("\\copy public.t to 'x'", "parse failed"),
    ("not sql at all !!!", "parse failed"),
]


@pytest.mark.parametrize("sql", ACCEPT)
def test_validator_accepts_benign(sql):
    vq = validate_sql(sql)
    assert vq.text == sql


@pytest.mark.parametrize("sql,needle", REJECT)
def test_validator_rejects_bypass(sql, needle):
    with pytest.raises(SqlValidationError) as ei:
        validate_sql(sql)
    assert needle.lower() in str(ei.value).lower()


def test_validator_rejects_nul_and_oversize():
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT 1\x00")
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT '" + "a" * (17 * 1024) + "'")
    with pytest.raises(SqlValidationError):
        validate_sql("   ")


def test_denylist_via_alias_still_rejected():
    # aliasing does not launder the relation name
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT p.email FROM public.profiles p")


def test_build_script_wraps_select_readonly_with_limit():
    s = build_script(validate_sql("SELECT * FROM foundation_grants"), 50)
    assert s.startswith("BEGIN READ ONLY;")
    assert "statement_timeout" in s
    assert "LIMIT 51" in s  # max_rows + 1
    assert s.rstrip().endswith("COMMIT;")


def test_build_script_explain_runs_bare_no_subquery_wrap():
    s = build_script(validate_sql("EXPLAIN SELECT 1"), 50)
    assert "BEGIN READ ONLY;" in s
    assert "__ro_q" not in s  # EXPLAIN is not subquery-wrappable
    assert "EXPLAIN SELECT 1;" in s


# ---------------------------------------------------------------------------
# Relay pipeline — substrate spine with execute() stubbed.
# ---------------------------------------------------------------------------

@pytest.fixture
def db_cfg(tmp_path):
    pw = tmp_path / "fedpulse-db.pass"
    pw.write_text("db-secret-not-a-bearer", encoding="utf-8")
    return FedpulseDbConfig(
        container="supabase_db_fedpulse", db_name="postgres",
        db_host="127.0.0.1", db_port=5432, db_user="sylva_ro",
        password_file=pw,
    )


@pytest.fixture
def relay_cfg(tmp_path):
    token = tmp_path / "fedpulse.token"
    token.write_text("test-bearer-secret-value", encoding="utf-8")
    return RORelayConfig(
        socket_path=str(tmp_path / "fedpulse-ro.sock"),
        token_path=str(token),
        route="/query",
        budget_kind="fedpulse_ro_queries",
        audit_surface_prefix="fedpulse_ro",
        concurrency=2,
    )


@pytest.fixture
def relay(relay_cfg, db_cfg):
    return FedpulseRORelay(relay_cfg, db_cfg)


@pytest.fixture
def stub_autonomy(monkeypatch):
    """Patch the REAL autonomy.{killswitch,budget,audit} module functions.

    The relay imports these lazily via ``from autonomy import <mod>``, which
    resolves the attribute on the already-imported ``autonomy`` package — so a
    ``sys.modules`` swap is bypassed once another test (e.g. test_budget.py) has
    imported the real submodule during collection. Patching the module
    functions in place is order-independent and mirrors production (the relay
    really does call the real module)."""
    import importlib

    state = {"quiesced": False, "allowed": True, "debits": [], "audits": []}

    ks = importlib.import_module("autonomy.killswitch")
    bud = importlib.import_module("autonomy.budget")
    aud = importlib.import_module("autonomy.audit")

    def _debit(surface, kind, amount=1, *, audit=True):
        state["debits"].append((surface, kind, amount))
        return {"allowed": state["allowed"], "degrade": not state["allowed"],
                "kind": kind, "usage": {}}

    monkeypatch.setattr(ks, "guard", lambda surface: state["quiesced"])
    monkeypatch.setattr(bud, "debit", _debit)
    monkeypatch.setattr(aud, "record", lambda **kw: state["audits"].append(kw))
    return state


def _ok_rows(monkeypatch, relay, columns, rows):
    def _fake_execute(validated):
        return True, {"columns": list(columns), "rows": [list(r) for r in rows],
                      "row_count": len(rows), "truncated": False}
    monkeypatch.setattr(relay, "execute", _fake_execute)


def test_pipeline_happy_path(relay, stub_autonomy, monkeypatch):
    _ok_rows(monkeypatch, relay, ["count"], [["42"]])
    status, payload = relay.handle_request({"sql": "SELECT count(*) FROM foundation_grants"})
    assert status == 200
    assert payload["rows"] == [["42"]]
    assert stub_autonomy["debits"] == [("fedpulse_ro:relay", "fedpulse_ro_queries", 1)]
    assert any(a["outcome"] == "ok" for a in stub_autonomy["audits"])


def test_pipeline_rejects_bad_sql_before_execute(relay, stub_autonomy, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(relay, "execute", lambda v: called.__setitem__("n", called["n"] + 1) or (True, {}))
    status, payload = relay.handle_request({"sql": "SELECT * FROM auth.users"})
    assert status == 422
    assert called["n"] == 0  # never reached the executor
    # rejected at validation → NOT debited (validation precedes the budget stage)
    assert stub_autonomy["debits"] == []


def test_pipeline_kill_switch_blocks(relay, stub_autonomy, monkeypatch):
    _ok_rows(monkeypatch, relay, ["c"], [["1"]])
    stub_autonomy["quiesced"] = True
    status, payload = relay.handle_request({"sql": "SELECT 1"})
    assert status == 503
    assert stub_autonomy["debits"] == []  # kill-switch precedes debit


def test_pipeline_budget_cap_refuses(relay, stub_autonomy, monkeypatch):
    _ok_rows(monkeypatch, relay, ["c"], [["1"]])
    stub_autonomy["allowed"] = False
    status, payload = relay.handle_request({"sql": "SELECT 1"})
    assert status == 429


def test_pipeline_debit_is_admission_before_execute(relay, stub_autonomy, monkeypatch):
    import importlib
    order = []
    bud = importlib.import_module("autonomy.budget")
    orig_debit = bud.debit

    def _tracking_debit(*a, **k):
        order.append("debit")
        return orig_debit(*a, **k)
    monkeypatch.setattr(bud, "debit", _tracking_debit)

    def _exec(v):
        order.append("execute")
        return True, {"columns": [], "rows": [], "row_count": 0, "truncated": False}
    monkeypatch.setattr(relay, "execute", _exec)

    relay.handle_request({"sql": "SELECT 1"})
    assert order == ["debit", "execute"]  # debit-on-admission, never after


def test_result_dlp_redacts_credential_shaped_cell(relay, stub_autonomy, monkeypatch):
    token = "ghp_" + "a" * 36
    _ok_rows(monkeypatch, relay, ["token"], [[token], ["benign-value"]])
    status, payload = relay.handle_request({"sql": "SELECT token FROM foundation_grants"})
    assert status == 200
    assert token not in str(payload["rows"])
    assert payload["dlp_redacted"] is True
    assert payload["rows"][1] == ["benign-value"]  # benign row untouched


def test_bearer_ok_and_bad(relay):
    assert relay._bearer_ok("Bearer test-bearer-secret-value") is True
    assert relay._bearer_ok("Bearer wrong") is False
    assert relay._bearer_ok(None) is False


def test_bearer_in_payload_refused(relay, stub_autonomy, monkeypatch):
    _ok_rows(monkeypatch, relay, ["c"], [["1"]])
    # the bearer value itself embedded in the SQL → refuse (422) before execute
    status, payload = relay.handle_request(
        {"sql": "SELECT 'test-bearer-secret-value' AS x"})
    assert status == 422


def test_max_rows_bounds(relay, stub_autonomy, monkeypatch):
    _ok_rows(monkeypatch, relay, ["c"], [["1"]])
    for bad in [0, -1, 5000, "10", 3.5, True]:
        status, _ = relay.handle_request({"sql": "SELECT 1", "max_rows": bad})
        assert status == 400


def test_argv_is_non_root_shell_false_no_password(relay):
    argv = relay._argv()
    assert argv[0] == "docker"
    assert "-u" in argv and argv[argv.index("-u") + 1] == "postgres"
    assert "sylva_ro" in argv  # -U user
    # password is forwarded via env name only — never appears as an argv value
    assert "db-secret-not-a-bearer" not in argv
    assert "PGPASSWORD" in argv  # the NAME is forwarded (bare -e), value from env
