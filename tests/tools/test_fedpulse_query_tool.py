"""Tests for fedpulse_query routed through the host FedPulse RO relay (PRD-048).

The tool forwards a single read-only SQL statement to the relay over a unix
socket. These tests stub the relay call and assert the tool's fail-closed
behavior, result shaping, error-status mapping, and the DLP flag passthrough.
No real socket / DB.
"""

import json
from unittest.mock import patch

import pytest

import tools.fedpulse_query_tool as t


def _relay_ok(columns=None, rows=None, **extra):
    body = {
        "columns": columns if columns is not None else ["count"],
        "rows": rows if rows is not None else [["42"]],
        "row_count": len(rows) if rows is not None else 1,
        "truncated": False,
        "dlp_redacted": False,
    }
    body.update(extra)
    return (200, body)


@pytest.fixture(autouse=True)
def _relay_present(monkeypatch):
    # Default: relay reachable, bearer readable, not cron.
    monkeypatch.setattr(t, "_relay_available", lambda: True)
    monkeypatch.setattr(t, "_read_bearer", lambda: "test-token")
    monkeypatch.setattr(t, "_in_cron_context", lambda: False)


class TestHappyPath:
    def test_select_returns_rows(self):
        with patch("tools.fedpulse_query_tool._call_relay",
                   return_value=_relay_ok(["count"], [["42"]])):
            out = json.loads(t.fedpulse_query(sql="SELECT count(*) FROM foundation_grants"))
        assert out["success"] is True
        assert out["columns"] == ["count"]
        assert out["rows"] == [["42"]]
        assert out["row_count"] == 1

    def test_max_rows_forwarded_and_clamped(self):
        with patch("tools.fedpulse_query_tool._call_relay", return_value=_relay_ok()) as call:
            t.fedpulse_query(sql="SELECT 1", max_rows=5000)
        # positional args: (sql, max_rows, surface) — clamped to the 1000 ceiling
        assert call.call_args.args[1] == 1000

    def test_max_rows_loose_type_coerced(self):
        with patch("tools.fedpulse_query_tool._call_relay", return_value=_relay_ok()) as call:
            t.fedpulse_query(sql="SELECT 1", max_rows="not-an-int")
        assert call.call_args.args[1] == t._DEFAULT_MAX_ROWS


class TestRelayDownFailsClosed:
    def test_relay_unavailable_refuses_no_fallback(self, monkeypatch):
        monkeypatch.setattr(t, "_relay_available", lambda: False)
        out = json.loads(t.fedpulse_query(sql="SELECT 1"))
        assert out["success"] is False
        assert out.get("blocked") == "relay_down"

    def test_relay_transport_error_fails_closed(self):
        with patch("tools.fedpulse_query_tool._call_relay", side_effect=OSError("connrefused")):
            out = json.loads(t.fedpulse_query(sql="SELECT 1"))
        assert out["success"] is False
        assert out.get("blocked") == "relay_error"


class TestSecretRefusal:
    @pytest.mark.parametrize("secret", [
        "sk-ant-api03-AbCdEf0123456789xyzTOKEN",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ])
    def test_secret_in_sql_refused_no_call(self, secret):
        with patch("tools.fedpulse_query_tool._call_relay") as call:
            out = json.loads(t.fedpulse_query(sql=f"SELECT '{secret}'"))
        assert out["success"] is False
        assert out.get("blocked") == "secret_in_sql"
        call.assert_not_called()


class TestRelayErrors:
    @pytest.mark.parametrize("status,blocked", [
        (400, "query_rejected"), (401, "relay_auth"), (422, "query_rejected"),
        (429, "budget"), (503, "quiesced_or_busy"),
    ])
    def test_relay_error_statuses_mapped(self, status, blocked):
        with patch("tools.fedpulse_query_tool._call_relay",
                   return_value=(status, {"error": "nope"})):
            out = json.loads(t.fedpulse_query(sql="SELECT 1"))
        assert out["success"] is False
        assert out.get("blocked") == blocked


class TestResultShaping:
    def test_dlp_flag_passthrough(self):
        with patch("tools.fedpulse_query_tool._call_relay",
                   return_value=_relay_ok(["tok"], [["[REDACTED-CREDENTIAL]"]], dlp_redacted=True)):
            out = json.loads(t.fedpulse_query(sql="SELECT tok FROM foundation_grants"))
        assert out["dlp_redacted"] is True

    def test_truncated_flag_passthrough(self):
        with patch("tools.fedpulse_query_tool._call_relay",
                   return_value=_relay_ok(["g"], [["1"], ["2"]], row_count=2, truncated=True)):
            out = json.loads(t.fedpulse_query(sql="SELECT g FROM generate_series(1,9) g", max_rows=2))
        assert out["truncated"] is True

    def test_oversize_result_capped(self):
        big_rows = [["z" * 5000] for _ in range(20)]
        with patch("tools.fedpulse_query_tool._call_relay",
                   return_value=_relay_ok(["x"], big_rows, row_count=20)):
            out = json.loads(t.fedpulse_query(sql="SELECT x FROM t"))
        assert out.get("truncated") is True
        assert "truncated" in out["result"]


class TestInputValidation:
    def test_empty_sql_rejected(self):
        out = json.loads(t.fedpulse_query(sql="   "))
        assert out["success"] is False

    def test_oversize_sql_rejected(self):
        out = json.loads(t.fedpulse_query(sql="SELECT '" + "a" * (t._MAX_SQL_CHARS + 1) + "'"))
        assert out["success"] is False


class TestSurfaceLabel:
    def test_cron_surface_forwarded(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        with patch("tools.fedpulse_query_tool._call_relay", return_value=_relay_ok()) as call:
            t.fedpulse_query(sql="SELECT 1")
        # surface (3rd positional arg) reflects the run identity (cron floor)
        assert call.call_args.args[2] in ("cron", "delegated_child", "orchestrated_headless", "proactive")


class TestAvailability:
    def test_check_fn_true_when_relay_present(self, monkeypatch):
        monkeypatch.setattr(t, "_relay_available", lambda: True)
        assert t.check_fedpulse_query_requirements() is True

    def test_check_fn_false_when_relay_absent(self, monkeypatch):
        monkeypatch.setattr(t, "_relay_available", lambda: False)
        assert t.check_fedpulse_query_requirements() is False
