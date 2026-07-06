"""HTTP-over-UDS layer tests (PRD-035) — the request parsing / framing / bearer
extraction the unit handle_consult tests don't exercise. Drives a real bound
relay socket with a real HTTP client, stubbing only the claude spawn + autonomy.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time
import types

import pytest

from relay.advisory_relay import AdvisoryRelay, RelayConfig, create_server


class _UnixConn(http.client.HTTPConnection):
    def __init__(self, path, timeout=10):
        super().__init__("localhost", timeout=timeout)
        self._p = path

    def connect(self):
        so = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        so.settimeout(self.timeout)
        so.connect(self._p)
        self.sock = so


@pytest.fixture
def live_relay(tmp_path, monkeypatch):
    # Stub autonomy so no real ledger is touched.
    state = {"quiesced": False, "allowed": True}
    ks = types.ModuleType("autonomy.killswitch"); ks.guard = lambda s: state["quiesced"]
    bud = types.ModuleType("autonomy.budget")
    bud.debit = lambda *a, **k: {"allowed": state["allowed"], "degrade": not state["allowed"]}
    aud = types.ModuleType("autonomy.audit"); aud.record = lambda **k: None
    pkg = types.ModuleType("autonomy")
    for n, m in [("autonomy", pkg), ("autonomy.killswitch", ks),
                 ("autonomy.budget", bud), ("autonomy.audit", aud)]:
        monkeypatch.setitem(sys.modules, n, m)

    token = tmp_path / "client.token"; token.write_text("live-bearer", encoding="utf-8")
    os.chmod(token, 0o600)
    sock = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), f"relaytest-{os.getpid()}.sock")
    cfg = RelayConfig(socket_path=sock, token_path=str(token), model="pinned",
                      isolated_home=str(tmp_path / "home"))
    relay = AdvisoryRelay(cfg)
    relay._canary_binary_sig = relay._binary_signature()
    monkeypatch.setattr(relay, "_spawn_claude", lambda a: (True, "advisory body"))
    server = create_server(relay)
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    time.sleep(0.15)
    yield sock, state
    server.shutdown()
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass


def _post(sock, body, bearer="live-bearer", path="/consult"):
    c = _UnixConn(sock)
    try:
        raw = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(raw))}
        if bearer is not None:
            headers["Authorization"] = f"Bearer {bearer}"
        c.request("POST", path, body=raw, headers=headers)
        r = c.getresponse()
        return r.status, json.loads(r.read().decode())
    finally:
        c.close()


def test_health_no_bearer(live_relay):
    sock, _ = live_relay
    c = _UnixConn(sock)
    try:
        c.request("GET", "/health")
        r = c.getresponse()
        assert r.status == 200
        assert json.loads(r.read())["status"] == "ok"
    finally:
        c.close()


def test_missing_bearer_401(live_relay):
    sock, _ = live_relay
    st, _ = _post(sock, {"prompt": "x"}, bearer=None)
    assert st == 401


def test_wrong_bearer_401(live_relay):
    sock, _ = live_relay
    st, _ = _post(sock, {"prompt": "x"}, bearer="nope")
    assert st == 401


def test_happy_path_200(live_relay):
    sock, _ = live_relay
    st, body = _post(sock, {"prompt": "review this"})
    assert st == 200
    assert body["advisory_text"] == "advisory body"


def test_unknown_post_path_404(live_relay):
    sock, _ = live_relay
    st, _ = _post(sock, {"prompt": "x"}, path="/nope")
    assert st == 404


def test_invalid_json_body_400(live_relay):
    sock, _ = live_relay
    c = _UnixConn(sock)
    try:
        raw = b"not json{{"
        c.request("POST", "/consult", body=raw,
                  headers={"Authorization": "Bearer live-bearer",
                           "Content-Length": str(len(raw))})
        r = c.getresponse()
        assert r.status == 400
    finally:
        c.close()


def test_oversize_body_413(live_relay):
    sock, _ = live_relay
    st, _ = _post(sock, {"prompt": "a" * (48_000 + 5000)})
    assert st == 413


def test_quiesced_503(live_relay):
    sock, state = live_relay
    state["quiesced"] = True
    st, _ = _post(sock, {"prompt": "x"})
    assert st == 503
