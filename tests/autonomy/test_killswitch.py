"""PRD-028 R-6 / AC-008, AC-009 — flag-file kill switch."""

from autonomy import killswitch


def test_engage_and_status():
    assert killswitch.is_quiesced() is False
    killswitch.quiesce("test reason")
    assert killswitch.is_quiesced() is True
    st = killswitch.status()
    assert st["quiesced"] is True
    assert "test reason" in st.get("detail", "")


def test_rearm_is_explicit():
    killswitch.quiesce("x")
    assert killswitch.rearm() is True
    assert killswitch.is_quiesced() is False
    # re-arm when not engaged returns False
    assert killswitch.rearm() is False


def test_guard_skips_when_quiesced():
    assert killswitch.guard("cron") is False  # not engaged -> don't skip
    killswitch.quiesce("halt")
    assert killswitch.guard("cron") is True   # engaged -> skip the work


def test_flag_is_a_file_not_an_rpc(tmp_path, monkeypatch):
    # AC-009: a directly-invoked path (no gateway) still sees the flag because
    # it is a filesystem stat, independent of any running service.
    flag = tmp_path / "QUIESCE"
    monkeypatch.setenv("HERMES_AUTONOMY_QUIESCE_FLAG", str(flag))
    assert killswitch.is_quiesced() is False
    flag.write_text("manual\n")
    assert killswitch.is_quiesced() is True
