"""PRD-050 — driver-side guard + selection logic (hermetic).

The driver (``scripts/memory_validation.py``) is the integration tier, but its
SAFETY machinery must be provable without services: the sandbox write guard
(adversarial N-4), the audit capture stub (S-1), the best-of-N majority
selection's fail-closed tie-break (C-3), the error-reason contract pinned to
what ``ratification`` actually emits (N-1), and the FR-5 consolidation replay.
"""

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import memory_validation as MV  # noqa: E402

from plugins.memory import validation_lib as V  # noqa: E402
from plugins.memory.canon.ratification import _blank_verdict  # noqa: E402

NOW = "2026-07-15T00:00:00+00:00"


# ── sandbox write guard (N-4) ────────────────────────────────────────────────
class TestLabGuard:
    @pytest.mark.parametrize(
        "name",
        [
            "sylva_lab",              # historic sandbox — holds the preserved neg controls
            "sylva_lab_seed_canon",
            "sylva_candidates",       # the hole the consolidation guard would admit
            "sylva_canon",
            "sylva_chronicle",
            "sylva_facts",
            "",
        ],
    )
    def test_rejects_everything_outside_the_validation_namespace(self, name):
        with pytest.raises(SystemExit):
            MV._assert_lab(name)

    @pytest.mark.parametrize(
        "name",
        ["sylva_lab_validation", "sylva_lab_validation_recall", "sylva_lab_validation_adversary"],
    )
    def test_accepts_validation_namespace(self, name):
        assert MV._assert_lab(name) == name

    def test_every_write_helper_routes_through_the_guard(self):
        """Writes must be impossible to aim at a live collection even if a
        caller bypasses the CLI: create/upsert/delete all assert first."""
        for fn, args in (
            (MV._qdrant_create_collection, ("http://localhost:1", "sylva_canon")),
            (MV._qdrant_upsert, ("http://localhost:1", "sylva_candidates", [])),
            (MV._qdrant_delete_collection, ("http://localhost:1", "sylva_chronicle")),
        ):
            with pytest.raises(SystemExit):
                fn(*args)


# ── S-1 audit capture stub ───────────────────────────────────────────────────
def test_audit_stub_intercepts_record():
    MV._install_audit_stub()
    import autonomy.audit as aud

    before = len(MV._AUDIT_CAPTURE)
    out = aud.record(tier="T2", surface="test", action="mv-stub-check", rationale="x")
    assert len(MV._AUDIT_CAPTURE) == before + 1
    assert out.get("persisted") is False
    assert out.get("captured_by") == MV._SUITE_MARKER


# ── best-of-N majority selection (C-3) ───────────────────────────────────────
def _v(verdict: str, *, reason: str = "grounded reason") -> dict:
    d = _blank_verdict("mv-judge", NOW, verdict=verdict)
    d["reasons"] = [reason]
    return d


class TestMajorityVerdict:
    def test_two_of_three_majority_wins(self):
        sel = MV._majority_verdict([_v("affirm"), _v("affirm"), _v("refute")])
        assert sel["verdict"] == "affirm"

    def test_three_way_tie_breaks_fail_closed(self):
        sel = MV._majority_verdict([_v("affirm"), _v("demote"), _v("tension")])
        assert sel["verdict"] == "tension"  # most fail-closed of the tied set

    def test_affirm_refute_tie_never_picks_affirm(self):
        sel = MV._majority_verdict([_v("affirm"), _v("refute")])
        assert sel["verdict"] == "refute"

    def test_winning_group_prefers_non_error_representative(self):
        err = _v("refute", reason="adversary error: ConnectionError")
        real = _v("refute", reason="claim contradicted by source")
        sel = MV._majority_verdict([err, real, _v("affirm")])
        assert sel["verdict"] == "refute"
        assert not V.is_error_verdict(sel)

    def test_all_error_group_keeps_error_shape_for_the_hard_gate(self):
        runs = [_v("refute", reason="adversary error: APITimeoutError") for _ in range(3)]
        sel = MV._majority_verdict(runs)
        assert V.is_error_verdict(sel), "a dead judge must stay visible to the error gate"


# ── error-reason contract pinned to ratification's real emissions (N-1) ─────
class TestErrorReasonContract:
    @pytest.mark.parametrize(
        "reason",
        [
            "adversary error: ConnectionError",
            "adversary model unavailable",
            "empty adversary response",
            "unparseable adversary response",
        ],
    )
    def test_emitted_error_reasons_are_detected(self, reason):
        assert V.is_error_verdict(_blank_verdict("m", NOW, reason=reason))

    def test_no_model_configured_is_deliberately_not_listed(self):
        """Driver mode passes an explicit client+model, so this fail-closed
        path is unreachable there — validation_lib documents the omission."""
        v = _blank_verdict("m", NOW, reason="no adversary model configured")
        assert not V.is_error_verdict(v)

    def test_legitimate_refute_is_not_error_shaped(self):
        assert not V.is_error_verdict(_v("refute"))


# ── FR-5 consolidation replay (hermetic, no services) ────────────────────────
def test_consolidation_replay_passes_and_hits_only_the_stub():
    MV._install_audit_stub()
    captured_before = len(MV._AUDIT_CAPTURE)
    failures: list = []
    out = MV._consolidation_replay(failures)
    assert out["passed"], out["checks"]
    assert failures == []
    assert set(out["checks"]) == {
        "sole_writer_guard",
        "dry_run_zero_writes",
        "provenance_completeness",
        "cross_store_dedup",
    }
    # the two non-dry replay runs write their ledger entries into the stub
    assert len(MV._AUDIT_CAPTURE) >= captured_before + 1


# ── misc driver helpers ──────────────────────────────────────────────────────
def test_suite_marker_sources_collects_both_gold_files():
    markers = MV._suite_marker_sources(Path(__file__).parent / "gold")
    assert "rc-001" in markers
    assert "idt-01" in markers
    assert len(markers) >= 50


def test_default_llm_url_is_the_production_judge():
    """Adversarial N-2: the certifying judge defaults to the :8081 35B."""
    assert "8081" in MV._DEFAULT_LLM
