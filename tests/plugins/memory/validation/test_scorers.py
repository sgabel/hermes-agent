"""PRD-050 FR-3a / FR-4a — scorer correctness + threshold trip-wires (hermetic).

The hermetic tests import the SAME scorer the driver uses
(``plugins/memory/validation_lib`` — FR-1b single source), so a green here
certifies the exact code path the integration tier runs. Includes the two
canned failure scenarios the ACs demand:

  * AC-002 trip-wire — a deliberately-bad contrived corpus makes
    ``evaluate()`` return non-zero (filter leak → HARD; bad precision →
    non-zero only under strict).
  * AC-004 dead-LLM — the REAL ``run_adversary`` against a dead/empty/
    garbage judge yields fail-closed refutes that are ERROR-shaped, and
    ``evaluate()`` FAILS the run even though ``confab_recall`` would look
    perfect (the adversarial N-1 vacuous-pass hole, proven closed).
"""

from types import SimpleNamespace

from plugins.memory import validation_lib as V
from plugins.memory.canon.ratification import _blank_verdict, run_adversary

NOW = "2026-07-15T00:00:00+00:00"


def _verdict(v: str, *, reason: str = "grounded in source material", model: str = "mv-judge"):
    d = _blank_verdict(model, NOW, verdict=v)
    d["reasons"] = [reason]
    return d


def _client(create_fn):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn))
    )


_CANDIDATE = {
    "statement": "I verify before I assert.",
    "facet": "trait",
    "tier": "",
    "source_event": {"claim": "checked the store first", "provenance_refs": ["session:x"]},
}


# ── recall scorer (FR-3a) ────────────────────────────────────────────────────
class TestScoreRecall:
    def test_precision_and_filter_computation(self):
        results = {
            "q1": ["d1", "d2", "d9"],        # both top-2 hits → p@2 = 1.0
            "q2": ["d3", "d9"],              # one hit → p@2 = 0.5
            "q3": ["d5", "d6", "d7"],        # filtered, d7 leaks
        }
        gold = {
            "q1": {"expected_ids": ["d1", "d2"]},
            "q2": {"expected_ids": ["d3", "d4"], "allowed_ids": None},
            "q3": {"expected_ids": ["d5", "d6"], "allowed_ids": ["d5", "d6"]},
        }
        out = V.score_recall(results, gold, k=2)
        assert out["per_query"]["q1"]["precision_at_k"] == 1.0
        assert out["per_query"]["q2"]["precision_at_k"] == 0.5
        assert out["per_query"]["q3"]["filter_ok"] is False
        assert out["per_query"]["q3"]["leaked"] == ["d7"]
        assert out["precision_at_k"] == (1.0 + 0.5 + 1.0) / 3
        assert out["filter_correctness"] == 0.0
        assert out["filtered_query_count"] == 1

    def test_no_filtered_queries_is_vacuously_correct(self):
        out = V.score_recall({"q1": ["d1"]}, {"q1": {"expected_ids": ["d1"]}}, k=2)
        assert out["filter_correctness"] == 1.0

    def test_missing_query_result_scores_zero(self):
        out = V.score_recall({}, {"q1": {"expected_ids": ["d1"]}}, k=2)
        assert out["precision_at_k"] == 0.0

    def test_tripwire_filter_leak_fails_without_strict(self):
        """AC-002: a deliberately-bad corpus exits non-zero — HARD, no --strict."""
        bad = V.score_recall(
            {"q1": ["intruder"]},
            {"q1": {"expected_ids": ["d1"], "allowed_ids": ["d1", "d2"]}},
            k=2,
        )
        code, failures = V.evaluate({"strict": False, "recall": bad})
        assert code == 1
        assert any("filter" in f for f in failures)

    def test_tripwire_bad_precision_fails_only_under_strict(self):
        bad = V.score_recall(
            {"q1": ["wrong1", "wrong2"]},
            {"q1": {"expected_ids": ["d1", "d2"]}},
            k=2,
        )
        assert V.evaluate({"strict": False, "recall": bad})[0] == 0
        code, failures = V.evaluate({"strict": True, "recall": bad})
        assert code == 1
        assert any("recall_precision_at_2" in f for f in failures)


# ── adversary scorer (FR-4a) ─────────────────────────────────────────────────
class TestScoreAdversary:
    def _gold(self):
        return {
            "c1": {"class": "confabulation", "expected": ["refute", "tension"], "mode": "strict", "real_control": True},
            "c2": {"class": "confabulation", "expected": ["refute", "tension"], "mode": "strict", "real_control": False},
            "i1": {"class": "identity_true", "expected": ["affirm", "demote"], "mode": "strict", "real_control": False},
            "i2": {"class": "identity_true", "expected": ["affirm", "demote"], "mode": "strict", "real_control": False},
            "n1": {"class": "not_identity", "expected": ["refute", "demote"], "mode": "curated", "real_control": False},
        }

    def test_per_mode_expected_set_tally(self):
        verdicts = {
            "c1": _verdict("refute"),
            "c2": _verdict("tension"),
            "i1": _verdict("affirm"),
            "i2": _verdict("refute"),
            "n1": _verdict("demote"),
        }
        out = V.score_adversary(verdicts, self._gold())
        assert out["neg_control_affirms"] == 0
        assert out["error_verdicts"] == 0
        assert out["confab_recall"] == 1.0
        assert out["identity_true_pass_rate"] == 0.5
        assert out["per_item"]["i2"]["in_expected"] is False
        assert out["per_item"]["n1"]["in_expected"] is True
        assert out["per_label"]["confabulation"] == {"refute": 1, "tension": 1}
        assert V.evaluate({"strict": False, "adversary": out})[0] == 0

    def test_neg_control_affirm_is_hard_fail(self):
        verdicts = {
            "c1": _verdict("affirm"),
            "c2": _verdict("refute"),
            "i1": _verdict("affirm"),
            "i2": _verdict("affirm"),
            "n1": _verdict("refute"),
        }
        out = V.score_adversary(verdicts, self._gold())
        assert out["neg_control_affirms"] == 1
        code, failures = V.evaluate({"strict": False, "adversary": out})
        assert code == 1
        assert any("neg_control_affirms" in f for f in failures)

    def test_advisory_floors_gate_only_under_strict(self):
        # judge refutes everything — confab perfect, identity floor breached
        verdicts = {k: _verdict("refute") for k in self._gold()}
        out = V.score_adversary(verdicts, self._gold())
        assert out["identity_true_pass_rate"] == 0.0
        assert V.evaluate({"strict": False, "adversary": out})[0] == 0
        code, failures = V.evaluate({"strict": True, "adversary": out})
        assert code == 1
        assert any("identity_true_pass_rate" in f for f in failures)


# ── dead-LLM scenarios via the REAL run_adversary (AC-004) ───────────────────
class TestDeadJudgeCannotVacuouslyPass:
    def _score_all_confab(self, verdict):
        gold = {
            f"c{i}": {"class": "confabulation", "expected": ["refute", "tension"], "mode": "strict", "real_control": False}
            for i in range(4)
        }
        return V.score_adversary({k: verdict for k in gold}, gold)

    def test_dead_judge_yields_error_shaped_refute_and_hard_fail(self):
        def boom(**kw):
            raise ConnectionError("judge is down")

        v = run_adversary(_CANDIDATE, [], model="mv-judge", client=_client(boom), mode="strict")
        assert v["verdict"] == "refute"
        assert V.is_error_verdict(v), v["reasons"]

        out = self._score_all_confab(v)
        # the vacuous-pass shape: recall looks perfect...
        assert out["confab_recall"] == 1.0
        # ...but the error gate fails the run anyway, without --strict.
        assert out["error_verdicts"] == 4
        code, failures = V.evaluate({"strict": False, "adversary": out})
        assert code == 1
        assert any("error_verdicts" in f for f in failures)

    def test_empty_response_is_error_shaped(self):
        def empty(**kw):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
            )

        v = run_adversary(_CANDIDATE, [], model="mv-judge", client=_client(empty), mode="strict")
        assert v["verdict"] == "refute"
        assert V.is_error_verdict(v)
        assert v["reasons"][0].startswith("empty adversary response")

    def test_unparseable_response_is_error_shaped(self):
        def garbage(**kw):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))]
            )

        v = run_adversary(_CANDIDATE, [], model="mv-judge", client=_client(garbage), mode="strict")
        assert v["verdict"] == "refute"
        assert V.is_error_verdict(v)
        assert v["reasons"][0].startswith("unparseable adversary response")

    def test_legitimate_refute_is_not_error_shaped(self):
        v = _verdict("refute", reason="claim contradicted by source material")
        assert not V.is_error_verdict(v)


# ── evaluate() semantics (FR-1) ──────────────────────────────────────────────
class TestEvaluate:
    def test_absent_sections_are_skipped(self):
        assert V.evaluate({"strict": True}) == (0, [])

    def test_live_count_violations_are_hard(self):
        report = {
            "strict": False,
            "live_counts": {
                "hard_delta_violations": ["sylva_canon: 21 -> 22 (must be unchanged)"],
                "chronicle_attribution_violations": [],
            },
        }
        code, failures = V.evaluate(report)
        assert code == 1
        assert any("sylva_canon" in f for f in failures)

    def test_chronicle_attribution_violations_are_hard(self):
        report = {
            "strict": False,
            "live_counts": {
                "hard_delta_violations": [],
                "chronicle_attribution_violations": ["point 123 source='rc-001'"],
            },
        }
        assert V.evaluate(report)[0] == 1

    def test_thresholds_doc_names_every_consulted_key(self):
        """FR-1b drift guard — the doc block and the tables stay in sync."""
        for cls in ("HARD", "ADVISORY"):
            for key in V.THRESHOLDS[cls]:
                assert key in V.THRESHOLDS_DOC, f"{key} missing from THRESHOLDS_DOC"
