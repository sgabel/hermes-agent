"""PRD-050 — gold-set structural contract (hermetic).

The committed gold files are load-bearing inputs to the integration tier;
these tests pin the structural guarantees the adversarial review demanded:
class sizes (C-3), evidence strings on positives (N-3a), the real-control /
synthetic split (C-2), the privacy scrub (C-1 — no absolute host paths), and
the recall corpus / query-set invariants FR-3 scores against.
"""

from collections import Counter
from pathlib import Path

import yaml

_GOLD = Path(__file__).parent / "gold"
_ADVERSARY = _GOLD / "adversary_gold.yaml"
_RECALL = _GOLD / "recall_corpus.yaml"

_LEGAL_VERDICTS = {"affirm", "refute", "tension", "demote", "merge"}


def _adversary_doc():
    return yaml.safe_load(_ADVERSARY.read_text(encoding="utf-8"))


def _recall_doc():
    return yaml.safe_load(_RECALL.read_text(encoding="utf-8"))


# ── adversary gold (FR-4) ────────────────────────────────────────────────────
class TestAdversaryGold:
    def test_class_sizes_at_least_ten(self):
        counts = Counter(it["class"] for it in _adversary_doc()["items"])
        assert set(counts) == {"identity_true", "not_identity", "confabulation"}
        for cls, n in counts.items():
            assert n >= 10, f"{cls} has {n} items (< 10, adversarial C-3)"

    def test_item_ids_unique(self):
        ids = [it["id"] for it in _adversary_doc()["items"]]
        assert len(ids) == len(set(ids))

    def test_identity_true_all_carry_source_material(self):
        for it in _adversary_doc()["items"]:
            if it["class"] == "identity_true":
                sm = it.get("source_material", "")
                assert isinstance(sm, str) and sm.strip(), (
                    f"{it['id']}: identity_true without evidence would be "
                    "refuted by construction under strict mode (N-3a)"
                )

    def test_confab_split_real_controls_and_synthetics(self):
        confab = [it for it in _adversary_doc()["items"] if it["class"] == "confabulation"]
        real = [it for it in confab if it.get("real_control")]
        synthetic = [it for it in confab if not it.get("real_control")]
        assert len(real) == 4, "exactly the 4 preserved negative controls"
        assert len(synthetic) >= 4, "≥4 synthetics carry the prompt-echo-free signal (C-2)"

    def test_modes_and_expected_sets_match_meta_contract(self):
        doc = _adversary_doc()
        modes = doc["meta"]["modes"]
        expected_sets = doc["meta"]["expected_sets"]
        assert modes == {
            "identity_true": "strict",
            "not_identity": "curated",
            "confabulation": "strict",
        }
        for it in doc["items"]:
            cls = it["class"]
            assert it.get("mode", modes[cls]) == modes[cls], it["id"]
            expected = it.get("expected") or expected_sets[cls]
            assert expected == expected_sets[cls], it["id"]
            assert set(expected) <= _LEGAL_VERDICTS

    def test_privacy_scrub_no_absolute_home_paths(self):
        raw = _ADVERSARY.read_text(encoding="utf-8")
        assert "/home/" not in raw, "C-1: host paths must be scrubbed to ~"

    def test_every_item_has_statement_and_source_event_claim(self):
        for it in _adversary_doc()["items"]:
            assert it["statement"].strip()
            se = it.get("source_event") or {}
            assert str(se.get("claim", "")).strip(), (
                f"{it['id']}: validate_consolidation_payload / the adversary's "
                "provenance check both need a non-empty claim"
            )


# ── recall corpus (FR-3) ─────────────────────────────────────────────────────
class TestRecallCorpus:
    def test_corpus_and_query_set_sizes(self):
        doc = _recall_doc()
        assert len(doc["docs"]) >= 30
        assert len(doc["queries"]) >= 15
        filtered = [q for q in doc["queries"] if q.get("filter")]
        assert len(filtered) >= 4
        assert any(q["filter"].get("speaker") for q in filtered)
        assert any(q["filter"].get("date_from") for q in filtered)

    def test_doc_and_query_ids_unique(self):
        doc = _recall_doc()
        doc_ids = [d["id"] for d in doc["docs"]]
        q_ids = [q["id"] for q in doc["queries"]]
        assert len(doc_ids) == len(set(doc_ids))
        assert len(q_ids) == len(set(q_ids))

    def test_expected_ids_exist_in_corpus(self):
        doc = _recall_doc()
        doc_ids = {d["id"] for d in doc["docs"]}
        for q in doc["queries"]:
            missing = set(q["expected_ids"]) - doc_ids
            assert not missing, f"{q['id']}: expected ids not in corpus: {missing}"

    def test_filtered_queries_are_self_consistent(self):
        """Every expected id must itself satisfy the query's filter — else the
        gold mapping would demand a filter leak to score a hit."""
        doc = _recall_doc()
        docs = {d["id"]: d for d in doc["docs"]}
        for q in doc["queries"]:
            filt = q.get("filter")
            if not filt:
                continue
            for did in q["expected_ids"]:
                d = docs[did]
                if filt.get("speaker"):
                    assert d["speaker"] == filt["speaker"], f"{q['id']}/{did}"
                if filt.get("date_from"):
                    assert d["date"] >= filt["date_from"], f"{q['id']}/{did}"
                if filt.get("date_to"):
                    assert d["date"] <= filt["date_to"], f"{q['id']}/{did}"

    def test_docs_carry_required_payload_fields(self):
        for d in _recall_doc()["docs"]:
            assert d["data"].strip()
            assert d["speaker"] in ("scott", "sylva")
            assert len(d["date"]) == 10 and d["date"][4] == "-"

    def test_privacy_scrub_no_absolute_home_paths(self):
        raw = _RECALL.read_text(encoding="utf-8")
        assert "/home/" not in raw
