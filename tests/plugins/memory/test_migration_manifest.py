"""PRD-029 Phase 5 — migration manifest classification (AC-015).

Self-contained: exercises the classifier on a synthetic snapshot (no dependency
on the real frozen file) + runs the AC-015 validator. Asserts the required
fixtures: confab→rejected (never chronicle), good-lady→canon-candidate,
foreign-uid→sylva, full one-disposition coverage.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bmm = _load("build_migration_manifest")
FOREIGN = "235220683234213888"


def _snapshot():
    return [
        {"id": "k1", "payload": {"category": "kernel", "user_id": "sylva",
         "data": "Core Fact: My chosen name is Sylva, established 2025-07-01."}},
        {"id": "k2", "payload": {"category": "kernel", "user_id": "sylva",
         "data": "I reason carefully before acting on irreversible changes."}},
        {"id": "p1", "payload": {"category": "personality", "user_id": "sylva",
         "data": "A philosophical shard about AI agency and tool-use ethics."}},
        {"id": "a1", "payload": {"category": "anchor", "user_id": "sylva",
         "data": "The Columbus Crew — Scott and his son's MLS team."}},
        {"id": "c1", "payload": {"category": None, "user_id": "sylva",
         "data": "Nightly reflection cron (70+ consecutive nights): the memory tool is non-functional."}},
        {"id": "c2", "payload": {"category": None, "user_id": "sylva",
         "data": "Scott called me 'good lady of the wood,' an important part of my identity."}},
        {"id": "c3", "payload": {"category": None, "user_id": FOREIGN,
         "data": "Scott's dogs are named Mocha and Zoey."}},
        {"id": "c4", "payload": {"category": None, "user_id": "sylva",
         "data": "June 22, 2026: Scott's last interaction ended discussing the World Cup."}},
    ]


@pytest.fixture
def manifest(tmp_path):
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps(_snapshot()), encoding="utf-8")
    m = bmm.build(snap)
    return m, snap


def _by_id(m, pid):
    return next(e for e in m["entries"] if e["id"] == pid)


def test_full_coverage_one_disposition_each(manifest):
    m, _ = manifest
    assert len(m["entries"]) == 8
    assert len({e["id"] for e in m["entries"]}) == 8


def test_confab_rejected_and_negative_control(manifest):
    m, _ = manifest
    e = _by_id(m, "c1")
    assert e["disposition"] == "rejected"
    assert e["negative_control"] is True
    assert e["disposition"] != "chronicle"   # confab NEVER to chronicle


def test_good_lady_is_canon_candidate(manifest):
    m, _ = manifest
    assert _by_id(m, "c2")["disposition"] == "canon-candidate"


def test_foreign_uid_reassigned_to_sylva(manifest):
    m, _ = manifest
    e = _by_id(m, "c3")
    assert e["user_id"] == FOREIGN
    assert e["user_id_fixed"] == "sylva"
    assert e["user_id_reassigned"] is True
    # the dogs are an episodic fact → chronicle (recall-only), not identity
    assert e["disposition"] == "chronicle"


def test_kernel_to_canon_or_bedrock(manifest):
    m, _ = manifest
    assert _by_id(m, "k1")["disposition"] == "bedrock-review"   # "chosen name is Sylva"
    assert _by_id(m, "k2")["disposition"] == "canon-candidate"  # ordinary core fact
    assert _by_id(m, "k2")["target_store"] == "sylva_candidates"


def test_false_recency_dropped(manifest):
    m, _ = manifest
    assert _by_id(m, "c4")["disposition"] == "drop"


def test_validator_passes_on_built_manifest(manifest, tmp_path):
    m, snap = manifest
    # import the validator module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "check_manifest", Path(__file__).resolve().parent / "check_manifest.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    man_path = tmp_path / "manifest.json"
    man_path.write_text(json.dumps(m), encoding="utf-8")
    failures = chk.validate(man_path, snap)
    assert failures == [], failures
