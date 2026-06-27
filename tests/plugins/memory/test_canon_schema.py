"""PRD-029 Phase 2 — canon schema + total-order sort key.

Covers the load-bearing schema contract Phases 3/4 build on: the fact/meaning
split, enum membership, the bedrock-not-in-collection invariant (AC-001/009),
and the deterministic sort key (AC-003 / F-02 / NF-4).
"""

import pytest

from plugins.memory.canon import schema
from plugins.memory.canon.schema import (
    RENDER_ORDER_SENTINEL,
    CanonSchemaError,
    make_payload,
    make_source_event,
    sort_key,
    validate_payload,
)


def _valid(**over):
    base = dict(
        statement="I protect our work before shipping fast.",
        facet="value",
        tier="core",
        source_event=make_source_event("hardened security at 02:14", ["sess:1#5-9"]),
        interpretation="this reflects that I value protecting our work",
    )
    base.update(over)
    return make_payload(**base)


def test_source_event_shape():
    se = make_source_event("x happened", ["ref1"])
    assert se == {"claim": "x happened", "provenance_refs": ["ref1"]}
    # the adversary's only verifiable surface; interpretation is a sibling field
    p = _valid()
    assert "interpretation" in p and "source_event" in p
    assert "interpretation" not in p["source_event"]


def test_validate_accepts_valid():
    validate_payload(_valid())


@pytest.mark.parametrize("bad", [
    {"facet": "not-a-facet"},
    {"tier": "not-a-tier"},
    {"status": "not-a-status"},
    {"statement": "   "},
])
def test_validate_rejects_bad_enums(bad):
    with pytest.raises(CanonSchemaError):
        validate_payload(_valid(**bad))


def test_bedrock_rejected_from_collection():
    # AC-001/AC-009: bedrock lives in SOUL.md, never as a collection row.
    with pytest.raises(CanonSchemaError):
        validate_payload(_valid(tier="bedrock"))


def test_validate_rejects_malformed_source_event():
    p = _valid()
    p["source_event"] = {"claim": "x"}  # missing provenance_refs
    with pytest.raises(CanonSchemaError):
        validate_payload(p)


def test_sort_key_tier_then_order_then_id():
    # core (rank 1) before peripheral (rank 2); within tier by render_order; ties by id
    keys = [
        sort_key("zzz", {"tier": "peripheral", "render_order": 0}),
        sort_key("aaa", {"tier": "core", "render_order": 5}),
        sort_key("bbb", {"tier": "core", "render_order": 5}),
        sort_key("ccc", {"tier": "core", "render_order": 1}),
    ]
    order = [k for k in sorted(keys)]
    # core/order1, core/order5/aaa, core/order5/bbb, peripheral
    assert order[0] == (1, 1, "ccc")
    assert order[1] == (1, 5, "aaa")
    assert order[2] == (1, 5, "bbb")
    assert order[3] == (2, 0, "zzz")


def test_sort_key_missing_render_order_sorts_after():
    with_order = sort_key("a", {"tier": "core", "render_order": 10_000})
    without = sort_key("a", {"tier": "core"})
    assert without[1] == RENDER_ORDER_SENTINEL
    assert with_order < without  # explicit order wins over sentinel


def test_facet_and_tier_enums_stable():
    # Phase 4 depends on these literals — guard against silent drift.
    assert schema.FACETS == (
        "value", "trait", "relationship", "selffact", "commitment", "mode", "framing",
    )
    assert schema.TIER_RANK == {"bedrock": 0, "core": 1, "peripheral": 2}
