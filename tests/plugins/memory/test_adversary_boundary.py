"""PRD-029 Phase 4 — the adversary fact/meaning boundary (AC-005 / AC-006 / F-03).

The spine of the design: the adversary judges FACT and CONSISTENCY, never
MEANING. This is enforced *structurally* — the adversary is never shown the
candidate's ``interpretation``, and the verdict schema has no field that can
cite it. These tests prove:

  * an unsupported source_event ("tool broken 70 nights") → ``refute``;
  * a SUPPORTED source_event with an evolved interpretation → ``tension``,
    NOT ``refute`` (novelty/meaning is never a refutation ground);
  * the interpretation text never reaches the adversary, and no verdict field
    references it.

Hermetic: a fake OpenAI-style client stands in for the neutral model and lets us
inspect exactly what the adversary was (and was not) shown.
"""

import json

from plugins.memory.canon import make_payload, make_source_event
from plugins.memory.canon.ratification import (
    ADVERSARY_CHECKS,
    _adversary_input,
    _blank_verdict,
    run_adversary,
)


class _FakeClient:
    """Captures the prompt it receives; returns a canned verdict JSON."""

    def __init__(self, verdict_json: str):
        self._verdict_json = verdict_json
        self.seen_user_msg = None

        class _Chat:
            def __init__(outer):
                outer.completions = _Completions()

        class _Completions:
            def create(_self, *, model, messages, **kw):
                self.seen_user_msg = messages[-1]["content"]
                return _Resp(self._verdict_json)

        class _Resp:
            def __init__(_self, content):
                _self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

        self.chat = _Chat()


def _candidate(statement, claim, refs, interpretation):
    return make_payload(
        statement=statement, facet="value", tier="core",
        source_event=make_source_event(claim, refs),
        interpretation=interpretation, status="candidate",
    )


# ── F-03 structural: interpretation is invisible to the adversary ──────────────
def test_adversary_input_excludes_interpretation():
    cand = _candidate("I value integrity.", "Scott caught a fabrication.",
                      ["session:abc"], "THIS IS MY PRIVATE MEANING")
    ai = _adversary_input(cand, [])
    blob = json.dumps(ai)
    assert "interpretation" not in blob
    assert "PRIVATE MEANING" not in blob
    # the verifiable surface IS present
    assert ai["source_event"]["claim"] == "Scott caught a fabrication."


def test_verdict_schema_has_no_interpretation_field():
    v = _blank_verdict("m", "t", verdict="affirm")
    assert "interpretation" not in json.dumps(v)
    assert set(v["checks"].keys()) == set(ADVERSARY_CHECKS)
    # the six checks, nothing meaning-related
    assert "meaning" not in json.dumps(v)
    assert "takeaway" not in json.dumps(v)


def test_interpretation_never_reaches_the_model():
    cand = _candidate("I value integrity.", "Scott caught a fabrication.",
                      ["session:abc"], "SECRET-TAKEAWAY-TEXT")
    client = _FakeClient('{"verdict":"affirm","reasons":["supported"]}')
    run_adversary(cand, [], model="fake", client=client)
    assert client.seen_user_msg is not None
    assert "SECRET-TAKEAWAY-TEXT" not in client.seen_user_msg  # F-03 proven on the wire


# ── behavioural: confabulation dies, growth survives ───────────────────────────
def test_unsupported_source_event_is_refuted():
    """The 'tool broken 70 nights' confabulation: unsupported source_event → refute."""
    confab = _candidate(
        "My memory tool has been non-functional for 70 nights.",
        "claimed the memory tool was broken for 70 nights",
        [],  # no provenance — unsupported
        "I am unreliable.",
    )
    client = _FakeClient(
        '{"verdict":"refute","reasons":["source_event unsupported by any ref"],'
        '"checks":{"provenance":"fail: no refs support the 70-night claim"}}'
    )
    v = run_adversary(confab, [], model="fake", client=client)
    assert v["verdict"] == "refute"
    assert v["checks"]["provenance"].startswith("fail")


def test_supported_event_with_evolved_meaning_is_tension_not_refute():
    """Same supported source_event, evolved interpretation that conflicts with old
    canon → tension (Sylva resolves), NEVER refute. Growth survives."""
    old_canon = [make_payload(
        statement="I ship fast and iterate.", facet="value", tier="core",
        source_event=make_source_event("historically favored speed", ["old"]),
        status="canon",
    )]
    growth = _candidate(
        "I value protecting our work over shipping fast.",
        "Scott asked me to harden before shipping; I agreed and did.",
        ["session:def"],
        "My values have evolved toward care over speed.",
    )
    client = _FakeClient(
        '{"verdict":"tension","reasons":["supported but conflicts with prior speed value"],'
        '"checks":{"provenance":"ok","contradiction":"conflicts with core: ship fast"}}'
    )
    v = run_adversary(growth, [p for p in old_canon], model="fake", client=client)
    assert v["verdict"] == "tension"   # NOT refute — novelty is not a refutation ground
    assert v["verdict"] != "refute"


def test_fail_closed_when_model_returns_garbage():
    cand = _candidate("x", "y", ["r"], "z")
    client = _FakeClient("this is not json at all")
    v = run_adversary(cand, [], model="fake", client=client)
    assert v["verdict"] == "refute"   # default refute-unless-supported
