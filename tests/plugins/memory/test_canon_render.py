"""PRD-029 Phase 2 — deterministic self-brief render (AC-002/003).

The render is *assembled, never generated*: a pure function of (SOUL.md bedrock,
ratified canon). These tests pin determinism (byte-identical across calls and
input orderings), the total-order sort, budget truncation, and the SOUL.md-only
fallback when canon is empty.
"""

from plugins.memory.canon import assemble_brief
from plugins.memory.canon.render import _CHARS_PER_TOKEN
from plugins.memory.canon.schema import make_payload, make_source_event


def _entry(pid, statement, tier="core", render_order=None):
    return (
        pid,
        make_payload(
            statement=statement,
            facet="value",
            tier=tier,
            render_order=render_order,
            source_event=make_source_event(statement, ["ref"]),
        ),
    )


SOUL = "I am Sylva. Good lady of the wood."


def test_empty_canon_returns_soul_only():
    # AC-018 ordering / fallback: with no canon, brief == SOUL.md bedrock.
    assert assemble_brief(SOUL, [], canon_token_budget=4096) == SOUL


def test_no_soul_no_canon_returns_none():
    assert assemble_brief(None, [], canon_token_budget=4096) is None


def test_soul_prepended_ahead_of_canon():
    brief = assemble_brief(SOUL, [_entry("a", "I keep my word.")], 4096)
    assert brief.startswith(SOUL)
    assert "I keep my word." in brief
    assert brief.index(SOUL) < brief.index("I keep my word.")


def test_deterministic_across_input_order():
    # AC-003: byte-identical regardless of scroll order (Qdrant returns unordered).
    entries = [
        _entry("c3", "third", render_order=3),
        _entry("c1", "first", render_order=1),
        _entry("c2", "second", render_order=2),
    ]
    forward = assemble_brief(SOUL, entries, 4096)
    reverse = assemble_brief(SOUL, list(reversed(entries)), 4096)
    assert forward == reverse
    # rendered in render_order, not input order
    assert forward.index("first") < forward.index("second") < forward.index("third")


def test_core_before_peripheral():
    entries = [
        _entry("p", "peripheral fact", tier="peripheral", render_order=0),
        _entry("c", "core value", tier="core", render_order=9),
    ]
    brief = assemble_brief(SOUL, entries, 4096)
    assert brief.index("core value") < brief.index("peripheral fact")


def test_budget_truncates_in_sorted_order():
    # Budget small enough to admit only the first core statement.
    s1 = "x" * 20  # core, render_order 1
    s2 = "y" * 20  # core, render_order 2 (should be dropped)
    entries = [_entry("a", s1, render_order=1), _entry("b", s2, render_order=2)]
    budget_tokens = int((len(s1) + 5) / _CHARS_PER_TOKEN)  # room for s1, not both
    brief = assemble_brief(SOUL, entries, budget_tokens)
    assert s1 in brief
    assert s2 not in brief


def test_budget_zero_yields_soul_only():
    brief = assemble_brief(SOUL, [_entry("a", "dropped")], canon_token_budget=0)
    assert brief == SOUL


def test_render_self_brief_stable_in_process():
    # AC-003 (in-process half; the verify suite covers cross-restart). Seed a
    # known SOUL.md into the isolated test HERMES_HOME so the render reflects it
    # deterministically (the harness lazily seeds a default SOUL.md otherwise) —
    # then assert byte-identical across calls. sylva_canon is empty here, so the
    # brief is SOUL.md-only, exercising the pre-seeding fallback path.
    from hermes_constants import get_hermes_home

    from plugins.memory.canon import render_self_brief

    (get_hermes_home() / "SOUL.md").write_text(SOUL, encoding="utf-8")
    first = render_self_brief()
    assert first == render_self_brief()
    assert first == SOUL  # empty canon → SOUL.md-only
