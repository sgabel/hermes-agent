"""PRD-023 FR-1/FR-2: session_search result content bounding.

Covers:
  - AC-001: discover path caps per-message `content` AND serialized `tool_calls`
            with an explicit marker; fields at/under the cap are unchanged.
  - AC-002: read/scroll use the generous cap (~4,000); discover uses ~1,200 — the
            paths differ and scroll matches read.
  - AC-003: a session_search(limit=3) over long-content + large-tool_calls rows
            returns a bounded payload (< 20 KB), not 95 KB+, and is valid JSON.
  - AC-004: session_search is pinned in the budget system at 30,000.
"""

import json

import pytest

from tools.session_search_tool import (
    _DISCOVER_CONTENT_CAP,
    _READ_CONTENT_CAP,
    _shape_message,
    _truncate_text,
)


# --------------------------------------------------------------------------
# AC-001 — discover caps content + tool_calls; small fields untouched
# --------------------------------------------------------------------------
class TestShapeMessageContentCap:
    def test_long_content_truncated_with_marker(self):
        msg = {"id": 1, "role": "user", "content": "x" * 5000, "timestamp": 0}
        out = _shape_message(msg, content_cap=_DISCOVER_CONTENT_CAP)
        assert out["content"].startswith("x" * _DISCOVER_CONTENT_CAP)
        assert "… [truncated 3800 chars]" in out["content"]
        # cap + marker only — not the original 5000.
        assert len(out["content"]) < 1300

    def test_content_at_or_under_cap_unchanged(self):
        msg = {"id": 1, "role": "user", "content": "hello world", "timestamp": 0}
        out = _shape_message(msg, content_cap=_DISCOVER_CONTENT_CAP)
        assert out["content"] == "hello world"

    def test_empty_content_preserved(self):
        # tool-call-only assistant turns carry empty/None content — must survive.
        msg = {"id": 1, "role": "assistant", "content": None, "timestamp": 0}
        out = _shape_message(msg, content_cap=_DISCOVER_CONTENT_CAP)
        assert "content" in out
        assert out["content"] is None

    def test_large_tool_calls_serialized_then_truncated_to_str(self):
        # tool_calls arrives as a list-of-dicts; must become a truncated STRING,
        # never a sliced structure (the re-review's correctness requirement).
        tool_calls = [{"id": "a", "function": {"name": "f", "arguments": "y" * 40000}}]
        msg = {
            "id": 1,
            "role": "assistant",
            "content": "ok",
            "tool_calls": tool_calls,
            "timestamp": 0,
        }
        out = _shape_message(msg, content_cap=_DISCOVER_CONTENT_CAP)
        assert isinstance(out["tool_calls"], str)
        assert "… [truncated" in out["tool_calls"]
        assert len(out["tool_calls"]) < 1300

    def test_small_tool_calls_kept_structured(self):
        # Under the cap, tool_calls stays a structured list (no needless stringify).
        tool_calls = [{"id": "a", "function": {"name": "f", "arguments": "{}"}}]
        msg = {
            "id": 1,
            "role": "assistant",
            "content": "ok",
            "tool_calls": tool_calls,
            "timestamp": 0,
        }
        out = _shape_message(msg, content_cap=_DISCOVER_CONTENT_CAP)
        assert out["tool_calls"] == tool_calls

    def test_capped_entry_is_valid_json(self):
        # The whole response is json.dumps'd downstream — a truncated entry must
        # re-serialize cleanly (the structural-corruption guard).
        tool_calls = [{"id": "a", "function": {"name": "f", "arguments": "y" * 40000}}]
        msg = {
            "id": 1,
            "role": "assistant",
            "content": "z" * 9000,
            "tool_calls": tool_calls,
            "timestamp": 0,
        }
        out = _shape_message(msg, content_cap=_DISCOVER_CONTENT_CAP)
        round_trip = json.loads(json.dumps(out, ensure_ascii=False))
        assert round_trip["id"] == 1

    def test_zero_cap_is_unbounded_legacy(self):
        # Default cap=0 keeps the old byte-identical behavior for un-migrated callers.
        msg = {"id": 1, "role": "user", "content": "x" * 5000, "timestamp": 0}
        out = _shape_message(msg)
        assert len(out["content"]) == 5000


# --------------------------------------------------------------------------
# AC-002 — per-path caps differ; scroll == read (generous), discover tighter
# --------------------------------------------------------------------------
class TestPerPathCapsDiffer:
    def test_discover_tighter_than_read(self):
        assert _DISCOVER_CONTENT_CAP < _READ_CONTENT_CAP

    def test_read_cap_value(self):
        msg = {"id": 1, "role": "user", "content": "x" * 9000, "timestamp": 0}
        out = _shape_message(msg, content_cap=_READ_CONTENT_CAP)
        assert out["content"].startswith("x" * _READ_CONTENT_CAP)
        assert len(out["content"]) < _READ_CONTENT_CAP + 64

    def test_discover_cap_value(self):
        msg = {"id": 1, "role": "user", "content": "x" * 9000, "timestamp": 0}
        out = _shape_message(msg, content_cap=_DISCOVER_CONTENT_CAP)
        assert len(out["content"]) < _DISCOVER_CONTENT_CAP + 64


class TestTruncateHelper:
    def test_under_cap_noop(self):
        assert _truncate_text("short", 100) == "short"

    def test_over_cap_marked(self):
        out = _truncate_text("a" * 50, 10)
        assert out == "a" * 10 + "… [truncated 40 chars]"

    def test_zero_cap_noop(self):
        assert _truncate_text("a" * 50, 0) == "a" * 50

    def test_non_string_passthrough(self):
        assert _truncate_text(None, 10) is None


# --------------------------------------------------------------------------
# AC-003 — discover payload is bounded end-to-end (< 20 KB), not 95 KB+
# --------------------------------------------------------------------------
def _iter_payload_messages(payload):
    for r in payload["results"]:
        for key in ("bookend_start", "messages", "bookend_end"):
            for m in r.get(key) or []:
                yield m


def test_discover_payload_bounded(tmp_path):
    """A real session_search(limit=3) over a heavy day (mostly-normal messages with
    a few 159 KB content floods + 40 KB tool_calls blobs — the actual crash shape)
    returns a bounded payload, not the pre-fix 95 KB+, and every message is capped."""
    pytest.importorskip("hermes_state")
    from hermes_state import SessionDB
    from tools.session_search_tool import session_search

    db = SessionDB(db_path=tmp_path / "state.db")
    flood_content = "needle " + ("filler " * 22000)        # ~159 KB (matches live max)
    big_args = "z" * 40000                                  # 40 KB tool_calls blob
    for s in ("s1", "s2", "s3"):
        db.create_session(s, source="cli")
        # Mostly small messages around the match...
        for i in range(5):
            db.append_message(s, role="user", content=f"needle context line {i}")
            db.append_message(s, role="assistant", content=f"reply {i} about the work")
        # ...with the heavy flood messages a real heavy day produces.
        db.append_message(s, role="user", content=flood_content)
        db.append_message(
            s,
            role="assistant",
            content=flood_content,
            tool_calls=json.dumps(
                [{"id": "c", "type": "function",
                  "function": {"name": "f", "arguments": big_args}}]
            ),
        )

    raw = session_search(query="needle", limit=3, db=db)
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["mode"] == "discover"

    # FR-1 guarantee: no single message floods — content and serialized tool_calls
    # are each bounded to the discover cap (+ marker), killing the 159 KB vector.
    for m in _iter_payload_messages(payload):
        if isinstance(m.get("content"), str):
            assert len(m["content"]) <= _DISCOVER_CONTENT_CAP + 64
        if isinstance(m.get("tool_calls"), str):
            assert len(m["tool_calls"]) <= _DISCOVER_CONTENT_CAP + 64

    # Aggregate for a realistic heavy day stays well under the pre-fix 95 KB+
    # (a single old result could carry a 159 KB message). FR-2 then caps the
    # whole tool result at 30 KB downstream for the truly pathological case.
    assert len(raw) < 20_000, f"payload too large: {len(raw)} bytes"


# --------------------------------------------------------------------------
# AC-004 — budget backstop threshold pinned for session_search
# --------------------------------------------------------------------------
def test_session_search_pinned_threshold():
    from tools.budget_config import DEFAULT_BUDGET, PINNED_THRESHOLDS

    assert PINNED_THRESHOLDS["session_search"] == 30_000
    # Pinned wins immediately — not the 100K default, not a registry value.
    assert DEFAULT_BUDGET.resolve_threshold("session_search") == 30_000


# --------------------------------------------------------------------------
# AC-000 — FR-0 per-model context_length override is scoped, not global
# --------------------------------------------------------------------------
class TestPerModelContextOverride:
    """The local-Qwen pin must resolve 65,536 for qwen@:8081 and leave any other
    model / endpoint to self-resolve (so a cloud-driver swap needs no revisit)."""

    CP = [{
        "name": "local-qwen",
        "base_url": "http://localhost:8081/v1",
        "models": {"qwen3.6-35b-a3b": {"context_length": 65536}},
    }]

    def test_qwen_at_local_endpoint_resolves_65536(self):
        from hermes_cli.config import get_custom_provider_context_length
        got = get_custom_provider_context_length(
            model="qwen3.6-35b-a3b",
            base_url="http://localhost:8081/v1",
            custom_providers=self.CP,
        )
        assert got == 65536

    def test_cloud_model_not_clamped(self):
        from hermes_cli.config import get_custom_provider_context_length
        for model, url in (
            ("claude-opus-4-8", "https://api.anthropic.com"),
            ("gpt-5.5", "https://api.openai.com/v1"),
        ):
            assert get_custom_provider_context_length(
                model=model, base_url=url, custom_providers=self.CP
            ) is None

    def test_same_endpoint_other_model_not_clamped(self):
        from hermes_cli.config import get_custom_provider_context_length
        assert get_custom_provider_context_length(
            model="some-other-model",
            base_url="http://localhost:8081/v1",
            custom_providers=self.CP,
        ) is None

    def test_nameless_entry_is_dropped(self):
        # Guards the re-review NEEDS-FIX: a custom_providers entry without `name`
        # is silently dropped by normalization, so the override would never fire.
        from hermes_cli.config import get_compatible_custom_providers
        nameless = {"config": {
            "custom_providers": [{
                "base_url": "http://localhost:8081/v1",
                "models": {"qwen3.6-35b-a3b": {"context_length": 65536}},
            }],
        }}
        compatible = get_compatible_custom_providers(nameless["config"])
        assert all(e.get("name") or e.get("provider_key") for e in compatible)
