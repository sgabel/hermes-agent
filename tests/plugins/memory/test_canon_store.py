"""PRD-029 Phase 2 — CanonStore direct-Qdrant round-trip (integration).

Gated on a reachable Qdrant. Uses a throwaway collection so it never touches the
live sylva_canon/sylva_candidates. Proves the read path is a filtered *scroll*
(no embedding, no semantic search — AC-002) and that get_canon → assemble_brief
renders in the deterministic total order against real Qdrant data.
"""

import uuid

import pytest
import requests

from plugins.memory.canon import CanonStore, assemble_brief
from plugins.memory.canon.schema import VECTOR_DIM, make_payload, make_source_event

_QDRANT = "http://localhost:6333"


def _qdrant_up() -> bool:
    try:
        return requests.get(f"{_QDRANT}/collections", timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qdrant_up(), reason="Qdrant not reachable on localhost:6333")


@pytest.fixture
def temp_collection():
    name = f"test_canon_{uuid.uuid4().hex[:8]}"
    store = CanonStore(qdrant_url=_QDRANT)
    store.ensure_collections(collections=(name,))
    try:
        yield store, name
    finally:
        requests.delete(f"{_QDRANT}/collections/{name}", timeout=10)


def _point(pid, statement, tier="core", render_order=None, status="canon"):
    payload = make_payload(
        statement=statement,
        facet="value",
        tier=tier,
        status=status,
        render_order=render_order,
        source_event=make_source_event(statement, ["ref"]),
    )
    # explicit dummy vector — render ignores vectors, so no TEI dependency.
    return {"id": pid, "payload": payload, "vector": [0.01] * VECTOR_DIM}


def test_ensure_is_idempotent(temp_collection):
    store, name = temp_collection
    assert store.ensure_collections(collections=(name,)) == []  # already exists
    assert store.collection_exists(name)


def test_upsert_then_get_canon_filters_and_renders(temp_collection):
    store, name = temp_collection
    ids = [str(uuid.uuid4()) for _ in range(3)]
    store.upsert(name, [
        _point(ids[0], "third value", render_order=3),
        _point(ids[1], "first value", render_order=1),
        _point(ids[2], "second value", render_order=2),
        # a non-canon row that must be filtered OUT by status
        _point(str(uuid.uuid4()), "a mere candidate", status="candidate"),
    ])
    entries = store.get_canon(status="canon", collection=name)
    statements = {p["statement"] for _i, p in entries}
    assert statements == {"first value", "second value", "third value"}
    assert "a mere candidate" not in statements  # status filter works

    brief = assemble_brief("BEDROCK", entries, 4096)
    assert brief.index("first value") < brief.index("second value") < brief.index("third value")


def test_get_canon_missing_collection_returns_empty():
    store = CanonStore(qdrant_url=_QDRANT)
    assert store.get_canon(collection="does_not_exist_xyz") == []
