#!/usr/bin/env python3
"""Create the PRD-029 canon collections (sylva_canon, sylva_candidates).

Idempotent: skips collections that already exist. Run on the host (resolves
Qdrant from mem0.json, falling back to localhost:6333) before seeding (Phase 5):

    cd ~/hermes/hermes-agent && source venv/bin/activate
    python3 scripts/canon_init.py

This only provisions empty, typed collections — it writes no identity content.
With them empty, render_self_brief() returns the SOUL.md-only fallback, so
provisioning is a zero-runtime-change step.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.memory.canon import CANDIDATES_COLLECTION, CANON_COLLECTION, CanonStore


def main() -> int:
    store = CanonStore.from_config()
    print(f"Qdrant: {store._qdrant_url}")
    created = store.ensure_collections()
    for name in (CANON_COLLECTION, CANDIDATES_COLLECTION):
        exists = store.collection_exists(name)
        flag = "created" if name in created else "exists"
        print(f"  {name}: {flag} ({store.count(name)} points)" if exists else f"  {name}: MISSING")
    if not all(store.collection_exists(n) for n in (CANON_COLLECTION, CANDIDATES_COLLECTION)):
        print("FAIL: a canon collection is missing", file=sys.stderr)
        return 1
    print("canon collections ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
