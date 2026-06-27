"""CanonStore — direct Qdrant + TEI for sylva_canon / sylva_candidates.

Bypasses the mem0 client/backend entirely: ``mem0_add`` binds one fixed
collection at construction (`_backend.py`) and cannot target the canon
collections (Pass-2 STOP-3), and mem0 OSS metadata filtering is thin. This is
the bespoke direct-Qdrant writer/reader the PRD specifies (AC-004) — the same
pattern :class:`~plugins.memory.mem0.chronicle.ChronicleSearcher` uses for reads,
extended with collection management + upserts.

Phase 2 uses this for: collection creation (AC-001), the render's filtered read
(AC-002/003), and test fixtures. Phases 3/5 are the production writers; this
module gives them the primitive but is collection-agnostic (callers pass the
target), so it never hardwires "who writes what".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from .schema import (
    CANDIDATES_COLLECTION,
    CANON_COLLECTION,
    LAYER_IDENTITY,
    VECTOR_DIM,
    VECTOR_DISTANCE,
)

logger = logging.getLogger(__name__)

# Host defaults (Qdrant/TEI published on localhost). In-container the real URLs
# come from mem0.json via from_config(); these are the fallback.
_DEFAULT_QDRANT_URL = "http://localhost:6333"
_DEFAULT_TEI_URL = "http://localhost:8085"

_TIMEOUT = 10
# Payload fields we filter on at read time — indexed defensively so a future
# strict-mode Qdrant doesn't reject the filtered scroll.
_INDEXED_FIELDS = ("layer", "status", "tier", "facet")


class CanonStore:
    """Typed identity store over Qdrant (sylva_canon + sylva_candidates)."""

    def __init__(
        self,
        qdrant_url: str = _DEFAULT_QDRANT_URL,
        tei_url: str = _DEFAULT_TEI_URL,
    ) -> None:
        self._qdrant_url = qdrant_url.rstrip("/")
        self._tei_url = tei_url.rstrip("/")

    # ── construction from live config ───────────────────────────────────────
    @classmethod
    def from_config(cls) -> "CanonStore":
        """Build a store with URLs resolved (in precedence order) from explicit
        env overrides, the live mem0.json OSS config (container DNS in prod), or
        localhost. Mirrors the mem0 plugin's resolution so canon and chronicle
        agree on endpoints.

        mem0.json carries *container* DNS (``http://qdrant:6333``) which is
        unreachable from the host (where tooling/tests run), so a configured
        Qdrant URL that does not answer is transparently swapped for the
        localhost default — making the one code path work both in-container and
        on the host without env juggling.
        """
        import os

        qdrant_url = _DEFAULT_QDRANT_URL
        tei_url = _DEFAULT_TEI_URL
        try:
            import json

            from hermes_constants import get_hermes_home

            mem0_path = get_hermes_home() / "mem0.json"
            cfg = json.loads(mem0_path.read_text(encoding="utf-8"))
            oss = cfg.get("oss") if isinstance(cfg.get("oss"), dict) else {}
            vs_cfg = (oss.get("vector_store") or cfg.get("vector_store") or {}).get("config", {})
            emb_cfg = (oss.get("embedder") or cfg.get("embedder") or {}).get("config", {})
            qdrant_url = vs_cfg.get("url") or qdrant_url
            tei_url = emb_cfg.get("openai_base_url") or tei_url
        except Exception as e:  # pragma: no cover - config best-effort
            logger.debug("CanonStore.from_config falling back to localhost: %s", e)

        # Explicit env overrides win outright.
        qdrant_url = os.environ.get("HERMES_CANON_QDRANT_URL", qdrant_url)
        tei_url = os.environ.get("HERMES_CANON_TEI_URL", tei_url)

        # Host fallback: a configured (container) URL that doesn't answer →
        # localhost. Skip the probe if it's already the localhost default.
        if qdrant_url != _DEFAULT_QDRANT_URL and not cls._reachable(qdrant_url):
            logger.debug("Qdrant %s unreachable; using %s", qdrant_url, _DEFAULT_QDRANT_URL)
            qdrant_url = _DEFAULT_QDRANT_URL
            if tei_url != _DEFAULT_TEI_URL:
                tei_url = _DEFAULT_TEI_URL
        return cls(qdrant_url=qdrant_url, tei_url=tei_url)

    @staticmethod
    def _reachable(qdrant_url: str) -> bool:
        try:
            return requests.get(
                f"{qdrant_url.rstrip('/')}/collections", timeout=2
            ).status_code == 200
        except Exception:
            return False

    # ── collection management ───────────────────────────────────────────────
    def collection_exists(self, collection: str) -> bool:
        try:
            r = requests.get(
                f"{self._qdrant_url}/collections/{collection}", timeout=3
            )
            return r.status_code == 200
        except Exception:
            return False

    def ensure_collections(
        self, collections: Tuple[str, ...] = (CANON_COLLECTION, CANDIDATES_COLLECTION)
    ) -> List[str]:
        """Idempotently create the canon collections (1024-dim Cosine, on-disk)
        with keyword indexes on the filtered fields. Returns names created."""
        created: List[str] = []
        for name in collections:
            if self.collection_exists(name):
                continue
            resp = requests.put(
                f"{self._qdrant_url}/collections/{name}",
                json={
                    "vectors": {
                        "size": VECTOR_DIM,
                        "distance": VECTOR_DISTANCE,
                        "on_disk": True,
                    }
                },
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            for field in _INDEXED_FIELDS:
                try:
                    requests.put(
                        f"{self._qdrant_url}/collections/{name}/index",
                        json={"field_name": field, "field_schema": "keyword"},
                        timeout=_TIMEOUT,
                    ).raise_for_status()
                except Exception as e:  # pragma: no cover - index best-effort
                    logger.debug("index %s on %s failed: %s", field, name, e)
            created.append(name)
        return created

    # ── reads ───────────────────────────────────────────────────────────────
    def get_canon(
        self,
        *,
        layer: str = LAYER_IDENTITY,
        status: str = "canon",
        collection: str = CANON_COLLECTION,
        limit: int = 1000,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Unordered filtered scroll → ``[(point_id, payload), …]``.

        Deliberately unordered: Qdrant ``order_by`` 400s on the
        created_at/render_order payload fields (not range-indexed), so ordering
        is the caller's job in Python (see :func:`schema.sort_key`). Returns []
        if the collection does not exist yet (render degrades to SOUL.md-only).
        """
        must: List[Dict[str, Any]] = []
        if layer:
            must.append({"key": "layer", "match": {"value": layer}})
        if status:
            must.append({"key": "status", "match": {"value": status}})

        out: List[Tuple[str, Dict[str, Any]]] = []
        next_offset: Any = None
        while True:
            body: Dict[str, Any] = {
                "limit": min(limit, 256),
                "with_payload": True,
                "with_vector": False,
            }
            if must:
                body["filter"] = {"must": must}
            if next_offset is not None:
                body["offset"] = next_offset
            resp = requests.post(
                f"{self._qdrant_url}/collections/{collection}/points/scroll",
                json=body,
                timeout=_TIMEOUT,
            )
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            result = resp.json().get("result", {})
            for pt in result.get("points", []):
                out.append((str(pt.get("id")), pt.get("payload") or {}))
            next_offset = result.get("next_page_offset")
            if next_offset is None or len(out) >= limit:
                break
        return out

    def count(self, collection: str) -> int:
        try:
            r = requests.post(
                f"{self._qdrant_url}/collections/{collection}/points/count",
                json={"exact": True},
                timeout=_TIMEOUT,
            )
            if r.status_code != 200:
                return 0
            return int(r.json().get("result", {}).get("count", 0))
        except Exception:
            return 0

    # ── writes (used by Phase 3/5 + tests; collection-agnostic) ─────────────
    def embed(self, text: str) -> List[float]:
        """TEI bge-m3 1024-dim embedding (for upserts/dedup; render never embeds)."""
        resp = requests.post(
            f"{self._tei_url}/embed",
            json={"inputs": text, "truncate": True},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0] if isinstance(data[0], list) else data
        return data

    def upsert(
        self,
        collection: str,
        points: List[Dict[str, Any]],
    ) -> None:
        """Upsert points ``[{id, payload, vector?}, …]`` via direct Qdrant PUT.

        If a point omits ``vector``, the ``statement`` is embedded via TEI so
        later phases (dedup/contradiction) have real vectors. Tests pass an
        explicit vector to stay hermetic (render ignores vectors entirely).
        """
        wire: List[Dict[str, Any]] = []
        for p in points:
            vec = p.get("vector")
            if vec is None:
                vec = self.embed(p["payload"]["statement"])
            wire.append({"id": p["id"], "payload": p["payload"], "vector": vec})
        resp = requests.put(
            f"{self._qdrant_url}/collections/{collection}/points",
            params={"wait": "true"},
            json={"points": wire},
            timeout=_TIMEOUT * 3,
        )
        resp.raise_for_status()
