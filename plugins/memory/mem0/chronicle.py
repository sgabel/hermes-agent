"""Chronicle searcher — direct Qdrant + TEI for sylva_chronicle.

Bypasses the mem0 client (which is bound to sylva_memories).
Uses the same bge-m3 embedder as mem0 OSS for vector consistency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_QDRANT_URL = "http://localhost:6333"
_DEFAULT_TEI_URL = "http://localhost:8085"
_DEFAULT_COLLECTION = "sylva_chronicle"
_EMBED_TIMEOUT = 10
_SEARCH_TIMEOUT = 10


class ChronicleSearcher:
    """Semantic search over the sylva_chronicle Qdrant collection."""

    def __init__(
        self,
        qdrant_url: str = _DEFAULT_QDRANT_URL,
        tei_url: str = _DEFAULT_TEI_URL,
        collection: str = _DEFAULT_COLLECTION,
    ):
        # Host fallback (FR-2, mirrors CanonStore.from_config at
        # plugins/memory/canon/store.py:92): mem0.json carries *container* DNS
        # (``http://qdrant:6333`` / ``http://tei-bge-m3:80``) which is
        # unreachable from a host process (tests, host-side cockpit/voice). A
        # configured URL that does not answer is transparently swapped for the
        # localhost default so the one code path — search AND on_session_end
        # writes — works both in-container and on the host without env juggling.
        qdrant_url, tei_url = self._resolve_endpoints(qdrant_url, tei_url)
        self._qdrant_url = qdrant_url.rstrip("/")
        self._tei_url = tei_url.rstrip("/")
        self._collection = collection

    @staticmethod
    def _reachable(qdrant_url: str) -> bool:
        try:
            return requests.get(
                f"{qdrant_url.rstrip('/')}/collections", timeout=2
            ).status_code == 200
        except Exception:
            return False

    @staticmethod
    def _tei_reachable(tei_url: str) -> bool:
        try:
            return requests.get(
                f"{tei_url.rstrip('/')}/info", timeout=2
            ).status_code == 200
        except Exception:
            return False

    @classmethod
    def _resolve_endpoints(cls, qdrant_url: str, tei_url: str) -> tuple[str, str]:
        """Resolve each endpoint INDEPENDENTLY: a configured (container-DNS) URL
        that doesn't answer from this process falls back to its localhost
        default. Probing them separately (vs CanonStore's single-Qdrant-probe-
        swaps-both) covers the case where Qdrant is reachable but TEI is still a
        container URL — otherwise embeds would fail while search/scroll worked.
        The localhost default is left untouched (skip the probe)."""
        q = qdrant_url or _DEFAULT_QDRANT_URL
        t = tei_url or _DEFAULT_TEI_URL
        if q != _DEFAULT_QDRANT_URL and not cls._reachable(q):
            logger.debug("Chronicle Qdrant %s unreachable; falling back to %s", q, _DEFAULT_QDRANT_URL)
            q = _DEFAULT_QDRANT_URL
        if t != _DEFAULT_TEI_URL and not cls._tei_reachable(t):
            logger.debug("Chronicle TEI %s unreachable; falling back to %s", t, _DEFAULT_TEI_URL)
            t = _DEFAULT_TEI_URL
        return q, t

    def embed(self, text: str) -> List[float]:
        """Get embedding from TEI (bge-m3, 1024-dim)."""
        resp = requests.post(
            f"{self._tei_url}/embed",
            json={"inputs": text, "truncate": True},
            timeout=_EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        # TEI returns [[float, ...]] for single input
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0] if isinstance(data[0], list) else data
        return data

    def search(
        self,
        query: str,
        *,
        speaker: str = "any",
        date_from: str = "",
        date_to: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Vector search with optional payload filters.

        Args:
            query: Search text.
            speaker: "scott", "sylva", or "any" (no filter).
            date_from: Inclusive start date (YYYY-MM-DD).
            date_to: Inclusive end date (YYYY-MM-DD).
            top_k: Max results (capped at 15).

        Returns:
            List of dicts with keys: data, speaker, date, source, score.
        """
        top_k = min(top_k, 15)
        vector = self.embed(query)

        # Build Qdrant filter conditions
        must_conditions: List[Dict[str, Any]] = []

        if speaker and speaker != "any":
            must_conditions.append({
                "key": "speaker",
                "match": {"value": speaker},
            })

        if date_from:
            must_conditions.append({
                "key": "date",
                "range": {"gte": date_from},
            })

        if date_to:
            must_conditions.append({
                "key": "date",
                "range": {"lte": date_to},
            })

        body: Dict[str, Any] = {
            "vector": vector,
            "limit": top_k,
            "with_payload": True,
        }

        if must_conditions:
            body["filter"] = {"must": must_conditions}

        resp = requests.post(
            f"{self._qdrant_url}/collections/{self._collection}/points/search",
            json=body,
            timeout=_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json().get("result", [])

        # Deduplicate by data content — same text can appear in multiple points
        seen: set[str] = set()
        deduped: list[dict] = []
        for point in result:
            data = point.get("payload", {}).get("data", "")
            if not data or data in seen:
                continue
            seen.add(data)
            deduped.append({
                "data": data,
                "speaker": point.get("payload", {}).get("speaker", ""),
                "date": point.get("payload", {}).get("date", ""),
                "source": point.get("payload", {}).get("source", ""),
                "score": round(point.get("score", 0.0), 4),
            })
        return deduped

    def is_available(self) -> bool:
        """Quick health check — collection exists and TEI is reachable."""
        try:
            r = requests.get(
                f"{self._qdrant_url}/collections/{self._collection}",
                timeout=3,
            )
            if r.status_code != 200:
                return False
            r2 = requests.get(f"{self._tei_url}/info", timeout=3)
            return r2.status_code == 200
        except Exception:
            return False
