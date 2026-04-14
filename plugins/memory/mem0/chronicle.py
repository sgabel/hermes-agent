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
        self._qdrant_url = qdrant_url.rstrip("/")
        self._tei_url = tei_url.rstrip("/")
        self._collection = collection

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
