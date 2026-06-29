"""PRD-037 FR-2 — ChronicleSearcher container→localhost endpoint fallback.

mem0.json carries container DNS (``http://qdrant:6333`` / ``http://tei-bge-m3:80``)
which is unreachable from a host process. The searcher must transparently fall
back to the localhost defaults so chronicle_search AND on_session_end writes work
both in-container and on the host (AC-005). Mirrors CanonStore.from_config.
"""

from unittest.mock import patch

from plugins.memory.mem0.chronicle import (
    ChronicleSearcher,
    _DEFAULT_QDRANT_URL,
    _DEFAULT_TEI_URL,
)


def test_unreachable_container_url_falls_back_to_localhost():
    # Configured container URLs that do not answer → both swapped to localhost.
    with patch.object(ChronicleSearcher, "_reachable", return_value=False), \
         patch.object(ChronicleSearcher, "_tei_reachable", return_value=False):
        s = ChronicleSearcher(
            qdrant_url="http://qdrant:6333",
            tei_url="http://tei-bge-m3:80",
        )
    assert s._qdrant_url == _DEFAULT_QDRANT_URL
    assert s._tei_url == _DEFAULT_TEI_URL


def test_reachable_container_url_is_kept():
    # In-container: both configured URLs answer → keep them (no localhost swap).
    with patch.object(ChronicleSearcher, "_reachable", return_value=True), \
         patch.object(ChronicleSearcher, "_tei_reachable", return_value=True):
        s = ChronicleSearcher(
            qdrant_url="http://qdrant:6333",
            tei_url="http://tei-bge-m3:80",
        )
    assert s._qdrant_url == "http://qdrant:6333"
    assert s._tei_url == "http://tei-bge-m3:80"


def test_tei_falls_back_independently_of_qdrant():
    # M4 (Codex): Qdrant reachable but TEI is an unreachable container URL →
    # only TEI swaps to localhost (embeds would otherwise fail silently).
    with patch.object(ChronicleSearcher, "_reachable", return_value=True), \
         patch.object(ChronicleSearcher, "_tei_reachable", return_value=False):
        s = ChronicleSearcher(
            qdrant_url="http://qdrant:6333",
            tei_url="http://tei-bge-m3:80",
        )
    assert s._qdrant_url == "http://qdrant:6333"
    assert s._tei_url == _DEFAULT_TEI_URL


def test_localhost_default_skips_probe():
    # Already-localhost defaults must NOT be probed (no network on the hot path).
    with patch.object(ChronicleSearcher, "_reachable", side_effect=AssertionError(
        "should not probe the localhost default"
    )), patch.object(ChronicleSearcher, "_tei_reachable", side_effect=AssertionError(
        "should not probe the localhost default"
    )):
        s = ChronicleSearcher()
    assert s._qdrant_url == _DEFAULT_QDRANT_URL
    assert s._tei_url == _DEFAULT_TEI_URL


def test_trailing_slash_normalized_after_fallback():
    with patch.object(ChronicleSearcher, "_reachable", return_value=True), \
         patch.object(ChronicleSearcher, "_tei_reachable", return_value=True):
        s = ChronicleSearcher(qdrant_url="http://qdrant:6333/", tei_url="http://tei-bge-m3:80/")
    assert not s._qdrant_url.endswith("/")
    assert not s._tei_url.endswith("/")
