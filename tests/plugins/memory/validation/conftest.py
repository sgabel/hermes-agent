"""PRD-050 validation-suite conftest — hermetic-tier env hygiene.

Adversarial C-4: ``plugins/memory/canon/store.py`` honors
``HERMES_CANON_QDRANT_URL`` / ``HERMES_CANON_TEI_URL`` as outright overrides,
but the top-level tests/ conftest does NOT blank them — a developer shell
export would silently point "hermetic" tests at a live store. Every module in
this package is fully network-stubbed regardless; this fixture removes the
env escape hatch too.
"""

import pytest


@pytest.fixture(autouse=True)
def _blank_canon_env(monkeypatch):
    monkeypatch.delenv("HERMES_CANON_QDRANT_URL", raising=False)
    monkeypatch.delenv("HERMES_CANON_TEI_URL", raising=False)
