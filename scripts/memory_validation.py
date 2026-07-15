#!/usr/bin/env python3
"""PRD-050 — memory validation suite driver (FR-1).

One failable entry point over the four memory-quality pillars:

    ingest     — run the FR-2 hermetic ingest-fidelity test module (pytest
                 subprocess); the driver folds its pass/fail into the report.
    recall     — FR-3b integration tier: seed ``<prefix>_recall`` from the
                 committed corpus, embed via real TEI, search via
                 ChronicleSearcher, score precision@2 + filter-correctness.
    adversary  — FR-4b integration tier: run the REAL ``run_adversary`` over the
                 committed gold set against the production judge (best-of-3
                 majority per item), stamp verdicts onto ``<prefix>_adversary``
                 lab copies only; plus the FR-5 hermetic consolidation replay.
    canon      — FR-6 read-only live verify (``--live`` gated): every
                 ``sylva_canon`` point governed (adversary_verdict+ratified_by),
                 double ``render_self_brief()`` byte-identical.
    all        — ingest → recall → adversary → canon(--live only).

Exit code is non-zero iff a HARD threshold (or, under ``--strict``, an
ADVISORY floor) is breached — thresholds live in ONE place,
``plugins/memory/validation_lib.py`` (FR-1b), and are printed at run start.

Safety invariants (adversarial S-1 / N-4 / N-5 — do not weaken):

  * The FIRST action in ``main()`` installs an in-process audit capture stub
    over ``autonomy.audit.record`` BEFORE any plugin import chain runs —
    otherwise the ingest/consolidation paths would append to the REAL
    ``~/.hermes/autonomy/audit.jsonl``. The report carries the real ledger's
    line count before/after as proof (AC-010).
  * Every write-target collection MUST ``startswith("sylva_lab_validation")``
    — deliberately stricter than consolidation's ``_assert_writable_target``,
    which admits the live ``sylva_candidates`` and the historic ``sylva_lab``
    (that one holds the preserved negative controls).
  * Live collections are never mutated: ``sylva_canon`` / ``sylva_candidates``
    / ``sylva_facts`` get a HARD delta=0 count check; ``sylva_chronicle`` is
    attribution-checked (its live session-end writer races a raw delta), plus
    a growth warning.

Module top stays stdlib-only: plugin imports (including validation_lib, whose
package ``__init__`` pulls hermes_cli.config) happen inside functions, after
the audit stub is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── constants ────────────────────────────────────────────────────────────────
_LAB_PREFIX_REQUIRED = "sylva_lab_validation"
_SUITE_MARKER = "prd050-memory-validation"
_GOLD_DIR_DEFAULT = _REPO_ROOT / "tests" / "plugins" / "memory" / "validation" / "gold"
_INGEST_TEST = _REPO_ROOT / "tests" / "plugins" / "memory" / "validation" / "test_ingest_fidelity.py"

_DEFAULT_QDRANT = "http://localhost:6333"
_DEFAULT_TEI = "http://localhost:8085"
# adversarial N-2: the certifying judge defaults to the PRODUCTION adversary
# model (the 35B at :8081), not the aux 4B.
_DEFAULT_LLM = "http://localhost:8081/v1"

# live collections the suite must never mutate
_HARD_DELTA_COLLECTIONS = ("sylva_canon", "sylva_candidates", "sylva_facts")
_CHRONICLE = "sylva_chronicle"

# fail-closed severity order for best-of-N ties (lower = safer to pick):
# never let a tie accidentally promote toward affirm.
_VERDICT_SEVERITY = {"refute": 0, "tension": 1, "demote": 2, "merge": 3, "affirm": 4}

_AUDIT_CAPTURE: List[Dict[str, Any]] = []


# ── S-1: audit capture stub (must run before any plugin import) ─────────────
def _install_audit_stub() -> None:
    """Rebind ``autonomy.audit.record`` to an in-process capturer.

    Both write paths the suite can trigger resolve ``record`` at call time
    (``from autonomy import audit; audit.record(...)``), so rebinding the
    module attribute intercepts every ledger write for the life of this
    process. Installed as the first action of ``main()`` (adversarial S-1).
    """
    import autonomy.audit as _aud

    def _capture(**kw: Any) -> Dict[str, Any]:
        _AUDIT_CAPTURE.append(kw)
        return {"persisted": False, "captured_by": _SUITE_MARKER, **kw}

    _aud.record = _capture  # type: ignore[assignment]


def _real_ledger_lines() -> Optional[int]:
    """Line count of the REAL audit ledger (read-only; AC-010 proof)."""
    try:
        from hermes_constants import get_hermes_home

        p = get_hermes_home() / "autonomy" / "audit.jsonl"
    except Exception:
        p = Path.home() / ".hermes" / "autonomy" / "audit.jsonl"
    try:
        if not p.exists():
            return 0
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


# ── write-target guard (adversarial N-4) ─────────────────────────────────────
def _assert_lab(collection: str) -> str:
    """Refuse any write target outside the validation sandbox namespace."""
    if not collection.startswith(_LAB_PREFIX_REQUIRED):
        raise SystemExit(
            f"REFUSING write target {collection!r}: every suite write target "
            f"must start with {_LAB_PREFIX_REQUIRED!r} (PRD-050 sandbox guard)"
        )
    return collection


# ── Qdrant helpers (reads unrestricted; writes guarded) ──────────────────────
def _qdrant_count(qdrant_url: str, collection: str) -> Optional[int]:
    import requests

    try:
        r = requests.post(
            f"{qdrant_url.rstrip('/')}/collections/{collection}/points/count",
            json={"exact": True},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        return int(r.json().get("result", {}).get("count", 0))
    except Exception:
        return None


def _qdrant_delete_collection(qdrant_url: str, collection: str) -> None:
    import requests

    _assert_lab(collection)
    requests.delete(
        f"{qdrant_url.rstrip('/')}/collections/{collection}", timeout=15
    ).raise_for_status()


def _qdrant_create_collection(qdrant_url: str, collection: str) -> None:
    import requests

    _assert_lab(collection)
    requests.put(
        f"{qdrant_url.rstrip('/')}/collections/{collection}",
        json={"vectors": {"size": 1024, "distance": "Cosine", "on_disk": True}},
        timeout=15,
    ).raise_for_status()
    # keyword indexes mirror sylva_chronicle (speaker/date filtered search)
    for field in ("speaker", "date"):
        try:
            requests.put(
                f"{qdrant_url.rstrip('/')}/collections/{collection}/index",
                json={"field_name": field, "field_schema": "keyword"},
                timeout=15,
            ).raise_for_status()
        except Exception:
            pass  # best-effort — unindexed filters still work in non-strict mode


def _qdrant_upsert(qdrant_url: str, collection: str, points: List[Dict[str, Any]]) -> None:
    import requests

    _assert_lab(collection)
    requests.put(
        f"{qdrant_url.rstrip('/')}/collections/{collection}/points",
        params={"wait": "true"},
        json={"points": points},
        timeout=60,
    ).raise_for_status()


def _chronicle_scan(qdrant_url: str, marker_sources: set) -> Tuple[List[str], Optional[int]]:
    """Full payload-only scroll of sylva_chronicle → attribution violations.

    Returns ``(violations, points_seen)``. A violation is any point whose
    ``suite`` payload equals the suite marker or whose ``source`` is one of the
    suite's own doc/item ids — none should ever exist in the live chronicle
    (adversarial N-5: raw deltas race the live session-end writer).
    """
    import requests

    violations: List[str] = []
    seen = 0
    offset: Any = None
    try:
        while True:
            body: Dict[str, Any] = {"limit": 1000, "with_payload": True, "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            r = requests.post(
                f"{qdrant_url.rstrip('/')}/collections/{_CHRONICLE}/points/scroll",
                json=body,
                timeout=30,
            )
            r.raise_for_status()
            result = r.json().get("result", {})
            for pt in result.get("points", []):
                seen += 1
                payload = pt.get("payload") or {}
                src = str(payload.get("source", ""))
                if payload.get("suite") == _SUITE_MARKER or src in marker_sources:
                    violations.append(f"point {pt.get('id')} source={src!r}")
            offset = result.get("next_page_offset")
            if offset is None:
                break
    except Exception as e:
        return [f"chronicle scan failed: {type(e).__name__}: {e}"], None
    return violations, seen


# ── ingest tier (FR-2 — the hermetic module IS the single source, N-7) ───────
def _run_ingest(report: Dict[str, Any], failures: List[str]) -> None:
    if not _INGEST_TEST.exists():
        report["ingest"] = {"passed": False, "error": f"missing {_INGEST_TEST}"}
        failures.append(f"[HARD] ingest: test module missing ({_INGEST_TEST})")
        return
    env = dict(os.environ, TZ="UTC")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(_INGEST_TEST), "-q"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
        timeout=600,
    )
    tail = (proc.stdout + proc.stderr)[-2000:]
    report["ingest"] = {"passed": proc.returncode == 0, "returncode": proc.returncode, "output_tail": tail}
    if proc.returncode != 0:
        failures.append(f"[HARD] ingest fidelity module failed (pytest rc={proc.returncode})")


# ── recall tier (FR-3b) ──────────────────────────────────────────────────────
def _run_recall(
    args: argparse.Namespace,
    report: Dict[str, Any],
    failures: List[str],
    created: List[str],
) -> None:
    import yaml

    from plugins.memory.mem0.chronicle import ChronicleSearcher

    gold = yaml.safe_load((Path(args.gold) / "recall_corpus.yaml").read_text(encoding="utf-8"))
    docs = gold["docs"]
    queries = gold["queries"]
    k = int(gold.get("meta", {}).get("precision_k", 2))
    collection = _assert_lab(f"{args.sandbox_prefix}_recall")

    searcher = ChronicleSearcher(
        qdrant_url=args.qdrant_url, tei_url=args.tei_url, collection=collection
    )

    # (re)create the sandbox collection idempotently
    _qdrant_delete_ignore_missing(args.qdrant_url, collection)
    _qdrant_create_collection(args.qdrant_url, collection)
    created.append(collection)

    points = []
    for d in docs:
        vec = searcher.embed(d["data"])
        points.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{_SUITE_MARKER}:recall:{d['id']}")),
                "payload": {
                    "data": d["data"],
                    "speaker": d["speaker"],
                    "date": d["date"],
                    "source": d["id"],
                    "category": "journal",
                    "suite": _SUITE_MARKER,
                },
                "vector": vec,
            }
        )
    _qdrant_upsert(args.qdrant_url, collection, points)

    results_by_query: Dict[str, List[str]] = {}
    gold_mapping: Dict[str, Dict[str, Any]] = {}
    for q in queries:
        filt = q.get("filter") or {}
        hits = searcher.search(
            q["query"],
            speaker=filt.get("speaker", "any") or "any",
            date_from=filt.get("date_from", ""),
            date_to=filt.get("date_to", ""),
            top_k=5,
        )
        results_by_query[q["id"]] = [h["source"] for h in hits]
        allowed = None
        if filt:
            allowed = [
                d["id"]
                for d in docs
                if (not filt.get("speaker") or d["speaker"] == filt["speaker"])
                and (not filt.get("date_from") or d["date"] >= filt["date_from"])
                and (not filt.get("date_to") or d["date"] <= filt["date_to"])
            ]
        gold_mapping[q["id"]] = {"expected_ids": q["expected_ids"], "allowed_ids": allowed}

    from plugins.memory import validation_lib as V

    report["recall"] = V.score_recall(results_by_query, gold_mapping, k=k)
    report["recall"]["collection"] = collection
    report["recall"]["doc_count"] = len(docs)


def _qdrant_delete_ignore_missing(qdrant_url: str, collection: str) -> None:
    try:
        _qdrant_delete_collection(qdrant_url, collection)
    except SystemExit:
        raise
    except Exception:
        pass


# ── adversary tier (FR-4b) + consolidation replay (FR-5) ─────────────────────
def _probe_model(llm_url: str) -> str:
    """Resolve the served model id from ``/v1/models`` (explicit beats probe)."""
    import requests

    r = requests.get(f"{llm_url.rstrip('/')}/models", timeout=10)
    r.raise_for_status()
    data = r.json().get("data") or []
    if not data:
        raise SystemExit(f"judge probe: {llm_url}/models returned no models")
    return str(data[0].get("id", ""))


def _config_adversary_model() -> str:
    """Best-effort: the config-resolved canon_adversary model (empty = main model)."""
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
        m = cfg_get(cfg, "auxiliary", "canon_adversary", "model") or ""
        if not m:
            m = cfg_get(cfg, "model", "default") or ""
        return str(m)
    except Exception:
        return ""


def _majority_verdict(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best-of-N majority by verdict string; ties break fail-closed (C-3).

    The representative dict for the winning verdict prefers a non-error-shaped
    run — but if every run in the winning group is error-shaped (dead judge),
    the error shape survives so the HARD ``error_verdicts`` gate can fire.
    """
    from plugins.memory import validation_lib as V

    counts = Counter(str(r.get("verdict", "")) for r in runs)
    top_n = max(counts.values())
    tied = [v for v, c in counts.items() if c == top_n]
    chosen = min(tied, key=lambda v: _VERDICT_SEVERITY.get(v, -1))
    group = [r for r in runs if str(r.get("verdict", "")) == chosen]
    non_error = [r for r in group if not V.is_error_verdict(r)]
    return (non_error or group)[0]


def _run_adversary(
    args: argparse.Namespace,
    report: Dict[str, Any],
    failures: List[str],
    created: List[str],
) -> None:
    import yaml
    from openai import OpenAI

    from plugins.memory import validation_lib as V
    from plugins.memory.canon.ratification import run_adversary
    from plugins.memory.canon.store import CanonStore

    gold_doc = yaml.safe_load((Path(args.gold) / "adversary_gold.yaml").read_text(encoding="utf-8"))
    meta = gold_doc.get("meta", {})
    items = gold_doc["items"]

    # judge resolution (adversarial N-1/N-2): model EXPLICIT, api_key non-empty —
    # model=None or api_key="" silently becomes a fail-closed refute, the exact
    # vacuous-pass hole the error-verdict hard gate exists to catch.
    model = args.model or _probe_model(args.llm_url)
    if not model:
        raise SystemExit("judge model unresolved: pass --model or fix --llm-url")
    client = OpenAI(base_url=args.llm_url, api_key="local")
    config_model = _config_adversary_model()

    collection = _assert_lab(f"{args.sandbox_prefix}_adversary")
    store = CanonStore(qdrant_url=args.qdrant_url, tei_url=args.tei_url)

    # lab copies — verdicts are stamped ONLY here, via direct set_payload
    # (NEVER route_verdict/run_ratification: those write canon + the ledger).
    _qdrant_delete_ignore_missing(args.qdrant_url, collection)
    _qdrant_create_collection(args.qdrant_url, collection)
    created.append(collection)
    lab_ids: Dict[str, str] = {}
    lab_points = []
    for it in items:
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{_SUITE_MARKER}:adversary:{it['id']}"))
        lab_ids[it["id"]] = pid
        lab_points.append(
            {
                "id": pid,
                "payload": {
                    "statement": it["statement"],
                    "facet": it.get("facet", ""),
                    "source_event": it.get("source_event")
                    or {"claim": "", "provenance_refs": []},
                    "class": it["class"],
                    "mode": it.get("mode", ""),
                    "status": "lab_validation",
                    "suite": _SUITE_MARKER,
                    "source": it["id"],
                },
                "vector": store.embed(it["statement"]),
            }
        )
    _qdrant_upsert(args.qdrant_url, collection, lab_points)

    modes = meta.get("modes", {})
    expected_sets = meta.get("expected_sets", {})
    verdicts_by_item: Dict[str, Dict[str, Any]] = {}
    runs_by_item: Dict[str, List[Dict[str, Any]]] = {}
    gold_mapping: Dict[str, Dict[str, Any]] = {}

    for it in items:
        cls = it["class"]
        mode = it.get("mode") or modes.get(cls, "strict")
        candidate = {
            "statement": it["statement"],
            "facet": it.get("facet", ""),
            "tier": it.get("tier", ""),
            "source_event": it.get("source_event") or {"claim": "", "provenance_refs": []},
        }
        runs = []
        for _ in range(max(1, args.runs)):
            runs.append(
                run_adversary(
                    candidate,
                    [],
                    model=model,
                    client=client,
                    evidence=it.get("source_material", "") or "",
                    mode=mode,
                )
            )
        selected = _majority_verdict(runs)
        verdicts_by_item[it["id"]] = selected
        runs_by_item[it["id"]] = [
            {
                "verdict": r.get("verdict", ""),
                "error_shaped": V.is_error_verdict(r),
                "reason0": (r.get("reasons") or [""])[0][:200],
            }
            for r in runs
        ]
        gold_mapping[it["id"]] = {
            "class": cls,
            "expected": it.get("expected") or expected_sets.get(cls, []),
            "mode": mode,
            "real_control": bool(it.get("real_control")),
        }
        # stamp the lab copy only (direct payload merge — no routing, no ledger)
        store.set_payload(
            _assert_lab(collection),
            lab_ids[it["id"]],
            {"adversary_verdict": selected, "suite": _SUITE_MARKER},
        )

    adv = V.score_adversary(verdicts_by_item, gold_mapping)
    adv["judge_model"] = model
    adv["config_adversary_model"] = config_model
    adv["judge_model_mismatch"] = bool(config_model) and (model != config_model)
    adv["runs_per_item"] = max(1, args.runs)
    adv["runs_by_item"] = runs_by_item
    adv["collection"] = collection
    # per-reason error breakdown (N-1 report requirement)
    err_reasons: Counter = Counter()
    for v in verdicts_by_item.values():
        if V.is_error_verdict(v):
            err_reasons[(v.get("reasons") or [""])[0][:80]] += 1
    adv["error_reason_breakdown"] = dict(err_reasons)
    report["adversary"] = adv

    # FR-5 — consolidation proposer discipline replay (hermetic, in-process)
    report["consolidation_replay"] = _consolidation_replay(failures)


def _consolidation_replay(failures: List[str]) -> Dict[str, Any]:
    """FR-5: replay ``run_consolidation`` with injected fakes — no network, no
    real DB, agency gatherer stubbed (never reads the live ledger/kanban)."""
    from plugins.memory.canon import consolidation as C
    from plugins.memory.canon.schema import content_hash, validate_consolidation_payload

    checks: Dict[str, str] = {}

    class _Store:
        def __init__(self, existing=None):
            self.upserts: List[tuple] = []
            self._existing = existing or {}

        def ensure_collections(self, collections):
            pass

        def upsert(self, collection, points):
            self.upserts.append((collection, points))

        def get_canon(self, *, collection, status, **_kw):
            return list(self._existing.get(collection, []))

    class _DB:
        def __init__(self):
            self._sessions = [{"id": "s1", "source": "tui", "last_active": "2026-07-15"}]

        def list_sessions_rich(self, *, limit, exclude_sources, min_message_count, order_by_last_active):
            return self._sessions[:limit]

        def get_messages_as_conversation(self, sid):
            return [{"role": "user", "content": "hello from the validation replay"}]

        def close(self):
            pass

    def _proposal(stmt: str) -> Dict[str, Any]:
        return {
            "statement": stmt,
            "facet": "value",
            "tier": "core",
            "source_event": {
                "claim": "validation replay fixture event",
                "provenance_refs": ["session:mv-replay"],
            },
            "interpretation": "fixture interpretation",
        }

    stmt_a = "I verify live state before asserting it."
    stmt_b = "I prefer reversible changes with a tested rollback path."

    orig_agency = C._gather_agency_layer
    C._gather_agency_layer = lambda: []  # hermetic: no live ledger/kanban reads
    try:
        # 1. sole-writer target guard — sylva_canon must raise before any work
        try:
            C.run_consolidation(
                target_collection="sylva_canon",
                store=_Store(),
                db=_DB(),
                derive_fn=lambda s, a: ([_proposal(stmt_a)], "mv-model"),
            )
            checks["sole_writer_guard"] = "FAIL: writing sylva_canon did not raise"
        except ValueError:
            checks["sole_writer_guard"] = "ok"

        # 2. dry_run derives but never writes
        st = _Store()
        res = C.run_consolidation(
            store=st,
            db=_DB(),
            dry_run=True,
            derive_fn=lambda s, a: ([_proposal(stmt_a)], "mv-model"),
        )
        if st.upserts == [] and res.dry_run:
            checks["dry_run_zero_writes"] = "ok"
        else:
            checks["dry_run_zero_writes"] = f"FAIL: upserts={len(st.upserts)}"

        # 3. provenance completeness — every written payload passes the
        #    consolidation-scoped validator (run_id/content_hash/claim)
        st = _Store()
        C.run_consolidation(
            store=st,
            db=_DB(),
            derive_fn=lambda s, a: ([_proposal(stmt_a), _proposal(stmt_b)], "mv-model"),
        )
        try:
            payloads = [p["payload"] for _, pts in st.upserts for p in pts]
            if not payloads:
                checks["provenance_completeness"] = "FAIL: no payloads written"
            else:
                for p in payloads:
                    validate_consolidation_payload(p)
                checks["provenance_completeness"] = f"ok ({len(payloads)} payloads)"
        except Exception as e:
            checks["provenance_completeness"] = f"FAIL: {e}"

        # 4. cross-store dedup vs pre-seeded hashes — A is already known
        seeded_row = ("pre", {"content_hash": content_hash(stmt_a), "status": "candidate"})
        st = _Store(
            existing={
                C.CANDIDATES_COLLECTION: [seeded_row],
                "sylva_canon": [seeded_row],
            }
        )
        res = C.run_consolidation(
            store=st,
            db=_DB(),
            derive_fn=lambda s, a: ([_proposal(stmt_a), _proposal(stmt_b)], "mv-model"),
        )
        written = [p["payload"]["statement"] for _, pts in st.upserts for p in pts]
        if res.candidates_written == 1 and written == [stmt_b]:
            checks["cross_store_dedup"] = "ok"
        else:
            checks["cross_store_dedup"] = f"FAIL: written={written!r}"
    finally:
        C._gather_agency_layer = orig_agency

    passed = all(v.startswith("ok") for v in checks.values())
    for name, v in checks.items():
        if not v.startswith("ok"):
            failures.append(f"[HARD] consolidation replay: {name}: {v}")
    return {"checks": checks, "passed": passed}


# ── canon tier (FR-6, read-only, --live gated) ───────────────────────────────
def _run_canon(args: argparse.Namespace, report: Dict[str, Any], failures: List[str]) -> None:
    if not args.live:
        report["canon"] = {"skipped": "requires --live (read-only live verify)"}
        return

    # check_seed / render resolve endpoints via env → point them at our args
    os.environ["HERMES_CANON_QDRANT_URL"] = args.qdrant_url
    os.environ["HERMES_CANON_TEI_URL"] = args.tei_url

    from plugins.memory.canon.render import _read_soul_md, render_self_brief
    from tests.plugins.memory.check_seed import validate

    canon_failures = validate("sylva_canon")
    for f in canon_failures:
        failures.append(f"[HARD] canon verify: {f}")

    brief1 = render_self_brief() or ""
    brief2 = render_self_brief() or ""
    double_identical = brief1 == brief2
    non_empty = bool(brief1.strip())
    if not non_empty:
        failures.append("[HARD] canon verify: render_self_brief returned empty")
    if not double_identical:
        failures.append("[HARD] canon verify: double render not byte-identical")

    soul = _read_soul_md() or ""
    soul_preamble_present = bool(soul.strip()) and soul.strip()[:64] in brief1
    if soul.strip() and not soul_preamble_present:
        failures.append("[HARD] canon verify: SOUL.md preamble missing from brief")

    report["canon"] = {
        "governance_failures": canon_failures,
        "double_render_identical": double_identical,
        "brief_non_empty": non_empty,
        "soul_preamble_present": soul_preamble_present,
        "brief_chars": len(brief1),
        # runbook §2: compare against the in-container hash at deploy time
        "brief_sha256": hashlib.sha256(brief1.encode("utf-8")).hexdigest(),
    }


# ── marker set for the chronicle attribution scan ────────────────────────────
def _suite_marker_sources(gold_dir: Path) -> set:
    import yaml

    markers: set = set()
    for fname, key in (("recall_corpus.yaml", "docs"), ("adversary_gold.yaml", "items")):
        try:
            doc = yaml.safe_load((gold_dir / fname).read_text(encoding="utf-8"))
            markers.update(str(entry["id"]) for entry in doc.get(key, []))
        except Exception:
            pass
    return markers


# ── main ─────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    # S-1: FIRST action — before argparse touches nothing, but critically
    # before any plugins.* / hermes_cli.* import chain can run.
    _install_audit_stub()

    ap = argparse.ArgumentParser(
        description="PRD-050 memory validation suite (failable driver)"
    )
    ap.add_argument("suite", choices=["ingest", "recall", "adversary", "canon", "all"])
    ap.add_argument("--gold", default=str(_GOLD_DIR_DEFAULT), help="gold-set directory")
    ap.add_argument(
        "--qdrant-url",
        default=os.environ.get("HERMES_CANON_QDRANT_URL") or _DEFAULT_QDRANT,
    )
    ap.add_argument(
        "--tei-url",
        default=os.environ.get("HERMES_CANON_TEI_URL") or _DEFAULT_TEI,
    )
    ap.add_argument(
        "--llm-url",
        default=os.environ.get("HERMES_VALIDATION_LLM_URL") or _DEFAULT_LLM,
        help="judge endpoint (default: the production adversary 35B)",
    )
    ap.add_argument("--model", default="", help="judge model id (default: probe /v1/models)")
    ap.add_argument("--sandbox-prefix", default=_LAB_PREFIX_REQUIRED)
    ap.add_argument("--json", dest="json_path", default="", help="write the machine report here")
    ap.add_argument("--live", action="store_true", help="enable the read-only live canon verify")
    ap.add_argument("--strict", action="store_true", help="advisory floors also fail the run")
    ap.add_argument("--keep", action="store_true", help="retain sandbox collections")
    ap.add_argument("--runs", type=int, default=3, help="adversary runs per item (majority)")
    args = ap.parse_args(argv)

    # even a caller-supplied prefix must stay inside the validation namespace
    _assert_lab(args.sandbox_prefix)

    from plugins.memory import validation_lib as V

    print(f"PRD-050 memory validation — suite={args.suite} strict={args.strict}")
    print(f"thresholds: {V.THRESHOLDS_DOC}")
    print(
        f"endpoints: qdrant={args.qdrant_url} tei={args.tei_url} llm={args.llm_url} "
        f"sandbox_prefix={args.sandbox_prefix}"
    )

    started = datetime.now(timezone.utc).isoformat()
    failures: List[str] = []
    created: List[str] = []
    report: Dict[str, Any] = {
        "prd": "PRD-050",
        "suite": args.suite,
        "strict": args.strict,
        "started_at": started,
        "thresholds": V.THRESHOLDS,
        "endpoints": {
            "qdrant": args.qdrant_url,
            "tei": args.tei_url,
            "llm": args.llm_url,
        },
    }

    ledger_before = _real_ledger_lines()
    counts_before = {
        c: _qdrant_count(args.qdrant_url, c)
        for c in (*_HARD_DELTA_COLLECTIONS, _CHRONICLE)
    }

    try:
        if args.suite in ("ingest", "all"):
            _run_ingest(report, failures)
        if args.suite in ("recall", "all"):
            _run_recall(args, report, failures, created)
        if args.suite in ("adversary", "all"):
            _run_adversary(args, report, failures, created)
        if args.suite in ("canon", "all"):
            if args.suite == "all" and not args.live:
                report["canon"] = {"skipped": "all without --live: canon verify skipped"}
            else:
                _run_canon(args, report, failures)
    except SystemExit:
        raise
    except Exception as e:
        report["error"] = {"type": type(e).__name__, "detail": str(e), "trace": traceback.format_exc()[-3000:]}
        failures.append(f"[HARD] driver error in suite {args.suite!r}: {type(e).__name__}: {e}")
    finally:
        # live-safety accounting happens even on a mid-suite error
        counts_after = {
            c: _qdrant_count(args.qdrant_url, c)
            for c in (*_HARD_DELTA_COLLECTIONS, _CHRONICLE)
        }
        hard_violations: List[str] = []
        for c in _HARD_DELTA_COLLECTIONS:
            b, a = counts_before.get(c), counts_after.get(c)
            if b is not None and a is not None and a != b:
                hard_violations.append(f"{c}: {b} -> {a} (must be unchanged)")
        attribution_violations: List[str] = []
        chronicle_growth = 0
        touched_qdrant = args.suite in ("recall", "adversary", "canon", "all")
        if touched_qdrant and counts_before.get(_CHRONICLE) is not None:
            markers = _suite_marker_sources(Path(args.gold))
            attribution_violations, _seen = _chronicle_scan(args.qdrant_url, markers)
            b = counts_before.get(_CHRONICLE) or 0
            a = counts_after.get(_CHRONICLE) or 0
            chronicle_growth = a - b
            if chronicle_growth > 0:
                print(
                    f"WARNING: sylva_chronicle grew by {chronicle_growth} during the run "
                    "(live session-end writer is active; attribution check governs)"
                )
        report["live_counts"] = {
            "before": counts_before,
            "after": counts_after,
            "hard_delta_violations": hard_violations,
            "chronicle_attribution_violations": attribution_violations,
            "chronicle_growth": chronicle_growth,
        }

        ledger_after = _real_ledger_lines()
        report["audit"] = {
            "stub_installed": True,
            "captured_records": len(_AUDIT_CAPTURE),
            "real_ledger_lines_before": ledger_before,
            "real_ledger_lines_after": ledger_after,
            "real_ledger_delta": (
                (ledger_after - ledger_before)
                if (ledger_before is not None and ledger_after is not None)
                else None
            ),
        }
        if (
            ledger_before is not None
            and ledger_after is not None
            and ledger_after != ledger_before
        ):
            failures.append(
                f"[HARD] real audit ledger grew {ledger_before} -> {ledger_after} "
                "(the S-1 stub must intercept every record)"
            )

        if not args.keep:
            for c in created:
                try:
                    _qdrant_delete_collection(args.qdrant_url, c)
                except Exception as e:
                    print(f"WARNING: teardown of {c} failed: {e}")
        elif created:
            print(f"--keep: retained sandbox collections: {', '.join(created)}")

    exit_code, threshold_failures = V.evaluate(report)
    failures.extend(threshold_failures)
    rc = 1 if failures else 0

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["failures"] = failures
    report["exit_code"] = rc

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"report written: {args.json_path}")

    if failures:
        print(f"\nFAIL — {len(failures)} finding(s):")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("\nPASS — all gates green" + (" (strict)" if args.strict else ""))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
