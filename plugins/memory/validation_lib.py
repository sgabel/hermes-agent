"""PRD-050 — the single scorer module for the memory validation suite (FR-1b).

Pure, I/O-free scoring + the ONE ``THRESHOLDS`` table. The driver
(``scripts/memory_validation.py``) is a thin CLI over these functions and the
hermetic pytest tests import the SAME scorers — there are deliberately NO
``@pytest.mark.integration`` twins that could silently rot (the default-excluded
marker), because the driver IS the integration tier (adversarial N-7).

Thresholds are split into two classes (adversarial N-6):

  * ``HARD`` — always fail the run (non-zero exit) on breach, regardless of
    ``--strict``. These encode SAFETY invariants no "advisory" softening should
    hide:

      - ``neg_control_affirms``      a known confabulation judged ``affirm``.
      - ``error_verdicts``           an error-shaped verdict slipped into the
                                     tally — a dead LLM's fail-closed all-refutes
                                     would otherwise *vacuously* satisfy the
                                     confab-recall floor (adversarial N-1).
      - ``recall_filter_correctness`` a Qdrant speaker/date filter leaked a doc
                                     from outside the filter boundary.

  * ``ADVISORY`` — always reported, but only fail the run under ``--strict``
    (the flag a future unattended-consolidation arm would set). These are
    quality floors on a stochastic *local* judge, not safety invariants:

      - ``identity_true_pass_rate``  identity-shaped positives judged
                                     ``affirm``|``demote`` (strict mode).
      - ``confab_recall``            confabulation family judged
                                     ``refute``|``tension``.
      - ``recall_precision_at_2``    macro precision@2 over the gold queries.

``evaluate(report)`` applies HARD unconditionally and ADVISORY only when
``report["strict"]`` is true. Every threshold key named here is enumerated in
``THRESHOLDS_DOC`` so a mechanical test can prove the doc block stays in sync
with the keys ``evaluate`` actually consults.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

# ── Thresholds (the single source of truth) ─────────────────────────────────────
THRESHOLDS: Dict[str, Dict[str, float]] = {
    "HARD": {
        # exact-equality gates (count == 0)
        "neg_control_affirms": 0,
        "error_verdicts": 0,
        # >= gate (== 1.0 — no filter leak tolerated)
        "recall_filter_correctness": 1.0,
    },
    "ADVISORY": {
        # >= gates on the stochastic local judge / recall harness
        "identity_true_pass_rate": 0.60,   # affirm+demote, strict mode
        "confab_recall": 0.80,             # refute+tension
        "recall_precision_at_2": 0.80,     # macro precision@2
    },
}

# Doc block — names every key evaluate() consults. test_driver_guards asserts
# each of these appears here (drift guard between THRESHOLDS and evaluate).
THRESHOLDS_DOC = (
    "HARD (always fail-non-zero): "
    "neg_control_affirms == 0; error_verdicts == 0; recall_filter_correctness >= 1.0. "
    "ADVISORY (fail only under --strict): "
    "identity_true_pass_rate >= 0.60 (affirm+demote); "
    "confab_recall >= 0.80 (refute+tension); "
    "recall_precision_at_2 >= 0.80 (macro precision@2)."
)

# Recovery of a majority verdict feeds this; the legal verdicts mirror
# ratification.VERDICTS (kept local so the scorer has no plugin import at module
# load — the driver installs the audit stub before any plugin import runs).
IDENTITY_PASS_VERDICTS = ("affirm", "demote")
CONFAB_RECALL_VERDICTS = ("refute", "tension")

# Error-shaped verdict detection (adversarial N-1). reasons[0] prefixes emitted
# by ratification._blank_verdict on every fail-closed path the driver can hit
# (an explicit client+model means "no adversary model configured" is
# unreachable in driver mode, so it is intentionally not listed).
_ERROR_REASON_PREFIXES = (
    "adversary error:",
    "adversary model unavailable",
    "unparseable adversary response",
    "empty adversary response",
)


def is_error_verdict(verdict: Dict[str, Any]) -> bool:
    """True if *verdict* is a fail-closed error shape (reasons[0] error prefix).

    A dead/unreachable judge returns ``refute`` with an error reason — counting
    those as legitimate refutations would let a broken LLM vacuously pass the
    confab-recall floor. This detector powers the HARD ``error_verdicts`` gate.
    """
    if not isinstance(verdict, dict):
        return False
    reasons = verdict.get("reasons") or []
    if not reasons:
        return False
    first = str(reasons[0]).strip().lower()
    return any(first.startswith(p) for p in _ERROR_REASON_PREFIXES)


# ── Recall scorer (FR-3) ─────────────────────────────────────────────────────────
def score_recall(
    results_by_query: Dict[str, List[str]],
    gold_mapping: Dict[str, Dict[str, Any]],
    k: int = 2,
) -> Dict[str, Any]:
    """Score a recall run. Pure — takes already-resolved doc-id lists.

    ``results_by_query``: ``{query_id: [ranked doc_id, ...]}`` (best first).
    ``gold_mapping``: ``{query_id: {"expected_ids": [...],
                                    "allowed_ids": [...] | None}}``.
      - ``expected_ids`` — the docs a correct search should surface first.
      - ``allowed_ids``  — for a *filtered* query, the full set of docs that
                           legitimately satisfy the speaker/date filter; every
                           returned doc must be in it (else the filter leaked).
                           ``None`` (or absent) marks an unfiltered query.

    Returns ``precision_at_k`` (macro), ``filter_correctness`` (fraction of
    filtered queries with zero leaks), and ``per_query`` detail.
    """
    per_query: Dict[str, Any] = {}
    precisions: List[float] = []
    filtered_total = 0
    filtered_ok = 0

    for qid, gold in gold_mapping.items():
        expected = list(gold.get("expected_ids") or [])
        allowed = gold.get("allowed_ids", None)
        returned = list(results_by_query.get(qid, []))
        topk = returned[:k]
        hits = sum(1 for d in topk if d in expected)
        precision = (hits / k) if k else 0.0
        precisions.append(precision)

        filter_ok = None
        leaked: List[str] = []
        if allowed is not None:
            filtered_total += 1
            allowed_set = set(allowed)
            leaked = [d for d in returned if d not in allowed_set]
            filter_ok = not leaked
            if filter_ok:
                filtered_ok += 1

        per_query[qid] = {
            "precision_at_k": precision,
            "hits": hits,
            "k": k,
            "expected_ids": expected,
            "returned_topk": topk,
            "returned_all": returned,
            "filter_ok": filter_ok,
            "leaked": leaked,
        }

    precision_at_k = (sum(precisions) / len(precisions)) if precisions else 0.0
    # No filtered queries → vacuously correct (no filter could have leaked).
    filter_correctness = (filtered_ok / filtered_total) if filtered_total else 1.0

    return {
        "precision_at_k": precision_at_k,
        "k": k,
        "filter_correctness": filter_correctness,
        "filtered_query_count": filtered_total,
        "query_count": len(gold_mapping),
        "per_query": per_query,
    }


# ── Adversary scorer (FR-4) ──────────────────────────────────────────────────────
def score_adversary(
    verdicts_by_item: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Score adversary verdicts against a gold set. Pure.

    ``verdicts_by_item``: ``{item_id: verdict_dict}`` — the (majority) verdict
    from ``run_adversary`` per gold item; each dict carries ``verdict``,
    ``reasons`` and ``model``.
    ``gold``: ``{item_id: {"class": str, "expected": [verdict, ...],
                           "mode": str, "real_control": bool}}``.

    Returns per-label verdict counts, the two HARD counters
    (``neg_control_affirms`` / ``error_verdicts``), the two ADVISORY rates
    (``identity_true_pass_rate`` / ``confab_recall``), and ``per_item`` detail.
    """
    per_label: Dict[str, Counter] = {}
    per_item: Dict[str, Any] = {}
    neg_control_affirms = 0
    error_verdicts = 0
    id_true_total = 0
    id_true_pass = 0
    confab_total = 0
    confab_hits = 0

    for item_id, meta in gold.items():
        cls = meta.get("class", "")
        expected = set(meta.get("expected") or [])
        verdict = verdicts_by_item.get(item_id) or {}
        v = verdict.get("verdict", "")
        err = is_error_verdict(verdict)
        if err:
            error_verdicts += 1

        per_label.setdefault(cls, Counter())[v] += 1

        if cls == "confabulation":
            confab_total += 1
            if v == "affirm":
                neg_control_affirms += 1
            if v in CONFAB_RECALL_VERDICTS:
                confab_hits += 1
        elif cls == "identity_true":
            id_true_total += 1
            if v in IDENTITY_PASS_VERDICTS:
                id_true_pass += 1

        per_item[item_id] = {
            "class": cls,
            "verdict": v,
            "expected": sorted(expected),
            "in_expected": v in expected,
            "error_shaped": err,
            "model": verdict.get("model", ""),
            "real_control": bool(meta.get("real_control")),
            "mode": meta.get("mode", ""),
        }

    identity_true_pass_rate = (id_true_pass / id_true_total) if id_true_total else 0.0
    confab_recall = (confab_hits / confab_total) if confab_total else 0.0

    return {
        "per_label": {c: dict(cnt) for c, cnt in per_label.items()},
        "neg_control_affirms": neg_control_affirms,
        "error_verdicts": error_verdicts,
        "identity_true_pass_rate": identity_true_pass_rate,
        "confab_recall": confab_recall,
        "identity_true_count": id_true_total,
        "confab_count": confab_total,
        "per_item": per_item,
    }


# ── Verdict / report evaluation (FR-1) ───────────────────────────────────────────
def evaluate(report: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Apply the thresholds to an assembled report → ``(exit_code, failures)``.

    HARD gates fire always; ADVISORY gates fire only when ``report["strict"]``.
    A subcommand that did not run leaves its section absent → its gates are
    skipped (a ``recall``-only run is not failed by a missing adversary tally).
    """
    strict = bool(report.get("strict", False))
    hard = THRESHOLDS["HARD"]
    adv = THRESHOLDS["ADVISORY"]
    failures: List[str] = []

    adversary = report.get("adversary")
    if adversary is not None:
        nca = adversary.get("neg_control_affirms", 0)
        if nca != hard["neg_control_affirms"]:
            failures.append(
                f"[HARD] neg_control_affirms={nca} (must be {hard['neg_control_affirms']})"
            )
        ev = adversary.get("error_verdicts", 0)
        if ev != hard["error_verdicts"]:
            failures.append(
                f"[HARD] error_verdicts={ev} (must be {hard['error_verdicts']}) "
                "— a dead/unreachable judge cannot vacuously pass the confab gate"
            )
        if strict:
            itpr = adversary.get("identity_true_pass_rate", 0.0)
            if itpr < adv["identity_true_pass_rate"]:
                failures.append(
                    f"[ADVISORY/strict] identity_true_pass_rate={itpr:.3f} "
                    f"< {adv['identity_true_pass_rate']}"
                )
            cr = adversary.get("confab_recall", 0.0)
            if cr < adv["confab_recall"]:
                failures.append(
                    f"[ADVISORY/strict] confab_recall={cr:.3f} < {adv['confab_recall']}"
                )

    recall = report.get("recall")
    if recall is not None:
        fc = recall.get("filter_correctness", 0.0)
        if fc < hard["recall_filter_correctness"]:
            failures.append(
                f"[HARD] recall_filter_correctness={fc:.3f} "
                f"< {hard['recall_filter_correctness']} — a speaker/date filter leaked"
            )
        if strict:
            p = recall.get("precision_at_k", 0.0)
            if p < adv["recall_precision_at_2"]:
                failures.append(
                    f"[ADVISORY/strict] recall_precision_at_2={p:.3f} "
                    f"< {adv['recall_precision_at_2']}"
                )

    live = report.get("live_counts")
    if live is not None:
        for v in live.get("hard_delta_violations", []) or []:
            failures.append(f"[HARD] live-collection delta: {v}")
        for v in live.get("chronicle_attribution_violations", []) or []:
            failures.append(f"[HARD] chronicle attribution: {v}")

    return (1 if failures else 0), failures
