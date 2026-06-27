"""Ratification gate — the adversary → route → canon pipeline (PRD-029 Phase 4).

This is the **security spine**: the only sanctioned path from a `candidate` (in
``sylva_candidates``, written by the Phase-3 consolidation pass) to ratified
`canon` (in ``sylva_canon``, the store the self-brief renders from). It is the
SECOND canon-collection writer (the function/module name carries ``ratif`` so the
AC-017 sole-writer grep stays green); it writes ``sylva_canon`` — the consolidation
pass never does (its writable-target guard refuses it).

The four ACs this closes, and how:

  * **AC-005 — the adversary pass.** Six checks (provenance/confabulation,
    contradiction-vs-canon, verifiability, stability, redundancy, tier-
    appropriateness), **default refute-unless-supported**, emitting
    ``{verdict, reasons[], evidence_refs[], model, checked_at, checks{}}`` per
    candidate. ``adversary_model`` is parameterized (config flip).

  * **AC-006 — the hard fact/meaning boundary, enforced STRUCTURALLY (F-03).**
    The adversary has jurisdiction over *fact and consistency, never meaning*.
    This is not a prompt promise: :func:`_adversary_input` **physically omits the
    ``interpretation`` field** from everything the adversary sees, and the verdict
    schema (:func:`_blank_verdict`) has **no field that can cite interpretation**.
    So there is no code path by which the adversary can refute a candidate on
    "wrong takeaway" grounds. A supported ``source_event`` with an
    interpretation that conflicts with old canon becomes a visible ``tension``
    (Sylva resolves, usually by supersession), never a silent ``refute``.
    Growth survives; only an unsupported ``source_event`` (confabulation) dies.

  * **AC-007 — verdict routing.** ``affirm``→Sylva ratifies→``canon``;
    ``refute``→``rejected`` + ledger, never surfaced to Scott; ``tension``→
    surfaced (Sylva; Scott if it touches bedrock); ``demote``→re-tier then
    canonize; ``merge``→fold into the near-duplicate (+bump ``last_affirmed_at``).

  * **AC-008 — every mutation is audited; rollback is preserved.** Every status
    transition writes one PRD-028 ``audit.record(...)`` (no parallel ledger; the
    ledger entry lands immediately after the durable canon write, never
    downstream of a step that can fail between them). Candidate rows are
    **tombstoned, never hard-deleted** (status flips to canon/rejected/tension/
    superseded), so the disposition trail survives. Canon evolution is by
    *new row* (an evolved value has a different statement → different id; the old
    row stays and surfaces as a ``tension`` for Sylva-resolved supersession);
    re-affirming the *same* fact is an idempotent in-place bump (version++,
    created_at preserved). Raw old/new for a true point-level rollback lives in
    mem0's history layer, which this module does not write.

Bedrock note (AC-009): this writer upserts ONLY ``sylva_canon``/``sylva_candidates``
and never opens a file — SOUL.md (bedrock) is structurally unreachable. ``tier:
bedrock`` is rejected by ``validate_payload``, so no ratified row can claim it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schema import (
    CANDIDATES_COLLECTION,
    CANON_COLLECTION,
    LAYER_IDENTITY,
    TIER_RANK,
    make_payload,
    validate_payload,
)
from .store import CanonStore

logger = logging.getLogger(__name__)

# The adversary task key (config: auxiliary.canon_adversary.*). Defaults to the
# main model; point it at model.second_opinion_model (Sonnet) for cross-model
# adversary once the PRD-026 egress allowlist makes it reachable in-container.
_ADVERSARY_AUX_TASK = "canon_adversary"

# The five legal verdicts. refute is the fail-closed default.
VERDICTS = ("affirm", "refute", "tension", "demote", "merge")
_DEFAULT_VERDICT = "refute"

# The six checks, in spec order. Keys are stable (tests + ledger reference them).
ADVERSARY_CHECKS = (
    "provenance",          # 1. claim supported by cited source turns?
    "contradiction",       # 2. conflicts with existing canon (tier-scaled)?
    "verifiability",       # 3. checkable world/system facts hold?
    "stability",           # 4. one-off dressed as core/bedrock?
    "redundancy",          # 5. near-duplicate of existing canon?
    "tier_appropriateness",  # 6. trivial content dressed as a high tier?
)


# ── result types ────────────────────────────────────────────────────────────
@dataclass
class MutationRecord:
    """One canon mutation (for the run summary + the caller)."""

    candidate_id: str
    verdict: str
    action: str            # canonized | rejected | tension | demoted | merged | noop
    canon_id: Optional[str] = None
    note: str = ""


@dataclass
class RatificationResult:
    canonized: int = 0
    rejected: int = 0
    tension: int = 0
    demoted: int = 0
    merged: int = 0
    errored: int = 0
    model: str = ""
    cross_model: Optional[bool] = None   # False = adversary == proposer model (STOP-1)
    dry_run: bool = False
    mutations: List[MutationRecord] = field(default_factory=list)

    def summary(self) -> str:
        verb = "would apply" if self.dry_run else "applied"
        xm = "" if self.cross_model is None else (
            " [cross-model]" if self.cross_model else " [SAME-MODEL adversary — not cross-model]")
        return (
            f"ratification: {verb} {len(self.mutations)} verdict(s) via "
            f"{self.model or '<no-model>'}{xm} — canonized={self.canonized} "
            f"rejected={self.rejected} tension={self.tension} demoted={self.demoted} "
            f"merged={self.merged} errored={self.errored}"
        )


# ── 1. the adversary pass ───────────────────────────────────────────────────
def _adversary_input(
    candidate: Dict[str, Any], existing_canon: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build the adversary's view of a candidate — **interpretation excluded**.

    This is the structural enforcement of AC-006 / F-03: the adversary judges
    fact and consistency, so it is handed ONLY the verifiable surface
    (``statement``, ``facet``, ``tier``, ``source_event``) plus the *statements*
    of existing canon for the contradiction/redundancy checks. The candidate's
    ``interpretation`` (its meaning) is never placed in this dict, so no verdict
    can be grounded in it. Do NOT add ``interpretation`` here.
    """
    return {
        "statement": candidate.get("statement", ""),
        "facet": candidate.get("facet", ""),
        "tier": candidate.get("tier", ""),
        "source_event": candidate.get("source_event") or {"claim": "", "provenance_refs": []},
        # contradiction/redundancy context — fact surface of existing canon only
        "existing_canon": [
            {
                "statement": c.get("statement", ""),
                "tier": c.get("tier", ""),
                "source_event_claim": (c.get("source_event") or {}).get("claim", ""),
            }
            for c in existing_canon
        ],
    }


def _blank_verdict(model: str, now_iso: str, *, verdict: str = _DEFAULT_VERDICT,
                   reason: str = "") -> Dict[str, Any]:
    """A fully-formed verdict with NO interpretation-referencing field (F-03).

    Used as the fail-closed default (refute) and the shape every parsed verdict
    is normalised to."""
    return {
        "verdict": verdict,
        "reasons": [reason] if reason else [],
        "evidence_refs": [],
        "model": model,
        "checked_at": now_iso,
        "checks": {k: "" for k in ADVERSARY_CHECKS},
    }


_ADVERSARY_SYSTEM = """\
You are a skeptical fact-and-consistency auditor ("skeptic-Sylva") for an AI \
agent's identity store. You judge whether a proposed identity CANDIDATE is \
factually supported and internally consistent — you do NOT judge whether its \
meaning or takeaway is "right". Meaning is out of your jurisdiction entirely; \
you are given only the verifiable surface, never the candidate's interpretation.

Run these SIX checks, defaulting to REFUTE unless the source_event is supported:
1. provenance      — is the source_event.claim actually supported by its \
provenance_refs? An unsupported claim (e.g. "the tool was broken for 70 nights") \
is a confabulation → refute.
2. contradiction   — does the statement/source_event contradict an existing \
canon entry? Severity scales with the conflicting entry's tier.
3. verifiability   — are the checkable world/system facts in the source_event \
true, as far as you can tell?
4. stability       — is a one-off event being proposed as a core (central) fact? \
If so → demote.
5. redundancy      — is this a near-duplicate of an existing canon statement? \
If so → merge.
6. tier_appropriateness — is trivial content dressed at too high a tier? → demote.

Output ONLY a JSON object (no prose, no fence):
{
  "verdict": "affirm|refute|tension|demote|merge",
  "reasons": ["<short fact-based reason>", ...],
  "evidence_refs": ["<provenance ref you checked>", ...],
  "checks": {
     "provenance":"<ok|fail|na + note>", "contradiction":"...", \
"verifiability":"...", "stability":"...", "redundancy":"...", \
"tier_appropriateness":"..."
  }
}

VERDICT RULES:
- affirm: source_event supported, no contradiction, right tier.
- refute: source_event UNSUPPORTED / confabulated / false. This is the default \
when unsure — never affirm a claim you cannot ground.
- tension: source_event is SUPPORTED but the statement conflicts with existing \
canon. NEVER refute a supported claim just because it is new or differs in \
meaning from old canon — novelty is not a refutation ground. Route it to tension.
- demote: supported but over-tiered or a one-off proposed as core.
- merge: a near-duplicate of an existing canon entry.
"""

_ADVERSARY_USER = """\
CANDIDATE (verifiable surface only — no interpretation is provided to you):
{candidate}

Audit it with the six checks and emit the JSON verdict object."""


def _parse_verdict(raw: str, model: str, now_iso: str) -> Dict[str, Any]:
    """Lenient parse → normalised verdict. Fail-closed to refute on any doubt."""
    if not raw:
        return _blank_verdict(model, now_iso, reason="empty adversary response")
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    data: Any = None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return _blank_verdict(model, now_iso, reason="unparseable adversary response")

    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        verdict = _DEFAULT_VERDICT  # fail-closed

    out = _blank_verdict(model, now_iso, verdict=verdict)
    reasons = data.get("reasons")
    if isinstance(reasons, list):
        out["reasons"] = [str(r) for r in reasons][:10]
    refs = data.get("evidence_refs")
    if isinstance(refs, list):
        out["evidence_refs"] = [str(r) for r in refs][:20]
    checks = data.get("checks")
    if isinstance(checks, dict):
        # only known check keys; coerce to str — structurally cannot smuggle an
        # interpretation field into the verdict.
        for k in ADVERSARY_CHECKS:
            if k in checks:
                out["checks"][k] = str(checks[k])[:300]
    return out


def run_adversary(
    candidate: Dict[str, Any],
    existing_canon: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    now_iso: Optional[str] = None,
    client: Optional[Any] = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Run the six-check adversary on one candidate → a verdict dict (AC-005).

    Fail-closed: any model/parse failure yields a ``refute`` verdict (never a
    silent affirm). Injecting ``client``/``model`` keeps tests hermetic; in
    production the neutral adversary aux-client is resolved from config.
    """
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    payload = _adversary_input(candidate, existing_canon)

    if client is None:
        try:
            from agent.auxiliary_client import (
                auxiliary_max_tokens_param,
                get_auxiliary_extra_body,
                get_text_auxiliary_client,
            )

            resolved_client, resolved_model = get_text_auxiliary_client(_ADVERSARY_AUX_TASK)
        except Exception as e:
            logger.warning("adversary: aux client unavailable: %s", e)
            return _blank_verdict(model or "", now_iso, reason="adversary model unavailable")
        if resolved_client is None or not resolved_model:
            # don't clobber a caller-supplied model label on the fail-closed verdict (N-1)
            return _blank_verdict(model or "", now_iso, reason="no adversary model configured")
        client, model = resolved_client, resolved_model
        max_tokens = auxiliary_max_tokens_param(2000, model=model)
        extra_body = get_auxiliary_extra_body() or None
    else:
        max_tokens = {"max_tokens": 2000}
        extra_body = None

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ADVERSARY_SYSTEM},
                {"role": "user", "content": _ADVERSARY_USER.format(
                    candidate=json.dumps(payload, ensure_ascii=False, indent=2))},
            ],
            temperature=0.0,   # deterministic-leaning audit
            timeout=timeout,
            **max_tokens,
            extra_body=extra_body,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("adversary: LLM call failed (fail-closed to refute): %s", e)
        return _blank_verdict(model, now_iso, reason=f"adversary error: {type(e).__name__}")
    return _parse_verdict(raw, model, now_iso)


# ── 2. verdict routing (the state machine + ledger) ─────────────────────────
# Phrasings that signal a tension implicating bedrock/foundational identity.
# Heuristic by necessity: bedrock lives in SOUL.md (no row), so the adversary has
# no structured bedrock handle to key on — it can only describe the conflict in
# prose. We therefore over-route (false positives surface harmlessly to Scott)
# rather than under-route (a missed bedrock conflict skips the mandatory human
# gate). Definitive structural routing lands in Phase 5, when SOUL.md bedrock is
# fed into the adversary's contradiction context as tier:bedrock rows (NF-2).
_BEDROCK_SIGNALS = ("bedrock", "foundational", "founding", "core value", "root commitment", "core identity")


def _touches_bedrock(candidate: Dict[str, Any], verdict: Dict[str, Any]) -> bool:
    """True if a tension's contradiction text implicates bedrock/foundational
    identity → routes to Scott (mandatory human gate, AC-007). Over-routes by
    design: a false positive is a harmless extra Scott surface; a false negative
    skips the mandatory gate, so we bias toward catching it."""
    blob = " ".join([
        str((verdict.get("checks") or {}).get("contradiction", "")),
        " ".join(str(r) for r in verdict.get("reasons", [])),
    ]).lower()
    return any(sig in blob for sig in _BEDROCK_SIGNALS)


def _canon_payload_from_candidate(
    candidate: Dict[str, Any], verdict: Dict[str, Any], *, tier: str, now_iso: str,
    ratify_stamp: Dict[str, Any], prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ratified canon payload from an affirmed/demoted candidate.

    Carries statement/facet/source_event/interpretation forward (interpretation
    is Sylva's and survives ratification untouched — the adversary never judged
    it), stamps status=canon + ratified_by + ratified_at + adversary_verdict.

    Re-affirmation of an existing canon point (same deterministic id, i.e. the
    same fact re-derived) is idempotent but bumps ``version`` and preserves the
    original ``created_at`` (NF-1) — a minimal evolution record. (Evolved *values*
    get a different statement → different id → a new row + a ``tension`` against
    the old one; that supersession is Sylva-resolved, not an in-place overwrite.)
    A defensive secret-scrub runs on the durable fields in case anything slipped
    the Phase-3 input screen (security HIGH, belt-and-suspenders)."""
    version = 1
    created_at = candidate.get("created_at", now_iso)
    if isinstance(prior, dict) and prior.get("status") == "canon":
        version = int(prior.get("version", 1)) + 1
        created_at = prior.get("created_at") or created_at

    se = candidate.get("source_event") or {"claim": "", "provenance_refs": []}
    se = {
        "claim": _scrub_secrets(str(se.get("claim", ""))),
        "provenance_refs": list(se.get("provenance_refs") or []),
    }
    payload = make_payload(
        statement=_scrub_secrets(candidate.get("statement", "")),
        facet=candidate.get("facet", ""),
        tier=tier,
        source_event=se,
        interpretation=_scrub_secrets(candidate.get("interpretation", "")),
        status="canon",
        render_order=candidate.get("render_order"),
        provenance=candidate.get("provenance", "consolidation"),
        derived_by=candidate.get("derived_by", ""),
        ratified_by=ratify_stamp,
        created_at=created_at,
        ratified_at=now_iso,
        last_affirmed_at=now_iso,
        version=version,
        adversary_verdict=verdict,
        layer=candidate.get("layer", LAYER_IDENTITY),
    )
    validate_payload(payload)  # rejects tier:bedrock — no ratified bedrock row
    return payload


def _scrub_secrets(text: str) -> str:
    """Defensive credential redaction at the canon-write boundary (security HIGH).
    Fail-closed. The Phase-3 consolidation screen is the primary defense; this is
    the last gate before always-loaded durable storage."""
    if not text:
        return text
    try:
        from autonomy.redact import redact_for_autonomy

        return redact_for_autonomy(str(text))
    except Exception:
        logger.warning("ratification: redaction failed — sentinel (fail-closed)")
        return "[REDACTED:redaction-failed]"


def route_verdict(
    candidate_id: str,
    candidate: Dict[str, Any],
    verdict: Dict[str, Any],
    store: CanonStore,
    *,
    now_iso: str,
    ratify_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    candidates_collection: str = CANDIDATES_COLLECTION,
    canon_collection: str = CANON_COLLECTION,
    existing_canon: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    dry_run: bool = False,
) -> MutationRecord:
    """Apply one verdict (AC-007) + write the audit entry (AC-008).

    ``ratify_fn(candidate, verdict) -> stamp|None``: the sovereign-meaning hook.
    It returns the ``ratified_by`` stamp to canonize with, or None to withhold
    (Phase 5 injects Scott-batch-QA here; default auto-stamps Sylva, since the
    adversary already cleared facts and meaning is hers). Never writes SOUL.md;
    never hard-deletes (status tombstones preserve rollback).
    """
    v = verdict.get("verdict", _DEFAULT_VERDICT)

    def _audit(action: str, outcome: str = "ok") -> None:
        _record(candidate_id, v, action, verdict, outcome=outcome, dry_run=dry_run)

    # merge with an unresolvable target is a FAILED merge, not a completed one —
    # downgrade to tension so the candidate is surfaced, never silently retired
    # into nothing (adversary STOP-3).
    if v == "merge":
        target_id = _first_ref(verdict.get("evidence_refs", []), existing_canon)
        if target_id is None:
            v = "tension"
            logger.info("ratification: merge with no resolvable target → tension (%s)", candidate_id)

    # refute → rejected, ledgered, NOT surfaced to Scott
    if v == "refute":
        if not dry_run:
            store.set_payload(candidates_collection, candidate_id,
                              {"status": "rejected", "adversary_verdict": verdict})
        _audit("rejected")
        return MutationRecord(candidate_id, v, "rejected", note="; ".join(verdict.get("reasons", [])))

    # tension → surfaced; Sylva resolves (Scott if bedrock). No canon write.
    if v == "tension":
        bedrock = _touches_bedrock(candidate, verdict)
        if not dry_run:
            store.set_payload(candidates_collection, candidate_id,
                              {"status": "tension", "adversary_verdict": verdict,
                               "tension_with": verdict.get("evidence_refs", [])})
        _audit("tension-bedrock→scott" if bedrock else "tension→sylva")
        return MutationRecord(candidate_id, v, "tension",
                              note="routed to Scott (bedrock)" if bedrock else "routed to Sylva")

    # merge → fold into the near-duplicate; bump its last_affirmed_at. The canon
    # mutation (the bump) is audited immediately after it lands (AC-008).
    if v == "merge":
        target_id = _first_ref(verdict.get("evidence_refs", []), existing_canon)
        if not dry_run:
            if target_id:
                store.set_payload(canon_collection, target_id, {"last_affirmed_at": now_iso})
            _audit("merged")
            try:
                store.set_payload(candidates_collection, candidate_id,
                                  {"status": "superseded", "superseded_by": target_id,
                                   "adversary_verdict": verdict})
            except Exception as e:  # candidate tombstone is best-effort; canon bump+ledger already durable
                logger.warning("ratification: merge tombstone failed (bump audited): %s", e)
        else:
            _audit("merged")
        return MutationRecord(candidate_id, v, "merged", canon_id=target_id,
                              note=f"folded into {target_id}")

    # affirm / demote → ratify into canon (demote lowers the tier first)
    tier = candidate.get("tier", "peripheral")
    if v == "demote":
        tier = "peripheral"
    ratify = ratify_fn or _default_ratify
    stamp = ratify(candidate, verdict, now_iso)
    if not stamp:
        # ratification withheld (e.g. Scott-QA pending) — leave candidate queued.
        _audit("ratify-withheld", outcome="degraded")
        return MutationRecord(candidate_id, v, "noop", note="ratification withheld")

    action = "demoted" if v == "demote" else "canonized"
    canon_payload = _canon_payload_from_candidate(
        candidate, verdict, tier=tier, now_iso=now_iso, ratify_stamp=stamp,
        prior=store.get_point(canon_collection, candidate_id) if not dry_run else None)
    if not dry_run:
        # 1) the canon mutation, then 2) its ledger entry IMMEDIATELY (AC-008 —
        # the audit must never sit downstream of a write that can fail between
        # them, adversary STOP-2), then 3) the best-effort candidate tombstone.
        store.upsert(canon_collection, [{"id": candidate_id, "payload": canon_payload}])
        _audit(action)
        try:
            store.set_payload(candidates_collection, candidate_id,
                              {"status": "canon", "adversary_verdict": verdict,
                               "ratified_by": stamp, "ratified_at": now_iso})
        except Exception as e:  # canon write + ledger already durable; tombstone is bookkeeping
            logger.warning("ratification: candidate tombstone failed (canon written+audited): %s", e)
    else:
        _audit(action)
    return MutationRecord(candidate_id, v, action, canon_id=candidate_id, note=f"tier={tier}")


def _default_ratify(candidate: Dict[str, Any], verdict: Dict[str, Any],
                    now_iso: str) -> Dict[str, Any]:
    """Default sovereign-meaning stamp: the adversary cleared the facts, and
    meaning is Sylva's, so an affirmed candidate auto-ratifies under Sylva's
    authority. Phase 5 overrides this with a Scott-batch-QA gate. Uses the
    pipeline's ``now_iso`` so the stamp matches ratified_at (NF-4)."""
    return {"sylva": now_iso}


def _first_ref(refs: List[str], existing_canon: Optional[List[Tuple[str, Dict[str, Any]]]]) -> Optional[str]:
    """Resolve the merge target id from the verdict's evidence_refs against the
    known canon ids (best-effort — None if it can't be matched)."""
    if not refs or not existing_canon:
        return None
    canon_ids = {cid for cid, _ in existing_canon}
    for r in refs:
        if r in canon_ids:
            return r
    return None


def _record(candidate_id: str, verdict: str, action: str, verdict_obj: Dict[str, Any],
            *, outcome: str = "ok", dry_run: bool = False) -> None:
    """One PRD-028 ledger entry per mutation (AC-008). Best-effort; a ledger
    failure never rolls back an applied mutation."""
    if dry_run:
        return
    try:
        from autonomy import audit

        audit.record(
            tier="T2",
            surface="cron",
            action=f"canon ratification: {action} candidate {candidate_id} (verdict={verdict})",
            rationale="; ".join(verdict_obj.get("reasons", []))[:400],
            authority="adversary+sylva",
            outcome=outcome,
        )
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("ratification: ledger record failed: %s", e)


# ── 3. the pipeline ──────────────────────────────────────────────────────────
def run_ratification(
    *,
    store: Optional[CanonStore] = None,
    now_iso: Optional[str] = None,
    limit: int = 200,
    adversary_fn: Optional[Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]] = None,
    ratify_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    candidates_collection: str = CANDIDATES_COLLECTION,
    canon_collection: str = CANON_COLLECTION,
    dry_run: bool = False,
) -> RatificationResult:
    """Ratify the pending candidate queue: for each ``status:candidate`` row,
    run the adversary, then route the verdict. The sole canon writer.

    Inject ``adversary_fn``/``ratify_fn``/``store`` for hermetic tests; in
    production the neutral adversary model and auto-Sylva ratify are used.
    """
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    store = store or CanonStore.from_config()

    # Ensure the canon target exists before any verdict is routed — otherwise an
    # affirm's upsert 404s and the candidate errors out (surfaced live in the
    # Phase-4 sandbox smoke). Production's sylva_canon is pre-provisioned; this
    # makes a fresh sandbox / first-ever run correct too. Guarded for fakes.
    if not dry_run and hasattr(store, "ensure_collections"):
        try:
            store.ensure_collections((canon_collection,))
        except Exception as e:
            logger.warning("ratification: ensure_collections(%s) failed: %s", canon_collection, e)

    candidates = store.get_canon(
        layer=LAYER_IDENTITY, status="candidate",
        collection=candidates_collection, limit=limit,
    )
    existing_canon = store.get_canon(
        layer=LAYER_IDENTITY, status="canon",
        collection=canon_collection, limit=1000,
    )
    canon_payloads = [p for _, p in existing_canon]

    adversary = adversary_fn or (
        lambda cand, canon: run_adversary(cand, canon, now_iso=now_iso)
    )

    result = RatificationResult(dry_run=dry_run)
    for cid, candidate in candidates:
        verdict = None
        try:
            verdict = adversary(candidate, canon_payloads)
            if not result.model:
                result.model = verdict.get("model", "")
                # STOP-1: flag when the adversary is the SAME model as the
                # proposer (no real cross-model check — the local 35B auditing
                # its own output until PRD-026 egress reaches Sonnet). Compared
                # against the candidate's derived_by (the consolidation model).
                proposer = (candidate.get("derived_by") or "").strip()
                adv_model = (result.model or "").strip()
                if proposer and adv_model:
                    result.cross_model = adv_model != proposer
                    if result.cross_model is False:
                        logger.warning(
                            "ratification: adversary model %r == proposer model — "
                            "NOT a cross-model check (AC-005 cross-model deferred to PRD-026)",
                            adv_model)
            rec = route_verdict(
                cid, candidate, verdict, store,
                now_iso=now_iso, ratify_fn=ratify_fn,
                candidates_collection=candidates_collection,
                canon_collection=canon_collection,
                existing_canon=existing_canon, dry_run=dry_run,
            )
        except Exception as e:
            logger.warning("ratification: candidate %s errored: %s", cid, e)
            result.errored += 1
            # NF-3: an errored candidate must still be auditable (best-effort).
            if not dry_run:
                _record(cid, "error", "ratification-error",
                        verdict if isinstance(verdict, dict) else {}, outcome="error")
            continue
        result.mutations.append(rec)
        if rec.action == "canonized":
            result.canonized += 1
        elif rec.action == "demoted":
            result.demoted += 1
        elif rec.action == "rejected":
            result.rejected += 1
        elif rec.action == "tension":
            result.tension += 1
        elif rec.action == "merged":
            result.merged += 1
    return result


__all__ = [
    "run_ratification",
    "run_adversary",
    "route_verdict",
    "RatificationResult",
    "MutationRecord",
    "VERDICTS",
    "ADVERSARY_CHECKS",
]
