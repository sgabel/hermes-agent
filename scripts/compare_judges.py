#!/usr/bin/env python3
"""PRD-029 Phase 5 experiment — compare adversary judges (Qwen vs Sonnet).

Runs the CURATED-mode adversary over an existing sandbox candidate set with TWO
judges in parallel and stores both verdicts side by side:
  * Qwen  — local aux client on :8081 (auxiliary.canon_adversary)
  * Sonnet — the `claude` CLI (Max OAuth, no metered cost) — same path as ask_claude

Both judges see the IDENTICAL curated prompt (the seed-calibrated adversary posture
in ratification._ADVERSARY_SYSTEM_CURATED) + the same recovered-journal evidence, so
the only variable is the judge model. Verdicts are written to the payload as
``adversary_verdict`` (Qwen) and ``adversary_verdict_sonnet`` — nothing is promoted.

Egress note: the Sonnet path sends the prompt off-host to Anthropic, so the
assembled prompt is secret-scrubbed (fail-closed) before it leaves.

Usage:
    python3 scripts/compare_judges.py --tag gemma            # judge sylva_lab_seed_candidates_gemma
    python3 scripts/compare_judges.py --tag gemma --limit 5  # quick sample
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.memory.canon import CanonStore  # noqa: E402
from plugins.memory.canon import ratification as R  # noqa: E402
from seed_canon import load_soul_bedrock_rows, recover_journal, _qdrant_url, _SANDBOX_CAND  # noqa: E402

_SONNET_MODEL = "claude-sonnet-4-6"
_SONNET_WORKERS = 4   # cap for Max quota / rate limits


def _scrub(text: str) -> str:
    try:
        from autonomy.redact import redact_for_autonomy
        return redact_for_autonomy(text)
    except Exception:
        return "[REDACTED:redaction-failed]"


def _sonnet_judge(view: dict, now_iso: str) -> dict:
    """Judge one candidate via the claude CLI (Sonnet). Returns a verdict dict."""
    combined = (R._ADVERSARY_SYSTEM_CURATED + "\n\n"
                + R._ADVERSARY_USER.format(candidate=json.dumps(view, ensure_ascii=False, indent=2)))
    combined = _scrub(combined)  # fail-closed before off-host egress
    try:
        from tools.environments.local import _make_run_env
        env = _make_run_env({})
    except Exception:
        env = None
    try:
        proc = subprocess.run(
            ["claude", "-p", "--model", _SONNET_MODEL, "--output-format", "json", combined],
            env=env, capture_output=True, text=True, timeout=180,
        )
    except Exception as e:
        return R._blank_verdict(_SONNET_MODEL, now_iso, reason=f"sonnet cli error: {type(e).__name__}")
    if proc.returncode != 0:
        return R._blank_verdict(_SONNET_MODEL, now_iso, reason=f"sonnet cli exit {proc.returncode}")
    raw = (proc.stdout or "").strip()
    try:
        env_json = json.loads(raw)
        result_text = env_json.get("result", raw) if isinstance(env_json, dict) else raw
    except Exception:
        result_text = raw
    return R._parse_verdict(result_text, _SONNET_MODEL, now_iso)


def _judge_one(store, coll, cid, payload, bedrock, qdrant, now_iso):
    evidence, _ = recover_journal(payload.get("legacy_date"), qdrant)
    view = R._adversary_input(payload, bedrock, evidence=evidence)
    # Qwen (local aux client), curated mode
    qwen = R.run_adversary(payload, bedrock, evidence=evidence, mode="curated", now_iso=now_iso)
    # Sonnet (claude CLI), identical curated prompt
    sonnet = _sonnet_judge(view, now_iso)
    store.set_payload(coll, cid, {
        "adversary_verdict": qwen,
        "adversary_verdict_sonnet": sonnet,
    })
    return cid, qwen.get("verdict"), sonnet.get("verdict"), payload.get("statement", "")[:70]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gemma")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    coll = f"{_SANDBOX_CAND}_{args.tag}" if args.tag else _SANDBOX_CAND
    store = CanonStore.from_config()
    qdrant = _qdrant_url()
    bedrock = load_soul_bedrock_rows()
    now_iso = datetime.now(timezone.utc).isoformat()
    cands = store.get_canon(status="candidate", collection=coll, limit=args.limit or 1000)
    print(f"judging {len(cands)} candidates in {coll} — Qwen (local) + Sonnet (claude CLI), "
          f"curated mode, bedrock-context={len(bedrock)}")

    rows = []
    with ThreadPoolExecutor(max_workers=_SONNET_WORKERS) as ex:
        futs = [ex.submit(_judge_one, store, coll, cid, p, bedrock, qdrant, now_iso)
                for cid, p in cands]
        for f in as_completed(futs):
            try:
                cid, qv, sv, stmt = f.result()
            except Exception as e:
                print("  judge error:", e); continue
            flag = "" if qv == sv else "  ← DISAGREE"
            rows.append((qv, sv, stmt))
            print(f"  qwen={qv:8s} sonnet={sv:8s} {stmt}{flag}")

    from collections import Counter
    qd = Counter(r[0] for r in rows)
    sd = Counter(r[1] for r in rows)
    agree = sum(1 for r in rows if r[0] == r[1])
    print("\n=== DISTRIBUTIONS (curated mode) ===")
    print("  Qwen  :", dict(qd))
    print("  Sonnet:", dict(sd))
    print(f"  agreement: {agree}/{len(rows)} ({100*agree//max(len(rows),1)}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
