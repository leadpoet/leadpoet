"""End-to-end test: fresh ICPs → qual_engine → production validator score.

For each of the 20 Sonar-generated ICPs:
  1. Run qual_engine.qualify() to produce CompanyMatch list
  2. Take the top match
  3. Serialize via to_submission_dict() → production CompanyOutput shape
  4. Call production score_company() and record final_score (0-100)
  5. Aggregate
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))                     # qual_engine
sys.path.insert(0, "/Users/tasnimul/Desktop/leadpoet")                    # gateway / qualification
sys.path.insert(0, "/Users/tasnimul/Desktop/leadpoet/qualification")      # nested scoring imports

# Load .env so qual_engine + production scorer both have keys
env_path = Path("/Users/tasnimul/Desktop/leadpoet/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
# Alias for the production scorer. The qual_engine accepts either
# OPENROUTER_API_KEY or OPENROUTER_KEY. The production scorer (lead_scorer →
# verification_helpers) reads QUALIFICATION_OPENROUTER_API_KEY specifically.
# Alias them all to the same value so both pipelines can authenticate.
_master_key = (
    os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENROUTER_KEY")
    or os.environ.get("QUALIFICATION_OPENROUTER_API_KEY")
)
if _master_key:
    os.environ["OPENROUTER_API_KEY"] = _master_key
    os.environ["OPENROUTER_KEY"] = _master_key
    os.environ["QUALIFICATION_OPENROUTER_API_KEY"] = _master_key

from qual_engine.entry import qualify
from qual_engine.layers.l8_output import to_submission_dict
from qual_engine.models import ICPPrompt as QualICPPrompt

# Production scorer + schemas
from gateway.qualification.models import (
    CompanyOutput as ProdCompanyOutput,
    ICPPrompt as ProdICPPrompt,
)
from qualification.scoring.lead_scorer import score_company


FRESH_PATH = Path(__file__).parent / "fresh_icps_20260525.json"


def _to_qual_icp(d: dict) -> QualICPPrompt:
    return QualICPPrompt(
        icp_id=d.get("icp_id", ""),
        prompt=d.get("prompt", "") or "",
        industry=d.get("industry", "") or "",
        sub_industry=d.get("sub_industry", "") or "",
        geography=d.get("geography", "") or "",
        country=d.get("country", "") or "",
        employee_count=d.get("employee_count", "") or "",
        company_stage=d.get("company_stage", "") or "",
        product_service=d.get("product_service", "") or "",
        intent_signals=d.get("intent_signals") or [],
    )


def _to_prod_icp(d: dict) -> ProdICPPrompt:
    return ProdICPPrompt(
        icp_id=d.get("icp_id", ""),
        prompt=d.get("prompt", "") or "",
        industry=d.get("industry", "") or "Software",
        sub_industry=d.get("sub_industry", "") or "",
        target_roles=[],
        target_seniority="",
        employee_count=d.get("employee_count", "") or "50-200",
        company_stage=d.get("company_stage", "") or "Series A",
        geography=d.get("geography", "") or "United States",
        country=d.get("country", "") or "United States",
        product_service=d.get("product_service", "") or "Software",
        intent_signals=d.get("intent_signals") or [],
    )


async def run_one(icp_dict: dict) -> dict:
    qual_icp = _to_qual_icp(icp_dict)
    prod_icp = _to_prod_icp(icp_dict)

    print(f"\n=== {qual_icp.icp_id}  {qual_icp.industry}/{qual_icp.sub_industry} ===")
    t0 = time.time()
    try:
        result = await qualify(qual_icp)
    except Exception as e:
        return {
            "icp_id": qual_icp.icp_id, "ok": False,
            "error": f"qualify crashed: {type(e).__name__}: {e}",
            "elapsed_s": round(time.time() - t0, 1),
        }

    elapsed = round(time.time() - t0, 1)
    cost = result.cost_breakdown.get("total_usd", 0)
    if not result.matches:
        print(f"   qualify: 0 matches (abstain: {result.abstention_reason})")
        return {
            "icp_id": qual_icp.icp_id,
            "industry": qual_icp.industry,
            "ok": True,
            "qualify_elapsed_s": elapsed,
            "qualify_cost_usd": cost,
            "match_count": 0,
            "abstention": result.abstention_reason,
            "prod_final_score": 0,
        }

    # Top match → serialize → validate as production schema
    top = result.matches[0]
    submission_dict = to_submission_dict(top.company)
    try:
        prod_co = ProdCompanyOutput.model_validate(submission_dict)
    except Exception as e:
        return {
            "icp_id": qual_icp.icp_id, "ok": False,
            "error": f"prod schema validation failed: {e}",
            "qualify_elapsed_s": elapsed,
            "qualify_cost_usd": cost,
        }

    # Production scorer
    print(f"   qualify: {result.total_matches} matches, top={top.company.company_name} (conf={top.overall_confidence})")
    print(f"   running production score_company()...")
    t1 = time.time()
    try:
        score_breakdown = await score_company(
            company=prod_co,
            icp=prod_icp,
            run_cost_usd=cost,
            run_time_seconds=elapsed,
            seen_companies=set(),
            force_fail_reason=None,
        )
    except Exception as e:
        return {
            "icp_id": qual_icp.icp_id, "ok": False,
            "error": f"score_company crashed: {type(e).__name__}: {str(e)[:200]}",
            "qualify_elapsed_s": elapsed,
            "qualify_cost_usd": cost,
            "match_name": top.company.company_name,
        }

    score_elapsed = round(time.time() - t1, 1)
    return {
        "icp_id": qual_icp.icp_id,
        "industry": qual_icp.industry,
        "geography": qual_icp.geography,
        "ok": True,
        "qualify_elapsed_s": elapsed,
        "qualify_cost_usd": cost,
        "score_elapsed_s": score_elapsed,
        "match_count": result.total_matches,
        "match_name": top.company.company_name,
        "match_internal_conf": top.overall_confidence,
        # Production score breakdown
        "prod_icp_fit": score_breakdown.icp_fit,
        "prod_intent_signal_final": score_breakdown.intent_signal_final,
        "prod_intent_signal_raw": score_breakdown.intent_signal_raw,
        "prod_time_decay": score_breakdown.time_decay_multiplier,
        "prod_cost_penalty": score_breakdown.cost_penalty,
        "prod_time_penalty": score_breakdown.time_penalty,
        "prod_final_score": score_breakdown.final_score,
        "prod_failure_reason": score_breakdown.failure_reason,
    }


async def main():
    icps = json.loads(FRESH_PATH.read_text())
    print(f"Loaded {len(icps)} fresh ICPs")
    print(f"Running qual_engine + production score_company on each (sequentially to avoid rate limits)...")

    log_path = Path(__file__).parent / f"prod_score_results_{int(time.time())}.jsonl"
    results = []
    t_start = time.time()
    for icp_dict in icps:
        r = await run_one(icp_dict)
        results.append(r)
        with open(log_path, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"\n{'='*100}")
    print(f"PRODUCTION-VALIDATOR SCORE — wall {time.time()-t_start:.1f}s")
    print(f"{'='*100}")
    print(f"{'ICP':22} {'INDUSTRY':35} {'#M':>3} {'CAND':18} {'icp_fit':>8} {'intent':>8} {'FINAL':>6}")
    print("-" * 110)
    total_final = 0
    total_runs = 0
    success_runs = 0
    for r in results:
        if not r.get("ok"):
            print(f"{r['icp_id']:22} ERROR {r.get('error','')[:60]}")
            continue
        ind = (r.get("industry") or "")[:35]
        if r.get("match_count", 0) == 0:
            print(f"{r['icp_id']:22} {ind:35} {0:>3} {'(no match)':18} {'-':>8} {'-':>8} {0:>6}")
            total_runs += 1
            continue
        name = (r.get("match_name") or "")[:18]
        icp_fit = r.get("prod_icp_fit", 0)
        intent = r.get("prod_intent_signal_final", 0)
        final = r.get("prod_final_score", 0)
        print(f"{r['icp_id']:22} {ind:35} {r['match_count']:>3} {name:18} {icp_fit:>8.1f} {intent:>8.1f} {final:>6.1f}")
        total_final += final
        total_runs += 1
        if final > 0:
            success_runs += 1
    print("-" * 110)
    print()
    if total_runs:
        avg_final = total_final / total_runs
        print(f"  ICPs run        : {total_runs}")
        print(f"  Non-zero score  : {success_runs}/{total_runs} ({success_runs/total_runs*100:.0f}%)")
        print(f"  Average score   : {avg_final:.1f} / 100")
        print(f"  Sum of scores   : {total_final:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
