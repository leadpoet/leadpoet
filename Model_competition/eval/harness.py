"""Run qualify() across all golden ICPs and report metrics."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from qual_engine.entry import qualify
from qual_engine.models import ICPPrompt


def normalize_company(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def domain_match(a: str, b: str) -> bool:
    from qual_engine.validators.text_match import normalize_domain
    return normalize_domain(a) == normalize_domain(b)


async def run_one(icp_yaml: dict) -> dict:
    icp = ICPPrompt(
        icp_id=icp_yaml["icp_id"],
        prompt=icp_yaml.get("prompt", ""),
        industry=icp_yaml["industry"],
        sub_industry=icp_yaml.get("sub_industry", ""),
        geography=icp_yaml.get("geography", ""),
        country=icp_yaml.get("country", ""),
        employee_count=icp_yaml.get("employee_count", ""),
        company_stage=icp_yaml.get("company_stage", ""),
        product_service=icp_yaml.get("product_service", ""),
        intent_signals=icp_yaml.get("intent_signals", []),
    )

    result = await qualify(icp)

    expected = icp_yaml.get("expected_one_of") or []
    must_not = icp_yaml.get("must_not_return") or []
    expected_norm = {normalize_company(e["name"]) for e in expected}
    must_not_norm = {normalize_company(b["name"]) for b in must_not}

    returned_names = [m.company.company_name for m in result.matches]
    n_returned = len(returned_names)

    # Count correct matches (any returned company in expected list)
    correct_count = sum(
        1 for m in result.matches
        if normalize_company(m.company.company_name) in expected_norm
        or any(
            normalize_company(e["name"]) in normalize_company(m.company.company_name)
            or normalize_company(m.company.company_name) in normalize_company(e["name"])
            for e in expected
        )
    )

    # False positives: any returned company in must_not
    false_positives = sum(
        1 for m in result.matches
        if normalize_company(m.company.company_name) in must_not_norm
    )

    return {
        "icp_id": icp.icp_id,
        "n_returned": n_returned,
        "n_correct": correct_count,
        "n_false_positive": false_positives,
        "returned_companies": [
            {"name": m.company.company_name, "confidence": m.overall_confidence}
            for m in result.matches
        ],
        "abstention_reason": result.abstention_reason,
        "cost_usd": result.cost_breakdown.get("total_usd", 0),
        "latency_s": round(result.latency_ms / 1000, 1),
        "expected_one_of": [e["name"] for e in expected],
    }


async def main():
    here = Path(__file__).parent
    icps = yaml.safe_load((here / "golden_icps.yaml").read_text())
    print(f"Running {len(icps)} golden ICPs through qualify()...")
    print("=" * 80)

    results = []
    for icp_yaml in icps:
        print(f"\n[{icp_yaml['icp_id']}] {icp_yaml['prompt']}")
        r = await run_one(icp_yaml)
        results.append(r)
        if r["n_returned"] > 0:
            print(f"  → returned {r['n_returned']} companies "
                  f"({r['n_correct']} correct, {r['n_false_positive']} false-positive)  "
                  f"cost=${r['cost_usd']:.4f}  {r['latency_s']}s")
            for c in r["returned_companies"]:
                print(f"     • {c['name']:30s} confidence={c['confidence']}")
            print(f"     expected one of: {r['expected_one_of']}")
        else:
            print(f"  → ABSTAIN: {r['abstention_reason']}  cost=${r['cost_usd']:.4f}  {r['latency_s']}s")

    # Aggregate
    n = len(results)
    total_returned = sum(r["n_returned"] for r in results)
    total_correct  = sum(r["n_correct"]  for r in results)
    total_fp       = sum(r["n_false_positive"] for r in results)
    icps_w_answer  = sum(1 for r in results if r["n_returned"] > 0)
    total_cost     = sum(r["cost_usd"] for r in results)

    precision = (total_correct / total_returned * 100) if total_returned else 0

    print("\n" + "=" * 80)
    print(" SUMMARY (multi-answer mode)")
    print("=" * 80)
    print(f"  total ICPs:             {n}")
    print(f"  ICPs with ≥1 match:     {icps_w_answer}")
    print(f"  total companies returned: {total_returned}")
    print(f"  avg companies per ICP:    {total_returned/n:.1f}")
    print(f"  correct matches:        {total_correct}/{total_returned} ({precision:.0f}% precision)")
    print(f"  false positives:        {total_fp}")
    print(f"  total cost:             ${total_cost:.4f}")
    print(f"  avg cost per ICP:       ${total_cost/n:.4f}")

    # Save report
    report_path = here / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(exist_ok=True, parents=True)
    report_path.write_text(json.dumps({
        "summary": {
            "n": n,
            "icps_with_answer": icps_w_answer,
            "total_companies_returned": total_returned,
            "avg_companies_per_icp": round(total_returned / n, 2),
            "correct_matches": total_correct,
            "precision_pct": round(precision, 1),
            "false_positives": total_fp,
            "total_cost_usd": round(total_cost, 4),
        },
        "results": results,
    }, indent=2))
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
