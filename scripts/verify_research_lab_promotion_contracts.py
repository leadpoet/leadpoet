#!/usr/bin/env python3
"""Verify Research Lab promotion and public benchmark contracts locally."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.public_benchmarks import build_public_benchmark_report, sanitize_benchmark_item_summary
from gateway.research_lab.promotion import _daily_counts_from_score_bundle
from leadpoet_verifier.economics import build_champion_reward_obligation


def main() -> int:
    errors: list[str] = []
    report = _public_report()
    encoded = json.dumps(report, sort_keys=True).lower()
    for forbidden in (
        "sk-or-",
        "https://",
        "http://",
        "image_digest",
        "candidate_patch_manifest",
        "private_model_manifest_doc",
        ".dkr.ecr",
        "judge_prompt",
    ):
        if forbidden in encoded:
            errors.append(f"public benchmark report leaked forbidden marker: {forbidden}")
    if "intent_signals" not in encoded:
        errors.append("public benchmark report must expose exact intent_signals for public ICPs")
    if report["schema_version"] != "1.2":
        errors.append(f"public benchmark report schema was not 1.2: {report['schema_version']}")
    if report["item_count"] != 6:
        errors.append(f"public benchmark report expected 6 total ICPs, got {report['item_count']}")
    if report["public_icp_count"] != 3:
        errors.append(f"public benchmark report expected 3 public ICPs, got {report['public_icp_count']}")
    if report["private_holdout_icp_count"] != 3:
        errors.append(
            f"public benchmark report expected 3 private holdout ICPs, got {report['private_holdout_icp_count']}"
        )
    split = report["visibility_split"]
    if split["public_strength_counts"] != {"strong": 1, "weak": 2}:
        errors.append(f"public split did not expose 2 weak / 1 strong ICPs: {split['public_strength_counts']}")
    if split["private_strength_counts"] != {"strong": 2, "weak": 1}:
        errors.append(f"private split did not reserve 1 weak / 2 strong ICPs: {split['private_strength_counts']}")
    if report["zero_lead_icp_count"] != 1:
        errors.append("zero-lead ICP count did not match expected value")
    if report["low_intent_fit_icp_count"] != 1:
        errors.append("public low-intent ICP count did not match expected value")
    if report["low_icp_fit_count"] != 0:
        errors.append("ICP mismatch count must not be derived from avg_icp_fit")
    if report["failure_category_counts"].get("hallucinated_or_generic_intent") != 1:
        errors.append("hallucinated/generic intent failure was not counted")
    issue_counts = report.get("model_issue_counts", {})
    if issue_counts.get("zero_company_results") != 1:
        errors.append("model_issue_counts did not include the public zero-company ICP")
    if issue_counts.get("low_intent_fit") != 1:
        errors.append("model_issue_counts did not include the public low-intent ICP")
    if issue_counts.get("hallucinated_or_generic_intent") != 1:
        errors.append("model_issue_counts did not include the public hallucinated/generic ICP")
    issue_icps = report.get("model_issue_public_icps", {})
    for issue_key in ("zero_company_results", "low_intent_fit", "hallucinated_or_generic_intent"):
        rows = issue_icps.get(issue_key) if isinstance(issue_icps, dict) else None
        if not isinstance(rows, list) or not rows:
            errors.append(f"model issue {issue_key} did not map to public ICP rows")

    daily_counts = _daily_counts_from_score_bundle(_score_bundle())
    if daily_counts != {str(day): 6 for day in range(100, 110)}:
        errors.append(f"daily ICP counts did not parse from score bundle refs: {daily_counts}")

    obligation = build_champion_reward_obligation(
        {
            "uid": 9,
            "miner_hotkey": "5Fminer111111111111111111111111111111111111",
            "island": "generalist",
            "candidate_id": "candidate:" + "a" * 64,
            "score_bundle_id": "score_bundle:" + "b" * 64,
            "run_id": "11111111-1111-4111-8111-111111111111",
            "evaluation_epoch": 1000,
            "start_epoch": 1001,
            "improvement_points": 1.25,
            "threshold_points": 1.0,
            "daily_icp_counts": {str(day): 6 for day in range(100, 110)},
        },
        {
            "champion_threshold_points": 1.0,
            "champion_min_alpha_percent": 2.0,
            "champion_extra_alpha_percent_per_point": 0.1,
            "champion_max_alpha_percent": 5.0,
            "champion_eval_days": 10,
            "champion_icps_per_day": 6,
            "reward_epochs": 20,
        },
    )
    if obligation["status"] != "active":
        errors.append(f"champion obligation was not active: {obligation}")
    if float(obligation["threshold_points"]) != 1.0:
        errors.append("champion obligation did not use the 1-point threshold")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab promotion contracts verified: sanitized public report, daily ICP count parsing, "
        "and 1-point champion obligation threshold."
    )
    return 0


def _public_report() -> dict[str, object]:
    benchmark_items = []
    summaries = []
    rows = [
        ("icp_a", "Healthcare", "Revenue Cycle", "hiring revenue operations leaders", 0.0, 0),
        ("icp_b", "Manufacturing", "Industrial Automation", "expansion into new facilities", 22.0, 3),
        ("icp_c", "Financial Services", "Risk", "new compliance audit program", 35.0, 2),
        ("icp_d", "Software", "Developer Tools", "migration to cloud data warehouse", 72.0, 4),
        ("icp_e", "Logistics", "Cold Chain", "opening new fulfillment centers", 83.0, 4),
        ("icp_f", "Cybersecurity", "IAM", "security platform implementation", 91.0, 5),
    ]
    for rank, (icp_id, industry, sub_industry, signal, score, company_count) in enumerate(rows, start=1):
        item = {
            "icp_ref": f"qualification_private_icp_sets:100:{icp_id}",
            "icp_hash": "sha256:" + f"{rank:064x}"[-64:],
            "set_id": 100,
            "day_index": 1,
            "day_rank": rank,
            "intent_signal_signature": signal,
            "icp": {
                "icp_id": icp_id,
                "prompt": f"Find companies for {industry}; see https://example.com/private",
                "industry": industry,
                "sub_industry": sub_industry,
                "country": "United States",
                "employee_count": "51-200",
                "product_service": f"{industry} platform",
                "intent_signals": [signal],
            },
        }
        benchmark_items.append(item)
        summaries.append(
            sanitize_benchmark_item_summary(
                item=item,
                score=score,
                company_count=company_count,
                score_breakdowns=[
                    {
                        "final_score": score,
                        "icp_fit": min(score / 2, 50.0),
                        "intent_signal_final": min(score / 2, 50.0),
                        "failure_reason": "Intent fabrication detected (generic claim)" if score == 0.0 else None,
                    }
                ],
            )
        )
    return build_public_benchmark_report(
        benchmark_date="2026-06-23",
        rolling_window_hash="sha256:" + "3" * 64,
        aggregate_score=36.0,
        per_icp_summaries=summaries,
        benchmark_items=benchmark_items,
        public_icps_per_day=3,
        public_weak_per_day=2,
    )


def _score_bundle() -> dict[str, object]:
    per_icp_results = []
    for set_id in range(100, 110):
        for idx in range(6):
            per_icp_results.append(
                {
                    "icp_ref": f"qualification_private_icp_sets:{set_id}:icp_{idx}",
                    "icp_hash": "sha256:" + f"{set_id:064x}"[-64:],
                    "base_company_scores": [50],
                    "candidate_company_scores": [55],
                }
            )
    return {"aggregates": {"per_icp_results": per_icp_results}}


if __name__ == "__main__":
    raise SystemExit(main())
