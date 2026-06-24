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
        "intent_signals",
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
    if report["zero_lead_icp_count"] != 1:
        errors.append("zero-lead ICP count did not match expected value")
    if report["failure_category_counts"].get("hallucinated_or_generic_intent") != 1:
        errors.append("hallucinated/generic intent failure was not counted")

    daily_counts = _daily_counts_from_score_bundle(_score_bundle())
    if daily_counts != {str(day): 5 for day in range(100, 110)}:
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
            "daily_icp_counts": {str(day): 5 for day in range(100, 110)},
        },
        {
            "champion_threshold_points": 1.0,
            "champion_min_alpha_percent": 2.0,
            "champion_extra_alpha_percent_per_point": 0.1,
            "champion_max_alpha_percent": 5.0,
            "champion_eval_days": 10,
            "champion_icps_per_day": 5,
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
    summaries = [
        sanitize_benchmark_item_summary(
            item={
                "icp_ref": "qualification_private_icp_sets:100:icp_a",
                "icp_hash": "sha256:" + "1" * 64,
                "icp": {
                    "industry": "Healthcare",
                    "sub_industry": "Revenue Cycle",
                    "country": "United States",
                    "employee_count": "51-200",
                    "intent_signals": ["hiring revenue operations leaders"],
                },
            },
            score=72.0,
            company_count=4,
            score_breakdowns=[
                {"final_score": 72.0, "icp_fit": 32.0, "intent_signal_final": 40.0, "failure_reason": None}
            ],
        ),
        sanitize_benchmark_item_summary(
            item={
                "icp_ref": "qualification_private_icp_sets:100:icp_b",
                "icp_hash": "sha256:" + "2" * 64,
                "icp": {
                    "industry": "Manufacturing",
                    "sub_industry": "Industrial automation",
                    "country": "United States",
                    "employee_count": "201-1000",
                    "intent_signals": ["expansion into new facilities"],
                },
            },
            score=0.0,
            company_count=0,
            score_breakdowns=[
                {
                    "final_score": 0.0,
                    "icp_fit": 0.0,
                    "intent_signal_final": 0.0,
                    "failure_reason": "Intent fabrication detected (generic claim)",
                }
            ],
        ),
    ]
    return build_public_benchmark_report(
        benchmark_date="2026-06-23",
        rolling_window_hash="sha256:" + "3" * 64,
        aggregate_score=36.0,
        per_icp_summaries=summaries,
    )


def _score_bundle() -> dict[str, object]:
    per_icp_results = []
    for set_id in range(100, 110):
        for idx in range(5):
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
