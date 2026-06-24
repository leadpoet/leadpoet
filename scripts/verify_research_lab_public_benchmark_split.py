#!/usr/bin/env python3
"""Verify Research Lab public/private benchmark disclosure split."""

from __future__ import annotations

from pathlib import Path
import json
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.bundles import contains_secret_material
from gateway.research_lab.public_benchmarks import (
    build_benchmark_visibility_split,
    build_public_benchmark_report,
    sanitize_benchmark_item_summary,
)


def main() -> int:
    errors: list[str] = []
    deterministic_errors = _verify_controlled_split()
    fuzz_errors = _verify_fuzzed_splits(seed=71060, runs=200)
    secret_errors = _verify_secret_rejection()
    migration_errors = _verify_migration_policy()
    errors.extend(deterministic_errors)
    errors.extend(fuzz_errors)
    errors.extend(secret_errors)
    errors.extend(migration_errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Research Lab public benchmark split verified: controlled 30/30 split and 200 fuzzed splits passed.")
    return 0


def _verify_controlled_split() -> list[str]:
    items, summaries = _fixture(seed=71)
    split = build_benchmark_visibility_split(
        rolling_window_hash="sha256:" + "9" * 64,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=3,
        public_weak_per_day=2,
    )
    report = build_public_benchmark_report(
        benchmark_date="2026-06-24",
        rolling_window_hash="sha256:" + "9" * 64,
        aggregate_score=51.0,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=3,
        public_weak_per_day=2,
    )
    errors = _assert_split(split)
    encoded = json.dumps(report, sort_keys=True).lower()
    if report.get("schema_version") != "1.1":
        errors.append("public report schema must be 1.1")
    if report.get("public_icp_count") != 30:
        errors.append(f"public report must expose 30 ICPs, got {report.get('public_icp_count')}")
    if report.get("private_holdout_icp_count") != 30:
        errors.append(f"public report must withhold 30 ICPs, got {report.get('private_holdout_icp_count')}")
    if len(report.get("public_icps", [])) != 30:
        errors.append("public_icps array must contain exactly 30 entries")
    if "intent_signals" not in encoded:
        errors.append("public ICPs must include exact intent_signals")
    for forbidden in (
        "https://",
        "http://",
        "sk-or-",
        "service_role",
        "candidate_patch_manifest",
        "image_digest",
        "hidden_icp",
        "icp_plaintext",
        "private_repo",
        "proxy_url",
        "judge_prompt",
    ):
        if forbidden in encoded:
            errors.append(f"public report leaked forbidden marker: {forbidden}")
    if contains_secret_material(report):
        errors.append("public report failed secret material guard")
    return errors


def _verify_secret_rejection() -> list[str]:
    items, summaries = _fixture(seed=99)
    for index, item in enumerate(items):
        updated = dict(item)
        public_icp = dict(updated["icp"])  # type: ignore[index]
        public_icp["prompt"] = "this must fail sk-or-secret"
        updated["icp"] = public_icp
        items[index] = updated
    try:
        build_public_benchmark_report(
            benchmark_date="2026-06-24",
            rolling_window_hash="sha256:" + "8" * 64,
            aggregate_score=51.0,
            benchmark_items=items,
            per_icp_summaries=summaries,
            public_icps_per_day=3,
            public_weak_per_day=2,
        )
    except ValueError:
        return []
    return ["public benchmark report must reject forbidden secret marker values"]


def _verify_migration_policy() -> list[str]:
    sql_path = ROOT / "scripts" / "39-research-lab-public-benchmark-icp-disclosure.sql"
    sql = sql_path.read_text(encoding="utf-8").lower()
    errors: list[str] = []
    if "grant " in sql:
        errors.append("migration 39 must not create new grants")
    if "intent_signals" not in sql:
        errors.append("migration 39 must document/handle public intent_signals")
    for marker in (
        "sk-or-",
        "service_role",
        "private_repo",
        "judge_prompt",
        "hidden_icp",
        "icp_plaintext",
        "image_digest",
        "candidate_patch_manifest",
        "proxy[_-]?url",
        "https?://",
    ):
        if marker not in sql:
            errors.append(f"migration 39 missing safety marker: {marker}")
    return errors


def _verify_fuzzed_splits(*, seed: int, runs: int) -> list[str]:
    errors: list[str] = []
    rng = random.Random(seed)
    for run in range(runs):
        items, summaries = _fixture(seed=rng.randint(1, 10_000_000), tie_scores=(run % 7 == 0))
        first = build_benchmark_visibility_split(
            rolling_window_hash="sha256:" + f"{run + 1:064x}"[-64:],
            benchmark_items=items,
            per_icp_summaries=summaries,
            public_icps_per_day=3,
            public_weak_per_day=2,
        )
        second = build_benchmark_visibility_split(
            rolling_window_hash="sha256:" + f"{run + 1:064x}"[-64:],
            benchmark_items=list(reversed(list(reversed(items)))),
            per_icp_summaries=list(reversed(list(reversed(summaries)))),
            public_icps_per_day=3,
            public_weak_per_day=2,
        )
        if first != second:
            errors.append(f"fuzz run {run}: split must be deterministic")
            break
        split_errors = _assert_split(first)
        if split_errors:
            errors.extend(f"fuzz run {run}: {error}" for error in split_errors)
            break
    return errors


def _assert_split(split: dict[str, object]) -> list[str]:
    errors: list[str] = []
    items = split.get("items") if isinstance(split.get("items"), list) else []
    public = [item for item in items if item.get("visibility") == "public"]
    private = [item for item in items if item.get("visibility") == "private"]
    public_refs = {str(item.get("icp_ref")) for item in public}
    private_refs = {str(item.get("icp_ref")) for item in private}
    if len(items) != 60:
        errors.append(f"split must contain 60 items, got {len(items)}")
    if len(public) != 30:
        errors.append(f"split must expose 30 public ICPs, got {len(public)}")
    if len(private) != 30:
        errors.append(f"split must reserve 30 private ICPs, got {len(private)}")
    if public_refs & private_refs:
        errors.append("public/private ICP refs must be disjoint")
    if len(public_refs | private_refs) != 60:
        errors.append("all split ICP refs must be unique")
    if split.get("public_strength_counts") != {"strong": 10, "weak": 20}:
        errors.append(f"public split must be 20 weak / 10 strong, got {split.get('public_strength_counts')}")
    if split.get("private_strength_counts") != {"strong": 20, "weak": 10}:
        errors.append(f"private split must be 10 weak / 20 strong, got {split.get('private_strength_counts')}")
    per_day: dict[int, dict[str, int]] = {}
    for item in items:
        day = int(item.get("day_index") or 0)
        visibility = str(item.get("visibility") or "")
        per_day.setdefault(day, {"public": 0, "private": 0})
        per_day[day][visibility] += 1
    for day, counts in per_day.items():
        if counts != {"public": 3, "private": 3}:
            errors.append(f"day {day} must split 3 public / 3 private, got {counts}")
    return errors


def _fixture(*, seed: int, tie_scores: bool = False) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    industries = [
        "Healthcare",
        "Manufacturing",
        "Financial Services",
        "Software",
        "Logistics",
        "Cybersecurity",
        "Energy",
        "Construction",
    ]
    signals = [
        "hiring operations leaders",
        "new compliance audit",
        "warehouse expansion",
        "cloud migration",
        "security platform rollout",
        "funding announced",
        "new facility opening",
        "partnership implementation",
    ]
    items: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for day in range(1, 11):
        scores = [10, 20, 30, 70, 80, 90] if not tie_scores else [50, 50, 50, 50, 50, 50]
        rng.shuffle(scores)
        for rank in range(1, 7):
            industry = industries[(day + rank + rng.randint(0, 3)) % len(industries)]
            signal = signals[(day * rank + rng.randint(0, 5)) % len(signals)]
            icp_id = f"icp_{day}_{rank}"
            score = float(scores[rank - 1])
            item = {
                "icp_ref": f"qualification_private_icp_sets:{100 + day}:{icp_id}",
                "icp_hash": "sha256:" + f"{day * 100 + rank:064x}"[-64:],
                "set_id": 100 + day,
                "day_index": day,
                "day_rank": rank,
                "intent_signal_signature": signal,
                "icp": {
                    "icp_id": icp_id,
                    "industry": industry,
                    "sub_industry": f"{industry} sub {rank}",
                    "country": "United States",
                    "employee_count": "51-200" if rank % 2 else "201-1000",
                    "product_service": f"{industry} platform",
                    "intent_signals": [signal],
                    "prompt": f"Find {industry} companies with signal {signal}; source https://example.com/private",
                },
            }
            items.append(item)
            summaries.append(
                sanitize_benchmark_item_summary(
                    item=item,
                    score=score,
                    company_count=max(0, int(score // 20)),
                    score_breakdowns=[
                        {
                            "final_score": score,
                            "icp_fit": min(score / 2, 50.0),
                            "intent_signal_final": min(score / 2, 50.0),
                            "failure_reason": None if score > 0 else "Intent fabrication detected",
                        }
                    ],
                )
            )
    return items, summaries


if __name__ == "__main__":
    raise SystemExit(main())
