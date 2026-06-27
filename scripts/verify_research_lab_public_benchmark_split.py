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
from gateway.research_lab.bundles import sha256_json
from gateway.research_lab.public_benchmarks import (
    build_benchmark_visibility_split,
    build_public_benchmark_report,
    sanitize_benchmark_item_summary,
)


def main() -> int:
    errors: list[str] = []
    deterministic_errors = _verify_controlled_split()
    skewed_errors = _verify_skewed_global_split()
    launch_config_errors = _verify_20_icp_total_split()
    issue_scope_errors = _verify_public_only_model_issues()
    fuzz_errors = _verify_fuzzed_splits(seed=71060, runs=200)
    secret_errors = _verify_secret_rejection()
    migration_errors = _verify_migration_policy()
    errors.extend(deterministic_errors)
    errors.extend(skewed_errors)
    errors.extend(launch_config_errors)
    errors.extend(issue_scope_errors)
    errors.extend(fuzz_errors)
    errors.extend(secret_errors)
    errors.extend(migration_errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Research Lab public benchmark split verified: global ranked public/private splits passed.")
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
    if report.get("schema_version") != "1.2":
        errors.append("public report schema must be 1.2")
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


def _verify_skewed_global_split() -> list[str]:
    items, summaries = _skewed_fixture()
    split = build_benchmark_visibility_split(
        rolling_window_hash="sha256:" + "7" * 64,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=3,
        public_weak_per_day=2,
    )
    errors = _assert_split(split)
    public_by_day: dict[int, int] = {}
    for item in split.get("items", []):
        if item.get("visibility") == "public":
            day = int(item.get("day_index") or 0)
            public_by_day[day] = public_by_day.get(day, 0) + 1
    if set(public_by_day.values()) == {3}:
        errors.append("skewed split must not force exactly 3 public ICPs from every day")
    return errors


def _verify_20_icp_total_split() -> list[str]:
    items, summaries = _fixture_20()
    split = build_benchmark_visibility_split(
        rolling_window_hash="sha256:" + "6" * 64,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=1,
        public_weak_per_day=1,
        public_total_icps=10,
        public_weak_total=7,
    )
    report = build_public_benchmark_report(
        benchmark_date="2026-06-25",
        rolling_window_hash="sha256:" + "6" * 64,
        aggregate_score=42.0,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=1,
        public_weak_per_day=1,
        public_total_icps=10,
        public_weak_total=7,
    )
    errors: list[str] = []
    split_items = split.get("items") if isinstance(split.get("items"), list) else []
    public = [item for item in split_items if item.get("visibility") == "public"]
    private = [item for item in split_items if item.get("visibility") == "private"]
    if len(split_items) != 20:
        errors.append(f"20-ICP launch split must contain 20 items, got {len(split_items)}")
    if len(public) != 10:
        errors.append(f"20-ICP launch split must expose 10 public ICPs, got {len(public)}")
    if len(private) != 10:
        errors.append(f"20-ICP launch split must reserve 10 private ICPs, got {len(private)}")
    if split.get("public_strength_counts") != {"strong": 3, "weak": 7}:
        errors.append(f"20-ICP launch public split must be 7 weak / 3 strong, got {split.get('public_strength_counts')}")
    if split.get("private_strength_counts") != {"strong": 7, "weak": 3}:
        errors.append(f"20-ICP launch private split must be 3 weak / 7 strong, got {split.get('private_strength_counts')}")
    ranked = sorted(
        split_items,
        key=lambda item: (
            float(item.get("score") or 0.0),
            _test_split_tiebreaker(split, item),
        ),
    )
    weak_pool = ranked[:10]
    strong_pool = ranked[10:]
    expected_public_refs = {
        str(item.get("icp_ref"))
        for item in [*weak_pool[:7], *list(reversed(strong_pool))[:3]]
    }
    actual_public_refs = {str(item.get("icp_ref")) for item in public}
    if actual_public_refs != expected_public_refs:
        errors.append("20-ICP launch split must pick the global 7 weakest and global 3 strongest ICPs")
    if report.get("public_icp_count") != 10 or report.get("private_holdout_icp_count") != 10:
        errors.append(
            "20-ICP launch report must expose 10 public ICPs and withhold 10 private ICPs, "
            f"got public={report.get('public_icp_count')} private={report.get('private_holdout_icp_count')}"
        )
    return errors


def _verify_public_only_model_issues() -> list[str]:
    items, summaries = _fixture_20()
    split = build_benchmark_visibility_split(
        rolling_window_hash="sha256:" + "5" * 64,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=1,
        public_weak_per_day=1,
        public_total_icps=10,
        public_weak_total=7,
    )
    public_refs = [
        str(item.get("icp_ref"))
        for item in split.get("items", [])
        if item.get("visibility") == "public"
    ]
    private_refs = [
        str(item.get("icp_ref"))
        for item in split.get("items", [])
        if item.get("visibility") == "private"
    ]
    if len(public_refs) < 3 or not private_refs:
        return ["public-only issue fixture did not produce enough public/private ICPs"]

    for summary in summaries:
        ref = str(summary.get("icp_ref"))
        diagnostics = dict(summary.get("diagnostics") or {})
        diagnostics["failure_categories"] = []
        diagnostics["avg_intent_signal_final"] = 40.0
        if ref == public_refs[0]:
            summary["company_count"] = 0
            diagnostics["failure_categories"] = ["runtime_provider_error", "provider_http_5xx"]
            diagnostics["avg_intent_signal_final"] = 0.0
        elif ref == public_refs[1]:
            summary["company_count"] = 2
            diagnostics["avg_intent_signal_final"] = 10.0
        elif ref == public_refs[2]:
            summary["company_count"] = 2
            diagnostics["avg_icp_fit"] = 0.0
        elif ref == private_refs[0]:
            summary["company_count"] = 3
            diagnostics["failure_categories"] = [
                "company_verification_failed",
                "hallucinated_or_generic_intent",
            ]
        summary["diagnostics"] = diagnostics

    report = build_public_benchmark_report(
        benchmark_date="2026-06-25",
        rolling_window_hash="sha256:" + "5" * 64,
        aggregate_score=42.0,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=1,
        public_weak_per_day=1,
        public_total_icps=10,
        public_weak_total=7,
    )
    errors: list[str] = []
    issue_counts = report.get("model_issue_counts", {})
    if issue_counts.get("provider_http_5xx") != 1:
        errors.append(f"public provider issue count wrong: {issue_counts}")
    if issue_counts.get("runtime_provider_error"):
        errors.append("runtime_provider_error must be suppressed when provider_http_5xx is present")
    if issue_counts.get("zero_company_results") != 1:
        errors.append(f"public zero-company count wrong: {issue_counts}")
    if issue_counts.get("low_intent_fit") != 1:
        errors.append(f"public low-intent count wrong: {issue_counts}")
    if issue_counts.get("icp_or_geo_mismatch") or report.get("low_icp_fit_count"):
        errors.append("ICP mismatch must not be derived from avg_icp_fit")
    for private_key in ("company_verification_failed", "hallucinated_or_generic_intent"):
        if issue_counts.get(private_key) or report.get("failure_category_counts", {}).get(private_key):
            errors.append(f"private issue leaked into public model issues: {private_key}")
    issue_icps = report.get("model_issue_public_icps", {})
    provider_rows = issue_icps.get("provider_http_5xx") if isinstance(issue_icps, dict) else None
    if not isinstance(provider_rows, list) or len(provider_rows) != 1:
        errors.append("provider issue must map to exactly one public ICP")
    elif provider_rows[0].get("icp_ref") != public_refs[0]:
        errors.append("provider issue mapped to the wrong public ICP")
    private_bucket_failures = [
        row.get("failure_categories")
        for row in report.get("icp_buckets", [])
        if row.get("visibility") == "private" and row.get("failure_categories")
    ]
    if private_bucket_failures:
        errors.append("private icp_buckets rows must not expose failure_categories")
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
    ranked = sorted(
        items,
        key=lambda item: (
            float(item.get("score") or 0.0),
            _test_split_tiebreaker(split, item),
        ),
    )
    weak_pool = ranked[: len(ranked) // 2]
    strong_pool = ranked[len(ranked) // 2 :]
    expected_public_weak_refs = {
        str(item.get("icp_ref"))
        for item in weak_pool[:20]
    }
    expected_public_strong_refs = {
        str(item.get("icp_ref"))
        for item in list(reversed(strong_pool))[:10]
    }
    actual_public_weak_refs = {
        str(item.get("icp_ref"))
        for item in public
        if item.get("strength_label") == "weak"
    }
    actual_public_strong_refs = {
        str(item.get("icp_ref"))
        for item in public
        if item.get("strength_label") == "strong"
    }
    if actual_public_weak_refs != expected_public_weak_refs:
        errors.append("public weak ICPs must be the global 20 lowest-scoring ICPs")
    if actual_public_strong_refs != expected_public_strong_refs:
        errors.append("public strong ICPs must be the global 10 highest-scoring ICPs")
    return errors


def _test_split_tiebreaker(split: dict[str, object], item: dict[str, object]) -> str:
    return sha256_json(
        {
            "rolling_window_hash": split.get("rolling_window_hash"),
            "icp_ref": item.get("icp_ref"),
            "icp_hash": item.get("icp_hash"),
            "score": round(float(item.get("score") or 0.0), 6),
        }
    )


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


def _fixture_20() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    items: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    scores = [3, 97, 17, 83, 29, 71, 41, 59, 7, 93, 13, 89, 23, 79, 37, 67, 47, 53, 31, 61]
    for day in range(1, 11):
        for rank in range(1, 3):
            index = (day - 1) * 2 + (rank - 1)
            score = float(scores[index])
            icp_id = f"launch_{day}_{rank}"
            item = {
                "icp_ref": f"qualification_private_icp_sets:{300 + day}:{icp_id}",
                "icp_hash": "sha256:" + f"{day * 100 + rank:064x}"[-64:],
                "set_id": 300 + day,
                "day_index": day,
                "day_rank": rank,
                "intent_signal_signature": f"launch signal {day}-{rank}",
                "icp": {
                    "icp_id": icp_id,
                    "industry": f"Industry {day}",
                    "sub_industry": f"Sub {rank}",
                    "country": "United States",
                    "employee_count": "51-200",
                    "product_service": "Launch benchmark",
                    "intent_signals": [f"launch signal {day}-{rank}"],
                },
            }
            items.append(item)
            summaries.append(
                sanitize_benchmark_item_summary(
                    item=item,
                    score=score,
                    company_count=max(1, int(score // 20)),
                    score_breakdowns=[
                        {
                            "final_score": score,
                            "icp_fit": min(score / 2, 50.0),
                            "intent_signal_final": min(score / 2, 50.0),
                            "failure_reason": None,
                        }
                    ],
                )
            )
    return items, summaries


def _skewed_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    items: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for day in range(1, 11):
        for rank in range(1, 7):
            score = float(day * 10 + rank)
            industry = f"Industry {day}"
            signal = f"signal {rank}"
            icp_id = f"skewed_{day}_{rank}"
            item = {
                "icp_ref": f"qualification_private_icp_sets:{200 + day}:{icp_id}",
                "icp_hash": "sha256:" + f"{day * 100 + rank:064x}"[-64:],
                "set_id": 200 + day,
                "day_index": day,
                "day_rank": rank,
                "intent_signal_signature": signal,
                "icp": {
                    "icp_id": icp_id,
                    "industry": industry,
                    "sub_industry": f"{industry} sub {rank}",
                    "country": "United States",
                    "employee_count": "51-200",
                    "product_service": f"{industry} platform",
                    "intent_signals": [signal],
                },
            }
            items.append(item)
            summaries.append(
                sanitize_benchmark_item_summary(
                    item=item,
                    score=score,
                    company_count=rank,
                    score_breakdowns=[
                        {
                            "final_score": score,
                            "icp_fit": min(score / 2, 50.0),
                            "intent_signal_final": min(score / 2, 50.0),
                            "failure_reason": None,
                        }
                    ],
                )
            )
    return items, summaries


if __name__ == "__main__":
    raise SystemExit(main())
