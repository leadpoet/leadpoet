"""Pure construction of the existing daily baseline benchmark summary.

This module contains no database, provider, Docker, KMS, or network I/O. The
gateway host and the measured scoring executor call the same function so the
existing benchmark document can be compared byte-for-byte.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json


def _average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summary_has_unresolved_runtime_error(item_summary: Mapping[str, Any]) -> bool:
    diagnostics = item_summary.get("diagnostics")
    return isinstance(diagnostics, Mapping) and bool(diagnostics.get("runtime_error"))


def build_baseline_health(
    *,
    per_icp_summaries: Sequence[Mapping[str, Any]],
    retried: int,
    recovered: int,
    max_unresolved_icps: int,
    day_jump_points: float | None = None,
) -> dict[str, Any]:
    unresolved = sum(
        1 for summary in per_icp_summaries if _summary_has_unresolved_runtime_error(summary)
    )
    result = {
        "unresolved_provider_errors": unresolved,
        "gate_passed": unresolved <= int(max_unresolved_icps),
        "decision": "observe_only",
        "retried": int(retried),
        "recovered": int(recovered),
        "max_unresolved_icps": int(max_unresolved_icps),
    }
    if day_jump_points is not None:
        result["day_jump_points"] = round(float(day_jump_points), 4)
    return result


def artifact_serving_version_doc(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_artifact_hash": artifact.get("model_artifact_hash"),
        "manifest_hash": artifact.get("manifest_hash"),
        "manifest_uri": artifact.get("manifest_uri"),
        "git_commit_sha": artifact.get("git_commit_sha"),
        "image_ref_hash": sha256_json({"image_ref": artifact.get("image_digest")}),
        "config_hash": artifact.get("config_hash"),
        "component_registry_version": artifact.get("component_registry_version"),
        "scoring_adapter_version": artifact.get("scoring_adapter_version"),
        "build_id": artifact.get("build_id"),
    }


def baseline_serving_model_version_doc(
    *,
    artifact: Mapping[str, Any],
    benchmark_date: str,
    benchmark_attempt: int,
    rolling_window_hash: str,
    evaluation_epoch: int,
) -> dict[str, Any]:
    doc = {
        "schema_version": "research_lab_serving_model_version.v1",
        "result_role": "private_baseline_rebenchmark",
        "run_id": "private_baseline_rebenchmark:%s:attempt:%s"
        % (benchmark_date, benchmark_attempt),
        "ticket_id": "",
        "candidate_id": "",
        "private_model_version_id": "",
        "evaluation_epoch": int(evaluation_epoch),
        "benchmark_date": str(benchmark_date),
        "benchmark_attempt": int(benchmark_attempt),
        "run_scope": "private_baseline_rebenchmark",
        "benchmark_id": "rolling_icp_window:%s" % rolling_window_hash,
        "benchmark_split_ref": "research_lab_rolling_icp_window:%s" % rolling_window_hash,
        "icp_set_hash": str(rolling_window_hash),
        "scoring_code_version": "qualification-company-scorer:v1",
        "evaluator_version": "leadpoet-gateway-private-baseline:v1",
        "parent_model": artifact_serving_version_doc(artifact),
        "candidate_patch_hash": "",
        "candidate_source_diff_hash": "",
        "candidate_build_ref": "",
    }
    doc["version_stamp_hash"] = sha256_json(
        {key: value for key, value in doc.items() if key != "version_stamp_hash"}
    )
    return doc


def with_baseline_evaluation_contexts(
    per_icp_summaries: Sequence[Mapping[str, Any]],
    *,
    benchmark_date: str,
    benchmark_attempt: int,
    rolling_window_hash: str,
    evaluation_epoch: int,
    serving_model_version_hash: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, summary in enumerate(per_icp_summaries):
        item = dict(summary)
        existing = item.get("evaluation_context") if isinstance(item.get("evaluation_context"), Mapping) else {}
        result_payload = {key: value for key, value in item.items() if key != "evaluation_context"}
        item["evaluation_context"] = {
            **dict(existing),
            "schema_version": "research_lab_evaluation_context.v1",
            "result_index": int(index),
            "icp_ref": str(item.get("icp_ref") or ""),
            "icp_hash": str(item.get("icp_hash") or ""),
            "benchmark_id": "rolling_icp_window:%s" % rolling_window_hash,
            "benchmark_split_ref": "research_lab_rolling_icp_window:%s" % rolling_window_hash,
            "icp_set_hash": str(rolling_window_hash),
            "input_window_hash": str(rolling_window_hash),
            "run_id": "private_baseline_rebenchmark:%s:attempt:%s"
            % (benchmark_date, benchmark_attempt),
            "ticket_id": "",
            "candidate_id": "",
            "evaluation_epoch": int(evaluation_epoch),
            "benchmark_date": str(benchmark_date),
            "benchmark_attempt": int(benchmark_attempt),
            "run_scope": "private_baseline_rebenchmark",
            "provider_cache_day": str(benchmark_date),
            "serving_model_version_hash": str(serving_model_version_hash),
            "result_row_hash": sha256_json(result_payload),
        }
        enriched.append(item)
    return enriched


def daily_noise_budget_doc(
    *,
    benchmark_date: str,
    rolling_window_hash: str,
    per_icp_summaries: Sequence[Mapping[str, Any]],
    aggregate_score: float,
) -> dict[str, Any]:
    scores = [float(row.get("score") or 0.0) for row in per_icp_summaries]
    count = len(scores)
    mean = sum(scores) / count if count else 0.0
    variance = (
        sum((score - mean) ** 2 for score in scores) / (count - 1)
        if count > 1
        else 0.0
    )
    sd = variance ** 0.5
    se = sd / (count ** 0.5) if count else 0.0
    return {
        "schema_version": "research_lab_daily_noise_budget.v1",
        "benchmark_date": str(benchmark_date),
        "rolling_window_hash": str(rolling_window_hash),
        "icp_count": count,
        "aggregate_score": round(float(aggregate_score), 6),
        "mean_icp_score": round(mean, 6),
        "sample_sd": round(sd, 6),
        "standard_error": round(se, 6),
        "confidence_band_95": {
            "lower": round(mean - 1.96 * se, 6),
            "upper": round(mean + 1.96 * se, 6),
        },
        "zero_score_count": sum(1 for score in scores if score <= 0.0),
        "high_volatility": bool(count >= 5 and sd >= 25.0),
        "observability_only": True,
    }


def build_baseline_score_summary(
    *,
    artifact_manifest: Mapping[str, Any],
    benchmark_date: str,
    benchmark_attempt: int,
    rolling_window_hash: str,
    evaluation_epoch: int,
    benchmark_items: Sequence[Mapping[str, Any]],
    per_icp_summaries: Sequence[Mapping[str, Any]],
    public_icps_per_day: int,
    public_weak_per_day: int,
    public_total_icps: int,
    public_weak_total: int,
    retried: int,
    recovered: int,
    max_unresolved_icps: int,
    day_jump_points: float | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    from gateway.research_lab.public_benchmarks import build_benchmark_visibility_split

    aggregate_score = _average(
        [summary["score"] for summary in per_icp_summaries]
    )
    baseline_health = build_baseline_health(
        per_icp_summaries=per_icp_summaries,
        retried=retried,
        recovered=recovered,
        max_unresolved_icps=max_unresolved_icps,
        day_jump_points=day_jump_points,
    )
    visibility_split = build_benchmark_visibility_split(
        rolling_window_hash=rolling_window_hash,
        benchmark_items=benchmark_items,
        per_icp_summaries=per_icp_summaries,
        public_icps_per_day=public_icps_per_day,
        public_weak_per_day=public_weak_per_day,
        public_total_icps=public_total_icps,
        public_weak_total=public_weak_total,
    )
    serving_model_version = baseline_serving_model_version_doc(
        artifact=artifact_manifest,
        benchmark_date=benchmark_date,
        benchmark_attempt=benchmark_attempt,
        rolling_window_hash=rolling_window_hash,
        evaluation_epoch=evaluation_epoch,
    )
    enriched_summaries = with_baseline_evaluation_contexts(
        per_icp_summaries,
        benchmark_date=benchmark_date,
        benchmark_attempt=benchmark_attempt,
        rolling_window_hash=rolling_window_hash,
        evaluation_epoch=evaluation_epoch,
        serving_model_version_hash=str(serving_model_version["version_stamp_hash"]),
    )
    noise_budget = daily_noise_budget_doc(
        benchmark_date=benchmark_date,
        rolling_window_hash=rolling_window_hash,
        per_icp_summaries=enriched_summaries,
        aggregate_score=aggregate_score,
    )
    score_summary_doc = {
        "schema_version": "1.0",
        "benchmark_quality": "passed",
        "benchmark_attempt": int(benchmark_attempt),
        "rolling_window_hash": str(rolling_window_hash),
        "serving_model_version": serving_model_version,
        "per_icp_summaries": enriched_summaries,
        "visibility_split": visibility_split,
        "daily_noise_budget": noise_budget,
        "aggregate_score": aggregate_score,
        "baseline_health": baseline_health,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
    }
    return {
        "aggregate_score": aggregate_score,
        "baseline_health": baseline_health,
        "serving_model_version": serving_model_version,
        "per_icp_summaries": enriched_summaries,
        "visibility_split": visibility_split,
        "daily_noise_budget": noise_budget,
        "score_summary_doc": score_summary_doc,
    }
