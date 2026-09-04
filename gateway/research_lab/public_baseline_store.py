"""Simple persistence for the daily public-baseline rebenchmark."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from gateway.research_lab.store import (
    deterministic_uuid,
    insert_row,
    now_iso,
    select_one,
    update_row,
)


TABLE = "research_lab_daily_rebenchmarks"
TERMINAL_STATUSES = {"completed", "failed"}


def _result_ref(value: Mapping[str, Any]) -> str:
    return str(value.get("icp_ref") or "").strip()


def validate_progress_results(
    values: Sequence[Mapping[str, Any]], *, expected_icp_count: int
) -> list[dict[str, Any]]:
    if len(values) > int(expected_icp_count):
        raise ValueError("daily rebenchmark has more ICP results than expected")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        row = dict(value)
        icp_ref = _result_ref(row)
        if not icp_ref or icp_ref in seen:
            raise ValueError("daily rebenchmark ICP results must have unique refs")
        if str(row.get("status") or "") not in TERMINAL_STATUSES:
            raise ValueError("daily rebenchmark ICP result has invalid status")
        seen.add(icp_ref)
        results.append(row)
    return results


async def get_or_create_run(
    *,
    benchmark_date: str,
    baseline_id: str,
    baseline_repository: str,
    baseline_entrypoint: str,
    rolling_window_hash: str,
    window_doc: Mapping[str, Any],
    evaluation_epoch: int,
    expected_icp_count: int,
    worker_ref: str,
) -> dict[str, Any]:
    filters = (
        ("benchmark_date", benchmark_date),
        ("baseline_id", baseline_id),
        ("rolling_window_hash", rolling_window_hash),
    )
    existing = await select_one(TABLE, filters=filters)
    if existing is not None:
        return existing
    run_id = deterministic_uuid(
        "daily_public_baseline",
        benchmark_date,
        baseline_id,
        rolling_window_hash,
    )
    row = {
        "run_id": run_id,
        "benchmark_date": benchmark_date,
        "baseline_id": baseline_id,
        "baseline_repository": baseline_repository,
        "baseline_entrypoint": baseline_entrypoint,
        "rolling_window_hash": rolling_window_hash,
        "window_doc": dict(window_doc),
        "evaluation_epoch": int(evaluation_epoch),
        "status": "running",
        "expected_icp_count": int(expected_icp_count),
        "completed_icp_count": 0,
        "per_icp_results": [],
        "usage_doc": {},
        "score_summary_doc": {},
        "public_report_doc": {},
        "error_doc": {},
        "worker_ref": worker_ref,
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }
    try:
        return await insert_row(TABLE, row)
    except Exception:
        existing = await select_one(TABLE, filters=filters)
        if existing is not None:
            return existing
        raise


async def save_progress(
    *,
    run_id: str,
    expected_icp_count: int,
    per_icp_results: Sequence[Mapping[str, Any]],
    usage_doc: Mapping[str, Any],
    worker_ref: str,
) -> dict[str, Any]:
    results = validate_progress_results(
        per_icp_results,
        expected_icp_count=expected_icp_count,
    )
    return await update_row(
        TABLE,
        {
            "status": "running",
            "completed_icp_count": len(results),
            "per_icp_results": results,
            "usage_doc": dict(usage_doc),
            "error_doc": {},
            "worker_ref": worker_ref,
            "updated_at": now_iso(),
            "completed_at": None,
        },
        filters=(("run_id", run_id),),
    )


async def complete_run(
    *,
    run_id: str,
    expected_icp_count: int,
    per_icp_results: Sequence[Mapping[str, Any]],
    aggregate_score: float,
    usage_doc: Mapping[str, Any],
    score_summary_doc: Mapping[str, Any],
    public_report_doc: Mapping[str, Any],
    worker_ref: str,
) -> dict[str, Any]:
    results = validate_progress_results(
        per_icp_results,
        expected_icp_count=expected_icp_count,
    )
    if len(results) != int(expected_icp_count):
        raise ValueError("daily rebenchmark cannot complete with missing ICP results")
    if any(str(row.get("status") or "") != "completed" for row in results):
        raise ValueError("daily rebenchmark cannot complete with failed ICP results")
    return await update_row(
        TABLE,
        {
            "status": "completed",
            "completed_icp_count": len(results),
            "per_icp_results": results,
            "aggregate_score": float(aggregate_score),
            "usage_doc": dict(usage_doc),
            "score_summary_doc": dict(score_summary_doc),
            "public_report_doc": dict(public_report_doc),
            "error_doc": {},
            "worker_ref": worker_ref,
            "updated_at": now_iso(),
            "completed_at": now_iso(),
        },
        filters=(("run_id", run_id),),
    )


async def fail_run(
    *, run_id: str, error_code: str, error_message: str, worker_ref: str
) -> dict[str, Any]:
    return await update_row(
        TABLE,
        {
            "status": "failed",
            "error_doc": {
                "code": str(error_code or "daily_rebenchmark_failed")[:120],
                "message": str(error_message or "")[:500],
            },
            "worker_ref": worker_ref,
            "updated_at": now_iso(),
            "completed_at": now_iso(),
        },
        filters=(("run_id", run_id),),
    )


async def load_completed_run(
    *, benchmark_date: str, baseline_id: str
) -> dict[str, Any] | None:
    return await select_one(
        TABLE,
        filters=(
            ("benchmark_date", benchmark_date),
            ("baseline_id", baseline_id),
            ("status", "completed"),
        ),
    )

