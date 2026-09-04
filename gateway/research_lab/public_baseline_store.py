"""Simple persistence for the daily public-baseline rebenchmark."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from gateway.db.client import get_write_client


TABLE = "research_lab_daily_rebenchmarks"
TERMINAL_STATUSES = {"completed", "failed"}
LEASE_SECONDS = 45 * 60


class DailyRebenchmarkBusy(RuntimeError):
    """Another worker owns the current daily run lease."""


def _lease_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS)).isoformat()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_filters(query: Any, filters: Sequence[tuple[Any, ...]]) -> Any:
    for value in filters:
        if len(value) == 2:
            query = query.eq(value[0], value[1])
        elif len(value) == 3 and value[1] == "neq":
            query = query.neq(value[0], value[2])
        else:
            raise ValueError("unsupported public-baseline store filter")
    return query


async def select_one(
    table: str, *, filters: Sequence[tuple[Any, ...]]
) -> dict[str, Any] | None:
    def _call() -> Any:
        query = get_write_client().table(table).select("*")
        return _apply_filters(query, filters).limit(1).execute()

    response = await asyncio.to_thread(_call)
    rows = getattr(response, "data", None) or []
    return dict(rows[0]) if rows else None


async def select_many(
    table: str,
    *,
    filters: Sequence[tuple[Any, ...]],
    order_by: Sequence[tuple[str, bool]],
    limit: int,
) -> list[dict[str, Any]]:
    def _call() -> Any:
        query = _apply_filters(get_write_client().table(table).select("*"), filters)
        for field, descending in order_by:
            query = query.order(field, desc=descending)
        return query.limit(int(limit)).execute()

    response = await asyncio.to_thread(_call)
    return [dict(row) for row in (getattr(response, "data", None) or [])]


async def insert_row(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
    def _call() -> Any:
        return get_write_client().table(table).insert(dict(row)).execute()

    response = await asyncio.to_thread(_call)
    rows = getattr(response, "data", None) or []
    if not rows:
        raise RuntimeError(f"{table}: insert returned no row")
    return dict(rows[0])


async def update_row(
    table: str,
    values: Mapping[str, Any],
    *,
    filters: Sequence[tuple[Any, ...]],
) -> dict[str, Any]:
    def _call() -> Any:
        query = get_write_client().table(table).update(dict(values))
        return _apply_filters(query, filters).execute()

    response = await asyncio.to_thread(_call)
    rows = getattr(response, "data", None) or []
    if not rows:
        raise RuntimeError(f"{table}: update returned no row")
    return dict(rows[0])


async def call_rpc(name: str, params: Mapping[str, Any]) -> Any:
    def _call() -> Any:
        return get_write_client().rpc(name, dict(params)).execute()

    response = await asyncio.to_thread(_call)
    return getattr(response, "data", None)


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


async def _update_unless_completed(
    run_id: str,
    values: Mapping[str, Any],
    *,
    claim_token: str,
) -> dict[str, Any]:
    """Apply one update without allowing a stale worker to reopen a result."""

    expected = dict(values)
    try:
        return await update_row(
            TABLE,
            expected,
            filters=(
                ("run_id", run_id),
                ("status", "neq", "completed"),
                ("claim_token", claim_token),
            ),
        )
    except Exception:
        current = await select_one(TABLE, filters=(("run_id", run_id),))
        if current is not None and str(current.get("status") or "") == "completed":
            return current
        if current is not None and all(
            current.get(key) == value for key, value in expected.items()
        ):
            return current
        raise


async def get_or_create_run(
    *,
    benchmark_date: str,
    baseline_id: str,
    baseline_repository: str,
    baseline_entrypoint: str,
    window_doc: Mapping[str, Any],
    benchmark_input_doc: Mapping[str, Any],
    evaluation_epoch: int,
    expected_icp_count: int,
    worker_ref: str,
    claim_token: str,
) -> dict[str, Any]:
    filters = (
        ("benchmark_date", benchmark_date),
        ("baseline_id", baseline_id),
    )
    existing = await select_one(TABLE, filters=filters)
    if existing is not None:
        if str(existing.get("status") or "") != "running":
            return existing
        claimed = await call_rpc(
            "research_lab_claim_daily_rebenchmark",
            {
                "p_run_id": existing["run_id"],
                "p_claim_token": claim_token,
                "p_worker_ref": worker_ref,
                "p_lease_seconds": LEASE_SECONDS,
            },
        )
        if not isinstance(claimed, Mapping) or claimed.get("claim_status") != "claimed":
            raise DailyRebenchmarkBusy("daily public rebenchmark is already running")
        row = claimed.get("run")
        if not isinstance(row, Mapping):
            raise RuntimeError("daily public rebenchmark claim returned no row")
        return dict(row)
    run_id = str(uuid4())
    row = {
        "run_id": run_id,
        "benchmark_date": benchmark_date,
        "baseline_id": baseline_id,
        "baseline_repository": baseline_repository,
        "baseline_entrypoint": baseline_entrypoint,
        "window_doc": dict(window_doc),
        "benchmark_input_doc": dict(benchmark_input_doc),
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
        "claim_token": claim_token,
        "lease_expires_at": _lease_expiry(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }
    try:
        return await insert_row(TABLE, row)
    except Exception:
        existing = await select_one(TABLE, filters=filters)
        if existing is not None and str(existing.get("status") or "") != "running":
            return existing
        if existing is not None:
            claimed = await call_rpc(
                "research_lab_claim_daily_rebenchmark",
                {
                    "p_run_id": existing["run_id"],
                    "p_claim_token": claim_token,
                    "p_worker_ref": worker_ref,
                    "p_lease_seconds": LEASE_SECONDS,
                },
            )
            if isinstance(claimed, Mapping) and claimed.get("claim_status") == "claimed":
                claimed_row = claimed.get("run")
                if isinstance(claimed_row, Mapping):
                    return dict(claimed_row)
            raise DailyRebenchmarkBusy("daily public rebenchmark is already running")
        raise


async def save_progress(
    *,
    run_id: str,
    expected_icp_count: int,
    per_icp_results: Sequence[Mapping[str, Any]],
    usage_doc: Mapping[str, Any],
    worker_ref: str,
    claim_token: str,
) -> dict[str, Any]:
    results = validate_progress_results(
        per_icp_results,
        expected_icp_count=expected_icp_count,
    )
    return await _update_unless_completed(
        run_id,
        {
            "status": "running",
            "completed_icp_count": sum(
                str(row.get("status") or "") == "completed" for row in results
            ),
            "per_icp_results": results,
            "usage_doc": dict(usage_doc),
            "error_doc": {},
            "worker_ref": worker_ref,
            "lease_expires_at": _lease_expiry(),
            "updated_at": now_iso(),
            "completed_at": None,
        },
        claim_token=claim_token,
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
    claim_token: str,
) -> dict[str, Any]:
    results = validate_progress_results(
        per_icp_results,
        expected_icp_count=expected_icp_count,
    )
    if len(results) != int(expected_icp_count):
        raise ValueError("daily rebenchmark cannot complete with missing ICP results")
    if any(str(row.get("status") or "") != "completed" for row in results):
        raise ValueError("daily rebenchmark cannot complete with failed ICP results")
    return await _update_unless_completed(
        run_id,
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
            "claim_token": "",
            "lease_expires_at": None,
            "updated_at": now_iso(),
            "completed_at": now_iso(),
        },
        claim_token=claim_token,
    )


async def fail_run(
    *,
    run_id: str,
    error_code: str,
    error_message: str,
    worker_ref: str,
    claim_token: str,
) -> dict[str, Any]:
    return await _update_unless_completed(
        run_id,
        {
            "status": "failed",
            "error_doc": {
                "code": str(error_code or "daily_rebenchmark_failed")[:120],
                "message": str(error_message or "")[:500],
            },
            "worker_ref": worker_ref,
            "claim_token": "",
            "lease_expires_at": None,
            "updated_at": now_iso(),
            "completed_at": now_iso(),
        },
        claim_token=claim_token,
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


async def load_latest_completed_run(*, baseline_id: str) -> dict[str, Any] | None:
    rows = await select_many(
        TABLE,
        filters=(("baseline_id", baseline_id), ("status", "completed")),
        order_by=(("benchmark_date", True),),
        limit=1,
    )
    return rows[0] if rows else None
