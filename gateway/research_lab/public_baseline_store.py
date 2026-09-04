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
    values: Sequence[Mapping[str, Any]],
    *,
    expected_icp_count: int,
    expected_icp_refs: Sequence[str],
) -> list[dict[str, Any]]:
    if len(values) > int(expected_icp_count):
        raise ValueError("daily rebenchmark has more ICP results than expected")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_refs = {str(value) for value in expected_icp_refs}
    if len(expected_refs) != int(expected_icp_count):
        raise ValueError("daily rebenchmark expected ICP refs are invalid")
    for value in values:
        row = dict(value)
        icp_ref = _result_ref(row)
        if not icp_ref or icp_ref in seen or icp_ref not in expected_refs:
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
        return await claim_run(
            existing,
            worker_ref=worker_ref,
            claim_token=claim_token,
        )
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
        "attempt_count": 1,
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
            return await claim_run(
                existing,
                worker_ref=worker_ref,
                claim_token=claim_token,
            )
        raise


async def load_run(*, benchmark_date: str, baseline_id: str) -> dict[str, Any] | None:
    return await select_one(
        TABLE,
        filters=(("benchmark_date", benchmark_date), ("baseline_id", baseline_id)),
    )


async def claim_run(
    row: Mapping[str, Any], *, worker_ref: str, claim_token: str
) -> dict[str, Any]:
    claimed = await call_rpc(
        "research_lab_claim_daily_rebenchmark",
        {
            "p_run_id": row["run_id"],
            "p_claim_token": claim_token,
            "p_worker_ref": worker_ref,
            "p_lease_seconds": LEASE_SECONDS,
        },
    )
    if not isinstance(claimed, Mapping):
        raise RuntimeError("daily public rebenchmark claim returned no result")
    status = str(claimed.get("claim_status") or "")
    if status not in {"claimed", "exhausted"}:
        raise DailyRebenchmarkBusy("daily public rebenchmark is already running")
    claimed_row = claimed.get("run")
    if not isinstance(claimed_row, Mapping):
        raise RuntimeError("daily public rebenchmark claim returned no row")
    return dict(claimed_row)


async def retry_failed_run(
    row: Mapping[str, Any], *, worker_ref: str, claim_token: str
) -> dict[str, Any] | None:
    """Start the one allowed whole-run retry for a failed daily run."""

    attempt_count = int(row.get("attempt_count") or 1)
    if attempt_count >= 2:
        return None
    retried = await call_rpc(
        "research_lab_retry_daily_rebenchmark",
        {
            "p_run_id": row["run_id"],
            "p_expected_attempt": attempt_count,
            "p_claim_token": claim_token,
            "p_worker_ref": worker_ref,
            "p_lease_seconds": LEASE_SECONDS,
        },
    )
    if not isinstance(retried, Mapping):
        raise RuntimeError("daily public rebenchmark retry returned no result")
    status = str(retried.get("retry_status") or "")
    if status == "exhausted":
        return None
    if status != "retried":
        raise DailyRebenchmarkBusy("daily public rebenchmark retry state changed")
    retried_row = retried.get("run")
    if not isinstance(retried_row, Mapping):
        raise RuntimeError("daily public rebenchmark retry returned no row")
    return dict(retried_row)


async def recover_invalid_completed_run(
    row: Mapping[str, Any], *, worker_ref: str, claim_token: str
) -> dict[str, Any]:
    """Use the one remaining attempt to replace an invalid completed row."""

    run_id = str(row.get("run_id") or "")
    attempt_count = int(row.get("attempt_count") or 1)
    timestamp = now_iso()
    reset = {
        "completed_icp_count": 0,
        "aggregate_score": None,
        "per_icp_results": [],
        "usage_doc": {},
        "score_summary_doc": {},
        "public_report_doc": {},
    }
    if attempt_count >= 2:
        values = {
            **reset,
            "status": "failed",
            "error_doc": {
                "code": "invalid_completed_run",
                "message": "completed daily rebenchmark failed structural validation",
            },
            "worker_ref": worker_ref,
            "claim_token": "",
            "lease_expires_at": None,
            "updated_at": timestamp,
            "completed_at": timestamp,
        }
    else:
        values = {
            **reset,
            "status": "running",
            "attempt_count": attempt_count + 1,
            "error_doc": {},
            "worker_ref": worker_ref,
            "claim_token": claim_token,
            "lease_expires_at": _lease_expiry(),
            "started_at": timestamp,
            "updated_at": timestamp,
            "completed_at": None,
        }
    try:
        return await update_row(
            TABLE,
            values,
            filters=(
                ("run_id", run_id),
                ("status", "completed"),
                ("attempt_count", attempt_count),
            ),
        )
    except Exception:
        current = await select_one(TABLE, filters=(("run_id", run_id),))
        if (
            current is not None
            and attempt_count < 2
            and str(current.get("status") or "") == "running"
            and int(current.get("attempt_count") or 0) == attempt_count + 1
            and str(current.get("claim_token") or "") == claim_token
        ):
            return current
        if (
            current is not None
            and attempt_count >= 2
            and str(current.get("status") or "") == "failed"
            and str((current.get("error_doc") or {}).get("code") or "")
            == "invalid_completed_run"
        ):
            return current
        if current is not None:
            raise DailyRebenchmarkBusy(
                "daily public rebenchmark completion recovery state changed"
            )
        raise


async def reset_progress(
    *, run_id: str, worker_ref: str, claim_token: str
) -> dict[str, Any]:
    """Restart one interrupted attempt without mixing baseline versions."""

    return await update_row(
        TABLE,
        {
            "completed_icp_count": 0,
            "aggregate_score": None,
            "per_icp_results": [],
            "usage_doc": {},
            "score_summary_doc": {},
            "public_report_doc": {},
            "error_doc": {},
            "worker_ref": worker_ref,
            "lease_expires_at": _lease_expiry(),
            "updated_at": now_iso(),
            "completed_at": None,
        },
        filters=(
            ("run_id", run_id),
            ("status", "running"),
            ("claim_token", claim_token),
        ),
    )


async def save_progress(
    *,
    run_id: str,
    expected_icp_count: int,
    per_icp_results: Sequence[Mapping[str, Any]],
    expected_icp_refs: Sequence[str],
    usage_doc: Mapping[str, Any],
    worker_ref: str,
    claim_token: str,
) -> dict[str, Any]:
    results = validate_progress_results(
        per_icp_results,
        expected_icp_count=expected_icp_count,
        expected_icp_refs=expected_icp_refs,
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


async def renew_claim(
    *, run_id: str, worker_ref: str, claim_token: str
) -> dict[str, Any]:
    """Extend one active lease without changing benchmark progress."""

    return await update_row(
        TABLE,
        {
            "worker_ref": worker_ref,
            "lease_expires_at": _lease_expiry(),
            "updated_at": now_iso(),
        },
        filters=(
            ("run_id", run_id),
            ("status", "running"),
            ("claim_token", claim_token),
        ),
    )


async def complete_run(
    *,
    run_id: str,
    expected_icp_count: int,
    per_icp_results: Sequence[Mapping[str, Any]],
    expected_icp_refs: Sequence[str],
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
        expected_icp_refs=expected_icp_refs,
    )
    if len(results) != int(expected_icp_count):
        raise ValueError("daily rebenchmark cannot complete with missing ICP results")
    if any(str(row.get("status") or "") != "completed" for row in results):
        raise ValueError("daily rebenchmark cannot complete with failed ICP results")
    if {_result_ref(row) for row in results} != {
        str(value) for value in expected_icp_refs
    }:
        raise ValueError("daily rebenchmark cannot complete with different ICP refs")
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
