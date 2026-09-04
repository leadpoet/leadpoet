"""Daily rebenchmark for the public open-source sourcing baseline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
import os
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from gateway.research_lab import public_baseline_store
from gateway.research_lab.daily_icp_set import (
    DailyIcpSetUnavailable,
    daily_icp_set_from_input_doc,
    fetch_daily_icp_set,
    utc_day_start,
    utc_set_id_for_datetime,
)
from gateway.research_lab.public_baseline_runner import (
    BASELINE_ENTRYPOINT,
    BASELINE_ID,
    BASELINE_REPOSITORY,
    PublicBaselineDockerRunner,
    PublicBaselineRunError,
)
from leadpoet_canonical.production_parity_boundary_v2 import (
    configured_rebenchmark_now_v2,
)
from qualification.scoring.competition import (
    CompetitionCompanyScorer,
    competition_score_from_breakdowns,
    scorer_breakdown_has_retryable_infrastructure_failure,
)
from qualification.scoring.evaluation_clock import use_evaluation_date


logger = logging.getLogger(__name__)


def _safe_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {str(exc)}"
    for env_name in (
        "OPENROUTER_API_KEY",
        "DEEPLINE_API_KEY",
        "SCRAPINGDOG_API_KEY",
    ):
        secret = str(os.getenv(env_name, "") or "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:500]


def _company_goal(icp: Mapping[str, Any]) -> int:
    try:
        return max(1, min(5, int(icp.get("max_companies") or 5)))
    except (TypeError, ValueError):
        return 5


def _benchmark_summary(
    *,
    icp_ref: str,
    score: float,
    sourced_count: int,
    breakdowns: list[Mapping[str, Any]],
) -> dict[str, Any]:
    count = len(breakdowns)
    return {
        "icp_ref": str(icp_ref),
        "score": round(float(score), 6),
        "company_count": count,
        "sourced_count": int(sourced_count),
        "average_icp_fit": round(
            sum(float(row.get("icp_fit") or 0.0) for row in breakdowns)
            / max(1, count),
            6,
        ),
        "average_intent_score": round(
            sum(
                float(row.get("intent_signal_final") or 0.0)
                for row in breakdowns
            )
            / max(1, count),
            6,
        ),
    }


def _existing_results(row: Mapping[str, Any], expected_refs: set[str]) -> list[dict[str, Any]]:
    raw = row.get("per_icp_results")
    if not isinstance(raw, list):
        return []
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, Mapping):
            continue
        item = dict(value)
        ref = str(item.get("icp_ref") or "")
        if ref not in expected_refs or ref in seen:
            continue
        seen.add(ref)
        results.append(item)
    return results


def _usage_summary(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in results if str(row.get("status") or "") == "completed"]
    provider_cost = sum(
        float((row.get("usage") or {}).get("provider_cost_usd") or 0.0)
        for row in completed
        if isinstance(row.get("usage"), Mapping)
    )
    combined_values = [
        float((row.get("usage") or {}).get("combined_cost_usd"))
        for row in completed
        if isinstance(row.get("usage"), Mapping)
        and (row.get("usage") or {}).get("combined_cost_usd") is not None
    ]
    return {
        "completed_attempts": len(completed),
        "provider_calls": sum(
            int((row.get("usage") or {}).get("provider_call_count") or 0)
            for row in completed
            if isinstance(row.get("usage"), Mapping)
        ),
        "provider_cost_usd": round(provider_cost, 6),
        "combined_cost_usd": (
            round(sum(combined_values), 6)
            if len(combined_values) == len(completed)
            else None
        ),
        "model": next(
            (
                str((row.get("usage") or {}).get("model") or "")
                for row in completed
                if isinstance(row.get("usage"), Mapping)
                and (row.get("usage") or {}).get("model")
            ),
            "",
        ),
    }


def _result_by_ref(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("icp_ref") or ""): row for row in results}


def _ordered_results(
    items: tuple[dict[str, Any], ...], results_by_ref: Mapping[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        dict(results_by_ref[str(item["icp_ref"])])
        for item in items
        if str(item["icp_ref"]) in results_by_ref
    ]


async def run_daily_public_rebenchmark(
    *,
    config: Any,
    worker_ref: str,
    evaluation_epoch: int,
    now: datetime | None = None,
    runner_factory: Callable[[], PublicBaselineDockerRunner] = PublicBaselineDockerRunner,
    scorer_factory: Callable[[], CompetitionCompanyScorer] = CompetitionCompanyScorer,
    fetch_window: Callable[..., Any] = fetch_daily_icp_set,
    store: Any = public_baseline_store,
) -> dict[str, Any]:
    """Run or resume one daily baseline against every ICP in today's set."""

    current = configured_rebenchmark_now_v2(
        now=now or datetime.now(timezone.utc)
    ).astimezone(timezone.utc)
    benchmark_date = current.date().isoformat()
    start_offset = max(
        0,
        min(86399, int(getattr(config, "baseline_start_utc_offset_seconds", 0) or 0)),
    )
    earliest_start = utc_day_start(current) + timedelta(seconds=start_offset)
    if current < earliest_start:
        return {
            "status": "waiting_for_daily_icp_activation",
            "benchmark_date": benchmark_date,
            "scheduled_start_at": earliest_start.isoformat(),
        }

    expected_set_id = utc_set_id_for_datetime(current)
    try:
        window = await fetch_window(set_id=expected_set_id, active_at=current)
    except DailyIcpSetUnavailable as exc:
        return {
            "status": "daily_icp_window_not_ready",
            "benchmark_date": benchmark_date,
            "expected_fresh_set_id": expected_set_id,
            "error": str(exc)[:500],
        }

    expected_count = len(window.benchmark_items)
    claim_token = uuid4().hex
    try:
        run = await store.get_or_create_run(
            benchmark_date=benchmark_date,
            baseline_id=BASELINE_ID,
            baseline_repository=BASELINE_REPOSITORY,
            baseline_entrypoint=BASELINE_ENTRYPOINT,
            window_doc=window.public_doc,
            benchmark_input_doc=window.input_doc,
            evaluation_epoch=int(evaluation_epoch),
            expected_icp_count=expected_count,
            worker_ref=worker_ref,
            claim_token=claim_token,
        )
    except public_baseline_store.DailyRebenchmarkBusy:
        return {
            "status": "in_progress_elsewhere",
            "benchmark_date": benchmark_date,
        }
    if int(run.get("expected_icp_count") or 0) != expected_count:
        raise RuntimeError("stored daily rebenchmark ICP count does not match today's set")
    if (
        str(run.get("baseline_repository") or "") != BASELINE_REPOSITORY
        or str(run.get("baseline_entrypoint") or "") != BASELINE_ENTRYPOINT
    ):
        raise RuntimeError("stored daily rebenchmark baseline identity differs")
    window = daily_icp_set_from_input_doc(
        run.get("benchmark_input_doc") or {},
        required_set_id=expected_set_id,
    )
    if dict(run.get("window_doc") or {}) != window.public_doc:
        raise RuntimeError("stored daily rebenchmark ICP references differ")
    if str(run.get("status") or "") == "completed":
        return {
            "status": "already_benchmarked",
            "benchmark_date": benchmark_date,
            "baseline_run_id": str(run.get("run_id") or ""),
            "aggregate_score": float(run.get("aggregate_score") or 0.0),
            "completed_icp_count": int(run.get("completed_icp_count") or 0),
        }
    if str(run.get("status") or "") == "failed":
        return {
            "status": "already_failed",
            "benchmark_date": benchmark_date,
            "baseline_run_id": str(run.get("run_id") or ""),
            "error_code": str((run.get("error_doc") or {}).get("code") or ""),
        }

    expected_refs = {str(item["icp_ref"]) for item in window.benchmark_items}
    results = _existing_results(run, expected_refs)
    results_by_ref = _result_by_ref(results)
    runner: PublicBaselineDockerRunner | None = None
    started = time.monotonic()
    try:
        pending_items = [
            item
            for item in window.benchmark_items
            if str(results_by_ref.get(str(item["icp_ref"]), {}).get("status") or "")
            != "completed"
        ]
        if pending_items:
            runner = runner_factory()
            preflight = await asyncio.to_thread(runner.preflight)
            model = str(preflight.get("model") or "")
            scorer = scorer_factory()
        else:
            model = str(_usage_summary(results).get("model") or "")
            scorer = None
        for item in pending_items:
            icp_ref = str(item["icp_ref"])
            try:
                if runner is None or scorer is None:
                    raise RuntimeError("daily rebenchmark runner is unavailable")
                execution = await asyncio.to_thread(
                    runner.run_icp,
                    item["icp"],
                    evaluation_date=benchmark_date,
                    max_companies=_company_goal(item["icp"]),
                )
                with use_evaluation_date(benchmark_date):
                    breakdowns = await scorer.score_with_breakdowns(
                        execution.companies,
                        item["icp"],
                        False,
                    )
                if any(
                    scorer_breakdown_has_retryable_infrastructure_failure(value)
                    for value in breakdowns
                ):
                    raise RuntimeError(
                        "retryable scorer verification infrastructure failure"
                    )
                score_result = competition_score_from_breakdowns(
                    item["icp"], breakdowns
                )
                score = float(score_result["per_icp_score"])
                summary = _benchmark_summary(
                    icp_ref=icp_ref,
                    score=score,
                    sourced_count=len(execution.companies),
                    breakdowns=breakdowns,
                )
                results_by_ref[icp_ref] = {
                    "icp_ref": icp_ref,
                    "status": "completed",
                    "companies": execution.companies,
                    "score_breakdowns": breakdowns,
                    "summary": summary,
                    "usage": {
                        "model": model,
                        "model_usage": execution.usage,
                        "provider_call_count": len(execution.provider_calls),
                        "provider_cost_usd": execution.provider_cost_usd,
                        "model_cost_usd": execution.model_cost_usd,
                        "combined_cost_usd": execution.combined_cost_usd,
                        "latency_seconds": execution.latency_seconds,
                    },
                }
            except Exception as exc:
                code = (
                    "baseline_run_failed"
                    if isinstance(exc, PublicBaselineRunError)
                    else "baseline_scoring_failed"
                )
                results_by_ref[icp_ref] = {
                    "icp_ref": icp_ref,
                    "status": "failed",
                    "error": {"code": code, "message": _safe_error(exc)},
                }
                ordered = _ordered_results(window.benchmark_items, results_by_ref)
                usage_doc = _usage_summary(ordered)
                await store.save_progress(
                    run_id=str(run["run_id"]),
                    expected_icp_count=expected_count,
                    per_icp_results=ordered,
                    usage_doc=usage_doc,
                    worker_ref=worker_ref,
                    claim_token=claim_token,
                )
                await store.fail_run(
                    run_id=str(run["run_id"]),
                    error_code=code,
                    error_message=_safe_error(exc),
                    worker_ref=worker_ref,
                    claim_token=claim_token,
                )
                logger.warning(
                    "daily_public_rebenchmark_icp_failed icp_ref=%s error=%s",
                    icp_ref,
                    type(exc).__name__,
                )
                return {
                    "status": "failed",
                    "benchmark_date": benchmark_date,
                    "baseline_run_id": str(run["run_id"]),
                    "failed_icp_ref": icp_ref,
                    "error_code": code,
                }

            ordered = _ordered_results(window.benchmark_items, results_by_ref)
            await store.save_progress(
                run_id=str(run["run_id"]),
                expected_icp_count=expected_count,
                per_icp_results=ordered,
                usage_doc=_usage_summary(ordered),
                worker_ref=worker_ref,
                claim_token=claim_token,
            )

        ordered = _ordered_results(window.benchmark_items, results_by_ref)
        summaries = [dict(row["summary"]) for row in ordered]
        if len(summaries) != expected_count:
            raise RuntimeError("daily rebenchmark finished with missing ICP results")
        aggregate_score = sum(float(row.get("score") or 0.0) for row in summaries) / len(
            summaries
        )
        usage_doc = _usage_summary(ordered)
        score_summary_doc = {
            "schema_version": "research_lab_daily_rebenchmark.v1",
            "baseline": {
                "id": BASELINE_ID,
                "repository": BASELINE_REPOSITORY,
                "entrypoint": BASELINE_ENTRYPOINT,
            },
            "benchmark_date": benchmark_date,
            "icp_set_id": int(window.set_id),
            "evaluation_epoch": int(evaluation_epoch),
            "aggregate_score": round(aggregate_score, 6),
            "per_icp_summaries": summaries,
            "usage": usage_doc,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        public_report = {
            "schema_version": "research_lab_public_baseline_report.v1",
            "baseline": {
                "id": BASELINE_ID,
                "repository": BASELINE_REPOSITORY,
                "entrypoint": BASELINE_ENTRYPOINT,
            },
            "benchmark_date": benchmark_date,
            "icp_set_id": int(window.set_id),
            "aggregate_score": round(aggregate_score, 6),
            "completed_icp_count": expected_count,
            "per_icp": [
                {
                    "icp_ref": str(summary.get("icp_ref") or ""),
                    "score": float(summary.get("score") or 0.0),
                    "company_count": int(summary.get("company_count") or 0),
                }
                for summary in summaries
            ],
        }
        completed = await store.complete_run(
            run_id=str(run["run_id"]),
            expected_icp_count=expected_count,
            per_icp_results=ordered,
            aggregate_score=aggregate_score,
            usage_doc=usage_doc,
            score_summary_doc=score_summary_doc,
            public_report_doc=public_report,
            worker_ref=worker_ref,
            claim_token=claim_token,
        )
        return {
            "status": "completed",
            "benchmark_date": benchmark_date,
            "baseline_run_id": str(completed.get("run_id") or run["run_id"]),
            "icp_set_id": int(window.set_id),
            "aggregate_score": round(aggregate_score, 6),
            "completed_icp_count": expected_count,
        }
    except Exception as exc:
        try:
            await store.fail_run(
                run_id=str(run["run_id"]),
                error_code="daily_rebenchmark_failed",
                error_message=_safe_error(exc),
                worker_ref=worker_ref,
                claim_token=claim_token,
            )
        except Exception:
            logger.exception(
                "daily_public_rebenchmark_failure_persistence_failed run_id=%s",
                str(run.get("run_id") or ""),
            )
            raise
        logger.warning(
            "daily_public_rebenchmark_failed run_id=%s error=%s",
            str(run.get("run_id") or ""),
            type(exc).__name__,
        )
        return {
            "status": "failed",
            "benchmark_date": benchmark_date,
            "baseline_run_id": str(run.get("run_id") or ""),
            "error_code": "daily_rebenchmark_failed",
        }
    finally:
        if runner is not None:
            await asyncio.to_thread(runner.close)
