"""Best-effort execution telemetry for Research Lab scoring.

This module is deliberately outside the canonical evaluator/bundle builders.
Every write is additive and failure-contained: telemetry can describe scoring,
but it can never decide scoring, promotion, rewards, reimbursements, or weights.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from gateway.research_lab.store import (
    canonical_hash,
    deterministic_uuid,
    insert_row,
    select_many,
    select_one,
)


logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_RUN_TYPES = {
    "private_baseline_rebenchmark",
    "candidate_scoring",
    "promotion_confirmation",
}
_MODEL_ROLES = {"reference", "candidate"}
_EXECUTION_KINDS = {
    "model_invocation",
    "checkpoint_reuse",
    "gate_skip",
    "latch_skip",
}
_DUPLICATE_MARKERS = ("duplicate key", "unique constraint", "23505")
_DEFAULT_WRITE_ATTEMPTS = 3


def telemetry_enabled(config: Any | None = None) -> bool:
    if config is not None and hasattr(config, "scoring_telemetry_v2_enabled"):
        return bool(getattr(config, "scoring_telemetry_v2_enabled"))
    return str(os.getenv("RESEARCH_LAB_SCORING_TELEMETRY_V2") or "false").strip().lower() in _TRUTHY


def scoring_identity(identity_doc: Mapping[str, Any]) -> str:
    """Return a stable scoring identity without exposing the identity payload."""

    digest = canonical_hash(
        {
            "schema_version": "research_lab_scoring_identity.v2",
            **dict(identity_doc),
        }
    )
    return "scoring:" + digest


def private_baseline_benchmark_id(
    *, benchmark_date: str, rolling_window_hash: str, reference_artifact_hash: str
) -> str:
    window = str(rolling_window_hash or "").removeprefix("sha256:")
    artifact = str(reference_artifact_hash or "").removeprefix("sha256:")
    return f"private_baseline:{benchmark_date}:{window}:{artifact}"


def opaque_checkpoint_ref(*parts: Any) -> str:
    return "scoring_checkpoint:" + canonical_hash(parts).removeprefix("sha256:")


def failure_fingerprint(error: BaseException | str | None) -> str | None:
    if error is None:
        return None
    if isinstance(error, BaseException):
        payload = {"class": error.__class__.__name__, "message": str(error)}
    else:
        payload = {"class": "message", "message": str(error)}
    return canonical_hash(payload)


def _is_duplicate(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _DUPLICATE_MARKERS)


async def _sleep_before_retry(attempt: int) -> None:
    await asyncio.sleep(min(1.0, 0.1 * (2 ** max(0, attempt - 1))))


async def _best_effort_insert(
    table: str,
    row: Mapping[str, Any],
    *,
    key_field: str,
    log_tag: str,
    attempts: int = _DEFAULT_WRITE_ATTEMPTS,
) -> dict[str, Any] | None:
    """Insert one immutable row with bounded retries and idempotent duplicates."""

    key_value = row.get(key_field)
    last_exc: BaseException | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            return await insert_row(table, dict(row))
        except Exception as exc:  # noqa: BLE001 - telemetry is failure-contained
            last_exc = exc
            if _is_duplicate(exc) and key_value is not None:
                try:
                    existing = await select_one(table, filters=((key_field, key_value),))
                except Exception:  # noqa: BLE001 - final warning below is sufficient
                    existing = None
                if existing is not None:
                    return existing
            if attempt < max(1, int(attempts)):
                await _sleep_before_retry(attempt)
    logger.warning(
        "%s table=%s key=%s attempts=%s error=%s",
        log_tag,
        table,
        str(key_value or "")[:120],
        max(1, int(attempts)),
        str(last_exc or "unknown")[:240],
    )
    return None


async def _next_run_attempt(
    scoring_id: str, *, minimum: int = 0
) -> tuple[int, Mapping[str, Any] | None]:
    rows = await select_many(
        "research_lab_scoring_run_current",
        columns="*",
        filters=(("scoring_id", scoring_id),),
        order_by=(("run_attempt", True),),
        limit=1,
    )
    if not rows:
        return max(0, int(minimum)), None
    try:
        return max(int(minimum), int(rows[0].get("run_attempt") or 0) + 1), rows[0]
    except (TypeError, ValueError):
        return max(0, int(minimum)), rows[0]


def _incomplete_run_is_stale(row: Mapping[str, Any] | None) -> bool:
    if not row:
        return False
    status = str(row.get("current_run_status") or "")
    if status in {"completed", "failed", "cancelled", "restarted", "paused"}:
        return False
    try:
        stale_seconds = max(
            60,
            int(os.getenv("RESEARCH_LAB_SCORING_TELEMETRY_STALE_SECONDS", "180")),
        )
    except ValueError:
        stale_seconds = 180
    raw = (
        row.get("last_heartbeat_at")
        or row.get("current_status_at")
        or row.get("created_at")
    )
    if not raw:
        # A concurrently allocated row may not have its first lifecycle event
        # yet. Never declare that fresh row stale merely because status is NULL.
        return False
    try:
        observed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return (datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds() > stale_seconds


@dataclass(frozen=True)
class ScoringRunContext:
    scoring_id: str
    scoring_run_id: str
    run_type: str
    run_attempt: int
    worker_ref: str
    expected_icp_count: int
    scheduler_type: str = "serial"
    benchmark_id: str = ""
    benchmark_date: str = ""
    candidate_id: str = ""
    source_run_id: str = ""
    ticket_id: str = ""
    rolling_window_hash: str = ""
    reference_artifact_hash: str = ""
    reference_manifest_hash: str = ""
    candidate_artifact_hash: str = ""
    candidate_manifest_hash: str = ""
    baseline_benchmark_bundle_id: str = ""
    source_score_bundle_id: str = ""
    evaluation_epoch: int = 0
    resumed_from_scoring_run_id: str = ""

    def linked_ids(self) -> dict[str, str]:
        return {
            "scoring_id": self.scoring_id,
            "scoring_run_id": self.scoring_run_id,
        }


@dataclass(frozen=True)
class IcpExecutionContext:
    scoring_id: str
    scoring_run_id: str
    icp_execution_id: str
    icp_ref: str
    icp_hash: str
    icp_ordinal: int
    model_role: str
    retry_round: int
    attempt_ordinal: int
    execution_kind: str
    phase: str
    worker_ref: str
    source_job_id: str = ""
    reused_from_execution_id: str = ""

    def linked_ids(self) -> dict[str, str]:
        return {
            "scoring_id": self.scoring_id,
            "scoring_run_id": self.scoring_run_id,
            "icp_execution_id": self.icp_execution_id,
        }


async def allocate_scoring_run(
    *,
    identity_doc: Mapping[str, Any],
    run_type: str,
    worker_ref: str,
    expected_icp_count: int,
    scheduler_type: str = "serial",
    minimum_run_attempt: int = 0,
    benchmark_id: str = "",
    benchmark_date: str = "",
    candidate_id: str = "",
    source_run_id: str = "",
    ticket_id: str = "",
    rolling_window_hash: str = "",
    reference_artifact_hash: str = "",
    reference_manifest_hash: str = "",
    candidate_artifact_hash: str = "",
    candidate_manifest_hash: str = "",
    baseline_benchmark_bundle_id: str = "",
    source_score_bundle_id: str = "",
    evaluation_epoch: int = 0,
    resumed_from_scoring_run_id: str = "",
) -> ScoringRunContext | None:
    if run_type not in _RUN_TYPES:
        raise ValueError(f"unsupported scoring telemetry run_type: {run_type}")
    scoring_id = scoring_identity(identity_doc)
    scoring_run_id = str(uuid4())
    run_attempt = max(0, int(minimum_run_attempt))
    stale_predecessor: Mapping[str, Any] | None = None
    for allocation_try in range(1, 6):
        try:
            run_attempt, latest_run = await _next_run_attempt(
                scoring_id, minimum=run_attempt
            )
        except Exception as exc:  # noqa: BLE001 - scoring proceeds without V2
            logger.warning(
                "research_lab_scoring_telemetry_run_attempt_read_failed scoring_id=%s error=%s",
                scoring_id[:64],
                str(exc)[:240],
            )
            return None
        latest_run_id = str((latest_run or {}).get("scoring_run_id") or "")
        if (
            not resumed_from_scoring_run_id
            and latest_run_id
            and _incomplete_run_is_stale(latest_run)
        ):
            stale_predecessor = latest_run
        effective_resumed_from = resumed_from_scoring_run_id or str(
            (stale_predecessor or {}).get("scoring_run_id") or ""
        )
        payload = {
            "scoring_run_id": scoring_run_id,
            "schema_version": "2.0",
            "scoring_id": scoring_id,
            "run_type": run_type,
            "run_attempt": run_attempt,
            "source_run_id": source_run_id or None,
            "ticket_id": ticket_id or None,
            "candidate_id": candidate_id or None,
            "benchmark_id": benchmark_id or None,
            "benchmark_date": benchmark_date or None,
            "rolling_window_hash": rolling_window_hash or None,
            "reference_artifact_hash": reference_artifact_hash or None,
            "reference_manifest_hash": reference_manifest_hash or None,
            "candidate_artifact_hash": candidate_artifact_hash or None,
            "candidate_manifest_hash": candidate_manifest_hash or None,
            "baseline_benchmark_bundle_id": baseline_benchmark_bundle_id or None,
            "source_score_bundle_id": source_score_bundle_id or None,
            "evaluation_epoch": max(0, int(evaluation_epoch)),
            "expected_icp_count": max(0, int(expected_icp_count)),
            "scheduler_type": scheduler_type,
            "worker_ref": worker_ref,
            "resumed_from_scoring_run_id": effective_resumed_from or None,
        }
        row = {**payload, "anchored_hash": canonical_hash(payload)}
        try:
            inserted = await insert_row("research_lab_scoring_runs", row)
        except Exception as exc:  # noqa: BLE001 - allocation conflict or telemetry outage
            if _is_duplicate(exc) and "run_attempt" in str(exc).lower() and allocation_try < 5:
                run_attempt += 1
                continue
            logger.warning(
                "research_lab_scoring_telemetry_run_insert_failed scoring_id=%s attempt=%s error=%s",
                scoring_id[:64],
                run_attempt,
                str(exc)[:240],
            )
            return None
        context = ScoringRunContext(
            scoring_id=scoring_id,
            scoring_run_id=str(inserted.get("scoring_run_id") or scoring_run_id),
            run_type=run_type,
            run_attempt=run_attempt,
            worker_ref=worker_ref,
            expected_icp_count=max(0, int(expected_icp_count)),
            scheduler_type=scheduler_type,
            benchmark_id=benchmark_id,
            benchmark_date=benchmark_date,
            candidate_id=candidate_id,
            source_run_id=source_run_id,
            ticket_id=ticket_id,
            rolling_window_hash=rolling_window_hash,
            reference_artifact_hash=reference_artifact_hash,
            reference_manifest_hash=reference_manifest_hash,
            candidate_artifact_hash=candidate_artifact_hash,
            candidate_manifest_hash=candidate_manifest_hash,
            baseline_benchmark_bundle_id=baseline_benchmark_bundle_id,
            source_score_bundle_id=source_score_bundle_id,
            evaluation_epoch=max(0, int(evaluation_epoch)),
            resumed_from_scoring_run_id=effective_resumed_from,
        )
        if stale_predecessor is not None:
            predecessor = ScoringRunContext(
                scoring_id=str(stale_predecessor.get("scoring_id") or scoring_id),
                scoring_run_id=str(stale_predecessor.get("scoring_run_id") or ""),
                run_type=str(stale_predecessor.get("run_type") or run_type),
                run_attempt=int(stale_predecessor.get("run_attempt") or 0),
                worker_ref=str(stale_predecessor.get("worker_ref") or worker_ref),
                expected_icp_count=int(
                    stale_predecessor.get("expected_icp_count") or 0
                ),
            )
            await emit_run_event(
                predecessor,
                "restarted",
                event_ordinal=run_attempt,
                event_doc={"resumed_by_scoring_run_id": context.scoring_run_id},
            )
        return context
    return None


async def emit_run_event(
    run: ScoringRunContext | None,
    event_type: str,
    *,
    event_ordinal: int = 0,
    retryable: bool | None = None,
    failure_category: str = "",
    error: BaseException | str | None = None,
    checkpoint_ref: str = "",
    checkpoint_hash: str = "",
    telemetry_degraded: bool = False,
    benchmark_bundle_id: str = "",
    score_bundle_id: str = "",
    promotion_event_id: str = "",
    event_doc: Mapping[str, Any] | None = None,
    occurred_at: str | None = None,
) -> dict[str, Any] | None:
    if run is None:
        return None
    occurred = occurred_at or datetime.now(timezone.utc).isoformat()
    event_id = deterministic_uuid(
        "scoring_run_event", run.scoring_run_id, event_type, int(event_ordinal)
    )
    payload = {
        "event_id": event_id,
        "schema_version": "2.0",
        "scoring_id": run.scoring_id,
        "scoring_run_id": run.scoring_run_id,
        "event_type": event_type,
        "event_ordinal": max(0, int(event_ordinal)),
        "occurred_at": occurred,
        "worker_ref": run.worker_ref,
        "retryable": retryable,
        "failure_category": failure_category or None,
        "failure_fingerprint": failure_fingerprint(error),
        "checkpoint_ref": checkpoint_ref or None,
        "checkpoint_hash": checkpoint_hash or None,
        "telemetry_degraded": bool(telemetry_degraded),
        "benchmark_bundle_id": benchmark_bundle_id or None,
        "score_bundle_id": score_bundle_id or None,
        "promotion_event_id": promotion_event_id or None,
        "event_doc": dict(event_doc or {}),
    }
    row = {**payload, "anchored_hash": canonical_hash(payload)}
    return await _best_effort_insert(
        "research_lab_scoring_run_events",
        row,
        key_field="event_id",
        log_tag="research_lab_scoring_telemetry_run_event_insert_failed",
    )


async def create_icp_execution(
    run: ScoringRunContext | None,
    *,
    icp_ref: str,
    icp_hash: str,
    icp_ordinal: int,
    model_role: str,
    retry_round: int = 0,
    attempt_ordinal: int | None = None,
    execution_kind: str = "model_invocation",
    phase: str = "all",
    source_job_id: str = "",
    reused_from_execution_id: str = "",
) -> IcpExecutionContext | None:
    if run is None:
        return None
    if model_role not in _MODEL_ROLES:
        raise ValueError(f"unsupported model_role: {model_role}")
    if execution_kind not in _EXECUTION_KINDS:
        raise ValueError(f"unsupported execution_kind: {execution_kind}")
    ordinal = max(0, int(retry_round if attempt_ordinal is None else attempt_ordinal))
    execution_id = str(uuid4())
    payload = {
        "icp_execution_id": execution_id,
        "schema_version": "2.0",
        "scoring_id": run.scoring_id,
        "scoring_run_id": run.scoring_run_id,
        "icp_ref": str(icp_ref),
        "icp_hash": str(icp_hash or "") or None,
        "icp_ordinal": max(0, int(icp_ordinal)),
        "model_role": model_role,
        "retry_round": max(0, int(retry_round)),
        "attempt_ordinal": ordinal,
        "execution_kind": execution_kind,
        "phase": phase,
        "worker_ref": run.worker_ref,
        "source_job_id": source_job_id or None,
        "reused_from_execution_id": reused_from_execution_id or None,
    }
    row = {**payload, "anchored_hash": canonical_hash(payload)}
    inserted = await _best_effort_insert(
        "research_lab_scoring_icp_executions",
        row,
        key_field="icp_execution_id",
        log_tag="research_lab_scoring_telemetry_icp_execution_insert_failed",
    )
    if inserted is None:
        return None
    return IcpExecutionContext(
        scoring_id=run.scoring_id,
        scoring_run_id=run.scoring_run_id,
        icp_execution_id=str(inserted.get("icp_execution_id") or execution_id),
        icp_ref=str(icp_ref),
        icp_hash=str(icp_hash or ""),
        icp_ordinal=max(0, int(icp_ordinal)),
        model_role=model_role,
        retry_round=max(0, int(retry_round)),
        attempt_ordinal=ordinal,
        execution_kind=execution_kind,
        phase=phase,
        worker_ref=run.worker_ref,
        source_job_id=source_job_id,
        reused_from_execution_id=reused_from_execution_id,
    )


async def emit_icp_event(
    execution: IcpExecutionContext | None,
    event_type: str,
    *,
    event_ordinal: int = 0,
    score: float | None = None,
    sourced_company_count: int | None = None,
    scored_company_count: int | None = None,
    retryable: bool | None = None,
    failure_category: str = "",
    error: BaseException | str | None = None,
    checkpoint_ref: str = "",
    checkpoint_hash: str = "",
    result_row_hash: str = "",
    telemetry_degraded: bool = False,
    event_doc: Mapping[str, Any] | None = None,
    occurred_at: str | None = None,
) -> dict[str, Any] | None:
    if execution is None:
        return None
    occurred = occurred_at or datetime.now(timezone.utc).isoformat()
    event_id = deterministic_uuid(
        "scoring_icp_event",
        execution.icp_execution_id,
        event_type,
        int(event_ordinal),
    )
    payload = {
        "event_id": event_id,
        "schema_version": "2.0",
        "scoring_id": execution.scoring_id,
        "scoring_run_id": execution.scoring_run_id,
        "icp_execution_id": execution.icp_execution_id,
        "event_type": event_type,
        "event_ordinal": max(0, int(event_ordinal)),
        "occurred_at": occurred,
        "score": score,
        "sourced_company_count": sourced_company_count,
        "scored_company_count": scored_company_count,
        "retryable": retryable,
        "failure_category": failure_category or None,
        "failure_fingerprint": failure_fingerprint(error),
        "checkpoint_ref": checkpoint_ref or None,
        "checkpoint_hash": checkpoint_hash or None,
        "result_row_hash": result_row_hash or None,
        "telemetry_degraded": bool(telemetry_degraded),
        "event_doc": dict(event_doc or {}),
    }
    row = {**payload, "anchored_hash": canonical_hash(payload)}
    return await _best_effort_insert(
        "research_lab_scoring_icp_events",
        row,
        key_field="event_id",
        log_tag="research_lab_scoring_telemetry_icp_event_insert_failed",
    )


def result_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only counts/scores safe for the telemetry ledger."""

    score: float | None = None
    if row.get("score") is not None:
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            score = None
    elif isinstance(row.get("candidate_company_scores"), Sequence):
        values = [float(value or 0.0) for value in row.get("candidate_company_scores") or ()]
        score = float(sum(values) / len(values)) if values else 0.0
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), Mapping) else {}
    funnel = (
        row.get("funnel")
        if isinstance(row.get("funnel"), Mapping)
        else diagnostics.get("funnel") if isinstance(diagnostics.get("funnel"), Mapping) else {}
    )
    sourced = funnel.get("sourced") if isinstance(funnel, Mapping) else None
    if sourced is None:
        sourced = row.get("sourced_count") or row.get("model_output_count")
    scored = funnel.get("scored") if isinstance(funnel, Mapping) else None
    if scored is None:
        scored = row.get("company_count")
    if scored is None and isinstance(row.get("candidate_company_scores"), Sequence):
        scored = len(row.get("candidate_company_scores") or ())
    return {
        "score": score,
        "sourced_company_count": int(sourced) if sourced is not None else None,
        "scored_company_count": int(scored) if scored is not None else None,
        "result_row_hash": canonical_hash(
            {key: value for key, value in dict(row).items() if not str(key).startswith("_")}
        ),
    }


@dataclass
class ScoringTelemetrySession:
    """In-memory coordinator for one scoring run's immutable DB records."""

    run: ScoringRunContext | None
    executions: dict[tuple[str, str, int], IcpExecutionContext] = field(default_factory=dict)
    latest_attempt: dict[tuple[str, str], int] = field(default_factory=dict)
    terminal_execution_ids: set[str] = field(default_factory=set)
    degraded: bool = False

    async def _emit_icp(
        self,
        execution: IcpExecutionContext | None,
        event_type: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        result = await emit_icp_event(execution, event_type, **kwargs)
        if execution is not None and result is None:
            self.degraded = True
        return result

    async def plan(
        self,
        *,
        icp_ref: str,
        icp_hash: str,
        icp_ordinal: int,
        model_role: str,
        phase: str = "all",
        held: bool = False,
        execution_kind: str = "model_invocation",
        reused_from_execution_id: str = "",
        source_job_id: str = "",
    ) -> IcpExecutionContext | None:
        key = (model_role, str(icp_ref), 0)
        existing = self.executions.get(key)
        if existing is not None:
            return existing
        execution = await create_icp_execution(
            self.run,
            icp_ref=icp_ref,
            icp_hash=icp_hash,
            icp_ordinal=icp_ordinal,
            model_role=model_role,
            execution_kind=execution_kind,
            phase=phase,
            source_job_id=source_job_id,
            reused_from_execution_id=reused_from_execution_id,
        )
        if execution is None:
            self.degraded = True
            return None
        self.executions[key] = execution
        self.latest_attempt[(model_role, str(icp_ref))] = 0
        await self._emit_icp(execution, "held" if held else "queued")
        return execution

    async def attempt_started(
        self,
        *,
        icp_ref: str,
        icp_hash: str,
        icp_ordinal: int,
        model_role: str,
        retry_round: int,
        phase: str = "all",
        source_job_id: str = "",
    ) -> IcpExecutionContext | None:
        key = (model_role, str(icp_ref), max(0, int(retry_round)))
        execution = self.executions.get(key)
        if execution is None:
            execution = await create_icp_execution(
                self.run,
                icp_ref=icp_ref,
                icp_hash=icp_hash,
                icp_ordinal=icp_ordinal,
                model_role=model_role,
                retry_round=retry_round,
                attempt_ordinal=retry_round,
                phase=phase,
                source_job_id=source_job_id,
            )
            if execution is None:
                self.degraded = True
                return None
            self.executions[key] = execution
            await self._emit_icp(execution, "queued")
        self.latest_attempt[(model_role, str(icp_ref))] = max(0, int(retry_round))
        await self._emit_icp(execution, "started")
        return execution

    def execution_for(
        self, *, icp_ref: str, model_role: str = "candidate", retry_round: int | None = None
    ) -> IcpExecutionContext | None:
        round_no = (
            self.latest_attempt.get((model_role, str(icp_ref)), 0)
            if retry_round is None
            else max(0, int(retry_round))
        )
        return self.executions.get((model_role, str(icp_ref), round_no))

    async def lifecycle(self, action: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Evaluator-facing no-throw lifecycle callback."""

        try:
            icp_ref = str(payload.get("icp_ref") or payload.get("icp_hash") or "")
            model_role = str(payload.get("model_role") or "candidate")
            retry_value = payload.get("retry_round")
            retry_round = max(0, int(retry_value or 0)) if retry_value is not None else None
            execution = self.execution_for(
                icp_ref=icp_ref, model_role=model_role, retry_round=retry_round
            )
            if action == "attempt_started":
                start_retry_round = max(0, int(retry_round or 0))
                execution = await self.attempt_started(
                    icp_ref=icp_ref,
                    icp_hash=str(payload.get("icp_hash") or ""),
                    icp_ordinal=max(0, int(payload.get("icp_ordinal") or 0)),
                    model_role=model_role,
                    retry_round=start_retry_round,
                    phase=str(payload.get("phase") or "all"),
                    source_job_id=str(payload.get("source_job_id") or ""),
                )
            elif action == "sourcing_completed":
                await self._emit_icp(
                    execution,
                    "sourcing_completed",
                    sourced_company_count=(
                        int(payload["sourced_company_count"])
                        if payload.get("sourced_company_count") is not None
                        else None
                    ),
                )
            elif action == "scoring_started":
                await self._emit_icp(execution, "scoring_started")
            elif action == "attempt_failed":
                await self._emit_icp(
                    execution,
                    "failed",
                    retryable=bool(payload.get("retryable")),
                    failure_category=str(payload.get("failure_category") or "runtime_error"),
                    error=str(payload.get("error") or ""),
                )
                if execution is not None:
                    self.terminal_execution_ids.add(execution.icp_execution_id)
            elif action == "attempt_skipped":
                # The evaluator still checkpoints a deterministic zero-score
                # row for timeout-latch skips. Defer the terminal event until
                # the checkpoint callback confirms durability.
                pass
            elif action == "gate_skipped":
                await self._emit_icp(
                    execution,
                    "skipped",
                    failure_category=str(payload.get("failure_category") or "public_gate_rejected"),
                    event_doc={"outcome": "public_gate_rejected"},
                )
                if execution is not None:
                    self.terminal_execution_ids.add(execution.icp_execution_id)
            if execution is None:
                return None
            return {
                **execution.linked_ids(),
                "retry_round": execution.retry_round,
                "model_role": execution.model_role,
            }
        except Exception as exc:  # noqa: BLE001 - observer cannot affect evaluator
            self.degraded = True
            logger.warning(
                "research_lab_scoring_telemetry_lifecycle_failed action=%s error=%s",
                action,
                str(exc)[:240],
            )
            return None

    async def complete_result(
        self,
        row: Mapping[str, Any],
        *,
        model_role: str,
        checkpoint_ref: str = "",
        checkpoint_hash: str = "",
        checkpoint_persisted: bool = True,
        outcome: str = "scored",
        terminal_event: str = "completed",
    ) -> None:
        icp_ref = str(row.get("icp_ref") or row.get("icp_hash") or "")
        execution = self.execution_for(icp_ref=icp_ref, model_role=model_role)
        metrics = result_metrics(row)
        await self._emit_icp(
            execution,
            terminal_event,
            **metrics,
            checkpoint_ref=checkpoint_ref,
            checkpoint_hash=checkpoint_hash,
            telemetry_degraded=self.degraded or not checkpoint_persisted,
            event_doc={
                "outcome": outcome,
                "checkpoint_persisted": bool(checkpoint_persisted),
            },
        )
        if execution is not None:
            self.terminal_execution_ids.add(execution.icp_execution_id)

    async def skip_unstarted(
        self,
        *,
        icp_ref: str,
        model_role: str,
        failure_category: str,
    ) -> None:
        execution = self.execution_for(icp_ref=icp_ref, model_role=model_role, retry_round=0)
        await self._emit_icp(
            execution,
            "skipped",
            failure_category=failure_category,
            event_doc={"outcome": failure_category},
        )
        if execution is not None:
            self.terminal_execution_ids.add(execution.icp_execution_id)

    async def cancel_active(self, *, failure_category: str, error: BaseException | str | None = None) -> None:
        for execution in tuple(self.executions.values()):
            if execution.icp_execution_id in self.terminal_execution_ids:
                continue
            await self._emit_icp(
                execution,
                "cancelled",
                failure_category=failure_category,
                error=error,
                telemetry_degraded=self.degraded,
            )
            self.terminal_execution_ids.add(execution.icp_execution_id)

    async def heartbeat_loop(
        self,
        stop: asyncio.Event,
        *,
        interval_seconds: float = 60.0,
    ) -> None:
        """Emit run and active-execution heartbeats until ``stop`` is set."""

        interval = max(1.0, float(interval_seconds))
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            now = datetime.now(timezone.utc)
            ordinal = int(now.timestamp() // 60)
            await emit_run_event(
                self.run,
                "heartbeat",
                event_ordinal=ordinal,
                telemetry_degraded=self.degraded,
                occurred_at=now.isoformat(),
            )
            for execution in tuple(self.executions.values()):
                if execution.icp_execution_id in self.terminal_execution_ids:
                    continue
                await self._emit_icp(
                    execution,
                    "heartbeat",
                    event_ordinal=ordinal,
                    telemetry_degraded=self.degraded,
                    occurred_at=now.isoformat(),
                )


async def load_scoring_session(scoring_run_id: str) -> ScoringTelemetrySession | None:
    """Hydrate a persisted run for another global-queue worker.

    The queue is distributed, so the worker that claims an ICP is often not
    the worker that allocated and planned the run. Hydration is read-only and
    failure-contained; a miss simply degrades to legacy scoring behavior.
    """

    if not str(scoring_run_id or ""):
        return None
    try:
        row = await select_one(
            "research_lab_scoring_runs",
            filters=(("scoring_run_id", str(scoring_run_id)),),
        )
        if row is None:
            return None
        run = ScoringRunContext(
            scoring_id=str(row.get("scoring_id") or ""),
            scoring_run_id=str(row.get("scoring_run_id") or ""),
            run_type=str(row.get("run_type") or ""),
            run_attempt=int(row.get("run_attempt") or 0),
            worker_ref=str(row.get("worker_ref") or ""),
            expected_icp_count=int(row.get("expected_icp_count") or 0),
            scheduler_type=str(row.get("scheduler_type") or "serial"),
            benchmark_id=str(row.get("benchmark_id") or ""),
            benchmark_date=str(row.get("benchmark_date") or ""),
            candidate_id=str(row.get("candidate_id") or ""),
            source_run_id=str(row.get("source_run_id") or ""),
            ticket_id=str(row.get("ticket_id") or ""),
            rolling_window_hash=str(row.get("rolling_window_hash") or ""),
            reference_artifact_hash=str(row.get("reference_artifact_hash") or ""),
            reference_manifest_hash=str(row.get("reference_manifest_hash") or ""),
            candidate_artifact_hash=str(row.get("candidate_artifact_hash") or ""),
            candidate_manifest_hash=str(row.get("candidate_manifest_hash") or ""),
            baseline_benchmark_bundle_id=str(row.get("baseline_benchmark_bundle_id") or ""),
            source_score_bundle_id=str(row.get("source_score_bundle_id") or ""),
            evaluation_epoch=int(row.get("evaluation_epoch") or 0),
            resumed_from_scoring_run_id=str(row.get("resumed_from_scoring_run_id") or ""),
        )
        session = ScoringTelemetrySession(run)
        execution_rows = await select_many(
            "research_lab_scoring_icp_executions",
            filters=(("scoring_run_id", run.scoring_run_id),),
            order_by=(("created_at", False),),
            limit=max(1000, run.expected_icp_count * 10),
        )
        for execution_row in execution_rows:
            execution = IcpExecutionContext(
                scoring_id=str(execution_row.get("scoring_id") or run.scoring_id),
                scoring_run_id=str(execution_row.get("scoring_run_id") or run.scoring_run_id),
                icp_execution_id=str(execution_row.get("icp_execution_id") or ""),
                icp_ref=str(execution_row.get("icp_ref") or ""),
                icp_hash=str(execution_row.get("icp_hash") or ""),
                icp_ordinal=int(execution_row.get("icp_ordinal") or 0),
                model_role=str(execution_row.get("model_role") or "candidate"),
                retry_round=int(execution_row.get("retry_round") or 0),
                attempt_ordinal=int(execution_row.get("attempt_ordinal") or 0),
                execution_kind=str(execution_row.get("execution_kind") or "model_invocation"),
                phase=str(execution_row.get("phase") or "all"),
                worker_ref=str(execution_row.get("worker_ref") or run.worker_ref),
                source_job_id=str(execution_row.get("source_job_id") or ""),
                reused_from_execution_id=str(execution_row.get("reused_from_execution_id") or ""),
            )
            key = (execution.model_role, execution.icp_ref, execution.retry_round)
            session.executions[key] = execution
            latest_key = (execution.model_role, execution.icp_ref)
            session.latest_attempt[latest_key] = max(
                session.latest_attempt.get(latest_key, 0), execution.retry_round
            )
        if execution_rows:
            event_rows = await select_many(
                "research_lab_scoring_icp_events",
                columns="icp_execution_id,event_type",
                filters=(("scoring_run_id", run.scoring_run_id),),
                limit=max(2000, run.expected_icp_count * 20),
            )
            session.terminal_execution_ids.update(
                str(event.get("icp_execution_id") or "")
                for event in event_rows
                if str(event.get("event_type") or "")
                in {"completed", "failed", "cancelled", "skipped"}
            )
        return session
    except Exception as exc:  # noqa: BLE001 - queue scoring cannot depend on telemetry
        logger.warning(
            "research_lab_scoring_telemetry_hydration_failed scoring_run_id=%s error=%s",
            str(scoring_run_id)[:64],
            str(exc)[:240],
        )
        return None


def checkpoint_telemetry_index(
    session: ScoringTelemetrySession | None,
    rows: Iterable[Mapping[str, Any]],
    *,
    model_role: str,
) -> dict[str, Any]:
    if session is None or session.run is None:
        return {}
    executions: dict[str, dict[str, Any]] = {}
    for row in rows:
        ref = str(row.get("icp_ref") or row.get("icp_hash") or "")
        if not ref:
            continue
        execution = session.execution_for(icp_ref=ref, model_role=model_role)
        if execution is None:
            continue
        executions[ref] = {
            "icp_execution_id": execution.icp_execution_id,
            "scoring_run_id": execution.scoring_run_id,
            "model_role": execution.model_role,
            "retry_round": execution.retry_round,
            "result_row_hash": result_metrics(row)["result_row_hash"],
        }
    return {
        "schema_version": "research_lab_scoring_checkpoint_telemetry.v2",
        "scoring_id": session.run.scoring_id,
        "scoring_run_id": session.run.scoring_run_id,
        "executions": executions,
    }
