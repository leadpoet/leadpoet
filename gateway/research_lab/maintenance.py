"""Research Lab operator maintenance controls."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from .config import DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS, ResearchLabGatewayConfig
from .store import (
    create_candidate_evaluation_event,
    create_gateway_control_event,
    create_queue_event,
    select_all,
    select_many,
    select_one,
)


logger = logging.getLogger(__name__)

AUTORESEARCH_MAINTENANCE_CONTROL_KEY = "autoresearch_maintenance"
AUTORESEARCH_PROXY_PREFIXES = (
    "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY",
    "RESEARCH_LAB_WORKER_PROXY",
    "RESEARCH_LAB_WORKER_HTTPS_PROXY",
)


async def get_autoresearch_maintenance_state() -> dict[str, Any]:
    try:
        row = await select_one(
            "research_lab_gateway_control_current",
            filters=(("control_key", AUTORESEARCH_MAINTENANCE_CONTROL_KEY),),
        )
    except Exception as exc:
        logger.warning("research_lab_maintenance_state_unavailable: %s", str(exc)[:240])
        return {
            "control_key": AUTORESEARCH_MAINTENANCE_CONTROL_KEY,
            "paused": True,
            "status": "unavailable_fail_closed",
            "unavailable": True,
            "fail_closed": True,
            "error": str(exc)[:240],
        }
    return normalize_autoresearch_maintenance_state(row)


def normalize_autoresearch_maintenance_state(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "control_key": AUTORESEARCH_MAINTENANCE_CONTROL_KEY,
            "paused": False,
            "status": "inactive",
        }
    status = str(row.get("current_control_status") or row.get("control_status") or "inactive")
    event_doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
    return {
        "control_key": str(row.get("control_key") or AUTORESEARCH_MAINTENANCE_CONTROL_KEY),
        "paused": status == "active",
        "status": status,
        "event_type": row.get("current_event_type") or row.get("event_type"),
        "reason": row.get("current_reason") or row.get("reason"),
        "actor_ref": row.get("actor_ref"),
        "event_seq": row.get("current_event_seq") or row.get("seq"),
        "event_hash": row.get("current_event_hash") or row.get("anchored_hash"),
        "status_at": row.get("current_status_at") or row.get("created_at"),
        "event_doc": dict(event_doc),
    }


async def is_autoresearch_maintenance_paused() -> bool:
    return bool((await get_autoresearch_maintenance_state()).get("paused"))


async def set_autoresearch_maintenance_paused(
    *,
    paused: bool,
    reason: str,
    actor_ref: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await create_gateway_control_event(
        control_key=AUTORESEARCH_MAINTENANCE_CONTROL_KEY,
        event_type="pause_requested" if paused else "resume_requested",
        control_status="active" if paused else "inactive",
        actor_ref=actor_ref,
        reason=reason,
        event_doc={
            "schema_version": "1.0",
            "maintenance_mode": "cooperative_checkpoint_pause",
            **(event_doc or {}),
        },
    )


async def autoresearch_queue_status_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in ("queued", "started", "paused"):
        rows = await select_all(
            "research_loop_run_queue_current",
            columns="run_id,current_queue_status",
            filters=(("current_queue_status", status),),
            max_rows=10000,
        )
        counts[status] = len(rows)
    return counts


def _stale_after_seconds(config: ResearchLabGatewayConfig) -> int:
    return max(
        60,
        int(config.active_loop_stale_after_seconds or DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
    )


def _status_is_stale(value: object, stale_after_seconds: int) -> bool:
    status_at = _parse_iso(value)
    if status_at is None:
        return True
    if status_at.tzinfo is None:
        status_at = status_at.replace(tzinfo=timezone.utc)
    now = datetime.now(status_at.tzinfo)
    return (now - status_at).total_seconds() >= stale_after_seconds


async def pause_pending_autoresearch_runs(
    *,
    actor_ref: str | None = None,
    reason: str = "maintenance_pause",
) -> dict[str, int]:
    """Move non-running queue work out of queued/started while maintenance is active.

    Queued runs can be paused immediately because workers stop claiming work once
    maintenance is active. Started runs are paused only when their queue heartbeat
    is stale; truly active workers should checkpoint and mark themselves paused.
    """

    config = ResearchLabGatewayConfig.from_env()
    stale_after_seconds = _stale_after_seconds(config)
    capacity_doc = autoresearch_queue_capacity_doc(config)
    result = {
        "queued_paused": 0,
        "stale_started_paused": 0,
        "active_started_left_running": 0,
    }

    for status in ("queued", "started"):
        rows = await select_all(
            "research_loop_run_queue_current",
            columns=(
                "run_id,ticket_id,current_queue_status,current_status_at,"
                "current_event_hash,queue_priority,worker_ref"
            ),
            filters=(("current_queue_status", status),),
            max_rows=10000,
        )
        for row in rows:
            if status == "started" and not _status_is_stale(
                row.get("current_status_at"),
                stale_after_seconds,
            ):
                result["active_started_left_running"] += 1
                continue
            run_id = str(row.get("run_id") or "")
            ticket_id = str(row.get("ticket_id") or "")
            if not run_id or not ticket_id:
                continue
            pause_reason = "maintenance_pause_queued" if status == "queued" else "maintenance_pause_stale_started"
            try:
                await create_queue_event(
                    run_id=run_id,
                    ticket_id=ticket_id,
                    event_type="paused",
                    queue_priority=int(row.get("queue_priority") or 0),
                    worker_ref=actor_ref,
                    reason=pause_reason,
                    event_doc={
                        "schema_version": "1.0",
                        **capacity_doc,
                        "pause_source": "research_lab_maintenance_cli",
                        "operator_reason": reason,
                        "previous_queue_status": status,
                        "previous_worker_ref": row.get("worker_ref"),
                        "previous_event_hash": row.get("current_event_hash"),
                        "previous_status_at": row.get("current_status_at"),
                        "stale_after_seconds": stale_after_seconds,
                    },
                )
                if status == "queued":
                    result["queued_paused"] += 1
                else:
                    result["stale_started_paused"] += 1
            except Exception as exc:
                logger.warning(
                    "research_lab_maintenance_pause_row_failed run_id=%s status=%s error=%s",
                    run_id,
                    status,
                    str(exc)[:240],
                )
    return result


async def requeue_paused_autoresearch_runs(*, actor_ref: str | None = None, reason: str = "maintenance_resume") -> int:
    config = ResearchLabGatewayConfig.from_env()
    capacity_doc = autoresearch_queue_capacity_doc(config)
    rows = await select_all(
        "research_loop_run_queue_current",
        columns="run_id,ticket_id,current_queue_status,queue_priority,current_event_hash,current_status_at",
        filters=(("current_queue_status", "paused"),),
        max_rows=10000,
    )
    requeued = 0
    for row in rows:
        try:
            await create_queue_event(
                run_id=str(row["run_id"]),
                ticket_id=str(row["ticket_id"]),
                event_type="queued",
                queue_priority=int(row.get("queue_priority") or 0),
                worker_ref=actor_ref,
                reason=reason,
                event_doc={
                    "schema_version": "1.0",
                    **capacity_doc,
                    "resume_source": "research_lab_maintenance_cli",
                    "previous_event_hash": row.get("current_event_hash"),
                    "previous_status_at": row.get("current_status_at"),
                },
            )
            requeued += 1
        except Exception as exc:
            if _is_queue_capacity_conflict(exc):
                logger.info(
                    "research_lab_maintenance_resume_capacity_limited run_id=%s error=%s",
                    row.get("run_id"),
                    str(exc)[:240],
                )
                continue
            logger.warning(
                "research_lab_maintenance_resume_row_failed run_id=%s error=%s",
                row.get("run_id"),
                str(exc)[:240],
            )
    return requeued


# A candidate is safe to requeue when it failed the baseline-readiness race: the
# scorer reached the private-holdout gate before the matching private baseline had
# completed. Once that baseline exists, the same candidate can score cleanly. Anything
# else (invalid patch, lost benchmark, runtime error) is a real failure and is NOT
# auto-requeued unless the operator passes force.
RECOVERABLE_CANDIDATE_FAILURE_MARKERS = ("matching_completed_private_baseline_required",)


def _parse_iso(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


async def requeue_failed_candidate(
    *,
    candidate_id: str,
    reason: str = "baseline_ready",
    actor_ref: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Append-only requeue of a candidate that was terminally ``failed`` by the baseline
    race, so a scoring worker re-claims and re-scores it against the now-completed
    private baseline. Verifies (1) the candidate is currently ``failed``, (2) the failure
    is the baseline race, (3) a completed+passed private baseline now exists that was
    created after the failure, (4) the failed event is the latest event. ``force`` skips
    the failure-class and baseline-existence checks; ``dry_run`` reports without writing."""
    events = await select_many(
        "research_lab_candidate_evaluation_events",
        columns="seq,event_type,candidate_status,reason,event_doc,run_id,ticket_id,created_at",
        filters=(("candidate_id", candidate_id),),
        order_by=(("seq", True),),
        limit=1,
    )
    if not events:
        return {"ok": False, "error": "candidate_not_found", "candidate_id": candidate_id}
    latest = events[0]
    status = str(latest.get("candidate_status") or "")
    event_type = str(latest.get("event_type") or "")
    failure_reason = str(latest.get("reason") or "")
    event_doc = latest.get("event_doc") if isinstance(latest.get("event_doc"), Mapping) else {}
    error_detail = str(event_doc.get("error") or "")
    failed_at = _parse_iso(latest.get("created_at"))

    # (1) + (4): the latest event must itself be a failure (nothing has moved on since).
    if status != "failed" or event_type != "failed":
        return {
            "ok": False,
            "error": "candidate_not_in_failed_state",
            "candidate_id": candidate_id,
            "candidate_status": status,
            "latest_event_type": event_type,
        }

    # (2): the failure must be the recoverable baseline race.
    is_baseline_race = failure_reason == "baseline_not_ready" or any(
        marker in error_detail for marker in RECOVERABLE_CANDIDATE_FAILURE_MARKERS
    )
    if not is_baseline_race and not force:
        return {
            "ok": False,
            "error": "failure_not_baseline_race",
            "candidate_id": candidate_id,
            "failure_reason": failure_reason,
            "error_detail": error_detail[:240],
            "hint": "pass force=True only if you are certain this candidate is safe to rescore",
        }

    # (3): a completed+passed private baseline created after the failure must now exist.
    baselines = await select_many(
        "research_lab_private_model_benchmark_current",
        columns=(
            "private_model_manifest_hash,rolling_window_hash,"
            "current_benchmark_status,benchmark_quality,created_at"
        ),
        filters=(("current_benchmark_status", "completed"), ("benchmark_quality", "passed")),
        order_by=(("created_at", True),),
        limit=20,
    )
    newer_baselines = [
        b
        for b in baselines
        if failed_at is not None
        and (_parse_iso(b.get("created_at")) or failed_at) > failed_at
    ]
    if not newer_baselines and not force:
        return {
            "ok": False,
            "error": "no_completed_baseline_after_failure",
            "candidate_id": candidate_id,
            "failed_at": latest.get("created_at"),
        }

    plan = {
        "ok": True,
        "action": "requeue-candidate",
        "candidate_id": candidate_id,
        "prior_failed_seq": latest.get("seq"),
        "failure_reason": failure_reason,
        "matching_baselines_after_failure": len(newer_baselines),
        "forced": bool(force),
    }
    if dry_run:
        return {**plan, "dry_run": True}

    event = await create_candidate_evaluation_event(
        candidate_id=candidate_id,
        run_id=str(latest["run_id"]),
        ticket_id=str(latest["ticket_id"]),
        event_type="queued",
        candidate_status="queued",
        reason=reason,
        event_doc={
            "operator_action": "requeue-candidate",
            "recovered_from": "baseline_not_ready_race",
            "prior_failed_seq": latest.get("seq"),
            "actor_ref": actor_ref or default_actor_ref(),
            "forced": bool(force),
        },
    )
    return {
        **plan,
        "requeued_event_id": event.get("event_id"),
        "requeued_event_seq": event.get("seq"),
        "requeued_event_hash": event.get("anchored_hash"),
    }


async def requeue_failed_loop(
    *,
    run_id: str | None = None,
    ticket_id: str | None = None,
    reason: str = "operator_resume",
    actor_ref: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Append-only requeue of an auto-research loop run whose queue row is terminally
    ``failed``, so a worker re-claims it and resumes from its last checkpoint (or restarts
    from scratch when none exists). The stale reaper recovers ``started`` and ``paused``
    runs, never ``failed`` ones; this is the operator path to bring one back.
    """
    if not run_id and not ticket_id:
        return {"ok": False, "error": "run_id_or_ticket_id_required"}

    columns = (
        "run_id,ticket_id,current_queue_status,queue_priority,"
        "current_event_hash,current_status_at"
    )
    if run_id:
        qrow = await select_one(
            "research_loop_run_queue_current",
            columns=columns,
            filters=(("run_id", run_id),),
        )
    else:
        rows = await select_many(
            "research_loop_run_queue_current",
            columns=columns,
            filters=(("ticket_id", ticket_id),),
            order_by=(("current_status_at", True),),
            limit=1,
        )
        qrow = rows[0] if rows else None

    if not qrow:
        return {"ok": False, "error": "loop_run_not_found", "run_id": run_id, "ticket_id": ticket_id}

    run_id = str(qrow["run_id"])
    ticket_id = str(qrow["ticket_id"])
    queue_status = str(qrow.get("current_queue_status") or "")
    if queue_status != "failed" and not force:
        return {
            "ok": False,
            "error": "loop_not_in_failed_state",
            "run_id": run_id,
            "queue_status": queue_status,
            "hint": "pass force=True to requeue a run that is not in the failed state",
        }

    loop = await select_one(
        "research_lab_auto_research_loop_current",
        columns="run_id,current_loop_status,current_event_seq",
        filters=(("run_id", run_id),),
    )
    loop_status = str(loop.get("current_loop_status") or "") if loop else ""
    if loop_status == "completed" and not force:
        return {
            "ok": False,
            "error": "loop_already_completed",
            "run_id": run_id,
            "loop_status": loop_status,
        }

    checkpoints = await select_many(
        "research_lab_auto_research_loop_events",
        columns="seq",
        filters=(("run_id", run_id), ("event_type", "checkpoint_saved")),
        order_by=(("seq", True),),
        limit=1,
    )
    resume_mode = "resume_from_checkpoint" if checkpoints else "restart_from_scratch"
    config = ResearchLabGatewayConfig.from_env()
    capacity_doc = autoresearch_queue_capacity_doc(config)

    plan = {
        "ok": True,
        "action": "requeue-loop",
        "run_id": run_id,
        "ticket_id": ticket_id,
        "prior_queue_status": queue_status,
        "loop_status": loop_status or None,
        "resume_mode": resume_mode,
        "checkpoint_seq": (checkpoints[0].get("seq") if checkpoints else None),
        "forced": bool(force),
    }
    if dry_run:
        return {**plan, "dry_run": True}

    try:
        event = await create_queue_event(
            run_id=run_id,
            ticket_id=ticket_id,
            event_type="queued",
            queue_priority=int(qrow.get("queue_priority") or 0),
            worker_ref=actor_ref,
            reason=reason,
            event_doc={
                "schema_version": "1.0",
                **capacity_doc,
                "operator_action": "requeue-loop",
                "resume_source": "research_lab_maintenance_cli",
                "resume_mode": resume_mode,
                "previous_queue_status": queue_status,
                "previous_event_hash": qrow.get("current_event_hash"),
                "previous_status_at": qrow.get("current_status_at"),
                "actor_ref": actor_ref or default_actor_ref(),
                "forced": bool(force),
            },
        )
    except Exception as exc:
        if _is_queue_capacity_conflict(exc):
            return {
                "ok": False,
                "error": "queue_capacity_or_hotkey_conflict",
                "run_id": run_id,
                "detail": str(exc)[:240],
                "hint": "the miner already has an active run or the lab is at capacity; retry later",
            }
        raise
    return {
        **plan,
        "requeued_event_id": event.get("event_id"),
        "requeued_event_seq": event.get("seq"),
        "requeued_event_hash": event.get("anchored_hash"),
    }


async def wait_until_autoresearch_drained(timeout_seconds: int, poll_seconds: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    last_counts: dict[str, int] = {}
    pause_actions: list[dict[str, int]] = []
    while True:
        state = await get_autoresearch_maintenance_state()
        if state.get("paused"):
            action = await pause_pending_autoresearch_runs(
                actor_ref=default_actor_ref(),
                reason="maintenance_wait_drained",
            )
            if any(int(value) for value in action.values()):
                pause_actions.append(action)
        last_counts = await autoresearch_queue_status_counts()
        if int(last_counts.get("queued", 0)) == 0 and int(last_counts.get("started", 0)) == 0:
            return {"drained": True, "counts": last_counts, "pause_actions": pause_actions[-10:]}
        if time.monotonic() >= deadline:
            return {"drained": False, "counts": last_counts, "pause_actions": pause_actions[-10:]}
        await asyncio.sleep(max(1.0, float(poll_seconds)))


def default_actor_ref() -> str:
    user = os.getenv("USER") or "operator"
    try:
        host = os.uname().nodename
    except AttributeError:
        host = "unknown-host"
    return f"{user}@{host}"


def dumps_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, default=str)


def autoresearch_queue_capacity_doc(config: ResearchLabGatewayConfig | None = None) -> dict[str, int | str]:
    config = config or ResearchLabGatewayConfig.from_env()
    return {
        "autoresearch_capacity_policy": "proxy_worker_capacity:v1",
        "autoresearch_capacity": int(_autoresearch_loop_capacity(config)),
        "active_loop_stale_after_seconds": max(
            60,
            int(config.active_loop_stale_after_seconds or DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
        ),
    }


def _autoresearch_loop_capacity(config: ResearchLabGatewayConfig) -> int:
    proxy_count = _configured_autoresearch_proxy_count()
    total_workers = max(0, int(config.hosted_worker_total_workers or 0))
    if config.hosted_worker_require_proxy and not proxy_count and not config.hosted_worker_proxy_url:
        return 0
    if proxy_count and total_workers:
        return max(1, min(proxy_count, total_workers))
    if proxy_count:
        return max(1, proxy_count)
    if total_workers:
        return max(1, total_workers)
    if config.hosted_worker_proxy_url:
        return 1
    return 0


def _configured_autoresearch_proxy_count() -> int:
    count = 0
    for index in range(1, 501):
        if any(os.getenv(f"{prefix}_{index}", "").strip() for prefix in AUTORESEARCH_PROXY_PREFIXES):
            count += 1
    return count


def _is_queue_capacity_conflict(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_queue_capacity_conflict" in message
        or "research_lab_queue_hotkey_conflict" in message
        or "23505" in message
    )
