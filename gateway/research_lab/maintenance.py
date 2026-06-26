"""Research Lab operator maintenance controls."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Mapping

from .config import DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS, ResearchLabGatewayConfig
from .store import create_gateway_control_event, create_queue_event, select_all, select_one


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
            if not _is_queue_capacity_conflict(exc):
                raise
            logger.info(
                "research_lab_maintenance_resume_capacity_limited run_id=%s error=%s",
                row.get("run_id"),
                str(exc)[:240],
            )
    return requeued


async def wait_until_autoresearch_drained(timeout_seconds: int, poll_seconds: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    last_counts: dict[str, int] = {}
    while True:
        last_counts = await autoresearch_queue_status_counts()
        if int(last_counts.get("queued", 0)) == 0 and int(last_counts.get("started", 0)) == 0:
            return {"drained": True, "counts": last_counts}
        if time.monotonic() >= deadline:
            return {"drained": False, "counts": last_counts}
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
