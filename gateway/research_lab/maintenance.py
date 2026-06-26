"""Research Lab operator maintenance controls."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Mapping

from .store import create_gateway_control_event, create_queue_event, select_all, select_one


logger = logging.getLogger(__name__)

AUTORESEARCH_MAINTENANCE_CONTROL_KEY = "autoresearch_maintenance"


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
            "paused": False,
            "status": "unavailable",
            "unavailable": True,
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
    rows = await select_all(
        "research_loop_run_queue_current",
        columns="run_id,ticket_id,current_queue_status,queue_priority,current_event_hash,current_status_at",
        filters=(("current_queue_status", "paused"),),
        max_rows=10000,
    )
    requeued = 0
    for row in rows:
        await create_queue_event(
            run_id=str(row["run_id"]),
            ticket_id=str(row["ticket_id"]),
            event_type="queued",
            queue_priority=int(row.get("queue_priority") or 0),
            worker_ref=actor_ref,
            reason=reason,
            event_doc={
                "schema_version": "1.0",
                "resume_source": "research_lab_maintenance_cli",
                "previous_event_hash": row.get("current_event_hash"),
                "previous_status_at": row.get("current_status_at"),
            },
        )
        requeued += 1
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
