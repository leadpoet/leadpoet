"""Research Lab operator maintenance controls."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .config import DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS, ResearchLabGatewayConfig
from .key_vault import (
    OpenRouterKeyVaultError,
    decrypt_openrouter_key,
    preflight_openrouter_key,
)
from .public_activity import derive_public_loop_outcome, safe_project_public_loop_activity
from .tee_protocol import legacy_v1_enabled
from .ticket_lifecycle import (
    TERMINAL_TICKET_STATUSES,
    UNPAID_TICKET_TTL_SECONDS,
    is_ticket_expiry_conflict,
)
from .store import (
    create_auto_research_loop_event,
    create_candidate_evaluation_event,
    create_gateway_control_event,
    create_queue_event,
    create_ticket_event,
    select_all,
    select_many,
    select_one,
)


logger = logging.getLogger(__name__)
_POSTGREST_TIMESTAMP_RE = re.compile(
    r"^(?P<prefix>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
    r"\.(?P<fraction>\d{1,9})(?P<suffix>Z|[+-]\d{2}(?::?\d{2})?)?$"
)

AUTORESEARCH_MAINTENANCE_CONTROL_KEY = "autoresearch_maintenance"
SCORING_MAINTENANCE_CONTROL_KEY = "scoring_maintenance"
AUTORESEARCH_PROXY_PREFIXES = (
    "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY",
    "RESEARCH_LAB_WORKER_PROXY",
    "RESEARCH_LAB_WORKER_HTTPS_PROXY",
)
TERMINAL_QUEUE_STATUSES = frozenset({"completed", "failed", "cancelled", "tombstoned"})
ACTIVE_QUEUE_STATUSES = frozenset({"queued", "started", "paused"})
TICKET_LIFECYCLE_QUEUE_BATCH_SIZE = 100
UNPAID_TICKET_EXPIRY_CANDIDATE_VIEW = "research_lab_unpaid_ticket_expiry_candidates"

# Named single-owner maintenance leases. Only the lease holder runs the global
# sweeps for that scope; other workers skip them this pass.
MAINTENANCE_LEASE_HOSTED = "hosted_worker_maintenance"
MAINTENANCE_LEASE_SCORING = "scoring_worker_recovery"


def make_lease_holder_ref(worker_ref: str) -> str:
    """Globally-unique lease-holder token for one worker process/boot.

    ``worker_ref`` (e.g. ``research-lab-worker-1``) is a *stable* name reused by
    every replica and every restart, so it must never be the lease holder id:
    two overlapping gateway processes would present the same holder_ref and the
    acquire RPC would treat the second as the incumbent renewing (both would
    hold the lease and both would run the global sweeps). This appends host,
    PID, and a per-boot UUID so each live process owns a distinct token, while
    the human-readable ``worker_ref`` stays available for logs and event
    attribution. Kept stable for the process lifetime so lease renewal matches.
    """
    try:
        node = os.uname().nodename
    except Exception:
        node = "unknown-host"
    return f"{worker_ref}#{node}#{os.getpid()}#{uuid.uuid4().hex}"


async def try_acquire_maintenance_lease(
    *,
    lease_name: str,
    holder_ref: str,
    ttl_seconds: int,
) -> bool:
    """Acquire or renew a single-owner maintenance lease; True iff held.

    Fail-closed: any error (contention or Supabase failure) returns False so
    the caller skips the global sweep this pass. The current holder — or the
    next taker after the lease expires — performs it, so work is never
    duplicated across workers or replicas, at worst briefly delayed.
    """
    from gateway.research_lab.store import call_rpc

    try:
        result = await call_rpc(
            "research_lab_acquire_maintenance_lease",
            {
                "p_lease_name": str(lease_name),
                "p_holder_ref": str(holder_ref),
                "p_ttl_seconds": int(ttl_seconds),
            },
        )
    except Exception as exc:
        logger.warning(
            "research_lab_maintenance_lease_acquire_failed name=%s error=%s",
            lease_name,
            str(exc)[:200],
        )
        return False
    payload = result[0] if isinstance(result, list) and result else result
    if isinstance(payload, Mapping):
        return bool(payload.get("acquired"))
    return False


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


async def get_scoring_maintenance_state() -> dict[str, Any]:
    try:
        row = await select_one(
            "research_lab_gateway_control_current",
            filters=(("control_key", SCORING_MAINTENANCE_CONTROL_KEY),),
        )
    except Exception as exc:
        logger.warning("research_lab_scoring_maintenance_state_unavailable: %s", str(exc)[:240])
        return {
            "control_key": SCORING_MAINTENANCE_CONTROL_KEY,
            "paused": True,
            "status": "unavailable_fail_closed",
            "unavailable": True,
            "fail_closed": True,
            "error": str(exc)[:240],
        }
    return normalize_scoring_maintenance_state(row)


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


def normalize_scoring_maintenance_state(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "control_key": SCORING_MAINTENANCE_CONTROL_KEY,
            "paused": False,
            "status": "inactive",
        }
    status = str(row.get("current_control_status") or row.get("control_status") or "inactive")
    event_doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
    return {
        "control_key": str(row.get("control_key") or SCORING_MAINTENANCE_CONTROL_KEY),
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


async def is_scoring_maintenance_paused() -> bool:
    return bool((await get_scoring_maintenance_state()).get("paused"))


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


async def set_scoring_maintenance_paused(
    *,
    paused: bool,
    reason: str,
    actor_ref: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await create_gateway_control_event(
        control_key=SCORING_MAINTENANCE_CONTROL_KEY,
        event_type="pause_requested" if paused else "resume_requested",
        control_status="active" if paused else "inactive",
        actor_ref=actor_ref,
        reason=reason,
        event_doc={
            "schema_version": "1.0",
            "maintenance_mode": "cooperative_scoring_pause",
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


async def candidate_scoring_status_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in ("queued", "assigned", "evaluating"):
        rows = await select_all(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id,current_candidate_status",
            filters=(("current_candidate_status", status),),
            max_rows=10000,
        )
        counts[status] = len(rows)
    return counts


def _terminal_ticket_event_for_queue_rows(queue_rows: list[Mapping[str, Any]]) -> str | None:
    if not queue_rows:
        return None
    statuses = {str(row.get("current_queue_status") or "") for row in queue_rows}
    if not statuses or not statuses.issubset(TERMINAL_QUEUE_STATUSES):
        return None
    if "completed" in statuses:
        return "completed"
    if "failed" in statuses:
        return "failed"
    if "cancelled" in statuses:
        return "cancelled"
    return "tombstoned"


def _age_seconds(value: object) -> int | None:
    parsed = _parse_iso(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()))


def _chunks(values: list[str], size: int) -> list[list[str]]:
    chunk_size = max(1, int(size or TICKET_LIFECYCLE_QUEUE_BATCH_SIZE))
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


async def _queue_rows_by_ticket_id(ticket_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {ticket_id: [] for ticket_id in ticket_ids}
    for chunk in _chunks(ticket_ids, TICKET_LIFECYCLE_QUEUE_BATCH_SIZE):
        rows = await select_all(
            "research_loop_run_queue_current",
            columns=(
                "run_id,ticket_id,current_queue_status,current_reason,current_event_hash,"
                "current_status_at,worker_ref"
            ),
            filters=(("ticket_id", "in", chunk),),
            order_by=(("current_status_at", True),),
            max_rows=10000,
        )
        for row in rows:
            ticket_id = str(row.get("ticket_id") or "")
            if ticket_id in grouped:
                grouped[ticket_id].append(row)
    return grouped


async def find_terminal_queue_open_tickets(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return tickets still open even though all expected queue runs are terminal."""
    tickets = await select_all(
        "research_loop_ticket_current",
        columns=(
            "ticket_id,miner_hotkey,requested_loop_count,current_ticket_status,"
            "current_event_hash,current_status_at,created_at"
        ),
        filters=(),
        order_by=(("current_status_at", True), ("created_at", True)),
        max_rows=10000,
    )
    tickets = [
        ticket
        for ticket in tickets
        if str(ticket.get("ticket_id") or "")
        and str(ticket.get("current_ticket_status") or "") not in TERMINAL_TICKET_STATUSES
    ]
    queues_by_ticket_id = await _queue_rows_by_ticket_id([str(ticket["ticket_id"]) for ticket in tickets])
    planned: list[dict[str, Any]] = []
    for ticket in tickets:
        ticket_id = str(ticket.get("ticket_id") or "")
        current_ticket_status = str(ticket.get("current_ticket_status") or "")
        queue_rows = queues_by_ticket_id.get(ticket_id) or []
        terminal_event = _terminal_ticket_event_for_queue_rows(queue_rows)
        requested_loop_count = max(1, int(ticket.get("requested_loop_count") or 1))
        if not terminal_event or len(queue_rows) < requested_loop_count:
            continue
        statuses: dict[str, int] = {}
        for row in queue_rows:
            status = str(row.get("current_queue_status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        latest_queue_status_at = max(
            (str(row.get("current_status_at") or "") for row in queue_rows),
            default=None,
        )
        planned.append(
            {
                "ticket_id": ticket_id,
                "miner_hotkey": ticket.get("miner_hotkey"),
                "current_ticket_status": current_ticket_status,
                "target_ticket_status": terminal_event,
                "requested_loop_count": requested_loop_count,
                "queue_count": len(queue_rows),
                "queue_status_counts": statuses,
                "ticket_event_hash": ticket.get("current_event_hash"),
                "ticket_status_at": ticket.get("current_status_at"),
                "ticket_age_seconds": _age_seconds(ticket.get("created_at") or ticket.get("current_status_at")),
                "ticket_status_age_seconds": _age_seconds(ticket.get("current_status_at")),
                "latest_queue_status_at": latest_queue_status_at,
                "latest_queue_status_age_seconds": _age_seconds(latest_queue_status_at),
                "queue_event_hashes": [
                    row.get("current_event_hash")
                    for row in queue_rows
                    if row.get("current_event_hash")
                ],
            }
        )
        if len(planned) >= max(1, int(limit or 50)):
            break
    return planned


async def find_expirable_unpaid_tickets(
    *,
    ticket_ids: Sequence[str] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    bounded_limit = max(1, min(10000, int(limit or 100)))
    normalized_ids = tuple(dict.fromkeys(str(item).strip() for item in (ticket_ids or ()) if str(item).strip()))
    filters: tuple[tuple[Any, ...], ...] = ()
    if normalized_ids:
        filters = (("ticket_id", "in", normalized_ids),)
    rows = await select_all(
        UNPAID_TICKET_EXPIRY_CANDIDATE_VIEW,
        columns=(
            "ticket_id,miner_hotkey,current_ticket_status,current_event_seq,current_event_hash,"
            "current_status_at,created_at,unpaid_expires_at"
        ),
        filters=filters,
        order_by=(("unpaid_expires_at", False), ("ticket_id", False)),
        batch_size=min(1000, bounded_limit),
        max_rows=bounded_limit,
        allow_partial=True,
    )
    return rows[:bounded_limit]


async def expire_unpaid_tickets(
    *,
    ticket_ids: Sequence[str] | None = None,
    limit: int = 100,
    reason: str = "unpaid_ticket_expired_after_24h",
    actor_ref: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    try:
        planned = await find_expirable_unpaid_tickets(ticket_ids=ticket_ids, limit=limit)
    except Exception as exc:
        logger.warning(
            "research_lab_unpaid_ticket_expiry_scan_failed type=%s error=%s",
            type(exc).__name__,
            str(exc)[:240],
        )
        return {
            "ok": False,
            "dry_run": bool(dry_run),
            "action": "expire-unpaid-tickets",
            "planned_count": 0,
            "expired_count": 0,
            "skipped_count": 0,
            "error": "expiry_candidate_scan_unavailable",
        }
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "expire-unpaid-tickets",
            "planned_count": len(planned),
            "planned": planned,
        }

    effective_actor_ref = actor_ref or default_actor_ref()
    expired: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    projection_pending: list[str] = []
    config = ResearchLabGatewayConfig.from_env()
    for plan in planned:
        ticket_id = str(plan.get("ticket_id") or "")
        try:
            current_candidate = await select_one(
                UNPAID_TICKET_EXPIRY_CANDIDATE_VIEW,
                columns=(
                    "ticket_id,current_ticket_status,current_event_seq,current_event_hash,"
                    "current_status_at,created_at,unpaid_expires_at"
                ),
                filters=(("ticket_id", ticket_id),),
            )
        except Exception as exc:
            logger.warning(
                "research_lab_unpaid_ticket_expiry_recheck_failed ticket_id=%s error=%s",
                ticket_id,
                str(exc)[:240],
            )
            skipped.append({"ticket_id": ticket_id, "reason": "eligibility_recheck_unavailable"})
            continue
        if not current_candidate:
            skipped.append({"ticket_id": ticket_id, "reason": "no_longer_eligible"})
            continue
        event_doc = {
            "schema_version": "1.0",
            "source": "unpaid_ticket_expiry",
            "policy_version": "research_lab_unpaid_ticket_expiry:v1",
            "ttl_seconds": UNPAID_TICKET_TTL_SECONDS,
            "operator_reason": reason,
            "actor_ref": effective_actor_ref,
            "previous_ticket_status": current_candidate.get("current_ticket_status"),
            "previous_ticket_event_hash": current_candidate.get("current_event_hash"),
            "ticket_created_at": current_candidate.get("created_at"),
            "unpaid_expires_at": current_candidate.get("unpaid_expires_at"),
        }
        try:
            event = await create_ticket_event(
                ticket_id=ticket_id,
                event_type="expired",
                actor_hotkey=None,
                reason=reason,
                event_doc=event_doc,
            )
        except Exception as exc:
            skip_reason = "expiry_race_lost" if is_ticket_expiry_conflict(exc) else "expiry_event_insert_failed"
            logger.warning(
                "research_lab_unpaid_ticket_expiry_insert_failed ticket_id=%s reason=%s error=%s",
                ticket_id,
                skip_reason,
                str(exc)[:240],
            )
            skipped.append({"ticket_id": ticket_id, "reason": skip_reason})
            continue
        expired.append(
            {
                "ticket_id": ticket_id,
                "previous_ticket_status": current_candidate.get("current_ticket_status"),
                "unpaid_expires_at": current_candidate.get("unpaid_expires_at"),
                "event_seq": event.get("seq"),
                "event_hash": event.get("anchored_hash"),
            }
        )
        projection = await safe_project_public_loop_activity(
            ticket_id,
            source_ref=f"unpaid_ticket_expiry:{ticket_id}",
            reason=reason,
            config=config,
            force=True,
        )
        if projection is None:
            projection_pending.append(ticket_id)
            logger.warning(
                "research_lab_unpaid_ticket_expiry_projection_pending ticket_id=%s",
                ticket_id,
            )
    return {
        "ok": not any(item.get("reason") == "expiry_event_insert_failed" for item in skipped),
        "dry_run": False,
        "action": "expire-unpaid-tickets",
        "planned_count": len(planned),
        "expired_count": len(expired),
        "skipped_count": len(skipped),
        "projection_pending_count": len(projection_pending),
        "expired": expired,
        "skipped": skipped,
        "projection_pending_ticket_ids": projection_pending,
    }


async def ticket_lifecycle_health(*, sample_limit: int = 25) -> dict[str, Any]:
    open_rows = await select_all(
        "research_loop_ticket_current",
        columns="ticket_id,current_ticket_status,current_status_at,created_at",
        filters=(),
        max_rows=10000,
    )
    open_tickets = [
        row
        for row in open_rows
        if str(row.get("current_ticket_status") or "") not in TERMINAL_TICKET_STATUSES
    ]
    open_ticket_ages = [
        age
        for row in open_tickets
        if (age := _age_seconds(row.get("created_at") or row.get("current_status_at"))) is not None
    ]
    queue_status_counts: dict[str, int] = {}
    oldest_active_queue_age_seconds: int | None = None
    for status in sorted(ACTIVE_QUEUE_STATUSES):
        rows = await select_all(
            "research_loop_run_queue_current",
            columns="run_id,current_queue_status,current_status_at",
            filters=(("current_queue_status", status),),
            max_rows=10000,
        )
        queue_status_counts[status] = len(rows)
        ages = [
            age for row in rows if (age := _age_seconds(row.get("current_status_at"))) is not None
        ]
        if ages:
            oldest = max(ages)
            oldest_active_queue_age_seconds = (
                oldest
                if oldest_active_queue_age_seconds is None
                else max(oldest_active_queue_age_seconds, oldest)
            )
    terminal_queue_open_tickets = await find_terminal_queue_open_tickets(limit=sample_limit)
    terminal_ticket_ages = [
        int(row["ticket_age_seconds"])
        for row in terminal_queue_open_tickets
        if row.get("ticket_age_seconds") is not None
    ]
    config = ResearchLabGatewayConfig.from_env()
    warning_seconds = config.ticket_lifecycle_age_warning_seconds
    unpaid_expiry_rows: list[dict[str, Any]] = []
    unpaid_expiry_available = True
    try:
        unpaid_expiry_rows = await find_expirable_unpaid_tickets(limit=10000)
    except Exception as exc:
        unpaid_expiry_available = False
        logger.warning(
            "research_lab_unpaid_ticket_expiry_health_unavailable type=%s error=%s",
            type(exc).__name__,
            str(exc)[:240],
        )
    unpaid_expiry_ages = [
        age
        for row in unpaid_expiry_rows
        if (age := _age_seconds(row.get("created_at"))) is not None
    ]
    return {
        "ok": len(terminal_queue_open_tickets) == 0,
        "open_ticket_count": len(open_tickets),
        "oldest_open_ticket_age_seconds": max(open_ticket_ages) if open_ticket_ages else None,
        "active_queue_status_counts": queue_status_counts,
        "oldest_active_queue_age_seconds": oldest_active_queue_age_seconds,
        "terminal_queue_open_ticket_count": len(terminal_queue_open_tickets),
        "oldest_terminal_queue_open_ticket_age_seconds": (
            max(terminal_ticket_ages) if terminal_ticket_ages else None
        ),
        "age_warning_seconds": warning_seconds,
        "age_warning": bool(
            (terminal_ticket_ages and max(terminal_ticket_ages) >= warning_seconds)
            or (
                oldest_active_queue_age_seconds is not None
                and oldest_active_queue_age_seconds >= warning_seconds
            )
        ),
        "sample_limit": sample_limit,
        "samples": terminal_queue_open_tickets,
        "unpaid_ticket_expiry": {
            "enabled": config.unpaid_ticket_expiry_enabled,
            "available": unpaid_expiry_available,
            "interval_seconds": config.unpaid_ticket_expiry_interval_seconds,
            "limit": config.unpaid_ticket_expiry_limit,
            "worker_index": config.unpaid_ticket_expiry_worker_index,
            "ttl_seconds": UNPAID_TICKET_TTL_SECONDS,
            "eligible_count": len(unpaid_expiry_rows),
            "oldest_eligible_age_seconds": max(unpaid_expiry_ages) if unpaid_expiry_ages else None,
            "samples": unpaid_expiry_rows[: max(1, int(sample_limit or 25))],
        },
    }


async def _revalidate_terminal_ticket_plan(plan: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
    ticket_id = str(plan.get("ticket_id") or "")
    if not ticket_id:
        return False, "missing_ticket_id", None
    current = await select_one(
        "research_loop_ticket_current",
        columns="ticket_id,current_ticket_status,current_event_hash,current_status_at",
        filters=(("ticket_id", ticket_id),),
    )
    if not current:
        return False, "ticket_missing", None
    current_status = str(current.get("current_ticket_status") or "")
    if current_status in TERMINAL_TICKET_STATUSES:
        return False, "ticket_already_terminal", current
    planned_hash = str(plan.get("ticket_event_hash") or "")
    current_hash = str(current.get("current_event_hash") or "")
    if planned_hash and current_hash and planned_hash != current_hash:
        return False, "ticket_projection_changed", current
    queue_rows = (await _queue_rows_by_ticket_id([ticket_id])).get(ticket_id) or []
    terminal_event = _terminal_ticket_event_for_queue_rows(queue_rows)
    requested_loop_count = max(1, int(plan.get("requested_loop_count") or 1))
    if terminal_event != str(plan.get("target_ticket_status") or ""):
        return False, "queue_terminal_status_changed", current
    if len(queue_rows) < requested_loop_count:
        return False, "missing_expected_queue_rows", current
    return True, "ready", current


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


async def requeue_paused_autoresearch_runs(
    *, actor_ref: str | None = None, reason: str = "maintenance_resume"
) -> dict[str, Any]:
    config = ResearchLabGatewayConfig.from_env()
    capacity_doc = autoresearch_queue_capacity_doc(config)
    rows = await select_all(
        "research_loop_run_queue_current",
        columns=(
            "run_id,ticket_id,current_queue_status,current_reason,queue_priority,"
            "current_event_hash,current_status_at"
        ),
        filters=(("current_queue_status", "paused"),),
        max_rows=10000,
    )
    result: dict[str, Any] = {
        "found_paused": len(rows),
        "requeued": 0,
        "capacity_limited": 0,
        "failed": 0,
        "blocked": [],
    }
    for row in rows:
        if str(row.get("current_reason") or "") == "blocked_for_credit":
            result["blocked"].append(
                {
                    "run_id": row.get("run_id"),
                    "ticket_id": row.get("ticket_id"),
                    "stage": "blocked_for_credit",
                    "error": "explicit credit preflight resume required",
                }
            )
            continue
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
            result["requeued"] += 1
        except Exception as exc:
            error = str(exc)[:240]
            if _is_queue_capacity_conflict(exc):
                result["capacity_limited"] += 1
                result["blocked"].append(
                    {
                        "run_id": row.get("run_id"),
                        "ticket_id": row.get("ticket_id"),
                        "stage": "capacity_guard",
                        "error": error,
                    }
                )
                logger.info(
                    "research_lab_maintenance_resume_capacity_limited run_id=%s error=%s",
                    row.get("run_id"),
                    error,
                )
                continue
            result["failed"] += 1
            result["blocked"].append(
                {
                    "run_id": row.get("run_id"),
                    "ticket_id": row.get("ticket_id"),
                    "stage": "queue_event_insert",
                    "error": error,
                }
            )
            logger.warning(
                "research_lab_maintenance_resume_row_failed run_id=%s error=%s",
                row.get("run_id"),
                error,
            )
    return result


async def reconcile_terminal_loop_projections(
    *,
    run_id: str | None = None,
    limit: int = 50,
    reason: str = "terminal_queue_reconciler",
    actor_ref: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    queue_rows = await select_all(
        "research_loop_run_queue_current",
        columns=(
            "run_id,ticket_id,current_queue_status,current_reason,current_event_seq,"
            "current_event_hash,current_status_at,worker_ref"
        ),
        filters=(("current_queue_status", "in", ["completed", "failed"]),),
        order_by=(("current_status_at", True),),
        max_rows=max(1, int(limit or 50) * 5),
    )
    planned: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    for qrow in queue_rows:
        current_run_id = str(qrow.get("run_id") or "")
        if run_id and current_run_id != str(run_id):
            continue
        loop = await select_one(
            "research_lab_auto_research_loop_current",
            columns=(
                "run_id,ticket_id,receipt_id,current_loop_status,current_event_type,"
                "current_event_seq,current_event_hash,current_status_at"
            ),
            filters=(("run_id", current_run_id),),
        )
        loop_status = str(loop.get("current_loop_status") or "") if loop else ""
        if loop_status in {"completed", "failed"}:
            continue
        queue_status = str(qrow.get("current_queue_status") or "")
        event_type = "loop_completed" if queue_status == "completed" else "loop_failed"
        loop_status_target = "completed" if queue_status == "completed" else "failed"
        plan = {
            "run_id": current_run_id,
            "ticket_id": str(qrow.get("ticket_id") or ""),
            "queue_status": queue_status,
            "loop_status": loop_status or None,
            "event_type": event_type,
            "loop_status_target": loop_status_target,
            "queue_event_hash": qrow.get("current_event_hash"),
            "loop_event_hash": loop.get("current_event_hash") if loop else None,
        }
        planned.append(plan)
        if len(planned) >= max(1, int(limit or 50)):
            break
    if dry_run:
        return {"ok": True, "dry_run": True, "action": "reconcile-loop-projections", "planned": planned}

    for plan in planned:
        event_doc = {
            "schema_version": "1.0",
            "source": "terminal_queue_reconciler",
            "operator_reason": reason,
            "actor_ref": actor_ref or default_actor_ref(),
            "queue_status": plan["queue_status"],
            "previous_loop_status": plan["loop_status"],
            "queue_event_hash": plan["queue_event_hash"],
            "previous_loop_event_hash": plan["loop_event_hash"],
        }
        event = await create_auto_research_loop_event(
            run_id=str(plan["run_id"]),
            ticket_id=str(plan["ticket_id"]),
            receipt_id=None,
            event_type=str(plan["event_type"]),
            loop_status=str(plan["loop_status_target"]),
            worker_ref=actor_ref or default_actor_ref(),
            provider_usage=[],
            event_doc=event_doc,
        )
        await safe_project_public_loop_activity(
            str(plan["ticket_id"]),
            source_ref=f"terminal_queue_reconciler:{plan['run_id']}",
            reason=reason,
            force=True,
        )
        repaired.append({**plan, "event_seq": event.get("seq"), "event_hash": event.get("anchored_hash")})
    return {
        "ok": True,
        "dry_run": False,
        "action": "reconcile-loop-projections",
        "planned_count": len(planned),
        "repaired_count": len(repaired),
        "repaired": repaired,
    }


async def reconcile_terminal_ticket_statuses(
    *,
    ticket_id: str | None = None,
    limit: int = 50,
    reason: str = "terminal_ticket_status_reconciler",
    actor_ref: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    planned = await find_terminal_queue_open_tickets(limit=max(1, int(limit or 50)))
    if ticket_id:
        planned = [row for row in planned if str(row.get("ticket_id") or "") == str(ticket_id)]
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "reconcile-terminal-tickets",
            "planned_count": len(planned),
            "planned": planned,
        }

    repaired: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for plan in planned:
        ready, skip_reason, current_ticket = await _revalidate_terminal_ticket_plan(plan)
        if not ready:
            skipped.append(
                {
                    "ticket_id": plan.get("ticket_id"),
                    "reason": skip_reason,
                    "planned_status": plan.get("target_ticket_status"),
                    "current_ticket_status": (
                        current_ticket.get("current_ticket_status") if current_ticket else None
                    ),
                    "current_event_hash": current_ticket.get("current_event_hash") if current_ticket else None,
                }
            )
            continue
        event_doc = {
            "schema_version": "1.0",
            "source": "terminal_ticket_status_reconciler",
            "operator_reason": reason,
            "actor_ref": actor_ref or default_actor_ref(),
            "previous_ticket_status": plan.get("current_ticket_status"),
            "target_ticket_status": plan.get("target_ticket_status"),
            "requested_loop_count": plan.get("requested_loop_count"),
            "queue_count": plan.get("queue_count"),
            "queue_status_counts": plan.get("queue_status_counts"),
            "previous_ticket_event_hash": plan.get("ticket_event_hash"),
            "queue_event_hashes": plan.get("queue_event_hashes") or [],
        }
        event = await create_ticket_event(
            ticket_id=str(plan["ticket_id"]),
            event_type=str(plan["target_ticket_status"]),
            actor_hotkey=None,
            reason=reason,
            event_doc=event_doc,
        )
        repaired.append(
            {
                **plan,
                "event_seq": event.get("seq"),
                "event_hash": event.get("anchored_hash"),
            }
        )
        try:
            await safe_project_public_loop_activity(
                str(plan["ticket_id"]),
                source_ref=f"terminal_ticket_status_reconciler:{plan['ticket_id']}",
                reason=reason,
                force=True,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_terminal_ticket_public_projection_failed ticket_id=%s error=%s",
                plan.get("ticket_id"),
                str(exc)[:240],
            )
    return {
        "ok": True,
        "dry_run": False,
        "action": "reconcile-terminal-tickets",
        "planned_count": len(planned),
        "repaired_count": len(repaired),
        "skipped_count": len(skipped),
        "repaired": repaired,
        "skipped": skipped,
    }


async def repair_public_loop_cards(
    *,
    ticket_id: str | None = None,
    limit: int = 50,
    reason: str = "operator_public_card_repair",
    dry_run: bool = True,
) -> dict[str, Any]:
    if ticket_id:
        ticket_ids = [str(ticket_id)]
    else:
        tickets = await select_all(
            "research_loop_ticket_current",
            columns="ticket_id,current_status_at,created_at",
            filters=(),
            order_by=(("current_status_at", True), ("created_at", True)),
            max_rows=max(1, int(limit or 50)),
        )
        ticket_ids = [str(row.get("ticket_id")) for row in tickets if row.get("ticket_id")]
    planned: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    for current_ticket_id in ticket_ids[: max(1, int(limit or 50))]:
        derived = await _derive_public_loop_outcome_for_ticket(current_ticket_id)
        if not derived:
            continue
        planned.append(derived)
        if dry_run:
            continue
        await safe_project_public_loop_activity(
            current_ticket_id,
            source_ref=f"operator_public_card_repair:{current_ticket_id}",
            reason=reason,
            force=True,
        )
        repaired.append(derived)
    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "action": "repair-public-cards",
        "planned": planned,
        "repaired_count": len(repaired),
    }


async def _derive_public_loop_outcome_for_ticket(ticket_id: str) -> dict[str, Any] | None:
    ticket = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
    if not ticket:
        return None
    queue_rows = await select_many(
        "research_loop_run_queue_current",
        filters=(("ticket_id", ticket_id),),
        order_by=(("current_status_at", True),),
        limit=1000,
    )
    receipt_rows = await select_many(
        "research_loop_receipt_current",
        filters=(("ticket_id", ticket_id),),
        order_by=(("current_status_at", True),),
        limit=1000,
    )
    candidate_rows = await select_many(
        "research_lab_candidate_evaluation_current",
        filters=(("ticket_id", ticket_id),),
        order_by=(("current_status_at", True),),
        limit=1000,
    )
    score_bundle_rows = await select_many(
        "research_evaluation_score_bundle_current",
        filters=(("ticket_id", ticket_id),),
        order_by=(("current_status_at", True),),
        limit=1000,
    )
    promotion_rows = []
    candidate_ids = [str(row.get("candidate_id")) for row in candidate_rows if row.get("candidate_id")]
    if candidate_ids:
        promotion_rows = await select_many(
            "research_lab_candidate_promotion_events",
            filters=(("candidate_id", "in", candidate_ids),),
            order_by=(("created_at", True),),
            limit=1000,
        )
    outcome = derive_public_loop_outcome(
        ticket=ticket,
        queue_rows=queue_rows,
        receipt_rows=receipt_rows,
        candidate_rows=candidate_rows,
        score_bundle_rows=score_bundle_rows,
        promotion_event_rows=promotion_rows,
        improvement_threshold_points=ResearchLabGatewayConfig.from_env().improvement_threshold_points,
    )
    return {
        "ticket_id": ticket_id,
        "run_id": outcome.run_id,
        "outcome_label": outcome.outcome_label,
        "outcome_band": outcome.outcome_band,
        "event_type": outcome.event_type,
        "candidate_count": outcome.candidate_count,
        "scored_candidate_count": outcome.scored_candidate_count,
    }


# A candidate is safe to requeue when it failed the baseline-readiness race: the
# scorer reached the private-holdout gate before the matching private baseline had
# completed. Once that baseline exists, the same candidate can score cleanly. Anything
# else (invalid patch, lost benchmark, runtime error) is a real failure and is NOT
# auto-requeued unless the operator passes force.
RECOVERABLE_CANDIDATE_FAILURE_MARKERS = ("matching_completed_private_baseline_required",)


def _parse_iso(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        match = _POSTGREST_TIMESTAMP_RE.match(text)
        if not match:
            return None
        suffix = match.group("suffix") or ""
        if suffix == "Z":
            suffix = "+00:00"
        elif re.fullmatch(r"[+-]\d{2}", suffix):
            suffix = f"{suffix}:00"
        elif re.fullmatch(r"[+-]\d{4}", suffix):
            suffix = f"{suffix[:3]}:{suffix[3:]}"
        fraction = (match.group("fraction") + "000000")[:6]
        try:
            return datetime.fromisoformat(f"{match.group('prefix')}.{fraction}{suffix}")
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


async def requeue_stale_started_autoresearch_runs(
    *,
    reason: str = "operator_requeue_stale_started",
    actor_ref: str | None = None,
    dry_run: bool = True,
    max_batch_size: int = 25,
) -> dict[str, Any]:
    """Dry-run-first recovery for queue rows wedged in ``started``.

    This is the operator wrapper around the worker stale reaper: it discovers
    stale ``started`` queue rows, skips rows whose loop has already completed,
    then delegates to ``requeue_failed_loop(force=True)`` so every write remains
    append-only and uses the existing checkpoint/resume planning logic.
    """
    config = ResearchLabGatewayConfig.from_env()
    stale_after_seconds = _stale_after_seconds(config)
    rows = await select_all(
        "research_loop_run_queue_current",
        columns=(
            "run_id,ticket_id,current_queue_status,current_status_at,"
            "current_event_hash,queue_priority,worker_ref"
        ),
        filters=(("current_queue_status", "started"),),
        order_by=(("current_status_at", False),),
        max_rows=max(1, int(max_batch_size)),
    )
    stale_rows = [
        row
        for row in rows
        if _status_is_stale(row.get("current_status_at"), stale_after_seconds)
    ]
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in stale_rows:
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        loop = await select_one(
            "research_lab_auto_research_loop_current",
            columns="run_id,current_loop_status,current_status_at,current_event_type,current_worker_ref",
            filters=(("run_id", run_id),),
        )
        loop_status = str(loop.get("current_loop_status") or "") if loop else ""
        if loop_status == "completed":
            skipped.append(
                {
                    "run_id": run_id,
                    "ticket_id": row.get("ticket_id"),
                    "skipped": "loop_already_completed",
                    "loop_status": loop_status,
                    "loop_event_type": loop.get("current_event_type") if loop else None,
                }
            )
            continue
        results.append(
            await requeue_failed_loop(
                run_id=run_id,
                reason=reason,
                actor_ref=actor_ref,
                dry_run=dry_run,
                force=True,
            )
        )
    return {
        "ok": True,
        "action": "requeue-stale-started-runs",
        "dry_run": bool(dry_run),
        "stale_after_seconds": stale_after_seconds,
        "discovered_started": len(rows),
        "stale_started": len(stale_rows),
        "planned_or_requeued": len(results),
        "skipped": skipped,
        "results": results,
    }


async def resume_credit_blocked_run(
    *,
    run_id: str,
    reason: str = "credit_preflight_passed_resume",
    actor_ref: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    qrow = await select_one(
        "research_loop_run_queue_current",
        columns=(
            "run_id,ticket_id,current_queue_status,current_reason,queue_priority,"
            "current_event_hash,current_status_at"
        ),
        filters=(("run_id", run_id),),
    )
    if not qrow:
        return {"ok": False, "error": "loop_run_not_found", "run_id": run_id}
    if str(qrow.get("current_queue_status") or "") != "paused" or str(qrow.get("current_reason") or "") != "blocked_for_credit":
        return {
            "ok": False,
            "error": "run_not_blocked_for_credit",
            "run_id": run_id,
            "queue_status": qrow.get("current_queue_status"),
            "reason": qrow.get("current_reason"),
        }
    key_doc = await _preflight_openrouter_key_for_run(str(qrow["ticket_id"]))
    if not key_doc.get("ok"):
        return {"ok": False, "run_id": run_id, **key_doc}
    plan = {
        "ok": True,
        "action": "resume-credit-blocked-run",
        "run_id": run_id,
        "ticket_id": str(qrow["ticket_id"]),
        "preflight": key_doc,
    }
    if dry_run:
        return {**plan, "dry_run": True}
    config = ResearchLabGatewayConfig.from_env()
    event = await create_queue_event(
        run_id=str(qrow["run_id"]),
        ticket_id=str(qrow["ticket_id"]),
        event_type="queued",
        queue_priority=int(qrow.get("queue_priority") or 0),
        worker_ref=actor_ref,
        reason=reason,
        event_doc={
            "schema_version": "1.0",
            **autoresearch_queue_capacity_doc(config),
            "resume_source": "research_lab_credit_preflight_cli",
            "previous_reason": qrow.get("current_reason"),
            "previous_event_hash": qrow.get("current_event_hash"),
            "previous_status_at": qrow.get("current_status_at"),
            "actor_ref": actor_ref or default_actor_ref(),
        },
    )
    return {**plan, "dry_run": False, "event_seq": event.get("seq"), "event_hash": event.get("anchored_hash")}


async def _preflight_openrouter_key_for_run(ticket_id: str) -> dict[str, Any]:
    ticket = await select_one(
        "research_loop_ticket_current",
        columns="ticket_id,miner_hotkey,miner_openrouter_key_ref,miner_openrouter_key_handling",
        filters=(("ticket_id", ticket_id),),
    )
    if not ticket:
        return {"ok": False, "error": "ticket_not_found", "ticket_id": ticket_id}
    try:
        if legacy_v1_enabled():
            raw_key = await _resolve_openrouter_key_for_ticket(ticket)
            doc = await asyncio.to_thread(preflight_openrouter_key, raw_key)
        else:
            key_ref = str(ticket.get("miner_openrouter_key_ref") or "")
            miner_hotkey = str(ticket.get("miner_hotkey") or "")
            if not key_ref.startswith("encrypted_ref:openrouter:") or not miner_hotkey:
                raise RuntimeError("encrypted OpenRouter key ref is required")
            from gateway.research_lab.attested_coordinator_v2 import (
                preflight_openrouter_key_ref_v2,
            )
            from gateway.research_lab.chain import (
                resolve_research_lab_evaluation_epoch,
            )

            config = ResearchLabGatewayConfig.from_env()
            epoch_id, _block, _source = await resolve_research_lab_evaluation_epoch(
                config.evaluation_epoch
            )
            authority = await preflight_openrouter_key_ref_v2(
                key_ref=key_ref,
                miner_hotkey=miner_hotkey,
                epoch_id=int(epoch_id),
                sequence=0,
            )
            result = authority.get("result")
            if not isinstance(result, Mapping) or not isinstance(
                result.get("preflight_doc"), Mapping
            ):
                raise RuntimeError("attested OpenRouter preflight result is invalid")
            doc = dict(result["preflight_doc"])
    except Exception as exc:
        return {"ok": False, "error": "openrouter_preflight_failed", "detail": str(exc)[:240]}
    remaining = doc.get("limit_remaining")
    try:
        remaining_value = float(remaining) if remaining is not None else None
    except (TypeError, ValueError):
        remaining_value = None
    if remaining_value is not None and remaining_value <= 0:
        return {"ok": False, "error": "openrouter_credit_still_blocked", "limit_remaining": remaining}
    return {
        "ok": True,
        "key_hash": doc.get("key_hash"),
        "limit": doc.get("limit"),
        "limit_remaining": remaining,
        "limit_reset": doc.get("limit_reset"),
    }


async def _resolve_openrouter_key_for_ticket(ticket: Mapping[str, Any]) -> str:
    config = ResearchLabGatewayConfig.from_env()
    key_ref = str(ticket.get("miner_openrouter_key_ref") or "")
    miner_hotkey = str(ticket.get("miner_hotkey") or "")
    if key_ref.startswith("encrypted_ref:openrouter:"):
        row = await select_one(
            "research_lab_openrouter_key_refs",
            filters=(("key_ref", key_ref), ("miner_hotkey", miner_hotkey)),
        )
        if not row:
            raise OpenRouterKeyVaultError(
                "encrypted OpenRouter key ref was not found"
            )
        return await asyncio.to_thread(
            decrypt_openrouter_key,
            ciphertext_b64=str(row["encrypted_key_ciphertext"]),
            miner_hotkey=miner_hotkey,
            key_ref=key_ref,
        )
    env_name = str(config.miner_openrouter_key_env_var or "")
    if config.miner_openrouter_key_ref_env_map_json:
        try:
            mapping = json.loads(config.miner_openrouter_key_ref_env_map_json)
        except json.JSONDecodeError as exc:
            raise OpenRouterKeyVaultError(
                "OpenRouter key-ref env map is invalid"
            ) from exc
        if isinstance(mapping, Mapping) and mapping.get(key_ref):
            env_name = str(mapping[key_ref])
    if not env_name:
        raise OpenRouterKeyVaultError(
            "no OpenRouter key env var configured for miner key ref"
        )
    raw_key = os.getenv(env_name, "").strip()
    if not raw_key:
        raise OpenRouterKeyVaultError(
            f"configured OpenRouter key env var is empty: {env_name}"
        )
    return raw_key


async def rebase_stale_parent_candidates(
    *,
    candidate_id: str | None = None,
    limit: int = 25,
    max_batch_size: int = 5,
    actor_ref: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    filters: tuple[tuple[Any, ...], ...]
    if candidate_id:
        filters = (("candidate_id", candidate_id),)
        max_rows = 1
    else:
        filters = (("current_candidate_status", "rejected"), ("current_reason", "stale_parent_needs_rescore"))
        max_rows = max(1, int(limit or 25))
    candidates = await select_all(
        "research_lab_candidate_evaluation_current",
        columns="*",
        filters=filters,
        order_by=(("current_status_at", False),),
        max_rows=max_rows,
    )
    candidates = candidates[: max(1, min(int(limit or 25), int(max_batch_size or 5)))]
    planned: list[dict[str, Any]] = []
    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate in candidates:
        cid = str(candidate.get("candidate_id") or "")
        if not cid:
            continue
        existing_rebase = await select_many(
            "research_lab_candidate_promotion_events",
            columns="candidate_id,derived_candidate_id,event_type,promotion_status,created_at",
            filters=(("candidate_id", cid), ("event_type", "rebase_queued")),
            order_by=(("created_at", True),),
            limit=1,
        )
        if existing_rebase:
            skipped.append({"candidate_id": cid, "reason": "already_has_rebase_queued"})
            continue
        item = {
            "candidate_id": cid,
            "run_id": candidate.get("run_id"),
            "ticket_id": candidate.get("ticket_id"),
            "parent_artifact_hash": candidate.get("parent_artifact_hash"),
        }
        planned.append(item)
        if dry_run:
            continue
        from .scoring_worker import ResearchLabGatewayScoringWorker

        worker = ResearchLabGatewayScoringWorker(
            ResearchLabGatewayConfig.from_env(),
            worker_ref=actor_ref or default_actor_ref(),
        )
        result = await worker._maybe_rebase_stale_candidate_before_scoring(
            candidate,
            evaluation_epoch=0,
            elapsed_seconds=lambda: 0.0,
        )
        processed.append({**item, "result": result})
    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "action": "rebase-stale-candidates",
        "planned": planned,
        "processed": processed,
        "skipped": skipped,
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
        "autoresearch_hotkey_capacity": max(
            1,
            int(config.max_active_autoresearch_loops_per_hotkey or 1),
        ),
        "active_loop_stale_after_seconds": max(
            60,
            int(config.active_loop_stale_after_seconds or DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
        ),
    }


def _autoresearch_loop_capacity(config: ResearchLabGatewayConfig) -> int:
    proxy_count = _configured_autoresearch_proxy_count()
    total_workers = max(0, int(config.hosted_worker_total_workers or 0))
    if config.hosted_worker_require_proxy and not proxy_count and not config.hosted_worker_proxy_url:
        # Strict V2 keeps worker proxy plaintext out of the parent process.
        # The configured total is the sealed profile fleet's bound capacity.
        return total_workers
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


async def reconcile_paused_loop_projections(
    *,
    run_id: str | None = None,
    limit: int = 50,
    reason: str = "paused_queue_reconciler",
    actor_ref: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Align loop projections with queue rows that are canonically paused.

    Cooperative pauses (operator pause, blocked_for_credit) land on the queue
    row, but historically the loop event schema had no paused state, so the
    loop projection kept displaying those runs as running. Requires migration
    88 (loop_paused/loop_resumed event types); an insert rejected by the
    events constraint is reported per-run instead of failing the sweep.
    """
    queue_rows = await select_all(
        "research_loop_run_queue_current",
        columns=(
            "run_id,ticket_id,current_queue_status,current_reason,current_event_seq,"
            "current_event_hash,current_status_at,worker_ref"
        ),
        filters=(("current_queue_status", "paused"),),
        order_by=(("current_status_at", True),),
        max_rows=max(1, int(limit or 50) * 5),
    )
    planned: list[dict[str, Any]] = []
    for qrow in queue_rows:
        current_run_id = str(qrow.get("run_id") or "")
        if run_id and current_run_id != str(run_id):
            continue
        loop = await select_one(
            "research_lab_auto_research_loop_current",
            columns=(
                "run_id,ticket_id,receipt_id,current_loop_status,current_event_type,"
                "current_event_seq,current_event_hash,current_status_at"
            ),
            filters=(("run_id", current_run_id),),
        )
        loop_status = str(loop.get("current_loop_status") or "") if loop else ""
        if loop_status != "running":
            continue
        planned.append(
            {
                "run_id": current_run_id,
                "ticket_id": str(qrow.get("ticket_id") or ""),
                "queue_status": "paused",
                "queue_reason": str(qrow.get("current_reason") or ""),
                "loop_status": loop_status,
                "loop_status_target": "paused",
                "queue_event_hash": qrow.get("current_event_hash"),
                "loop_event_hash": loop.get("current_event_hash") if loop else None,
            }
        )
        if len(planned) >= max(1, int(limit or 50)):
            break
    if dry_run:
        return {"ok": True, "dry_run": True, "action": "reconcile-paused-loop-projections", "planned": planned}

    repaired: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for plan in planned:
        event_doc = {
            "schema_version": "1.0",
            "source": "paused_queue_reconciler",
            "operator_reason": reason,
            "actor_ref": actor_ref or default_actor_ref(),
            "queue_status": plan["queue_status"],
            "queue_reason": plan["queue_reason"],
            "previous_loop_status": plan["loop_status"],
            "queue_event_hash": plan["queue_event_hash"],
            "previous_loop_event_hash": plan["loop_event_hash"],
        }
        try:
            event = await create_auto_research_loop_event(
                run_id=str(plan["run_id"]),
                ticket_id=str(plan["ticket_id"]),
                receipt_id=None,
                event_type="loop_paused",
                loop_status="paused",
                worker_ref=actor_ref or default_actor_ref(),
                provider_usage=[],
                event_doc=event_doc,
            )
        except Exception as exc:  # noqa: BLE001 - report per-run, keep sweeping
            failed.append({**plan, "error": str(exc)[:200]})
            continue
        await safe_project_public_loop_activity(
            str(plan["ticket_id"]),
            source_ref=f"paused_queue_reconciler:{plan['run_id']}",
            reason=reason,
            force=True,
        )
        repaired.append({**plan, "event_seq": event.get("seq"), "event_hash": event.get("anchored_hash")})
    return {
        "ok": not failed,
        "dry_run": False,
        "action": "reconcile-paused-loop-projections",
        "planned_count": len(planned),
        "repaired_count": len(repaired),
        "repaired": repaired,
        "failed": failed,
    }


async def reconcile_champion_reward_statuses(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
    limit: int = 50,
    reason: str = "champion_reward_status_reconciler",
    actor_ref: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Mark champion rewards whose scheduled obligation is fully retired as paid.

    Paid-to-date is reconstructed only from allocation inputs whose exact
    validator weight extrinsic has finalized on chain.  Allocation snapshots
    that were merely produced or published cannot close an obligation.
    """
    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.allocations import (
        ACTIVE_CHAMPION_STATUSES,
        _champion_finalized_paid_alpha_to_date,
        _decimal,
        _rate_float,
    )
    from .store import create_champion_reward_event

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    reward_rows: list[dict[str, Any]] = []
    for status in sorted(ACTIVE_CHAMPION_STATUSES):
        reward_rows.extend(
            await select_all(
                "research_lab_champion_reward_current",
                filters=(("current_reward_status", status),),
                max_rows=max(1, int(limit or 50) * 5),
            )
        )
    paid_by_reward = await _champion_finalized_paid_alpha_to_date(
        epoch=effective_epoch,
        netuid=effective_netuid,
        champion_rows=reward_rows,
    )
    planned: list[dict[str, Any]] = []
    for row in reward_rows:
        reward_id = str(row.get("champion_reward_id") or "")
        if not reward_id:
            continue
        desired = _decimal(row.get("desired_alpha_percent") or 0)
        epoch_count = int(row.get("epoch_count") or 0)
        total_due = desired * epoch_count
        paid = min(total_due, _decimal(paid_by_reward.get(reward_id, 0)))
        remaining = total_due - paid
        if desired <= 0 or epoch_count <= 0 or remaining > 0:
            continue
        planned.append(
            {
                "champion_reward_id": reward_id,
                "miner_uid": row.get("miner_uid"),
                "current_reward_status": str(row.get("current_reward_status") or ""),
                "reward_status_target": "paid",
                "total_due_alpha_percent": _rate_float(total_due),
                "paid_alpha_percent_to_date": _rate_float(paid),
            }
        )
        if len(planned) >= max(1, int(limit or 50)):
            break
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "reconcile-champion-reward-statuses",
            "epoch": effective_epoch,
            "planned": planned,
        }

    repaired: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for plan in planned:
        try:
            event = await create_champion_reward_event(
                champion_reward_id=str(plan["champion_reward_id"]),
                event_type="paid",
                reward_status="paid",
                reason=reason,
                event_doc={
                    "schema_version": "1.0",
                    "source": "finalized_chain_champion_status_reconciler",
                    "settlement_authority": "finalized_v2_weight_extrinsics",
                    "actor_ref": actor_ref or default_actor_ref(),
                    "epoch": effective_epoch,
                    "previous_reward_status": plan["current_reward_status"],
                    "total_due_alpha_percent": plan["total_due_alpha_percent"],
                    "paid_alpha_percent_to_date": plan["paid_alpha_percent_to_date"],
                },
            )
        except Exception as exc:  # noqa: BLE001 - continue the idempotent sweep
            logger.warning(
                "research_lab_champion_status_reconcile_write_failed "
                "reward_id=%s error=%s",
                plan["champion_reward_id"],
                str(exc)[:240],
            )
            failed.append(
                {
                    "champion_reward_id": plan["champion_reward_id"],
                    "error": str(exc)[:240],
                }
            )
            continue
        repaired.append({**plan, "event_seq": event.get("seq"), "event_hash": event.get("anchored_hash")})
    return {
        "ok": not failed,
        "dry_run": False,
        "action": "reconcile-champion-reward-statuses",
        "epoch": effective_epoch,
        "planned_count": len(planned),
        "repaired_count": len(repaired),
        "repaired": repaired,
        "failed": failed,
    }


async def reconcile_source_add_reward_statuses(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
    limit: int = 50,
    reason: str = "source_add_reward_fully_delivered",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Stop forward payments on SOURCE_ADD rewards whose obligation is delivered.

    Paid-to-date is reconstructed from the per-epoch allocation snapshots —
    the same first-class SOURCE_ADD rows the allocator itself settles from.
    A reward whose snapshots already sum to its full promised alpha
    (``alpha_percent`` x ``reward_epochs``) gets a ``stopped_forward`` event
    appended, which removes it from the active-status set the allocator
    reads. This is the guard against the allocator's epoch counter freezing
    (as during the 2026-07-20 stateful-epoch switch) while weight setting
    keeps reusing the last computed payout pattern: without the stop event,
    a finished reward keeps collecting every epoch with nothing counting
    it down.
    """

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.allocations import (
        ACTIVE_CHAMPION_STATUSES,
        _decimal,
        _rate_float,
        _source_add_paid_alpha_to_date,
    )
    from .store import insert_row

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    reward_rows: list[dict[str, Any]] = []
    for status in sorted(ACTIVE_CHAMPION_STATUSES):
        reward_rows.extend(
            await select_all(
                "research_lab_source_add_reward_current",
                filters=(("current_reward_status", status),),
                max_rows=max(1, int(limit or 50) * 5),
            )
        )
    paid_by_reward = await _source_add_paid_alpha_to_date(
        epoch=effective_epoch,
        netuid=effective_netuid,
        source_rows=reward_rows,
    )
    planned: list[dict[str, Any]] = []
    planned_refs: set[str] = set()
    for row in reward_rows:
        reward_ref = str(row.get("reward_ref") or "")
        if not reward_ref or reward_ref in planned_refs:
            continue
        desired = _decimal(
            row.get("desired_alpha_percent") or row.get("alpha_percent") or 0
        )
        epoch_count = int(row.get("epoch_count") or row.get("reward_epochs") or 0)
        total_due = desired * epoch_count
        paid = _decimal(paid_by_reward.get(reward_ref, 0))
        if desired <= 0 or epoch_count <= 0 or paid < total_due:
            continue
        planned_refs.add(reward_ref)
        planned.append(
            {
                "reward_ref": reward_ref,
                "miner_hotkey": str(row.get("miner_hotkey") or ""),
                "current_reward_status": str(row.get("current_reward_status") or ""),
                "reward_status_target": "stopped_forward",
                "total_due_alpha_percent": _rate_float(total_due),
                "paid_alpha_percent_to_date": _rate_float(paid),
                "next_seq": int(row.get("current_event_seq") or 0) + 1,
            }
        )
        if len(planned) >= max(1, int(limit or 50)):
            break
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "reconcile-source-add-reward-statuses",
            "epoch": effective_epoch,
            "planned": planned,
        }

    stopped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for plan in planned:
        try:
            await insert_row(
                "research_lab_source_add_reward_events",
                {
                    "reward_ref": plan["reward_ref"],
                    "seq": plan["next_seq"],
                    "reward_status": "stopped_forward",
                    "reason": "%s paid=%.4f due=%.4f epoch=%s"
                    % (
                        reason,
                        plan["paid_alpha_percent_to_date"],
                        plan["total_due_alpha_percent"],
                        effective_epoch,
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001 - continue the idempotent sweep
            logger.warning(
                "research_lab_source_add_status_reconcile_write_failed "
                "reward_ref=%s error=%s",
                plan["reward_ref"],
                str(exc)[:240],
            )
            failed.append(
                {"reward_ref": plan["reward_ref"], "error": str(exc)[:240]}
            )
            continue
        stopped.append(plan)
    return {
        "ok": not failed,
        "dry_run": False,
        "action": "reconcile-source-add-reward-statuses",
        "epoch": effective_epoch,
        "planned_count": len(planned),
        "stopped_count": len(stopped),
        "stopped": stopped,
        "failed": failed,
    }


async def backfill_source_add_reward_v2_authority(
    *,
    epoch: int | None = None,
    limit: int = 1000,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Idempotently attest measured pre-V2 SOURCE_ADD obligations."""

    from gateway.research_lab.allocations import (
        SETTLEMENT_TRACKED_CHAMPION_STATUSES,
    )
    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graph_by_ref_v2,
    )
    from gateway.research_lab.v2_authority import (
        attest_historical_source_add_reward_v2,
    )
    from gateway.tee.reward_executor_v2 import source_add_reward_row_projection_v2
    from leadpoet_canonical.attested_v2 import sha256_json

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    rows: list[dict[str, Any]] = []
    for status in sorted(SETTLEMENT_TRACKED_CHAMPION_STATUSES):
        rows.extend(
            await select_all(
                "research_lab_source_add_reward_current",
                filters=(("current_reward_status", status),),
                order_by=(("created_at", False),),
                max_rows=max(1, int(limit)),
                allow_partial=False,
            )
        )
    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("start_epoch") or 0),
            str(row.get("reward_ref") or ""),
        ),
    )[: max(1, int(limit))]
    covered: list[str] = []
    planned: list[str] = []
    for row in rows:
        reward_ref = str(row.get("reward_ref") or "")
        expected_output = sha256_json(
            source_add_reward_row_projection_v2(
                "source_add_leg%d" % int(row.get("leg") or 0),
                {**dict(row), "initial_reward_status": "active"},
            )
        )
        try:
            graph = await load_business_artifact_graph_by_ref_v2(
                artifact_kind="source_add_reward_decision",
                artifact_ref=reward_ref,
            )
            root_hash = str(graph.get("root_receipt_hash") or "")
            root = next(
                (
                    receipt
                    for receipt in graph.get("receipts") or ()
                    if isinstance(receipt, Mapping)
                    and receipt.get("receipt_hash") == root_hash
                ),
                None,
            )
            if (
                not isinstance(root, Mapping)
                or root.get("purpose") != "research_lab.reward_decision.v2"
                or root.get("output_root") != expected_output
            ):
                raise RuntimeError("stored SOURCE_ADD V2 receipt differs")
            covered.append(reward_ref)
        except Exception as exc:
            logger.info(
                "research_lab_source_add_v2_backfill_required "
                "reward_ref=%s reason=%s",
                reward_ref,
                str(exc)[:200],
            )
            planned.append(reward_ref)
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "backfill-source-add-v2-authority",
            "epoch": effective_epoch,
            "inspected_count": len(rows),
            "already_covered_count": len(covered),
            "planned_count": len(planned),
            "planned_reward_refs": planned,
        }

    migrated: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for reward_ref in planned:
        try:
            outcome = await attest_historical_source_add_reward_v2(
                epoch_id=effective_epoch,
                reward_ref=reward_ref,
            )
            receipt = outcome.get("execution_receipt") or outcome.get("receipt") or {}
            migrated.append(
                {
                    "reward_ref": reward_ref,
                    "receipt_hash": str(receipt.get("receipt_hash") or ""),
                }
            )
        except Exception as exc:
            logger.exception(
                "research_lab_source_add_v2_backfill_failed reward_ref=%s",
                reward_ref,
            )
            failed.append(
                {
                    "reward_ref": reward_ref,
                    "error": str(exc)[:300],
                }
            )
    return {
        "ok": not failed,
        "dry_run": False,
        "action": "backfill-source-add-v2-authority",
        "epoch": effective_epoch,
        "inspected_count": len(rows),
        "already_covered_count": len(covered),
        "migrated_count": len(migrated),
        "migrated": migrated,
        "failed": failed,
    }


async def backfill_champion_reward_v2_authority(
    *,
    epoch: int | None = None,
    limit: int = 1000,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Idempotently attest immutable pre-V2 champion obligations."""

    from gateway.research_lab.allocations import (
        SETTLEMENT_TRACKED_CHAMPION_STATUSES,
    )
    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graph_by_ref_v2,
    )
    from gateway.research_lab.v2_authority import (
        attest_historical_champion_reward_v2,
    )
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2
    from leadpoet_canonical.attested_v2 import sha256_json

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    rows: list[dict[str, Any]] = []
    for status in sorted(SETTLEMENT_TRACKED_CHAMPION_STATUSES):
        rows.extend(
            await select_all(
                "research_lab_champion_reward_current",
                filters=(("current_reward_status", status),),
                order_by=(("created_at", False),),
                max_rows=max(1, int(limit)),
                allow_partial=False,
            )
        )
    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("start_epoch") or 0),
            str(row.get("champion_reward_id") or ""),
        ),
    )[: max(1, int(limit))]
    covered: list[str] = []
    planned: list[str] = []
    for row in rows:
        reward_id = str(row.get("champion_reward_id") or "")
        expected_output = sha256_json(champion_reward_row_projection_v2(row))
        try:
            graph = await load_business_artifact_graph_by_ref_v2(
                artifact_kind="champion_reward_decision",
                artifact_ref=reward_id,
            )
            root_hash = str(graph.get("root_receipt_hash") or "")
            root = next(
                (
                    receipt
                    for receipt in graph.get("receipts") or ()
                    if isinstance(receipt, Mapping)
                    and receipt.get("receipt_hash") == root_hash
                ),
                None,
            )
            if (
                not isinstance(root, Mapping)
                or root.get("purpose") != "research_lab.reward_decision.v2"
                or root.get("output_root") != expected_output
            ):
                raise RuntimeError("stored champion V2 receipt differs")
            covered.append(reward_id)
        except Exception as exc:
            logger.info(
                "research_lab_champion_v2_backfill_required reward_id=%s reason=%s",
                reward_id,
                str(exc)[:200],
            )
            planned.append(reward_id)
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "backfill-champion-v2-authority",
            "epoch": effective_epoch,
            "inspected_count": len(rows),
            "already_covered_count": len(covered),
            "planned_count": len(planned),
            "planned_champion_reward_ids": planned,
        }

    migrated: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for reward_id in planned:
        try:
            outcome = await attest_historical_champion_reward_v2(
                epoch_id=effective_epoch,
                champion_reward_id=reward_id,
            )
            receipt = outcome.get("execution_receipt") or outcome.get("receipt") or {}
            migrated.append(
                {
                    "champion_reward_id": reward_id,
                    "receipt_hash": str(receipt.get("receipt_hash") or ""),
                }
            )
        except Exception as exc:
            logger.exception(
                "research_lab_champion_v2_backfill_failed reward_id=%s",
                reward_id,
            )
            failed.append(
                {
                    "champion_reward_id": reward_id,
                    "error": str(exc)[:300],
                }
            )
    return {
        "ok": not failed,
        "dry_run": False,
        "action": "backfill-champion-v2-authority",
        "epoch": effective_epoch,
        "inspected_count": len(rows),
        "already_covered_count": len(covered),
        "migrated_count": len(migrated),
        "migrated": migrated,
        "failed": failed,
    }


async def backfill_champion_settlement_v2_authority(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
    limit: int = 1000,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Classify missing pre-V2 champion allocation epochs.

    The cutover readiness report is the source of work. Invalid historical
    rows and allocation-hash conflicts remain blocking findings; they are
    never converted into payment authority by this command.
    """

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.v2_authority import (
        classify_historical_champion_allocation_v2,
    )
    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = (
        int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    )
    normalized_limit = max(1, int(limit or 1000))
    before = await champion_v2_cutover_readiness_report(
        epoch=effective_epoch,
        netuid=effective_netuid,
    )
    missing = list(
        before.get("missing_historical_classifications")
        or before.get("missing_historical_settlements")
        or ()
    )
    planned = [
        {
            "epoch": int(item["epoch"]),
            "allocation_hash": str(item["allocation_hash"]),
        }
        for item in missing
        if isinstance(item, Mapping)
        and item.get("reason")
        == "missing_finalized_chain_classification_authority"
        and item.get("epoch") is not None
        and item.get("allocation_hash")
    ]
    planned = sorted(planned, key=lambda item: item["epoch"])[
        :normalized_limit
    ]
    blocked = [
        dict(item)
        for item in missing
        if not (
            isinstance(item, Mapping)
            and item.get("reason")
            == "missing_finalized_chain_classification_authority"
        )
    ]
    if dry_run:
        return {
            "ok": not blocked,
            "dry_run": True,
            "action": "backfill-champion-v2-settlements",
            "epoch": effective_epoch,
            "netuid": effective_netuid,
            "planned_count": len(planned),
            "planned": planned,
            "blocked": blocked,
            "readiness_before": before,
        }

    classified: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for item in planned:
        settlement_epoch = int(item["epoch"])
        try:
            outcome = await classify_historical_champion_allocation_v2(
                epoch_id=effective_epoch,
                netuid=effective_netuid,
                settlement_epoch_id=settlement_epoch,
            )
            receipt = (
                outcome.get("execution_receipt")
                or outcome.get("receipt")
                or {}
            )
            result = outcome.get("result") or {}
            classification = str(outcome.get("status") or "")
            classified.append(
                {
                    **item,
                    "classification": classification,
                    "settlement_hash": str(
                        result.get("settlement_hash") or ""
                    ),
                    "finding_hash": str(result.get("finding_hash") or ""),
                    "receipt_hash": str(receipt.get("receipt_hash") or ""),
                }
            )
        except Exception as exc:  # noqa: BLE001 - continue the idempotent sweep
            logger.exception(
                "research_lab_champion_v2_settlement_backfill_failed "
                "netuid=%s settlement_epoch=%s",
                effective_netuid,
                settlement_epoch,
            )
            failed.append(
                {
                    **item,
                    "error": str(exc)[:300],
                }
            )
    after = await champion_v2_cutover_readiness_report(
        epoch=effective_epoch,
        netuid=effective_netuid,
    )
    return {
        "ok": not blocked and not failed,
        "dry_run": False,
        "action": "backfill-champion-v2-settlements",
        "epoch": effective_epoch,
        "netuid": effective_netuid,
        "planned_count": len(planned),
        "classified_count": len(classified),
        "finalized_count": sum(
            1
            for item in classified
            if item.get("classification") == "finalized"
        ),
        "nonfinalized_count": sum(
            1
            for item in classified
            if item.get("classification") == "not_finalized"
        ),
        # Compatibility aliases for the existing operator command response.
        "migrated_count": len(classified),
        "migrated": classified,
        "blocked": blocked,
        "failed": failed,
        "readiness_before": before,
        "readiness_after": after,
    }


async def champion_v2_cutover_readiness_report(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
) -> dict[str, Any]:
    """Return the operator-visible 100% positive-balance receipt gate."""

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.champion_settlement_v2 import (
        champion_v2_cutover_readiness,
    )
    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = (
        int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    )
    return await champion_v2_cutover_readiness(
        epoch=effective_epoch,
        netuid=effective_netuid,
    )


async def _resolve_maintenance_epoch(epoch: int | None) -> int:
    """Resolve an operator command epoch without requiring gateway lifespan state."""

    if epoch is not None:
        return int(epoch)
    from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch

    resolved_epoch, _block, _source = (
        await resolve_research_lab_evaluation_epoch()
    )
    return int(resolved_epoch)
