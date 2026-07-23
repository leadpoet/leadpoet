"""Supabase persistence helpers for Research Lab gateway endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from typing import Any, Iterable, Mapping
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

from gateway.db.client import get_write_client
from research_lab.canonical import sha256_json

logger = logging.getLogger(__name__)

# A read on the weight-critical path (allocation graph, publication,
# finalization) that hits a transient edge/proxy failure — a Cloudflare or
# gateway 5xx/timeout in front of Supabase, or a dropped connection — must
# not burn a whole 72-minute epoch. These are idempotent reads, so a bounded
# retry is safe. The classifier is an allowlist: anything that is not a
# recognized transient propagates immediately, exactly as before, so genuine
# query-logic errors still fail closed.
_TRANSIENT_READ_ATTEMPTS = 4
_TRANSIENT_READ_BACKOFF_SECONDS = (0.25, 0.75, 1.5)
_TRANSIENT_ERROR_SIGNATURES = (
    "cloudflare",
    "<html",
    "json could not be generated",
    "bad gateway",
    "gateway time-out",
    "gateway timeout",
    "service temporarily unavailable",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "connection refused",
    "server disconnected",
    "timed out",
    "timeout",
)
_TRANSIENT_ERROR_TYPE_SIGNATURES = (
    "timeout",
    "connection",
    "connecterror",
    "readerror",
    "remoteprotocol",
    "serverdisconnected",
)


def _is_transient_read_error(exc: BaseException) -> bool:
    """Return whether a read failure is a retryable edge/network transient.

    Fail-safe: only a recognized transient returns True. An unknown error —
    including a genuine PostgREST/Postgres query error — returns False and
    propagates unchanged.
    """

    type_name = type(exc).__name__.lower()
    if any(token in type_name for token in _TRANSIENT_ERROR_TYPE_SIGNATURES):
        return True
    message = str(getattr(exc, "message", "") or "").lower()
    detail = str(exc).lower()
    haystack = message + "\n" + detail
    # A genuine PostgREST logic error carries a SQLSTATE or PGRST code; never
    # retry those even if some transient token also appears in the payload.
    code = str(getattr(exc, "code", "") or "").strip().lower()
    edge_codes = {"408", "429", "500", "502", "503", "504", "520", "521", "522", "523", "524"}
    if code in edge_codes:
        return True
    if code and code not in edge_codes and (code.startswith("pgrst") or len(code) == 5):
        return False
    return any(token in haystack for token in _TRANSIENT_ERROR_SIGNATURES)


async def _execute_read_with_retry(call, *, label: str):
    """Run an idempotent PostgREST read, retrying only transient failures."""

    last_exc: BaseException | None = None
    for attempt in range(_TRANSIENT_READ_ATTEMPTS):
        try:
            return await asyncio.to_thread(call)
        except Exception as exc:  # noqa: BLE001 - reclassified below
            if not _is_transient_read_error(exc) or attempt == (
                _TRANSIENT_READ_ATTEMPTS - 1
            ):
                raise
            last_exc = exc
            backoff = _TRANSIENT_READ_BACKOFF_SECONDS[
                min(attempt, len(_TRANSIENT_READ_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "transient_read_retry label=%s attempt=%s/%s type=%s error=%s",
                label,
                attempt + 1,
                _TRANSIENT_READ_ATTEMPTS,
                type(exc).__name__,
                str(exc)[:160],
            )
            await asyncio.sleep(backoff)
    # Unreachable: the loop either returns or raises, but keep mypy honest.
    assert last_exc is not None
    raise last_exc



RESEARCH_LAB_UUID_NAMESPACE = uuid5(NAMESPACE_URL, "leadpoet:research_lab:gateway")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def scoring_dispatch_event_anchor_payload(payload: dict[str, Any], dispatch_event_id: str) -> dict[str, Any]:
    return {**payload, "dispatch_event_id": str(dispatch_event_id)}


def deterministic_uuid(*parts: Any) -> str:
    return str(uuid5(RESEARCH_LAB_UUID_NAMESPACE, canonical_hash(parts)))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_filters(query: Any, filters: Iterable[tuple[Any, ...]]) -> Any:
    for raw_filter in filters:
        if len(raw_filter) == 2:
            field, value = raw_filter
            query = query.eq(field, str(value) if isinstance(value, UUID) else value)
            continue
        if len(raw_filter) != 3:
            raise ValueError(f"invalid PostgREST filter spec: {raw_filter!r}")
        field, operator, value = raw_filter
        value = str(value) if isinstance(value, UUID) else value
        if operator == "eq":
            query = query.eq(field, value)
        elif operator == "neq":
            query = query.neq(field, value)
        elif operator == "lt":
            query = query.lt(field, value)
        elif operator == "lte":
            query = query.lte(field, value)
        elif operator == "gt":
            query = query.gt(field, value)
        elif operator == "gte":
            query = query.gte(field, value)
        elif operator == "in":
            query = query.in_(field, value)
        else:
            raise ValueError(f"unsupported PostgREST filter operator: {operator}")
    return query


async def insert_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
    def _call() -> Any:
        return get_write_client().table(table).insert(row).execute()

    response = await asyncio.to_thread(_call)
    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError(f"{table}: insert returned no rows")
    return dict(data[0])


async def call_rpc(function_name: str, params: Mapping[str, Any]) -> Any:
    """Call one service-role PostgREST function without blocking the event loop."""
    def _call() -> Any:
        return get_write_client().rpc(function_name, dict(params)).execute()

    response = await asyncio.to_thread(_call)
    return getattr(response, "data", None)


def insert_row_sync(table: str, row: dict[str, Any]) -> dict[str, Any]:
    response = get_write_client().table(table).insert(row).execute()
    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError(f"{table}: insert returned no rows")
    return dict(data[0])


async def persist_source_add_submission(record_doc: dict[str, Any]) -> None:
    """Persist a SOURCE_ADD submission's funnel stages (W5, append-only).

    One row per ``stage_history`` entry, keyed UNIQUE(submission_id, seq).
    Idempotent: already-persisted stage rows are skipped, so calling this
    after every funnel transition writes only the new stages. The full
    submission doc rides on the newest row (earlier rows keep a stub) so the
    current view always returns the freshest state.
    """

    submission_id = str(record_doc.get("submission_id") or "")
    stage_history = [str(stage) for stage in (record_doc.get("stage_history") or [])]
    if not submission_id or not stage_history:
        raise ValueError("source_add submission doc requires submission_id and stage_history")
    yield_value = record_doc.get("measured_trial_yield")
    measured_yield = float(yield_value) if isinstance(yield_value, (int, float)) and float(yield_value) >= 0 else None
    last_seq = len(stage_history) - 1
    precheck_doc = record_doc.get("precheck_doc") if isinstance(record_doc.get("precheck_doc"), Mapping) else {}
    for seq, stage in enumerate(stage_history):
        row = {
            "submission_id": submission_id,
            "adapter_id": str(record_doc.get("adapter_id") or ""),
            "miner_hotkey": str(record_doc.get("miner_hotkey") or ""),
            "stage": stage,
            "seq": seq,
            "measured_trial_yield": measured_yield if seq == last_seq else None,
            "submission_doc": record_doc if seq == last_seq else {},
            "precheck_status": str(record_doc.get("precheck_status") or "") if seq == last_seq else "",
            "precheck_doc": dict(precheck_doc) if seq == last_seq else {},
            "source_identity_hash": str(record_doc.get("source_identity_hash") or "") if seq == last_seq else "",
        }
        try:
            await insert_row("research_lab_source_add_submissions", row)
        except Exception as exc:
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower() or "23505" in str(exc):
                continue
            raise


async def update_row(
    table: str,
    values: dict[str, Any],
    *,
    filters: Iterable[tuple[Any, ...]],
) -> dict[str, Any]:
    def _call() -> Any:
        query = get_write_client().table(table).update(values)
        query = _apply_filters(query, filters)
        return query.execute()

    response = await asyncio.to_thread(_call)
    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError(f"{table}: update returned no rows")
    return dict(data[0])


async def select_one(
    table: str,
    *,
    columns: str = "*",
    filters: Iterable[tuple[str, Any]],
) -> dict[str, Any] | None:
    normalized_filters = tuple(filters)

    def _call() -> Any:
        query = get_write_client().table(table).select(columns)
        query = _apply_filters(query, normalized_filters)
        return query.limit(1).execute()

    response = await _execute_read_with_retry(
        _call, label="select_one:%s" % table
    )
    data = getattr(response, "data", None) or []
    return dict(data[0]) if data else None


async def select_many(
    table: str,
    *,
    columns: str = "*",
    filters: Iterable[tuple[str, Any]],
    order_by: Iterable[tuple[str, bool]] = (),
    limit: int = 100,
) -> list[dict[str, Any]]:
    normalized_filters = tuple(filters)
    normalized_order = tuple(order_by)

    def _call() -> Any:
        query = get_write_client().table(table).select(columns)
        query = _apply_filters(query, normalized_filters)
        for field, desc in normalized_order:
            query = query.order(field, desc=desc)
        return query.limit(limit).execute()

    response = await _execute_read_with_retry(
        _call, label="select_many:%s" % table
    )
    return [dict(row) for row in (getattr(response, "data", None) or [])]


async def select_all(
    table: str,
    *,
    columns: str = "*",
    filters: Iterable[tuple[str, Any]],
    order_by: Iterable[tuple[str, bool]] = (),
    batch_size: int = 1000,
    max_rows: int = 10000,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    """Fetch rows with explicit PostgREST pagination for weight-critical paths."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    normalized_filters = tuple(filters)
    normalized_order = tuple(order_by)
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < max_rows:
        end = min(offset + batch_size - 1, max_rows - 1)

        def _call() -> Any:
            query = get_write_client().table(table).select(columns)
            query = _apply_filters(query, normalized_filters)
            for field, desc in normalized_order:
                query = query.order(field, desc=desc)
            return query.range(offset, end).execute()

        response = await _execute_read_with_retry(
            _call, label="select_all:%s" % table
        )
        batch = [dict(row) for row in (getattr(response, "data", None) or [])]
        rows.extend(batch)
        if len(batch) < batch_size:
            return rows
        offset += batch_size
    if allow_partial:
        return rows
    raise RuntimeError(f"{table}: paginated select exceeded max_rows={max_rows}")


async def next_event_seq(table: str, key_field: str, key_value: Any) -> int:
    def _call() -> Any:
        return (
            get_write_client()
            .table(table)
            .select("seq")
            .eq(key_field, str(key_value))
            .order("seq", desc=True)
            .limit(1)
            .execute()
        )

    response = await asyncio.to_thread(_call)
    data = getattr(response, "data", None) or []
    return int(data[0]["seq"]) + 1 if data else 0


def _is_seq_conflict(exc: BaseException) -> bool:
    """True for a UNIQUE(key, seq) violation — the signature of a concurrent event-seq
    race. Other unique violations (e.g. content-addressed hashes) are NOT retried, since
    they indicate a genuine duplicate rather than a seq race."""
    message = str(exc).lower()
    is_unique = "duplicate key" in message or "unique constraint" in message or "23505" in message
    return is_unique and "seq" in message


async def append_event_with_seq(
    table: str,
    key_field: str,
    key_value: Any,
    build_payload: Any,
    *,
    attempts: int = 5,
) -> dict[str, Any]:
    """Allocate the next event seq and insert atomically against concurrent appends.

    ``next_event_seq`` is read-max-then-insert, so two concurrent appends for the same
    key can pick the same seq; the DB ``UNIQUE(key, seq)`` constraint rejects the loser.
    This retries the loser (re-read seq, rebuild payload, re-insert) so both appends land
    instead of one crashing. The row is built identically to the legacy inline form
    (``event_id`` + ``schema_version`` + payload + ``anchored_hash`` over the payload),
    so audit hashes are unchanged. ``build_payload(seq)`` returns the payload dict.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        seq = await next_event_seq(table, key_field, key_value)
        payload = build_payload(seq)
        row = {
            "event_id": str(uuid4()),
            "schema_version": "1.0",
            **payload,
            "anchored_hash": canonical_hash(payload),
        }
        try:
            return await insert_row(table, row)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_seq_conflict(exc) and attempt < int(attempts):
                continue
            raise
    assert last_exc is not None  # pragma: no cover - loop always returns or raises
    raise last_exc


async def payment_ref_exists(block_hash: str, extrinsic_index: int) -> bool:
    row = await select_one(
        "research_loop_start_payments",
        columns="payment_id",
        filters=(("block_hash", block_hash), ("extrinsic_index", extrinsic_index)),
    )
    return row is not None


async def _existing_or_recovered_event(
    event_table: str,
    key_field: str,
    key_value: Any,
    create_opening_event: Any,
) -> dict[str, Any]:
    """Return an idempotency event, recreating seq=0 if a prior insert crashed."""
    event = await select_one(
        event_table,
        filters=((key_field, key_value), ("seq", 0)),
    )
    if event:
        return event
    existing_events = await select_many(
        event_table,
        filters=((key_field, key_value),),
        order_by=(("seq", False),),
        limit=1,
    )
    if existing_events:
        return existing_events[0]
    try:
        return await create_opening_event()
    except Exception:
        event = await select_one(
            event_table,
            filters=((key_field, key_value), ("seq", 0)),
        )
        if event:
            return event
        existing_events = await select_many(
            event_table,
            filters=((key_field, key_value),),
            order_by=(("seq", False),),
            limit=1,
        )
        if existing_events:
            return existing_events[0]
        raise


async def create_openrouter_key_ref(
    *,
    key_ref: str,
    miner_hotkey: str,
    key_hash: str,
    encrypted_key_ciphertext: str,
    kms_key_id: str,
    encryption_context_hash: str,
    preflight_doc: dict[str, Any],
    encrypted_management_key_ciphertext: str | None = None,
    management_key_hash: str | None = None,
    management_kms_key_id: str | None = None,
    management_encryption_context_hash: str | None = None,
    openrouter_workspace_hash: str | None = None,
    privacy_status: str = "not_configured",
    privacy_verified_at: str | None = None,
    privacy_proof_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = await select_one("research_lab_openrouter_key_refs", filters=(("key_ref", key_ref),))
    if existing:
        return existing
    row = {
        "key_ref": key_ref,
        "schema_version": "1.0",
        "miner_hotkey": miner_hotkey,
        "key_hash": key_hash,
        "encrypted_key_ciphertext": encrypted_key_ciphertext,
        "kms_key_id": kms_key_id,
        "encryption_context_hash": encryption_context_hash,
        "preflight_status": "passed",
        "preflight_doc": preflight_doc,
        "encrypted_management_key_ciphertext": encrypted_management_key_ciphertext,
        "management_key_hash": management_key_hash,
        "management_kms_key_id": management_kms_key_id,
        "management_encryption_context_hash": management_encryption_context_hash,
        "openrouter_workspace_hash": openrouter_workspace_hash,
        "privacy_status": privacy_status,
        "privacy_verified_at": privacy_verified_at,
        "privacy_proof_doc": privacy_proof_doc or {},
    }
    row["anchored_hash"] = canonical_hash(row)
    return await insert_row("research_lab_openrouter_key_refs", row)


def create_openrouter_privacy_proof_event_sync(
    *,
    key_ref: str,
    miner_hotkey: str,
    proof_status: str,
    proof_doc: dict[str, Any],
    run_id: str | None = None,
    stage: str = "",
) -> dict[str, Any]:
    row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        "key_ref": key_ref,
        "miner_hotkey": miner_hotkey,
        "run_id": run_id,
        "stage": str(stage or "unknown"),
        "proof_status": proof_status,
        "proof_doc": proof_doc,
        "created_at": now_iso(),
    }
    row["anchored_hash"] = canonical_hash(row)
    return insert_row_sync("research_lab_openrouter_privacy_proof_events", row)


async def create_ticket(request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ticket_id = deterministic_uuid("ticket", request.miner_hotkey, request.idempotency_key)
    existing_ticket = await select_one("research_loop_tickets", filters=(("ticket_id", ticket_id),))
    if existing_ticket:
        existing_event = await _existing_or_recovered_event(
            "research_loop_ticket_events",
            "ticket_id",
            ticket_id,
            lambda: create_ticket_event(
                ticket_id=ticket_id,
                event_type="opened",
                actor_hotkey=request.miner_hotkey,
                reason="ticket_created",
                event_doc={"ticket_hash": existing_ticket.get("ticket_hash")},
            ),
        )
        return existing_ticket, existing_event

    ticket_payload = {
        "ticket_id": ticket_id,
        "schema_version": "1.0",
        "miner_hotkey": request.miner_hotkey,
        "island": request.island,
        "brief_id": None,
        "brief_sanitized_ref": request.brief_sanitized_ref,
        "requested_loop_count": request.requested_loop_count,
        "ticket_status": "opened",
        "loop_start_fee_required_usd": request.loop_start_fee_required_usd,
        "loop_start_fee_payment_ref": None,
        "miner_openrouter_key_ref": request.miner_openrouter_key_ref,
        "miner_openrouter_key_handling": request.miner_openrouter_key_handling,
        "miner_openrouter_preflight_status": "not_run" if request.miner_openrouter_key_ref else None,
        "ticket_hash": "",
        "ticket_doc": {
            "idempotency_key_hash": canonical_hash(request.idempotency_key),
            "source": "gateway_research_lab_api",
            "brief_public_summary": getattr(request, "brief_public_summary", None),
            "research_model_tier": getattr(request, "research_model_tier", "default"),
            "requested_compute_budget_usd": float(getattr(request, "requested_compute_budget_usd", 5.0)),
            "max_compute_budget_usd": float(getattr(request, "max_compute_budget_usd", 25.0)),
            "budget_policy_version": "research-lab-budget:v1",
        },
    }
    ticket_payload["ticket_hash"] = canonical_hash(ticket_payload)
    ticket = await insert_row("research_loop_tickets", ticket_payload)
    event = await create_ticket_event(
        ticket_id=ticket_id,
        event_type="opened",
        actor_hotkey=request.miner_hotkey,
        reason="ticket_created",
        event_doc={"ticket_hash": ticket_payload["ticket_hash"]},
    )
    return ticket, event


async def create_ticket_event(
    *,
    ticket_id: str,
    event_type: str,
    actor_hotkey: str | None,
    reason: str,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_loop_ticket_events",
        "ticket_id",
        ticket_id,
        lambda seq: {
            "ticket_id": ticket_id,
            "seq": seq,
            "event_type": event_type,
            "actor_hotkey": actor_hotkey,
            "reason": reason,
            "event_doc": event_doc or {},
        },
    )


async def create_credit_event(
    *,
    credit_id: str,
    ticket_id: str,
    payment_id: str | None,
    payment_ref: str,
    miner_hotkey: str,
    event_type: str,
    credit_status: str,
    reason: str,
    consumed_by_loop_id: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_loop_start_credit_events",
        "credit_id",
        credit_id,
        lambda seq: {
            "credit_id": credit_id,
            "ticket_id": ticket_id,
            "payment_id": payment_id,
            "payment_ref": payment_ref,
            "miner_hotkey": miner_hotkey,
            "seq": seq,
            "event_type": event_type,
            "credit_status": credit_status,
            "reason": reason,
            "consumed_by_loop_id": consumed_by_loop_id,
            "event_doc": event_doc or {},
        },
    )


async def create_queue_event(
    *,
    run_id: str,
    ticket_id: str,
    event_type: str,
    queue_priority: int,
    reason: str,
    worker_ref: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_loop_run_queue_events",
        "run_id",
        run_id,
        lambda seq: {
            "run_id": run_id,
            "ticket_id": ticket_id,
            "seq": seq,
            "event_type": event_type,
            "queue_priority": queue_priority,
            "worker_ref": worker_ref,
            "reason": reason,
            "event_doc": event_doc or {},
        },
    )


async def create_gateway_control_event(
    *,
    control_key: str,
    event_type: str,
    control_status: str,
    reason: str,
    actor_ref: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_lab_gateway_control_events",
        "control_key",
        control_key,
        lambda seq: {
            "control_key": control_key,
            "seq": seq,
            "event_type": event_type,
            "control_status": control_status,
            "actor_ref": actor_ref,
            "reason": reason,
            "event_doc": event_doc or {},
        },
    )


async def create_loop_start_payment(
    *,
    ticket_id: str,
    payment_ref: str,
    block_hash: str,
    extrinsic_index: int,
    network: str,
    netuid: int,
    miner_hotkey: str,
    payment_info: dict[str, Any],
    required_usd: float,
    payment_kind: str = "loop_start",
    run_id: str | None = None,
    compute_budget_usd: float | None = None,
    extra_verification_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verification_doc = {
        "call_function": payment_info.get("call_function"),
        "amount_rao": payment_info.get("amount_rao"),
        "payment_kind": payment_kind,
    }
    if run_id:
        verification_doc["run_id"] = run_id
    if compute_budget_usd is not None:
        verification_doc["compute_budget_usd"] = float(compute_budget_usd)
    if extra_verification_doc:
        verification_doc.update(extra_verification_doc)
    row = {
        "payment_id": str(uuid4()),
        "schema_version": "1.0",
        "ticket_id": ticket_id,
        "payment_ref": payment_ref,
        "block_hash": block_hash,
        "extrinsic_index": extrinsic_index,
        "network": network,
        "netuid": netuid,
        "miner_hotkey": miner_hotkey,
        "miner_coldkey": payment_info.get("sender_coldkey"),
        "destination_wallet": payment_info.get("destination") or "",
        "required_usd": required_usd,
        "amount_tao": payment_info.get("amount_tao", 0.0),
        "amount_usd": payment_info.get("amount_usd", 0.0),
        "tao_price_usd": payment_info.get("tao_price_at_payment", 0.0),
        "payment_status": "verified",
        "verification_error": None,
        "verification_doc": verification_doc,
        "verified_at": now_iso(),
    }
    return await insert_row("research_loop_start_payments", row)


async def create_receipt(request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_payload = {
        "receipt_id": str(uuid4()),
        "schema_version": "1.0",
        "ticket_id": str(request.ticket_id),
        "trajectory_id": str(request.trajectory_id) if request.trajectory_id else None,
        "run_id": str(request.run_id) if request.run_id else None,
        "loop_start_payment_id": str(request.loop_start_payment_id) if request.loop_start_payment_id else None,
        "loop_start_credit_id": request.loop_start_credit_id,
        "miner_hotkey": request.miner_hotkey,
        "island": request.island,
        "receipt_status": request.receipt_status,
        "loop_count": request.loop_count,
        "miner_openrouter_key_ref": request.miner_openrouter_key_ref,
        "provider_usage": request.provider_usage,
        "cost_ledger": request.cost_ledger,
        "receipt_hash": "",
        "public_receipt_ref": request.public_receipt_ref,
        "receipt_doc": request.receipt_doc,
    }
    receipt_payload["receipt_hash"] = canonical_hash(receipt_payload)
    receipt = await insert_row("research_loop_receipts", receipt_payload)
    event = await create_receipt_event(
        receipt_id=receipt["receipt_id"],
        ticket_id=str(request.ticket_id),
        event_type=request.receipt_status,
        receipt_status=request.receipt_status,
        event_doc={"receipt_hash": receipt_payload["receipt_hash"]},
    )
    return receipt, event


async def find_queued_receipt_for_run(run_id: str) -> dict[str, Any] | None:
    rows = await select_many(
        "research_loop_receipt_current",
        filters=(("run_id", run_id),),
        order_by=(("current_status_at", True),),
        limit=10,
    )
    for row in rows:
        if str(row.get("current_receipt_status") or row.get("receipt_status") or "") == "queued":
            return row
    return None


async def create_receipt_event(
    *,
    receipt_id: str,
    ticket_id: str,
    event_type: str,
    receipt_status: str,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_loop_receipt_events",
        "receipt_id",
        receipt_id,
        lambda seq: {
            "receipt_id": receipt_id,
            "ticket_id": ticket_id,
            "seq": seq,
            "event_type": event_type,
            "receipt_status": receipt_status,
            "event_doc": event_doc or {},
        },
    )


async def create_candidate_artifact(request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = dict(request.private_model_manifest)
    patch = dict(request.candidate_patch_manifest)
    candidate_kind = str(getattr(request, "candidate_kind", "image_build") or "image_build")
    if candidate_kind != "image_build":
        raise ValueError("new Research Lab candidates must be image_build artifacts")
    candidate_model_manifest = dict(getattr(request, "candidate_model_manifest", None) or {})
    if not candidate_model_manifest:
        raise ValueError("image_build candidate requires candidate_model_manifest")
    candidate_artifact_hash = str(
        candidate_model_manifest.get("model_artifact_hash")
        or patch["candidate_artifact_hash"]
    )
    candidate_id = "candidate:" + candidate_artifact_hash.split(":", 1)[1]
    candidate_patch_hash = sha256_json(patch)
    candidate_model_manifest_hash = str(
        candidate_model_manifest.get("manifest_hash")
        or patch.get("candidate_model_manifest_hash")
        or ""
    )
    row = {
        "candidate_id": candidate_id,
        "schema_version": "1.0",
        "run_id": str(request.run_id),
        "ticket_id": str(request.ticket_id),
        "receipt_id": str(request.receipt_id) if request.receipt_id else None,
        "miner_hotkey": request.miner_hotkey,
        "island": request.island,
        "parent_artifact_hash": str(patch["parent_artifact_hash"]),
        "candidate_artifact_hash": candidate_artifact_hash,
        "private_model_manifest_hash": str(artifact["manifest_hash"]),
        "private_model_manifest_doc": artifact,
        "candidate_patch_hash": candidate_patch_hash,
        "candidate_patch_manifest": patch,
        "hypothesis_doc": dict(request.hypothesis_doc or {}),
        "redacted_public_summary": request.redacted_public_summary or "",
        "anchored_hash": "",
    }
    row.update(
        {
            "candidate_kind": "image_build",
            "candidate_model_manifest_hash": candidate_model_manifest_hash,
            "candidate_model_manifest_doc": candidate_model_manifest,
            "candidate_source_diff_hash": str(getattr(request, "candidate_source_diff_hash", "") or ""),
            "candidate_build_doc": dict(getattr(request, "candidate_build_doc", None) or {}),
        }
    )
    tree_id = str(getattr(request, "git_tree_id", "") or "")
    if tree_id:
        row.update(
            {
                "git_tree_id": tree_id,
                "git_tree_node_id": str(request.git_tree_node_id),
                "git_tree_root_commit": str(request.git_tree_root_commit),
                "git_tree_node_commit": str(request.git_tree_node_commit),
                "git_tree_lineage_hash": str(request.git_tree_lineage_hash),
            }
        )
    row["anchored_hash"] = canonical_hash(row)
    existing = await select_one(
        "research_lab_candidate_artifacts",
        filters=(("candidate_id", candidate_id),),
    )
    if existing:
        if tree_id and not _candidate_artifact_content_matches(existing, row):
            raise ValueError(
                "existing candidate artifact differs from current Git-tree root or content"
            )
        await _record_candidate_tree_handoff(request=request, candidate_id=candidate_id)
        event = await _existing_or_recovered_event(
            "research_lab_candidate_evaluation_events",
            "candidate_id",
            candidate_id,
            lambda: create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(request.run_id),
                ticket_id=str(request.ticket_id),
                event_type="queued",
                candidate_status="queued",
                reason="candidate_generated_by_gateway_worker",
                event_doc={
                    "candidate_artifact_hash": candidate_artifact_hash,
                    "candidate_patch_hash": candidate_patch_hash,
                },
            ),
        )
        return existing, event
    if tree_id:
        inserted = await _create_candidate_tree_handoff(
            request=request,
            candidate_id=candidate_id,
            candidate_row=row,
        )
    else:
        inserted = await insert_row("research_lab_candidate_artifacts", row)
    event = await create_candidate_evaluation_event(
        candidate_id=candidate_id,
        run_id=str(request.run_id),
        ticket_id=str(request.ticket_id),
        event_type="queued",
        candidate_status="queued",
        reason="candidate_generated_by_gateway_worker",
        event_doc={
            "candidate_artifact_hash": candidate_artifact_hash,
            "candidate_patch_hash": candidate_patch_hash,
        },
    )
    return inserted, event


async def _record_candidate_tree_handoff(
    *, request: Any, candidate_id: str
) -> None:
    tree_id = str(getattr(request, "git_tree_id", "") or "")
    if not tree_id:
        return
    await record_candidate_tree_handoff(
        tree_id=tree_id,
        run_id=str(request.run_id),
        candidate_id=str(candidate_id),
        node_id=str(request.git_tree_node_id),
        root_git_commit=str(request.git_tree_root_commit),
        node_git_commit=str(request.git_tree_node_commit),
        lineage_hash=str(request.git_tree_lineage_hash),
    )


async def _create_candidate_tree_handoff(
    *,
    request: Any,
    candidate_id: str,
    candidate_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist a selected candidate and its active-root handoff."""

    tree_id = str(request.git_tree_id)
    node_id = str(request.git_tree_node_id)
    root_git_commit = str(request.git_tree_root_commit)
    node_git_commit = str(request.git_tree_node_commit)
    lineage_hash = str(request.git_tree_lineage_hash)
    handoff_doc = {
        "schema_version": "research_lab.git_tree_candidate_handoff.v1",
        "tree_id": tree_id,
        "run_id": str(request.run_id),
        "candidate_id": str(candidate_id),
        "node_id": node_id,
        "root_git_commit": root_git_commit,
        "node_git_commit": node_git_commit,
        "lineage_hash": lineage_hash,
    }
    handoff_hash = sha256_json(handoff_doc)
    completion_doc = {
        "schema_version": "research_lab.git_tree_completed.v1",
        "tree_id": tree_id,
        "run_id": str(request.run_id),
        "candidate_id": str(candidate_id),
        "node_id": node_id,
        "handoff_hash": handoff_hash,
        "paid_finalist_count": 1,
    }
    last_error: BaseException | None = None
    for _attempt in range(5):
        current = await select_one(
            "research_lab_autoresearch_tree_current",
            columns=(
                "tree_id,current_event_type,current_event_doc,current_event_hash"
            ),
            filters=(("tree_id", tree_id),),
        )
        if not current:
            raise RuntimeError("Git-tree candidate handoff tree is missing")
        current_type = str(current.get("current_event_type") or "")
        if current_type in {
            "tree_completed",
            "tree_failed",
            "tree_cancelled_root_changed",
        }:
            raise RuntimeError(
                f"Git-tree candidate handoff targets terminal tree {current_type}"
            )
        previous_event_hash = str(
            current.get("current_event_hash") or "sha256:" + "0" * 64
        )
        completed_event_hash = sha256_json(
            {
                "schema_version": "research_lab.git_tree_event.v1",
                "tree_id": tree_id,
                "event_type": "tree_completed",
                "node_id": node_id,
                "previous_event_hash": previous_event_hash,
                "event_doc": completion_doc,
            }
        )
        try:
            data = await call_rpc(
                "create_research_lab_git_tree_candidate_handoff",
                {
                    "requested_candidate_doc": dict(candidate_row),
                    "requested_tree_id": tree_id,
                    "requested_run_id": str(request.run_id),
                    "requested_candidate_id": str(candidate_id),
                    "requested_node_id": node_id,
                    "requested_root_git_commit": root_git_commit,
                    "requested_node_git_commit": node_git_commit,
                    "requested_lineage_hash": lineage_hash,
                    "requested_handoff_doc": handoff_doc,
                    "requested_handoff_hash": handoff_hash,
                    "requested_previous_event_hash": previous_event_hash,
                    "requested_completed_event_hash": completed_event_hash,
                },
            )
        except Exception as exc:
            last_error = exc
            candidate = await select_one(
                "research_lab_candidate_artifacts",
                filters=(("candidate_id", str(candidate_id)),),
            )
            handoff = await select_one(
                "research_lab_autoresearch_tree_handoffs",
                columns="tree_id,run_id,candidate_id,handoff_hash",
                filters=(("tree_id", tree_id),),
            )
            if (
                isinstance(candidate, Mapping)
                and _candidate_artifact_content_matches(
                    candidate,
                    candidate_row,
                )
                and isinstance(handoff, Mapping)
                and str(handoff.get("run_id") or "") == str(request.run_id)
                and str(handoff.get("candidate_id") or "") == str(candidate_id)
                and str(handoff.get("handoff_hash") or "") == handoff_hash
            ):
                return dict(candidate)
            message = str(exc).lower()
            if "stale_active_root" in message:
                raise
            if "40001" not in message and "event_conflict" not in message:
                raise
            continue
        if isinstance(data, list):
            data = data[0] if data else None
        candidate = data.get("candidate") if isinstance(data, Mapping) else None
        handoff = data.get("handoff") if isinstance(data, Mapping) else None
        if (
            not isinstance(candidate, Mapping)
            or str(candidate.get("candidate_id") or "") != str(candidate_id)
            or not isinstance(handoff, Mapping)
            or not isinstance(handoff.get("handoff"), Mapping)
            or not isinstance(handoff.get("completion_event"), Mapping)
        ):
            raise RuntimeError(
                "Git-tree candidate creation returned no durable handoff"
            )
        return dict(candidate)
    raise RuntimeError(
        "Git-tree candidate creation conflicted after retries"
    ) from last_error


def _candidate_artifact_content_matches(
    persisted: Mapping[str, Any],
    requested: Mapping[str, Any],
) -> bool:
    """Verify timeout recovery cannot accept another candidate's content."""

    return all(
        persisted.get(field) == requested.get(field)
        for field in (
            "candidate_artifact_hash",
            "candidate_model_manifest_hash",
            "candidate_model_manifest_doc",
            "candidate_source_diff_hash",
            "candidate_patch_hash",
            "private_model_manifest_hash",
        )
    )


async def record_candidate_tree_handoff(
    *,
    tree_id: str,
    run_id: str,
    candidate_id: str,
    node_id: str,
    root_git_commit: str,
    node_git_commit: str,
    lineage_hash: str,
) -> None:
    """Atomically bind one selected tree node to its scoring candidate."""

    handoff_doc = {
        "schema_version": "research_lab.git_tree_candidate_handoff.v1",
        "tree_id": tree_id,
        "run_id": str(run_id),
        "candidate_id": str(candidate_id),
        "node_id": str(node_id),
        "root_git_commit": str(root_git_commit),
        "node_git_commit": str(node_git_commit),
        "lineage_hash": str(lineage_hash),
    }
    handoff_hash = sha256_json(handoff_doc)
    completion_doc = {
        "schema_version": "research_lab.git_tree_completed.v1",
        "tree_id": tree_id,
        "run_id": str(run_id),
        "candidate_id": str(candidate_id),
        "node_id": str(node_id),
        "handoff_hash": handoff_hash,
        "paid_finalist_count": 1,
    }
    last_error: BaseException | None = None
    for _attempt in range(5):
        current = await select_one(
            "research_lab_autoresearch_tree_current",
            columns=(
                "tree_id,current_event_type,current_event_doc,current_event_hash"
            ),
            filters=(("tree_id", tree_id),),
        )
        if not current:
            raise RuntimeError("Git-tree candidate handoff tree is missing")
        current_type = str(current.get("current_event_type") or "")
        if current_type == "tree_completed":
            if dict(current.get("current_event_doc") or {}) != completion_doc:
                raise RuntimeError("Git-tree candidate completion differs")
            return
        if current_type in {"tree_failed", "tree_cancelled_root_changed"}:
            raise RuntimeError("Git-tree candidate handoff targets a terminal tree")
        previous_event_hash = str(
            current.get("current_event_hash") or "sha256:" + "0" * 64
        )
        completed_event_hash = sha256_json(
            {
                "schema_version": "research_lab.git_tree_event.v1",
                "tree_id": tree_id,
                "event_type": "tree_completed",
                "node_id": str(node_id),
                "previous_event_hash": previous_event_hash,
                "event_doc": completion_doc,
            }
        )
        try:
            data = await call_rpc(
                "record_research_lab_autoresearch_tree_handoff",
                {
                    "requested_tree_id": tree_id,
                    "requested_run_id": str(run_id),
                    "requested_candidate_id": str(candidate_id),
                    "requested_node_id": str(node_id),
                    "requested_root_git_commit": str(root_git_commit),
                    "requested_node_git_commit": str(node_git_commit),
                    "requested_lineage_hash": str(lineage_hash),
                    "requested_handoff_doc": handoff_doc,
                    "requested_handoff_hash": handoff_hash,
                    "requested_previous_event_hash": previous_event_hash,
                    "requested_completed_event_hash": completed_event_hash,
                },
            )
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "40001" not in message and "event_conflict" not in message:
                raise
            continue
        if isinstance(data, list):
            data = data[0] if data else None
        if (
            not isinstance(data, Mapping)
            or not isinstance(data.get("handoff"), Mapping)
            or not isinstance(data.get("completion_event"), Mapping)
        ):
            raise RuntimeError("Git-tree candidate handoff returned no row")
        return
    assert last_error is not None
    raise last_error


async def create_auto_research_loop_event(
    *,
    run_id: str,
    ticket_id: str,
    event_type: str,
    loop_status: str,
    worker_ref: str,
    receipt_id: str | None = None,
    node_id: str | None = None,
    elapsed_seconds: float = 0.0,
    candidate_artifact_hash: str | None = None,
    candidate_patch_hash: str | None = None,
    provider_usage: list[dict[str, Any]] | None = None,
    cost_ledger: dict[str, Any] | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_lab_auto_research_loop_events",
        "run_id",
        run_id,
        lambda seq: {
            "run_id": run_id,
            "ticket_id": ticket_id,
            "receipt_id": receipt_id,
            "seq": seq,
            "event_type": event_type,
            "loop_status": loop_status,
            "node_id": node_id,
            "worker_ref": worker_ref,
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "candidate_artifact_hash": candidate_artifact_hash,
            "candidate_patch_hash": candidate_patch_hash,
            "provider_usage": provider_usage or [],
            "cost_ledger": cost_ledger or {},
            "event_doc": event_doc or {},
        },
    )


async def latest_auto_research_checkpoint(run_id: str) -> dict[str, Any] | None:
    rows = await select_many(
        "research_lab_auto_research_loop_events",
        filters=(("run_id", run_id), ("event_type", "checkpoint_saved")),
        order_by=(("seq", True),),
        limit=1,
    )
    if not rows:
        return None
    event_doc = rows[0].get("event_doc")
    if isinstance(event_doc, dict) and isinstance(event_doc.get("checkpoint"), dict):
        return dict(event_doc["checkpoint"])
    return None


async def create_rolling_icp_window(window: Any) -> dict[str, Any]:
    """Persist the public hash/ref side of a Research Lab rolling ICP window."""
    public_doc = dict(window.public_doc if hasattr(window, "public_doc") else window)
    rolling_window_hash = str(public_doc["rolling_window_hash"])
    existing = await select_one(
        "research_lab_rolling_icp_windows",
        filters=(("rolling_window_hash", rolling_window_hash),),
    )
    if existing:
        return existing
    payload = {
        "rolling_window_hash": rolling_window_hash,
        "required_days": int(public_doc.get("required_days", 10)),
        "icps_per_day": int(public_doc.get("icps_per_day", 5)),
        "selected_set_count": int(public_doc.get("selected_set_count", 0)),
        "selected_icp_count": int(public_doc.get("selected_icp_count", 0)),
        "window_doc": public_doc,
    }
    row = {
        "schema_version": str(public_doc.get("schema_version") or "1.0"),
        **payload,
        "window_mode": public_doc.get("window_mode"),
        "selection_policy": public_doc.get("selection_policy"),
        "fresh_set_id": public_doc.get("fresh_set_id"),
        "fresh_icp_count": public_doc.get("fresh_icp_count"),
        "retained_icp_count": public_doc.get("retained_icp_count"),
        "min_new_icp_count": public_doc.get("min_new_icp_count"),
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_lab_rolling_icp_windows", row)


async def create_scoring_dispatch_event(
    *,
    dispatch_type: str,
    dispatch_status: str,
    worker_ref: str,
    proxy_ref_hash: str | None = None,
    candidate_id: str | None = None,
    run_id: str | None = None,
    ticket_id: str | None = None,
    rolling_window_hash: str | None = None,
    score_bundle_id: str | None = None,
    benchmark_bundle_id: str | None = None,
    scoring_id: str | None = None,
    scoring_run_id: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "dispatch_type": dispatch_type,
        "dispatch_status": dispatch_status,
        "candidate_id": candidate_id,
        "run_id": run_id,
        "ticket_id": ticket_id,
        "rolling_window_hash": rolling_window_hash,
        "score_bundle_id": score_bundle_id,
        "benchmark_bundle_id": benchmark_bundle_id,
        "worker_ref": worker_ref,
        "proxy_ref_hash": proxy_ref_hash,
        "event_doc": event_doc or {},
    }
    legacy_anchor_payload = dict(payload)
    # Migration 83 columns are opt-in: omit them entirely for legacy writers
    # so code can be deployed disabled without requiring schema cache refresh.
    if scoring_id or scoring_run_id:
        if not scoring_id or not scoring_run_id:
            raise ValueError("scoring dispatch telemetry ids must be provided together")
        payload["scoring_id"] = scoring_id
        payload["scoring_run_id"] = scoring_run_id
    dispatch_event_id = str(uuid4())
    row = {
        "dispatch_event_id": dispatch_event_id,
        "schema_version": "1.0",
        **payload,
        # Preserve the existing dispatch/audit anchor contract. The additive
        # telemetry link is independently append-only, FK-constrained, and
        # hash-anchored in the V2 tables; enabling or degrading telemetry must
        # not perturb legacy audit bundle inputs.
        "anchored_hash": canonical_hash(
            scoring_dispatch_event_anchor_payload(
                legacy_anchor_payload,
                dispatch_event_id,
            )
        ),
    }
    return await insert_row("research_lab_scoring_dispatch_events", row)


async def create_private_model_benchmark_bundle(
    *,
    benchmark_date: str,
    private_model_artifact_hash: str,
    private_model_manifest_hash: str,
    rolling_window_hash: str,
    evaluation_epoch: int,
    benchmark_attempt: int = 0,
    benchmark_quality: str = "passed",
    aggregate_score: float,
    scoring_worker_ref: str,
    proxy_ref_hash: str | None,
    signature_ref: str,
    score_summary_doc: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema_version = str(score_summary_doc.get("schema_version") or "1.0")
    if schema_version not in {"1.0", "1.1"}:
        raise ValueError(f"unsupported private benchmark schema version: {schema_version}")
    payload = {
        "benchmark_date": benchmark_date,
        "private_model_artifact_hash": private_model_artifact_hash,
        "private_model_manifest_hash": private_model_manifest_hash,
        "rolling_window_hash": rolling_window_hash,
        "evaluation_epoch": int(evaluation_epoch),
        "benchmark_attempt": int(benchmark_attempt),
        "benchmark_quality": str(benchmark_quality),
        "aggregate_score": float(aggregate_score),
        "scoring_worker_ref": scoring_worker_ref,
        "proxy_ref_hash": proxy_ref_hash,
        "signature_ref": signature_ref,
        "score_summary_doc": score_summary_doc,
    }
    benchmark_bundle_hash = canonical_hash(payload)
    benchmark_bundle_id = "private_benchmark:" + benchmark_bundle_hash.split(":", 1)[1]
    existing = await select_one(
        "research_lab_private_model_benchmark_bundles",
        filters=(("benchmark_bundle_id", benchmark_bundle_id),),
    )
    if existing:
        event = await _existing_or_recovered_event(
            "research_lab_private_model_benchmark_events",
            "benchmark_bundle_id",
            benchmark_bundle_id,
            lambda: create_private_model_benchmark_event(
                benchmark_bundle_id=benchmark_bundle_id,
                event_type="completed",
                benchmark_status="completed",
                event_doc={
                    "benchmark_bundle_hash": benchmark_bundle_hash,
                    "rolling_window_hash": rolling_window_hash,
                },
            ),
        )
        return existing, event
    row = {
        "benchmark_bundle_id": benchmark_bundle_id,
        "schema_version": schema_version,
        **payload,
        "benchmark_bundle_hash": benchmark_bundle_hash,
        "anchored_hash": benchmark_bundle_hash,
    }
    inserted = await insert_row("research_lab_private_model_benchmark_bundles", row)
    event = await create_private_model_benchmark_event(
        benchmark_bundle_id=benchmark_bundle_id,
        event_type="completed",
        benchmark_status="completed",
        event_doc={
            "benchmark_bundle_hash": benchmark_bundle_hash,
            "rolling_window_hash": rolling_window_hash,
        },
    )
    return inserted, event


async def create_private_model_benchmark_event(
    *,
    benchmark_bundle_id: str,
    event_type: str,
    benchmark_status: str,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_lab_private_model_benchmark_events",
        "benchmark_bundle_id",
        benchmark_bundle_id,
        lambda seq: {
            "benchmark_bundle_id": benchmark_bundle_id,
            "seq": seq,
            "event_type": event_type,
            "benchmark_status": benchmark_status,
            "event_doc": event_doc or {},
        },
    )


async def create_private_model_version(
    *,
    artifact_manifest: dict[str, Any],
    manifest_uri: str | None = None,
    source_candidate_id: str | None = None,
    source_score_bundle_id: str | None = None,
    source_benchmark_bundle_id: str | None = None,
    redacted_version_doc: dict[str, Any] | None = None,
    version_status: str = "active",
    reason: str = "private_model_version_registered",
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model_artifact_hash": str(artifact_manifest["model_artifact_hash"]),
        "private_model_manifest_hash": str(artifact_manifest["manifest_hash"]),
        "private_model_manifest_uri": str(manifest_uri or artifact_manifest["manifest_uri"]),
        "git_commit_sha": str(artifact_manifest["git_commit_sha"]),
        "config_hash": str(artifact_manifest["config_hash"]),
        "component_registry_version": str(artifact_manifest["component_registry_version"]),
        "scoring_adapter_version": str(artifact_manifest["scoring_adapter_version"]),
        "source_candidate_id": source_candidate_id,
        "source_score_bundle_id": source_score_bundle_id,
        "source_benchmark_bundle_id": source_benchmark_bundle_id,
        "signature_ref": str(artifact_manifest["signature_ref"]),
        "build_id": artifact_manifest.get("build_id"),
        "redacted_version_doc": redacted_version_doc or {},
    }
    version_hash = canonical_hash(payload)
    version_id = "private_model_version:" + version_hash
    existing = await select_one(
        "research_lab_private_model_versions",
        filters=(("private_model_version_id", version_id),),
    )
    reused_by_artifact_hash = False
    if not existing:
        existing = await select_one(
            "research_lab_private_model_versions",
            filters=(("model_artifact_hash", payload["model_artifact_hash"]),),
        )
        if existing:
            mismatches = []
            for field in (
                "private_model_manifest_hash",
                "git_commit_sha",
                "config_hash",
                "component_registry_version",
                "scoring_adapter_version",
                "signature_ref",
            ):
                existing_value = str(existing.get(field) or "")
                payload_value = str(payload.get(field) or "")
                if existing_value != payload_value:
                    mismatches.append(field)
            if mismatches:
                raise RuntimeError(
                    "research_lab_private_model_versions: existing model_artifact_hash "
                    f"{payload['model_artifact_hash']} conflicts on {', '.join(mismatches)}"
                )
            reused_by_artifact_hash = True
    if existing:
        event = await create_private_model_version_event(
            private_model_version_id=str(existing["private_model_version_id"]),
            event_type=version_status,
            version_status=version_status,
            reason=reason,
            event_doc={
                "version_hash": version_hash,
                "requested_private_model_version_id": version_id,
                "reused_existing_model_artifact_hash": reused_by_artifact_hash,
            },
        )
        return existing, event
    row = {
        "private_model_version_id": version_id,
        "schema_version": "1.0",
        **payload,
        "version_hash": version_hash,
        "anchored_hash": version_hash,
    }
    inserted = await insert_row("research_lab_private_model_versions", row)
    event = await create_private_model_version_event(
        private_model_version_id=version_id,
        event_type=version_status,
        version_status=version_status,
        reason=reason,
        event_doc={"version_hash": version_hash},
    )
    return inserted, event


async def create_private_model_version_event(
    *,
    private_model_version_id: str,
    event_type: str,
    version_status: str,
    reason: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_lab_private_model_version_events",
        "private_model_version_id",
        private_model_version_id,
        lambda seq: {
            "private_model_version_id": private_model_version_id,
            "seq": seq,
            "event_type": event_type,
            "version_status": version_status,
            "reason": reason,
            "event_doc": event_doc or {},
        },
    )


async def create_candidate_promotion_event(
    *,
    candidate_id: str,
    event_type: str,
    promotion_status: str,
    derived_candidate_id: str | None = None,
    source_score_bundle_id: str | None = None,
    derived_score_bundle_id: str | None = None,
    private_model_version_id: str | None = None,
    active_parent_artifact_hash: str | None = None,
    candidate_parent_artifact_hash: str | None = None,
    rolling_window_hash: str | None = None,
    improvement_points: float = 0.0,
    threshold_points: float = 1.0,
    worker_ref: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate_id,
        "derived_candidate_id": derived_candidate_id,
        "source_score_bundle_id": source_score_bundle_id,
        "derived_score_bundle_id": derived_score_bundle_id,
        "private_model_version_id": private_model_version_id,
        "event_type": event_type,
        "promotion_status": promotion_status,
        "active_parent_artifact_hash": active_parent_artifact_hash,
        "candidate_parent_artifact_hash": candidate_parent_artifact_hash,
        "rolling_window_hash": rolling_window_hash,
        "improvement_points": float(improvement_points),
        "threshold_points": float(threshold_points),
        "worker_ref": worker_ref,
        "event_doc": event_doc or {},
    }
    row = {
        "promotion_event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_lab_candidate_promotion_events", row)


def public_loop_card_id(ticket_id: str) -> str:
    return f"public_loop_card:{ticket_id}"


def public_loop_card_event_ref(*parts: Any) -> str:
    return "public_loop_card_event:" + canonical_hash(parts).split(":", 1)[1]


async def create_public_loop_card(
    *,
    ticket_id: str,
    miner_hotkey: str,
    research_area: str,
    research_focus_summary: str,
    topic_tags: list[str],
    topic_signature_hash: str,
    card_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    card_id = public_loop_card_id(ticket_id)
    existing = await select_one("research_lab_public_loop_cards", filters=(("card_id", card_id),))
    if existing:
        return existing
    payload = {
        "card_id": card_id,
        "schema_version": "1.0",
        "ticket_id": ticket_id,
        "miner_hotkey": miner_hotkey,
        "research_area": research_area,
        "research_focus_summary": research_focus_summary,
        "topic_tags": topic_tags,
        "topic_signature_hash": topic_signature_hash,
        "card_doc": card_doc or {},
    }
    payload["card_hash"] = canonical_hash(payload)
    payload["anchored_hash"] = payload["card_hash"]
    return await insert_row("research_lab_public_loop_cards", payload)


async def create_public_loop_card_event(
    *,
    event_ref: str,
    card_id: str,
    ticket_id: str,
    event_type: str,
    outcome_label: str,
    outcome_band: str,
    topic_tags: list[str],
    topic_signature_hash: str,
    run_id: str | None = None,
    receipt_id: str | None = None,
    candidate_count: int = 0,
    scored_candidate_count: int = 0,
    best_candidate_public_summary: str = "",
    last_activity_at: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = await select_one(
        "research_lab_public_loop_card_events",
        filters=(("event_ref", event_ref),),
    )
    if existing:
        return existing
    return await append_event_with_seq(
        "research_lab_public_loop_card_events",
        "card_id",
        card_id,
        lambda seq: {
            "event_ref": event_ref,
            "card_id": card_id,
            "ticket_id": ticket_id,
            "run_id": run_id,
            "receipt_id": receipt_id,
            "seq": seq,
            "event_type": event_type,
            "outcome_label": outcome_label,
            "outcome_band": outcome_band,
            "topic_tags": topic_tags,
            "topic_signature_hash": topic_signature_hash,
            "candidate_count": int(candidate_count),
            "scored_candidate_count": int(scored_candidate_count),
            "best_candidate_public_summary": best_candidate_public_summary,
            "last_activity_at": last_activity_at or now_iso(),
            "event_doc": event_doc or {},
        },
    )


async def create_private_repo_commit_event(
    *,
    commit_status: str,
    branch_name: str,
    candidate_id: str | None = None,
    score_bundle_id: str | None = None,
    private_model_version_id: str | None = None,
    git_commit_sha: str | None = None,
    private_repo_ref_hash: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate_id,
        "score_bundle_id": score_bundle_id,
        "private_model_version_id": private_model_version_id,
        "commit_status": commit_status,
        "git_commit_sha": git_commit_sha,
        "branch_name": branch_name,
        "private_repo_ref_hash": private_repo_ref_hash,
        "event_doc": event_doc or {},
    }
    row = {
        "commit_event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_lab_private_repo_commit_events", row)


async def create_public_benchmark_report(
    *,
    benchmark_date: str,
    benchmark_bundle_id: str,
    private_model_artifact_hash: str,
    private_model_manifest_hash: str,
    rolling_window_hash: str,
    aggregate_score: float,
    benchmark_attempt: int = 0,
    benchmark_quality: str = "passed",
    report_doc: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "benchmark_date": benchmark_date,
        "benchmark_bundle_id": benchmark_bundle_id,
        "private_model_artifact_hash": private_model_artifact_hash,
        "private_model_manifest_hash": private_model_manifest_hash,
        "rolling_window_hash": rolling_window_hash,
        "aggregate_score": float(aggregate_score),
        "benchmark_attempt": int(benchmark_attempt),
        "benchmark_quality": str(benchmark_quality),
        "report_doc": report_doc,
    }
    report_hash = canonical_hash(payload)
    report_id = "public_benchmark:" + report_hash
    existing = await select_one(
        "research_lab_public_benchmark_reports",
        filters=(("report_id", report_id),),
    )
    if existing:
        event = await _existing_or_recovered_event(
            "research_lab_public_benchmark_report_events",
            "report_id",
            report_id,
            lambda: create_public_benchmark_report_event(
                report_id=report_id,
                event_type="published",
                report_status="published",
                event_doc={"report_hash": report_hash, "benchmark_bundle_id": benchmark_bundle_id},
            ),
        )
        return existing, event
    row = {
        "report_id": report_id,
        "schema_version": "1.0",
        **payload,
        "report_hash": report_hash,
        "anchored_hash": report_hash,
    }
    inserted = await insert_row("research_lab_public_benchmark_reports", row)
    event = await create_public_benchmark_report_event(
        report_id=report_id,
        event_type="published",
        report_status="published",
        event_doc={"report_hash": report_hash, "benchmark_bundle_id": benchmark_bundle_id},
    )
    return inserted, event


async def create_public_benchmark_report_event(
    *,
    report_id: str,
    event_type: str,
    report_status: str,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_lab_public_benchmark_report_events",
        "report_id",
        report_id,
        lambda seq: {
            "report_id": report_id,
            "seq": seq,
            "event_type": event_type,
            "report_status": report_status,
            "event_doc": event_doc or {},
        },
    )


def _champion_reward_corpus_refs(run_id: str) -> dict[str, str]:
    """P18: explicit trajectory/execution-trace refs for a reward's run.

    Lazy import — the projector imports this store module, so a top-level
    import would cycle. Never raises (refs are additive metadata)."""
    try:
        from gateway.research_lab.trajectory_projector import (
            execution_trace_id_for_run,
            trajectory_id_for_run,
        )

        return {
            "trajectory_ref": f"trajectory:{trajectory_id_for_run(run_id)}",
            "execution_trace_ref": f"execution_trace:{execution_trace_id_for_run(run_id)}",
        }
    except Exception:  # noqa: BLE001 - additive metadata only
        return {}


async def create_champion_reward_obligation(
    *,
    obligation: dict[str, Any],
    ticket_id: str | None = None,
    obligation_doc: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = await select_one(
        "research_lab_champion_reward_obligations",
        filters=(("champion_reward_id", obligation["champion_reward_id"]),),
    )
    if existing:
        event = await _existing_or_recovered_event(
            "research_lab_champion_reward_events",
            "champion_reward_id",
            obligation["champion_reward_id"],
            lambda: create_champion_reward_event(
                champion_reward_id=obligation["champion_reward_id"],
                event_type="active",
                reward_status="active",
                reason="created_from_gateway_promotion_event",
                event_doc={
                    "candidate_id": obligation.get("candidate_id"),
                    "score_bundle_id": obligation.get("score_bundle_id"),
                },
            ),
        )
        return existing, event
    row = {
        "champion_reward_id": obligation["champion_reward_id"],
        "schema_version": "1.0",
        "score_bundle_id": obligation.get("score_bundle_id") or None,
        "candidate_id": obligation.get("candidate_id") or None,
        "run_id": obligation["run_id"],
        "ticket_id": ticket_id,
        "miner_hotkey": obligation["miner_hotkey"],
        "miner_uid": int(obligation["uid"]),
        "island": obligation["island"],
        "policy_id": str((obligation_doc or {}).get("policy_id") or "research-lab-promotion-v1"),
        "evaluation_epoch": int(obligation["evaluation_epoch"]),
        "start_epoch": int(obligation["start_epoch"]),
        "epoch_count": int(obligation["epoch_count"]),
        "improvement_points": float(obligation["improvement_points"]),
        "threshold_points": float(obligation["threshold_points"]),
        "desired_alpha_percent": float(obligation["desired_alpha_percent"]),
        "source_score_bundle_hash": (obligation_doc or {}).get("source_score_bundle_hash"),
        "input_hash": obligation["input_hash"],
        "anchored_hash": obligation["anchored_hash"],
        "obligation_doc": {
            **(obligation_doc or {}),
            # P18 (trajectoryimprovements.md): reward → trajectory resolves via
            # explicit refs instead of re-deriving the deterministic uuid5s.
            "corpus_refs": _champion_reward_corpus_refs(str(obligation["run_id"])),
        },
    }
    inserted = await insert_row("research_lab_champion_reward_obligations", row)
    event = await create_champion_reward_event(
        champion_reward_id=obligation["champion_reward_id"],
        event_type="active",
        reward_status="active",
        reason="created_from_gateway_promotion_event",
        event_doc={"candidate_id": obligation.get("candidate_id"), "score_bundle_id": obligation.get("score_bundle_id")},
    )
    return inserted, event


async def create_champion_reward_event(
    *,
    champion_reward_id: str,
    event_type: str,
    reward_status: str,
    reason: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_lab_champion_reward_events",
        "champion_reward_id",
        champion_reward_id,
        lambda seq: {
            "champion_reward_id": champion_reward_id,
            "seq": seq,
            "event_type": event_type,
            "reward_status": reward_status,
            "reason": reason,
            "event_doc": event_doc or {},
        },
    )


async def create_research_lab_emission_allocation_snapshot(
    *,
    epoch: int,
    netuid: int,
    policy_id: str,
    snapshot_status: str,
    allocation_doc: dict[str, Any],
) -> dict[str, Any]:
    allocation_hash = str(allocation_doc["allocation_hash"])
    allocation_id = "lab_allocation:" + allocation_hash
    existing = await select_one(
        "research_lab_emission_allocation_snapshots",
        filters=(("allocation_id", allocation_id),),
    )
    if existing:
        return existing
    row = {
        "allocation_id": allocation_id,
        "schema_version": "1.0",
        "epoch": int(epoch),
        "netuid": int(netuid),
        "policy_id": str(policy_id),
        "snapshot_status": str(snapshot_status),
        "lab_cap_alpha_percent": float(allocation_doc.get("lab_cap_percent") or 0.0),
        "source_add_alpha_percent": float(allocation_doc.get("source_add_alpha_percent") or 0.0),
        "reimbursement_alpha_percent": float(allocation_doc.get("reimbursement_alpha_percent") or 0.0),
        "champion_alpha_percent": float(allocation_doc.get("champion_alpha_percent") or 0.0),
        "queued_champion_alpha_percent": float(allocation_doc.get("queued_champion_alpha_percent") or 0.0),
        "unallocated_alpha_percent": float(allocation_doc.get("unallocated_percent") or 0.0),
        "input_hash": str(allocation_doc.get("input_hash") or ""),
        "allocation_hash": allocation_hash,
        "allocation_doc": allocation_doc,
    }
    return await insert_row("research_lab_emission_allocation_snapshots", row)


async def create_signed_audit_bundle(
    *,
    epoch: int,
    bundle_doc: dict[str, Any],
    signature_ref: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_bundle_hash = canonical_hash(bundle_doc)
    audit_bundle_id = "research_lab_audit:" + audit_bundle_hash.split(":", 1)[1]
    existing = await select_one(
        "research_lab_signed_audit_bundles",
        filters=(("audit_bundle_id", audit_bundle_id),),
    )
    if existing:
        event = await _existing_or_recovered_event(
            "research_lab_signed_audit_bundle_events",
            "audit_bundle_id",
            audit_bundle_id,
            lambda: create_signed_audit_bundle_event(
                audit_bundle_id=audit_bundle_id,
                event_type="created",
                audit_status="created",
                event_doc={"audit_bundle_hash": audit_bundle_hash},
            ),
        )
        return existing, event
    payload = {
        "audit_bundle_id": audit_bundle_id,
        "epoch": int(epoch),
        "audit_bundle_hash": audit_bundle_hash,
        "signature_ref": signature_ref,
        "bundle_doc": bundle_doc,
    }
    row = {
        "schema_version": "1.0",
        **payload,
        "anchored_hash": audit_bundle_hash,
    }
    inserted = await insert_row("research_lab_signed_audit_bundles", row)
    event = await create_signed_audit_bundle_event(
        audit_bundle_id=audit_bundle_id,
        event_type="created",
        audit_status="created",
        event_doc={"audit_bundle_hash": audit_bundle_hash},
    )
    return inserted, event


async def create_signed_audit_bundle_event(
    *,
    audit_bundle_id: str,
    event_type: str,
    audit_status: str,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_lab_signed_audit_bundle_events",
        "audit_bundle_id",
        audit_bundle_id,
        lambda seq: {
            "audit_bundle_id": audit_bundle_id,
            "seq": seq,
            "event_type": event_type,
            "audit_status": audit_status,
            "event_doc": event_doc or {},
        },
    )


async def create_arweave_epoch_audit_anchor(
    *,
    epoch: int,
    netuid: int,
    audit_kind: str,
    audit_bundle_id: str | None,
    audit_bundle_hash: str | None,
    allocation_hash: str | None,
    weights_hash: str | None,
    payload_hash: str,
    transparency_event_hash: str | None,
    tee_sequence: int | None,
    event_doc: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_payload = {
        "epoch": int(epoch),
        "netuid": int(netuid),
        "audit_kind": audit_kind,
        "payload_hash": payload_hash,
    }
    anchor_hash = canonical_hash(identity_payload)
    anchor_id = "research_lab_arweave_anchor:" + anchor_hash.split(":", 1)[1]
    existing = await select_one(
        "research_lab_arweave_epoch_audit_anchors",
        filters=(("anchor_id", anchor_id),),
    )
    payload = {
        "epoch": int(epoch),
        "netuid": int(netuid),
        "audit_kind": audit_kind,
        "audit_bundle_id": audit_bundle_id,
        "audit_bundle_hash": audit_bundle_hash,
        "allocation_hash": allocation_hash,
        "weights_hash": weights_hash,
        "payload_hash": payload_hash,
        "transparency_event_hash": transparency_event_hash,
        "tee_sequence": tee_sequence,
    }
    if existing:
        event = await select_one(
            "research_lab_arweave_epoch_audit_anchor_events",
            filters=(("anchor_id", anchor_id), ("event_type", "buffered")),
        )
        if event:
            return existing, event
        event = await create_arweave_epoch_audit_anchor_event(
            anchor_id=anchor_id,
            event_type="buffered",
            anchor_status="buffered",
            event_doc=event_doc or {},
        )
        return existing, event

    row = {
        "anchor_id": anchor_id,
        "schema_version": "1.0",
        **payload,
        "anchor_hash": anchor_hash,
        "anchored_hash": anchor_hash,
    }
    inserted = await insert_row("research_lab_arweave_epoch_audit_anchors", row)
    await create_arweave_epoch_audit_anchor_event(
        anchor_id=anchor_id,
        event_type="created",
        anchor_status="created",
        event_doc={
            "payload_hash": payload_hash,
            "audit_kind": audit_kind,
        },
    )
    event = await create_arweave_epoch_audit_anchor_event(
        anchor_id=anchor_id,
        event_type="buffered",
        anchor_status="buffered",
        event_doc=event_doc or {},
    )
    return inserted, event


async def create_arweave_epoch_audit_anchor_event(
    *,
    anchor_id: str,
    event_type: str,
    anchor_status: str,
    transparency_event_hash: str | None = None,
    tee_sequence: int | None = None,
    checkpoint_number: int | None = None,
    checkpoint_merkle_root: str | None = None,
    arweave_tx_id: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc = event_doc or {}
    return await append_event_with_seq(
        "research_lab_arweave_epoch_audit_anchor_events",
        "anchor_id",
        anchor_id,
        lambda seq: {
            "anchor_id": anchor_id,
            "seq": seq,
            "event_type": event_type,
            "anchor_status": anchor_status,
            "transparency_event_hash": transparency_event_hash or doc.get("transparency_event_hash") or doc.get("event_hash"),
            "tee_sequence": tee_sequence if tee_sequence is not None else doc.get("tee_sequence"),
            "checkpoint_number": (
                checkpoint_number
                if checkpoint_number is not None
                else doc.get("checkpoint_number")
            ),
            "checkpoint_merkle_root": checkpoint_merkle_root or doc.get("checkpoint_merkle_root"),
            "arweave_tx_id": arweave_tx_id or doc.get("arweave_tx_id"),
            "event_doc": doc,
        },
    )


async def create_participation_snapshot(
    *,
    island: str,
    lookback_start: str,
    lookback_end: str,
    distinct_funded_hotkeys: int,
    paid_loop_count: int,
    unique_brief_count: int,
    participation_score: float,
    policy_id: str,
    snapshot_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "island": island,
        "lookback_start": lookback_start,
        "lookback_end": lookback_end,
        "distinct_funded_hotkeys": max(0, int(distinct_funded_hotkeys)),
        "paid_loop_count": max(0, int(paid_loop_count)),
        "unique_brief_count": max(0, int(unique_brief_count)),
        "source_add_count": 0,
        "red_team_count": 0,
        "participation_score": float(participation_score),
        "policy_id": policy_id,
        "snapshot_doc": snapshot_doc or {},
    }
    input_hash = canonical_hash(payload)
    existing = await select_one(
        "research_island_participation_snapshots",
        filters=(("input_hash", input_hash),),
    )
    if existing:
        return existing
    snapshot_ref = "research_island_participation_snapshot:" + input_hash.split(":", 1)[1]
    row = {
        "participation_snapshot_id": str(uuid4()),
        "schema_version": "1.0",
        "snapshot_ref": snapshot_ref,
        **payload,
        "input_hash": input_hash,
    }
    return await insert_row("research_island_participation_snapshots", row)


def _award_supplement_events_enabled() -> bool:
    return os.getenv(
        "RESEARCH_LAB_REIMBURSEMENT_SUPPLEMENT_EVENTS", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}


async def _maybe_record_award_supplement(
    existing: Mapping[str, Any], award: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Record post-resume spend that exceeds the recorded award target (bug #25).

    Awards are append-only (first terminal event wins the base row), so a
    resumed run's extra spend used to vanish entirely. The base row cannot be
    rewritten; instead a supplemental ``awarded`` event carries the reconcilable
    delta for operators/allocators to consume.
    """
    if not _award_supplement_events_enabled():
        return None
    try:
        existing_status = str(existing.get("award_status") or "")
        if existing_status != "awarded":
            return None
        existing_target = int(existing.get("target_reimbursement_microusd") or 0)
        new_target = int(award.get("target_reimbursement_microusd") or 0)
        if new_target <= existing_target:
            return None
        event = await create_reimbursement_award_event(
            award_id=str(existing["award_id"]),
            event_type="awarded",
            award_status="awarded",
            event_doc={
                "award_id": str(existing["award_id"]),
                "reason": "supplemental_spend_after_first_terminal_event",
                "previous_target_reimbursement_microusd": existing_target,
                "recomputed_target_reimbursement_microusd": new_target,
                "supplemental_target_reimbursement_microusd": new_target - existing_target,
                "target_reimbursement_microusd": new_target,
            },
        )
        logger.warning(
            "research_lab_reimbursement_supplement_recorded award_id=%s previous_microusd=%s new_microusd=%s",
            str(existing.get("award_id") or "")[:24],
            existing_target,
            new_target,
        )
        return event
    except Exception as exc:
        logger.warning(
            "research_lab_reimbursement_supplement_failed award_id=%s error=%s",
            str(existing.get("award_id") or "")[:24],
            str(exc)[:200],
        )
        return None


async def create_reimbursement_award(
    *,
    award: dict[str, Any],
    receipt_id: str | None,
    participation_snapshot_id: str | None,
    policy_id: str,
    award_doc: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = await select_one("research_reimbursement_awards", filters=(("award_id", award["award_id"]),))
    if not existing:
        existing = await select_one("research_reimbursement_awards", filters=(("run_id", str(award["run_id"])),))
    if existing:
        existing_award_id = str(existing["award_id"])
        event = await _existing_or_recovered_event(
            "research_reimbursement_award_events",
            "award_id",
            existing_award_id,
            lambda: create_reimbursement_award_event(
                award_id=existing_award_id,
                event_type=str(existing.get("award_status") or award["status"]),
                award_status=str(existing.get("award_status") or award["status"]),
                event_doc={
                    "award_id": existing_award_id,
                    "target_reimbursement_microusd": int(
                        existing.get("target_reimbursement_microusd")
                        or award["target_reimbursement_microusd"]
                    ),
                },
            ),
        )
        supplement_event = await _maybe_record_award_supplement(existing, award)
        return existing, supplement_event or event
    row = {
        "award_id": str(award["award_id"]),
        "schema_version": "1.0",
        "receipt_id": receipt_id,
        "participation_snapshot_id": participation_snapshot_id,
        "run_id": str(award["run_id"]),
        "miner_hotkey": str(award["miner_hotkey"]),
        "island": str(award["island"]),
        "run_day": str(award["run_day"]),
        "policy_id": policy_id,
        "award_status": str(award["status"]),
        "participation_score": float(award["participation_score"]),
        "participation_fraction": float(award["participation_fraction"]),
        "rebate_rate": float(award["rebate_rate"]),
        "eligible_cost_microusd": int(award["eligible_cost_microusd"]),
        "target_reimbursement_microusd": int(award["target_reimbursement_microusd"]),
        "reimbursement_epochs": int(award["reimbursement_epochs"]),
        "loop_start_fee_included": bool(award["loop_start_fee_included"]),
        "input_hash": str(award["input_hash"]),
        "award_doc": award_doc or award,
    }
    inserted = await insert_row("research_reimbursement_awards", row)
    event = await create_reimbursement_award_event(
        award_id=str(award["award_id"]),
        event_type=str(award["status"]),
        award_status=str(award["status"]),
        event_doc={"award_id": str(award["award_id"]), "target_reimbursement_microusd": int(award["target_reimbursement_microusd"])},
    )
    return inserted, event


async def create_reimbursement_award_event(
    *,
    award_id: str,
    event_type: str,
    award_status: str,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_reimbursement_award_events",
        "award_id",
        award_id,
        lambda seq: {
            "award_id": award_id,
            "seq": seq,
            "event_type": event_type,
            "award_status": award_status,
            "event_doc": event_doc or {},
        },
    )


async def create_reimbursement_schedule(
    *,
    schedule: dict[str, Any],
    schedule_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = await select_one("research_reimbursement_schedules", filters=(("schedule_id", schedule["schedule_id"]),))
    if existing:
        return existing
    doc = schedule_doc or schedule
    row = {
        "schedule_id": str(schedule["schedule_id"]),
        "schema_version": "1.0",
        "award_id": str(schedule["award_id"]),
        "schedule_status": str(schedule["status"]),
        "start_epoch": int(schedule["start_epoch"]),
        "epoch_count": int(schedule["epoch_count"]),
        "total_microusd": int(schedule["total_microusd"]),
        "entries": list(schedule.get("entries", [])),
        "schedule_hash": canonical_hash(doc),
        "schedule_doc": doc,
    }
    return await insert_row("research_reimbursement_schedules", row)


async def create_candidate_evaluation_event(
    *,
    candidate_id: str,
    run_id: str,
    ticket_id: str,
    event_type: str,
    candidate_status: str,
    reason: str,
    evaluator_ref: str | None = None,
    score_bundle_id: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_lab_candidate_evaluation_events",
        "candidate_id",
        candidate_id,
        lambda seq: {
            "candidate_id": candidate_id,
            "run_id": run_id,
            "ticket_id": ticket_id,
            "seq": seq,
            "event_type": event_type,
            "candidate_status": candidate_status,
            "evaluator_ref": evaluator_ref,
            "reason": reason,
            "score_bundle_id": score_bundle_id,
            "event_doc": event_doc or {},
        },
    )


_CONDITIONAL_VALIDATION_EVENT_TYPES = {
    "preliminary_gate_passed",
    "preliminary_gate_failed",
    "conditional_started",
    "retryable_failure",
    "conditional_completed",
    "final_pass",
    "final_fail",
}


async def create_conditional_validation_event(
    *,
    candidate_id: str,
    event_type: str,
    assignment_hash: str,
    policy_hash: str,
    rolling_window_hash: str,
    baseline_benchmark_bundle_id: str,
    source_ref: str,
    decision_score: float | None = None,
    threshold_points: float | None = None,
    queue_generation_id: str | None = None,
    source_score_bundle_id: str | None = None,
    failure_class: str | None = None,
    event_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one private conditional-validation lifecycle event idempotently."""

    normalized_type = str(event_type or "").strip()
    if normalized_type not in _CONDITIONAL_VALIDATION_EVENT_TYPES:
        raise ValueError(f"unsupported conditional validation event: {normalized_type}")
    score = float(decision_score) if decision_score is not None else None
    threshold = float(threshold_points) if threshold_points is not None else None
    if score is not None and (not math.isfinite(score) or score < 0.0 or score > 100.0):
        raise ValueError("conditional validation decision score must be finite and within 0-100")
    if threshold is not None and (
        not math.isfinite(threshold) or threshold < 0.0 or threshold > 100.0
    ):
        raise ValueError("conditional validation threshold must be finite and within 0-100")
    if not str(source_ref or "").strip():
        raise ValueError("conditional validation event requires source_ref")

    payload = {
        "schema_version": "1.1",
        "candidate_id": str(candidate_id),
        "event_type": normalized_type,
        "assignment_hash": str(assignment_hash),
        "policy_hash": str(policy_hash),
        "rolling_window_hash": str(rolling_window_hash),
        "baseline_benchmark_bundle_id": str(baseline_benchmark_bundle_id),
        "source_ref": str(source_ref),
        "decision_score": round(score, 6) if score is not None else None,
        "threshold_points": round(threshold, 6) if threshold is not None else None,
        "queue_generation_id": str(queue_generation_id) if queue_generation_id else None,
        "source_score_bundle_id": (
            str(source_score_bundle_id) if source_score_bundle_id else None
        ),
        "failure_class": str(failure_class or "") or None,
        "event_doc": dict(event_doc or {}),
    }
    event_hash = canonical_hash(payload)
    event_id = deterministic_uuid("conditional_validation", event_hash)

    if normalized_type == "retryable_failure":
        existing_filters = (("event_id", event_id),)
    else:
        existing_filters = (
            ("candidate_id", str(candidate_id)),
            ("event_type", normalized_type),
            ("assignment_hash", str(assignment_hash)),
        )
    existing = await select_one(
        "research_lab_conditional_validation_events",
        filters=existing_filters,
    )
    if existing is not None:
        _validate_conditional_validation_event(existing, payload)
        return existing

    row = {
        "event_id": event_id,
        **payload,
        "event_hash": event_hash,
    }
    try:
        return await insert_row("research_lab_conditional_validation_events", row)
    except Exception as exc:
        message = str(exc).lower()
        if not any(marker in message for marker in ("duplicate key", "unique constraint", "23505")):
            raise
        recovered = await select_one(
            "research_lab_conditional_validation_events",
            filters=existing_filters,
        )
        if recovered is None:
            raise
        _validate_conditional_validation_event(recovered, payload)
        return recovered


def _validate_conditional_validation_event(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for field in (
        "schema_version",
        "candidate_id",
        "event_type",
        "assignment_hash",
        "policy_hash",
        "rolling_window_hash",
        "baseline_benchmark_bundle_id",
        "decision_score",
        "threshold_points",
        "failure_class",
    ):
        actual = existing.get(field)
        wanted = expected.get(field)
        if field in {"decision_score", "threshold_points"}:
            actual = None if actual is None else round(float(actual), 6)
            wanted = None if wanted is None else round(float(wanted), 6)
        if actual != wanted:
            raise RuntimeError(
                "conditional validation event idempotency conflict:"
                f"{field}:stored={actual!r}:expected={wanted!r}"
            )


async def create_score_bundle(request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = dict(request.score_bundle)
    score_bundle_hash = str(bundle["score_bundle_hash"])
    score_bundle_id = "score_bundle:" + score_bundle_hash.split(":", 1)[1]
    existing = await select_one(
        "research_evaluation_score_bundles",
        filters=(("score_bundle_id", score_bundle_id),),
    )
    if existing:
        event = await _existing_or_recovered_event(
            "research_evaluation_score_bundle_events",
            "score_bundle_id",
            score_bundle_id,
            lambda: create_score_bundle_event(
                score_bundle_id=score_bundle_id,
                event_type=request.bundle_status,
                event_status=request.bundle_status,
                reason="score_bundle_created",
                event_doc={"score_bundle_hash": score_bundle_hash},
            ),
        )
        return existing, event

    row = {
        "score_bundle_id": score_bundle_id,
        "schema_version": str(bundle.get("schema_version") or "1.0"),
        "run_id": bundle["run_id"],
        "ticket_id": bundle["ticket_id"],
        "receipt_id": str(request.receipt_id) if request.receipt_id else None,
        "miner_hotkey": bundle["miner_hotkey"],
        "island": bundle["island"],
        "evaluation_epoch": int(bundle["evaluation_epoch"]),
        "bundle_status": request.bundle_status,
        "parent_artifact_hash": bundle["parent_artifact_hash"],
        "candidate_artifact_hash": bundle["candidate_artifact_hash"],
        "private_model_manifest_hash": bundle["private_model_manifest_hash"],
        "candidate_patch_hash": bundle["candidate_patch_hash"],
        "icp_set_hash": bundle["icp_set_hash"],
        "scoring_version": bundle["scoring_version"],
        "evaluator_version": bundle["evaluator_version"],
        "score_bundle_hash": score_bundle_hash,
        "anchored_hash": bundle["anchored_hash"],
        "signature_ref": bundle["signature_ref"],
        "score_bundle_doc": bundle,
    }
    inserted = await insert_row("research_evaluation_score_bundles", row)
    event = await create_score_bundle_event(
        score_bundle_id=score_bundle_id,
        event_type=request.bundle_status,
        event_status=request.bundle_status,
        reason="score_bundle_created",
        event_doc={"score_bundle_hash": score_bundle_hash},
    )
    return inserted, event


async def create_score_bundle_event(
    *,
    score_bundle_id: str,
    event_type: str,
    event_status: str,
    reason: str,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_evaluation_score_bundle_events",
        "score_bundle_id",
        score_bundle_id,
        lambda seq: {
            "score_bundle_id": score_bundle_id,
            "seq": seq,
            "event_type": event_type,
            "event_status": event_status,
            "reason": reason,
            "event_doc": event_doc or {},
        },
    )


async def create_scoring_category_result(
    *,
    source_kind: str,
    source_bundle_ref: str,
    category: str,
    assignment_hash: str,
    policy_hash: str,
    rolling_window_hash: str,
    icp_count: int,
    aggregate_score: float,
    scoring_run_id: str | None = None,
    candidate_id: str | None = None,
    delta_vs_baseline: float | None = None,
) -> dict[str, Any]:
    """Persist one aggregate-only category result idempotently.

    Hidden ICP identities and per-ICP scores remain solely in the private source
    bundle. This derivative table is service-role-only operational telemetry.
    """

    score = float(aggregate_score)
    delta = float(delta_vs_baseline) if delta_vs_baseline is not None else None
    if not math.isfinite(score) or score < 0.0 or score > 100.0:
        raise ValueError("category aggregate score must be finite and between 0 and 100")
    if delta is not None and not math.isfinite(delta):
        raise ValueError("category delta must be finite")
    if int(icp_count) <= 0:
        raise ValueError("category ICP count must be positive")
    payload = {
        "schema_version": "1.1",
        "source_kind": str(source_kind),
        "source_bundle_ref": str(source_bundle_ref),
        "category": str(category),
        "assignment_hash": str(assignment_hash),
        "policy_hash": str(policy_hash),
        "rolling_window_hash": str(rolling_window_hash),
        "icp_count": int(icp_count),
        "aggregate_score": round(score, 6),
        "scoring_run_id": str(scoring_run_id) if scoring_run_id else None,
        "candidate_id": str(candidate_id) if candidate_id else None,
        "delta_vs_baseline": (
            round(delta, 6)
            if delta is not None
            else None
        ),
    }
    result_hash = canonical_hash(payload)
    category_result_id = "scoring_category:" + result_hash.split(":", 1)[1]
    existing = await select_one(
        "research_lab_scoring_category_results",
        filters=(("category_result_id", category_result_id),),
    )
    if existing is not None:
        _validate_scoring_category_result(existing, payload)
        return existing
    row = {
        "category_result_id": category_result_id,
        **payload,
        "result_doc": {
            "schema_version": "1.1",
            "source_kind": payload["source_kind"],
            "category": payload["category"],
            "icp_count": payload["icp_count"],
            "aggregate_score": payload["aggregate_score"],
            "delta_vs_baseline": payload["delta_vs_baseline"],
        },
        "result_hash": result_hash,
        "anchored_hash": result_hash,
    }
    try:
        return await insert_row("research_lab_scoring_category_results", row)
    except Exception as exc:
        message = str(exc).lower()
        if not any(
            marker in message
            for marker in ("duplicate key", "unique constraint", "23505")
        ):
            raise
        recovered = await select_one(
            "research_lab_scoring_category_results",
            filters=(
                ("source_bundle_ref", source_bundle_ref),
                ("category", category),
                ("assignment_hash", assignment_hash),
            ),
        )
        if recovered is None:
            raise
        _validate_scoring_category_result(recovered, payload)
        return recovered


def _validate_scoring_category_result(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Reject an idempotency collision that points at different evidence."""

    for field in (
        "schema_version",
        "source_kind",
        "source_bundle_ref",
        "category",
        "assignment_hash",
        "policy_hash",
        "rolling_window_hash",
        "scoring_run_id",
        "candidate_id",
        "icp_count",
        "aggregate_score",
        "delta_vs_baseline",
    ):
        actual = existing.get(field)
        wanted = expected.get(field)
        if field in {"aggregate_score", "delta_vs_baseline"}:
            actual = None if actual is None else round(float(actual), 6)
            wanted = None if wanted is None else round(float(wanted), 6)
        elif field == "icp_count":
            actual = int(actual or 0)
            wanted = int(wanted or 0)
        else:
            actual = None if actual is None else str(actual)
            wanted = None if wanted is None else str(wanted)
        if actual != wanted:
            raise RuntimeError(
                "research_lab_scoring_category_result_conflict:"
                f"{field}:{actual!r}!={wanted!r}"
            )
