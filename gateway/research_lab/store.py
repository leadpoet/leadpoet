"""Supabase persistence helpers for Research Lab gateway endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

from gateway.db.client import get_write_client


RESEARCH_LAB_UUID_NAMESPACE = uuid5(NAMESPACE_URL, "leadpoet:research_lab:gateway")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def deterministic_uuid(*parts: Any) -> str:
    return str(uuid5(RESEARCH_LAB_UUID_NAMESPACE, canonical_hash(parts)))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def insert_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
    def _call() -> Any:
        return get_write_client().table(table).insert(row).execute()

    response = await asyncio.to_thread(_call)
    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError(f"{table}: insert returned no rows")
    return dict(data[0])


async def select_one(
    table: str,
    *,
    columns: str = "*",
    filters: Iterable[tuple[str, Any]],
) -> dict[str, Any] | None:
    def _call() -> Any:
        query = get_write_client().table(table).select(columns)
        for field, value in filters:
            query = query.eq(field, str(value) if isinstance(value, UUID) else value)
        return query.limit(1).execute()

    response = await asyncio.to_thread(_call)
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
    def _call() -> Any:
        query = get_write_client().table(table).select(columns)
        for field, value in filters:
            query = query.eq(field, str(value) if isinstance(value, UUID) else value)
        for field, desc in order_by:
            query = query.order(field, desc=desc)
        return query.limit(limit).execute()

    response = await asyncio.to_thread(_call)
    return [dict(row) for row in (getattr(response, "data", None) or [])]


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


async def payment_ref_exists(block_hash: str, extrinsic_index: int) -> bool:
    row = await select_one(
        "research_loop_start_payments",
        columns="payment_id",
        filters=(("block_hash", block_hash), ("extrinsic_index", extrinsic_index)),
    )
    return row is not None


async def create_ticket(request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ticket_id = deterministic_uuid("ticket", request.miner_hotkey, request.idempotency_key)
    existing_ticket = await select_one("research_loop_tickets", filters=(("ticket_id", ticket_id),))
    if existing_ticket:
        existing_event = await select_one(
            "research_loop_ticket_events",
            filters=(("ticket_id", ticket_id), ("seq", 0)),
        )
        if not existing_event:
            raise RuntimeError("existing Research Lab ticket is missing its opening event")
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
    seq = await next_event_seq("research_loop_ticket_events", "ticket_id", ticket_id)
    payload = {
        "ticket_id": ticket_id,
        "seq": seq,
        "event_type": event_type,
        "actor_hotkey": actor_hotkey,
        "reason": reason,
        "event_doc": event_doc or {},
    }
    row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_loop_ticket_events", row)


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
    seq = await next_event_seq("research_loop_start_credit_events", "credit_id", credit_id)
    payload = {
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
    }
    row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_loop_start_credit_events", row)


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
    seq = await next_event_seq("research_loop_run_queue_events", "run_id", run_id)
    payload = {
        "run_id": run_id,
        "ticket_id": ticket_id,
        "seq": seq,
        "event_type": event_type,
        "queue_priority": queue_priority,
        "worker_ref": worker_ref,
        "reason": reason,
        "event_doc": event_doc or {},
    }
    row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_loop_run_queue_events", row)


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
) -> dict[str, Any]:
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
        "verification_doc": {
            "call_function": payment_info.get("call_function"),
            "amount_rao": payment_info.get("amount_rao"),
        },
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


async def create_receipt_event(
    *,
    receipt_id: str,
    ticket_id: str,
    event_type: str,
    receipt_status: str,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seq = await next_event_seq("research_loop_receipt_events", "receipt_id", receipt_id)
    payload = {
        "receipt_id": receipt_id,
        "ticket_id": ticket_id,
        "seq": seq,
        "event_type": event_type,
        "receipt_status": receipt_status,
        "event_doc": event_doc or {},
    }
    row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_loop_receipt_events", row)


async def create_score_bundle(request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = dict(request.score_bundle)
    score_bundle_hash = str(bundle["score_bundle_hash"])
    score_bundle_id = "score_bundle:" + score_bundle_hash.split(":", 1)[1]
    existing = await select_one(
        "research_evaluation_score_bundles",
        filters=(("score_bundle_id", score_bundle_id),),
    )
    if existing:
        event = await select_one(
            "research_evaluation_score_bundle_events",
            filters=(("score_bundle_id", score_bundle_id), ("seq", 0)),
        )
        if not event:
            raise RuntimeError("existing Research Lab score bundle is missing its opening event")
        return existing, event

    row = {
        "score_bundle_id": score_bundle_id,
        "schema_version": "1.0",
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
    seq = await next_event_seq("research_evaluation_score_bundle_events", "score_bundle_id", score_bundle_id)
    payload = {
        "score_bundle_id": score_bundle_id,
        "seq": seq,
        "event_type": event_type,
        "event_status": event_status,
        "reason": reason,
        "event_doc": event_doc or {},
    }
    row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_evaluation_score_bundle_events", row)
