"""Supabase persistence helpers for Research Lab gateway endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

from gateway.db.client import get_write_client
from research_lab.canonical import sha256_json


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


async def create_openrouter_key_ref(
    *,
    key_ref: str,
    miner_hotkey: str,
    key_hash: str,
    encrypted_key_ciphertext: str,
    kms_key_id: str,
    encryption_context_hash: str,
    preflight_doc: dict[str, Any],
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
    }
    row["anchored_hash"] = canonical_hash(row)
    return await insert_row("research_lab_openrouter_key_refs", row)


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


async def create_candidate_artifact(request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = dict(request.private_model_manifest)
    patch = dict(request.candidate_patch_manifest)
    candidate_artifact_hash = str(patch["candidate_artifact_hash"])
    candidate_id = "candidate:" + candidate_artifact_hash.split(":", 1)[1]
    candidate_patch_hash = sha256_json(patch)
    existing = await select_one(
        "research_lab_candidate_artifacts",
        filters=(("candidate_id", candidate_id),),
    )
    if existing:
        event = await select_one(
            "research_lab_candidate_evaluation_events",
            filters=(("candidate_id", candidate_id), ("seq", 0)),
        )
        if not event:
            raise RuntimeError("existing Research Lab candidate is missing its opening event")
        return existing, event

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
    row["anchored_hash"] = canonical_hash(row)
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
    seq = await next_event_seq("research_lab_auto_research_loop_events", "run_id", run_id)
    payload = {
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
    }
    row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_lab_auto_research_loop_events", row)


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
    row = {
        "participation_snapshot_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "input_hash": input_hash,
    }
    return await insert_row("research_island_participation_snapshots", row)


async def create_reimbursement_award(
    *,
    award: dict[str, Any],
    receipt_id: str | None,
    participation_snapshot_id: str | None,
    policy_id: str,
    award_doc: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = await select_one("research_reimbursement_awards", filters=(("award_id", award["award_id"]),))
    if existing:
        event = await select_one(
            "research_reimbursement_award_events",
            filters=(("award_id", award["award_id"]), ("seq", 0)),
        )
        if not event:
            raise RuntimeError("existing Research Lab reimbursement award is missing its opening event")
        return existing, event
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
    seq = await next_event_seq("research_reimbursement_award_events", "award_id", award_id)
    payload = {
        "award_id": award_id,
        "seq": seq,
        "event_type": event_type,
        "award_status": award_status,
        "event_doc": event_doc or {},
    }
    row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_reimbursement_award_events", row)


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
    seq = await next_event_seq("research_lab_candidate_evaluation_events", "candidate_id", candidate_id)
    payload = {
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
    }
    row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        **payload,
        "anchored_hash": canonical_hash(payload),
    }
    return await insert_row("research_lab_candidate_evaluation_events", row)


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
