"""Research Lab production gateway API.

The namespace is production-facing but inert by default. All mutating routes
require explicit Research Lab flags and write only Research Lab tables/events.
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Request

from gateway.qualification.api.payment import get_payment_info, verify_payment
from gateway.qualification.utils.chain import (
    BITTENSOR_NETUID,
    BITTENSOR_NETWORK,
    is_hotkey_registered as chain_is_hotkey_registered,
    verify_hotkey_signature,
)
from gateway.utils.bans import is_hotkey_banned

from .bundles import build_shadow_report_bundle
from .config import ResearchLabGatewayConfig
from .models import (
    ResearchLabLoopStartRequest,
    ResearchLabLoopStartResponse,
    ResearchLabProbeRequest,
    ResearchLabReceiptCreateRequest,
    ResearchLabReceiptResponse,
    ResearchLabTicketCreateRequest,
    ResearchLabTicketResponse,
)
from .store import (
    canonical_hash,
    create_credit_event,
    create_loop_start_payment,
    create_queue_event,
    create_receipt,
    create_ticket,
    create_ticket_event,
    payment_ref_exists,
    select_many,
    select_one,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research-lab", tags=["research-lab"])


@router.get("/status")
async def research_lab_status() -> dict[str, object]:
    config = ResearchLabGatewayConfig.from_env()
    return {
        "service": "leadpoet-research-lab-gateway",
        "status": "configured" if config.api_enabled else "disabled",
        **config.public_status(),
    }


@router.post("/tickets", response_model=ResearchLabTicketResponse)
async def create_research_lab_ticket(payload: ResearchLabTicketCreateRequest, request: Request):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    await _verify_signed_miner(payload)

    try:
        ticket, event = await create_ticket(payload)
    except Exception as exc:
        _raise_storage_error(exc)

    logger.info("Research Lab ticket opened: ticket_id=%s hotkey=%s", ticket["ticket_id"], payload.miner_hotkey[:16])
    return ResearchLabTicketResponse(
        ticket_id=ticket["ticket_id"],
        status=event["event_type"],
        event_id=event["event_id"],
        event_seq=int(event["seq"]),
        ticket_hash=ticket["ticket_hash"],
    )


@router.post("/probes")
async def create_research_lab_probe(payload: ResearchLabProbeRequest):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.probes_enabled, "Research Lab probes are disabled")
    await _verify_signed_miner(payload)

    ticket = await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)
    try:
        event = await create_ticket_event(
            ticket_id=str(payload.ticket_id),
            event_type="probe_created",
            actor_hotkey=payload.miner_hotkey,
            reason="private_probe_requested",
            event_doc={"probe_ref": payload.probe_ref, "ticket_hash": ticket.get("ticket_hash")},
        )
    except Exception as exc:
        _raise_storage_error(exc)

    return {
        "ticket_id": str(payload.ticket_id),
        "probe_ref": payload.probe_ref,
        "status": "probe_created",
        "event_id": event["event_id"],
        "event_seq": int(event["seq"]),
    }


@router.post("/loop-start", response_model=ResearchLabLoopStartResponse)
async def start_research_lab_paid_loop(payload: ResearchLabLoopStartRequest):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.paid_loops_enabled, "Research Lab paid loops are disabled")
    _require_enabled(config.hosted_runs_enabled, "Research Lab hosted runs are disabled")
    if config.miner_openrouter_key_required and payload.miner_openrouter_preflight_status != "passed":
        raise HTTPException(status_code=400, detail="miner OpenRouter key preflight must pass before queueing")
    await _verify_signed_miner(payload)
    await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)

    payment_ref = f"{payload.payment_block_hash}:{payload.payment_extrinsic_index}"
    if await payment_ref_exists(payload.payment_block_hash, payload.payment_extrinsic_index):
        raise HTTPException(status_code=409, detail="Research Lab loop-start payment has already been used")

    payment_valid, payment_error = await verify_payment(
        block_hash=payload.payment_block_hash,
        extrinsic_index=payload.payment_extrinsic_index,
        miner_hotkey=payload.miner_hotkey,
        required_usd=config.loop_start_fee_usd,
    )
    if not payment_valid:
        raise HTTPException(status_code=402, detail=f"loop-start payment verification failed: {payment_error}")

    payment_info = await get_payment_info(payload.payment_block_hash, payload.payment_extrinsic_index)
    if not payment_info:
        raise HTTPException(status_code=402, detail="loop-start payment details were unavailable after verification")

    payment = await create_loop_start_payment(
        ticket_id=str(payload.ticket_id),
        payment_ref=payment_ref,
        block_hash=payload.payment_block_hash,
        extrinsic_index=payload.payment_extrinsic_index,
        network=BITTENSOR_NETWORK,
        netuid=BITTENSOR_NETUID,
        miner_hotkey=payload.miner_hotkey,
        payment_info=payment_info,
        required_usd=config.loop_start_fee_usd,
    )

    run_id = str(uuid4())
    try:
        await create_ticket_event(
            ticket_id=str(payload.ticket_id),
            event_type="funded",
            actor_hotkey=payload.miner_hotkey,
            reason="loop_start_payment_verified",
            event_doc={
                "payment_id": payment["payment_id"],
                "payment_ref": payment_ref,
                "miner_openrouter_key_ref": payload.miner_openrouter_key_ref,
                "miner_openrouter_key_handling": payload.miner_openrouter_key_handling,
            },
        )
        await create_queue_event(
            run_id=run_id,
            ticket_id=str(payload.ticket_id),
            event_type="queued",
            queue_priority=0,
            reason="paid_loop_queued",
            event_doc={"payment_id": payment["payment_id"], "requested_loop_count": payload.requested_loop_count},
        )
        await create_ticket_event(
            ticket_id=str(payload.ticket_id),
            event_type="queued",
            actor_hotkey=payload.miner_hotkey,
            reason="paid_loop_queued",
            event_doc={"payment_id": payment["payment_id"], "run_id": run_id},
        )
    except Exception as exc:
        credit_id = "loop_start_credit:" + canonical_hash(
            {"ticket_id": str(payload.ticket_id), "payment_ref": payment_ref, "run_id": run_id}
        ).split(":", 1)[1][:32]
        await create_credit_event(
            credit_id=credit_id,
            ticket_id=str(payload.ticket_id),
            payment_id=payment["payment_id"],
            payment_ref=payment_ref,
            miner_hotkey=payload.miner_hotkey,
            event_type="granted",
            credit_status="available",
            reason="queue_failed_before_work_started",
            event_doc={"error": str(exc)[:200]},
        )
        logger.exception("Research Lab queue failed after payment; retry credit preserved")
        return ResearchLabLoopStartResponse(
            ticket_id=str(payload.ticket_id),
            run_id=run_id,
            payment_id=payment["payment_id"],
            payment_ref=payment_ref,
            queued=False,
            credit_preserved=True,
            credit_id=credit_id,
            status="credit_preserved_after_queue_failure",
        )

    return ResearchLabLoopStartResponse(
        ticket_id=str(payload.ticket_id),
        run_id=run_id,
        payment_id=payment["payment_id"],
        payment_ref=payment_ref,
        queued=True,
        status="queued",
    )


@router.post("/receipts", response_model=ResearchLabReceiptResponse)
async def create_research_lab_receipt(
    payload: ResearchLabReceiptCreateRequest,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.receipts_enabled, "Research Lab receipt writes are disabled")
    _require_internal_key(config, x_leadpoet_internal_key)

    try:
        receipt, _event = await create_receipt(payload)
    except Exception as exc:
        _raise_storage_error(exc)

    return ResearchLabReceiptResponse(
        receipt_id=receipt["receipt_id"],
        receipt_hash=receipt["receipt_hash"],
        status=receipt["receipt_status"],
    )


@router.get("/tickets/{ticket_id}")
async def get_research_lab_ticket(ticket_id: str):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    row = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab ticket not found")
    return row


@router.get("/receipts/{receipt_id}")
async def get_research_lab_receipt(receipt_id: str):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    row = await select_one("research_loop_receipt_current", filters=(("receipt_id", receipt_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab receipt not found")
    return row


@router.get("/reports/shadow/{epoch}")
async def get_research_lab_shadow_report(epoch: int):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.shadow_bundles_enabled, "Research Lab shadow bundles are disabled")
    _require_enabled(config.shadow_weights_enabled, "Research Lab shadow weights are disabled")

    weight_rows = await select_many(
        "research_loop_shadow_weight_inputs",
        filters=(("epoch", epoch),),
        limit=1000,
    )
    ticket_rows = await select_many(
        "research_loop_ticket_current",
        filters=(),
        limit=1000,
    )
    queue_rows = await select_many(
        "research_loop_run_queue_current",
        filters=(),
        limit=1000,
    )
    receipt_rows = await select_many(
        "research_loop_receipt_current",
        filters=(),
        limit=1000,
    )
    reimbursement_rows = await select_many(
        "research_reimbursement_award_current",
        filters=(),
        limit=1000,
    )
    try:
        return build_shadow_report_bundle(
            epoch=epoch,
            weight_input_snapshots=weight_rows,
            ticket_rows=ticket_rows,
            queue_rows=queue_rows,
            receipt_rows=receipt_rows,
            reimbursement_rows=reimbursement_rows,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _verify_signed_miner(payload: object) -> None:
    signature_valid = verify_hotkey_signature(
        hotkey=payload.miner_hotkey,
        signature=payload.signature,
        message_data=payload.signed_payload(),
    )
    if not signature_valid:
        raise HTTPException(status_code=401, detail="invalid miner hotkey signature")

    is_banned, ban_reason = await is_hotkey_banned(payload.miner_hotkey)
    if is_banned:
        raise HTTPException(status_code=403, detail=f"hotkey is banned: {ban_reason}")

    is_registered, _role = await chain_is_hotkey_registered(payload.miner_hotkey)
    if not is_registered:
        raise HTTPException(status_code=403, detail="hotkey is not registered on this subnet")


async def _get_ticket_for_miner(ticket_id: str, miner_hotkey: str) -> dict[str, object]:
    ticket = await select_one(
        "research_loop_tickets",
        filters=(("ticket_id", ticket_id), ("miner_hotkey", miner_hotkey)),
    )
    if not ticket:
        raise HTTPException(status_code=404, detail="Research Lab ticket not found for miner")
    return ticket


def _require_enabled(enabled: bool, detail: str) -> None:
    if not enabled:
        raise HTTPException(status_code=403, detail=detail)


def _require_internal_key(config: ResearchLabGatewayConfig, provided: Optional[str]) -> None:
    if not config.internal_api_key:
        raise HTTPException(status_code=403, detail="Research Lab internal API key is not configured")
    if not provided or not secrets.compare_digest(provided, config.internal_api_key):
        raise HTTPException(status_code=401, detail="invalid Research Lab internal API key")


def _raise_storage_error(exc: Exception) -> None:
    message = str(exc)
    if "does not exist" in message or "relation" in message:
        raise HTTPException(status_code=503, detail="Research Lab SQL migrations are not applied") from exc
    raise HTTPException(status_code=500, detail="Research Lab storage operation failed") from exc
