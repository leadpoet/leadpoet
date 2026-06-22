"""Research Lab production gateway API.

The namespace is production-facing but inert by default. All mutating routes
require explicit Research Lab flags and write only Research Lab tables/events.
"""

from __future__ import annotations

import asyncio
import logging
import re
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
from .key_vault import (
    OpenRouterKeyVaultError,
    encrypt_openrouter_key,
    openrouter_key_ref,
    preflight_openrouter_key,
)
from .models import (
    ResearchLabLoopStartRequest,
    ResearchLabLoopStartResponse,
    ResearchLabLoopTopUpRequest,
    ResearchLabLoopTopUpResponse,
    ResearchLabOpenRouterKeyRegisterRequest,
    ResearchLabOpenRouterKeyRegisterResponse,
    ResearchLabProbeRequest,
    ResearchLabReceiptCreateRequest,
    ResearchLabReceiptResponse,
    ResearchLabScoreBundleCreateRequest,
    ResearchLabScoreBundleResponse,
    ResearchLabTicketCreateRequest,
    ResearchLabTicketResponse,
)
from .store import (
    canonical_hash,
    create_credit_event,
    create_loop_start_payment,
    create_openrouter_key_ref,
    create_queue_event,
    create_receipt,
    create_score_bundle,
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
    _validate_requested_model_and_budget(
        config,
        research_model_tier=payload.research_model_tier,
        requested_compute_budget_usd=payload.requested_compute_budget_usd,
        max_compute_budget_usd=payload.max_compute_budget_usd,
    )

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


@router.post("/openrouter-keys", response_model=ResearchLabOpenRouterKeyRegisterResponse)
async def register_research_lab_openrouter_key(payload: ResearchLabOpenRouterKeyRegisterRequest):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    if not config.openrouter_key_kms_key_id:
        raise HTTPException(status_code=503, detail="Research Lab OpenRouter key vault KMS key is not configured")
    await _verify_signed_miner(payload)

    try:
        preflight_doc = await asyncio.to_thread(preflight_openrouter_key, payload.openrouter_api_key)
        key_hash = str(preflight_doc["key_hash"])
        key_ref = openrouter_key_ref(miner_hotkey=payload.miner_hotkey, key_hash=key_hash)
        encrypted = await asyncio.to_thread(
            encrypt_openrouter_key,
            raw_key=payload.openrouter_api_key,
            kms_key_id=config.openrouter_key_kms_key_id,
            miner_hotkey=payload.miner_hotkey,
            key_ref=key_ref,
        )
        await create_openrouter_key_ref(
            key_ref=key_ref,
            miner_hotkey=payload.miner_hotkey,
            key_hash=key_hash,
            encrypted_key_ciphertext=encrypted["ciphertext_b64"],
            kms_key_id=encrypted["kms_key_id"],
            encryption_context_hash=encrypted["encryption_context_hash"],
            preflight_doc={
                "source": "openrouter_current_key",
                "limit": preflight_doc.get("limit"),
                "limit_remaining": preflight_doc.get("limit_remaining"),
                "limit_reset": preflight_doc.get("limit_reset"),
                "usage": preflight_doc.get("usage"),
                "is_free_tier": preflight_doc.get("is_free_tier"),
                "is_management_key": preflight_doc.get("is_management_key"),
                "expires_at": preflight_doc.get("expires_at"),
                "key_label": payload.key_label,
            },
        )
    except OpenRouterKeyVaultError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _raise_storage_error(exc)

    return ResearchLabOpenRouterKeyRegisterResponse(
        key_ref=key_ref,
        preflight_status="passed",
        key_hash=key_hash,
        limit_remaining=preflight_doc.get("limit_remaining"),
        limit_reset=preflight_doc.get("limit_reset"),
    )


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
    ticket = await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)
    budget_doc = _effective_budget_doc(
        config,
        ticket=ticket,
        research_model_tier=payload.research_model_tier,
        requested_compute_budget_usd=payload.requested_compute_budget_usd,
        max_compute_budget_usd=payload.max_compute_budget_usd,
    )

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

    run_id = str(uuid4())
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
        payment_kind="loop_start",
        run_id=run_id,
        compute_budget_usd=budget_doc["requested_compute_budget_usd"],
        extra_verification_doc={
            "research_model_tier": budget_doc["research_model_tier"],
            "max_compute_budget_usd": budget_doc["max_compute_budget_usd"],
        },
    )

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
                **budget_doc,
            },
        )
        await create_queue_event(
            run_id=run_id,
            ticket_id=str(payload.ticket_id),
            event_type="queued",
            queue_priority=0,
            reason="paid_loop_queued",
            event_doc={
                "payment_id": payment["payment_id"],
                "payment_kind": "loop_start",
                "requested_loop_count": payload.requested_loop_count,
                **budget_doc,
            },
        )
        await create_ticket_event(
            ticket_id=str(payload.ticket_id),
            event_type="queued",
            actor_hotkey=payload.miner_hotkey,
            reason="paid_loop_queued",
            event_doc={"payment_id": payment["payment_id"], "run_id": run_id, **budget_doc},
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


@router.post("/loop-topups", response_model=ResearchLabLoopTopUpResponse)
async def top_up_research_lab_paid_loop(payload: ResearchLabLoopTopUpRequest):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.paid_loops_enabled, "Research Lab paid loops are disabled")
    _require_enabled(config.hosted_runs_enabled, "Research Lab hosted runs are disabled")
    if config.miner_openrouter_key_required and payload.miner_openrouter_preflight_status != "passed":
        raise HTTPException(status_code=400, detail="miner OpenRouter key preflight must pass before queueing")
    await _verify_signed_miner(payload)
    ticket = await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)

    budget_doc = _effective_budget_doc(
        config,
        ticket=ticket,
        research_model_tier=payload.research_model_tier,
        requested_compute_budget_usd=payload.additional_compute_budget_usd,
        max_compute_budget_usd=None,
    )
    budget_doc["additional_compute_budget_usd"] = float(payload.additional_compute_budget_usd)
    budget_doc["topup_reason"] = payload.topup_reason
    if payload.continue_from_run_id:
        budget_doc["continue_from_run_id"] = str(payload.continue_from_run_id)

    payment_ref = f"{payload.payment_block_hash}:{payload.payment_extrinsic_index}"
    if await payment_ref_exists(payload.payment_block_hash, payload.payment_extrinsic_index):
        raise HTTPException(status_code=409, detail="Research Lab top-up payment has already been used")

    payment_valid, payment_error = await verify_payment(
        block_hash=payload.payment_block_hash,
        extrinsic_index=payload.payment_extrinsic_index,
        miner_hotkey=payload.miner_hotkey,
        required_usd=float(payload.additional_compute_budget_usd),
    )
    if not payment_valid:
        raise HTTPException(status_code=402, detail=f"top-up payment verification failed: {payment_error}")

    payment_info = await get_payment_info(payload.payment_block_hash, payload.payment_extrinsic_index)
    if not payment_info:
        raise HTTPException(status_code=402, detail="top-up payment details were unavailable after verification")

    run_id = str(uuid4())
    payment = await create_loop_start_payment(
        ticket_id=str(payload.ticket_id),
        payment_ref=payment_ref,
        block_hash=payload.payment_block_hash,
        extrinsic_index=payload.payment_extrinsic_index,
        network=BITTENSOR_NETWORK,
        netuid=BITTENSOR_NETUID,
        miner_hotkey=payload.miner_hotkey,
        payment_info=payment_info,
        required_usd=float(payload.additional_compute_budget_usd),
        payment_kind="top_up",
        run_id=run_id,
        compute_budget_usd=float(payload.additional_compute_budget_usd),
        extra_verification_doc=budget_doc,
    )

    try:
        await create_ticket_event(
            ticket_id=str(payload.ticket_id),
            event_type="funded",
            actor_hotkey=payload.miner_hotkey,
            reason="loop_topup_payment_verified",
            event_doc={
                "payment_id": payment["payment_id"],
                "payment_ref": payment_ref,
                "miner_openrouter_key_ref": payload.miner_openrouter_key_ref,
                "miner_openrouter_key_handling": payload.miner_openrouter_key_handling,
                **budget_doc,
            },
        )
        await create_queue_event(
            run_id=run_id,
            ticket_id=str(payload.ticket_id),
            event_type="queued",
            queue_priority=-1,
            reason="loop_topup_queued",
            event_doc={
                "payment_id": payment["payment_id"],
                "payment_kind": "top_up",
                **budget_doc,
            },
        )
        await create_ticket_event(
            ticket_id=str(payload.ticket_id),
            event_type="queued",
            actor_hotkey=payload.miner_hotkey,
            reason="loop_topup_queued",
            event_doc={"payment_id": payment["payment_id"], "run_id": run_id, **budget_doc},
        )
    except Exception as exc:
        _raise_storage_error(exc)

    return ResearchLabLoopTopUpResponse(
        ticket_id=str(payload.ticket_id),
        run_id=run_id,
        continued_from_run_id=str(payload.continue_from_run_id) if payload.continue_from_run_id else None,
        topup_payment_id=payment["payment_id"],
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


@router.post("/evaluations/score-bundles", response_model=ResearchLabScoreBundleResponse)
async def create_research_lab_score_bundle(
    payload: ResearchLabScoreBundleCreateRequest,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.evaluation_bundles_enabled, "Research Lab evaluation bundle writes are disabled")
    _require_internal_key(config, x_leadpoet_internal_key)

    try:
        bundle, event = await create_score_bundle(payload)
    except Exception as exc:
        _raise_storage_error(exc)

    return ResearchLabScoreBundleResponse(
        score_bundle_id=bundle["score_bundle_id"],
        score_bundle_hash=bundle["score_bundle_hash"],
        status=bundle["bundle_status"],
        event_id=event["event_id"],
        event_seq=int(event["seq"]),
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


@router.get("/evaluations/score-bundles/{score_bundle_id}")
async def get_research_lab_score_bundle(score_bundle_id: str):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    row = await select_one("research_evaluation_score_bundle_current", filters=(("score_bundle_id", score_bundle_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab evaluation score bundle not found")
    return row


@router.get("/evaluations/latest/{epoch}")
async def get_research_lab_latest_evaluation_bundles(epoch: int):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    rows = await select_many(
        "research_evaluation_score_bundle_current",
        filters=(("evaluation_epoch", epoch), ("bundle_status", "scored"), ("current_event_status", "scored")),
        limit=1000,
    )
    return {
        "schema_version": "1.0",
        "bundle_type": "research_lab_evaluation_score_bundle_page",
        "epoch": int(epoch),
        "count": len(rows),
        "score_bundles": rows,
        "on_chain_submission_allowed": False,
    }


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


def _validate_requested_model_and_budget(
    config: ResearchLabGatewayConfig,
    *,
    research_model_tier: str | None,
    requested_compute_budget_usd: float | None,
    max_compute_budget_usd: float | None,
) -> None:
    try:
        config.resolve_auto_research_model(research_model_tier)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if requested_compute_budget_usd is not None:
        _validate_compute_budget(config, requested_compute_budget_usd, "requested_compute_budget_usd")
    if max_compute_budget_usd is not None:
        _validate_compute_budget(config, max_compute_budget_usd, "max_compute_budget_usd")
    if (
        requested_compute_budget_usd is not None
        and max_compute_budget_usd is not None
        and float(requested_compute_budget_usd) > float(max_compute_budget_usd)
    ):
        raise HTTPException(status_code=400, detail="requested_compute_budget_usd cannot exceed max_compute_budget_usd")


def _effective_budget_doc(
    config: ResearchLabGatewayConfig,
    *,
    ticket: dict[str, object],
    research_model_tier: str | None,
    requested_compute_budget_usd: float | None,
    max_compute_budget_usd: float | None,
) -> dict[str, object]:
    ticket_doc = ticket.get("ticket_doc") if isinstance(ticket.get("ticket_doc"), dict) else {}
    effective_tier = research_model_tier or str(ticket_doc.get("research_model_tier") or config.default_auto_research_model_tier)
    try:
        resolved_tier, _model, tier_doc = config.resolve_auto_research_model(effective_tier)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    requested_budget = (
        requested_compute_budget_usd
        if requested_compute_budget_usd is not None
        else ticket_doc.get("requested_compute_budget_usd", config.default_compute_budget_usd)
    )
    max_budget = (
        max_compute_budget_usd
        if max_compute_budget_usd is not None
        else ticket_doc.get("max_compute_budget_usd", tier_doc.get("max_compute_budget_usd", config.max_compute_budget_usd))
    )
    _validate_compute_budget(config, float(requested_budget), "requested_compute_budget_usd")
    _validate_compute_budget(config, float(max_budget), "max_compute_budget_usd")
    if float(requested_budget) > float(max_budget):
        raise HTTPException(status_code=400, detail="requested_compute_budget_usd cannot exceed max_compute_budget_usd")
    return {
        "research_model_tier": resolved_tier,
        "requested_compute_budget_usd": float(requested_budget),
        "max_compute_budget_usd": float(max_budget),
        "budget_policy_version": "research-lab-budget:v1",
    }


def _validate_compute_budget(config: ResearchLabGatewayConfig, value: float, field_name: str) -> None:
    lower = max(0.0, float(config.min_compute_budget_usd))
    upper = max(lower, float(config.max_compute_budget_usd))
    if float(value) < lower or float(value) > upper:
        raise HTTPException(status_code=400, detail=f"{field_name} must be between {lower:.2f} and {upper:.2f}")


def _raise_storage_error(exc: Exception) -> None:
    message = _redact_storage_error_text(str(exc))
    json_detail = getattr(exc, "json", None)
    if callable(json_detail):
        try:
            json_detail = json_detail()
        except Exception:
            json_detail = None
    logger.warning(
        "research_lab_storage_error type=%s detail=%s json=%s",
        type(exc).__name__,
        message,
        _redact_storage_error_text(str(json_detail)) if json_detail else None,
    )
    if "does not exist" in message or "relation" in message:
        raise HTTPException(status_code=503, detail="Research Lab SQL migrations are not applied") from exc
    raise HTTPException(status_code=500, detail="Research Lab storage operation failed") from exc


def _redact_storage_error_text(value: str) -> str:
    value = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "[REDACTED_OPENROUTER_KEY]", value or "")
    value = re.sub(r"sb_(secret|publishable)_[A-Za-z0-9_-]+", "[REDACTED_SUPABASE_KEY]", value)
    return value[:2000]
