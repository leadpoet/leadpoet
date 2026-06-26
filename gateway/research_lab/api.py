"""Research Lab production gateway API.

The namespace is production-facing but inert by default. All mutating routes
require explicit Research Lab flags and write only Research Lab tables/events.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import re
import secrets
import time
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Query, Request

from gateway.qualification.api.payment import get_payment_info, verify_payment
from gateway.qualification.utils.chain import (
    BITTENSOR_NETUID,
    BITTENSOR_NETWORK,
    is_hotkey_registered as chain_is_hotkey_registered,
    verify_hotkey_signature,
)
from gateway.utils.bans import is_hotkey_banned

from .allocations import build_research_lab_allocation_bundle
from .arweave_audit import latest_arweave_anchor
from .bundles import build_research_lab_audit_bundle, build_shadow_report_bundle, contains_secret_material
from .config import DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS, ResearchLabGatewayConfig
from .key_vault import (
    OpenRouterKeyVaultError,
    encrypt_openrouter_key,
    openrouter_key_ref,
    preflight_openrouter_key,
)
from .maintenance import get_autoresearch_maintenance_state
from .models import (
    ResearchLabCandidateEvaluationResultRequest,
    ResearchLabCandidateEvaluationResultResponse,
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
from .public_activity import (
    fetch_public_loop_detail,
    fetch_public_loop_rows,
    safe_project_public_loop_activity,
)
from .store import (
    canonical_hash,
    create_candidate_evaluation_event,
    create_credit_event,
    create_loop_start_payment,
    create_openrouter_key_ref,
    create_queue_event,
    create_receipt,
    create_receipt_event,
    create_score_bundle,
    create_ticket,
    create_ticket_event,
    payment_ref_exists,
    select_all,
    select_many,
    select_one,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research-lab", tags=["research-lab"])
_OPENROUTER_KEY_REGISTRATION_ATTEMPTS: dict[str, list[float]] = {}
_OPENROUTER_KEY_REGISTER_MIN_SECONDS = 60.0
_OPENROUTER_KEY_REGISTER_MAX_PER_HOUR = 6
ACTIVE_AUTORESEARCH_QUEUE_STATUSES = {"queued", "started", "paused"}
AUTORESEARCH_PROXY_PREFIXES = (
    "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY",
    "RESEARCH_LAB_WORKER_PROXY",
    "RESEARCH_LAB_WORKER_HTTPS_PROXY",
)


@router.get("/status")
async def research_lab_status() -> dict[str, object]:
    config = ResearchLabGatewayConfig.from_env()
    maintenance = await get_autoresearch_maintenance_state()
    maintenance_public = {
        key: maintenance.get(key)
        for key in ("paused", "status", "reason", "status_at", "unavailable")
        if key in maintenance
    }
    latest_public_benchmark = None
    if config.api_enabled and config.reports_enabled:
        try:
            rows = await select_many(
                "research_lab_public_benchmark_report_current",
                filters=(("current_report_status", "published"),),
                order_by=(("benchmark_date", True), ("created_at", True)),
                limit=1,
            )
            if rows:
                latest_public_benchmark = {
                    "benchmark_date": rows[0].get("benchmark_date"),
                    "report_id": rows[0].get("report_id"),
                    "aggregate_score": rows[0].get("aggregate_score"),
                    "report_doc": rows[0].get("report_doc"),
                }
        except Exception as exc:
            logger.warning("research_lab_public_benchmark_status_unavailable: %s", str(exc)[:200])
            latest_public_benchmark = {"status": "unavailable"}
    return {
        "service": "leadpoet-research-lab-gateway",
        "status": "configured" if config.api_enabled else "disabled",
        **config.public_status(),
        "maintenance": maintenance_public,
        "latest_public_benchmark": latest_public_benchmark,
    }


@router.post("/tickets", response_model=ResearchLabTicketResponse)
async def create_research_lab_ticket(payload: ResearchLabTicketCreateRequest, request: Request):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    await _verify_signed_miner(payload)
    await _require_autoresearch_not_paused()
    await _enforce_autoresearch_loop_capacity(config, payload.miner_hotkey)
    island = _validate_allowed_research_island(config, payload.island)
    _require_default_research_model_tier(config, payload.research_model_tier)
    budget_doc = _effective_budget_doc(
        config,
        ticket={"ticket_doc": {}},
        research_model_tier=payload.research_model_tier,
        requested_compute_budget_usd=payload.requested_compute_budget_usd,
        max_compute_budget_usd=payload.max_compute_budget_usd,
    )
    payload_to_store = payload.model_copy(
        update={
            "island": island,
            "loop_start_fee_required_usd": float(config.loop_start_fee_usd),
            "research_model_tier": str(budget_doc["research_model_tier"]),
            "requested_compute_budget_usd": float(budget_doc["requested_compute_budget_usd"]),
            "max_compute_budget_usd": float(budget_doc["max_compute_budget_usd"]),
        }
    )

    try:
        ticket, event = await create_ticket(payload_to_store)
    except Exception as exc:
        _raise_storage_error(exc)

    await safe_project_public_loop_activity(
        str(ticket["ticket_id"]),
        source_ref=f"ticket_event:{event['event_id']}",
        reason="ticket_created",
        config=config,
    )
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
    _enforce_openrouter_key_registration_rate_limit(payload.miner_hotkey)

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
    await _require_autoresearch_not_paused()
    if config.miner_openrouter_key_required and payload.miner_openrouter_preflight_status != "passed":
        raise HTTPException(status_code=400, detail="miner OpenRouter key preflight must pass before queueing")
    await _verify_signed_miner(payload)
    ticket = await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)
    await _validate_miner_openrouter_key_ref(
        config,
        miner_hotkey=payload.miner_hotkey,
        key_ref=payload.miner_openrouter_key_ref,
        key_handling=payload.miner_openrouter_key_handling,
    )
    await _enforce_autoresearch_loop_capacity(config, payload.miner_hotkey)
    _validate_allowed_research_island(config, str(ticket.get("island") or ""))
    _require_default_research_model_tier(config, payload.research_model_tier)
    budget_doc = _effective_budget_doc(
        config,
        ticket=ticket,
        research_model_tier=payload.research_model_tier,
        requested_compute_budget_usd=payload.requested_compute_budget_usd,
        max_compute_budget_usd=payload.max_compute_budget_usd,
    )
    loop_start_fee_usd = _ticket_loop_start_fee_usd(ticket, config)

    run_id = str(uuid4())
    consumed_credit: dict[str, Any] | None = None
    if payload.credit_id:
        payment = await _consume_loop_start_credit(
            payload=payload,
            run_id=run_id,
        )
        consumed_credit = payment.pop("_credit")
        payment_ref = str(payment["payment_ref"])
    else:
        assert payload.payment_block_hash is not None
        assert payload.payment_extrinsic_index is not None
        payment_ref = f"{payload.payment_block_hash}:{payload.payment_extrinsic_index}"
        if await payment_ref_exists(payload.payment_block_hash, payload.payment_extrinsic_index):
            raise HTTPException(status_code=409, detail="Research Lab loop-start payment has already been used")

        payment_valid, payment_error = await verify_payment(
            block_hash=payload.payment_block_hash,
            extrinsic_index=payload.payment_extrinsic_index,
            miner_hotkey=payload.miner_hotkey,
            required_usd=loop_start_fee_usd,
        )
        if not payment_valid:
            raise HTTPException(status_code=402, detail=f"loop-start payment verification failed: {payment_error}")

        payment_info = await get_payment_info(payload.payment_block_hash, payload.payment_extrinsic_index)
        if not payment_info:
            raise HTTPException(status_code=402, detail="loop-start payment details were unavailable after verification")

        try:
            payment = await create_loop_start_payment(
                ticket_id=str(payload.ticket_id),
                payment_ref=payment_ref,
                block_hash=payload.payment_block_hash,
                extrinsic_index=payload.payment_extrinsic_index,
                network=BITTENSOR_NETWORK,
                netuid=BITTENSOR_NETUID,
                miner_hotkey=payload.miner_hotkey,
                payment_info=payment_info,
                required_usd=loop_start_fee_usd,
                payment_kind="loop_start",
                run_id=run_id,
                compute_budget_usd=budget_doc["requested_compute_budget_usd"],
                extra_verification_doc={
                    "research_model_tier": budget_doc["research_model_tier"],
                    "max_compute_budget_usd": budget_doc["max_compute_budget_usd"],
                },
            )
        except Exception as exc:
            if _is_duplicate_payment_error(exc):
                raise HTTPException(status_code=409, detail="Research Lab loop-start payment has already been used") from exc
            _raise_storage_error(exc)

    try:
        await create_ticket_event(
            ticket_id=str(payload.ticket_id),
            event_type="funded",
            actor_hotkey=payload.miner_hotkey,
            reason="loop_start_credit_consumed" if consumed_credit else "loop_start_payment_verified",
            event_doc={
                "payment_id": payment["payment_id"],
                "payment_ref": payment_ref,
                "loop_start_credit_id": payload.credit_id,
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
                "payment_ref": payment_ref,
                "payment_kind": "loop_start_credit" if consumed_credit else "loop_start",
                "loop_start_credit_id": payload.credit_id,
                "miner_openrouter_key_ref": payload.miner_openrouter_key_ref,
                "miner_openrouter_key_handling": payload.miner_openrouter_key_handling,
                "requested_loop_count": payload.requested_loop_count,
                **_queue_capacity_doc(config),
                **budget_doc,
            },
        )
        capacity_error = await _post_queue_capacity_error(
            config,
            run_id=run_id,
            miner_hotkey=payload.miner_hotkey,
        )
        if capacity_error:
            await create_queue_event(
                run_id=run_id,
                ticket_id=str(payload.ticket_id),
                event_type="cancelled",
                queue_priority=0,
                reason="capacity_rejected_after_queue",
                event_doc={"error": capacity_error},
            )
            raise RuntimeError(capacity_error)
        await create_ticket_event(
            ticket_id=str(payload.ticket_id),
            event_type="queued",
            actor_hotkey=payload.miner_hotkey,
            reason="paid_loop_queued",
            event_doc={
                "payment_id": payment["payment_id"],
                "payment_ref": payment_ref,
                "loop_start_credit_id": payload.credit_id,
                "run_id": run_id,
                **budget_doc,
            },
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
            reason="queue_failed_after_credit_consumed" if consumed_credit else "queue_failed_before_work_started",
            event_doc={
                "error": str(exc)[:200],
                "replaces_credit_id": payload.credit_id,
            },
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

    await safe_project_public_loop_activity(
        str(payload.ticket_id),
        source_ref=f"loop_start_queue:{run_id}",
        reason="paid_loop_queued",
        config=config,
    )
    return ResearchLabLoopStartResponse(
        ticket_id=str(payload.ticket_id),
        run_id=run_id,
        payment_id=payment["payment_id"],
        payment_ref=payment_ref,
        queued=True,
        credit_id=payload.credit_id,
        status="queued",
    )


@router.post("/loop-topups", response_model=ResearchLabLoopTopUpResponse)
async def top_up_research_lab_paid_loop(payload: ResearchLabLoopTopUpRequest):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.paid_loops_enabled, "Research Lab paid loops are disabled")
    _require_enabled(config.hosted_runs_enabled, "Research Lab hosted runs are disabled")
    _require_enabled(config.loop_topups_enabled, "Research Lab loop top-ups are disabled for launch")
    await _require_autoresearch_not_paused()
    if config.miner_openrouter_key_required and payload.miner_openrouter_preflight_status != "passed":
        raise HTTPException(status_code=400, detail="miner OpenRouter key preflight must pass before queueing")
    await _verify_signed_miner(payload)
    ticket = await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)
    await _validate_miner_openrouter_key_ref(
        config,
        miner_hotkey=payload.miner_hotkey,
        key_ref=payload.miner_openrouter_key_ref,
        key_handling=payload.miner_openrouter_key_handling,
    )
    _validate_allowed_research_island(config, str(ticket.get("island") or ""))
    if not payload.continue_from_run_id:
        raise HTTPException(status_code=400, detail="continue_from_run_id is required for Research Lab top-ups")
    continuation_context = await _topup_continuation_context(
        ticket_id=str(payload.ticket_id),
        run_id=str(payload.continue_from_run_id),
    )
    _require_default_research_model_tier(config, payload.research_model_tier)

    budget_doc = _effective_budget_doc(
        config,
        ticket=ticket,
        research_model_tier=payload.research_model_tier,
        requested_compute_budget_usd=payload.additional_compute_budget_usd,
        max_compute_budget_usd=payload.additional_compute_budget_usd,
    )
    budget_doc["additional_compute_budget_usd"] = float(payload.additional_compute_budget_usd)
    budget_doc["topup_reason"] = payload.topup_reason
    budget_doc["continue_from_run_id"] = str(payload.continue_from_run_id)
    budget_doc["continuation_context"] = continuation_context

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
    try:
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
    except Exception as exc:
        if _is_duplicate_payment_error(exc):
            raise HTTPException(status_code=409, detail="Research Lab top-up payment has already been used") from exc
        _raise_storage_error(exc)

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
                "payment_ref": payment_ref,
                "payment_kind": "top_up",
                "miner_openrouter_key_ref": payload.miner_openrouter_key_ref,
                "miner_openrouter_key_handling": payload.miner_openrouter_key_handling,
                **_queue_capacity_doc(config),
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

    await safe_project_public_loop_activity(
        str(payload.ticket_id),
        source_ref=f"loop_topup_queue:{run_id}",
        reason="loop_topup_queued",
        config=config,
    )
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


@router.post("/evaluations/candidate-results", response_model=ResearchLabCandidateEvaluationResultResponse)
async def record_research_lab_candidate_result(
    payload: ResearchLabCandidateEvaluationResultRequest,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.receipts_enabled, "Research Lab receipt writes are disabled")
    if payload.score_bundle:
        _require_enabled(config.evaluation_bundles_enabled, "Research Lab evaluation bundle writes are disabled")
    _require_internal_key(config, x_leadpoet_internal_key)

    candidate = await select_one(
        "research_lab_candidate_evaluation_current",
        filters=(("candidate_id", payload.candidate_id),),
    )
    if not candidate:
        raise HTTPException(status_code=404, detail="Research Lab candidate not found")

    score_bundle_id = None
    score_bundle_hash = None
    if payload.score_bundle:
        _validate_score_bundle_matches_candidate(payload.score_bundle, candidate)
        bundle_request = ResearchLabScoreBundleCreateRequest(
            receipt_id=candidate.get("receipt_id"),
            bundle_status=payload.candidate_status,
            score_bundle=payload.score_bundle,
        )
        try:
            bundle, _bundle_event = await create_score_bundle(bundle_request)
        except Exception as exc:
            _raise_storage_error(exc)
        score_bundle_id = str(bundle["score_bundle_id"])
        score_bundle_hash = str(bundle["score_bundle_hash"])

    try:
        event = await create_candidate_evaluation_event(
            candidate_id=payload.candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_type=payload.candidate_status,
            candidate_status=payload.candidate_status,
            evaluator_ref=payload.evaluator_ref,
            reason=payload.reason or f"validator_reported_{payload.candidate_status}",
            score_bundle_id=score_bundle_id,
            event_doc={
                "score_bundle_id": score_bundle_id,
                "score_bundle_hash": score_bundle_hash,
                "result_doc": payload.result_doc,
            },
        )
    except Exception as exc:
        _raise_storage_error(exc)

    receipt_finalized = await _maybe_finalize_candidate_receipt(candidate)
    await safe_project_public_loop_activity(
        str(candidate["ticket_id"]),
        source_ref=f"candidate_result:{event['event_id']}",
        reason=f"candidate_{payload.candidate_status}",
        config=config,
    )
    return ResearchLabCandidateEvaluationResultResponse(
        candidate_id=payload.candidate_id,
        status=payload.candidate_status,
        event_id=event["event_id"],
        event_seq=int(event["seq"]),
        score_bundle_id=score_bundle_id,
        receipt_finalized=receipt_finalized,
    )


@router.get("/tickets/{ticket_id}")
async def get_research_lab_ticket(
    ticket_id: str,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    row = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab ticket not found")
    return row


@router.get("/receipts/{receipt_id}")
async def get_research_lab_receipt(
    receipt_id: str,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    row = await select_one("research_loop_receipt_current", filters=(("receipt_id", receipt_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab receipt not found")
    return row


@router.get("/evaluations/score-bundles/{score_bundle_id}")
async def get_research_lab_score_bundle(
    score_bundle_id: str,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    row = await select_one("research_evaluation_score_bundle_current", filters=(("score_bundle_id", score_bundle_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab evaluation score bundle not found")
    return row


@router.get("/public/loops")
async def get_research_lab_public_loops(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None, max_length=80),
    topic: Optional[str] = Query(default=None, max_length=80),
    research_area: Optional[str] = Query(default=None, max_length=80),
    since_days: int = Query(default=14, ge=0, le=90),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.public_activity_enabled, "Research Lab public activity is disabled")
    try:
        items, groups = await fetch_public_loop_rows(
            limit=limit,
            offset=offset,
            status=status,
            topic=topic,
            research_area=research_area,
            since_days=since_days,
        )
    except Exception as exc:
        _raise_storage_error(exc)
    return {
        "schema_version": "1.0",
        "items": items,
        "topic_groups": groups,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "returned": len(items),
        },
        "filters": {
            "status": status,
            "topic": topic,
            "research_area": research_area,
            "since_days": since_days,
        },
    }


@router.get("/public/loops/{ticket_id}")
async def get_research_lab_public_loop_detail(ticket_id: str):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.public_activity_enabled, "Research Lab public activity is disabled")
    try:
        detail = await fetch_public_loop_detail(ticket_id)
    except Exception as exc:
        _raise_storage_error(exc)
    if not detail:
        raise HTTPException(status_code=404, detail="Research Lab public activity card not found")
    return {"schema_version": "1.0", **detail}


@router.get("/public/topic-groups")
async def get_research_lab_public_topic_groups(
    since_days: int = Query(default=14, ge=0, le=90),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.public_activity_enabled, "Research Lab public activity is disabled")
    try:
        _items, groups = await fetch_public_loop_rows(limit=1, offset=0, since_days=since_days)
    except Exception as exc:
        _raise_storage_error(exc)
    return {
        "schema_version": "1.0",
        "topic_groups": groups,
        "since_days": since_days,
    }


@router.get("/evaluations/latest/{epoch}")
async def get_research_lab_latest_evaluation_bundles(
    epoch: int,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    rows = await select_all(
        "research_evaluation_score_bundle_current",
        filters=(("evaluation_epoch", epoch), ("bundle_status", "scored"), ("current_event_status", "scored")),
    )
    return {
        "schema_version": "1.0",
        "bundle_type": "research_lab_evaluation_score_bundle_page",
        "epoch": int(epoch),
        "count": len(rows),
        "score_bundles": rows,
        "on_chain_submission_allowed": False,
    }


@router.get("/benchmarks/public/latest")
async def get_latest_public_benchmark_report():
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    rows = await select_many(
        "research_lab_public_benchmark_report_current",
        filters=(("current_report_status", "published"),),
        order_by=(("benchmark_date", True), ("created_at", True)),
        limit=1,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Research Lab public benchmark report not found")
    return rows[0]


@router.get("/benchmarks/public/{benchmark_date}")
async def get_public_benchmark_report_by_date(benchmark_date: str):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", benchmark_date):
        raise HTTPException(status_code=400, detail="benchmark_date must be YYYY-MM-DD")
    rows = await select_many(
        "research_lab_public_benchmark_report_current",
        filters=(("benchmark_date", benchmark_date), ("current_report_status", "published")),
        order_by=(("created_at", True),),
        limit=1,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Research Lab public benchmark report not found")
    return rows[0]


@router.get("/audit/latest/{epoch}")
async def get_research_lab_latest_audit_bundle(epoch: int):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")

    signed_rows = await select_many(
        "research_lab_signed_audit_bundle_current",
        filters=(("epoch", epoch),),
        order_by=(("created_at", True),),
        limit=1,
    )
    if signed_rows:
        row = dict(signed_rows[0])
        row["arweave_anchor"] = await _safe_latest_arweave_anchor(epoch)
        return row

    ticket_rows = await select_many("research_loop_ticket_current", filters=(), limit=1000)
    queue_rows = await select_many("research_loop_run_queue_current", filters=(), limit=1000)
    receipt_rows = await select_many("research_loop_receipt_current", filters=(), limit=1000)
    candidate_rows = await select_many("research_lab_candidate_evaluation_current", filters=(), limit=1000)
    candidate_event_rows = await select_many("research_lab_candidate_evaluation_events", filters=(), limit=1000)
    loop_event_rows = await select_many("research_lab_auto_research_loop_events", filters=(), limit=1000)
    dispatch_event_rows = await select_many("research_lab_scoring_dispatch_events", filters=(), limit=1000)
    rolling_window_rows = await select_many("research_lab_rolling_icp_windows", filters=(), limit=1000)
    benchmark_rows = await select_many("research_lab_private_model_benchmark_current", filters=(), limit=1000)
    private_model_version_rows = await select_many("research_lab_private_model_version_current", filters=(), limit=1000)
    promotion_event_rows = await select_many("research_lab_candidate_promotion_events", filters=(), limit=1000)
    private_repo_commit_event_rows = await select_many("research_lab_private_repo_commit_events", filters=(), limit=1000)
    public_benchmark_report_rows = await select_many("research_lab_public_benchmark_report_current", filters=(), limit=1000)
    score_bundle_rows = await select_many(
        "research_evaluation_score_bundle_current",
        filters=(("evaluation_epoch", epoch),),
        limit=1000,
    )
    try:
        preview = build_research_lab_audit_bundle(
            epoch=epoch,
            ticket_rows=ticket_rows,
            queue_rows=queue_rows,
            receipt_rows=receipt_rows,
            candidate_rows=candidate_rows,
            candidate_event_rows=candidate_event_rows,
            loop_event_rows=loop_event_rows,
            dispatch_event_rows=dispatch_event_rows,
            rolling_window_rows=rolling_window_rows,
            benchmark_rows=benchmark_rows,
            private_model_version_rows=private_model_version_rows,
            promotion_event_rows=promotion_event_rows,
            private_repo_commit_event_rows=private_repo_commit_event_rows,
            public_benchmark_report_rows=public_benchmark_report_rows,
            score_bundle_rows=score_bundle_rows,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "schema_version": "1.0",
        "bundle_type": "research_lab_audit_bundle_preview",
        "epoch": int(epoch),
        "signed": False,
        "signature_ref": "",
        "bundle_doc": preview,
        "arweave_anchor": await _safe_latest_arweave_anchor(epoch),
        "on_chain_submission_allowed": False,
    }


@router.get("/audit/arweave/latest/{epoch}")
async def get_research_lab_latest_arweave_anchor(epoch: int):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    row = await _safe_latest_arweave_anchor(epoch)
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab Arweave audit anchor not found")
    return {
        "schema_version": "1.0",
        "bundle_type": "research_lab_arweave_audit_anchor_latest",
        "epoch": int(epoch),
        "anchor": row,
        "verification": {
            "download_checkpoint_url": (
                f"https://arweave.net/{row.get('current_arweave_tx_id')}"
                if row.get("current_arweave_tx_id")
                else None
            ),
            "expected_event_type": "RESEARCH_LAB_EPOCH_AUDIT",
            "expected_event_hash": row.get("transparency_event_hash")
            or row.get("current_transparency_event_hash"),
            "expected_tee_sequence": row.get("tee_sequence") or row.get("current_tee_sequence"),
            "expected_checkpoint_merkle_root": row.get("current_checkpoint_merkle_root"),
        },
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


@router.get("/allocations/live/{epoch}")
async def get_research_lab_live_allocation(epoch: int):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.shadow_bundles_enabled, "Research Lab report bundles are disabled")
    try:
        return await build_research_lab_allocation_bundle(
            config=config,
            epoch=int(epoch),
            netuid=BITTENSOR_NETUID,
            persist_snapshot=bool(config.reimbursements_enabled or config.weight_mutation_enabled),
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research Lab allocation unavailable: {str(exc)[:200]}") from exc


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


async def _validate_miner_openrouter_key_ref(
    config: ResearchLabGatewayConfig,
    *,
    miner_hotkey: str,
    key_ref: str,
    key_handling: str,
) -> None:
    if not config.miner_openrouter_key_required:
        return
    normalized_ref = str(key_ref or "").strip()
    normalized_handling = str(key_handling or "").strip()
    normalized_hotkey = str(miner_hotkey or "").strip()
    if not normalized_ref:
        raise HTTPException(status_code=400, detail="miner OpenRouter key ref is required")
    if normalized_handling == "encrypted_ref":
        if not normalized_ref.startswith("encrypted_ref:openrouter:"):
            raise HTTPException(status_code=400, detail="encrypted OpenRouter key ref is required")
        row = await select_one(
            "research_lab_openrouter_key_refs",
            columns="key_ref,miner_hotkey,preflight_status",
            filters=(("key_ref", normalized_ref), ("miner_hotkey", normalized_hotkey)),
        )
        if not row:
            raise HTTPException(status_code=400, detail="miner OpenRouter key ref was not found for this hotkey")
        if str(row.get("preflight_status") or "") != "passed":
            raise HTTPException(status_code=400, detail="miner OpenRouter key ref has not passed preflight")
        return
    if normalized_handling == "ephemeral_ref":
        env_name = _openrouter_env_name_for_ref(config, normalized_ref)
        if not env_name:
            raise HTTPException(status_code=400, detail="ephemeral OpenRouter key refs are not configured")
        if not os.getenv(env_name):
            raise HTTPException(status_code=400, detail="configured OpenRouter key env var is empty")
        return
    raise HTTPException(status_code=400, detail="unsupported miner OpenRouter key handling")


def _openrouter_env_name_for_ref(config: ResearchLabGatewayConfig, key_ref: str) -> str:
    if config.miner_openrouter_key_ref_env_map_json:
        try:
            mapping = json.loads(config.miner_openrouter_key_ref_env_map_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=503, detail="OpenRouter key ref env map is invalid") from exc
        if not isinstance(mapping, Mapping):
            raise HTTPException(status_code=503, detail="OpenRouter key ref env map must be an object")
        mapped = mapping.get(str(key_ref))
        if mapped:
            return str(mapped)
    return str(config.miner_openrouter_key_env_var or "")


def _require_enabled(enabled: bool, detail: str) -> None:
    if not enabled:
        raise HTTPException(status_code=403, detail=detail)


async def _require_autoresearch_not_paused() -> None:
    state = await get_autoresearch_maintenance_state()
    if not state.get("paused"):
        return
    raise HTTPException(
        status_code=503,
        detail={
            "code": "research_lab_maintenance_paused",
            "message": "Research Lab auto-research is paused for maintenance",
            "reason": state.get("reason"),
            "status_at": state.get("status_at"),
        },
    )


async def _safe_latest_arweave_anchor(epoch: int) -> dict[str, object] | None:
    try:
        return await latest_arweave_anchor(epoch=epoch, netuid=BITTENSOR_NETUID)
    except Exception as exc:
        logger.warning("research_lab_arweave_anchor_unavailable: %s", str(exc)[:200])
        return None


def _require_internal_key(config: ResearchLabGatewayConfig, provided: Optional[str]) -> None:
    if not config.internal_api_key:
        raise HTTPException(status_code=403, detail="Research Lab internal API key is not configured")
    if not provided or not secrets.compare_digest(provided, config.internal_api_key):
        raise HTTPException(status_code=401, detail="invalid Research Lab internal API key")


def _enforce_openrouter_key_registration_rate_limit(miner_hotkey: str) -> None:
    now = time.monotonic()
    key = str(miner_hotkey or "").strip()
    attempts = [
        ts
        for ts in _OPENROUTER_KEY_REGISTRATION_ATTEMPTS.get(key, [])
        if now - ts < 3600.0
    ]
    if attempts and now - attempts[-1] < _OPENROUTER_KEY_REGISTER_MIN_SECONDS:
        raise HTTPException(status_code=429, detail="OpenRouter key registration rate limit exceeded")
    if len(attempts) >= _OPENROUTER_KEY_REGISTER_MAX_PER_HOUR:
        raise HTTPException(status_code=429, detail="OpenRouter key registration hourly limit exceeded")
    attempts.append(now)
    _OPENROUTER_KEY_REGISTRATION_ATTEMPTS[key] = attempts


async def _consume_loop_start_credit(
    *,
    payload: ResearchLabLoopStartRequest,
    run_id: str,
) -> dict[str, Any]:
    credit = await select_one(
        "research_loop_start_credit_current",
        filters=(
            ("credit_id", str(payload.credit_id)),
            ("ticket_id", str(payload.ticket_id)),
            ("miner_hotkey", payload.miner_hotkey),
        ),
    )
    if not credit:
        raise HTTPException(status_code=404, detail="Research Lab loop-start credit not found")
    if str(credit.get("current_credit_status") or "") != "available":
        raise HTTPException(status_code=409, detail="Research Lab loop-start credit is not available")
    if not credit.get("payment_id"):
        raise HTTPException(status_code=409, detail="Research Lab loop-start credit is missing its payment reference")
    try:
        await create_credit_event(
            credit_id=str(credit["credit_id"]),
            ticket_id=str(credit["ticket_id"]),
            payment_id=str(credit["payment_id"]) if credit.get("payment_id") else None,
            payment_ref=str(credit["payment_ref"]),
            miner_hotkey=str(credit["miner_hotkey"]),
            event_type="consumed",
            credit_status="consumed",
            reason="loop_start_credit_consumed_before_queueing",
            consumed_by_loop_id=run_id,
            event_doc={"run_id": run_id},
        )
    except Exception as exc:
        if _is_credit_claim_race_error(exc):
            raise HTTPException(status_code=409, detail="Research Lab loop-start credit was already consumed") from exc
        _raise_storage_error(exc)
    return {
        "payment_id": str(credit.get("payment_id") or ""),
        "payment_ref": str(credit["payment_ref"]),
        "_credit": credit,
    }


def _is_credit_claim_race_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_credit_consume_conflict" in message
        or "research_loop_start_credit_events_credit_seq_key" in message
        or "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
    )


def _is_duplicate_payment_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    if "research_loop_start_payments" not in message:
        return False
    return (
        "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
        or "payment_ref" in message
        or "block_extrinsic" in message
    )


def _validate_requested_model_and_budget(
    config: ResearchLabGatewayConfig,
    *,
    research_model_tier: str | None,
    requested_compute_budget_usd: float | None,
    max_compute_budget_usd: float | None,
) -> None:
    _effective_budget_doc(
        config,
        ticket={"ticket_doc": {}},
        research_model_tier=research_model_tier,
        requested_compute_budget_usd=requested_compute_budget_usd,
        max_compute_budget_usd=max_compute_budget_usd,
    )


def _require_default_research_model_tier(config: ResearchLabGatewayConfig, research_model_tier: str | None) -> None:
    default_tier = str(config.default_auto_research_model_tier or "default")
    requested_tier = str(research_model_tier or default_tier)
    if requested_tier != default_tier:
        raise HTTPException(status_code=400, detail="miner model tier selection is disabled for launch")


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
    tier_cap = _tier_compute_budget_cap(config, tier_doc)
    requested_budget = (
        requested_compute_budget_usd
        if requested_compute_budget_usd is not None
        else ticket_doc.get("requested_compute_budget_usd", config.default_compute_budget_usd)
    )
    _validate_compute_budget(config, float(requested_budget), "requested_compute_budget_usd")
    requested_budget = float(requested_budget)
    if requested_budget > tier_cap:
        raise HTTPException(
            status_code=400,
            detail=f"requested_compute_budget_usd cannot exceed tier cap {tier_cap:.2f}",
        )
    user_max_budget = (
        max_compute_budget_usd
        if max_compute_budget_usd is not None
        else ticket_doc.get("max_compute_budget_usd")
    )
    effective_max_budget = tier_cap
    if user_max_budget is not None:
        _validate_compute_budget(config, float(user_max_budget), "max_compute_budget_usd")
        effective_max_budget = min(effective_max_budget, float(user_max_budget))
    if requested_budget > effective_max_budget:
        raise HTTPException(status_code=400, detail="requested_compute_budget_usd cannot exceed max_compute_budget_usd")
    return {
        "research_model_tier": resolved_tier,
        "requested_compute_budget_usd": requested_budget,
        "max_compute_budget_usd": float(effective_max_budget),
        "budget_policy_version": "research-lab-budget:v1",
    }


def _tier_compute_budget_cap(config: ResearchLabGatewayConfig, tier_doc: Mapping[str, Any]) -> float:
    lower = max(0.0, float(config.min_compute_budget_usd))
    global_cap = max(lower, float(config.max_compute_budget_usd))
    try:
        tier_cap = float(tier_doc.get("max_compute_budget_usd", global_cap))
    except (TypeError, ValueError):
        tier_cap = global_cap
    return min(global_cap, max(lower, tier_cap))


def _validate_compute_budget(config: ResearchLabGatewayConfig, value: float, field_name: str) -> None:
    lower = max(0.0, float(config.min_compute_budget_usd))
    upper = max(lower, float(config.max_compute_budget_usd))
    if float(value) < lower or float(value) > upper:
        raise HTTPException(status_code=400, detail=f"{field_name} must be between {lower:.2f} and {upper:.2f}")


def _normalize_research_island(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _validate_allowed_research_island(config: ResearchLabGatewayConfig, value: object) -> str:
    island = _normalize_research_island(value)
    allowed = {_normalize_research_island(item) for item in config.allowed_research_islands}
    if island not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Research Lab launch currently accepts only generalist research loops",
        )
    return island


def _ticket_loop_start_fee_usd(ticket: Mapping[str, Any], config: ResearchLabGatewayConfig) -> float:
    try:
        fee = float(ticket.get("loop_start_fee_required_usd"))
    except (TypeError, ValueError):
        fee = float(config.loop_start_fee_usd)
    if fee < 0:
        return float(config.loop_start_fee_usd)
    return fee


async def _enforce_autoresearch_loop_capacity(config: ResearchLabGatewayConfig, miner_hotkey: str) -> None:
    active_rows = [
        row
        for row in await _active_autoresearch_queue_rows()
        if _autoresearch_active_row_is_fresh(row, config)
    ]
    capacity = _autoresearch_loop_capacity(config)
    if capacity <= 0:
        raise HTTPException(
            status_code=409,
            detail="too many autoresearch loops right now, try again later",
        )
    if not active_rows:
        return
    ticket_map = await _ticket_rows_by_id(active_rows)
    normalized_hotkey = str(miner_hotkey or "").strip()
    for row in active_rows:
        ticket = ticket_map.get(str(row.get("ticket_id") or ""))
        if ticket and str(ticket.get("miner_hotkey") or "").strip() == normalized_hotkey:
            raise HTTPException(
                status_code=409,
                detail="autoresearch loop for this hotkey already running",
            )

    if len(active_rows) >= capacity:
        raise HTTPException(
            status_code=409,
            detail="too many autoresearch loops right now, try again later",
        )


async def _post_queue_capacity_error(
    config: ResearchLabGatewayConfig,
    *,
    run_id: str,
    miner_hotkey: str,
) -> str | None:
    active_rows = [
        row
        for row in await _active_autoresearch_queue_rows()
        if _autoresearch_active_row_is_fresh(row, config)
    ]
    capacity = _autoresearch_loop_capacity(config)
    if capacity <= 0:
        return "too many autoresearch loops right now, try again later"
    normalized_run_id = str(run_id or "")
    if not any(str(row.get("run_id") or "") == normalized_run_id for row in active_rows):
        return None
    ticket_map = await _ticket_rows_by_id(active_rows)
    normalized_hotkey = str(miner_hotkey or "").strip()
    same_hotkey_rows = [
        row
        for row in active_rows
        if str((ticket_map.get(str(row.get("ticket_id") or "")) or {}).get("miner_hotkey") or "").strip()
        == normalized_hotkey
    ]
    same_hotkey_rows.sort(key=_autoresearch_capacity_sort_key)
    if same_hotkey_rows and str(same_hotkey_rows[0].get("run_id") or "") != normalized_run_id:
        return "autoresearch loop for this hotkey already running"

    active_rows.sort(key=_autoresearch_capacity_sort_key)
    admitted_run_ids = {str(row.get("run_id") or "") for row in active_rows[:capacity]}
    if len(active_rows) > capacity and normalized_run_id not in admitted_run_ids:
        return "too many autoresearch loops right now, try again later"
    return None


async def _active_autoresearch_queue_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status in sorted(ACTIVE_AUTORESEARCH_QUEUE_STATUSES):
        rows.extend(
            await select_all(
                "research_loop_run_queue_current",
                columns="run_id,ticket_id,current_queue_status,current_status_at",
                filters=(("current_queue_status", status),),
                order_by=(("current_status_at", True),),
                batch_size=1000,
                max_rows=10000,
            )
        )
    return rows


def _autoresearch_capacity_sort_key(row: Mapping[str, Any]) -> tuple[datetime, str]:
    raw_status_at = row.get("current_status_at")
    try:
        status_at = datetime.fromisoformat(str(raw_status_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        status_at = datetime.max.replace(tzinfo=timezone.utc)
    if status_at.tzinfo is None:
        status_at = status_at.replace(tzinfo=timezone.utc)
    return status_at.astimezone(timezone.utc), str(row.get("run_id") or "")


async def _ticket_rows_by_id(queue_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    ticket_ids = {str(row.get("ticket_id") or "") for row in queue_rows if row.get("ticket_id")}
    if not ticket_ids:
        return {}
    ticket_rows: dict[str, dict[str, Any]] = {}
    for ticket_id in sorted(ticket_ids):
        row = await select_one(
            "research_loop_ticket_current",
            columns="ticket_id,miner_hotkey",
            filters=(("ticket_id", ticket_id),),
        )
        if row:
            ticket_rows[ticket_id] = row
    return ticket_rows


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


def _queue_capacity_doc(config: ResearchLabGatewayConfig) -> dict[str, int | str]:
    return {
        "autoresearch_capacity_policy": "proxy_worker_capacity:v1",
        "autoresearch_capacity": int(_autoresearch_loop_capacity(config)),
        "active_loop_stale_after_seconds": max(
            60,
            int(config.active_loop_stale_after_seconds or DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
        ),
    }


def _configured_autoresearch_proxy_count() -> int:
    count = 0
    for index in range(1, 501):
        if any(os.getenv(f"{prefix}_{index}", "").strip() for prefix in AUTORESEARCH_PROXY_PREFIXES):
            count += 1
    return count


def _autoresearch_active_row_is_fresh(row: Mapping[str, Any], config: ResearchLabGatewayConfig) -> bool:
    if str(row.get("current_queue_status") or "").strip().lower() == "paused":
        return True
    stale_after_seconds = max(
        60,
        int(config.active_loop_stale_after_seconds or DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
    )
    raw_status_at = row.get("current_status_at")
    if not raw_status_at:
        return True
    try:
        status_at = datetime.fromisoformat(str(raw_status_at).replace("Z", "+00:00"))
    except ValueError:
        return True
    if status_at.tzinfo is None:
        status_at = status_at.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - status_at.astimezone(timezone.utc)).total_seconds()
    if age_seconds <= stale_after_seconds:
        return True
    logger.info(
        "research_lab_autoresearch_capacity_ignored_stale_active_run run_id=%s status=%s age_seconds=%.0f stale_after_seconds=%s",
        row.get("run_id"),
        row.get("current_queue_status"),
        age_seconds,
        stale_after_seconds,
    )
    return False


async def _topup_continuation_context(*, ticket_id: str, run_id: str) -> dict[str, Any]:
    run = await select_one(
        "research_loop_run_queue_current",
        filters=(("run_id", run_id), ("ticket_id", ticket_id)),
    )
    if not run:
        raise HTTPException(status_code=404, detail="continue_from_run_id was not found for this ticket")
    candidate_rows = await select_many(
        "research_lab_candidate_evaluation_current",
        filters=(("run_id", run_id), ("ticket_id", ticket_id)),
        order_by=(("created_at", True),),
        limit=10,
    )
    event_rows = await select_many(
        "research_lab_auto_research_loop_events",
        filters=(("run_id", run_id), ("ticket_id", ticket_id)),
        order_by=(("seq", True),),
        limit=30,
    )
    candidate_summaries = []
    for row in candidate_rows[:5]:
        score_bundle_id = str(row.get("current_score_bundle_id") or "")
        score_summary = await _candidate_score_summary(score_bundle_id)
        candidate_doc = {
            "candidate_id": str(row.get("candidate_id") or ""),
            "status": str(row.get("current_candidate_status") or ""),
            "score_bundle_id": score_bundle_id,
            "redacted_public_summary": _safe_public_context_text(row.get("redacted_public_summary"), max_length=500),
        }
        if score_summary:
            candidate_doc["score_summary"] = score_summary
        candidate_summaries.append(candidate_doc)
    reflections = _topup_reflection_summaries(event_rows)
    return {
        "schema_version": "1.0",
        "prior_run_id": run_id,
        "prior_queue_status": str(run.get("current_queue_status") or ""),
        "prior_event_count": len(event_rows),
        "prior_candidate_count": len(candidate_rows),
        "prior_candidates": candidate_summaries,
        "prior_reflections": reflections,
    }


async def _candidate_score_summary(score_bundle_id: str) -> dict[str, Any] | None:
    if not score_bundle_id:
        return None
    row = await select_one(
        "research_evaluation_score_bundle_current",
        filters=(("score_bundle_id", score_bundle_id),),
    )
    if not row:
        return None
    bundle = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else {}
    aggregates = bundle.get("aggregates") if isinstance(bundle.get("aggregates"), Mapping) else {}
    summary = {
        "base_score": _float_or_none(aggregates.get("base_score")),
        "candidate_score": _float_or_none(aggregates.get("candidate_score")),
        "mean_delta": _float_or_none(aggregates.get("mean_delta")),
        "delta_lcb": _float_or_none(aggregates.get("delta_lcb")),
        "icp_count": _int_or_none(aggregates.get("icp_count")),
        "successful_icp_count": _int_or_none(aggregates.get("successful_icp_count")),
        "failure_count": _int_or_none(aggregates.get("failure_count")),
    }
    compact = {key: value for key, value in summary.items() if value is not None}
    return compact or None


def _topup_reflection_summaries(event_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    reflections: list[dict[str, str]] = []
    for row in event_rows:
        if str(row.get("event_type") or "") != "reflection_recorded":
            continue
        event_doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
        reflection = event_doc.get("reflection") if isinstance(event_doc.get("reflection"), Mapping) else {}
        if not reflection:
            continue
        item = {
            "worked": _safe_public_context_text(reflection.get("worked"), max_length=300),
            "failed": _safe_public_context_text(reflection.get("failed"), max_length=300),
            "why": _safe_public_context_text(reflection.get("why"), max_length=300),
            "next_question": _safe_public_context_text(reflection.get("next_question"), max_length=300),
            "decision": _safe_public_context_text(reflection.get("decision"), max_length=40),
        }
        clean = {key: value for key, value in item.items() if value}
        if clean and not contains_secret_material(clean):
            reflections.append(clean)
        if len(reflections) >= 5:
            break
    return reflections


def _safe_public_context_text(value: object, *, max_length: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)[:max_length]
    return "" if contains_secret_material(text) else text


def _float_or_none(value: object) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_score_bundle_matches_candidate(bundle: dict[str, object], candidate: dict[str, object]) -> None:
    expected = {
        "run_id": str(candidate["run_id"]),
        "ticket_id": str(candidate["ticket_id"]),
        "miner_hotkey": str(candidate["miner_hotkey"]),
        "island": str(candidate["island"]),
        "parent_artifact_hash": str(candidate["parent_artifact_hash"]),
        "candidate_artifact_hash": str(candidate["candidate_artifact_hash"]),
        "private_model_manifest_hash": str(candidate["private_model_manifest_hash"]),
        "candidate_patch_hash": str(candidate["candidate_patch_hash"]),
    }
    for key, expected_value in expected.items():
        actual_value = str(bundle.get(key) or "")
        if actual_value != expected_value:
            raise HTTPException(
                status_code=400,
                detail=f"score bundle {key} does not match Research Lab candidate",
            )


async def _maybe_finalize_candidate_receipt(candidate: dict[str, object]) -> bool:
    receipt_id = candidate.get("receipt_id")
    if not receipt_id:
        return False

    candidates = await select_many(
        "research_lab_candidate_evaluation_current",
        filters=(("run_id", str(candidate["run_id"])),),
        limit=1000,
    )
    if not candidates:
        return False

    terminal_statuses = {"scored", "failed", "rejected", "tombstoned"}
    status_counts: dict[str, int] = {}
    score_bundle_ids: list[str] = []
    for row in candidates:
        status = str(row.get("current_candidate_status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in terminal_statuses:
            return False
        score_bundle_id = row.get("current_score_bundle_id")
        if score_bundle_id:
            score_bundle_ids.append(str(score_bundle_id))

    receipt = await select_one(
        "research_loop_receipt_current",
        filters=(("receipt_id", str(receipt_id)),),
    )
    if not receipt or receipt.get("current_receipt_status") != "queued":
        return False

    event_doc = {
        "run_id": str(candidate["run_id"]),
        "candidate_status_counts": status_counts,
        "score_bundle_ids": score_bundle_ids,
        "finalization_source": "gateway_qualification_worker_results",
    }
    has_scored_candidate = status_counts.get("scored", 0) > 0
    await create_receipt_event(
        receipt_id=str(receipt_id),
        ticket_id=str(candidate["ticket_id"]),
        event_type="completed" if has_scored_candidate else "failed",
        receipt_status="completed" if has_scored_candidate else "failed",
        event_doc=event_doc,
    )
    await create_ticket_event(
        ticket_id=str(candidate["ticket_id"]),
        event_type="completed" if has_scored_candidate else "cancelled",
        actor_hotkey=None,
        reason=(
            "gateway_research_lab_candidate_evaluation_completed"
            if has_scored_candidate
            else "gateway_research_lab_candidate_evaluation_failed"
        ),
        event_doc=event_doc,
    )
    return True


def _raise_storage_error(exc: Exception) -> None:
    message = _redact_storage_error_text(str(exc))
    message_lower = message.lower()
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
    if "does not exist" in message_lower or "relation" in message_lower:
        raise HTTPException(status_code=503, detail="Research Lab SQL migrations are not applied") from exc
    if "research_lab_queue_hotkey_conflict" in message_lower:
        raise HTTPException(status_code=409, detail="autoresearch loop for this hotkey already running") from exc
    if "research_lab_queue_capacity_conflict" in message_lower:
        raise HTTPException(status_code=409, detail="too many autoresearch loops right now, try again later") from exc
    if _is_duplicate_payment_error(exc):
        raise HTTPException(status_code=409, detail="Research Lab payment has already been used") from exc
    raise HTTPException(status_code=500, detail="Research Lab storage operation failed") from exc


def _redact_storage_error_text(value: str) -> str:
    value = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "[REDACTED_OPENROUTER_KEY]", value or "")
    value = re.sub(r"sb_(secret|publishable)_[A-Za-z0-9_-]+", "[REDACTED_SUPABASE_KEY]", value)
    return value[:2000]
