"""Research Lab production gateway API.

The namespace is production-facing but inert by default. All mutating routes
require explicit Research Lab flags and write only Research Lab tables/events.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import copy
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
from gateway.utils.rate_limiter import reserve_submission_slot
from research_lab.eval.promotion_metric import (
    benchmark_relative_score_deltas,
    promotion_improvement_metric,
)

from .allocations import build_research_lab_allocation_bundle
from .arweave_audit import latest_arweave_anchor
from .bundles import build_research_lab_audit_bundle, build_shadow_report_bundle, contains_secret_material
from .candidate_generation_report import fetch_candidate_generation_failure_report
from .config import DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS, ResearchLabGatewayConfig
from .key_vault import (
    OpenRouterKeyVaultError,
    encrypt_openrouter_key,
    openrouter_key_ref,
    preflight_openrouter_key,
    verify_openrouter_workspace_privacy,
)
from .v2_credential_envelopes import (
    persist_openrouter_credential_envelope_v2,
)
from .maintenance import get_autoresearch_maintenance_state
from .ticket_intake_validation import validate_ticket_direction
from .ticket_lifecycle import (
    TERMINAL_TICKET_STATUSES,
    UNPAID_TICKET_ELIGIBLE_STATUSES,
    is_ticket_expiry_conflict,
    normalized_ticket_status,
    ticket_is_expired,
    ticket_is_house_arm,
    unpaid_ticket_deadline_passed,
    unpaid_ticket_expires_at,
)
from .candidate_diagnostics import (
    _sanitized_run_summary_from_terminal_event,
)
from .miner_diagnostics import (
    build_candidate_diagnostics,
    visibility_map_from_benchmark_split,
)
from .models import (
    ResearchLabCandidateEvaluationResultRequest,
    ResearchLabCandidateEvaluationResultResponse,
    ResearchLabLoopStartRequest,
    ResearchLabLoopStartResponse,
    ResearchLabLoopTopUpRequest,
    ResearchLabLoopTopUpResponse,
    ResearchLabOpenRouterKeyRegisterRequest,
    ResearchLabOpenRouterKeyRegisterResponse,
    ResearchLabOpenRouterCredentialRecipientRequest,
    ResearchLabOpenRouterCredentialRecipientV2,
    ResearchLabOpenRouterCredentialReleaseEvidenceV2,
    ResearchLabOpenRouterCredentialRecipientsResponse,
    ResearchLabLoopDiagnosticsRequest,
    ResearchLabProbeRequest,
    ResearchLabReceiptCreateRequest,
    ResearchLabReceiptResponse,
    ResearchLabResumeCreditBlockedRequest,
    ResearchLabResumeCreditBlockedResponse,
    ResearchLabScoreBundleCreateRequest,
    ResearchLabScoreBundleResponse,
    ResearchLabSourceAdapterSubmissionRequest,
    ResearchLabSourceAdapterSubmissionResponse,
    ResearchLabSourceMetadata,
    ResearchLabSourceAddCredentialRecipientRequest,
    ResearchLabCredentialRecipientResponse,
    ResearchLabSourceAdapterRecheckResponse,
    ResearchLabSourceAdapterProbeConfigureRequest,
    ResearchLabSourceAdapterProbeConfigureResponse,
    ResearchLabSourceAdapterProvisionRequest,
    ResearchLabSourceAdapterProvisionResponse,
    ResearchLabTicketCreateRequest,
    ResearchLabTicketResponse,
    reject_secret_material,
)
from .public_activity import (
    fetch_public_loop_detail,
    fetch_public_loop_rows,
    fetch_public_loop_summary,
    public_loop_outcome_closes_ticket,
    safe_project_public_loop_activity,
)
from .chain import resolve_research_lab_evaluation_epoch
from .promotion import private_repo_head_alignment_status
from .source_add_catalog import (
    ALREADY_SUBMITTED_DETAIL,
    PROVISION_STATUS_APPROVED_PENDING,
    PROVISION_STATUS_DISABLED,
    PROVISION_STATUS_ELIGIBLE,
    PROVISION_STATUSES,
    reject_source_add_secret_text,
    sanitize_source_add_doc,
    source_add_encrypted_envelope_valid,
)
from .tee_protocol import legacy_v1_enabled
from .store import (
    canonical_hash,
    call_rpc,
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
    insert_row,
    payment_ref_exists,
    select_all,
    select_many,
    select_one,
    update_row,
)
from gateway.research_lab.provider_evidence_proxy import (
    ProviderRegistryEntry,
    reserved_builtin_provider_ids_sync,
    validate_provider_registry_entries,
)
from gateway.research_lab.provider_capabilities import validate_capability_provider_doc
from research_lab.probe_catalog import ProviderProbeEndpoint, validate_probe_catalog
from research_lab.improvement_engine.config import ImprovementEngineConfig
from research_lab.source_add_execution import intake_source_add_submission
from research_lab.source_add_identity import (
    SOURCE_ADD_IDENTITY_VERSION,
    legacy_source_identity_hash,
    source_identity_alias_hashes_from_metadata,
    source_identity_hash_from_metadata,
)
from .source_add_workflow import (
    source_add_host_hash,
    source_add_probe_config_ref,
    source_add_ref,
    source_add_work_id,
)
from research_lab.improvement_engine.fix_generator import sanitized_miner_opportunity
from research_lab.improvement_engine.scanner import scan_for_issues


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


def _source_add_provision_credential_ref(miner_hotkey: str, adapter_id: str) -> str:
    return (
        "encrypted_ref:source_add:"
        + canonical_hash(
            {"adapter_id": str(adapter_id), "miner": str(miner_hotkey)}
        )[-32:]
    )


async def _source_add_credential_recipient(
    *,
    miner_hotkey: str,
    adapter_id: str,
    credential_ref: str,
) -> dict[str, Any]:
    from gateway.utils.tee_client import coordinator_tee_client

    try:
        return dict(
            await coordinator_tee_client.v2_get_source_add_ingress_recipient(
                miner_hotkey=str(miner_hotkey),
                adapter_ref="source_add:%s" % str(adapter_id),
                credential_ref=str(credential_ref),
            )
        )
    except Exception as exc:
        logger.warning(
            "SOURCE_ADD_V2_RECIPIENT_UNAVAILABLE type=%s", type(exc).__name__
        )
        raise HTTPException(
            status_code=503,
            detail="attested SOURCE_ADD credential recipient is unavailable",
        ) from exc


async def _seal_source_add_credential_v2(
    *,
    encrypted: Any,
    miner_hotkey: str,
    adapter_id: str,
    expected_credential_ref: str,
) -> dict[str, Any]:
    from gateway.tee.source_add_credential_ingress_v2 import (
        source_add_encryption_context,
    )
    from gateway.tee.source_add_runtime_v2 import (
        validate_source_add_credential_envelope_v2,
    )
    from gateway.utils.tee_client import coordinator_tee_client

    try:
        result = await coordinator_tee_client.v2_seal_source_add_ingress_credential(
            request_id=str(encrypted.request_id),
            ciphertext_b64=str(encrypted.ciphertext_b64),
        )
        envelope = validate_source_add_credential_envelope_v2(
            result.get("credential_envelope") or {}
        )
    except Exception as exc:
        logger.warning(
            "SOURCE_ADD_V2_CREDENTIAL_SEAL_FAILED type=%s", type(exc).__name__
        )
        raise HTTPException(
            status_code=400,
            detail="attested SOURCE_ADD credential ciphertext is invalid or expired",
        ) from exc
    expected_context = source_add_encryption_context(
        miner_hotkey=str(miner_hotkey),
        adapter_ref="source_add:%s" % str(adapter_id),
    )
    if (
        envelope.get("envelope_kind") != "coordinator_sealed"
        or envelope.get("credential_ref") != expected_credential_ref
        or envelope.get("encryption_context") != expected_context
    ):
        raise HTTPException(
            status_code=400,
            detail="attested SOURCE_ADD credential scope differs",
        )
    return {
        key: item
        for key, item in envelope.items()
        if key != "ciphertext_blob"
    }


async def _openrouter_credential_recipient_v2(
    *,
    miner_hotkey: str,
    credential_kind: str,
) -> dict[str, Any]:
    from gateway.utils.tee_client import coordinator_tee_client

    try:
        return dict(
            await coordinator_tee_client.v2_get_openrouter_ingress_recipient(
                miner_hotkey=str(miner_hotkey),
                credential_kind=str(credential_kind),
            )
        )
    except Exception as exc:
        logger.warning(
            "OPENROUTER_V2_RECIPIENT_UNAVAILABLE kind=%s type=%s",
            credential_kind,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="attested OpenRouter credential recipient is unavailable",
        ) from exc


async def _openrouter_credential_release_evidence_v2() -> dict[str, Any]:
    """Expose direct, immutable release evidence for miner-side verification."""

    import boto3
    from botocore.config import Config

    from gateway.tee.release_channel_v2 import (
        DEFAULT_BUCKET,
        DEFAULT_PREFIX,
        release_channel_key,
    )
    from gateway.utils.tee_client import coordinator_tee_client

    try:
        boot = dict(await coordinator_tee_client.v2_get_boot_identity())
        commit = str(boot.get("commit_sha") or "").lower()
        key = release_channel_key(commit, prefix=DEFAULT_PREFIX)
        client = boto3.client(
            "s3",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
            ),
        )
        head = await asyncio.to_thread(
            client.head_object,
            Bucket=DEFAULT_BUCKET,
            Key=key,
        )
        if str(head.get("ObjectLockMode") or "").upper() != "COMPLIANCE":
            raise RuntimeError("release channel is not Object-Locked in COMPLIANCE mode")
        version_id = str(head.get("VersionId") or "")
        if not version_id:
            raise RuntimeError("release channel version is unavailable")
        params = {
            "Bucket": DEFAULT_BUCKET,
            "Key": key,
            "VersionId": version_id,
        }
        get_url, head_url = await asyncio.gather(
            asyncio.to_thread(
                client.generate_presigned_url,
                "get_object",
                Params=params,
                ExpiresIn=300,
                HttpMethod="GET",
            ),
            asyncio.to_thread(
                client.generate_presigned_url,
                "head_object",
                Params=params,
                ExpiresIn=300,
                HttpMethod="HEAD",
            ),
        )
        return {
            "schema_version": "leadpoet.openrouter_release_evidence.v2",
            "coordinator_boot_identity": boot,
            "release_channel_version_id": version_id,
            "release_channel_get_url": str(get_url),
            "release_channel_head_url": str(head_url),
        }
    except Exception as exc:
        logger.warning(
            "OPENROUTER_V2_RELEASE_EVIDENCE_UNAVAILABLE type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="independently built OpenRouter recipient release evidence is unavailable",
        ) from exc


async def _seal_openrouter_credential_v2(
    *,
    encrypted: Any,
    miner_hotkey: str,
    credential_kind: str,
) -> dict[str, Any]:
    from gateway.tee.openrouter_credential_v2 import (
        validate_openrouter_ingress_envelope_v2,
    )
    from gateway.utils.tee_client import coordinator_tee_client
    from leadpoet_canonical.attested_v2 import sha256_bytes

    try:
        result = await coordinator_tee_client.v2_seal_openrouter_ingress_credential(
            request_id=str(encrypted.request_id),
            ciphertext_b64=str(encrypted.ciphertext_b64),
        )
        envelope = validate_openrouter_ingress_envelope_v2(
            result.get("credential_envelope") or {}
        )
    except Exception as exc:
        logger.warning(
            "OPENROUTER_V2_CREDENTIAL_SEAL_FAILED kind=%s type=%s",
            credential_kind,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=400,
            detail="attested OpenRouter credential ciphertext is invalid or expired",
        ) from exc
    if (
        envelope.get("credential_kind") != credential_kind
        or envelope.get("miner_hotkey_hash")
        != sha256_bytes(str(miner_hotkey).encode("utf-8"))
    ):
        raise HTTPException(
            status_code=400,
            detail="attested OpenRouter credential scope differs",
        )
    return dict(envelope)


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
    private_model_repo_head = None
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
    try:
        private_model_repo_head = await private_repo_head_alignment_status(config)
    except Exception as exc:
        logger.warning("research_lab_private_model_repo_head_status_unavailable: %s", str(exc)[:200])
        private_model_repo_head = {"status": "unavailable"}
    return {
        "service": "leadpoet-research-lab-gateway",
        "status": "configured" if config.api_enabled else "disabled",
        **config.public_status(),
        "maintenance": maintenance_public,
        "latest_public_benchmark": latest_public_benchmark,
        "private_model_repo_head": private_model_repo_head,
    }


@router.post("/tickets", response_model=ResearchLabTicketResponse)
async def create_research_lab_ticket(payload: ResearchLabTicketCreateRequest, request: Request):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.miner_submissions_enabled, "Research Lab miner submissions are disabled")
    await _verify_signed_miner(payload)
    await _enforce_research_lab_submission_rate_limit(payload.miner_hotkey, route="tickets")
    await _enforce_open_ticket_cap(config, payload.miner_hotkey)
    await _require_autoresearch_not_paused()
    await _enforce_autoresearch_loop_capacity(config, payload.miner_hotkey)
    island = _validate_allowed_research_island(config, payload.island)
    _require_default_research_model_tier(config, payload.research_model_tier)
    # Free intake validation: reject directions bound to uneditable paths or
    # already refused repeatedly against the active source BEFORE the miner
    # pays a loop-start fee or a planner call is spent.
    intake_rejection = await validate_ticket_direction(
        brief_public_summary=str(payload.brief_public_summary or ""),
        allowed_prefixes=config.code_edit_allowed_path_prefixes(),
        allowed_exact=config.code_edit_allowed_exact_paths(),
        allowed_suffixes=config.code_edit_allowed_suffixes(),
    )
    if intake_rejection is not None:
        raise HTTPException(status_code=422, detail=intake_rejection)
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

    current_ticket = await select_one(
        "research_loop_ticket_current",
        filters=(("ticket_id", str(ticket["ticket_id"])),),
    )
    if current_ticket and ticket_is_expired(current_ticket):
        _raise_ticket_expired(current_ticket)
    expiry_at = unpaid_ticket_expires_at(current_ticket or ticket)

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
        unpaid_expires_at=expiry_at.isoformat() if expiry_at else None,
    )


@router.post("/probes")
async def create_research_lab_probe(payload: ResearchLabProbeRequest):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.miner_submissions_enabled, "Research Lab miner submissions are disabled")
    _require_enabled(config.probes_enabled, "Research Lab probes are disabled")
    await _verify_signed_miner(payload)
    await _enforce_research_lab_submission_rate_limit(payload.miner_hotkey, route="probes")

    ticket = await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)
    await _require_ticket_mutable(ticket, enforce_unpaid_deadline=True)
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


async def _source_add_rpc(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Call one atomic SOURCE_ADD RPC and fail closed on schema/storage drift."""

    try:
        value = await call_rpc(name, params)
    except Exception as exc:
        logger.warning("SOURCE_ADD_RPC_FAILED rpc=%s type=%s", name, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="SOURCE_ADD workflow temporarily unavailable",
        ) from exc
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Mapping):
        value = value[0]
    if not isinstance(value, Mapping):
        logger.warning("SOURCE_ADD_RPC_INVALID_RESULT rpc=%s", name)
        raise HTTPException(
            status_code=503,
            detail="SOURCE_ADD workflow temporarily unavailable",
        )
    return dict(value)


@router.post(
    "/source-adapters/credential-recipient",
    response_model=ResearchLabCredentialRecipientResponse,
)
async def create_source_add_credential_recipient(
    payload: ResearchLabSourceAddCredentialRecipientRequest,
):
    """Retired: miners never submit provider credentials."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.source_add_enabled, "Research Lab SOURCE_ADD submissions are disabled")
    await _verify_signed_miner(payload)
    raise HTTPException(
        status_code=410,
        detail="SOURCE_ADD miner credentials are not accepted",
    )


@router.post("/source-adapters", response_model=ResearchLabSourceAdapterSubmissionResponse)
async def submit_research_lab_source_adapter(payload: ResearchLabSourceAdapterSubmissionRequest):
    """Atomically reserve a source identity and queue measured provenance."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.miner_submissions_enabled, "Research Lab miner submissions are disabled")
    _require_enabled(config.source_add_enabled, "Research Lab SOURCE_ADD submissions are disabled")
    # SOURCE_ADD intake mints emission rewards, so a maintenance pause on
    # scoring or autoresearch must also stop new source submissions: rewards
    # granted while everything else is frozen only drain the burn share.
    from gateway.research_lab.maintenance import (
        is_autoresearch_maintenance_paused,
        is_scoring_maintenance_paused,
    )

    if await is_scoring_maintenance_paused() or await is_autoresearch_maintenance_paused():
        raise HTTPException(
            status_code=503,
            detail="Research Lab maintenance is active; source adapter intake is paused",
        )
    await _verify_signed_miner(payload)
    await _enforce_research_lab_submission_rate_limit(payload.miner_hotkey, route="source_adapters")

    if payload.adapter_credential is not None or payload.adapter_credential_v2 is not None:
        raise HTTPException(
            status_code=400,
            detail="miners must not submit SOURCE_ADD API credentials",
        )

    source_metadata = payload.source_metadata.model_dump(mode="json")
    declared_domains = (
        payload.manifest.get("declared_base_domains")
        if isinstance(payload.manifest, Mapping)
        else None
    )
    if not isinstance(declared_domains, list) or any(
        not isinstance(item, str) for item in declared_domains
    ):
        raise HTTPException(
            status_code=400,
            detail="SOURCE_ADD declared_base_domains must be a list of domains",
        )
    try:
        source_identity_ref = source_identity_hash_from_metadata(
            source_metadata,
            declared_base_domains=declared_domains,
        )
        source_identity_aliases = source_identity_alias_hashes_from_metadata(
            source_metadata
        )
        legacy_identity_ref = legacy_source_identity_hash(
            api_base_url=str(source_metadata.get("api_base_url") or ""),
            documentation_url=str(source_metadata.get("documentation_url") or ""),
            declared_base_domains=declared_domains,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="SOURCE_ADD source identity is invalid",
        ) from exc

    record, errors = await asyncio.to_thread(
        intake_source_add_submission,
        payload.manifest,
        miner_hotkey=payload.miner_hotkey,
        raw_credential="",
        source_brief=payload.source_brief or "",
        submitted_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        existing_catalog_domains=(),
        existing_source_identity_hashes=(),
        source_identity_ref=source_identity_ref,
        open_submission_count_for_hotkey=0,
        submissions_last_day_for_hotkey=0,
        submissions_last_30d_for_hotkey=0,
    )
    if errors or record is None:
        raise HTTPException(status_code=400, detail="; ".join(errors or ["submission rejected"]))
    record_doc = record.to_dict()
    record_doc["source_metadata"] = source_metadata
    record_doc["source_identity_version"] = SOURCE_ADD_IDENTITY_VERSION
    work_id = source_add_work_id(
        record.submission_id,
        "provenance",
        "%s:%s" % (payload.idempotency_key, payload.timestamp),
    )
    admitted = await _source_add_rpc(
        "research_lab_source_add_admit",
        {
            "p_record_doc": record_doc,
            "p_identity_hash": source_identity_ref,
            "p_documentation_identity_hash": (
                source_identity_aliases[0] if source_identity_aliases else ""
            ),
            "p_legacy_identity_hash": legacy_identity_ref,
            "p_work_id": work_id,
            "p_max_open": int(config.source_add_max_concurrent_per_hotkey),
            "p_max_day": int(config.source_add_max_per_day_per_hotkey),
            "p_max_30d": int(config.source_add_max_per_30d_per_hotkey),
        },
    )
    status = str(admitted.get("status") or "")
    if status == "duplicate":
        raise HTTPException(status_code=409, detail=ALREADY_SUBMITTED_DETAIL)
    if status in {"hotkey_open_cap", "hotkey_day_cap", "hotkey_30d_cap"}:
        raise HTTPException(status_code=429, detail="SOURCE_ADD submission limit reached")
    if status != "admitted":
        logger.warning("SOURCE_ADD_ADMISSION_UNEXPECTED status=%s", status)
        raise HTTPException(status_code=503, detail="SOURCE_ADD workflow temporarily unavailable")

    return ResearchLabSourceAdapterSubmissionResponse(
        submission_id=record.submission_id,
        adapter_id=record.adapter_id,
        stage=str(admitted.get("stage") or "provenance_queued"),
        credential_ref=None,
        precheck_status=None,
        precheck_reasons=[],
    )


def _require_source_add_admin(authorization: str) -> None:
    expected = str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="SOURCE_ADD admin auth is not configured")
    parts = str(authorization or "").strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=403, detail="Forbidden")
    if not parts[1] or not secrets.compare_digest(parts[1], expected):
        raise HTTPException(status_code=403, detail="Forbidden")


def _source_add_submission_parts(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    doc = row.get("submission_doc") if isinstance(row.get("submission_doc"), Mapping) else {}
    manifest_doc = doc.get("manifest") if isinstance(doc.get("manifest"), Mapping) else {}
    source_metadata = doc.get("source_metadata") if isinstance(doc.get("source_metadata"), Mapping) else {}
    if not manifest_doc:
        raise HTTPException(status_code=400, detail="submission manifest is incomplete")
    try:
        source_metadata = ResearchLabSourceMetadata.model_validate(
            source_metadata
        ).model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="submission metadata is incomplete or invalid",
        ) from exc
    submission_id = str(row.get("submission_id") or doc.get("submission_id") or "")
    adapter_id = str(row.get("adapter_id") or doc.get("adapter_id") or manifest_doc.get("adapter_id") or "")
    miner_hotkey = str(row.get("miner_hotkey") or doc.get("miner_hotkey") or manifest_doc.get("miner_ref") or "")
    if not submission_id or not adapter_id or not miner_hotkey:
        raise HTTPException(status_code=400, detail="submission ownership fields are incomplete")
    return dict(doc), dict(manifest_doc), dict(source_metadata)


@router.post(
    "/admin/source-adapters/{submission_id}/recheck-provenance",
    response_model=ResearchLabSourceAdapterRecheckResponse,
)
async def recheck_research_lab_source_adapter_provenance(
    submission_id: str,
    authorization: str = Header(default=""),
):
    """Queue an owner-requested provenance recheck without doing work inline."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.source_add_enabled, "Research Lab SOURCE_ADD submissions are disabled")
    _require_source_add_admin(authorization)

    row = await select_one(
        "research_lab_source_add_submission_current",
        columns=(
            "submission_id,adapter_id,miner_hotkey,stage,seq,submission_doc,precheck_status,"
            "precheck_doc,source_identity_hash,source_identity_version"
        ),
        filters=(("submission_id", submission_id),),
    )
    if not row:
        raise HTTPException(status_code=404, detail="submission not found")
    stage = str(row.get("stage") or "")
    if stage in {"accepted", "rejected", "rejected_precheck", "functional_probe_failed"}:
        raise HTTPException(status_code=400, detail="terminal SOURCE_ADD submission cannot be rechecked")

    _doc, manifest, source_metadata = _source_add_submission_parts(row)
    declared = manifest.get("declared_base_domains") or []
    identity_hash = source_identity_hash_from_metadata(
        source_metadata,
        declared_base_domains=[str(item) for item in declared],
    )
    identity_aliases = source_identity_alias_hashes_from_metadata(source_metadata)
    legacy_hash = legacy_source_identity_hash(
        api_base_url=str(source_metadata.get("api_base_url") or ""),
        documentation_url=str(source_metadata.get("documentation_url") or ""),
        declared_base_domains=[str(item) for item in declared],
    )
    work_id = source_add_work_id(
        submission_id,
        "provenance",
        "operator-recheck:%s" % (int(row.get("seq") or 0) + 1),
    )
    queued = await _source_add_rpc(
        "research_lab_source_add_requeue_provenance",
        {
            "p_submission_id": submission_id,
            "p_identity_hash": identity_hash,
            "p_documentation_identity_hash": (
                identity_aliases[0] if identity_aliases else ""
            ),
            "p_legacy_identity_hash": legacy_hash,
            "p_work_id": work_id,
            "p_actor_ref": "operator:source-add-recheck",
        },
    )
    queue_status = str(queued.get("status") or "")
    if queue_status == "duplicate":
        raise HTTPException(status_code=409, detail=ALREADY_SUBMITTED_DETAIL)
    if queue_status == "missing":
        raise HTTPException(status_code=404, detail="submission not found")
    if queue_status != "queued":
        raise HTTPException(status_code=400, detail="SOURCE_ADD submission cannot be rechecked")
    precheck_doc = row.get("precheck_doc") if isinstance(row.get("precheck_doc"), Mapping) else {}
    return ResearchLabSourceAdapterRecheckResponse(
        submission_id=submission_id,
        adapter_id=str(row.get("adapter_id") or ""),
        stage=str(queued.get("stage") or "provenance_queued"),
        queue_status=queue_status,
        work_id=str(queued.get("work_id") or work_id),
        precheck_status=str(row.get("precheck_status") or "") or None,
        precheck_reasons=[str(item) for item in precheck_doc.get("reasons") or []],
        leg1_reward_status="not_evaluated",
    )


@router.post(
    "/admin/source-adapters/{submission_id}/credential-recipient",
    response_model=ResearchLabCredentialRecipientResponse,
)
async def create_admin_source_add_credential_recipient(
    submission_id: str,
    authorization: str = Header(default=""),
):
    """Return a one-use Nitro recipient scoped to a provenance-passed adapter."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.source_add_enabled, "Research Lab SOURCE_ADD submissions are disabled")
    _require_source_add_admin(authorization)
    row = await select_one(
        "research_lab_source_add_submission_current",
        columns=(
            "submission_id,adapter_id,miner_hotkey,stage,submission_doc,"
            "precheck_status"
        ),
        filters=(("submission_id", submission_id),),
    )
    if not row:
        raise HTTPException(status_code=404, detail="submission not found")
    if str(row.get("precheck_status") or "") != "provenance_precheck_passed":
        raise HTTPException(
            status_code=400,
            detail="SOURCE_ADD provenance pass is required",
        )
    doc = row.get("submission_doc") if isinstance(row.get("submission_doc"), Mapping) else {}
    manifest = doc.get("manifest") if isinstance(doc.get("manifest"), Mapping) else {}
    adapter_id = str(row.get("adapter_id") or doc.get("adapter_id") or manifest.get("adapter_id") or "")
    miner_hotkey = str(row.get("miner_hotkey") or doc.get("miner_hotkey") or manifest.get("miner_ref") or "")
    if not adapter_id or not miner_hotkey:
        raise HTTPException(status_code=400, detail="submission identity is incomplete")
    credential_ref = _source_add_provision_credential_ref(
        miner_hotkey,
        adapter_id,
    )
    return ResearchLabCredentialRecipientResponse(
        **await _source_add_credential_recipient(
            miner_hotkey=miner_hotkey,
            adapter_id=adapter_id,
            credential_ref=credential_ref,
        )
    )


@router.post(
    "/admin/source-adapters/{submission_id}/configure-test",
    response_model=ResearchLabSourceAdapterProbeConfigureResponse,
)
async def configure_research_lab_source_adapter_test(
    submission_id: str,
    payload: ResearchLabSourceAdapterProbeConfigureRequest,
    authorization: str = Header(default=""),
):
    """Persist one exact operator-approved probe and queue its V2 evaluation."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.source_add_enabled, "Research Lab SOURCE_ADD submissions are disabled")
    _require_source_add_admin(authorization)

    row = await select_one(
        "research_lab_source_add_submission_current",
        columns=(
            "submission_id,adapter_id,miner_hotkey,stage,seq,submission_doc,"
            "precheck_status,precheck_doc,source_identity_hash"
        ),
        filters=(("submission_id", submission_id),),
    )
    if not row:
        raise HTTPException(status_code=404, detail="submission not found")
    _doc, _manifest, source_metadata = _source_add_submission_parts(row)
    if str(row.get("precheck_status") or "") != "provenance_precheck_passed":
        raise HTTPException(status_code=400, detail="SOURCE_ADD provenance pass is required")

    adapter_id = str(row.get("adapter_id") or "")
    miner_hotkey = str(row.get("miner_hotkey") or "")
    if payload.base_url.rstrip("/") != str(
        source_metadata.get("api_base_url") or ""
    ).rstrip("/"):
        raise HTTPException(
            status_code=400,
            detail="SOURCE_ADD test base_url must match the submitted API base URL",
        )
    credential_envelope: dict[str, Any] = {}
    if payload.api_credential_v2 is not None:
        credential_ref = _source_add_provision_credential_ref(miner_hotkey, adapter_id)
        credential_envelope = await _seal_source_add_credential_v2(
            encrypted=payload.api_credential_v2,
            miner_hotkey=miner_hotkey,
            adapter_id=adapter_id,
            expected_credential_ref=credential_ref,
        )
        if not source_add_encrypted_envelope_valid(credential_envelope):
            raise HTTPException(
                status_code=500,
                detail="SOURCE_ADD credential sealing returned an invalid envelope",
            )

    probe_doc = {
        "schema_version": "leadpoet.source_add_probe_config.v2",
        "provider_id": "sourceadd_%s" % canonical_hash(
            {"submission_id": submission_id, "adapter_id": adapter_id}
        ).split(":", 1)[1][:16],
        "base_url": payload.base_url.rstrip("/"),
        "auth_kind": payload.auth_kind,
        "auth_name": payload.auth_name or "",
        "request_headers": dict(payload.request_headers),
        "probes": [item.model_dump(mode="json") for item in payload.probes],
    }
    credential_value_hash = str(credential_envelope.get("credential_value_hash") or "")
    config_ref = source_add_probe_config_ref(
        submission_id,
        probe_doc,
        credential_value_hash=credential_value_hash,
    )
    try:
        from gateway.tee.source_add_runtime_v2 import build_source_add_probe_route_v2

        build_source_add_probe_route_v2(
            {
                "submission_id": submission_id,
                "adapter_id": adapter_id,
                "miner_hotkey": miner_hotkey,
                "config_ref": config_ref,
                "probe_doc": probe_doc,
                "credential_envelope": credential_envelope,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail="SOURCE_ADD probe configuration is invalid") from exc

    work_id = source_add_work_id(
        submission_id,
        "functional_probe",
        "operator-config:%s" % config_ref,
    )
    queued = await _source_add_rpc(
        "research_lab_source_add_configure_probe",
        {
            "p_submission_id": submission_id,
            "p_config_ref": config_ref,
            "p_probe_doc": probe_doc,
            "p_credential_envelope": credential_envelope,
            "p_actor_ref": "operator:source-add-configure-test",
            "p_work_id": work_id,
            "p_host_hash": source_add_host_hash(payload.base_url),
        },
    )
    queue_status = str(queued.get("status") or "")
    if queue_status == "missing":
        raise HTTPException(status_code=404, detail="submission not found")
    if queue_status == "terminal":
        raise HTTPException(status_code=400, detail="SOURCE_ADD submission is terminal")
    if queue_status == "provenance_required":
        raise HTTPException(status_code=400, detail="SOURCE_ADD provenance pass is required")
    if queue_status not in {"queued", "already_configured"}:
        raise HTTPException(status_code=503, detail="SOURCE_ADD test could not be queued")
    return ResearchLabSourceAdapterProbeConfigureResponse(
        submission_id=submission_id,
        adapter_id=adapter_id,
        config_ref=config_ref,
        work_id=str(queued.get("work_id") or work_id),
        stage=str(queued.get("stage") or "functional_probe_queued"),
        queue_status=queue_status,
    )


@router.post(
    "/admin/source-adapters/{submission_id}/provision",
    response_model=ResearchLabSourceAdapterProvisionResponse,
)
async def provision_research_lab_source_adapter(
    submission_id: str,
    payload: ResearchLabSourceAdapterProvisionRequest,
    authorization: str = Header(default=""),
):
    """Atomically provision only the exact config that passed V2 testing."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.source_add_enabled, "Research Lab SOURCE_ADD submissions are disabled")
    _require_source_add_admin(authorization)

    status = str(payload.provision_status or "").strip()
    if status not in PROVISION_STATUSES:
        raise HTTPException(status_code=400, detail="invalid provision_status")
    if payload.api_credential_v2 is not None:
        raise HTTPException(
            status_code=400,
            detail="configure and pass the exact SOURCE_ADD test before provisioning",
        )

    row = await select_one(
        "research_lab_source_add_submission_current",
        columns=(
            "submission_id,adapter_id,miner_hotkey,stage,submission_doc,"
            "precheck_status,precheck_doc,source_identity_hash"
        ),
        filters=(("submission_id", submission_id),),
    )
    if not row:
        raise HTTPException(status_code=404, detail="submission not found")
    doc, manifest, source_metadata = _source_add_submission_parts(row)
    adapter_id = str(row.get("adapter_id") or "")
    miner_hotkey = str(row.get("miner_hotkey") or "")
    source_identity_ref = str(row.get("source_identity_hash") or "")
    if not source_identity_ref:
        raise HTTPException(status_code=400, detail="SOURCE_ADD identity reservation is missing")

    reserved_provider_ids = await asyncio.to_thread(reserved_builtin_provider_ids_sync)
    if payload.registry_provider_id in reserved_provider_ids:
        raise HTTPException(status_code=409, detail="registry_provider_id is reserved by a built-in provider")
    if payload.operator_notes:
        try:
            reject_source_add_secret_text(payload.operator_notes, field_name="operator_notes")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    probe_config: Mapping[str, Any] = {}
    probe_doc: Mapping[str, Any] = {}
    credential_envelope: dict[str, Any] = {}
    functional: Mapping[str, Any] = {}
    if status != PROVISION_STATUS_DISABLED:
        probe_config = await select_one(
            "research_lab_source_add_probe_config_current",
            filters=(("submission_id", submission_id), ("config_status", "active")),
        ) or {}
        functional = await select_one(
            "research_lab_source_add_functional_probe_current",
            filters=(("submission_id", submission_id),),
        ) or {}
        probe_doc = probe_config.get("probe_doc") if isinstance(probe_config.get("probe_doc"), Mapping) else {}
        credential_envelope = dict(probe_config.get("credential_envelope") or {})
        if (
            not probe_doc
            or str(functional.get("result_status") or "") != "passed"
            or str(functional.get("config_ref") or "") != str(probe_config.get("config_ref") or "")
        ):
            raise HTTPException(status_code=400, detail="current SOURCE_ADD functional pass is required")
        if payload.base_url and payload.base_url.rstrip("/") != str(probe_doc.get("base_url") or "").rstrip("/"):
            raise HTTPException(status_code=400, detail="provision base_url differs from tested config")
        if "auth_kind" in payload.model_fields_set and payload.auth_kind != str(probe_doc.get("auth_kind") or "none"):
            raise HTTPException(status_code=400, detail="provision auth_kind differs from tested config")
        if payload.auth_name and payload.auth_name != str(probe_doc.get("auth_name") or ""):
            raise HTTPException(status_code=400, detail="provision auth_name differs from tested config")
        if payload.request_headers and payload.request_headers != dict(probe_doc.get("request_headers") or {}):
            raise HTTPException(status_code=400, detail="provision headers differ from tested config")
        if payload.test_probes:
            submitted_tests = [item.model_dump(mode="json") for item in payload.test_probes]
            if submitted_tests != list(probe_doc.get("probes") or []):
                raise HTTPException(status_code=400, detail="provision test probes differ from tested config")

    try:
        probe_objects = [ProviderProbeEndpoint.from_mapping(item) for item in payload.probe_endpoints]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid probe_endpoints") from exc
    probe_errors = validate_probe_catalog(probe_objects)
    if any(item.provider_id != payload.registry_provider_id for item in probe_objects):
        probe_errors.append("probe endpoint provider_id differs from registry_provider_id")
    if probe_errors:
        raise HTTPException(status_code=400, detail="invalid probe_endpoints: " + "; ".join(probe_errors[:5]))
    if status != PROVISION_STATUS_DISABLED:
        tested_routes = {
            (str(item.get("method") or "").upper(), str(item.get("path") or ""))
            for item in probe_doc.get("probes") or []
            if isinstance(item, Mapping)
        }
        provisioned_routes = {(item.method, item.path) for item in probe_objects}
        if not tested_routes or tested_routes != provisioned_routes:
            raise HTTPException(
                status_code=400,
                detail="provisioned routes must exactly match the tested SOURCE_ADD routes",
            )
    probe_endpoints = [item.to_dict() for item in probe_objects]

    existing_provision = await select_one(
        "research_lab_source_add_provisioning_current",
        filters=(("adapter_id", adapter_id),),
    )
    if status == PROVISION_STATUS_DISABLED:
        if not existing_provision:
            raise HTTPException(status_code=400, detail="SOURCE_ADD source is not provisioned")
        existing_doc = existing_provision.get("provision_doc")
        if not isinstance(existing_doc, Mapping):
            raise HTTPException(status_code=503, detail="SOURCE_ADD current provisioning record is invalid")
        provision_doc = dict(existing_doc)
        credential_envelope = dict(existing_provision.get("credential_envelope") or {})
        registry_entry_doc = provision_doc.get("provider_registry_entry")
        if (
            not isinstance(registry_entry_doc, Mapping)
            or str(existing_provision.get("registry_provider_id") or "") != payload.registry_provider_id
        ):
            raise HTTPException(status_code=400, detail="registry_provider_id differs from current source")
    else:
        auth_kind = str(probe_doc.get("auth_kind") or "none")
        auth_name = str(probe_doc.get("auth_name") or "")
        credential_refs = (
            (str(credential_envelope.get("credential_ref") or ""),)
            if credential_envelope
            else ()
        )
        registry_entry = ProviderRegistryEntry(
            id=payload.registry_provider_id,
            base_url=str(probe_doc.get("base_url") or ""),
            auth_kind=auth_kind,
            auth_name=auth_name,
            credential_ref=credential_refs,
            cost_model=dict(payload.cost_model or {}),
            active=status == PROVISION_STATUS_ELIGIBLE,
            capability_policy={
                "routes": [
                    {"method": endpoint.method, "path": endpoint.path}
                    for endpoint in probe_objects
                ],
                "blocked_routes": [],
                "allow_unlisted_paths": False,
                "unlisted_methods": [],
                "model_policy": {"kind": "none"},
            },
            planner_summary={
                "provider_alias": payload.registry_provider_id,
                "endpoint_families": [
                    {
                        "endpoint_id": endpoint.endpoint_id,
                        "description": endpoint.description[:200],
                    }
                    for endpoint in probe_objects
                ],
                "model_policy": "",
                "probe_metadata": [endpoint.endpoint_id for endpoint in probe_objects],
            },
            probe_endpoints=tuple(probe_endpoints),
            origin="source_add",
            reward_eligible=True,
        )
        registry_errors = validate_provider_registry_entries([registry_entry])
        registry_errors.extend(validate_capability_provider_doc(registry_entry.to_dict()))
        if registry_errors:
            raise HTTPException(
                status_code=400,
                detail="invalid provider registry entry: " + "; ".join(registry_errors[:5]),
            )
        provision_doc = sanitize_source_add_doc(
            {
                "provider_registry_entry": registry_entry.to_dict(),
                "probe_endpoints": probe_endpoints,
                "request_headers": dict(probe_doc.get("request_headers") or {}),
                "operator_notes": payload.operator_notes or "",
                "source_metadata": source_metadata,
                "tested_config_ref": str(probe_config.get("config_ref") or ""),
            }
        )

    catalog_id = "source_catalog:" + canonical_hash({"adapter_id": adapter_id}).split(":", 1)[1][:16]
    existing_catalog = await select_one(
        "research_lab_source_catalog",
        columns="catalog_id,adapter_id",
        filters=(("adapter_id", adapter_id),),
    )
    if existing_catalog:
        catalog_id = str(existing_catalog.get("catalog_id") or catalog_id)
    catalog_row = {
        "catalog_id": catalog_id,
        "adapter_id": adapter_id,
        "miner_ref": miner_hotkey,
        "source_name": str(manifest.get("source_name") or "")[:200],
        "source_kind": str(manifest.get("source_kind") or "web"),
        "declared_base_domains": list(manifest.get("declared_base_domains") or []),
        "registry_provider_id": payload.registry_provider_id,
        "catalog_doc": sanitize_source_add_doc(
            {
                "source_metadata": source_metadata,
                "operator_notes": payload.operator_notes or "",
                "registry_provider_id": payload.registry_provider_id,
                "provision_status": status,
            }
        ),
        "source_identity_hash": source_identity_ref,
    }
    if (
        existing_provision
        and str(existing_provision.get("registry_provider_id") or "")
        == payload.registry_provider_id
        and str(existing_provision.get("provision_status") or "") == status
        and dict(existing_provision.get("provision_doc") or {}) == provision_doc
        and dict(existing_provision.get("credential_envelope") or {})
        == credential_envelope
    ):
        return ResearchLabSourceAdapterProvisionResponse(
            submission_id=submission_id,
            adapter_id=adapter_id,
            catalog_id=str(existing_provision.get("catalog_id") or catalog_id),
            registry_provider_id=payload.registry_provider_id,
            provision_status=status,
            provision_ref=str(existing_provision.get("provision_ref") or ""),
            credential_ref=str(credential_envelope.get("credential_ref") or "")
            or None,
        )
    provision_ref = source_add_ref(
        "source_add_provision",
        submission_id,
        payload.registry_provider_id,
        status,
        str(probe_config.get("config_ref") or "disabled"),
        str((existing_provision or {}).get("provision_ref") or "initial"),
        canonical_hash(catalog_row),
        canonical_hash(provision_doc),
    )
    provision_row = {
        "provision_ref": provision_ref,
        "submission_id": submission_id,
        "adapter_id": adapter_id,
        "miner_hotkey": miner_hotkey,
        "source_identity_hash": source_identity_ref,
        "registry_provider_id": payload.registry_provider_id,
        "provision_status": status,
        "provision_doc": provision_doc,
        "credential_envelope": credential_envelope,
    }

    existing_event = await select_one(
        "research_lab_source_add_provisioning_events",
        columns="provision_ref,catalog_id,provision_status",
        filters=(("provision_ref", provision_ref),),
    )
    if existing_event:
        return ResearchLabSourceAdapterProvisionResponse(
            submission_id=submission_id,
            adapter_id=adapter_id,
            catalog_id=str(existing_event.get("catalog_id") or catalog_id),
            registry_provider_id=payload.registry_provider_id,
            provision_status=str(existing_event.get("provision_status") or status),
            provision_ref=provision_ref,
            credential_ref=str(credential_envelope.get("credential_ref") or "")
            or None,
        )

    if status == PROVISION_STATUS_ELIGIBLE:
        _require_enabled(
            config.source_add_functional_probes_enabled,
            "SOURCE_ADD functional probes are disabled",
        )
        pending_doc = copy.deepcopy(provision_doc)
        pending_registry = pending_doc.get("provider_registry_entry")
        if not isinstance(pending_registry, dict):
            raise HTTPException(
                status_code=503,
                detail="SOURCE_ADD pending provisioning document is invalid",
            )
        pending_registry["active"] = False
        pending_catalog_row = copy.deepcopy(catalog_row)
        pending_catalog_doc = pending_catalog_row.get("catalog_doc")
        if isinstance(pending_catalog_doc, dict):
            pending_catalog_doc["provision_status"] = PROVISION_STATUS_APPROVED_PENDING
        pending_ref = source_add_ref(
            "source_add_provision",
            submission_id,
            payload.registry_provider_id,
            PROVISION_STATUS_APPROVED_PENDING,
            str(probe_config.get("config_ref") or ""),
            str((existing_provision or {}).get("provision_ref") or "initial"),
            canonical_hash(pending_catalog_row),
            canonical_hash(pending_doc),
        )
        pending_row = {
            **provision_row,
            "provision_ref": pending_ref,
            "provision_status": PROVISION_STATUS_APPROVED_PENDING,
            "provision_doc": pending_doc,
        }
        pending_matches = bool(
            existing_provision
            and str(existing_provision.get("registry_provider_id") or "")
            == payload.registry_provider_id
            and str(existing_provision.get("provision_status") or "")
            == PROVISION_STATUS_APPROVED_PENDING
            and dict(existing_provision.get("provision_doc") or {}) == pending_doc
            and dict(existing_provision.get("credential_envelope") or {})
            == credential_envelope
        )
        if pending_matches:
            pending_ref = str(existing_provision.get("provision_ref") or pending_ref)
            pending_catalog_id = str(existing_provision.get("catalog_id") or catalog_id)
        else:
            pending = await _source_add_rpc(
                "research_lab_source_add_finalize_provision",
                {
                    "p_submission_id": submission_id,
                    "p_catalog_row": pending_catalog_row,
                    "p_provision_row": pending_row,
                    "p_smoke_attempt": {},
                },
            )
            if str(pending.get("status") or "") not in {
                "provisioned",
                "already_provisioned",
            }:
                raise HTTPException(
                    status_code=503,
                    detail="SOURCE_ADD pending provisioning did not finalize",
                )
            pending_ref = str(pending.get("provision_ref") or pending_ref)
            pending_catalog_id = str(pending.get("catalog_id") or catalog_id)

        provision_ref = source_add_ref(
            "source_add_provision",
            submission_id,
            payload.registry_provider_id,
            status,
            str(probe_config.get("config_ref") or ""),
            pending_ref,
            canonical_hash(catalog_row),
            canonical_hash(provision_doc),
        )
        provision_row["provision_ref"] = provision_ref
        smoke_work_id = source_add_work_id(
            submission_id,
            "provisioning_smoke",
            provision_ref,
        )
        queued = await _source_add_rpc(
            "research_lab_source_add_enqueue_provision_smoke",
            {
                "p_work_id": smoke_work_id,
                "p_submission_id": submission_id,
                "p_config_ref": str(probe_config.get("config_ref") or ""),
                "p_host_hash": source_add_host_hash(
                    str(probe_doc.get("base_url") or "")
                ),
                "p_catalog_row": catalog_row,
                "p_provision_row": provision_row,
            },
        )
        queue_status = str(queued.get("status") or "")
        if queue_status in {
            "current_probe_config_required",
            "pending_approval_required",
        }:
            raise HTTPException(
                status_code=400,
                detail="current SOURCE_ADD functional proof and pending approval are required",
            )
        if queue_status not in {"queued", "already_queued"}:
            raise HTTPException(
                status_code=503,
                detail="SOURCE_ADD provisioning smoke could not be queued",
            )
        return ResearchLabSourceAdapterProvisionResponse(
            submission_id=submission_id,
            adapter_id=adapter_id,
            catalog_id=pending_catalog_id,
            registry_provider_id=payload.registry_provider_id,
            provision_status=PROVISION_STATUS_APPROVED_PENDING,
            provision_ref=pending_ref,
            credential_ref=str(credential_envelope.get("credential_ref") or "")
            or None,
            requested_provision_status=PROVISION_STATUS_ELIGIBLE,
            queue_status=queue_status,
            work_id=str(queued.get("work_id") or smoke_work_id),
        )

    finalized = await _source_add_rpc(
        "research_lab_source_add_finalize_provision",
        {
            "p_submission_id": submission_id,
            "p_catalog_row": catalog_row,
            "p_provision_row": provision_row,
            "p_smoke_attempt": {},
        },
    )
    final_status = str(finalized.get("status") or "")
    if final_status in {"functional_probe_required", "current_probe_config_required", "smoke_test_required"}:
        raise HTTPException(status_code=400, detail="current SOURCE_ADD functional proof is required")
    if final_status == "provision_config_differs_from_test":
        raise HTTPException(status_code=400, detail="provisioning configuration differs from tested config")
    if final_status == "registry_provider_conflict":
        raise HTTPException(status_code=409, detail="registry_provider_id is already in use")
    if final_status in {"missing", "catalog_missing"}:
        raise HTTPException(status_code=404, detail="SOURCE_ADD record not found")
    if final_status not in {"provisioned", "already_provisioned"}:
        raise HTTPException(status_code=503, detail="SOURCE_ADD provisioning did not finalize")
    return ResearchLabSourceAdapterProvisionResponse(
        submission_id=submission_id,
        adapter_id=adapter_id,
        catalog_id=str(finalized.get("catalog_id") or catalog_id),
        registry_provider_id=payload.registry_provider_id,
        provision_status=status,
        provision_ref=str(finalized.get("provision_ref") or provision_ref),
        credential_ref=str(credential_envelope.get("credential_ref") or "") or None,
    )


_TERMINAL_CANDIDATE_STATUSES = {"scored", "rejected", "failed"}


def _as_mapping(value: object) -> dict[str, object]:
    """Coerce a possibly-JSON-string DB column into a dict (empty on failure)."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


async def _build_ticket_candidate_diagnostics(
    ticket_id: str, candidate_id: str | None = None
) -> list[dict[str, object]]:
    """One sanitized diagnostics doc per TERMINAL candidate on the ticket.

    Shared by the signed miner endpoint (own-run) and the internal admin
    dashboard endpoint (any loop). Same builder, same sanitization — the ONLY
    difference between the two callers is the auth gate, so the exact same
    sanitized projection is returned regardless of who asks.
    """
    rows = await select_many(
        "research_lab_candidate_evaluation_current",
        filters=(("ticket_id", ticket_id),),
    )
    candidates = [
        row
        for row in (rows or [])
        if str(row.get("current_candidate_status") or "") in _TERMINAL_CANDIDATE_STATUSES
        and (not candidate_id or str(row.get("candidate_id")) == candidate_id)
    ]

    # Visibility split is deterministic per rolling window; cache across
    # candidates. CRITICAL: each candidate is classified by ITS OWN window's
    # split (keyed by the candidate bundle's icp_set_hash). A multi-day ticket
    # can hold candidates from different windows, and the same ICP can be public
    # in one window and sealed in another — so we must never reuse one window's
    # split for a candidate scored against a different window.
    vis_cache: dict[str, dict[str, str]] = {}
    diagnostics: list[dict[str, object]] = []
    for cand in candidates:
        bundle_doc: dict[str, object] = {}
        bundle_id = str(cand.get("current_score_bundle_id") or "")
        if bundle_id:
            brow = await select_one(
                "research_evaluation_score_bundle_current",
                filters=(("score_bundle_id", bundle_id),),
            )
            if brow:
                bundle_doc = _as_mapping(brow.get("score_bundle_doc"))
        # Fallback: rejected/failed candidates carry no current_score_bundle_id,
        # but a candidate rejected AT the scoring gate still produced a bundle
        # linked by candidate_artifact_hash — recover it so the miner gets the
        # per-ICP deltas + gate result (their most valuable "ran but lost" case).
        if not bundle_doc.get("aggregates"):
            art = str(cand.get("candidate_artifact_hash") or "")
            if art:
                brows = await select_many(
                    "research_evaluation_score_bundles",
                    filters=(("candidate_artifact_hash", art),),
                    order_by=(("created_at", True),),
                    limit=1,
                )
                if brows:
                    fallback = _as_mapping(brows[0].get("score_bundle_doc"))
                    if fallback.get("aggregates"):
                        bundle_doc = fallback
        window = str(bundle_doc.get("icp_set_hash") or "")
        if window and window not in vis_cache:
            bm = await select_one(
                "research_lab_private_model_benchmark_bundles",
                filters=(("rolling_window_hash", window),),
            )
            vis_cache[window] = visibility_map_from_benchmark_split(
                _as_mapping(bm.get("score_summary_doc")) if bm else None
            )
        diagnostics.append(
            build_candidate_diagnostics(
                candidate_id=str(cand.get("candidate_id") or ""),
                bundle_doc=bundle_doc,
                patch_manifest=_as_mapping(cand.get("candidate_patch_manifest")),
                visibility_by_ref=vis_cache.get(window, {}),
                candidate_status=str(cand.get("current_candidate_status") or ""),
                rejection_reason=str(cand.get("current_reason") or ""),
            )
        )
    return diagnostics


async def _build_ticket_run_summaries(
    ticket_id: str, run_id: str | None = None
) -> list[dict[str, Any]]:
    """One sanitized run-level summary per terminal loop run on the ticket.

    Zero-candidate runs produce no candidate diagnostics, so this is the
    owner's only window into WHY a paid loop stopped (stop_reason, last
    stage, call count, actual spend).
    """
    rows = await select_many(
        "research_lab_auto_research_loop_events",
        filters=(("ticket_id", ticket_id),),
        order_by=(("seq", False),),
        limit=2000,
    )
    failure_classes_by_run: dict[str, list[str]] = {}
    terminal_by_run: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for row in rows or []:
        row_run_id = str(row.get("run_id") or "")
        if not row_run_id or (run_id and row_run_id != str(run_id)):
            continue
        event_doc = row.get("event_doc")
        event_doc = event_doc if isinstance(event_doc, Mapping) else {}
        failure_class = str(event_doc.get("failure_class") or "")
        if failure_class:
            classes = failure_classes_by_run.setdefault(row_run_id, [])
            if failure_class not in classes:
                classes.append(failure_class)
        if str(row.get("event_type") or "") in {"loop_completed", "loop_failed"}:
            terminal_by_run[row_run_id] = (str(row.get("loop_status") or ""), event_doc)
    return [
        _sanitized_run_summary_from_terminal_event(
            terminal_run_id,
            loop_status,
            event_doc,
            failure_classes_by_run.get(terminal_run_id, []),
        )
        for terminal_run_id, (loop_status, event_doc) in sorted(terminal_by_run.items())
    ]


@router.post("/loop-diagnostics")
async def get_research_lab_loop_diagnostics(payload: ResearchLabLoopDiagnosticsRequest):
    """Sanitized own-run diagnostics for the miner who paid for the loop.

    Ownership-gated: the ticket must belong to the signing hotkey, so a miner
    can only read diagnostics for their own runs. Returns one diagnostics doc
    per TERMINAL candidate on the ticket (scored / rejected / failed), plus a
    run-level summary per terminal loop run — zero-candidate runs have no
    candidate diagnostics, and the run summary (stop_reason, last stage,
    call count, actual spend) is the owner's only way to diagnose them.
    """
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    await _verify_signed_miner(payload)
    # Ownership: 404 unless the ticket belongs to this hotkey.
    await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)

    diagnostics = await _build_ticket_candidate_diagnostics(
        str(payload.ticket_id), payload.candidate_id
    )
    run_summaries = await _build_ticket_run_summaries(str(payload.ticket_id))
    if not diagnostics and not run_summaries:
        raise HTTPException(status_code=404, detail="no terminal diagnostics for this ticket yet")
    return {
        "ticket_id": str(payload.ticket_id),
        "diagnostics": diagnostics,
        "run_summaries": run_summaries,
    }


@router.get("/admin/loops/{ticket_id}/diagnostics")
async def get_research_lab_admin_loop_diagnostics(
    ticket_id: str,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    """Loop-detail diagnostics for the internal admin dashboard.

    Internal-key gated (team-only), so unlike the public activity feed this may
    carry the full per-candidate diagnostics. Combines the loop's event
    timeline (the timestamps already shown when a loop is clicked) with the
    per-candidate diagnostics the scorer now captures — one payload the
    dashboard renders in the loop-detail panel, in line with the benchmark
    report. Reuses the exact sanitized builder used by the miner endpoint.
    """
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_internal_key(config, x_leadpoet_internal_key)

    try:
        detail = await fetch_public_loop_detail(ticket_id)
        diagnostics = await _build_ticket_candidate_diagnostics(ticket_id)
    except Exception as exc:
        _raise_storage_error(exc)
    if not detail and not diagnostics:
        raise HTTPException(status_code=404, detail="Research Lab loop not found")
    return {
        "schema_version": "1.0",
        "ticket_id": ticket_id,
        # event timeline the loop-click view already renders (timestamps)
        "card": (detail or {}).get("card"),
        "events": (detail or {}).get("events", []),
        # the per-candidate diagnostics now captured (funnel, patch, delta)
        "candidate_diagnostics": diagnostics,
    }


@router.post(
    "/openrouter-keys/credential-recipient",
    response_model=ResearchLabOpenRouterCredentialRecipientsResponse,
)
async def create_openrouter_credential_recipients(
    payload: ResearchLabOpenRouterCredentialRecipientRequest,
):
    """Return one-use Nitro-attested recipients for the miner key pair."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(
        config.miner_submissions_enabled,
        "Research Lab miner submissions are disabled",
    )
    await _verify_signed_miner(payload)
    runtime, management, release_evidence = await asyncio.gather(
        _openrouter_credential_recipient_v2(
            miner_hotkey=payload.miner_hotkey,
            credential_kind="runtime",
        ),
        _openrouter_credential_recipient_v2(
            miner_hotkey=payload.miner_hotkey,
            credential_kind="management",
        ),
        _openrouter_credential_release_evidence_v2(),
    )
    return ResearchLabOpenRouterCredentialRecipientsResponse(
        runtime=ResearchLabOpenRouterCredentialRecipientV2(**runtime),
        management=ResearchLabOpenRouterCredentialRecipientV2(**management),
        release_evidence=ResearchLabOpenRouterCredentialReleaseEvidenceV2(
            **release_evidence
        ),
    )


async def _register_openrouter_key_legacy(
    payload: ResearchLabOpenRouterKeyRegisterRequest,
    config: ResearchLabGatewayConfig,
) -> ResearchLabOpenRouterKeyRegisterResponse:
    if payload.openrouter_api_key_v2 or payload.openrouter_management_key_v2:
        raise HTTPException(
            status_code=400,
            detail="attested OpenRouter credentials require V2 mode",
        )
    runtime_key = str(payload.openrouter_api_key or "")
    management_key = str(payload.openrouter_management_key or "")
    if not runtime_key or not management_key:
        raise HTTPException(
            status_code=400,
            detail="plaintext OpenRouter credential pair is required in legacy mode",
        )
    if not config.openrouter_key_kms_key_id:
        raise HTTPException(
            status_code=503,
            detail="Research Lab OpenRouter key vault KMS key is not configured",
        )
    try:
        preflight_doc = await asyncio.to_thread(preflight_openrouter_key, runtime_key)
        key_hash = str(preflight_doc["key_hash"])
        privacy_proof_doc = await asyncio.to_thread(
            verify_openrouter_workspace_privacy,
            runtime_key=runtime_key,
            management_key=management_key,
            stage="key_registration",
        )
        management_key_hash = str(privacy_proof_doc["management_key_hash"])
        key_ref = openrouter_key_ref(
            miner_hotkey=payload.miner_hotkey,
            key_hash=key_hash,
            management_key_hash=management_key_hash,
        )
        encrypted = await asyncio.to_thread(
            encrypt_openrouter_key,
            raw_key=runtime_key,
            kms_key_id=config.openrouter_key_kms_key_id,
            miner_hotkey=payload.miner_hotkey,
            key_ref=key_ref,
        )
        encrypted_management = await asyncio.to_thread(
            encrypt_openrouter_key,
            raw_key=management_key,
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
            encrypted_management_key_ciphertext=(
                encrypted_management["ciphertext_b64"]
            ),
            management_key_hash=management_key_hash,
            management_kms_key_id=encrypted_management["kms_key_id"],
            management_encryption_context_hash=(
                encrypted_management["encryption_context_hash"]
            ),
            openrouter_workspace_hash=str(
                privacy_proof_doc["workspace_id_hash"]
            ),
            privacy_status="verified",
            privacy_verified_at=str(privacy_proof_doc["verified_at"]),
            privacy_proof_doc=privacy_proof_doc,
            preflight_doc={
                "source": "openrouter_current_key",
                "limit": preflight_doc.get("limit"),
                "limit_remaining": preflight_doc.get("limit_remaining"),
                "limit_reset": preflight_doc.get("limit_reset"),
                "usage": preflight_doc.get("usage"),
                "is_free_tier": preflight_doc.get("is_free_tier"),
                "is_management_key": preflight_doc.get("is_management_key"),
                "expires_at": preflight_doc.get("expires_at"),
                "key_label_hash": preflight_doc.get("key_label_hash"),
                "creator_user_id_hash": preflight_doc.get(
                    "creator_user_id_hash"
                ),
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


@router.post("/openrouter-keys", response_model=ResearchLabOpenRouterKeyRegisterResponse)
async def register_research_lab_openrouter_key(payload: ResearchLabOpenRouterKeyRegisterRequest):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.miner_submissions_enabled, "Research Lab miner submissions are disabled")
    await _verify_signed_miner(payload)
    _enforce_openrouter_key_registration_rate_limit(payload.miner_hotkey)

    if legacy_v1_enabled():
        return await _register_openrouter_key_legacy(payload, config)

    if payload.openrouter_api_key or payload.openrouter_management_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "plaintext OpenRouter credentials are retired; use the "
                "attested credential-recipient endpoint"
            ),
        )
    if (
        payload.openrouter_api_key_v2 is None
        or payload.openrouter_management_key_v2 is None
    ):
        raise HTTPException(
            status_code=400,
            detail="attested OpenRouter credential pair is required",
        )

    try:
        runtime_credential, management_credential = await asyncio.gather(
            _seal_openrouter_credential_v2(
                encrypted=payload.openrouter_api_key_v2,
                miner_hotkey=payload.miner_hotkey,
                credential_kind="runtime",
            ),
            _seal_openrouter_credential_v2(
                encrypted=payload.openrouter_management_key_v2,
                miner_hotkey=payload.miner_hotkey,
                credential_kind="management",
            ),
        )
        evaluation_epoch, _block, _source = await resolve_research_lab_evaluation_epoch(
            config.evaluation_epoch
        )
        from gateway.research_lab.attested_coordinator_v2 import (
            register_openrouter_credentials_v2,
        )
        from gateway.tee.openrouter_credential_v2 import (
            COORDINATOR_SEALED_KEY_ID,
            OPENROUTER_REGISTRATION_RESULT_SCHEMA_VERSION,
        )

        authority = await register_openrouter_credentials_v2(
            miner_hotkey=payload.miner_hotkey,
            key_label=payload.key_label,
            runtime_credential=runtime_credential,
            management_credential=management_credential,
            epoch_id=int(evaluation_epoch),
            sequence=int(payload.timestamp),
        )
        result = authority.get("result")
        if (
            not isinstance(result, Mapping)
            or result.get("schema_version")
            != OPENROUTER_REGISTRATION_RESULT_SCHEMA_VERSION
        ):
            raise RuntimeError("attested OpenRouter registration result is invalid")
        key_ref = str(result["key_ref"])
        key_hash = str(result["key_hash"])
        management_key_hash = str(result["management_key_hash"])
        preflight_doc = dict(result["preflight_doc"])
        privacy_proof_doc = dict(result["privacy_proof_doc"])
        envelopes = result.get("credential_envelopes")
        if (
            not isinstance(envelopes, list)
            or len(envelopes) != 2
            or {str(item.get("credential_kind") or "") for item in envelopes}
            != {"runtime", "management"}
        ):
            raise RuntimeError("attested OpenRouter credential envelopes are invalid")
        envelope_by_kind = {
            str(item["credential_kind"]): dict(item) for item in envelopes
        }
        runtime_envelope = envelope_by_kind["runtime"]
        management_envelope = envelope_by_kind["management"]
        await create_openrouter_key_ref(
            key_ref=key_ref,
            miner_hotkey=payload.miner_hotkey,
            key_hash=key_hash,
            encrypted_key_ciphertext=runtime_envelope["ciphertext_blob_b64"],
            kms_key_id=COORDINATOR_SEALED_KEY_ID,
            encryption_context_hash=runtime_envelope["encryption_context_hash"],
            encrypted_management_key_ciphertext=management_envelope["ciphertext_blob_b64"],
            management_key_hash=management_key_hash,
            management_kms_key_id=COORDINATOR_SEALED_KEY_ID,
            management_encryption_context_hash=management_envelope["encryption_context_hash"],
            openrouter_workspace_hash=str(privacy_proof_doc["workspace_id_hash"]),
            privacy_status="verified",
            privacy_verified_at=str(privacy_proof_doc["verified_at"]),
            privacy_proof_doc=privacy_proof_doc,
            preflight_doc={
                "source": "openrouter_current_key",
                "limit": preflight_doc.get("limit"),
                "limit_remaining": preflight_doc.get("limit_remaining"),
                "limit_reset": preflight_doc.get("limit_reset"),
                "usage": preflight_doc.get("usage"),
                "is_free_tier": preflight_doc.get("is_free_tier"),
                "is_management_key": preflight_doc.get("is_management_key"),
                "expires_at": preflight_doc.get("expires_at"),
                "key_label_hash": preflight_doc.get("key_label_hash"),
                "creator_user_id_hash": preflight_doc.get("creator_user_id_hash"),
                "key_label": payload.key_label,
            },
        )
        for envelope in envelopes:
            await persist_openrouter_credential_envelope_v2(
                envelope
            )
    except OpenRouterKeyVaultError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        from gateway.research_lab.attested_scoring_v2 import (
            AttestedScoringV2Error,
        )

        if isinstance(exc, AttestedScoringV2Error):
            authority_doc = exc.authority or {}
            attempts = authority_doc.get("transport_attempts") or []
            if any(
                isinstance(item, Mapping)
                and item.get("terminal_status") == "transport_failure"
                for item in attempts
            ):
                raise HTTPException(
                    status_code=503,
                    detail="OpenRouter credential verification transport is unavailable",
                ) from exc
            raise HTTPException(
                status_code=400,
                detail="OpenRouter credential verification failed",
            ) from exc
        _raise_storage_error(exc)

    return ResearchLabOpenRouterKeyRegisterResponse(
        key_ref=key_ref,
        preflight_status="passed",
        key_hash=key_hash,
        limit_remaining=preflight_doc.get("limit_remaining"),
        limit_reset=preflight_doc.get("limit_reset"),
    )


@router.post("/resume-credit-blocked", response_model=ResearchLabResumeCreditBlockedResponse)
async def resume_research_lab_credit_blocked(payload: ResearchLabResumeCreditBlockedRequest):
    """Miner self-service: after topping up their OpenRouter key, re-queue their
    credit-blocked (paused/blocked_for_credit) runs. Resumes from checkpoint (or
    restarts if stale) without consuming another loop-start payment. The hosted
    worker's credit preflight is the final gate, so a still-unfunded key just re-pauses.
    Core logic lives in recovery.resume_credit_blocked_runs_for_miner (fastapi-free)."""
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.miner_submissions_enabled, "Research Lab miner submissions are disabled")
    await _verify_signed_miner(payload)
    await _require_autoresearch_not_paused()

    from .recovery import resume_credit_blocked_runs_for_miner

    result = await resume_credit_blocked_runs_for_miner(payload.miner_hotkey, run_ids=payload.run_ids)
    return ResearchLabResumeCreditBlockedResponse(**result)


@router.post("/loop-start", response_model=ResearchLabLoopStartResponse)
async def start_research_lab_paid_loop(payload: ResearchLabLoopStartRequest):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.miner_submissions_enabled, "Research Lab miner submissions are disabled")
    _require_enabled(config.paid_loops_enabled, "Research Lab paid loops are disabled")
    _require_enabled(config.hosted_runs_enabled, "Research Lab hosted runs are disabled")
    await _verify_signed_miner(payload)
    await _require_autoresearch_not_paused()
    if config.miner_openrouter_key_required and payload.miner_openrouter_preflight_status != "passed":
        raise HTTPException(status_code=400, detail="miner OpenRouter key preflight must pass before queueing")
    ticket = await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)
    await _require_ticket_mutable(ticket, enforce_unpaid_deadline=not bool(payload.credit_id))
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
        await _require_autoresearch_not_paused()
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
        credit_id = await _preserve_loop_start_credit_after_queue_failure(
            ticket_id=str(payload.ticket_id),
            payment_id=str(payment["payment_id"]),
            payment_ref=payment_ref,
            miner_hotkey=payload.miner_hotkey,
            run_id=run_id,
            reason="queue_failed_after_credit_consumed" if consumed_credit else "queue_failed_before_work_started",
            error=exc,
            replaces_credit_id=payload.credit_id,
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
    _require_enabled(config.miner_submissions_enabled, "Research Lab miner submissions are disabled")
    _require_enabled(config.paid_loops_enabled, "Research Lab paid loops are disabled")
    _require_enabled(config.hosted_runs_enabled, "Research Lab hosted runs are disabled")
    _require_enabled(config.loop_topups_enabled, "Research Lab loop top-ups are disabled for launch")
    await _verify_signed_miner(payload)
    await _require_autoresearch_not_paused()
    if config.miner_openrouter_key_required and payload.miner_openrouter_preflight_status != "passed":
        raise HTTPException(status_code=400, detail="miner OpenRouter key preflight must pass before queueing")
    ticket = await _get_ticket_for_miner(str(payload.ticket_id), payload.miner_hotkey)
    await _require_ticket_mutable(ticket, enforce_unpaid_deadline=False)
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
        await _require_autoresearch_not_paused()
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
        credit_id = await _preserve_loop_start_credit_after_queue_failure(
            ticket_id=str(payload.ticket_id),
            payment_id=str(payment["payment_id"]),
            payment_ref=payment_ref,
            miner_hotkey=payload.miner_hotkey,
            run_id=run_id,
            reason="topup_queue_failed_before_work_started",
            error=exc,
            replaces_credit_id=None,
        )
        logger.exception("Research Lab top-up queue failed after payment; retry credit preserved")
        return ResearchLabLoopTopUpResponse(
            ticket_id=str(payload.ticket_id),
            run_id=run_id,
            continued_from_run_id=str(payload.continue_from_run_id) if payload.continue_from_run_id else None,
            topup_payment_id=payment["payment_id"],
            payment_ref=payment_ref,
            queued=False,
            credit_preserved=True,
            credit_id=credit_id,
            status="credit_preserved_after_topup_queue_failure",
        )

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


@router.get("/public/loops/summary")
async def get_research_lab_public_loop_summary(
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
        summary = await fetch_public_loop_summary(
            status=status,
            topic=topic,
            research_area=research_area,
            since_days=since_days,
        )
    except Exception as exc:
        _raise_storage_error(exc)
    return {
        "schema_version": "1.0",
        "summary": summary,
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


@router.get("/engine/issues")
async def list_research_lab_engine_issues(
    status: str = Query(default="open", max_length=32),
    priority: Optional[str] = Query(default=None, max_length=32),
    limit: int = Query(default=100, ge=1, le=500),
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    _require_improvement_engine_enabled()
    filters: list[tuple[str, Any]] = [("status", status)]
    if priority:
        filters.append(("priority", priority))
    rows = await select_many(
        "engine_issues",
        filters=tuple(filters),
        order_by=(("last_seen_at", True),),
        limit=limit,
    )
    return {"issues": [_engine_issue_public(row) for row in rows]}


@router.get("/engine/issues/{issue_id}")
async def get_research_lab_engine_issue(
    issue_id: str,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    _require_improvement_engine_enabled()
    row = await select_one("engine_issues", filters=(("id", issue_id),))
    if not row:
        row = await select_one("engine_issues", filters=(("issue_key", issue_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab Engine issue not found")
    return {"issue": _engine_issue_public(row)}


@router.post("/engine/issues/{issue_id}/status")
async def update_research_lab_engine_issue_status(
    issue_id: str,
    payload: dict[str, Any],
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    _require_improvement_engine_enabled()
    reject_secret_material(payload)
    status = str(payload.get("status") or "").strip()
    if status not in {"open", "in_review", "fixed", "ignored", "reopened"}:
        raise HTTPException(status_code=400, detail="unsupported Engine issue status")
    row = await select_one("engine_issues", filters=(("id", issue_id),))
    if not row:
        row = await select_one("engine_issues", filters=(("issue_key", issue_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab Engine issue not found")
    updated = await update_row(
        "engine_issues",
        {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()},
        filters=(("id", row["id"]),),
    )
    event = await insert_row(
        "engine_issue_events",
        {
            "issue_id": row["id"],
            "event_type": "status_changed",
            "event_doc": {
                "old_status": row.get("status"),
                "new_status": status,
                "reason": str(payload.get("reason") or "")[:500],
            },
        },
    )
    return {"issue": _engine_issue_public(updated), "event_id": event["id"]}


@router.post("/engine/scans")
async def run_research_lab_engine_scan(
    payload: Optional[dict[str, Any]] = None,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    engine_config = _require_improvement_engine_enabled()
    payload = payload or {}
    reject_secret_material(payload)
    result = await scan_for_issues(
        config=engine_config,
        dry_run=bool(payload.get("dry_run", True)),
        persist=bool(payload.get("persist", False)),
    )
    return result


@router.post("/engine/issues/{issue_id}/generate-fix")
async def generate_research_lab_engine_fix(
    issue_id: str,
    payload: Optional[dict[str, Any]] = None,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    engine_config = _require_improvement_engine_enabled()
    payload = payload or {}
    reject_secret_material(payload)
    row = await select_one("engine_issues", filters=(("id", issue_id),))
    if not row:
        row = await select_one("engine_issues", filters=(("issue_key", issue_id),))
    if not row:
        raise HTTPException(status_code=404, detail="Research Lab Engine issue not found")
    fix_doc = row.get("suggested_fix_doc") if isinstance(row.get("suggested_fix_doc"), Mapping) else {}
    response: dict[str, Any] = {
        "issue_id": row["id"],
        "issue_key": row["issue_key"],
        "dry_run": bool(payload.get("dry_run", True)),
        "auto_generate_patches_enabled": engine_config.auto_generate_patches,
        "fix_doc": fix_doc,
    }
    if bool(payload.get("include_miner_opportunity", False)):
        from research_lab.improvement_engine.models import EngineIssue

        issue = EngineIssue(
            issue_key=str(row["issue_key"]),
            title=str(row["title"]),
            status=str(row["status"]),
            priority=str(row["priority"]),
            category=str(row["category"]),
            fingerprint=str(row["fingerprint"]),
            first_seen_at=str(row["first_seen_at"]),
            last_seen_at=str(row["last_seen_at"]),
            occurrence_count=int(row.get("occurrence_count") or 0),
            severity_score=float(row.get("severity_score") or 0),
            confidence=float(row.get("confidence") or 0),
            root_cause_doc=dict(row.get("root_cause_doc") or {}),
            suggested_fix_doc=dict(fix_doc),
            evaluator_spec_doc=dict(row.get("evaluator_spec_doc") or {}),
            dataset_spec_doc=dict(row.get("dataset_spec_doc") or {}),
        )
        response["miner_opportunity"] = sanitized_miner_opportunity(issue)
    return response


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


@router.get("/reports/candidate-generation-failures")
async def get_research_lab_candidate_generation_failure_report(
    days: int = Query(default=7, ge=0, le=90),
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    try:
        return await fetch_candidate_generation_failure_report(days)
    except Exception as exc:
        _raise_storage_error(exc)


@router.get("/reports/daily-noise-budget/latest")
async def get_latest_research_lab_daily_noise_budget_report(
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    try:
        rows = await select_many(
            "research_lab_daily_noise_budget_report_current",
            filters=(),
            order_by=(("benchmark_date", True), ("benchmark_attempt", True), ("created_at", True)),
            limit=1,
        )
    except Exception as exc:
        _raise_storage_error(exc)
    if not rows:
        raise HTTPException(status_code=404, detail="Research Lab daily noise budget report not found")
    return rows[0]


@router.get("/reports/daily-noise-budget/{benchmark_date}")
async def get_research_lab_daily_noise_budget_report_by_date(
    benchmark_date: str,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_internal_key(config, x_leadpoet_internal_key)
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", benchmark_date):
        raise HTTPException(status_code=400, detail="benchmark_date must be YYYY-MM-DD")
    try:
        rows = await select_many(
            "research_lab_daily_noise_budget_report_current",
            filters=(("benchmark_date", benchmark_date),),
            order_by=(("benchmark_attempt", True), ("created_at", True)),
            limit=10,
        )
    except Exception as exc:
        _raise_storage_error(exc)
    if not rows:
        raise HTTPException(status_code=404, detail="Research Lab daily noise budget report not found")
    return {
        "schema_version": "1.0",
        "benchmark_date": benchmark_date,
        "reports": rows,
    }


@router.get("/audit/latest/{epoch}")
async def get_research_lab_latest_audit_bundle(
    epoch: int,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
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

    _require_internal_key(config, x_leadpoet_internal_key)

    ticket_rows = await _audit_preview_select_all("research_loop_ticket_current", current_view=True)
    queue_rows = await _audit_preview_select_all("research_loop_run_queue_current", current_view=True)
    receipt_rows = await _audit_preview_select_all("research_loop_receipt_current", current_view=True)
    candidate_rows = await _audit_preview_select_all("research_lab_candidate_evaluation_current", current_view=True)
    candidate_event_rows = await _audit_preview_select_all("research_lab_candidate_evaluation_events")
    loop_event_rows = await _audit_preview_select_all("research_lab_auto_research_loop_events")
    dispatch_event_rows = await _audit_preview_select_all("research_lab_scoring_dispatch_events")
    rolling_window_rows = await _audit_preview_select_all("research_lab_rolling_icp_windows")
    benchmark_rows = await _audit_preview_select_all("research_lab_private_model_benchmark_current", current_view=True)
    private_model_version_rows = await _audit_preview_select_all(
        "research_lab_private_model_version_current",
        current_view=True,
    )
    promotion_event_rows = await _audit_preview_select_all("research_lab_candidate_promotion_events")
    private_repo_commit_event_rows = await _audit_preview_select_all("research_lab_private_repo_commit_events")
    public_benchmark_report_rows = await _audit_preview_select_all(
        "research_lab_public_benchmark_report_current",
        current_view=True,
    )
    score_bundle_rows = await _audit_preview_select_all(
        "research_evaluation_score_bundle_current",
        filters=(("evaluation_epoch", epoch),),
        current_view=True,
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


async def _allocation_epoch_guard_and_persistence(
    config: ResearchLabGatewayConfig,
    epoch: int,
    internal_key: Optional[str],
) -> bool:
    """Reject future epochs; return whether this request may persist a snapshot.

    Anonymous GETs are read-only: an unauthenticated caller could otherwise
    mint active snapshots for arbitrary epochs (future rows for four epochs
    ahead were found persisted this way), which contaminates paid-to-date
    accounting. Only the authenticated validator path persists, and only for
    the current epoch it is about to submit.
    """
    from gateway.research_lab.allocations import allocation_snapshot_persistence_decision
    from gateway.utils.epoch import get_current_epoch_id_async

    try:
        current_epoch = await get_current_epoch_id_async()
    except Exception as exc:  # noqa: BLE001 - chain lookup outage must not break reads
        # Fail safe, not open: without the chain epoch we cannot prove the
        # requested epoch isn't in the future, so serve the computation but
        # never persist. The validator retries next cycle once the chain
        # lookup recovers, so persistence is delayed, not lost.
        logger.warning(
            "research_lab_allocation_epoch_guard_degraded epoch=%s error=%s",
            int(epoch),
            str(exc)[:120],
        )
        return False
    normalized_key = internal_key if isinstance(internal_key, str) else None
    decision = allocation_snapshot_persistence_decision(
        current_epoch=int(current_epoch),
        requested_epoch=int(epoch),
        provided_key=normalized_key,
        configured_key=str(getattr(config, "internal_api_key", "") or ""),
        live_allocation_enabled=bool(config.reimbursements_enabled or config.weight_mutation_enabled),
    )
    if decision == "future_epoch":
        raise HTTPException(
            status_code=422,
            detail=f"allocation epoch {int(epoch)} is in the future (current {int(current_epoch)})",
        )
    if decision == "key_not_configured":
        raise HTTPException(status_code=403, detail="Research Lab internal API key is not configured")
    if decision == "invalid_key":
        raise HTTPException(status_code=401, detail="invalid Research Lab internal API key")
    return decision == "persist"


@router.get("/allocations/live/{epoch}")
async def get_research_lab_live_allocation(
    epoch: int,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.shadow_bundles_enabled, "Research Lab report bundles are disabled")
    persist_snapshot = await _allocation_epoch_guard_and_persistence(
        config, int(epoch), x_leadpoet_internal_key
    )
    try:
        return await build_research_lab_allocation_bundle(
            config=config,
            epoch=int(epoch),
            netuid=BITTENSOR_NETUID,
            persist_snapshot=persist_snapshot,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research Lab allocation unavailable: {str(exc)[:200]}") from exc


# Assembling one attested allocation bundle is expensive: it reconstructs the
# full ancestry receipt graph (hundreds of chunked reads over the large
# receipt tables) and takes tens of seconds. The validator polls this endpoint
# inside a fixed on-chain submission window and retries on slow responses, so
# without coordination every retry — and every concurrent poll — launches a
# fresh rebuild. The rebuilds then contend for the database pool and the
# enclave, each one slowing past the validator's fetch timeout, so the
# validator never receives an allocation and its fail-closed guard blocks the
# weight submission for the whole epoch. The assembled bundle is deterministic
# for a given epoch (only a cosmetic bundle_id and generated_at timestamp vary
# between builds), so it is safe to build it at most once per epoch and serve
# every other caller from that result.
_AllocationCacheKey = tuple[int, bool]
_ALLOCATION_HANDOFF_CACHE: "OrderedDict[_AllocationCacheKey, tuple[float, dict[str, Any]]]" = (
    OrderedDict()
)
_ALLOCATION_BUILD_LOCKS: dict[_AllocationCacheKey, asyncio.Lock] = {}
_ALLOCATION_CACHE_TTL_SECONDS = 300.0
_ALLOCATION_CACHE_MAX_EPOCHS = 16


def _allocation_cache_key(epoch: int, persist_snapshot: bool) -> _AllocationCacheKey:
    return (int(epoch), bool(persist_snapshot))


def _allocation_handoff_cache_get(
    epoch: int,
    persist_snapshot: bool,
) -> Optional[dict[str, Any]]:
    key = _allocation_cache_key(epoch, persist_snapshot)
    entry = _ALLOCATION_HANDOFF_CACHE.get(key)
    if entry is None:
        return None
    expires_at, handoff = entry
    if time.monotonic() >= expires_at:
        _ALLOCATION_HANDOFF_CACHE.pop(key, None)
        return None
    _ALLOCATION_HANDOFF_CACHE.move_to_end(key)
    return handoff


def _allocation_handoff_cache_put(
    epoch: int,
    persist_snapshot: bool,
    handoff: dict[str, Any],
) -> None:
    key = _allocation_cache_key(epoch, persist_snapshot)
    _ALLOCATION_HANDOFF_CACHE[key] = (
        time.monotonic() + _ALLOCATION_CACHE_TTL_SECONDS,
        handoff,
    )
    _ALLOCATION_HANDOFF_CACHE.move_to_end(key)
    while len(_ALLOCATION_HANDOFF_CACHE) > _ALLOCATION_CACHE_MAX_EPOCHS:
        evicted, _ = _ALLOCATION_HANDOFF_CACHE.popitem(last=False)
        _ALLOCATION_BUILD_LOCKS.pop(evicted, None)


def _allocation_build_lock(
    epoch: int,
    persist_snapshot: bool,
) -> asyncio.Lock:
    key = _allocation_cache_key(epoch, persist_snapshot)
    lock = _ALLOCATION_BUILD_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _ALLOCATION_BUILD_LOCKS[key] = lock
    return lock


@router.get("/allocations/attested/{epoch}")
async def get_research_lab_attested_allocation(
    epoch: int,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    """Return the unchanged live allocation plus its enclave-signed sidecar."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.shadow_bundles_enabled, "Research Lab report bundles are disabled")
    # The guard rejects future epochs and decides snapshot persistence; it must
    # run on every request and is cheap relative to the bundle build.
    persist_snapshot = await _allocation_epoch_guard_and_persistence(
        config, int(epoch), x_leadpoet_internal_key
    )
    cached_handoff = _allocation_handoff_cache_get(
        int(epoch),
        persist_snapshot,
    )
    if cached_handoff is not None:
        return cached_handoff
    # Serialize builds per epoch so concurrent polls and retries wait for one
    # rebuild instead of launching contending ones. Re-check the cache after
    # acquiring the lock: the first builder populates it for everyone waiting.
    async with _allocation_build_lock(int(epoch), persist_snapshot):
        cached_handoff = _allocation_handoff_cache_get(
            int(epoch),
            persist_snapshot,
        )
        if cached_handoff is not None:
            return cached_handoff
        return await _build_and_cache_attested_allocation(
            config=config,
            epoch=int(epoch),
            persist_snapshot=persist_snapshot,
        )


async def _build_and_cache_attested_allocation(
    *,
    config: "ResearchLabGatewayConfig",
    epoch: int,
    persist_snapshot: bool,
) -> dict[str, Any]:
    attestation: dict[str, Any] = {}
    try:
        bundle = await build_research_lab_allocation_bundle(
            config=config,
            epoch=int(epoch),
            netuid=BITTENSOR_NETUID,
            persist_snapshot=persist_snapshot,
            attestation_out=attestation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Research Lab attested allocation unavailable: {str(exc)[:200]}",
        ) from exc
    if attestation.get("status") != "matched":
        raise HTTPException(
            status_code=503,
            detail=f"Research Lab attested allocation is not ready: {attestation.get('status', 'unknown')}",
        )
    receipt = attestation.get("execution_receipt") or attestation.get("receipt")
    receipt_graph = attestation.get("receipt_graph")
    lineage_bindings = attestation.get("lineage_bindings")
    lineage_complete = attestation.get("lineage_complete")
    persistence = attestation.get("persistence")
    if (
        not isinstance(receipt, Mapping)
        or not isinstance(receipt_graph, Mapping)
        or not isinstance(lineage_bindings, list)
        or lineage_complete is not True
        or not isinstance(persistence, Mapping)
    ):
        raise HTTPException(status_code=503, detail="Research Lab attested allocation receipt is incomplete")
    from leadpoet_canonical.allocation_handoff_v2 import (
        build_allocation_handoff_v2,
    )

    try:
        if receipt_graph.get("root_receipt_hash") != receipt.get("receipt_hash"):
            from gateway.research_lab.attested_v2_store import (
                load_receipt_graph_v2,
            )
            from leadpoet_canonical.attested_v2 import sha256_json

            receipt_graph = await load_receipt_graph_v2(
                str(receipt["receipt_hash"])
            )
            persistence = {
                "graph_hash": sha256_json(dict(receipt_graph)),
                "root_receipt_hash": str(receipt_graph["root_receipt_hash"]),
                "boot_count": len(receipt_graph["boot_identities"]),
                "receipt_count": len(receipt_graph["receipts"]),
                "transport_attempt_count": len(
                    receipt_graph["transport_attempts"]
                ),
                "host_operation_count": len(
                    receipt_graph["host_operations"]
                ),
            }
        handoff = build_allocation_handoff_v2(
            bundle=bundle,
            receipt_graph=receipt_graph,
            lineage_bindings=lineage_bindings,
            lineage_complete=lineage_complete,
            persistence=persistence,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Research Lab allocation V2 handoff is invalid",
        ) from exc
    # Only fully-assembled bundles are cached; failures above raise and are
    # retried by the next caller without poisoning the cache.
    _allocation_handoff_cache_put(
        int(epoch),
        persist_snapshot,
        handoff,
    )
    return handoff


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
        "research_loop_ticket_current",
        filters=(("ticket_id", ticket_id), ("miner_hotkey", miner_hotkey)),
    )
    if not ticket:
        raise HTTPException(status_code=404, detail="Research Lab ticket not found for miner")
    return ticket


def _raise_ticket_expired(ticket: Mapping[str, Any]) -> None:
    expiry_at = unpaid_ticket_expires_at(ticket)
    raise HTTPException(
        status_code=410,
        detail={
            "code": "research_lab_ticket_expired",
            "message": "Research Lab ticket expired before loop-start payment; create a new ticket",
            "ticket_id": str(ticket.get("ticket_id") or ""),
            "unpaid_expires_at": expiry_at.isoformat() if expiry_at else None,
        },
    )


async def _require_ticket_mutable(
    ticket: Mapping[str, Any],
    *,
    enforce_unpaid_deadline: bool,
) -> None:
    if ticket_is_expired(ticket):
        _raise_ticket_expired(ticket)
    if (
        not enforce_unpaid_deadline
        or ticket_is_house_arm(ticket)
        or normalized_ticket_status(ticket) not in UNPAID_TICKET_ELIGIBLE_STATUSES
        or not unpaid_ticket_deadline_passed(ticket)
    ):
        return
    try:
        eligible = await select_one(
            "research_lab_unpaid_ticket_expiry_candidates",
            columns="ticket_id,unpaid_expires_at",
            filters=(("ticket_id", str(ticket.get("ticket_id") or "")),),
        )
    except Exception as exc:
        # The DB insert trigger remains authoritative. This compatibility path
        # keeps a code-first, expiry-disabled deployment safe before migration 85.
        logger.warning(
            "research_lab_unpaid_ticket_expiry_precheck_unavailable ticket_id=%s error=%s",
            str(ticket.get("ticket_id") or ""),
            str(exc)[:240],
        )
        return
    if eligible:
        _raise_ticket_expired(ticket)


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
            columns=(
                "key_ref,miner_hotkey,preflight_status,privacy_status,"
                "encrypted_management_key_ciphertext,management_key_hash,openrouter_workspace_hash"
            ),
            filters=(("key_ref", normalized_ref), ("miner_hotkey", normalized_hotkey)),
        )
        if not row:
            raise HTTPException(status_code=400, detail="miner OpenRouter key ref was not found for this hotkey")
        if str(row.get("preflight_status") or "") != "passed":
            raise HTTPException(status_code=400, detail="miner OpenRouter key ref has not passed preflight")
        if str(row.get("privacy_status") or "") != "verified":
            raise HTTPException(status_code=400, detail="miner OpenRouter key ref has not passed privacy verification")
        if not str(row.get("encrypted_management_key_ciphertext") or "").strip():
            raise HTTPException(status_code=400, detail="miner OpenRouter management key is required")
        if not str(row.get("management_key_hash") or "").strip():
            raise HTTPException(status_code=400, detail="miner OpenRouter management key metadata is missing")
        if not str(row.get("openrouter_workspace_hash") or "").strip():
            raise HTTPException(status_code=400, detail="miner OpenRouter workspace privacy proof is missing")
        return
    if normalized_handling == "ephemeral_ref":
        raise HTTPException(status_code=400, detail="encrypted OpenRouter key ref with management proof is required")
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


async def _audit_preview_select_all(
    table: str,
    *,
    filters: tuple[tuple[Any, ...], ...] = (),
    current_view: bool = False,
) -> list[dict[str, Any]]:
    primary_order = (("current_status_at", True),) if current_view else (("created_at", True),)
    try:
        return await select_all(table, filters=filters, order_by=primary_order, max_rows=50000)
    except Exception as exc:
        logger.warning(
            "research_lab_audit_preview_select_order_fallback table=%s order=%s error=%s",
            table,
            primary_order,
            str(exc)[:200],
        )
        return await select_all(table, filters=filters, max_rows=50000)


def _require_internal_key(config: ResearchLabGatewayConfig, provided: Optional[str]) -> None:
    if not config.internal_api_key:
        raise HTTPException(status_code=403, detail="Research Lab internal API key is not configured")
    if not provided or not secrets.compare_digest(provided, config.internal_api_key):
        raise HTTPException(status_code=401, detail="invalid Research Lab internal API key")


def _require_improvement_engine_enabled() -> ImprovementEngineConfig:
    config = ImprovementEngineConfig.from_env()
    if not config.enabled:
        raise HTTPException(status_code=403, detail="Research Lab Improvement Engine is disabled")
    return config


def _engine_issue_public(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "issue_key": row.get("issue_key"),
        "title": row.get("title"),
        "status": row.get("status"),
        "priority": row.get("priority"),
        "category": row.get("category"),
        "fingerprint": row.get("fingerprint"),
        "first_seen_at": row.get("first_seen_at"),
        "last_seen_at": row.get("last_seen_at"),
        "occurrence_count": row.get("occurrence_count"),
        "severity_score": row.get("severity_score"),
        "confidence": row.get("confidence"),
        "root_cause_doc": row.get("root_cause_doc") if isinstance(row.get("root_cause_doc"), Mapping) else {},
        "suggested_fix_doc": row.get("suggested_fix_doc") if isinstance(row.get("suggested_fix_doc"), Mapping) else {},
        "evaluator_spec_doc": row.get("evaluator_spec_doc") if isinstance(row.get("evaluator_spec_doc"), Mapping) else {},
        "dataset_spec_doc": row.get("dataset_spec_doc") if isinstance(row.get("dataset_spec_doc"), Mapping) else {},
        "linked_trace_ids": row.get("linked_trace_ids") or [],
        "linked_score_bundle_hashes": row.get("linked_score_bundle_hashes") or [],
        "linked_run_ids": row.get("linked_run_ids") or [],
        "linked_ticket_ids": row.get("linked_ticket_ids") or [],
        "created_pr_url": row.get("created_pr_url"),
        "created_candidate_id": row.get("created_candidate_id"),
        "created_by_engine_version": row.get("created_by_engine_version"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


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


async def _enforce_research_lab_submission_rate_limit(miner_hotkey: str, *, route: str) -> None:
    try:
        allowed, reason, stats = reserve_submission_slot(str(miner_hotkey or ""))
    except Exception as exc:
        logger.warning(
            "research_lab_rate_limit_unavailable route=%s hotkey=%s error=%s",
            route,
            str(miner_hotkey or "")[:16],
            str(exc)[:240],
        )
        raise HTTPException(status_code=503, detail="Research Lab rate limiter unavailable") from exc
    if allowed:
        return
    raise HTTPException(
        status_code=429,
        detail={
            "code": "research_lab_rate_limited",
            "route": route,
            "message": reason or "Research Lab rate limit exceeded",
            "stats": stats,
        },
    )


async def _enforce_open_ticket_cap(config: ResearchLabGatewayConfig, miner_hotkey: str) -> None:
    rows = await select_all(
        "research_loop_ticket_current",
        columns="ticket_id,current_ticket_status,current_status_at",
        filters=(("miner_hotkey", str(miner_hotkey)),),
        order_by=(("current_status_at", True),),
        max_rows=500,
    )
    public_rows = await select_all(
        "research_lab_public_loop_card_current",
        columns="ticket_id,current_outcome_label,current_outcome_band,current_last_activity_at,created_at",
        filters=(("miner_hotkey", str(miner_hotkey)),),
        order_by=(("current_last_activity_at", True), ("created_at", True)),
        max_rows=500,
    )
    public_by_ticket_id = {
        str(row.get("ticket_id") or ""): row
        for row in public_rows
        if row.get("ticket_id")
    }
    open_rows = [
        row
        for row in rows
        if not public_loop_outcome_closes_ticket(public_by_ticket_id.get(str(row.get("ticket_id") or "")))
        if str(row.get("current_ticket_status") or "").strip().lower() not in TERMINAL_TICKET_STATUSES
    ]
    if len(open_rows) < int(config.max_open_tickets_per_hotkey):
        return
    raise HTTPException(
        status_code=429,
        detail={
            "code": "research_lab_open_ticket_cap_exceeded",
            "message": "too many open Research Lab tickets for this hotkey",
            "open_ticket_count": len(open_rows),
            "max_open_tickets": int(config.max_open_tickets_per_hotkey),
        },
    )


async def _preserve_loop_start_credit_after_queue_failure(
    *,
    ticket_id: str,
    payment_id: str,
    payment_ref: str,
    miner_hotkey: str,
    run_id: str,
    reason: str,
    error: BaseException,
    replaces_credit_id: str | None,
) -> str:
    credit_id = _deterministic_loop_start_credit_id(
        ticket_id=ticket_id,
        payment_ref=payment_ref,
        run_id=run_id,
    )
    event_doc = {
        "error": str(error)[:200],
        "replaces_credit_id": replaces_credit_id,
        "run_id": run_id,
    }
    last_exc: BaseException | None = None
    for attempt in range(1, 4):
        try:
            await create_credit_event(
                credit_id=credit_id,
                ticket_id=ticket_id,
                payment_id=payment_id,
                payment_ref=payment_ref,
                miner_hotkey=miner_hotkey,
                event_type="granted",
                credit_status="available",
                reason=reason,
                event_doc={**event_doc, "credit_preservation_attempt": attempt},
            )
            return credit_id
        except Exception as exc:
            last_exc = exc
            existing = await _existing_available_credit(credit_id)
            if existing:
                return credit_id
            await asyncio.sleep(0.2 * attempt)
    logger.exception(
        "research_lab_credit_preservation_failed credit_id=%s payment_ref=%s error=%s",
        credit_id,
        payment_ref,
        str(last_exc)[:240] if last_exc else "",
    )
    raise HTTPException(
        status_code=503,
        detail={
            "code": "research_lab_credit_preservation_failed",
            "message": "Payment was verified but retry credit could not be persisted; operator reconciliation required",
            "credit_id": credit_id,
            "payment_ref": payment_ref,
        },
    ) from last_exc


async def _existing_available_credit(credit_id: str) -> dict[str, Any] | None:
    try:
        row = await select_one(
            "research_loop_start_credit_current",
            filters=(("credit_id", credit_id),),
        )
    except Exception as exc:
        logger.warning(
            "research_lab_existing_credit_lookup_failed credit_id=%s error=%s",
            credit_id,
            str(exc)[:240],
        )
        return None
    if row and str(row.get("current_credit_status") or "") == "available":
        return row
    return None


def _deterministic_loop_start_credit_id(*, ticket_id: str, payment_ref: str, run_id: str) -> str:
    return "loop_start_credit:" + canonical_hash(
        {"ticket_id": ticket_id, "payment_ref": payment_ref, "run_id": run_id}
    ).split(":", 1)[1][:32]


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
    hotkey_capacity = max(1, int(config.max_active_autoresearch_loops_per_hotkey or 1))
    same_hotkey_count = sum(
        1
        for row in active_rows
        if str((ticket_map.get(str(row.get("ticket_id") or "")) or {}).get("miner_hotkey") or "").strip()
        == normalized_hotkey
    )
    if same_hotkey_count >= hotkey_capacity:
        raise HTTPException(
            status_code=409,
            detail="too many autoresearch loops for this hotkey already running",
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
    hotkey_capacity = max(1, int(config.max_active_autoresearch_loops_per_hotkey or 1))
    same_hotkey_admitted_run_ids = {
        str(row.get("run_id") or "") for row in same_hotkey_rows[:hotkey_capacity]
    }
    if len(same_hotkey_rows) > hotkey_capacity and normalized_run_id not in same_hotkey_admitted_run_ids:
        return "too many autoresearch loops for this hotkey already running"

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
        # V2 worker proxies are KMS-sealed and intentionally absent from the
        # parent environment. The explicit fleet size is bound into the
        # encrypted-profile transition report and worker process environment.
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


def _queue_capacity_doc(config: ResearchLabGatewayConfig) -> dict[str, int | str]:
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


def _configured_autoresearch_proxy_count() -> int:
    count = 0
    for index in range(1, 501):
        if any(os.getenv(f"{prefix}_{index}", "").strip() for prefix in AUTORESEARCH_PROXY_PREFIXES):
            count += 1
    return count


def _autoresearch_active_row_is_fresh(row: Mapping[str, Any], config: ResearchLabGatewayConfig) -> bool:
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
    mean_delta, delta_lcb = benchmark_relative_score_deltas(bundle)
    metric = promotion_improvement_metric(bundle)
    summary = {
        "base_score": (
            metric.baseline_aggregate_score
            if metric.baseline_aggregate_score is not None
            else _float_or_none(aggregates.get("base_score"))
        ),
        "candidate_score": (
            metric.candidate_total_score
            if metric.candidate_total_score is not None
            else _float_or_none(aggregates.get("candidate_score"))
        ),
        "mean_delta": mean_delta,
        "delta_lcb": delta_lcb,
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
    if is_ticket_expiry_conflict(exc):
        raise HTTPException(
            status_code=410,
            detail={
                "code": "research_lab_ticket_expired",
                "message": "Research Lab ticket expired before this operation completed; create a new ticket",
            },
        ) from exc
    if "does not exist" in message_lower or "relation" in message_lower:
        raise HTTPException(status_code=503, detail="Research Lab SQL migrations are not applied") from exc
    if "research_lab_queue_hotkey_conflict" in message_lower:
        raise HTTPException(
            status_code=409,
            detail="too many autoresearch loops for this hotkey already running",
        ) from exc
    if "research_lab_queue_capacity_conflict" in message_lower:
        raise HTTPException(status_code=409, detail="too many autoresearch loops right now, try again later") from exc
    if _is_duplicate_payment_error(exc):
        raise HTTPException(status_code=409, detail="Research Lab payment has already been used") from exc
    raise HTTPException(status_code=500, detail="Research Lab storage operation failed") from exc


def _redact_storage_error_text(value: str) -> str:
    value = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "[REDACTED_OPENROUTER_KEY]", value or "")
    value = re.sub(r"sb_(secret|publishable)_[A-Za-z0-9_-]+", "[REDACTED_SUPABASE_KEY]", value)
    return value[:2000]
