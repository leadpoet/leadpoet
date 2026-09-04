"""Research Lab production gateway API.

The namespace is production-facing but inert by default. All mutating routes
require explicit Research Lab flags and write only Research Lab tables/events.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import copy
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
import secrets
import time
from typing import Any, Mapping, Optional

import gzip
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute

from gateway.build_info import get_build_info
from gateway.qualification.utils.chain import (
    BITTENSOR_NETUID,
    ChainRegistrationUnavailable,
    check_hotkey_registration as chain_is_hotkey_registered,
    verify_hotkey_signature,
)
from gateway.utils.bans import is_hotkey_banned

from .allocations import build_research_lab_allocation_bundle
from .config import ResearchLabGatewayConfig
from .models import (
    ResearchLabSourceAdapterSubmissionRequest,
    ResearchLabSourceAdapterSubmissionResponse,
    ResearchLabSourceAddStatusItem,
    ResearchLabSourceAddStatusRequest,
    ResearchLabSourceAddStatusResponse,
    ResearchLabSourceMetadata,
    ResearchLabSourceAddCredentialRecipientRequest,
    ResearchLabCredentialRecipientResponse,
    ResearchLabSourceAdapterRecheckResponse,
    ResearchLabSourceAdapterProbeConfigureRequest,
    ResearchLabSourceAdapterProbeConfigureResponse,
    ResearchLabSourceAdapterProvisionRequest,
    ResearchLabSourceAdapterProvisionResponse,
)
from .source_add_catalog import (
    ALREADY_SUBMITTED_DETAIL,
    SOURCE_ADD_SUBMISSION_FAILED_DETAIL,
    PROVISION_STATUS_APPROVED_PENDING,
    PROVISION_STATUS_DISABLED,
    PROVISION_STATUS_ELIGIBLE,
    PROVISION_STATUSES,
    reject_source_add_secret_text,
    sanitize_source_add_doc,
    source_add_encrypted_envelope_valid,
)
from .store import (
    canonical_hash,
    call_rpc,
    select_one,
)
from gateway.research_lab.provider_evidence_proxy import (
    ProviderRegistryEntry,
    reserved_builtin_provider_ids_sync,
    validate_provider_registry_entries,
)
from gateway.research_lab.provider_capabilities import (
    normalize_source_add_planner_contract,
    validate_capability_provider_doc,
)
from research_lab.probe_catalog import ProviderProbeEndpoint, validate_probe_catalog
from research_lab.source_add_execution import intake_source_add_submission
from research_lab.source_add import source_add_contains_credential_material
from research_lab.source_add_identity import (
    SOURCE_ADD_IDENTITY_VERSION,
    legacy_source_identity_hash,
    normalize_source_add_provider_origin,
    source_identity_alias_hashes_from_metadata,
    source_identity_hash_from_metadata,
    source_provider_origin_hash_from_metadata,
)
from .source_add_workflow import (
    source_add_control_state,
    source_add_host_hash,
    source_add_probe_config_ref,
    source_add_ref,
    source_add_work_id,
)
from . import source_add_catalog as source_add_catalog_contract
from leadpoet_canonical.constants import EPOCH_LENGTH
from gateway.research_lab import allocation_handoff_disk_cache


logger = logging.getLogger(__name__)


class _ResearchLabCredentialSafeRoute(APIRoute):
    """Keep SOURCE_ADD validation failures generic and credential-free."""

    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def credential_safe_route_handler(request: Request):
            try:
                return await original_route_handler(request)
            except RequestValidationError:
                if request.url.path == "/research-lab/source-adapters":
                    return JSONResponse(
                        status_code=400,
                        content={
                            "detail": SOURCE_ADD_SUBMISSION_FAILED_DETAIL
                        },
                    )
                raise

        return credential_safe_route_handler


router = APIRouter(
    prefix="/research-lab",
    tags=["research-lab"],
    route_class=_ResearchLabCredentialSafeRoute,
)
_SOURCE_ADD_SUBMISSION_COOLDOWN_SECONDS = 20
_SOURCE_ADD_CAP_LIMIT_TYPES = {
    "hotkey_open_cap": "open_submissions",
    "hotkey_day_cap": "daily",
    "hotkey_30d_cap": "rolling_30d",
}


def _source_add_seconds_until_utc_midnight(now: datetime | None = None) -> int:
    """Seconds until the daily cap resets.

    The daily counter in ``research_lab_source_add_admit_v3`` is bounded by
    ``date_trunc('day', NOW() AT TIME ZONE 'UTC')``, so the reset is the next
    UTC midnight. Never returns 0, so a client at the boundary still backs off.
    """

    now = now or datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(1, int((midnight - now).total_seconds()))


def _source_add_cap_detail(status: str, config) -> dict:
    """Structured 429 body for the three per-hotkey submission caps.

    All three used to answer with the bare string "SOURCE_ADD submission
    limit reached", which tells a miner neither which limit it hit nor when to
    try again — so clients retry in a tight loop against a cap that will not
    move for hours. The shape mirrors the cooldown 429 above. The limits
    themselves are already published on the Research Lab config surface, so
    naming them here discloses nothing new, and the hotkey is signature-verified
    upstream, so a miner only ever learns about its own state.
    """

    limit_type = _SOURCE_ADD_CAP_LIMIT_TYPES[status]
    if status == "hotkey_open_cap":
        limit = int(config.source_add_max_concurrent_per_hotkey)
        retry_after = None
        message = (
            f"You already have {limit} submissions in review. Retry once one of "
            "them finishes."
        )
    elif status == "hotkey_day_cap":
        limit = int(config.source_add_max_per_day_per_hotkey)
        retry_after = _source_add_seconds_until_utc_midnight()
        message = (
            f"You have used all {limit} submissions for today. Retry in "
            f"{retry_after} seconds, when the daily allowance resets at UTC "
            "midnight."
        )
    else:
        limit = int(config.source_add_max_per_30d_per_hotkey)
        retry_after = None
        message = (
            f"You have used all {limit} submissions allowed in a rolling 30 "
            "days. Retry once your oldest submission ages out of the window."
        )

    stats: dict = {"limit_type": limit_type, "limit": limit}
    if retry_after is not None:
        stats["retry_after_seconds"] = retry_after
    return {
        "code": "research_lab_rate_limited",
        "route": "source_adapters",
        "message": message,
        "stats": stats,
    }
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


def _source_add_dispatcher_runtime_ready(request: Request) -> bool:
    """Project the same live dispatcher gate used by gateway middleware."""

    task = getattr(request.app.state, "source_add_dispatcher_task", None)
    try:
        return bool(task is not None and not task.done())
    except Exception as exc:
        logger.warning(
            "research_lab_source_add_dispatcher_status_unavailable type=%s",
            type(exc).__name__,
        )
        return False


@router.get("/status")
async def research_lab_status(request: Request) -> dict[str, object]:
    config = ResearchLabGatewayConfig.from_env()
    source_add_control = await source_add_control_state()
    public_status = config.public_status()
    source_add_public = dict(public_status.get("source_add") or {})
    source_add_public["control"] = {
        key: source_add_control.get(key)
        for key in ("paused", "status", "updated_at", "unavailable")
        if key in source_add_control
    }
    dispatcher_runtime_ready = _source_add_dispatcher_runtime_ready(request)
    source_add_public["effective_dispatcher_enabled"] = bool(
        config.source_add_enabled
        and config.source_add_dispatcher_enabled
        and dispatcher_runtime_ready
        and not source_add_control.get("paused", True)
    )
    # This is the public, fail-closed projection of every launch gate checked
    # before the source-adapter POST verifies a miner or persists anything.
    # Miner clients use it to exit before prompting when intake is closed.
    source_add_public["intake_enabled"] = bool(
        config.api_enabled
        and config.production_writes_enabled
        and config.source_add_enabled
        and config.source_add_dispatcher_enabled
        and dispatcher_runtime_ready
        and not source_add_control.get("paused", True)
    )
    return {
        "service": "leadpoet-research-lab-gateway",
        "status": "configured" if config.api_enabled else "disabled",
        **public_status,
        "source_add": source_add_public,
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


_SOURCE_ADD_FINAL_APPROVAL_STAGES = frozenset({"accepted"})


def _require_source_add_final_approval_mutable(row: Mapping[str, Any]) -> None:
    if str(row.get("stage") or "") in _SOURCE_ADD_FINAL_APPROVAL_STAGES:
        raise HTTPException(
            status_code=409,
            detail="SOURCE_ADD final approval is frozen",
        )


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


@router.post(
    "/source-adapters/status",
    response_model=ResearchLabSourceAddStatusResponse,
)
async def list_research_lab_source_add_status(
    payload: ResearchLabSourceAddStatusRequest,
    response: Response,
):
    """Return only the signing miner's sanitized SOURCE_ADD status page."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    await _verify_signed_miner(payload)

    try:
        rows = await call_rpc(
            "research_lab_source_add_miner_status_page_v1",
            {
                "p_miner_hotkey": payload.miner_hotkey,
                "p_cursor_submission_id": payload.cursor,
                "p_limit": payload.limit,
            },
        )
    except Exception as exc:
        logger.warning(
            "SOURCE_ADD_MINER_STATUS_READ_FAILED type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="SOURCE_ADD status is temporarily unavailable",
        ) from exc
    if not isinstance(rows, list) or len(rows) > payload.limit + 1:
        logger.warning("SOURCE_ADD_MINER_STATUS_INVALID_PAGE")
        raise HTTPException(
            status_code=503,
            detail="SOURCE_ADD status is temporarily unavailable",
        )

    items: list[ResearchLabSourceAddStatusItem] = []
    seen_submission_ids: set[str] = set()
    for raw_row in rows:
        if (
            not isinstance(raw_row, Mapping)
            or str(raw_row.get("schema_version") or "")
            != "leadpoet.source_add_miner_status.v1"
            or str(raw_row.get("miner_hotkey") or "") != payload.miner_hotkey
        ):
            logger.warning("SOURCE_ADD_MINER_STATUS_OWNERSHIP_MISMATCH")
            raise HTTPException(
                status_code=503,
                detail="SOURCE_ADD status is temporarily unavailable",
            )
        try:
            item = ResearchLabSourceAddStatusItem.model_validate(
                {
                    "submission_id": raw_row.get("submission_id"),
                    "source_name": raw_row.get("source_name"),
                    "submitted_at": raw_row.get("submitted_at"),
                    "updated_at": raw_row.get("updated_at"),
                    "decision_status": raw_row.get("decision_status"),
                    "decision_reason_code": raw_row.get("decision_reason_code"),
                    "decision_reason": raw_row.get("decision_reason"),
                    "reward_status": raw_row.get("reward_status"),
                    "alpha_percent": raw_row.get("alpha_percent"),
                    "reward_epochs": raw_row.get("reward_epochs"),
                    "start_epoch": raw_row.get("start_epoch"),
                    "end_epoch": raw_row.get("end_epoch"),
                }
            )
        except Exception as exc:
            logger.warning(
                "SOURCE_ADD_MINER_STATUS_INVALID_ROW type=%s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail="SOURCE_ADD status is temporarily unavailable",
            ) from exc
        if item.submission_id in seen_submission_ids:
            logger.warning("SOURCE_ADD_MINER_STATUS_DUPLICATE_ROW")
            raise HTTPException(
                status_code=503,
                detail="SOURCE_ADD status is temporarily unavailable",
            )
        seen_submission_ids.add(item.submission_id)
        items.append(item)

    has_more = len(items) > payload.limit
    visible_items = items[: payload.limit]
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Pragma"] = "no-cache"
    return ResearchLabSourceAddStatusResponse(
        schema_version="leadpoet.source_add_miner_status.v1",
        submissions=visible_items,
        next_cursor=(visible_items[-1].submission_id if has_more else None),
    )


@router.post("/source-adapters", response_model=ResearchLabSourceAdapterSubmissionResponse)
async def submit_research_lab_source_adapter(payload: ResearchLabSourceAdapterSubmissionRequest):
    """Atomically reserve a source identity and queue measured provenance."""

    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.production_writes_enabled, "Research Lab production writes are disabled")
    _require_enabled(config.source_add_enabled, "Research Lab SOURCE_ADD submissions are disabled")
    source_add_control = await source_add_control_state()
    if source_add_control.get("paused", True):
        raise HTTPException(
            status_code=503,
            detail="SOURCE_ADD workflow is paused",
        )
    if source_add_contains_credential_material(payload.signed_payload()):
        logger.warning("SOURCE_ADD_CREDENTIAL_MATERIAL_REJECTED")
        raise HTTPException(
            status_code=400,
            detail=SOURCE_ADD_SUBMISSION_FAILED_DETAIL,
        )
    await _verify_signed_miner(payload)

    if payload.adapter_credential is not None or payload.adapter_credential_v2 is not None:
        logger.warning("SOURCE_ADD_MINER_CREDENTIAL_FIELDS_REJECTED")
        raise HTTPException(
            status_code=400,
            detail=SOURCE_ADD_SUBMISSION_FAILED_DETAIL,
        )

    source_metadata = payload.source_metadata.model_dump(mode="json")
    try:
        current_model_uses_api = await asyncio.to_thread(
            source_add_catalog_contract.source_add_api_is_current_builtin_sync,
            str(source_metadata.get("api_base_url") or ""),
        )
    except Exception as exc:
        logger.warning(
            "SOURCE_ADD_BUILTIN_CATALOG_UNAVAILABLE type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="SOURCE_ADD workflow temporarily unavailable",
        ) from exc
    if current_model_uses_api:
        raise HTTPException(
            status_code=409,
            detail=SOURCE_ADD_SUBMISSION_FAILED_DETAIL,
        )
    declared_domains = (
        payload.manifest.get("declared_base_domains")
        if isinstance(payload.manifest, Mapping)
        else None
    )
    if not isinstance(declared_domains, list) or any(
        not isinstance(item, str) for item in declared_domains
    ):
        logger.warning("SOURCE_ADD_DECLARED_DOMAINS_REJECTED")
        raise HTTPException(
            status_code=400,
            detail=SOURCE_ADD_SUBMISSION_FAILED_DETAIL,
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
        provider_origin_host = normalize_source_add_provider_origin(
            str(source_metadata.get("api_base_url") or "")
        )
        provider_origin_ref = source_provider_origin_hash_from_metadata(
            source_metadata
        )
        if not provider_origin_host or not provider_origin_ref:
            raise ValueError("provider origin is unavailable")
    except ValueError as exc:
        logger.warning("SOURCE_ADD_IDENTITY_REJECTED")
        raise HTTPException(
            status_code=400,
            detail=SOURCE_ADD_SUBMISSION_FAILED_DETAIL,
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
        logger.warning(
            "SOURCE_ADD_INTAKE_REJECTED reason_count=%d",
            len(errors or ()),
        )
        raise HTTPException(
            status_code=400,
            detail=SOURCE_ADD_SUBMISSION_FAILED_DETAIL,
        )
    record_doc = record.to_dict()
    record_doc["source_metadata"] = source_metadata
    record_doc["source_identity_version"] = SOURCE_ADD_IDENTITY_VERSION
    record_doc["provider_origin_host"] = provider_origin_host
    record_doc["provider_origin_hash"] = provider_origin_ref
    work_id = source_add_work_id(
        record.submission_id,
        "provenance",
        "%s:%s" % (payload.idempotency_key, payload.timestamp),
    )
    admitted = await _source_add_rpc(
        "research_lab_source_add_admit_v3",
        {
            "p_record_doc": record_doc,
            "p_identity_hash": source_identity_ref,
            "p_documentation_identity_hash": (
                source_identity_aliases[0] if source_identity_aliases else ""
            ),
            "p_legacy_identity_hash": legacy_identity_ref,
            "p_provider_origin_hash": provider_origin_ref,
            "p_work_id": work_id,
            "p_max_open": int(config.source_add_max_concurrent_per_hotkey),
            "p_max_day": int(config.source_add_max_per_day_per_hotkey),
            "p_max_30d": int(config.source_add_max_per_30d_per_hotkey),
            "p_cooldown_seconds": _SOURCE_ADD_SUBMISSION_COOLDOWN_SECONDS,
        },
    )
    status = str(admitted.get("status") or "")
    if status == "duplicate":
        raise HTTPException(
            status_code=409,
            detail=SOURCE_ADD_SUBMISSION_FAILED_DETAIL,
        )
    if status == "route_cooldown":
        try:
            cooldown_seconds = int(admitted.get("cooldown_seconds") or 0)
            wait_seconds = int(admitted.get("wait_seconds") or 0)
        except (TypeError, ValueError):
            cooldown_seconds = 0
            wait_seconds = 0
        if (
            cooldown_seconds != _SOURCE_ADD_SUBMISSION_COOLDOWN_SECONDS
            or wait_seconds < 1
            or wait_seconds > cooldown_seconds
        ):
            logger.warning(
                "SOURCE_ADD_ADMISSION_INVALID_COOLDOWN cooldown=%s wait=%s",
                cooldown_seconds,
                wait_seconds,
            )
            raise HTTPException(
                status_code=503,
                detail="SOURCE_ADD workflow temporarily unavailable",
            )
        raise HTTPException(
            status_code=429,
            detail={
                "code": "research_lab_rate_limited",
                "route": "source_adapters",
                "message": (
                    f"Please wait {wait_seconds} seconds before submitting "
                    "another lead (anti-spam cooldown)."
                ),
                "stats": {
                    "limit_type": "cooldown",
                    "cooldown_seconds": cooldown_seconds,
                    "wait_seconds": wait_seconds,
                },
            },
        )
    if status in _SOURCE_ADD_CAP_LIMIT_TYPES:
        raise HTTPException(
            status_code=429,
            detail=_source_add_cap_detail(status, config),
        )
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
    provider_origin_hash = source_provider_origin_hash_from_metadata(
        source_metadata
    )
    if not provider_origin_hash:
        raise HTTPException(
            status_code=400,
            detail="submission metadata is incomplete or invalid",
        )
    work_id = source_add_work_id(
        submission_id,
        "provenance",
        "operator-recheck:%s" % (int(row.get("seq") or 0) + 1),
    )
    queued = await _source_add_rpc(
        "research_lab_source_add_requeue_provenance_v2",
        {
            "p_submission_id": submission_id,
            "p_identity_hash": identity_hash,
            "p_documentation_identity_hash": (
                identity_aliases[0] if identity_aliases else ""
            ),
            "p_legacy_identity_hash": legacy_hash,
            "p_provider_origin_hash": provider_origin_hash,
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
    _require_source_add_final_approval_mutable(row)
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
        "research_lab_source_add_configure_probe_v3",
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
    if queue_status == "final_approval_frozen":
        raise HTTPException(status_code=409, detail="SOURCE_ADD final approval is frozen")
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
    _require_source_add_final_approval_mutable(row)
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
        planner_summary: dict[str, Any] = {
            "provider_alias": payload.provider_alias
            or payload.registry_provider_id,
            "endpoint_families": [
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "description": endpoint.description[:200],
                }
                for endpoint in probe_objects
            ],
            "model_policy": "",
            "probe_metadata": [
                endpoint.endpoint_id for endpoint in probe_objects
            ],
        }
        if payload.routing_contract:
            try:
                planner_summary.update(
                    normalize_source_add_planner_contract(
                        payload.registry_provider_id,
                        payload.routing_contract,
                        estimated_cost_microusd_per_call=int(
                            payload.cost_model.get(
                                "est_cost_microusd_per_call", 0
                            )
                            or 0
                        ),
                        probe_endpoints=probe_endpoints,
                        tested_probes=[
                            dict(item)
                            for item in probe_doc.get("probes") or []
                            if isinstance(item, Mapping)
                        ],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="invalid SOURCE_ADD v8 routing contract",
                ) from exc
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
            planner_summary=planner_summary,
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
                "research_lab_source_add_finalize_provision_v3",
                {
                    "p_submission_id": submission_id,
                    "p_catalog_row": pending_catalog_row,
                    "p_provision_row": pending_row,
                    "p_smoke_attempt": {},
                },
            )
            pending_status = str(pending.get("status") or "")
            if pending_status == "final_approval_frozen":
                raise HTTPException(
                    status_code=409,
                    detail="SOURCE_ADD final approval is frozen",
                )
            if pending_status not in {
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
            "research_lab_source_add_enqueue_provision_smoke_v2",
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
        if queue_status == "final_approval_frozen":
            raise HTTPException(
                status_code=409,
                detail="SOURCE_ADD final approval is frozen",
            )
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
        "research_lab_source_add_finalize_provision_v3",
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
    if final_status == "final_approval_frozen":
        raise HTTPException(status_code=409, detail="SOURCE_ADD final approval is frozen")
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
    return _allocation_persistence_for_known_current_epoch(
        config=config,
        current_epoch=int(current_epoch),
        requested_epoch=int(epoch),
        internal_key=internal_key,
    )


def _allocation_persistence_for_known_current_epoch(
    *,
    config: ResearchLabGatewayConfig,
    current_epoch: int,
    requested_epoch: int,
    internal_key: Optional[str],
) -> bool:
    """Apply the normal persistence policy to an already-resolved epoch."""

    from gateway.research_lab.allocations import allocation_snapshot_persistence_decision

    normalized_key = internal_key if isinstance(internal_key, str) else None
    decision = allocation_snapshot_persistence_decision(
        current_epoch=int(current_epoch),
        requested_epoch=int(requested_epoch),
        provided_key=normalized_key,
        configured_key=str(getattr(config, "internal_api_key", "") or ""),
        live_allocation_enabled=bool(config.reimbursements_enabled or config.weight_mutation_enabled),
    )
    if decision == "future_epoch":
        raise HTTPException(
            status_code=422,
            detail=(
                f"allocation epoch {int(requested_epoch)} is in the future "
                f"(current {int(current_epoch)})"
            ),
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
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # This endpoint has no build task to report its failures, so the reason
        # for a 500 has to be recorded here or it is lost with the response.
        logger.error(
            "research_lab_live_allocation_build_failed epoch=%s "
            "persist_snapshot=%s error_type=%s error=%s",
            int(epoch),
            bool(persist_snapshot),
            type(exc).__name__,
            str(exc)[:240],
            exc_info=True,
        )
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=500, detail=str(exc)) from exc
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
_ALLOCATION_BUILD_TASKS: dict[
    _AllocationCacheKey,
    asyncio.Task[dict[str, Any]],
] = {}
# Finney targets roughly 12-second blocks. Retain a successful handoff for
# longer than one 360-block epoch so a preparation completed before the weight
# window survives until block 300, including normal finality/block-time drift.
_ALLOCATION_CACHE_TTL_SECONDS = float(EPOCH_LENGTH * 15)
_ALLOCATION_CACHE_MAX_EPOCHS = 16


def _allocation_cache_key(epoch: int, persist_snapshot: bool) -> _AllocationCacheKey:
    return (int(epoch), bool(persist_snapshot))


def _allocation_handoff_cache_get(
    epoch: int,
    persist_snapshot: bool,
) -> Optional[dict[str, Any]]:
    keys = [_allocation_cache_key(epoch, persist_snapshot)]
    if not persist_snapshot:
        # persist_snapshot only gates the authorized emission-snapshot write;
        # it never changes the returned document. A read-only caller may
        # therefore reuse a persisted handoff, but an authenticated caller must
        # never reuse a read-only result and skip its required persistence.
        keys.append(_allocation_cache_key(epoch, True))

    for key in keys:
        entry = _ALLOCATION_HANDOFF_CACHE.get(key)
        if entry is None:
            continue
        expires_at, handoff = entry
        if time.monotonic() >= expires_at:
            _ALLOCATION_HANDOFF_CACHE.pop(key, None)
            continue
        _ALLOCATION_HANDOFF_CACHE.move_to_end(key)
        return handoff
    return None


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
        completed = _ALLOCATION_BUILD_TASKS.get(evicted)
        if completed is not None and completed.done():
            _ALLOCATION_BUILD_TASKS.pop(evicted, None)


def _allocation_cache_release_commit() -> str:
    return str(get_build_info().get("git_commit") or "").strip().lower()


def _allocation_build_task(
    *,
    config: "ResearchLabGatewayConfig",
    epoch: int,
    persist_snapshot: bool,
) -> asyncio.Task[dict[str, Any]]:
    key = _allocation_cache_key(epoch, persist_snapshot)
    task = _ALLOCATION_BUILD_TASKS.get(key)
    if task is not None:
        return task
    if not persist_snapshot:
        # If an authenticated validator already owns the build, share that
        # server-owned task. Its persistence was authorized by that validator;
        # the anonymous waiter neither initiates nor upgrades persistence.
        persisted = _ALLOCATION_BUILD_TASKS.get(
            _allocation_cache_key(epoch, True)
        )
        if persisted is not None:
            return persisted
    task = asyncio.create_task(
        _build_and_cache_attested_allocation(
            config=config,
            epoch=int(epoch),
            persist_snapshot=bool(persist_snapshot),
        )
    )
    _ALLOCATION_BUILD_TASKS[key] = task

    def clear(completed: asyncio.Task[dict[str, Any]]) -> None:
        if _ALLOCATION_BUILD_TASKS.get(key) is completed:
            _ALLOCATION_BUILD_TASKS.pop(key, None)
        if completed.cancelled():
            logger.warning(
                "research_lab_allocation_build_cancelled epoch=%s persist_snapshot=%s",
                int(epoch),
                bool(persist_snapshot),
            )
            return
        error = completed.exception()
        if error is not None:
            logger.warning(
                "research_lab_allocation_build_failed epoch=%s "
                "persist_snapshot=%s error_type=%s error=%s",
                int(epoch),
                bool(persist_snapshot),
                type(error).__name__,
                str(error)[:240],
            )

    task.add_done_callback(clear)
    return task


async def _allocation_handoff_response(
    handoff: dict[str, Any],
    accept_encoding: Optional[str],
):
    """Serve the handoff gzip-compressed when the caller asks for it.

    The handoff is multi-MB, hash-dense JSON that compresses several-fold, and
    it is fetched inside the validator's bounded pre-submission budget — the
    dominant cost of a cold-epoch fetch should be the build, not the transfer.
    Callers that do not advertise gzip get the identity JSON exactly as
    before. Serialization + compression run in a worker thread so the response
    path never stalls the shared event loop.
    """

    if "gzip" not in str(accept_encoding or "").lower():
        return handoff

    def _encode() -> bytes:
        raw = json.dumps(handoff, separators=(",", ":")).encode("utf-8")
        return gzip.compress(raw, compresslevel=6)

    wire = await asyncio.to_thread(_encode)
    return Response(
        content=wire,
        media_type="application/json",
        headers={
            "Content-Encoding": "gzip",
            "Vary": "Accept-Encoding",
        },
    )


def _research_lab_attested_allocation_config() -> ResearchLabGatewayConfig:
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.shadow_bundles_enabled, "Research Lab report bundles are disabled")
    return config


async def _get_research_lab_attested_allocation_handoff(
    *,
    config: ResearchLabGatewayConfig,
    epoch: int,
    persist_snapshot: bool,
) -> dict[str, Any]:
    """Load or build one allocation handoff after persistence authorization."""

    cached_handoff = _allocation_handoff_cache_get(
        int(epoch),
        persist_snapshot,
    )
    if cached_handoff is not None:
        return cached_handoff
    # Warm-start after a process restart: the memory cache is wiped by every
    # gateway restart, and a restart between the block-180 prewarm and the
    # block-300 submission used to force a full cold rebuild (receipt-ancestry
    # reconstruction + fresh enclave attestation) inside the validator's 90s
    # fetch budget. The handoff is deterministic per epoch, the validator
    # re-validates it fail-closed, and any disk-cache failure falls open to
    # the normal build below.
    if _ALLOCATION_BUILD_TASKS.get(
        _allocation_cache_key(int(epoch), persist_snapshot)
    ) is None:
        disk_handoff = await asyncio.to_thread(
            allocation_handoff_disk_cache.load_handoff,
            int(BITTENSOR_NETUID),
            int(epoch),
            persist_snapshot,
            _allocation_cache_release_commit(),
        )
        if disk_handoff is not None:
            _allocation_handoff_cache_put(
                int(epoch),
                persist_snapshot,
                disk_handoff,
            )
            return disk_handoff
    # A concurrent persisted build can finish while this request yields to the
    # disk-cache lookup. Recheck memory before creating a second cold build.
    cached_handoff = _allocation_handoff_cache_get(
        int(epoch),
        persist_snapshot,
    )
    if cached_handoff is not None:
        return cached_handoff
    # Shield the complete build-and-cache task from client disconnects. The
    # previous lock only serialized live requests: when a validator timed out,
    # request cancellation released the lock before the finished authority
    # could be assembled and cached, so the next retry started over.
    built_handoff = await asyncio.shield(
        _allocation_build_task(
            config=config,
            epoch=int(epoch),
            persist_snapshot=persist_snapshot,
        )
    )
    return built_handoff


async def _get_research_lab_attested_allocation_for_resolved_current_epoch(
    *,
    epoch: int,
    current_epoch: int,
    internal_key: str,
) -> dict[str, Any]:
    """Build the pre-launch handoff using an exact epoch resolved by maintenance."""

    config = _research_lab_attested_allocation_config()
    persist_snapshot = _allocation_persistence_for_known_current_epoch(
        config=config,
        current_epoch=int(current_epoch),
        requested_epoch=int(epoch),
        internal_key=internal_key,
    )
    return await _get_research_lab_attested_allocation_handoff(
        config=config,
        epoch=int(epoch),
        persist_snapshot=persist_snapshot,
    )


@router.get("/allocations/attested/{epoch}")
async def get_research_lab_attested_allocation(
    epoch: int,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
    accept_encoding: Optional[str] = Header(default=None),
):
    """Return the unchanged live allocation plus its enclave-signed sidecar."""

    config = _research_lab_attested_allocation_config()
    # The guard rejects future epochs and decides snapshot persistence; it must
    # run on every request and is cheap relative to the bundle build.
    persist_snapshot = await _allocation_epoch_guard_and_persistence(
        config, int(epoch), x_leadpoet_internal_key
    )
    handoff = await _get_research_lab_attested_allocation_handoff(
        config=config,
        epoch=int(epoch),
        persist_snapshot=persist_snapshot,
    )
    return await _allocation_handoff_response(handoff, accept_encoding)


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
    except (asyncio.TimeoutError, TimeoutError) as exc:
        # A build that ran out of time is the one failure we can name from the
        # outside: only the status code survives into request telemetry (the
        # detail below reaches the caller, and the gateway log is not
        # collected), so give the timeout its own code instead of folding it
        # into the generic 500.
        raise HTTPException(
            status_code=504,
            detail=f"Research Lab attested allocation timed out: {str(exc)[:200]}",
        ) from exc
    except RuntimeError as exc:
        # The build path raises RuntimeError to REFUSE work it has decided not
        # to attempt — today that is the durable retry ladder declining another
        # generation. Nothing was built, so this is a temporary refusal rather
        # than a server fault, and 503 keeps it separable from a build that
        # tried and broke.
        raise HTTPException(
            status_code=503,
            detail=f"Research Lab attested allocation was refused: {str(exc)[:200]}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Research Lab attested allocation unavailable: {str(exc)[:200]}",
        ) from exc
    if attestation.get("status") != "matched":
        # The HTTPException raised below IS surfaced by the build task's
        # failure callback, but only generically (error_type=HTTPException).
        # The attestation status that caused it is not, so record it here:
        # a 503 inside the submission window should be greppable by the
        # status that produced it, not just by an access-log code.
        logger.error(
            "research_lab_attested_allocation_not_ready epoch=%s "
            "persist_snapshot=%s status=%s",
            int(epoch),
            bool(persist_snapshot),
            attestation.get("status", "unknown"),
        )
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
    # Best-effort restart-surviving copy so a gateway restart mid-window can
    # warm-start instead of rebuilding cold (fail-open on any disk error).
    await asyncio.to_thread(
        allocation_handoff_disk_cache.store_handoff,
        int(BITTENSOR_NETUID),
        int(epoch),
        persist_snapshot,
        _allocation_cache_release_commit(),
        handoff,
        ttl_seconds=_ALLOCATION_CACHE_TTL_SECONDS,
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

    try:
        is_registered, _role = await chain_is_hotkey_registered(
            payload.miner_hotkey
        )
    except ChainRegistrationUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="subnet registration check is temporarily unavailable; retry shortly",
            headers={"Retry-After": "30"},
        ) from exc
    if not is_registered:
        raise HTTPException(status_code=403, detail="hotkey is not registered on this subnet")
def _require_enabled(enabled: bool, detail: str) -> None:
    if not enabled:
        raise HTTPException(status_code=403, detail=detail)
