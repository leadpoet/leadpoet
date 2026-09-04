"""Restart-safe SOURCE_ADD provenance, catalog, and reward orchestration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlsplit

from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.source_add_provenance import (
    rebuild_attested_provenance_result_v2,
)
from gateway.research_lab.store import call_rpc, select_many, select_one
from gateway.research_lab.v2_authority import (
    authorize_reward_decision_v2,
    evaluate_source_add_functional_probe_v2,
    evaluate_source_add_provenance_v2,
)
from gateway.tee.reward_executor_v2 import source_add_reward_row_projection_v2
from leadpoet_canonical.attested_v2 import sha256_json
from research_lab.source_add_rewards import create_leg1_reward

from . import source_add_catalog as source_add_catalog_contract


logger = logging.getLogger(__name__)

_SECRET_QUERY_NAMES = frozenset(
    {"access_token", "api-key", "api_key", "apikey", "key", "token"}
)
_NON_DATA_PATH_PARTS = frozenset(
    {
        "docs",
        "documentation",
        "health",
        "healthcheck",
        "openapi.json",
        "ping",
        "ready",
        "readiness",
        "status",
        "swagger",
        "version",
    }
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TERMINAL_PROVIDER_FAILURE_TYPES = frozenset(
    {"response_too_large", "tls_failure"}
)
_DOCUMENTATION_HTTP_400_MAX_ATTEMPTS = 3


class SourceAddWorkflowError(RuntimeError):
    """SOURCE_ADD queue state or measured authority is inconsistent."""


async def source_add_control_state() -> dict[str, Any]:
    """Return the durable SOURCE_ADD queue control, failing closed on drift."""

    try:
        row = await select_one(
            "research_lab_source_add_control",
            columns="paused,reason,updated_at",
            filters=(("singleton", True),),
        )
    except Exception as exc:
        logger.warning(
            "SOURCE_ADD_CONTROL_UNAVAILABLE type=%s", type(exc).__name__
        )
        return {
            "paused": True,
            "status": "unavailable_fail_closed",
            "unavailable": True,
        }
    if not isinstance(row, Mapping):
        logger.warning("SOURCE_ADD_CONTROL_MISSING")
        return {
            "paused": True,
            "status": "unavailable_fail_closed",
            "unavailable": True,
        }
    paused = bool(row.get("paused", True))
    return {
        "paused": paused,
        "status": "paused" if paused else "active",
        "reason": str(row.get("reason") or ""),
        "updated_at": row.get("updated_at"),
        "unavailable": False,
    }


def source_add_ref(prefix: str, *parts: Any) -> str:
    digest = sha256_json({"prefix": prefix, "parts": list(parts)}).split(":", 1)[1]
    return "%s:%s" % (prefix, digest[:16])


def source_add_work_id(
    submission_id: str, work_kind: str, discriminator: str = "initial"
) -> str:
    return source_add_ref(
        "source_add_work", str(submission_id), str(work_kind), str(discriminator)
    )


def source_add_probe_config_ref(
    submission_id: str,
    probe_doc: Mapping[str, Any],
    *,
    credential_value_hash: str = "",
) -> str:
    return source_add_ref(
        "source_add_probe_config",
        submission_id,
        sha256_json(probe_doc),
        str(credential_value_hash or ""),
    )


def source_add_probe_attempt_ref(
    submission_id: str, work_id: str, attempt_number: int
) -> str:
    return source_add_ref(
        "source_add_probe_attempt", submission_id, work_id, int(attempt_number)
    )


def source_add_reward_intent_id(submission_id: str, adapter_id: str) -> str:
    return source_add_ref("source_add_reward_intent", submission_id, adapter_id, 1)


def _provenance_result_from_submission(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the exact attested provenance result from durable state."""

    submission_doc = row.get("submission_doc")
    precheck_doc = row.get("precheck_doc")
    if not isinstance(submission_doc, Mapping) or not isinstance(
        precheck_doc, Mapping
    ):
        raise SourceAddWorkflowError(
            "SOURCE_ADD credible provenance authority is missing"
        )
    try:
        return rebuild_attested_provenance_result_v2(
            submission_id=str(row.get("submission_id") or ""),
            precheck_status=str(row.get("precheck_status") or ""),
            precheck_doc=precheck_doc,
            submission_doc=submission_doc,
        )
    except ValueError as exc:
        raise SourceAddWorkflowError(str(exc)) from exc


def source_add_host_hash(base_url: str) -> str:
    host = str(urlsplit(str(base_url or "")).hostname or "").strip().lower()
    if not host:
        raise SourceAddWorkflowError("SOURCE_ADD provider host is missing")
    return sha256_json({"source_add_destination_host": host})


def _current_builtin_disabled_provision_row(
    *,
    submission_id: str,
    config_ref: str,
    catalog_row: Mapping[str, Any],
    provision_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact inactive provisioning event for a late builtin match."""

    provision_doc = provision_row.get("provision_doc")
    if not isinstance(provision_doc, Mapping):
        raise SourceAddWorkflowError(
            "SOURCE_ADD current-model rejection provisioning is invalid"
        )
    disabled_doc = source_add_catalog_contract.sanitize_source_add_doc(provision_doc)
    registry_entry = disabled_doc.get("provider_registry_entry")
    if not isinstance(registry_entry, Mapping):
        raise SourceAddWorkflowError(
            "SOURCE_ADD current-model rejection registry entry is invalid"
        )
    disabled_registry_entry = dict(registry_entry)
    disabled_registry_entry["active"] = False
    disabled_doc["provider_registry_entry"] = disabled_registry_entry

    disabled_row = dict(provision_row)
    disabled_row["provision_status"] = "disabled"
    disabled_row["provision_doc"] = disabled_doc
    disabled_row["provision_ref"] = source_add_ref(
        "source_add_provision",
        submission_id,
        str(provision_row.get("registry_provider_id") or ""),
        "disabled",
        config_ref,
        str(provision_row.get("provision_ref") or ""),
        sha256_json(dict(catalog_row)),
        sha256_json(disabled_doc),
    )
    return disabled_row


def build_automatic_probe_config(
    *,
    submission_id: str,
    adapter_id: str,
    source_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Select up to three safe executable GET examples without credentials."""

    auth_type = str(source_metadata.get("auth_type") or "").strip().lower()
    if auth_type != "none":
        return None, "operator_credential_required"
    base_url = str(source_metadata.get("api_base_url") or "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    try:
        port = parsed.port or 443
    except ValueError:
        return None, "invalid_api_base_url"
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or port != 443
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not _safe_executable_path(parsed.path or "/")
    ):
        return None, "https_api_base_url_required"

    probes: list[dict[str, Any]] = []
    examples = source_metadata.get("endpoint_examples")
    if not isinstance(examples, list):
        return None, "endpoint_examples_missing"
    for example in examples:
        if not isinstance(example, Mapping):
            continue
        method = str(example.get("method") or "").upper()
        path = str(example.get("path") or "").strip()
        purpose = str(example.get("purpose") or "").lower()
        if method != "GET" or not _safe_executable_path(path):
            continue
        path_parts = {part.lower() for part in path.rstrip("/").split("/") if part}
        if path_parts.intersection(_NON_DATA_PATH_PARTS) or any(
            marker in purpose
            for marker in ("health check", "documentation", "status endpoint")
        ):
            continue
        query = _parse_example_query(str(example.get("example_query") or ""))
        if query is None:
            continue
        probes.append(
            {"method": "GET", "path": path, "query": query, "body_json": None}
        )
        if len(probes) == 3:
            break
    if not probes:
        return None, "operator_probe_configuration_required"

    provider_suffix = sha256_json(
        {"submission_id": submission_id, "adapter_id": adapter_id}
    ).split(":", 1)[1][:16]
    return {
        "schema_version": "leadpoet.source_add_probe_config.v2",
        "provider_id": "sourceadd_%s" % provider_suffix,
        "base_url": base_url,
        "auth_kind": "none",
        "auth_name": "",
        "request_headers": {},
        "probes": probes,
    }, "automatic_safe_get_selected"


def _safe_executable_path(path: str) -> bool:
    return bool(
        path.startswith("/")
        and "?" not in path
        and "#" not in path
        and "%" not in path
        and "\\" not in path
        and not re.search(r"[{}<>\[\]]|(^|/):[A-Za-z_]", path)
        and not any(part in {".", ".."} for part in path.split("/"))
        and not any(ord(char) < 32 or ord(char) == 127 for char in path)
        and not any(char.isspace() for char in path)
    )


def _parse_example_query(value: str) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if text.lower() in {"", "n/a", "na", "no params", "none"}:
        return {}
    parsed: Any
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, Mapping):
            return None
        pairs = list(parsed.items())
    else:
        if "=" not in text:
            return None
        pairs = parse_qsl(text.lstrip("?"), keep_blank_values=True)
        if len({str(name) for name, _ in pairs}) != len(pairs):
            return None
    output: dict[str, Any] = {}
    for name, item in pairs:
        normalized_name = str(name).strip()
        if (
            not normalized_name
            or len(normalized_name) > 120
            or normalized_name.lower() in _SECRET_QUERY_NAMES
            or not isinstance(item, (str, int, float, bool))
            or len(str(item)) > 500
        ):
            return None
        output[normalized_name] = item
    return output


async def process_source_add_work_item(
    work: Mapping[str, Any], *, config: ResearchLabGatewayConfig
) -> dict[str, Any]:
    if _requires_uncertain_recovery(work):
        return await _finish_uncertain_provider_outcome(work)
    kind = str(work.get("work_kind") or "")
    if kind == "provenance":
        return await _process_provenance(work, config=config)
    if kind == "functional_probe":
        return await _process_functional_probe(work, config=config)
    if kind == "provisioning_smoke":
        return await _process_provisioning_smoke(work, config=config)
    if kind == "leg1_reward":
        return await _process_leg1_reward(work, config=config)
    raise SourceAddWorkflowError("unsupported SOURCE_ADD work kind")


async def _process_provenance(
    work: Mapping[str, Any], *, config: ResearchLabGatewayConfig
) -> dict[str, Any]:
    current = await _load_submission(str(work.get("submission_id") or ""))
    document = _submission_document(current)
    manifest = document.get("manifest")
    metadata = document.get("source_metadata")
    if not isinstance(manifest, Mapping) or not isinstance(metadata, Mapping):
        raise SourceAddWorkflowError("SOURCE_ADD provenance input is incomplete")
    work = await _begin_provider_execution(work)
    attempt_count = int(work.get("attempt_count") or 0)
    precheck, outcome = await evaluate_source_add_provenance_v2(
        submission_id=str(current["submission_id"]),
        source_name=str(manifest.get("source_name") or ""),
        source_kind=str(manifest.get("source_kind") or ""),
        declared_base_domains=[
            str(item) for item in manifest.get("declared_base_domains") or ()
        ],
        source_metadata=dict(metadata),
        epoch_id=max(0, int(getattr(config, "evaluation_epoch", 0) or 0)),
        sequence=attempt_count,
        timeout_seconds=int(config.source_add_probe_timeout_seconds),
    )
    precheck_doc = precheck.to_record_doc()
    provenance_result = outcome.get("result")
    if not isinstance(provenance_result, Mapping):
        raise SourceAddWorkflowError("SOURCE_ADD provenance result is missing")
    provenance_result = dict(provenance_result)
    reasons = {str(item) for item in precheck.reasons}
    docs_fetch = precheck_doc.get("docs_fetch")
    docs_http_status = docs_fetch.get("status", 0) if isinstance(
        docs_fetch, Mapping
    ) else 0
    if not isinstance(docs_http_status, int):
        docs_http_status = 0
    terminal_provider_failure = any(
        str(section.get("error_type") or "").strip().lower()
        in _TERMINAL_PROVIDER_FAILURE_TYPES
        for section in (
            precheck_doc.get("docs_fetch"),
            precheck_doc.get("archive_history"),
            precheck_doc.get("ai_mode"),
        )
        if isinstance(section, Mapping)
    )
    # ScrapingDog has historically surfaced transient shedding as HTTP 400,
    # so preserve two retries. A third 400 is deterministic enough to stop
    # consuming provider credits and leave the source for manual review.
    documentation_http_400_retryable = (
        docs_http_status != 400
        or attempt_count < _DOCUMENTATION_HTTP_400_MAX_ATTEMPTS
    )
    retryable_provider_failure = (
        not terminal_provider_failure
        and documentation_http_400_retryable
        and (
            any(
                item.endswith("provider_error")
                or item == "scrapingdog_key_missing"
                for item in reasons
            )
            or (
                "documentation_fetch_failed" in reasons
                and (
                    docs_http_status in {400, 408, 425, 429}
                    or 500 <= docs_http_status < 600
                )
            )
        )
    )
    if (
        precheck.precheck_status == "needs_manual_review"
        and retryable_provider_failure
        and _retry_allowed(
            work, attempt_count, config.source_add_probe_max_attempts
        )
    ):
        return await _finish_work(
            work,
            disposition="retry",
            stage="needs_manual_review",
            submission_doc={**document, "precheck_status": precheck.precheck_status, "precheck_doc": precheck_doc},
            precheck_status=precheck.precheck_status,
            precheck_doc=precheck_doc,
            result_doc={"status": "provenance_retryable", "reason_codes": sorted(reasons)},
            available_at=_retry_at(attempt_count),
        )

    probe_config = {}
    next_work = {}
    stage = precheck.precheck_status
    release_identity = precheck.precheck_status == "rejected_precheck"
    if precheck.precheck_status == "provenance_precheck_passed":
        if config.source_add_functional_probes_enabled:
            selected, selection_reason = build_automatic_probe_config(
                submission_id=str(current["submission_id"]),
                adapter_id=str(current["adapter_id"]),
                source_metadata=metadata,
            )
            if selected is not None:
                config_ref = source_add_probe_config_ref(
                    str(current["submission_id"]), selected
                )
                functional_work_id = source_add_work_id(
                    str(current["submission_id"]),
                    "functional_probe",
                    "%s:%s" % (config_ref, str(work.get("work_id") or "")),
                )
                probe_config = {
                    "config_ref": config_ref,
                    "probe_doc": selected,
                    "credential_envelope": {},
                    "actor_ref": "system:source-add-auto-probe",
                }
                next_work = {
                    "work_id": functional_work_id,
                    "work_kind": "functional_probe",
                    "priority": 20,
                    "job_doc": {
                        "config_ref": config_ref,
                        "host_hash": source_add_host_hash(str(selected["base_url"])),
                    },
                }
                stage = "functional_probe_queued"
                reasons.add(selection_reason)
            elif selection_reason == "operator_credential_required":
                stage = "awaiting_operator_credential"
                reasons.add(selection_reason)
            else:
                stage = "needs_manual_review"
                reasons.add(selection_reason)
        else:
            stage = "provenance_precheck_passed"
    precheck_doc = {**precheck_doc, "reasons": sorted(reasons)}
    result = await _finish_work(
        work,
        disposition="complete",
        stage=stage,
        submission_doc={
            **document,
            "precheck_status": precheck.precheck_status,
            "precheck_doc": precheck_doc,
            "provenance_receipt_hash": str(
                (outcome.get("execution_receipt") or outcome.get("receipt") or {}).get(
                    "receipt_hash"
                )
                or ""
            ),
            "provenance_artifact_hash": sha256_json(provenance_result),
            "provenance_result": provenance_result,
        },
        precheck_status=precheck.precheck_status,
        precheck_doc=precheck_doc,
        result_doc={"status": stage, "reason_codes": sorted(reasons)},
        probe_config=probe_config,
        next_work=next_work,
        release_identity=release_identity,
    )
    leg1_status = "not_eligible"
    if precheck.precheck_status == "provenance_precheck_passed":
        receipt = outcome.get("execution_receipt") or outcome.get("receipt") or {}
        receipt_hash = str(receipt.get("receipt_hash") or "")
        business_hash = str(receipt.get("output_root") or "")
        if not _HASH_RE.fullmatch(receipt_hash) or not _HASH_RE.fullmatch(
            business_hash
        ):
            raise SourceAddWorkflowError(
                "SOURCE_ADD provenance reward authority is invalid"
            )
        intent_id = source_add_reward_intent_id(
            str(current["submission_id"]), str(current["adapter_id"])
        )
        reward_work_id = source_add_work_id(
            str(current["submission_id"]), "leg1_reward", intent_id
        )
        try:
            queued = await _rpc(
                "research_lab_source_add_enqueue_leg1_after_provenance_v1",
                {
                    "p_submission_id": str(current["submission_id"]),
                    "p_intent_id": intent_id,
                    "p_reward_work_id": reward_work_id,
                    "p_provenance_receipt_hash": receipt_hash,
                    "p_provenance_artifact_hash": business_hash,
                },
            )
            leg1_status = str(queued.get("status") or "")
            if leg1_status not in {
                "queued",
                "already_queued",
                "already_created",
            }:
                raise SourceAddWorkflowError(
                    "SOURCE_ADD provenance Leg 1 enqueue returned %s"
                    % (leg1_status or "an empty status")
                )
        except Exception as exc:
            # Provenance completion is already durable. Reconciliation repairs
            # this bounded transaction gap from the same attested receipt,
            # without rerunning a paid provider call.
            leg1_status = "reconciliation_pending"
            logger.warning(
                "SOURCE_ADD_PROVENANCE_LEG1_RECONCILIATION_PENDING "
                "submission_id=%s type=%s",
                current["submission_id"],
                type(exc).__name__,
            )
    logger.info(
        "SOURCE_ADD_PROVENANCE_FINISHED submission_id=%s stage=%s leg1_status=%s",
        current["submission_id"],
        stage,
        leg1_status,
    )
    if leg1_status == "not_eligible":
        return result
    return {**result, "leg1_reward_status": leg1_status}


async def _process_functional_probe(
    work: Mapping[str, Any], *, config: ResearchLabGatewayConfig
) -> dict[str, Any]:
    if not config.source_add_functional_probes_enabled:
        return await _finish_work(
            work,
            disposition="retry",
            result_doc={"status": "functional_probes_disabled"},
            available_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    current = await _load_submission(str(work.get("submission_id") or ""))
    document = _submission_document(current)
    config_row = await select_one(
        "research_lab_source_add_probe_config_current",
        filters=(
            ("submission_id", str(current["submission_id"])),
            ("config_status", "active"),
        ),
    )
    if not isinstance(config_row, Mapping):
        return await _finish_work(
            work,
            disposition="complete",
            stage="needs_manual_review",
            submission_doc=document,
            precheck_status=str(current.get("precheck_status") or ""),
            precheck_doc=dict(current.get("precheck_doc") or {}),
            result_doc={"status": "probe_config_missing"},
        )
    probe_doc = config_row.get("probe_doc")
    credential_envelope = config_row.get("credential_envelope")
    if (
        isinstance(probe_doc, Mapping)
        and str(probe_doc.get("auth_kind") or "none") != "none"
        and not credential_envelope
    ):
        return await _finish_work(
            work,
            disposition="complete",
            stage="awaiting_operator_credential",
            submission_doc=document,
            precheck_status=str(current.get("precheck_status") or ""),
            precheck_doc=dict(current.get("precheck_doc") or {}),
            result_doc={"status": "operator_credential_required"},
        )

    attempt_number = int(work.get("attempt_count") or 0)
    attempt_ref = source_add_probe_attempt_ref(
        str(current["submission_id"]), str(work["work_id"]), attempt_number
    )
    work = await _begin_provider_execution(work)
    result, outcome = await evaluate_source_add_functional_probe_v2(
        submission_id=str(current["submission_id"]),
        config_ref=str(config_row.get("config_ref") or ""),
        evaluation_mode="functional_probe",
        epoch_id=max(0, int(getattr(config, "evaluation_epoch", 0) or 0)),
        sequence=attempt_number,
        artifact_ref=attempt_ref,
        timeout_seconds=int(config.source_add_probe_timeout_seconds),
    )
    functional_attempt = _functional_attempt_doc(
        attempt_ref=attempt_ref,
        evaluation_mode="functional_probe",
        config_ref=str(config_row.get("config_ref") or ""),
        result=result,
        outcome=outcome,
    )
    status = str(result["result_status"])
    workflow_result = dict(result)
    if status == "retryable" and _retry_allowed(
        work, attempt_number, config.source_add_probe_max_attempts
    ):
        return await _finish_work(
            work,
            disposition="retry",
            stage="functional_probe_retryable",
            submission_doc={**document, "functional_probe": functional_attempt},
            precheck_status=str(current.get("precheck_status") or ""),
            precheck_doc=dict(current.get("precheck_doc") or {}),
            result_doc=workflow_result,
            functional_attempt=functional_attempt,
            available_at=_retry_at(
                attempt_number,
                retry_after_seconds=int(result.get("retry_after_seconds") or 0),
            ),
        )

    release_identity = False
    stage = {
        "passed": "functional_probe_passed",
        "retryable": "needs_manual_review",
        "awaiting_operator": "awaiting_operator_credential",
        "manual_review": "needs_manual_review",
        "failed": "functional_probe_failed",
    }[status]
    if status == "failed":
        release_identity = True
    response = await _finish_work(
        work,
        disposition="complete",
        stage=stage,
        submission_doc={**document, "functional_probe": functional_attempt},
        precheck_status=str(current.get("precheck_status") or ""),
        precheck_doc=dict(current.get("precheck_doc") or {}),
        result_doc=workflow_result,
        functional_attempt=functional_attempt,
        release_identity=release_identity,
    )
    logger.info(
        "SOURCE_ADD_FUNCTIONAL_FINISHED submission_id=%s stage=%s status=%s",
        current["submission_id"],
        stage,
        status,
    )
    return response


async def _process_provisioning_smoke(
    work: Mapping[str, Any], *, config: ResearchLabGatewayConfig
) -> dict[str, Any]:
    """Run and atomically finalize one operator-requested eligibility smoke."""

    if not config.source_add_functional_probes_enabled:
        return await _finish_work(
            work,
            disposition="retry",
            result_doc={"status": "functional_probes_disabled"},
            available_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    current = await _load_submission(str(work.get("submission_id") or ""))
    document = _submission_document(current)
    job_doc = work.get("job_doc") if isinstance(work.get("job_doc"), Mapping) else {}
    catalog_row = job_doc.get("catalog_row")
    provision_row = job_doc.get("provision_row")
    config_ref = str(job_doc.get("config_ref") or "")
    if (
        not isinstance(catalog_row, Mapping)
        or not isinstance(provision_row, Mapping)
        or str(provision_row.get("provision_status") or "")
        != "provisioned_autoresearch_eligible"
    ):
        raise SourceAddWorkflowError("SOURCE_ADD provisioning smoke input is invalid")

    attempt_number = int(work.get("attempt_count") or 0)
    attempt_ref = source_add_probe_attempt_ref(
        str(current["submission_id"]), str(work["work_id"]), attempt_number
    )
    work = await _begin_provider_execution(work)
    result, outcome = await evaluate_source_add_functional_probe_v2(
        submission_id=str(current["submission_id"]),
        config_ref=config_ref,
        evaluation_mode="provisioning_smoke",
        epoch_id=max(0, int(getattr(config, "evaluation_epoch", 0) or 0)),
        sequence=attempt_number,
        artifact_ref=attempt_ref,
        timeout_seconds=int(config.source_add_probe_timeout_seconds),
    )
    smoke_attempt = _functional_attempt_doc(
        attempt_ref=attempt_ref,
        evaluation_mode="provisioning_smoke",
        config_ref=config_ref,
        result=result,
        outcome=outcome,
    )
    smoke_attempt["work_id"] = str(work["work_id"])
    smoke_attempt["attempt_number"] = attempt_number
    status = str(result.get("result_status") or "")
    if status == "passed":
        metadata = (
            document.get("source_metadata")
            if isinstance(document.get("source_metadata"), Mapping)
            else {}
        )
        provision_doc = (
            provision_row.get("provision_doc")
            if isinstance(provision_row.get("provision_doc"), Mapping)
            else {}
        )
        registry_entry = (
            provision_doc.get("provider_registry_entry")
            if isinstance(provision_doc.get("provider_registry_entry"), Mapping)
            else {}
        )
        submitted_base_url = str(metadata.get("api_base_url") or "")
        tested_base_url = str(registry_entry.get("base_url") or "")
        try:
            current_model_uses_api = await asyncio.to_thread(
                source_add_catalog_contract.source_add_api_is_current_builtin_sync,
                submitted_base_url,
                tested_base_url=tested_base_url,
            )
        except Exception as exc:
            retry_allowed = _retry_allowed(
                work, attempt_number, config.source_add_probe_max_attempts
            )
            logger.warning(
                "SOURCE_ADD_CURRENT_MODEL_CATALOG_UNAVAILABLE "
                "submission_id=%s type=%s",
                current["submission_id"],
                type(exc).__name__,
            )
            return await _finish_work(
                work,
                disposition="retry" if retry_allowed else "complete",
                submission_doc=document,
                precheck_status=str(current.get("precheck_status") or ""),
                precheck_doc=dict(current.get("precheck_doc") or {}),
                result_doc={"status": "current_model_catalog_unavailable"},
                functional_attempt=smoke_attempt,
                available_at=_retry_at(attempt_number) if retry_allowed else None,
            )
        if current_model_uses_api:
            logger.info(
                "SOURCE_ADD_FINAL_ACCEPTANCE_REJECTED submission_id=%s reason=current_model_provider",
                current["submission_id"],
            )
            rejected = await _rpc(
                "research_lab_source_add_reject_current_builtin_v3",
                {
                    "p_work_id": str(work["work_id"]),
                    "p_lease_token": str(work["lease_token"]),
                    "p_submission_id": str(current["submission_id"]),
                    "p_submission_doc": document,
                    "p_precheck_status": str(current.get("precheck_status") or ""),
                    "p_precheck_doc": dict(current.get("precheck_doc") or {}),
                    "p_catalog_row": dict(catalog_row),
                    "p_disabled_provision_row": (
                        _current_builtin_disabled_provision_row(
                            submission_id=str(current["submission_id"]),
                            config_ref=config_ref,
                            catalog_row=catalog_row,
                            provision_row=provision_row,
                        )
                    ),
                    "p_smoke_attempt": smoke_attempt,
                },
            )
            if str(rejected.get("status") or "") != "not_eligible":
                raise SourceAddWorkflowError(
                    "SOURCE_ADD current-model rejection did not finalize"
                )
            return rejected
        functional = await select_one(
            "research_lab_source_add_functional_probe_current",
            filters=(("submission_id", str(current["submission_id"])),),
        )
        if (
            not isinstance(functional, Mapping)
            or str(functional.get("adapter_id") or "")
            != str(current["adapter_id"])
            or str(functional.get("result_status") or "") != "passed"
            or not _HASH_RE.fullmatch(str(functional.get("receipt_hash") or ""))
            or not _HASH_RE.fullmatch(
                str(functional.get("business_artifact_hash") or "")
            )
        ):
            raise SourceAddWorkflowError(
                "SOURCE_ADD accepted functional proof is missing"
            )
        finalized = await _rpc(
            "research_lab_source_add_finalize_provision_smoke_v3",
            {
                "p_work_id": str(work["work_id"]),
                "p_lease_token": str(work["lease_token"]),
                "p_submission_id": str(current["submission_id"]),
                "p_catalog_row": dict(catalog_row),
                "p_provision_row": dict(provision_row),
                "p_smoke_attempt": smoke_attempt,
            },
        )
        final_status = str(finalized.get("status") or "")
        log = logger.info if final_status in {
            "provisioned",
            "already_provisioned",
        } else logger.warning
        log(
            "SOURCE_ADD_PROVISIONING_SMOKE_FINISHED submission_id=%s status=%s",
            current["submission_id"],
            final_status or "missing",
        )
        return finalized

    if status == "retryable" and _retry_allowed(
        work, attempt_number, config.source_add_probe_max_attempts
    ):
        disposition = "retry"
        available_at = _retry_at(
            attempt_number,
            retry_after_seconds=int(result.get("retry_after_seconds") or 0),
        )
    else:
        disposition = "complete"
        available_at = None
    finished = await _finish_work(
        work,
        disposition=disposition,
        submission_doc=document,
        precheck_status=str(current.get("precheck_status") or ""),
        precheck_doc=dict(current.get("precheck_doc") or {}),
        result_doc=dict(result),
        functional_attempt=smoke_attempt,
        available_at=available_at,
    )
    logger.info(
        "SOURCE_ADD_PROVISIONING_SMOKE_FINISHED submission_id=%s status=%s disposition=%s",
        current["submission_id"],
        status,
        disposition,
    )
    return finished


def _functional_attempt_doc(
    *,
    attempt_ref: str,
    evaluation_mode: str,
    config_ref: str,
    result: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = outcome.get("execution_receipt") or outcome.get("receipt") or {}
    receipt_hash = str(receipt.get("receipt_hash") or "") if isinstance(receipt, Mapping) else ""
    business_hash = str(receipt.get("output_root") or "") if isinstance(receipt, Mapping) else ""
    if not _HASH_RE.fullmatch(receipt_hash) or business_hash != sha256_json(dict(result)):
        raise SourceAddWorkflowError("SOURCE_ADD functional receipt binding differs")
    return {
        "attempt_ref": attempt_ref,
        "evaluation_mode": evaluation_mode,
        "config_ref": config_ref,
        "result_status": str(result["result_status"]),
        "route_hash": str(result["route_hash"]),
        "response_hash": str(result["response_hash"]),
        "status_class": str(result["status_class"]),
        "content_type": str(result["content_type"]),
        "byte_count": int(result["byte_count"]),
        "duration_ms": int(result["duration_ms"]),
        "retry_after_seconds": int(result.get("retry_after_seconds") or 0),
        "reason_codes": list(result["reason_codes"]),
        "receipt_hash": receipt_hash,
        "business_artifact_hash": business_hash,
        "result_doc": dict(result),
    }


async def _process_leg1_reward(
    work: Mapping[str, Any], *, config: ResearchLabGatewayConfig
) -> dict[str, Any]:
    if not config.source_add_rewards_enabled:
        return await _finish_work(
            work,
            disposition="retry",
            result_doc={"status": "rewards_disabled"},
            available_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    job_doc = work.get("job_doc") if isinstance(work.get("job_doc"), Mapping) else {}
    intent_id = str(job_doc.get("intent_id") or "")
    intent = await select_one(
        "research_lab_source_add_reward_intents",
        filters=(("intent_id", intent_id),),
    )
    if not isinstance(intent, Mapping):
        raise SourceAddWorkflowError("SOURCE_ADD reward intent is missing")
    slot = await _rpc(
        "research_lab_source_add_reserve_leg1_slot_v4",
        {
            "p_intent_id": intent_id,
            "p_work_id": str(work["work_id"]),
            "p_work_lease_token": str(work["lease_token"]),
            "p_daily_cap": int(config.source_add_leg1_max_per_utc_day),
            "p_slot_lease_seconds": int(config.source_add_work_lease_seconds),
        },
    )
    slot_status = str(slot.get("status") or "")
    if slot_status in {
        "already_created",
        "daily_cap_fifo",
        "fifo_wait",
        "lease_lost",
    }:
        return slot
    if slot_status != "reserved":
        raise SourceAddWorkflowError(
            "SOURCE_ADD reward slot reservation returned %s"
            % (slot_status or "an empty status")
        )

    current = await _load_submission(str(work.get("submission_id") or ""))
    document = _submission_document(current)
    provenance_result = _provenance_result_from_submission(current)
    receipt_hash = str(document.get("provenance_receipt_hash") or "")
    business_hash = sha256_json(provenance_result)
    if (
        str(intent.get("approval_kind") or "")
        != "provenance_precheck_passed"
        or not _HASH_RE.fullmatch(receipt_hash)
        or receipt_hash != str(intent.get("provenance_receipt_hash") or "")
        or business_hash != str(intent.get("provenance_artifact_hash") or "")
    ):
        raise SourceAddWorkflowError("SOURCE_ADD provenance reward intent differs")
    from gateway.research_lab.attested_v2_store import load_business_artifact_graph_v2

    provenance_graph = await load_business_artifact_graph_v2(
        artifact_kind="source_add_provenance",
        artifact_ref=str(current["submission_id"]),
        artifact_hash=business_hash,
    )
    current_epoch, _block, _source = await resolve_research_lab_evaluation_epoch(
        getattr(config, "evaluation_epoch", 0)
    )
    start_epoch = max(0, int(current_epoch)) + 1
    trigger = {
        "provenance_precheck_passed": True,
        "submission_id": str(current["submission_id"]),
        "precheck_status": "provenance_precheck_passed",
        "provenance_receipt_hash": receipt_hash,
        "provenance_artifact_hash": business_hash,
        "provenance_result_hash": sha256_json(provenance_result),
    }
    existing_rewards = []
    existing = await select_one(
        "research_lab_source_add_reward_current",
        filters=(("adapter_id", str(current["adapter_id"])), ("leg", 1)),
    )
    if existing:
        existing_rewards.append(dict(existing))
    leg1 = create_leg1_reward(
        adapter_id=str(current["adapter_id"]),
        miner_ref=str(current["miner_hotkey"]),
        start_epoch=start_epoch,
        existing_rewards=existing_rewards,
        alpha_percent=float(config.source_add_leg1_alpha_percent),
        reward_epochs=int(config.lab_reward_epochs),
        trigger_evidence=trigger,
    )
    if leg1 is None:
        raise SourceAddWorkflowError("SOURCE_ADD reward already exists after slot reservation")
    expected_decision_artifact_hash = sha256_json(
        source_add_reward_row_projection_v2("source_add_leg1", leg1.to_dict())
    )
    existing_decision_link = await select_one(
        "research_lab_attested_business_artifact_links_v2",
        filters=(
            ("artifact_kind", "source_add_reward_decision"),
            ("artifact_ref", leg1.reward_ref),
            ("artifact_hash", expected_decision_artifact_hash),
        ),
    )
    decision_receipt: Mapping[str, Any] | None = None
    if isinstance(existing_decision_link, Mapping):
        decision_graph = await load_business_artifact_graph_v2(
            artifact_kind="source_add_reward_decision",
            artifact_ref=leg1.reward_ref,
            artifact_hash=expected_decision_artifact_hash,
        )
        decision_root = str(decision_graph.get("root_receipt_hash") or "")
        decision_receipt = next(
            (
                receipt
                for receipt in decision_graph.get("receipts") or ()
                if isinstance(receipt, Mapping)
                and receipt.get("receipt_hash") == decision_root
            ),
            None,
        )
        if (
            not isinstance(decision_receipt, Mapping)
            or decision_receipt.get("purpose")
            != "research_lab.reward_decision.v2"
            or decision_receipt.get("output_root")
            != expected_decision_artifact_hash
            or decision_receipt.get("parent_receipt_hashes") != [receipt_hash]
        ):
            raise SourceAddWorkflowError(
                "SOURCE_ADD recovered reward decision differs"
            )
    else:
        authority = await authorize_reward_decision_v2(
            epoch_id=max(0, start_epoch - 1),
            decision_kind="source_add_leg1",
            decision_payload={
                "adapter_id": str(current["adapter_id"]),
                "miner_ref": str(current["miner_hotkey"]),
                "start_epoch": start_epoch,
                "existing_rewards": existing_rewards,
                "alpha_percent": float(config.source_add_leg1_alpha_percent),
                "reward_epochs": int(config.lab_reward_epochs),
                "provenance_result": provenance_result,
                "trigger_evidence": trigger,
            },
            expected_result={
                "decision_kind": "source_add_leg1",
                "reward": leg1.to_dict(),
            },
            artifact_kind="source_add_reward_decision",
            artifact_ref=leg1.reward_ref,
            parent_graphs=(provenance_graph,),
        )
        decision_receipt = authority.get("execution_receipt") or authority.get(
            "receipt"
        )
    if not isinstance(decision_receipt, Mapping):
        raise SourceAddWorkflowError("SOURCE_ADD reward decision receipt is missing")
    decision_receipt_hash = str(decision_receipt.get("receipt_hash") or "")
    decision_artifact_hash = str(decision_receipt.get("output_root") or "")
    if (
        not _HASH_RE.fullmatch(decision_receipt_hash)
        or decision_artifact_hash != expected_decision_artifact_hash
    ):
        raise SourceAddWorkflowError("SOURCE_ADD reward decision receipt is invalid")
    finalized = await _rpc(
        "research_lab_source_add_finalize_leg1_v4",
        {
            "p_intent_id": intent_id,
            "p_work_id": str(work["work_id"]),
            "p_work_lease_token": str(work["lease_token"]),
            "p_slot_lease_token": str(slot["slot_lease_token"]),
            "p_daily_cap": int(config.source_add_leg1_max_per_utc_day),
            "p_reward": {
                "reward_ref": leg1.reward_ref,
                "reward_kind": leg1.reward_kind,
                "alpha_percent": leg1.alpha_percent,
                "reward_epochs": leg1.reward_epochs,
                "start_epoch": leg1.start_epoch,
                "state": leg1.state,
                "trigger_evidence_doc": trigger,
                "public_label": leg1.public_label,
                "decision_receipt_hash": decision_receipt_hash,
                "decision_artifact_hash": decision_artifact_hash,
            },
            "p_submission_doc": document,
        },
    )
    finalized_status = str(finalized.get("status") or "")
    if finalized_status not in {
        "created",
        "daily_cap_fifo",
        "fifo_wait",
        "lease_lost",
        "slot_day_rolled",
    }:
        raise SourceAddWorkflowError(
            "SOURCE_ADD reward finalization returned %s"
            % (finalized_status or "an empty status")
        )
    logger.info(
        "SOURCE_ADD_LEG1_FINALIZED submission_id=%s status=%s reward_ref=%s",
        current["submission_id"],
        finalized.get("status"),
        finalized.get("reward_ref"),
    )
    return finalized


async def run_source_add_dispatcher(
    *, config_supplier: Any = ResearchLabGatewayConfig.from_env
) -> None:
    """Run one independent SOURCE_ADD dispatcher until gateway shutdown."""

    worker_id = "source-add:%s:%s" % (os.getpid(), id(asyncio.current_task()))
    logger.info("SOURCE_ADD_DISPATCHER_STARTED worker_id=%s", worker_id)
    queue_paused = False
    next_leg1_reconciliation_at = 0.0
    while True:
        poll_seconds = 2.0
        try:
            config = config_supplier()
            poll_seconds = max(
                0.25, float(config.source_add_dispatcher_poll_seconds)
            )
            if not (
                config.source_add_enabled
                and config.source_add_dispatcher_enabled
            ):
                await asyncio.sleep(poll_seconds)
                continue
            claim = await _rpc(
                "research_lab_source_add_claim_work",
                {
                    "p_worker_id": worker_id,
                    "p_lease_seconds": int(config.source_add_work_lease_seconds),
                },
            )
            claim_status = str(claim.get("status") or "")
            if claim_status == "paused":
                if not queue_paused:
                    logger.info("SOURCE_ADD_DISPATCHER_PAUSED worker_id=%s", worker_id)
                queue_paused = True
                await asyncio.sleep(poll_seconds)
                continue
            if queue_paused:
                logger.info("SOURCE_ADD_DISPATCHER_RESUMED worker_id=%s", worker_id)
                queue_paused = False
            if claim_status != "claimed":
                now = asyncio.get_running_loop().time()
                if now >= next_leg1_reconciliation_at:
                    next_leg1_reconciliation_at = now + 30.0
                    try:
                        reconciled = await _rpc(
                            "research_lab_source_add_reconcile_provenance_leg1_v1",
                            {},
                        )
                        if str(reconciled.get("status") or "") != "reconciled":
                            raise SourceAddWorkflowError(
                                "SOURCE_ADD Leg 1 reconciliation returned "
                                "an invalid status"
                            )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # New provenance passes enqueue their reward work in
                        # the same transaction. Reconciliation only repairs
                        # historical gaps and must not starve durable work.
                        logger.warning(
                            "SOURCE_ADD_LEG1_RECONCILIATION_FAILED type=%s",
                            type(exc).__name__,
                        )
                await asyncio.sleep(poll_seconds)
                continue
            work = claim.get("work")
            if not isinstance(work, Mapping):
                raise SourceAddWorkflowError("SOURCE_ADD claimed work is invalid")
            try:
                await process_source_add_work_item(work, config=config)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "SOURCE_ADD_WORK_FAILED work_id=%s kind=%s type=%s",
                    work.get("work_id"),
                    work.get("work_kind"),
                    type(exc).__name__,
                )
                await _recover_failed_claim(work, config=config)
        except asyncio.CancelledError:
            logger.info("SOURCE_ADD_DISPATCHER_STOPPED worker_id=%s", worker_id)
            raise
        except Exception as exc:
            logger.warning(
                "SOURCE_ADD_DISPATCHER_LOOP_FAILED type=%s", type(exc).__name__
            )
            await asyncio.sleep(max(1.0, poll_seconds))


async def _recover_failed_claim(
    work: Mapping[str, Any], *, config: ResearchLabGatewayConfig
) -> None:
    persisted = await select_one(
        "research_lab_source_add_work_items",
        filters=(("work_id", str(work.get("work_id") or "")),),
    )
    if isinstance(persisted, Mapping):
        same_lease = str(persisted.get("lease_token") or "") == str(
            work.get("lease_token") or ""
        )
        job_doc = (
            persisted.get("job_doc")
            if isinstance(persisted.get("job_doc"), Mapping)
            else {}
        )
        if same_lease and job_doc.get("provider_execution_state") == "started":
            await _finish_uncertain_provider_outcome(persisted)
            return
    attempt_count = int(work.get("attempt_count") or 0)
    if str(work.get("work_kind") or "") == "leg1_reward":
        await _finish_work(
            work,
            disposition="retry",
            result_doc={
                "status": "reward_worker_exception_retry",
                "attempt_count": attempt_count,
            },
            available_at=datetime.now(timezone.utc)
            + timedelta(seconds=min(3600, 60 * (2 ** min(6, attempt_count)))),
        )
        return
    disposition = (
        "retry"
        if _retry_allowed(work, attempt_count, config.source_add_probe_max_attempts)
        else "complete"
    )
    current = await _load_submission(str(work.get("submission_id") or ""))
    await _finish_work(
        work,
        disposition=disposition,
        stage="needs_manual_review" if disposition == "complete" else "",
        submission_doc=_submission_document(current),
        precheck_status=str(current.get("precheck_status") or ""),
        precheck_doc=dict(current.get("precheck_doc") or {}),
        result_doc={
            "status": "worker_exception_retry" if disposition == "retry" else "worker_exception_dead_letter",
            "attempt_count": attempt_count,
        },
        available_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )


def _requires_uncertain_recovery(work: Mapping[str, Any]) -> bool:
    job_doc = work.get("job_doc")
    return bool(
        isinstance(job_doc, Mapping)
        and job_doc.get("provider_execution_state") == "started"
        and job_doc.get("provider_execution_recovery")
    )


async def _begin_provider_execution(work: Mapping[str, Any]) -> dict[str, Any]:
    begun = await _rpc(
        "research_lab_source_add_begin_provider_execution",
        {
            "p_work_id": str(work.get("work_id") or ""),
            "p_lease_token": str(work.get("lease_token") or ""),
        },
    )
    if begun.get("status") != "started" or not isinstance(
        begun.get("work"), Mapping
    ):
        raise SourceAddWorkflowError(
            "SOURCE_ADD provider execution fence was not acquired"
        )
    return dict(begun["work"])


async def _finish_uncertain_provider_outcome(
    work: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed after a lost worker may already have sent provider I/O."""

    current = await _load_submission(str(work.get("submission_id") or ""))
    document = _submission_document(current)
    kind = str(work.get("work_kind") or "")
    stage = "" if kind == "provisioning_smoke" else "needs_manual_review"
    precheck_status = str(current.get("precheck_status") or "")
    precheck_doc = dict(current.get("precheck_doc") or {})
    if kind == "provenance":
        precheck_status = "needs_manual_review"
        precheck_doc = {
            **precheck_doc,
            "precheck_status": "needs_manual_review",
            "reasons": ["provider_execution_outcome_unknown_after_worker_loss"],
        }
    result = await _finish_work(
        work,
        disposition="complete",
        stage=stage,
        submission_doc=document,
        precheck_status=precheck_status,
        precheck_doc=precheck_doc,
        result_doc={
            "status": "provider_execution_outcome_unknown_after_worker_loss",
            "reason_codes": [
                "provider_execution_outcome_unknown_after_worker_loss"
            ],
            "attempt_count": int(work.get("attempt_count") or 0),
        },
    )
    logger.warning(
        "SOURCE_ADD_PROVIDER_EXECUTION_UNCERTAIN work_id=%s kind=%s",
        work.get("work_id"),
        kind,
    )
    return result


async def _load_submission(submission_id: str) -> dict[str, Any]:
    row = await select_one(
        "research_lab_source_add_submission_current",
        filters=(("submission_id", submission_id),),
    )
    if not isinstance(row, Mapping):
        raise SourceAddWorkflowError("SOURCE_ADD submission is missing")
    return dict(row)


def _submission_document(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("submission_doc")
    if not isinstance(value, Mapping):
        raise SourceAddWorkflowError("SOURCE_ADD submission document is missing")
    return dict(value)


def _retry_allowed(
    work: Mapping[str, Any], attempt_count: int, max_attempts: int
) -> bool:
    if attempt_count >= max(1, int(max_attempts)):
        return False
    raw = str(work.get("created_at") or "")
    try:
        created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created <= timedelta(hours=24)


def _retry_at(attempt_count: int, *, retry_after_seconds: int = 0) -> datetime:
    seconds = min(6 * 60 * 60, 60 * (2 ** max(0, int(attempt_count) - 1)))
    if retry_after_seconds:
        seconds = max(seconds, min(6 * 60 * 60, max(0, int(retry_after_seconds))))
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


async def _finish_work(
    work: Mapping[str, Any],
    *,
    disposition: str,
    stage: str = "",
    submission_doc: Mapping[str, Any] | None = None,
    precheck_status: str = "",
    precheck_doc: Mapping[str, Any] | None = None,
    result_doc: Mapping[str, Any] | None = None,
    functional_attempt: Mapping[str, Any] | None = None,
    probe_config: Mapping[str, Any] | None = None,
    next_work: Mapping[str, Any] | None = None,
    reward_intent: Mapping[str, Any] | None = None,
    available_at: datetime | None = None,
    release_identity: bool = False,
) -> dict[str, Any]:
    return await _rpc(
        "research_lab_source_add_finish_work",
        {
            "p_work_id": str(work.get("work_id") or ""),
            "p_lease_token": str(work.get("lease_token") or ""),
            "p_disposition": disposition,
            "p_stage": stage,
            "p_submission_doc": dict(submission_doc or {}),
            "p_precheck_status": precheck_status,
            "p_precheck_doc": dict(precheck_doc or {}),
            "p_result_doc": dict(result_doc or {}),
            "p_functional_attempt": dict(functional_attempt or {}),
            "p_probe_config": dict(probe_config or {}),
            "p_next_work": dict(next_work or {}),
            "p_reward_intent": dict(reward_intent or {}),
            "p_available_at": available_at.isoformat() if available_at else None,
            "p_release_identity": bool(release_identity),
        },
    )


async def _rpc(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    value = await call_rpc(name, params)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Mapping):
        value = value[0]
    if not isinstance(value, Mapping):
        raise SourceAddWorkflowError("%s returned an invalid result" % name)
    return dict(value)


__all__ = [
    "SourceAddWorkflowError",
    "build_automatic_probe_config",
    "process_source_add_work_item",
    "run_source_add_dispatcher",
    "source_add_control_state",
    "source_add_probe_attempt_ref",
    "source_add_probe_config_ref",
    "source_add_host_hash",
    "source_add_ref",
    "source_add_reward_intent_id",
    "source_add_work_id",
]
