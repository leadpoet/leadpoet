"""Gateway-owned Research Lab private scoring worker."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import functools
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from gateway.research_lab.bundles import build_research_lab_audit_bundle
from gateway.research_lab.attested_scoring import (
    canonical_json_bytes as attested_canonical_json_bytes,
    compare_baseline_score_summary as compare_attested_baseline_score_summary,
    compare_promotion_gate_decision as compare_attested_promotion_gate_decision,
    compare_promotion_metric as compare_attested_promotion_metric,
    compare_score_bundle as compare_attested_score_bundle,
    persist_attested_outcome_artifact_links,
    resolve_attested_artifact_lineage,
    sha256_bytes as attested_sha256_bytes,
)
from gateway.research_lab.autoresearch_authority_v2 import (
    attest_stale_parent_rebase_v2,
)
from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.code_build import (
    CodeEditBuildError,
    CodeEditCandidateBuilder,
    CodeEditPatchApplyError,
)
from gateway.research_lab.config import (
    DEFAULT_BASELINE_START_UTC_OFFSET_SECONDS,
    DEFAULT_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS,
    ResearchLabGatewayConfig,
)
from gateway.research_lab.icp_window import (
    RollingIcpWindowUnavailable,
    fetch_rolling_icp_window,
    reconstruct_icp_window_from_doc,
    utc_day_start,
    utc_set_id_for_datetime,
)
from gateway.research_lab.logging_utils import (
    compact_ref,
    event_error_diagnostics as _event_error_diagnostics,
    format_worker_block,
    runtime_error_diagnostics as _runtime_error_diagnostics,
    safe_event_error_text as _safe_event_error_text,
)
from gateway.research_lab.maintenance import (
    get_scoring_maintenance_state,
    set_scoring_maintenance_paused,
)
from gateway.research_lab.model_authority_v2 import (
    AttestedPrivateModelRunnerV2,
    V2_PROVIDER_PROFILE_ENV,
    retry_attested_model_runner_v2,
)
from gateway.research_lab.provider_preflight import (
    PREFLIGHT_REASON_PREFIX,
    preflight_gate,
)
from gateway.research_lab.provider_profiles_v2 import (
    BENCHMARK_MODEL_PROFILE,
    load_provider_profile_v2,
    ProviderProfileV2Error,
    require_worker_proxy_profile_v2,
)
from gateway.research_lab.tee_protocol import legacy_v1_enabled
from gateway.research_lab.models import ResearchLabCandidateArtifactCreateRequest, ResearchLabScoreBundleCreateRequest
from gateway.tee.scoring_executor import configuration_hash as scoring_configuration_hash
from gateway.research_lab.promotion import (
    CONFIRMATION_ATTEMPT_FAILED_REASON,
    CONFIRMATION_CLOSED_REASON,
    CONFIRMATION_HOLD_REASON,
    CONFIRMATION_NON_CLOSING_STATUSES,
    CONFIRMATION_RESULT_REASON,
    CONFIRMATION_STARTED_REASON,
    ResearchLabPromotionController,
    candidate_already_promoted,
    confirmation_attempt_budget,
    load_active_private_model,
    load_confirmation_state,
    private_source_push_retry_seconds,
    private_repo_head_alignment_status,
    promotion_confirmation_rerun_enabled,
    promotion_improvement_metric,
    reconcile_failed_private_source_pushes,
    sync_active_model_to_repo_head,
)
from gateway.research_lab.public_activity import safe_project_public_loop_activity
from gateway.research_lab.public_benchmarks import (
    build_benchmark_visibility_split,
    build_public_benchmark_report,
    sanitize_benchmark_item_summary,
)
from gateway.research_lab.scoring_telemetry import (
    ScoringTelemetrySession,
    allocate_scoring_run,
    checkpoint_telemetry_index,
    emit_icp_event,
    emit_run_event,
    load_scoring_session,
    opaque_checkpoint_ref,
    private_baseline_benchmark_id,
    result_metrics,
    telemetry_enabled as scoring_telemetry_enabled,
)
from gateway.research_lab.trajectory_projector import execution_trace_id_for_node
from gateway.research_lab.store import (
    canonical_hash,
    create_candidate_artifact,
    create_candidate_evaluation_event,
    create_candidate_promotion_event,
    create_conditional_validation_event,
    create_private_model_benchmark_bundle,
    create_private_model_benchmark_event,
    create_public_benchmark_report,
    create_receipt_event,
    create_rolling_icp_window,
    create_score_bundle,
    create_scoring_category_result,
    create_scoring_dispatch_event,
    create_signed_audit_bundle,
    create_ticket_event,
    deterministic_uuid,
    insert_row,
    select_all,
    select_many,
    select_one,
)
from research_lab.canonical import canonical_json, sha256_json
from research_lab.code_editing import (
    CodeEditDraft,
    extract_unified_diff_paths,
)
from research_lab.eval import (
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
    SealedBenchmarkSet,
    evaluate_private_model_pair,
    ensure_private_model_outputs,
    private_model_env_passthrough,
    sign_digest_with_kms,
)
from research_lab.eval.baseline_summary import (
    artifact_serving_version_doc as shared_artifact_serving_version_doc,
    baseline_serving_model_version_doc as shared_baseline_serving_model_version_doc,
    build_baseline_health as shared_build_baseline_health,
    build_baseline_score_summary,
    daily_noise_budget_doc as shared_daily_noise_budget_doc,
    with_baseline_evaluation_contexts as shared_with_baseline_evaluation_contexts,
)
from research_lab.eval.miner_report_stats import build_icp_stats
from research_lab.eval.evaluator import (
    ConditionalValidationRetryableError,
    INCONTAINER_TRACE_KMS_KEY_ENV,
    INCONTAINER_TRACE_S3_PREFIX_ENV,
    QualificationStyleCompanyScorer,
    _benchmark_style_score as _queue_benchmark_style_score,
    _critical_measurement_failures,
    _upload_incontainer_trace as _upload_incontainer_trace_doc,
    _fp_penalty_points,
    _fp_unverified_primary_penalty_points,
    benchmark_icp_score_from_company_scores,
    fp_penalty_total_from_breakdowns,
    build_holdout_gate_result,
    build_score_bundle_from_scored_icps,
    score_private_model_pair_items,
)
from gateway.research_lab import global_icp_queue
from research_lab.eval.private_runtime import (
    PROVIDER_COST_EVALUATION_SCOPE_ENV,
    begin_incontainer_trace_collection,
    end_incontainer_trace_collection,
    incontainer_trace_capture_enabled,
)
from research_lab.observability.langfuse_client import observation, run_trace_id as langfuse_run_trace_id
from research_lab.eval.provider_costs import (
    cost_event_from_trace_entry,
    summarize_provider_cost_trace_entries,
)
from research_lab.eval.promotion_metric import (
    PAIRED_LCB_PROMOTION_METRIC_VERSION,
    preliminary_promotion_gate_projection,
    promotion_gate_decision,
)
from research_lab.observability.redaction import miner_hotkey_hash
from research_lab.observability.tracing import finish_score_bundle_observation


logger = logging.getLogger(__name__)

STALE_PARENT_REBASE_REPAIR_MAX_TOKENS = 32_768
STALE_PARENT_REBASE_REPAIR_REASONING_BODY = {"enabled": True, "effort": "max"}
STALE_PARENT_REBASE_REPAIR_EMPTY_CONTENT_ATTEMPTS = 2
PRIVATE_BASELINE_FAST_EMPTY_ABORT_AFTER = 6
PRIVATE_BASELINE_FAST_EMPTY_ABORT_SECONDS = 90.0
_POSTGREST_TIMESTAMP_RE = re.compile(
    r"^(?P<prefix>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
    r"\.(?P<fraction>\d{1,9})(?P<suffix>Z|[+-]\d{2}(?::?\d{2})?)?$"
)


def _attested_score_bundle_evidence_roots(
    *,
    artifact: PrivateModelArtifactManifest,
    candidate_artifact: PrivateModelArtifactManifest,
    per_icp_results: Any,
    evidence_bundle_refs: Any,
    private_holdout_gate: Mapping[str, Any],
) -> dict[str, str]:
    roots = {
        "parent_model_manifest": str(artifact.manifest_hash),
        "candidate_model_manifest": str(candidate_artifact.manifest_hash),
        "model_outputs": attested_sha256_bytes(attested_canonical_json_bytes(per_icp_results)),
        "provider_evidence_refs": attested_sha256_bytes(
            attested_canonical_json_bytes(list(evidence_bundle_refs or ()))
        ),
    }
    image_digest = str(candidate_artifact.image_digest or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        roots["candidate_model_image"] = image_digest
    baseline_hash = str(
        private_holdout_gate.get("baseline_benchmark_hash")
        if isinstance(private_holdout_gate, Mapping)
        else ""
    ).lower()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", baseline_hash):
        roots["baseline_score_summary"] = baseline_hash
    return roots


async def _compare_candidate_score_bundle_in_enclave(
    *,
    evaluation_epoch: int,
    artifact: PrivateModelArtifactManifest,
    benchmark: SealedBenchmarkSet,
    patch: Mapping[str, Any],
    candidate_artifact: PrivateModelArtifactManifest,
    per_icp_results: Any,
    run_context: Mapping[str, Any],
    policy: Mapping[str, Any],
    private_holdout_gate: Mapping[str, Any],
    expected_score_bundle: Mapping[str, Any],
    parent_receipts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_parents: dict[str, dict[str, Any]] = {}
    direct_parent_hashes: set[str] = set()
    for receipt in parent_receipts or []:
        if not isinstance(receipt, Mapping):
            continue
        receipt_hash = str(receipt.get("receipt_hash") or "")
        if receipt_hash:
            normalized_parents[receipt_hash] = dict(receipt)
            direct_parent_hashes.add(receipt_hash)
    baseline_bundle_id = str(
        private_holdout_gate.get("baseline_benchmark_bundle_id") or ""
    )
    baseline_summary_hash = str(
        private_holdout_gate.get("baseline_benchmark_hash") or ""
    ).lower()
    if baseline_bundle_id and re.fullmatch(r"sha256:[0-9a-f]{64}", baseline_summary_hash):
        baseline_receipt, baseline_lineage = await resolve_attested_artifact_lineage(
            artifact_kind="benchmark_score_summary",
            artifact_ref=baseline_bundle_id,
            artifact_hash=baseline_summary_hash,
        )
        if baseline_receipt is not None:
            direct_parent_hashes.add(str(baseline_receipt["receipt_hash"]))
            for receipt in baseline_lineage:
                normalized_parents[str(receipt["receipt_hash"])] = dict(receipt)
    preliminary_proof = private_holdout_gate.get("preliminary_promotion_gate")
    if isinstance(preliminary_proof, Mapping):
        preliminary_bundle_hash = str(
            preliminary_proof.get("preliminary_score_bundle_hash") or ""
        ).lower()
        preliminary_receipt_hash = str(
            preliminary_proof.get("promotion_decision_receipt_hash") or ""
        ).lower()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", preliminary_bundle_hash):
            raise RuntimeError("conditional preliminary score-bundle hash is invalid")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", preliminary_receipt_hash):
            raise RuntimeError("conditional preliminary decision receipt is invalid")
        preliminary_receipt, preliminary_lineage = await resolve_attested_artifact_lineage(
            artifact_kind="promotion_decision",
            artifact_ref="score_bundle:" + preliminary_bundle_hash.split(":", 1)[1],
            artifact_hash=preliminary_bundle_hash,
        )
        if (
            preliminary_receipt is None
            or str(preliminary_receipt.get("receipt_hash") or "").lower()
            != preliminary_receipt_hash
        ):
            raise RuntimeError(
                "conditional preliminary decision lineage differs from the frozen proof"
            )
        direct_parent_hashes.add(preliminary_receipt_hash)
        for receipt in preliminary_lineage:
            normalized_parents[str(receipt["receipt_hash"])] = dict(receipt)
    return await compare_attested_score_bundle(
        epoch_id=int(evaluation_epoch),
        purpose="research_lab.candidate_score.v1",
        build_payload={
            "artifact_manifest": artifact.to_dict(),
            "benchmark": benchmark.to_dict(),
            "patch_manifest": dict(patch),
            "candidate_artifact_manifest": candidate_artifact.to_dict(),
            "per_icp_results": list(per_icp_results or ()),
            "run_context": dict(run_context),
            "policy": dict(policy),
            "extra_bundle_fields": {"private_holdout_gate": dict(private_holdout_gate)},
        },
        expected_score_bundle=expected_score_bundle,
        parent_receipts=list(normalized_parents.values()),
        direct_parent_receipt_hashes=sorted(direct_parent_hashes),
        evidence_roots=_attested_score_bundle_evidence_roots(
            artifact=artifact,
            candidate_artifact=candidate_artifact,
            per_icp_results=per_icp_results,
            evidence_bundle_refs=run_context.get("evidence_bundle_refs"),
            private_holdout_gate=private_holdout_gate,
        ),
    )


def _attested_execution_receipt(outcome: Mapping[str, Any], label: str) -> dict[str, Any]:
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ConditionalValidationRetryableError(
            f"conditional_preliminary_{label}_receipt_missing"
        )
    normalized = dict(receipt)
    for field in ("receipt_hash", "output_root"):
        value = str(normalized.get(field) or "").lower()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ConditionalValidationRetryableError(
                f"conditional_preliminary_{label}_{field}_invalid"
            )
        normalized[field] = value
    return normalized


async def _authorize_conditional_preliminary_gate(
    *,
    config: ResearchLabGatewayConfig,
    evaluation_epoch: int,
    candidate: Mapping[str, Any],
    artifact: PrivateModelArtifactManifest,
    benchmark: SealedBenchmarkSet,
    patch: Mapping[str, Any],
    candidate_artifact: PrivateModelArtifactManifest,
    preliminary_results: list[Mapping[str, Any]],
    run_context: Mapping[str, Any],
    policy: Mapping[str, Any],
    preliminary_gate: Mapping[str, Any],
    parent_receipts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attest the unchanged 20-ICP promotion gate before conditional work."""

    if len(preliminary_results) != int(preliminary_gate.get("public_icp_count") or 0) + int(
        preliminary_gate.get("private_holdout_icp_count") or 0
    ):
        raise ConditionalValidationRetryableError(
            "conditional_preliminary_result_count_mismatch"
        )
    projected_gate = preliminary_promotion_gate_projection(preliminary_gate)
    provisional_bundle = build_score_bundle_from_scored_icps(
        artifact_manifest=artifact,
        benchmark=benchmark,
        patch_manifest=patch,
        candidate_artifact_manifest=candidate_artifact.to_dict(),
        per_icp_results=preliminary_results,
        run_context=run_context,
        policy=policy,
        extra_bundle_fields={"private_holdout_gate": projected_gate},
    )
    bundle_outcome = await _compare_candidate_score_bundle_in_enclave(
        evaluation_epoch=int(evaluation_epoch),
        artifact=artifact,
        benchmark=benchmark,
        patch=patch,
        candidate_artifact=candidate_artifact,
        per_icp_results=preliminary_results,
        run_context=run_context,
        policy=policy,
        private_holdout_gate=projected_gate,
        expected_score_bundle=provisional_bundle,
        parent_receipts=parent_receipts,
    )
    metric = promotion_improvement_metric(provisional_bundle)
    metric_outcome = await compare_attested_promotion_metric(
        epoch_id=int(evaluation_epoch),
        score_bundle=provisional_bundle,
        expected_improvement_points=float(metric.improvement_points),
        expected_event_doc=metric.event_doc(),
    )

    active = await load_active_private_model(config, register_bootstrap=True)
    candidate_parent = str(
        candidate.get("parent_artifact_hash")
        or provisional_bundle.get("parent_artifact_hash")
        or artifact.model_artifact_hash
    )
    active_parent = str(active.artifact.model_artifact_hash)
    decision = promotion_gate_decision(
        provisional_bundle,
        candidate_kind=str(candidate.get("candidate_kind") or "patch"),
        candidate_parent=candidate_parent,
        active_parent=active_parent,
        threshold_points=float(config.improvement_threshold_points),
        auto_promotion_enabled=bool(config.auto_promotion_enabled),
    )
    decision_outcome = await compare_attested_promotion_gate_decision(
        epoch_id=int(evaluation_epoch),
        score_bundle=provisional_bundle,
        decision_payload={
            "candidate_kind": decision.candidate_kind,
            "candidate_parent": candidate_parent,
            "active_parent": active_parent,
            "threshold_points": decision.threshold_points,
            "auto_promotion_enabled": decision.auto_promotion_enabled,
        },
        expected_decision=decision.to_dict(),
        metric_outcome=metric_outcome,
    )
    if decision.status == "stale_parent_needs_rescore":
        raise StaleParentDuringScoring(
            active_artifact=active.artifact,
            candidate_parent=candidate_parent,
            progress={
                "phase": "conditional_preliminary_gate",
                "completed_icp_count": len(preliminary_results),
            },
        )
    if decision.status != "promotion_passed":
        raise ConditionalValidationRetryableError(
            "conditional_preliminary_attested_gate_not_passed:" + decision.status
        )

    bundle_receipt = _attested_execution_receipt(bundle_outcome, "score_bundle")
    metric_receipt = _attested_execution_receipt(metric_outcome, "metric")
    decision_receipt = _attested_execution_receipt(decision_outcome, "decision")
    proof = {
        "schema_version": "research_lab_preliminary_promotion_gate.v1",
        "status": decision.status,
        "preliminary_score_bundle_hash": str(
            provisional_bundle.get("score_bundle_hash") or ""
        ).lower(),
        "score_bundle_receipt_hash": bundle_receipt["receipt_hash"],
        "promotion_metric_receipt_hash": metric_receipt["receipt_hash"],
        "promotion_decision_receipt_hash": decision_receipt["receipt_hash"],
        "promotion_decision_output_root": decision_receipt["output_root"],
        "candidate_artifact_hash": str(candidate_artifact.model_artifact_hash),
        "candidate_parent_artifact_hash": candidate_parent,
        "active_parent_artifact_hash": active_parent,
        "rolling_window_hash": str(run_context.get("rolling_window_hash") or ""),
        "category_assignment_hash": str(
            preliminary_gate.get("category_assignment_hash") or ""
        ),
        "conditional_validation_policy_hash": str(
            preliminary_gate.get("conditional_validation_policy_hash") or ""
        ),
        "scoring_configuration_hash": scoring_configuration_hash(),
        "threshold_points": float(decision.threshold_points),
        "decision": decision.to_dict(),
    }
    return {**proof, "proof_hash": canonical_hash(proof)}


@functools.lru_cache(maxsize=1)
def _scoring_worker_source_hash() -> str:
    """Hash the executing worker source for controlled publication retries."""

    try:
        source = Path(__file__).read_bytes()
    except OSError as exc:
        logger.warning(
            "research_lab_scoring_worker_source_hash_fallback path=%s error=%s",
            str(__file__),
            _short_error(exc),
        )
        return sha256_json({"module": __name__, "source_available": False})
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _baseline_publication_retry_token_hash() -> str:
    token = str(os.getenv("RESEARCH_LAB_BASELINE_PUBLICATION_RETRY_TOKEN", "") or "").strip()
    return sha256_json({"retry_token": token}) if token else ""


def _latest_terminal_baseline_publication_failure(
    rows: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Return a terminal publication failure only when it is the latest event."""

    if not rows:
        return None
    latest = rows[0]
    doc = latest.get("event_doc") if isinstance(latest.get("event_doc"), Mapping) else {}
    if str(latest.get("dispatch_status") or "") != "failed":
        return None
    if str(doc.get("failure_phase") or "") != "publication":
        return None
    if not bool(doc.get("terminal_no_automatic_retry")):
        return None
    return latest


def _baseline_publication_retry_authorization(failure_row: Mapping[str, Any]) -> str:
    """Authorize one retry only after a worker change or a new operator token."""

    doc = failure_row.get("event_doc") if isinstance(failure_row.get("event_doc"), Mapping) else {}
    failed_source_hash = str(doc.get("scoring_worker_source_hash") or "")
    current_source_hash = _scoring_worker_source_hash()
    if failed_source_hash and failed_source_hash != current_source_hash:
        return "scoring_worker_source_changed"
    failed_token_hash = str(doc.get("publication_retry_token_hash") or "")
    current_token_hash = _baseline_publication_retry_token_hash()
    if current_token_hash and current_token_hash != failed_token_hash:
        return "operator_retry_token_changed"
    return ""


def _baseline_publication_retry_decision(
    rows: list[Mapping[str, Any]],
    *,
    scope_key: str,
    in_process_failures: set[str],
) -> tuple[bool, str]:
    """Return (blocked, authorization_reason) for the next baseline attempt."""

    if scope_key in in_process_failures:
        return True, ""
    failure = _latest_terminal_baseline_publication_failure(rows)
    if failure is None:
        return False, ""
    authorization = _baseline_publication_retry_authorization(failure)
    return (not bool(authorization)), authorization


def _artifact_manifest_mapping(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, Mapping):
        return dict(artifact)
    to_dict = getattr(artifact, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {
        field: getattr(artifact, field, None)
        for field in (
            "model_artifact_hash",
            "manifest_hash",
            "manifest_uri",
            "git_commit_sha",
            "image_digest",
            "config_hash",
            "component_registry_version",
            "scoring_adapter_version",
            "build_id",
        )
    }


def _artifact_serving_version_doc(artifact: PrivateModelArtifactManifest) -> dict[str, Any]:
    return shared_artifact_serving_version_doc(_artifact_manifest_mapping(artifact))


def _baseline_serving_model_version_doc(
    *,
    artifact: PrivateModelArtifactManifest,
    benchmark_date: str,
    benchmark_attempt: int,
    rolling_window_hash: str,
    evaluation_epoch: int,
) -> dict[str, Any]:
    return shared_baseline_serving_model_version_doc(
        artifact=_artifact_manifest_mapping(artifact),
        benchmark_date=benchmark_date,
        benchmark_attempt=benchmark_attempt,
        rolling_window_hash=rolling_window_hash,
        evaluation_epoch=evaluation_epoch,
    )


def _public_serving_model_version_doc(serving_doc: Mapping[str, Any]) -> dict[str, Any]:
    parent = serving_doc.get("parent_model") if isinstance(serving_doc.get("parent_model"), Mapping) else {}
    doc = {
        "schema_version": str(serving_doc.get("schema_version") or ""),
        "result_role": str(serving_doc.get("result_role") or ""),
        "run_id": str(serving_doc.get("run_id") or ""),
        "evaluation_epoch": int(serving_doc.get("evaluation_epoch") or 0),
        "benchmark_date": str(serving_doc.get("benchmark_date") or ""),
        "benchmark_attempt": serving_doc.get("benchmark_attempt"),
        "run_scope": str(serving_doc.get("run_scope") or ""),
        "icp_set_hash": str(serving_doc.get("icp_set_hash") or ""),
        "scoring_code_version": str(serving_doc.get("scoring_code_version") or ""),
        "evaluator_version": str(serving_doc.get("evaluator_version") or ""),
        "model_artifact_hash": str(parent.get("model_artifact_hash") or ""),
        "manifest_hash": str(parent.get("manifest_hash") or ""),
        "git_commit_sha": str(parent.get("git_commit_sha") or ""),
        "version_stamp_hash": str(serving_doc.get("version_stamp_hash") or ""),
    }
    doc["public_stamp_hash"] = sha256_json({key: value for key, value in doc.items() if key != "public_stamp_hash"})
    return doc


def _event_serving_model_version(score_bundle: Any) -> dict[str, Any]:
    """Return the public, hash-only serving stamp for an event document."""
    if not isinstance(score_bundle, Mapping):
        return _public_serving_model_version_doc({})
    serving_doc = score_bundle.get("serving_model_version")
    return _public_serving_model_version_doc(
        serving_doc if isinstance(serving_doc, Mapping) else {}
    )


def _with_baseline_evaluation_contexts(
    per_icp_summaries: list[dict[str, Any]],
    *,
    benchmark_date: str,
    benchmark_attempt: int,
    rolling_window_hash: str,
    evaluation_epoch: int,
    serving_model_version_hash: str,
) -> list[dict[str, Any]]:
    return shared_with_baseline_evaluation_contexts(
        per_icp_summaries,
        benchmark_date=benchmark_date,
        benchmark_attempt=benchmark_attempt,
        rolling_window_hash=rolling_window_hash,
        evaluation_epoch=evaluation_epoch,
        serving_model_version_hash=serving_model_version_hash,
    )


def _daily_noise_budget_doc(
    *,
    benchmark_date: str,
    rolling_window_hash: str,
    per_icp_summaries: list[Mapping[str, Any]],
    aggregate_score: float,
) -> dict[str, Any]:
    return shared_daily_noise_budget_doc(
        benchmark_date=benchmark_date,
        rolling_window_hash=rolling_window_hash,
        per_icp_summaries=per_icp_summaries,
        aggregate_score=aggregate_score,
    )


def _retry_runner_with_provider_cost_scope(
    runner: AttestedPrivateModelRunnerV2,
    *,
    retry_round: int,
) -> AttestedPrivateModelRunnerV2:
    """Clone a retry runner with a fresh provider-cost budget scope.

    The Docker runtime hashes this evaluation scope together with the actual
    ICP payload, so every ICP still gets an independent ledger. The extra retry
    seed prevents an in-attempt retry from inheriting the first-pass spend.
    """

    extra_env = dict(runner.spec.extra_env or {})
    base_scope = str(extra_env.get(PROVIDER_COST_EVALUATION_SCOPE_ENV) or "").strip()
    retry_scope = sha256_json(
        {
            "schema_version": "research_lab_provider_cost_retry_scope.v1",
            "base_scope": base_scope,
            "retry_round": int(retry_round),
            "retry_round_started_at_ms": int(time.time() * 1000),
        }
    )
    extra_env[PROVIDER_COST_EVALUATION_SCOPE_ENV] = retry_scope
    return retry_attested_model_runner_v2(
        runner,
        extra_env=extra_env,
    )


def _attested_receipts_from(*sources: Any) -> list[dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for source in sources:
        supplier = getattr(source, "attested_receipts", None)
        if not callable(supplier):
            continue
        for receipt in supplier():
            if not isinstance(receipt, Mapping):
                continue
            receipt_hash = str(receipt.get("receipt_hash") or "")
            if receipt_hash:
                receipts[receipt_hash] = dict(receipt)
    return [receipts[key] for key in sorted(receipts)]


def _attested_outcome_count(source: Any) -> int | None:
    supplier = getattr(source, "attested_outcome_count", None)
    if not callable(supplier):
        return None
    try:
        value = int(supplier())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _last_attested_receipt_hash(source: Any) -> str:
    supplier = getattr(source, "last_attested_receipt_hash", None)
    if not callable(supplier):
        return ""
    value = str(supplier() or "").lower()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        return value
    return ""


def _queue_current_attested_receipt_hashes(
    *,
    scorer: Any,
    scorer_outcome_count_before: int | None,
    receipt_hashes_before: set[str],
    receipt_hashes_after: set[str],
) -> list[str]:
    """Bind one queue result to the measured call that produced it.

    Receipt hashes are content-addressed and may repeat on a deterministic
    retry. The monotonic call count distinguishes that valid retry from stale
    receipts accumulated by earlier ICPs.
    """

    current_hashes = receipt_hashes_after - receipt_hashes_before
    scorer_outcome_count_after = _attested_outcome_count(scorer)
    if scorer_outcome_count_before is not None:
        if (
            scorer_outcome_count_after is None
            or scorer_outcome_count_after <= scorer_outcome_count_before
        ):
            raise ConditionalValidationRetryableError(
                "conditional_queue_attested_receipt_missing"
            )
        latest_scorer_receipt = _last_attested_receipt_hash(scorer)
        if not latest_scorer_receipt:
            raise ConditionalValidationRetryableError(
                "conditional_queue_attested_receipt_missing"
            )
        current_hashes.add(latest_scorer_receipt)
    if not current_hashes:
        raise ConditionalValidationRetryableError(
            "conditional_queue_attested_receipt_missing"
        )
    return sorted(current_hashes)


def _queue_attested_parent_receipts(
    docs: Mapping[str, Any],
    *live_sources: Any,
) -> list[dict[str, Any]]:
    receipts = {
        str(item["receipt_hash"]): dict(item)
        for item in _attested_receipts_from(*live_sources)
        if item.get("receipt_hash")
    }
    persisted = docs.get("attested_receipt_hashes") or []
    if not isinstance(persisted, list):
        raise ConditionalValidationRetryableError(
            "conditional_queue_receipt_sidecar_invalid"
        )
    for value in persisted:
        receipt_hash = str(value or "").lower()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt_hash):
            raise ConditionalValidationRetryableError(
                "conditional_queue_receipt_sidecar_invalid"
            )
        receipts.setdefault(receipt_hash, {"receipt_hash": receipt_hash})
    return [receipts[key] for key in sorted(receipts)]


def _queue_job_error_is_retryable(
    job: Mapping[str, Any],
    error: BaseException,
) -> bool:
    if (
        isinstance(error, ConditionalValidationRetryableError)
        and "queue_attested_receipt_missing" in str(error)
    ):
        return True
    if str(job.get("phase") or "") != "conditional":
        return False
    return bool(_candidate_scoring_failure_class(error)[1])


def _queue_scoring_item(
    ctx: Mapping[str, Any] | None,
    job: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Resolve one queued item, holding conditional work on missing state."""

    phase = str(job.get("phase") or "")
    if ctx is None:
        if phase == "conditional":
            raise ConditionalValidationRetryableError(
                "conditional_queue_scoring_context_missing"
            )
        return None
    items_by_ref = ctx.get("items_by_ref")
    item = (
        items_by_ref.get(str(job.get("icp_ref") or ""))
        if isinstance(items_by_ref, Mapping)
        else None
    )
    if item is None and phase == "conditional":
        raise ConditionalValidationRetryableError(
            "conditional_queue_scoring_item_missing"
        )
    return item if isinstance(item, Mapping) else None


async def _attested_model_parent_graphs(
    *,
    model_kind: str,
    artifact: PrivateModelArtifactManifest,
    candidate_id: str = "",
    epoch_id: int = 0,
) -> tuple[dict[str, Any], ...]:
    if legacy_v1_enabled():
        return ()
    if model_kind == "private":
        from gateway.research_lab.active_model_authority_v2 import (
            attest_active_private_model_v2,
        )

        authority = await attest_active_private_model_v2(
            artifact=artifact,
            epoch_id=int(epoch_id),
        )
        graph = authority.get("receipt_graph")
        if not isinstance(graph, Mapping):
            raise RuntimeError("active private model V2 receipt graph is missing")
        return (dict(graph),)
    if model_kind != "candidate" or not candidate_id:
        raise RuntimeError("candidate model V2 lineage identity is missing")
    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graph_v2,
    )

    graph = await load_business_artifact_graph_v2(
        artifact_kind="candidate_model",
        artifact_ref=str(candidate_id),
        artifact_hash=str(artifact.manifest_hash),
    )
    return (dict(graph),)


class StaleParentDuringScoring(RuntimeError):
    """Raised at an ICP boundary when a candidate's parent is no longer current."""

    def __init__(
        self,
        *,
        active_artifact: PrivateModelArtifactManifest,
        candidate_parent: str,
        progress: Mapping[str, Any],
    ) -> None:
        self.active_artifact = active_artifact
        self.candidate_parent = candidate_parent
        self.progress = dict(progress)
        completed = int(self.progress.get("completed_icp_count") or 0)
        super().__init__(
            "candidate parent changed during scoring: "
            f"candidate_parent={compact_ref(candidate_parent)} "
            f"active_parent={compact_ref(active_artifact.model_artifact_hash)} "
            f"completed_icps={completed}"
        )


def _idle_log_seconds() -> float:
    try:
        return max(10.0, float(os.getenv("RESEARCH_LAB_WORKER_IDLE_LOG_SECONDS", "60")))
    except ValueError:
        return 60.0


def _error_backoff_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("RESEARCH_LAB_WORKER_ERROR_BACKOFF_SECONDS", "60")))
    except ValueError:
        return 60.0


def _short_error(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:300]}"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return int(default)


def _worker_recycle_rss_mb() -> int:
    return _env_int("RESEARCH_LAB_SCORING_WORKER_RECYCLE_RSS_MB", 3072)


def _worker_recycle_max_jobs() -> int:
    return _env_int("RESEARCH_LAB_SCORING_WORKER_RECYCLE_JOBS", 16)


def _read_own_rss_mb(status_path: str = "/proc/self/status") -> int | None:
    """Current process resident set size in MB (None off-Linux/on failure)."""
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) < 2:
                        return None
                    return int(parts[1]) // 1024
    except Exception:
        return None
    return None


# Event-doc error sanitizers (bug #36) moved to logging_utils so the loop
# engine shares them; imported at the top with underscore aliases to keep
# existing call sites and tests stable.

_PROVIDER_429_RETRY_BACKOFF_SECONDS = 15.0
_PROVIDER_COST_CAP_ERROR_MARKERS = (
    "research_lab_provider_cost_cap_exceeded",
    "cost_cap_reached",
    "provider_cost_cap_blocked",
    "provider cost cap",
    "budget_soft_stop",
)


def _provider_cost_cap_error_text(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    if any(marker in lowered for marker in _PROVIDER_COST_CAP_ERROR_MARKERS):
        return True
    diagnostics = _runtime_error_diagnostics(error_text)
    try:
        return int(diagnostics.get("status") or 0) == 402
    except (TypeError, ValueError):
        return False


def _baseline_error_is_retryable(error_text: str) -> bool:
    """Transient provider/infra failures are retryable; model code bugs and
    auth/request errors are not.

    Must receive the full exception text: ``_short_error`` truncates to 300
    chars and can drop the status marker. Note ``_runtime_error_diagnostics``
    buckets 429 into ``provider_http_4xx`` — a plain 4xx check would mark the
    rate-limit error this classifier exists to catch as permanent, so 429 is
    matched explicitly before the 4xx rejection branch.
    """
    lowered = error_text.lower()
    diagnostics = _runtime_error_diagnostics(error_text)
    status = int(diagnostics.get("status") or 0)
    provider = str(diagnostics.get("provider") or "unknown")
    if _provider_cost_cap_error_text(error_text):
        return False
    if status in (408, 429) or status >= 500:
        return True
    # Scrapingdog's 400 "Something went wrong or profile not found" is
    # transient/data-shaped, not auth or request validation, and can recover on
    # retry. Production evidence showed 410 Gone retries did not produce usable
    # content, so 410 is treated as a terminal provider/data miss.
    if status == 400 and (provider == "scrapingdog" or "something went wrong" in lowered):
        return True
    if status in (400, 401, 402, 403, 404, 409, 410):
        return False
    if any(
        marker in lowered
        for marker in (
            "too many requests",
            "rate limit",
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
        )
    ):
        return True
    if any(
        marker in lowered
        for marker in (
            "exit status 137",
            "killed",
            "docker daemon",
            "no space left on device",
        )
    ):
        # Infra pressure (OOM-kill, daemon wedge) clears when the retry rounds
        # run fewer containers at once.
        return True
    return False


def _baseline_429_retry_backoff_seconds(error_text: str) -> float:
    """Return a short backoff only for explicit provider HTTP 429 errors."""

    diagnostics = _runtime_error_diagnostics(error_text)
    try:
        status = int(diagnostics.get("status") or 0)
    except (TypeError, ValueError):
        status = 0
    if status == 429:
        return _PROVIDER_429_RETRY_BACKOFF_SECONDS
    return 0.0


class BaselineHealthGateFailure(RuntimeError):
    """Raised when an explicitly configured baseline quality guard blocks a run."""

    def __init__(self, message: str, *, baseline_health: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.baseline_health = dict(baseline_health)


class ConfirmationMeasurementUnhealthy(RuntimeError):
    """Raised when a §5.2-2 confirmation measurement is too degraded to decide
    on (too many unresolved provider errors on either side). The attempt is
    recorded as failed and the candidate stays held, bounded by the
    claim-attempt budget — a flaky confirmation must neither reject a good
    candidate nor confirm a phantom improvement."""


# --- §0-N6: scorer-side burst isolation for the parallel baseline ----------
# Only Exa is container-isolated today; the qualification scoring calls
# (score_with_breakdowns — OpenRouter/Scrapingdog) burst N-wide on the prod
# keys during a parallel baseline. Opt-in dedicated benchmark scorer keys,
# applied ONLY within a baseline-batch scoring scope and falling back to prod
# values when unset — mirroring how the benchmark Exa key is plumbed.


def _benchmark_scorer_max_concurrency() -> int:
    """Optional cap on concurrent scorer calls inside a baseline batch.
    0 (default) = unlimited (scorer calls fan out at batch concurrency)."""
    try:
        return max(0, int(os.getenv("RESEARCH_LAB_BENCHMARK_SCORER_MAX_CONCURRENCY", "0")))
    except ValueError:
        return 0


def _provider_profile_has_override(profile: str, provider_id: str) -> bool:
    """Report encrypted profile presence without reading plaintext credentials."""

    if legacy_v1_enabled():
        legacy_env_names = {
            "exa": "RESEARCH_LAB_BENCHMARK_EXA_API_KEY",
            "openrouter": "RESEARCH_LAB_BENCHMARK_OPENROUTER_API_KEY",
            "scrapingdog": "RESEARCH_LAB_BENCHMARK_SCRAPINGDOG_API_KEY",
        }
        env_name = legacy_env_names.get(str(provider_id))
        return bool(env_name and os.getenv(env_name, "").strip())
    document = load_provider_profile_v2(profile)
    return str(provider_id) in dict(document["credential_ref_hashes"])


_SCORER_KEY_MODULE_ATTRS: tuple[tuple[str, str, str], ...] = (
    ("gateway.qualification.utils.helpers", "OPENROUTER_API_KEY", "openrouter"),
    ("qualification.scoring.verification_helpers", "OPENROUTER_API_KEY", "openrouter"),
    ("qualification.scoring.verification_helpers", "SCRAPINGDOG_API_KEY", "scrapingdog"),
)


@contextlib.contextmanager
def _benchmark_scorer_isolation():
    """Select benchmark scorer credentials for the active protocol.

    V2 leases its encrypted profile inside the measured runtime. Legacy mode
    temporarily applies the already-supported benchmark-specific host keys to
    the dedicated scoring worker and restores every value afterward.
    """

    if not legacy_v1_enabled():
        yield
        return
    scrapingdog = os.getenv(
        "RESEARCH_LAB_BENCHMARK_SCRAPINGDOG_API_KEY", ""
    ).strip()
    openrouter = os.getenv(
        "RESEARCH_LAB_BENCHMARK_OPENROUTER_API_KEY", ""
    ).strip()
    if not scrapingdog and not openrouter:
        yield
        return
    env_overrides: dict[str, str] = {}
    if scrapingdog:
        env_overrides["SCRAPINGDOG_API_KEY"] = scrapingdog
        env_overrides["QUALIFICATION_SCRAPINGDOG_API_KEY"] = scrapingdog
    if openrouter:
        env_overrides["QUALIFICATION_OPENROUTER_API_KEY"] = openrouter
    saved_env: dict[str, str | None] = {}
    saved_attrs: list[tuple[Any, str, Any]] = []
    try:
        for name, value in env_overrides.items():
            saved_env[name] = os.environ.get(name)
            os.environ[name] = value
        for module_name, attr, provider in _SCORER_KEY_MODULE_ATTRS:
            value = scrapingdog if provider == "scrapingdog" else openrouter
            if not value:
                continue
            try:
                module = importlib.import_module(module_name)
            except Exception:  # pragma: no cover - optional scorer stack
                continue
            if hasattr(module, attr):
                saved_attrs.append((module, attr, getattr(module, attr)))
                setattr(module, attr, value)
        yield
    finally:
        for module, attr, previous in saved_attrs:
            try:
                setattr(module, attr, previous)
            except Exception:  # pragma: no cover - best-effort restoration
                logger.warning(
                    "research_lab_benchmark_scorer_attr_restore_failed attr=%s",
                    attr,
                )
        for name, previous in saved_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _confirmation_lease_seconds() -> int:
    """Staleness window for a confirmation_rerun_started claim: a dead worker's
    in-flight confirmation becomes reclaimable after this many seconds."""
    try:
        return max(600, int(os.getenv("RESEARCH_LAB_CONFIRMATION_LEASE_SECONDS", "7200")))
    except ValueError:
        return 7200


def _collect_confirmation_scores(
    summaries: list[dict[str, Any]],
) -> tuple[dict[str, float], list[str], int]:
    """(icp_ref -> score, unresolved-error refs, non-empty output count) from a
    baseline-batch result. Reads the underscore orchestration fields without
    mutating the summaries (confirmation summaries are never stored raw)."""
    scores: dict[str, float] = {}
    unresolved: list[str] = []
    nonempty = 0
    for summary in summaries:
        ref = str(summary.get("icp_ref") or "")
        if summary.get("_nonempty"):
            nonempty += 1
        if not ref:
            continue
        scores[ref] = _safe_float(summary.get("score"), default=0.0)
        if summary.get("_runtime_error"):
            unresolved.append(ref)
    return scores, unresolved, nonempty


def _per_icp_checkpoint_enabled() -> bool:
    """Bug #31: persist per-ICP candidate results so a requeue/rescore resumes
    instead of re-running the whole multi-hour evaluation."""
    return os.getenv(
        "RESEARCH_LAB_SCORING_PER_ICP_CHECKPOINT", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _scoring_progress_s3_prefix(manifest_uri: str, candidate_id: str) -> tuple[str, str] | None:
    uri = str(manifest_uri or "")
    if not uri.startswith("s3://"):
        return None
    rest = uri[5:]
    bucket, sep, key = rest.partition("/")
    if not bucket or not sep or not key:
        return None
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    safe_candidate = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(candidate_id or "candidate")
    )[:96]
    return bucket, f"{base_prefix}/candidates/{safe_candidate}/scoring-progress/"


def _scoring_progress_s3_location(
    manifest_uri: str, candidate_id: str, window_hash: str
) -> tuple[str, str] | None:
    prefix = _scoring_progress_s3_prefix(manifest_uri, candidate_id)
    if prefix is None:
        return None
    bucket, object_prefix = prefix
    window_tag = str(window_hash or "").removeprefix("sha256:")[:16] or "window"
    return bucket, f"{object_prefix}{window_tag}.json"


def _completed_icp_count_from_progress_doc(doc: Mapping[str, Any] | None) -> int:
    if not isinstance(doc, Mapping):
        return 0
    candidates: list[Any] = [
        doc.get("completed_icp_count"),
        doc.get("completed_icps"),
    ]
    for nested_key in ("scoring_progress", "stale_progress", "progress"):
        nested = doc.get(nested_key)
        if isinstance(nested, Mapping):
            candidates.extend(
                [
                    nested.get("completed_icp_count"),
                    nested.get("completed_icps"),
                ]
            )
    for value in candidates:
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    rows = doc.get("per_icp_results")
    return len(rows) if isinstance(rows, list) else 0


def _safe_scoring_progress_summary(
    *,
    source: str,
    completed_icp_count: int,
    rolling_window_hash: str = "",
) -> dict[str, Any]:
    count = max(0, int(completed_icp_count or 0))
    summary: dict[str, Any] = {
        "source": str(source or "unknown")[:80],
        "completed_icp_count": count,
        "checkpoint_found": count > 0,
    }
    if rolling_window_hash:
        summary["rolling_window_hash"] = str(rolling_window_hash)
    return summary


def _latest_scoring_progress_from_events(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = _safe_scoring_progress_summary(
        source="candidate_events",
        completed_icp_count=0,
    )
    for row in rows:
        doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
        completed = _completed_icp_count_from_progress_doc(doc)
        if completed <= best["completed_icp_count"]:
            continue
        rolling_window_hash = ""
        if isinstance(doc, Mapping):
            progress_doc = doc.get("scoring_progress")
            rolling_window_hash = str(doc.get("rolling_window_hash") or "")
            if not rolling_window_hash and isinstance(progress_doc, Mapping):
                rolling_window_hash = str(progress_doc.get("rolling_window_hash") or "")
        best = _safe_scoring_progress_summary(
            source="candidate_events",
            completed_icp_count=completed,
            rolling_window_hash=rolling_window_hash,
        )
    return best


def _load_scoring_progress(
    bucket: str,
    object_key: str,
    *,
    window_hash: str,
    candidate_artifact_hash: str,
    commitment_hash: str = "",
) -> list[dict[str, Any]]:
    import boto3  # type: ignore

    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=object_key)["Body"].read()
        doc = json.loads(body.decode("utf-8"))
    except Exception as exc:
        logger.warning(
            "research_lab_scoring_progress_load_failed bucket=%s key=%s error=%s",
            str(bucket)[:120],
            str(object_key)[:240],
            _short_error(exc),
        )
        return []
    if not isinstance(doc, Mapping):
        return []
    if str(doc.get("rolling_window_hash") or "") != str(window_hash):
        return []
    if str(doc.get("candidate_artifact_hash") or "") != str(candidate_artifact_hash):
        return []
    if commitment_hash and str(doc.get("commitment_hash") or "") != str(commitment_hash):
        return []
    rows = doc.get("per_icp_results")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _load_latest_scoring_progress_summary(
    bucket: str,
    object_prefix: str,
    *,
    candidate_artifact_hash: str = "",
) -> dict[str, Any]:
    import boto3  # type: ignore

    try:
        s3 = boto3.client("s3")
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=object_prefix, MaxKeys=25)
    except Exception:
        return _safe_scoring_progress_summary(source="s3", completed_icp_count=0)
    contents = listing.get("Contents") if isinstance(listing, Mapping) else None
    if not isinstance(contents, list):
        return _safe_scoring_progress_summary(source="s3", completed_icp_count=0)
    ordered = sorted(
        (
            item for item in contents
            if isinstance(item, Mapping) and str(item.get("Key") or "").endswith(".json")
        ),
        key=lambda item: item.get("LastModified") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    for item in ordered:
        object_key = str(item.get("Key") or "")
        try:
            body = s3.get_object(Bucket=bucket, Key=object_key)["Body"].read()
            doc = json.loads(body.decode("utf-8"))
        except Exception:
            continue
        if not isinstance(doc, Mapping):
            continue
        if candidate_artifact_hash and str(doc.get("candidate_artifact_hash") or "") != candidate_artifact_hash:
            continue
        completed = _completed_icp_count_from_progress_doc(doc)
        if completed <= 0:
            continue
        return _safe_scoring_progress_summary(
            source="s3",
            completed_icp_count=completed,
            rolling_window_hash=str(doc.get("rolling_window_hash") or ""),
        )
    return _safe_scoring_progress_summary(source="s3", completed_icp_count=0)


def _store_scoring_progress(
    bucket: str,
    object_key: str,
    *,
    candidate_id: str,
    window_hash: str,
    candidate_artifact_hash: str,
    rows: list[dict[str, Any]],
    telemetry_index: Mapping[str, Any] | None = None,
    commitment_hash: str = "",
) -> str:
    import boto3  # type: ignore

    doc = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_candidate_scoring_progress",
        "candidate_id": str(candidate_id),
        "rolling_window_hash": str(window_hash),
        "candidate_artifact_hash": str(candidate_artifact_hash),
        "completed_icp_count": len(rows),
        "per_icp_results": rows,
    }
    if commitment_hash:
        doc["schema_version"] = "1.1"
        doc["commitment_hash"] = str(commitment_hash)
    if telemetry_index:
        # Observation-only top-level index. Never copied into per_icp_results
        # or the signed score bundle.
        doc["telemetry_index"] = dict(telemetry_index)
    encoded = json.dumps(doc, sort_keys=True, default=str).encode("utf-8")
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=encoded,
        ContentType="application/json",
    )
    expected_hash = canonical_hash(doc)
    readback = s3.get_object(Bucket=bucket, Key=object_key)["Body"].read()
    readback_doc = json.loads(readback.decode("utf-8"))
    if not isinstance(readback_doc, Mapping) or canonical_hash(readback_doc) != expected_hash:
        raise RuntimeError("research_lab_scoring_progress_readback_hash_mismatch")
    return expected_hash


def _baseline_progress_s3_location(
    manifest_uri: str,
    *,
    benchmark_date: str,
    window_hash: str,
    private_model_artifact_hash: str,
) -> tuple[str, str] | None:
    uri = str(manifest_uri or "")
    if not uri.startswith("s3://"):
        return None
    rest = uri[5:]
    bucket, sep, key = rest.partition("/")
    if not bucket or not sep or not key:
        return None
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    safe_date = "".join(
        ch if ch.isdigit() or ch == "-" else "-" for ch in str(benchmark_date or "date")
    )[:32]
    window_tag = str(window_hash or "").removeprefix("sha256:")[:16] or "window"
    artifact_tag = str(private_model_artifact_hash or "").removeprefix("sha256:")[:16] or "model"
    return (
        bucket,
        f"{base_prefix}/baselines/{safe_date}/scoring-progress/{window_tag}-{artifact_tag}.json",
    )


def _benchmark_item_ref_for_progress(item: Mapping[str, Any]) -> str:
    return str(item.get("icp_ref") or item.get("icp_hash") or "")


def _progress_rows_by_icp_ref(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        ref = str(row.get("icp_ref") or row.get("icp_hash") or "")
        if ref:
            indexed[ref] = dict(row)
    return indexed


def _retryable_measurement_checkpoint_row(row: Mapping[str, Any]) -> bool:
    reasons = {
        token.strip()
        for token in str(row.get("failure_reason") or "").split(";")
        if token.strip()
    }
    return bool(
        row.get("provider_excluded")
        or row.get("provider_cost_cap_blocked")
        or row.get("provider_cost_tracking_failed")
        or any(
            reason.startswith("reference_model_runtime_")
            or "provider_error" in reason
            or "timeout" in reason
            for reason in reasons
        )
    )


def _baseline_summary_nonempty(row: Mapping[str, Any]) -> bool:
    for key in ("company_count", "sourced_count", "model_output_count"):
        try:
            if int(row.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    breakdowns = row.get("score_breakdowns")
    return isinstance(breakdowns, list) and len(breakdowns) > 0


def _icp_company_goal(icp: Any) -> int | None:
    """The ICP's pinned company goal (max_companies), clamped, or None."""
    raw = icp.get("max_companies") if isinstance(icp, Mapping) else None
    if raw is None:
        return None
    try:
        goal = int(raw)
    except (TypeError, ValueError):
        return None
    return max(1, min(goal, 50))


def _baseline_summary_checkpointable(row: Mapping[str, Any]) -> bool:
    if row.get("_runtime_error"):
        return False
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        if diagnostics.get("runtime_error"):
            return False
        categories = {str(value) for value in diagnostics.get("failure_categories") or []}
        if categories.intersection({"runtime_provider_error", "scorer_provider_error", "provider_cost_tracking_failed"}):
            return False
        if "provider_cost_cap_blocked" in categories and not _baseline_summary_nonempty(row):
            return False
    return bool(str(row.get("icp_ref") or row.get("icp_hash") or ""))


def _baseline_progress_public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if not str(key).startswith("_")}


def _load_baseline_scoring_progress(
    bucket: str,
    object_key: str,
    *,
    benchmark_date: str,
    window_hash: str,
    private_model_artifact_hash: str,
    repo_git_sha: str = "",
    manifest_hash: str = "",
) -> list[dict[str, Any]]:
    import boto3  # type: ignore

    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=object_key)["Body"].read()
        doc = json.loads(body.decode("utf-8"))
    except Exception:
        return []
    if not isinstance(doc, Mapping):
        return []
    if str(doc.get("benchmark_date") or "") != str(benchmark_date):
        return []
    if str(doc.get("rolling_window_hash") or "") != str(window_hash):
        return []
    if str(doc.get("private_model_artifact_hash") or "") != str(private_model_artifact_hash):
        return []
    # A benchmark checkpoint is only reusable by the EXACT model that wrote it:
    # a mid-benchmark model change must rescore every ICP with the new model
    # (cost recovery comes from the provider-call cache, never from old score
    # rows). The artifact hash alone missed real change paths (active-row lag
    # vs current.json, promotions), so the repo commit and manifest hash are
    # bound too — any mismatch discards the checkpoint entirely.
    if repo_git_sha and str(doc.get("repo_git_sha") or ""):
        if str(doc.get("repo_git_sha")).lower() != str(repo_git_sha).lower():
            logger.warning(
                "research_lab_baseline_progress_rejected reason=repo_git_sha_changed "
                "checkpoint=%s current=%s",
                str(doc.get("repo_git_sha"))[:16], str(repo_git_sha)[:16])
            return []
    if manifest_hash and str(doc.get("manifest_hash") or ""):
        if str(doc.get("manifest_hash")) != str(manifest_hash):
            logger.warning(
                "research_lab_baseline_progress_rejected reason=manifest_hash_changed "
                "checkpoint=%s current=%s",
                str(doc.get("manifest_hash"))[:24], str(manifest_hash)[:24])
            return []
    rows = doc.get("per_icp_results")
    if not isinstance(rows, list):
        return []
    return [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and _baseline_summary_checkpointable(row)
    ]


def _store_baseline_scoring_progress(
    bucket: str,
    object_key: str,
    *,
    benchmark_date: str,
    window_hash: str,
    private_model_artifact_hash: str,
    rows: list[dict[str, Any]],
    telemetry_index: Mapping[str, Any] | None = None,
    repo_git_sha: str = "",
    manifest_hash: str = "",
) -> str:
    import boto3  # type: ignore

    safe_rows = [
        _baseline_progress_public_row(row)
        for row in rows
        if isinstance(row, Mapping) and _baseline_summary_checkpointable(row)
    ]
    doc = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_private_baseline_scoring_progress",
        "benchmark_date": str(benchmark_date),
        "rolling_window_hash": str(window_hash),
        "private_model_artifact_hash": str(private_model_artifact_hash),
        "repo_git_sha": str(repo_git_sha or ""),
        "manifest_hash": str(manifest_hash or ""),
        "completed_icp_count": len(safe_rows),
        "per_icp_results": safe_rows,
    }
    if telemetry_index:
        doc["telemetry_index"] = dict(telemetry_index)
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=object_key,
        Body=json.dumps(doc, sort_keys=True, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return canonical_hash(doc)


# --- §5.4 scorer-judge traces + baseline in-container trace collection ------
# The qualification scorer's per-company judgments are dense reward labels for
# a future Sales LLM; the private models' in-container provider traffic is the
# matching trajectory data. Both captures are pure observation: best-effort,
# pointer-only in any event/bundle doc, and inert without S3 configuration.

_SCORER_TRACE_CAPTURE_ENV = "RESEARCH_LAB_SCORER_TRACE_CAPTURE"
_SCORER_TRACE_S3_PREFIX_ENV = "RESEARCH_LAB_SCORER_TRACE_S3_PREFIX"
_TRACE_KMS_KEY_ENV = "RESEARCH_LAB_TRACE_KMS_KEY_ID"
_SCORER_TRACE_PUT_CONNECT_TIMEOUT_SECONDS = 5
_SCORER_TRACE_PUT_READ_TIMEOUT_SECONDS = 15
# Model-output fields safe to persist as scorer-INPUT context: company identity
# only — never page content, evidence text, or contact-level material.
_SCORER_TRACE_COMPANY_IDENTITY_FIELDS = (
    "company_name",
    "company_website",
    "company_linkedin",
    "industry",
    "sub_industry",
    "employee_count",
    "company_stage",
    "city",
    "state",
    "country",
)

# The sourcing model emits some identity fields under different key names
# than the canonical ones above ("subindustry" without the underscore,
# "hq_city"/"hq_state"/"hq_country" for the HQ location).  Read those as
# fallbacks so identity capture doesn't silently null those columns out.
_SCORER_TRACE_IDENTITY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "sub_industry": ("sub_industry", "subindustry"),
    "city": ("city", "hq_city"),
    "state": ("state", "hq_state"),
    "country": ("country", "hq_country"),
}


def _scorer_trace_capture_enabled() -> bool:
    return os.getenv(_SCORER_TRACE_CAPTURE_ENV, "true").strip().lower() in {"1", "true", "yes", "on"}


def _trace_kms_key_id() -> str:
    """KMS key used for S3 SSE-KMS trace encryption."""
    return str(os.getenv(_TRACE_KMS_KEY_ENV, "") or "").strip()


def _openrouter_include_reasoning_enabled() -> bool:
    return os.getenv("RESEARCH_LAB_LLM_INCLUDE_REASONING", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _trace_path_segment(value: object, *, fallback: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value or ""))[:96]
    return text.strip("-.") or fallback


def _scorer_trace_company_identity(output: Any) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        return {}
    identity: dict[str, Any] = {}
    for field in _SCORER_TRACE_COMPANY_IDENTITY_FIELDS:
        for key in _SCORER_TRACE_IDENTITY_FIELD_ALIASES.get(field, (field,)):
            value = str(output.get(key) or "").strip()
            if value:
                identity[field] = value
                break
    return identity


_REJECTED_COMPANIES_CAPTURE_ENV = "RESEARCH_LAB_REJECTED_COMPANIES_CAPTURE"
_REJECTED_COMPANIES_TABLE = "research_lab_rejected_companies"
_COMPANY_LABEL_EXAMPLES_CAPTURE_ENV = "RESEARCH_LAB_COMPANY_LABEL_EXAMPLES_CAPTURE"
_COMPANY_LABEL_EXAMPLES_TABLE = "research_lab_company_label_examples"


def _rejected_companies_capture_enabled() -> bool:
    return os.getenv(_REJECTED_COMPANIES_CAPTURE_ENV, "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _company_label_examples_capture_enabled() -> bool:
    return os.getenv(_COMPANY_LABEL_EXAMPLES_CAPTURE_ENV, "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _optional_score(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _optional_text(value: Any, limit: int = 500) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"(?i)sk-or-v1-[A-Za-z0-9_-]+", "[redacted-openrouter-key]", text)
    text = re.sub(r"(?i)sb_secret_[A-Za-z0-9_-]+", "[redacted-supabase-service-key]", text)
    text = re.sub(r"(?i)://([^/\s:@]+):([^/@\s]+)@", "://[redacted-credentials]@", text)
    return text[:limit]


def _optional_public_url(value: Any, limit: int = 500) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
    except ValueError:
        parts = None
    if parts and parts.scheme in {"http", "https"} and parts.netloc:
        safe_netloc = parts.hostname or ""
        if parts.port:
            safe_netloc = f"{safe_netloc}:{parts.port}"
        # Query/fragment frequently contain tokens or tracking IDs.
        return urlunsplit((parts.scheme, safe_netloc, parts.path[:limit], "", ""))[:limit]
    text = _optional_text(raw, limit=limit)
    return text[:limit] if text else None


def _company_identity_key(identity: Mapping[str, Any]) -> str:
    return "".join(
        ch
        for ch in (
            identity.get("company_name")
            or identity.get("company_website")
            or identity.get("company_linkedin")
            or ""
        ).lower()
        if ch.isalnum()
    )


def _is_duplicate_insert_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "duplicate key" in text or "unique constraint" in text or "23505" in text


def _model_side_for_label(*, is_reference_model: bool, candidate_id: str | None, context_ref: str) -> str:
    if not is_reference_model:
        return "candidate"
    if candidate_id:
        return "champion"
    return "baseline_arm"


async def _persist_company_label_examples(
    *,
    context_ref: str,
    icp_ref: str,
    icp_hash: str,
    is_reference_model: bool,
    outputs: Any,
    breakdowns: Any,
    scorer_trace_pointer: Mapping[str, Any] | None = None,
    candidate_id: str | None = None,
    model_manifest_hash: str | None = None,
    run_id: str | None = None,
    ticket_id: str | None = None,
    score_bundle_id: str | None = None,
) -> int:
    """Best-effort positive/negative company-label capture for offline corpus use.

    Full scorer docs remain in SSE-KMS S3; this table stores only identity,
    scores, pass/fail flags, and trace pointers. A failure here must never
    change scoring, promotion, reward, or weight behavior.
    """
    if not _company_label_examples_capture_enabled() or not breakdowns:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    model_side = _model_side_for_label(
        is_reference_model=bool(is_reference_model),
        candidate_id=str(candidate_id) if candidate_id else None,
        context_ref=str(context_ref or ""),
    )
    pointer = dict(scorer_trace_pointer or {}) if isinstance(scorer_trace_pointer, Mapping) else {}
    scorer_trace_ref = str(pointer.get("s3_ref") or "")
    scorer_trace_sha256 = str(pointer.get("sha256") or "")
    raw_trace_refs = [
        {
            "kind": "scorer_judgment_trace",
            "s3_ref": scorer_trace_ref,
            "sha256": scorer_trace_sha256,
        }
    ] if scorer_trace_ref or scorer_trace_sha256 else []
    written = 0
    for index, (output, breakdown) in enumerate(zip(outputs or (), breakdowns or ())):
        if not isinstance(output, Mapping) or not isinstance(breakdown, Mapping):
            continue
        identity = _scorer_trace_company_identity(output)
        company_key = _company_identity_key(identity)
        failure_reason = str(breakdown.get("failure_reason") or "").strip()
        intent_doc = output.get("intent") if isinstance(output.get("intent"), Mapping) else {}
        attr_doc = (
            output.get("required_attribute")
            if isinstance(output.get("required_attribute"), Mapping)
            else {}
        )
        dedup_key = sha256_json(
            {
                "context_ref": str(context_ref),
                "icp_hash": str(icp_hash or ""),
                "icp_ref": str(icp_ref),
                "model_side": model_side,
                "candidate_id": str(candidate_id or ""),
                "company": company_key,
                "failure_reason": failure_reason,
                "index": index if not company_key else None,
            }
        )
        row = {
            "label_id": deterministic_uuid("research_lab_company_label", dedup_key),
            "context_ref": str(context_ref),
            "run_id": str(run_id) if run_id else None,
            "ticket_id": str(ticket_id) if ticket_id else None,
            "candidate_id": str(candidate_id) if candidate_id else None,
            "score_bundle_id": str(score_bundle_id) if score_bundle_id else None,
            "model_manifest_hash": str(model_manifest_hash) if model_manifest_hash else None,
            "model_side": model_side,
            "is_reference_model": bool(is_reference_model),
            "icp_ref": str(icp_ref),
            "icp_hash": str(icp_hash or ""),
            "company_name": identity.get("company_name") or None,
            "company_website": identity.get("company_website") or None,
            "company_linkedin": identity.get("company_linkedin") or None,
            "industry": identity.get("industry") or None,
            "sub_industry": identity.get("sub_industry") or None,
            "employee_count": identity.get("employee_count") or None,
            "company_stage": identity.get("company_stage") or None,
            "city": identity.get("city") or None,
            "state": identity.get("state") or None,
            "country": identity.get("country") or None,
            "model_claimed_score": _optional_score(output.get("score")),
            "intent_source": _optional_text(intent_doc.get("source"), 80),
            "intent_claimed_signal": _optional_text(intent_doc.get("signal"), 200),
            "intent_evidence_url": _optional_public_url(intent_doc.get("url")),
            "intent_evidence_date": _optional_text(intent_doc.get("date"), 40),
            "attribute_evidence_url": _optional_public_url(attr_doc.get("evidence_url")),
            "final_score": float(breakdown.get("final_score", 0.0) or 0.0),
            "failure_reason": _optional_text(failure_reason, 500),
            "failure_stage": _optional_text(breakdown.get("stage_failed"), 120),
            "fit_passed": _optional_bool(breakdown.get("fit_passed")),
            "attribute_passed": _optional_bool(breakdown.get("attribute_passed")),
            "intent_passed": _optional_bool(breakdown.get("intent_passed")),
            "icp_fit": _optional_score(breakdown.get("icp_fit")),
            "intent_signal_raw": _optional_score(breakdown.get("intent_signal_raw")),
            "time_decay_multiplier": _optional_score(breakdown.get("time_decay_multiplier")),
            "intent_signal": _optional_score(breakdown.get("intent_signal_final")),
            "scorer_trace_ref": scorer_trace_ref or None,
            "scorer_trace_sha256": scorer_trace_sha256 or None,
            "raw_trace_refs": raw_trace_refs,
            "capture_doc": {
                "capture_kind": "scorer_company_label",
                "source": "score_with_breakdowns",
                "company_identity_present": bool(company_key),
            },
            "captured_at": now,
            "dedup_key": dedup_key,
        }
        try:
            await insert_row(_COMPANY_LABEL_EXAMPLES_TABLE, row)
            written += 1
        except Exception as exc:  # noqa: BLE001 - capture must never affect scoring
            if _is_duplicate_insert_error(exc):
                continue
            logger.warning(
                "research_lab_company_label_example_insert_failed context=%s icp_ref=%s model_side=%s error=%s",
                compact_ref(str(context_ref)),
                compact_ref(str(icp_ref)),
                model_side,
                _short_error(exc),
            )
            continue
    return written


async def _persist_rejected_companies(
    *,
    context_ref: str,
    icp_ref: str,
    icp_hash: str,
    is_reference_model: bool,
    outputs: Any,
    breakdowns: Any,
    candidate_id: str | None = None,
    model_manifest_hash: str | None = None,
) -> int:
    """Best-effort: persist model-sourced companies the harness REJECTED
    (final_score 0 / an explicit failure_reason) to
    ``research_lab_rejected_companies`` for later false-rejection analysis.

    Company identity only (name / site / linkedin / industry / size / country) —
    never page content or evidence text. Dedup: each unique rejection (icp_hash |
    company | failure_reason | model) is stored once via ``dedup_key`` + the
    table's UNIQUE constraint (a conflicting insert is swallowed). Never raises
    and never blocks scoring. Returns the number of rows written.
    """
    if not _rejected_companies_capture_enabled() or not breakdowns:
        return 0
    written = 0
    now = datetime.now(timezone.utc).isoformat()
    try:
        for output, breakdown in zip(outputs or (), breakdowns or ()):
            if not isinstance(breakdown, Mapping):
                continue
            final_score = float(breakdown.get("final_score", 0.0) or 0.0)
            failure_reason = str(breakdown.get("failure_reason") or "").strip()
            if final_score > 0.0 and not failure_reason:
                continue  # accepted / scored — not a rejection
            identity = _scorer_trace_company_identity(output)
            # Dedup identity: SAME company + SAME error + SAME icp -> one row
            # (regardless of baseline vs candidate). Key on the company NAME,
            # normalized to alphanumerics so case / punctuation / spacing / a
            # linkedin-slug present one run but not the next never splits it.
            company_key = "".join(
                ch
                for ch in (
                    identity.get("company_name")
                    or identity.get("company_website")
                    or identity.get("company_linkedin")
                    or ""
                ).lower()
                if ch.isalnum()
            )
            dedup_key = sha256_json(
                {
                    "icp_hash": str(icp_hash or ""),
                    "company": company_key,
                    "failure_reason": failure_reason,
                }
            )
            # Model-side claims: what the model itself asserted about this
            # company (nested intent / required_attribute docs + its own score).
            # Deliberately EXCLUDED: required_attribute.text (sealed-ICP
            # content) and evidence_quote (scraped page text).
            intent_doc = output.get("intent") if isinstance(output.get("intent"), Mapping) else {}
            attr_doc = (
                output.get("required_attribute")
                if isinstance(output.get("required_attribute"), Mapping)
                else {}
            )

            def _opt_text(value: Any, limit: int = 500) -> str | None:
                text = str(value or "").strip()
                return text[:limit] if text else None

            row = {
                "context_ref": str(context_ref),
                "is_reference_model": bool(is_reference_model),
                "candidate_id": (str(candidate_id) if candidate_id else None),
                "model_manifest_hash": (str(model_manifest_hash) if model_manifest_hash else None),
                "icp_ref": str(icp_ref),
                "icp_hash": str(icp_hash or ""),
                "company_name": identity.get("company_name") or None,
                "company_website": identity.get("company_website") or None,
                "company_linkedin": identity.get("company_linkedin") or None,
                "industry": identity.get("industry") or None,
                "sub_industry": identity.get("sub_industry") or None,
                "employee_count": identity.get("employee_count") or None,
                "company_stage": identity.get("company_stage") or None,
                "city": identity.get("city") or None,
                "state": identity.get("state") or None,
                "country": identity.get("country") or None,
                # model claims (for model-confidence vs harness-verdict analysis)
                "model_claimed_score": _optional_score(output.get("score")),
                "intent_source": _opt_text(intent_doc.get("source"), 80),
                "intent_claimed_signal": _opt_text(intent_doc.get("signal"), 200),
                "intent_evidence_url": _opt_text(intent_doc.get("url")),
                "intent_evidence_date": _opt_text(intent_doc.get("date"), 40),
                "attribute_evidence_url": _opt_text(attr_doc.get("evidence_url")),
                # harness outcome
                "final_score": final_score,
                "failure_reason": failure_reason or None,
                "failure_stage": (str(breakdown.get("stage_failed") or "").strip() or None),
                "fit_passed": _optional_bool(breakdown.get("fit_passed")),
                "attribute_passed": _optional_bool(breakdown.get("attribute_passed")),
                "intent_passed": _optional_bool(breakdown.get("intent_passed")),
                "icp_fit": _optional_score(breakdown.get("icp_fit")),
                "intent_signal_raw": _optional_score(breakdown.get("intent_signal_raw")),
                "time_decay_multiplier": _optional_score(breakdown.get("time_decay_multiplier")),
                "intent_signal": _optional_score(breakdown.get("intent_signal_final")),
                "captured_at": now,
                "dedup_key": dedup_key,
            }
            try:
                await insert_row(_REJECTED_COMPANIES_TABLE, row)
                written += 1
            except Exception:
                pass  # duplicate (UNIQUE dedup_key) or transient — best-effort
    except Exception:
        pass
    return written


class _ScorerTraceRecorder:
    """Best-effort SSE-KMS S3 capture of the qualification scorer's per-company
    judgments at the ``score_with_breakdowns`` boundary (§5.4).

    One JSON doc per (context, icp) under
    ``{prefix}/scorer-traces/{context_ref}/{icp_ref}.json`` — retried ICPs
    overwrite the same key, so the freshest attempt wins (matching the batch's
    replace-on-retry semantics). ``{prefix}`` comes from
    ``RESEARCH_LAB_SCORER_TRACE_S3_PREFIX`` or falls back to the private model
    manifest's bucket/prefix (like the scoring-progress writer); with neither
    resolvable the recorder counts-and-drops after one log line. Hard rules,
    mirroring ``_OpenRouterRawTraceRecorder``:

      * a capture failure can NEVER affect scoring — S3 writes are
        fire-and-forget on a small background pool, every synchronous step is
        exception-wrapped, and failures log once per context;
      * event/bundle docs receive ONLY ``{s3_ref, sha256}`` pointers — never
        breakdown content (the protected-material scanners reject content
        keys);
      * docs store scorer INPUTS limited to icp context refs plus company
        identity fields, and the scorer's full per-company breakdowns
        (including any reasoning text the scorer returns);
      * the pointer is optimistic: returned while the write is in flight, so a
        failed write leaves a dangling (never wrong) reference.

    Flag: ``RESEARCH_LAB_SCORER_TRACE_CAPTURE`` (default true; inert without
    S3/KMS config). Encryption: SSE-KMS via ``RESEARCH_LAB_TRACE_KMS_KEY_ID``.
    """

    def __init__(self, config: ResearchLabGatewayConfig):
        self.config = config
        self._lock = threading.Lock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending: set[concurrent.futures.Future[None]] = set()
        self._failure_logged_contexts: set[str] = set()
        self._destinations: dict[str, tuple[str, str] | None] = {}
        self._drop_logged = False
        self._dropped_docs = 0
        self._disabled = False

    def capture(
        self,
        *,
        context_ref: str,
        icp_ref: str,
        icp_hash: str = "",
        outputs: Any = (),
        breakdowns: Any = (),
        is_reference_model: bool = False,
        manifest_uri: str = "",
    ) -> dict[str, str] | None:
        """Queue one scorer-judgment doc for upload.

        Returns the ``{s3_ref, sha256}`` pointer for event/bundle docs, or None
        when capture is disabled/unconfigured or there is nothing to record.
        Never raises."""
        try:
            if self._disabled or not _scorer_trace_capture_enabled() or not breakdowns:
                return None
            destination = self._resolve_destination(manifest_uri)
            if destination is None:
                self._count_drop(reason="missing_s3_prefix", env_name=_SCORER_TRACE_S3_PREFIX_ENV)
                return None
            kms_key_id = _trace_kms_key_id()
            if not kms_key_id:
                self._count_drop(reason="missing_kms_key", env_name=_TRACE_KMS_KEY_ENV)
                return None
            bucket, key_prefix = destination
            safe_context = _trace_path_segment(context_ref, fallback="context")
            safe_icp = _trace_path_segment(icp_ref, fallback="icp")
            object_key = "/".join(
                segment
                for segment in (key_prefix, "scorer-traces", safe_context, f"{safe_icp}.json")
                if segment
            )
            doc = {
                "schema_version": "1.0",
                "artifact_type": "research_lab_scorer_judgment_trace",
                "context_ref": str(context_ref),
                "icp_ref": str(icp_ref),
                "icp_hash": str(icp_hash or ""),
                "is_reference_model": bool(is_reference_model),
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "sourced_count": len(outputs) if outputs else 0,
                "scored_count": len(breakdowns),
                "companies": [_scorer_trace_company_identity(output) for output in (outputs or ())],
                "score_breakdowns": [dict(item) for item in breakdowns if isinstance(item, Mapping)],
            }
            body = canonical_json(doc).encode("utf-8")
            digest = sha256_json(doc)
            self._submit(
                context_ref=str(context_ref),
                bucket=bucket,
                object_key=object_key,
                body=body,
                kms_key_id=kms_key_id,
            )
            return {"s3_ref": f"s3://{bucket}/{object_key}", "sha256": digest}
        except Exception as exc:
            self._log_failure_once(str(context_ref or "unknown"), exc)
            return None

    def flush(self, timeout_seconds: float = 10.0) -> None:
        """Wait for in-flight uploads (tests / orderly teardown only)."""
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            with self._lock:
                pending = tuple(self._pending)
            if not pending or time.monotonic() >= deadline:
                return
            for future in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                try:
                    future.exception(timeout=remaining)
                except Exception:
                    return

    def _resolve_destination(self, manifest_uri: str) -> tuple[str, str] | None:
        prefix_uri = str(os.getenv(_SCORER_TRACE_S3_PREFIX_ENV, "")).strip().rstrip("/")
        cache_key = prefix_uri or str(manifest_uri or "")
        with self._lock:
            if cache_key in self._destinations:
                return self._destinations[cache_key]
        if not prefix_uri:
            uri = str(manifest_uri or "").strip()
            if uri.startswith("s3://"):
                bucket, _sep, key = uri[5:].partition("/")
                base_prefix = key.rsplit("/", 1)[0] if "/" in key else ""
                if bucket:
                    prefix_uri = f"s3://{bucket}/{base_prefix}".rstrip("/")
        destination: tuple[str, str] | None = None
        if prefix_uri.startswith("s3://"):
            bucket, _sep, key_prefix = prefix_uri[5:].partition("/")
            if bucket:
                destination = (bucket, key_prefix.strip("/"))
        with self._lock:
            self._destinations[cache_key] = destination
        return destination

    def _count_drop(self, *, reason: str, env_name: str) -> None:
        with self._lock:
            self._dropped_docs += 1
            if self._drop_logged:
                return
            self._drop_logged = True
        # Local/dev inertness: no S3 destination — one log, then silent drops.
        logger.info(
            "research_lab_scorer_trace_capture_disabled reason=%s env=%s; "
            "scorer judgment capture is skipped for this process",
            reason,
            env_name,
        )

    def _submit(
        self,
        *,
        context_ref: str,
        bucket: str,
        object_key: str,
        body: bytes,
        kms_key_id: str,
    ) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="research-lab-scorer-trace",
                )
            executor = self._executor
        future = executor.submit(
            self._put_object,
            bucket=bucket,
            object_key=object_key,
            body=body,
            kms_key_id=kms_key_id,
        )
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(lambda done: self._consume_put_result(context_ref, done))

    def _put_object(self, *, bucket: str, object_key: str, body: bytes, kms_key_id: str) -> None:
        import boto3  # type: ignore

        client_kwargs: dict[str, Any] = {}
        try:
            from botocore.config import Config as BotoClientConfig  # type: ignore

            client_kwargs["config"] = BotoClientConfig(
                connect_timeout=_SCORER_TRACE_PUT_CONNECT_TIMEOUT_SECONDS,
                read_timeout=_SCORER_TRACE_PUT_READ_TIMEOUT_SECONDS,
                retries={"max_attempts": 2},
            )
        except Exception:  # pragma: no cover - botocore ships with boto3
            client_kwargs = {}
        put_kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": object_key,
            "Body": body,
            "ContentType": "application/json",
            # SSE-KMS at rest via an ENCRYPT_DECRYPT key, matching the
            # raw-trace recorder's §9.1 encryption requirement.
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": kms_key_id,
        }
        boto3.client("s3", **client_kwargs).put_object(**put_kwargs)

    def _consume_put_result(self, context_ref: str, future: "concurrent.futures.Future[None]") -> None:
        with self._lock:
            self._pending.discard(future)
        if future.cancelled():
            return
        exc = future.exception()
        if exc is None:
            return
        if isinstance(exc, ImportError):
            # boto3 absent: local/dev environment — go inert for the process.
            with self._lock:
                self._disabled = True
        self._log_failure_once(context_ref, exc)

    def _log_failure_once(self, context_ref: str, exc: BaseException) -> None:
        context_key = str(context_ref or "unknown")
        with self._lock:
            if context_key in self._failure_logged_contexts:
                return
            self._failure_logged_contexts.add(context_key)
        logger.warning(
            "research_lab_scorer_trace_capture_failed context=%s error=%s; capture is "
            "best-effort and scoring was not affected",
            compact_ref(context_key),
            _short_error(exc),
        )


class _TraceCapturingCompanyScorer:
    """``QualificationStyleCompanyScorer`` delegate that records each judgment
    through a ``_ScorerTraceRecorder`` (§5.4 candidate-eval scorer boundary).

    Pure observation: breakdowns and derived scores are passed through
    untouched, and the capture step is fully exception-contained — with the
    flag off or no S3 config the delegate behaves exactly like the default
    scorer the evaluator would have constructed itself.
    """

    def __init__(
        self,
        *,
        recorder: _ScorerTraceRecorder,
        context_ref: str,
        manifest_uri: str,
        benchmark_items: Any = (),
        pointer_map: dict[str, dict[str, str]] | None = None,
        inner: Any = None,
        candidate_id: str | None = None,
        candidate_model_manifest_hash: str | None = None,
        run_id: str | None = None,
        ticket_id: str | None = None,
        attested_epoch_id: int | None = None,
        attested_purpose: str = "",
        attested_provider_profile: str = "default",
    ):
        self._inner = (
            inner
            if inner is not None
            else QualificationStyleCompanyScorer(
                attested_epoch_id=attested_epoch_id,
                attested_purpose=attested_purpose,
                attested_provider_profile=attested_provider_profile,
            )
        )
        self._recorder = recorder
        self._context_ref = str(context_ref)
        self._manifest_uri = str(manifest_uri or "")
        self._candidate_id = str(candidate_id) if candidate_id else None
        self._candidate_model_manifest_hash = (
            str(candidate_model_manifest_hash) if candidate_model_manifest_hash else None
        )
        self._run_id = str(run_id) if run_id else None
        self._ticket_id = str(ticket_id) if ticket_id else None
        self._pointer_map = pointer_map
        # Per-ICP funnel counts (sourced -> fit -> verified -> intent -> scored)
        # for the candidate model, keyed by icp_ref. Read back onto each per-ICP
        # row via scorer_funnel_for.
        self._funnel_map: dict[str, dict[str, Any]] = {}
        self._evidence_types_map: dict[str, dict[str, Any]] = {}
        # The evaluator hands the scorer the raw ICP payload, not the benchmark
        # item, so refs are recovered via the payload's canonical hash.
        self._icp_refs: dict[str, tuple[str, str]] = {}
        for item in benchmark_items or ():
            try:
                icp = item.get("icp") if isinstance(item, Mapping) else None
                if isinstance(icp, Mapping):
                    self._icp_refs[sha256_json(icp)] = (
                        str(item.get("icp_ref") or item.get("icp_hash") or ""),
                        str(item.get("icp_hash") or ""),
                    )
            except Exception:  # noqa: BLE001 - ref recovery is best-effort
                continue

    async def __call__(
        self,
        companies: Any,
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[float]:
        breakdowns = await self.score_with_breakdowns(companies, icp, is_reference_model)
        return [float(item.get("final_score", 0.0) or 0.0) for item in breakdowns]

    async def score_with_breakdowns(
        self,
        companies: Any,
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[dict[str, Any]]:
        breakdowns = await self._inner.score_with_breakdowns(companies, icp, is_reference_model)
        try:
            icp_key = sha256_json(icp if isinstance(icp, Mapping) else {})
            icp_ref, icp_hash = self._icp_refs.get(icp_key, ("", ""))
            if not icp_ref:
                icp_ref = icp_key.removeprefix("sha256:")[:24]
            pointer = self._recorder.capture(
                context_ref=self._context_ref,
                icp_ref=icp_ref,
                icp_hash=icp_hash,
                outputs=list(companies or ()),
                breakdowns=breakdowns,
                is_reference_model=bool(is_reference_model),
                manifest_uri=self._manifest_uri,
            )
            if pointer and self._pointer_map is not None:
                self._pointer_map[icp_ref] = dict(pointer)
            if not is_reference_model and icp_ref:
                # Isolated from the outer trace-capture guard: a stats failure
                # must not also drop the judgment pointer, and it must log with
                # a distinct tag (a silent failure here blanks the per-ICP
                # funnel for the whole candidate — a real regression once went
                # unnoticed because it shared the generic wrapper warning).
                try:
                    # Per-breakdown intent-signal evidence lists feed the
                    # evidence-type (intent pass rate) panel; absent on the
                    # default scorer, so this stays best-effort.
                    signal_details = [
                        (b.get("intent_signals_detail") or [])
                        if isinstance(b, Mapping) else []
                        for b in (breakdowns or [])
                    ]
                    stats = build_icp_stats(
                        sourced_count=len(companies or ()),
                        breakdowns=breakdowns,
                        signal_details=signal_details if any(signal_details) else None,
                    )
                    funnel = stats.get("funnel")
                    if isinstance(funnel, Mapping):
                        self._funnel_map[icp_ref] = dict(funnel)
                    evidence_types = stats.get("evidence_types")
                    if isinstance(evidence_types, Mapping) and evidence_types:
                        self._evidence_types_map[icp_ref] = dict(evidence_types)
                except Exception:  # noqa: BLE001 - stats never affect scoring
                    logger.warning(
                        "research_lab_scorer_funnel_stats_failed context=%s icp_ref=%s",
                        compact_ref(self._context_ref),
                        compact_ref(icp_ref),
                        exc_info=True,
                    )
            # Persist rejected companies for false-rejection analysis (candidate
            # path). Best-effort; never affects scoring.
            await _persist_company_label_examples(
                context_ref=self._context_ref,
                icp_ref=icp_ref,
                icp_hash=icp_hash,
                is_reference_model=bool(is_reference_model),
                outputs=list(companies or ()),
                breakdowns=breakdowns,
                scorer_trace_pointer=pointer,
                candidate_id=getattr(self, "_candidate_id", None),
                model_manifest_hash=getattr(self, "_candidate_model_manifest_hash", None),
                run_id=getattr(self, "_run_id", None),
                ticket_id=getattr(self, "_ticket_id", None),
            )
            await _persist_rejected_companies(
                context_ref=self._context_ref,
                icp_ref=icp_ref,
                icp_hash=icp_hash,
                is_reference_model=bool(is_reference_model),
                outputs=list(companies or ()),
                breakdowns=breakdowns,
                candidate_id=getattr(self, "_candidate_id", None),
                model_manifest_hash=getattr(self, "_candidate_model_manifest_hash", None),
            )
        except Exception:  # noqa: BLE001 - capture can never affect scoring
            logger.warning(
                "research_lab_scorer_trace_wrapper_failed context=%s",
                compact_ref(self._context_ref),
                exc_info=True,
            )
        return breakdowns

    def scorer_trace_pointer_for(self, icp_ref: str) -> dict[str, str] | None:
        """P12: the evaluator's row builder picks up this ICP's judgment-trace
        pointer so ``scorer_trace_ref`` rides the per-ICP row into the bundle
        (previously the pointers reached only the scored event doc)."""
        if not self._pointer_map:
            return None
        pointer = self._pointer_map.get(str(icp_ref))
        return dict(pointer) if isinstance(pointer, Mapping) else None

    def scorer_funnel_for(self, icp_ref: str) -> dict[str, Any] | None:
        """Return this ICP's candidate funnel counts so they ride the per-ICP
        row into the bundle. Duck-typed; the default scorer has no funnel."""
        funnel = self._funnel_map.get(str(icp_ref))
        return dict(funnel) if isinstance(funnel, Mapping) else None

    def scorer_evidence_types_for(self, icp_ref: str) -> dict[str, Any] | None:
        """Return this ICP's per-evidence-type intent stats (intent pass rate
        panel). Duck-typed; the default scorer has none."""
        evidence = self._evidence_types_map.get(str(icp_ref))
        return dict(evidence) if isinstance(evidence, Mapping) else None

    def attested_receipts(self) -> list[dict[str, Any]]:
        """Expose the inner scorer's additive receipt sidecars."""

        supplier = getattr(self._inner, "attested_receipts", None)
        if not callable(supplier):
            return []
        return [dict(item) for item in supplier() if isinstance(item, Mapping)]

    def attested_outcome_count(self) -> int | None:
        """Expose measured-call progress independently from receipt deduplication."""

        return _attested_outcome_count(self._inner)

    def last_attested_receipt_hash(self) -> str:
        """Expose the receipt emitted by the latest measured scorer call."""

        return _last_attested_receipt_hash(self._inner)


def _upload_baseline_incontainer_trace(
    prefix: str,
    context_ref: str,
    icp_ref: str,
    entries: list[dict[str, Any]],
    kms_key_id: str,
) -> str:
    """Local equivalent of the evaluator's default-sink uploader for the
    baseline batch path, writing ``{prefix}/{context}/baseline/{icp_ref}.json``
    under the same ``RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX`` scheme."""
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise PrivateModelRuntimeError("boto3 is required for in-container trace S3 uploads") from exc
    if not prefix.startswith("s3://"):
        raise PrivateModelRuntimeError(f"invalid in-container trace S3 prefix: {prefix}")
    bucket, _, key_prefix = prefix[5:].partition("/")
    if not bucket:
        raise PrivateModelRuntimeError(f"invalid in-container trace S3 prefix: {prefix}")
    key = "/".join(
        part
        for part in (
            key_prefix.strip("/"),
            _trace_path_segment(context_ref, fallback="context"),
            "baseline",
            f"{_trace_path_segment(icp_ref, fallback='icp')}.json",
        )
        if part
    )
    if not kms_key_id:
        # P5/P13: never PUT in-container trace content without aws:kms.
        raise PrivateModelRuntimeError(
            "in-container trace upload requires a KMS key id (unencrypted uploads are refused)"
        )
    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_incontainer_trace",
        "run_ref": str(context_ref),
        "icp_ref": str(icp_ref),
        "call_count": len(entries),
        "entries": entries,
    }
    put_kwargs: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": canonical_json(payload).encode("utf-8"),
        "ContentType": "application/json",
        "ServerSideEncryption": "aws:kms",
        "SSEKMSKeyId": kms_key_id,
    }
    boto3.client("s3").put_object(**put_kwargs)
    return f"s3://{bucket}/{key}"


async def _publish_baseline_incontainer_trace(
    *,
    context_ref: str,
    icp_ref: str,
    entries: list[dict[str, Any]],
    drop_state: dict[str, Any],
) -> dict[str, Any]:
    """Persist one baseline/champion ICP's collected in-container trace entries
    and build the POINTER-ONLY diagnostics fields (mirroring the evaluator's
    ``_finalize_incontainer_trace`` semantics: upload failures are logged and
    swallowed). P5: a dropped capture must not look populated — no ref means
    ``incontainer_trace_call_count=0`` plus ``incontainer_trace_dropped=true``;
    the sha256 stays as the attestation the entries existed. Returns ``{}``
    when there is nothing to record."""
    if not entries:
        return {}
    ref = ""
    prefix = str(os.getenv(INCONTAINER_TRACE_S3_PREFIX_ENV) or "").strip().rstrip("/")
    kms_key_id = str(os.getenv(INCONTAINER_TRACE_KMS_KEY_ENV) or "").strip()
    if not prefix or not kms_key_id:
        reason = "no_s3_prefix" if not prefix else "missing_kms_key"
        drop_state["dropped_entries"] = int(drop_state.get("dropped_entries") or 0) + len(entries)
        if not drop_state.get("logged"):
            drop_state["logged"] = True
            log = logger.info if not prefix else logger.error
            log(
                "research_lab_incontainer_trace_dropped run_ref=%s reason=%s "
                "(set %s and %s to persist baseline in-container traces encrypted; "
                "further drops this run are silent)",
                context_ref,
                reason,
                INCONTAINER_TRACE_S3_PREFIX_ENV,
                INCONTAINER_TRACE_KMS_KEY_ENV,
            )
    else:
        try:
            ref = await asyncio.to_thread(
                _upload_baseline_incontainer_trace,
                prefix,
                str(context_ref),
                str(icp_ref),
                list(entries),
                kms_key_id,
            )
        except Exception:  # noqa: BLE001 - capture must never fail the run
            logger.warning(
                "research_lab_incontainer_trace_sink_failed icp_ref=%s entry_count=%s",
                icp_ref,
                len(entries),
                exc_info=True,
            )
            ref = ""
    fields = {
        "incontainer_trace_ref": ref,
        "incontainer_trace_sha256": sha256_json(list(entries)),
        "incontainer_trace_call_count": len(entries) if ref else 0,
    }
    # P13: truncation is filterable from the index, not just inside the blob.
    truncated_count = sum(1 for entry in entries if entry.get("truncated"))
    if truncated_count:
        fields["incontainer_trace_truncated_count"] = truncated_count
    provider_cost_summary = summarize_provider_cost_trace_entries(entries)
    if (
        provider_cost_summary.get("paid_call_count")
        or provider_cost_summary.get("cache_hit_count")
        or provider_cost_summary.get("blocked_call_count")
        or provider_cost_summary.get("tracking_failed_count")
    ):
        fields["provider_cost_summary"] = provider_cost_summary
        fields["provider_cost_total_usd"] = provider_cost_summary.get("total_cost_usd", 0.0)
        fields["provider_cost_paid_call_count"] = provider_cost_summary.get("paid_call_count", 0)
        fields["provider_cost_cap_blocked"] = bool(provider_cost_summary.get("cap_blocked"))
        fields["provider_cost_tracking_failed"] = int(provider_cost_summary.get("tracking_failed_count") or 0) > 0
    if not ref:
        fields["incontainer_trace_dropped"] = True
        fields["incontainer_trace_dropped_call_count"] = len(entries)
    return fields


def _apply_provider_cost_baseline_outcome(item_summary: dict[str, Any]) -> None:
    diagnostics = item_summary.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return
    summary = diagnostics.get("provider_cost_summary")
    if not isinstance(summary, Mapping):
        return
    cap_blocked = bool(summary.get("cap_blocked"))
    tracking_failed = int(summary.get("tracking_failed_count") or 0) > 0
    if not cap_blocked and not tracking_failed:
        return
    mutable_diagnostics = dict(diagnostics)
    categories = set(mutable_diagnostics.get("failure_categories") or [])
    if cap_blocked:
        categories.add("provider_cost_cap_blocked")
        mutable_diagnostics["provider_cost_cap_blocked"] = True
    if tracking_failed:
        categories.add("provider_cost_tracking_failed")
        mutable_diagnostics["provider_cost_tracking_failed"] = True
    mutable_diagnostics["failure_categories"] = sorted(categories)
    if tracking_failed or (cap_blocked and not _baseline_summary_nonempty(item_summary)):
        item_summary["score"] = 0.0
        item_summary["company_count"] = 0
    item_summary["diagnostics"] = mutable_diagnostics


def _provider_cost_cap_state(event: Mapping[str, Any]) -> str:
    if event.get("tracking_failed"):
        return "cost_tracking_failed"
    if event.get("cap_blocked"):
        return "blocked_before_call"
    if event.get("cap_exceeded_after_success"):
        return "exceeded_after_success"
    return "under_cap"


def _is_provider_cost_duplicate_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_provider_cost_events" in message
        and (
            "duplicate key" in message
            or "unique constraint" in message
            or "23505" in message
        )
    )


async def _persist_provider_cost_events(
    *,
    entries: list[dict[str, Any]],
    run_type: str,
    icp_ref: str,
    icp_hash: str = "",
    runner_role: str = "unknown",
    candidate_id: str = "",
    benchmark_date: str = "",
    rolling_window_hash: str = "",
    scoring_id: str = "",
    scoring_run_id: str = "",
    icp_execution_id: str = "",
) -> bool:
    all_persisted = True
    telemetry_ids = (str(scoring_id or ""), str(scoring_run_id or ""), str(icp_execution_id or ""))
    if any(telemetry_ids) and not all(telemetry_ids):
        logger.warning(
            "research_lab_provider_cost_telemetry_ids_incomplete scoring_id=%s scoring_run_id=%s icp_execution_id=%s",
            compact_ref(telemetry_ids[0]),
            compact_ref(telemetry_ids[1]),
            compact_ref(telemetry_ids[2]),
        )
        telemetry_ids = ("", "", "")
    for entry in entries:
        event = cost_event_from_trace_entry(entry)
        if not event:
            continue
        provider = str(event.get("provider") or "unknown")
        if provider not in {"exa", "or", "sd", "deepline"}:
            provider = "unknown"
        anchored_payload = {
            "event": event,
            "icp_ref": str(icp_ref),
            "icp_hash": str(icp_hash or ""),
            "runner_role": str(entry.get("runner_role") or runner_role or "unknown"),
            "trace_seq": entry.get("seq"),
        }
        if all(telemetry_ids):
            anchored_payload.update(
                {
                    "scoring_id": telemetry_ids[0],
                    "scoring_run_id": telemetry_ids[1],
                    "icp_execution_id": telemetry_ids[2],
                }
            )
        anchored_hash = sha256_json(anchored_payload)
        row = {
            "event_id": deterministic_uuid("provider_cost_event", anchored_hash),
            "run_scope": str(event.get("scope") or "unscoped"),
            "run_type": str(run_type or "unknown"),
            "candidate_id": str(candidate_id or "") or None,
            "benchmark_date": str(benchmark_date or "") or None,
            "rolling_window_hash": str(rolling_window_hash or "") or None,
            "icp_ref": str(icp_ref or ""),
            "icp_hash": str(icp_hash or ""),
            "runner_role": str(entry.get("runner_role") or runner_role or "unknown"),
            "provider": provider,
            "endpoint": str(event.get("endpoint") or "")[:200],
            "model": str(event.get("model") or "")[:200],
            "request_fingerprint": str(event.get("request_fingerprint") or "")[:64],
            "status_code": int(event.get("status_code") or 0),
            "billable": bool(event.get("billable")),
            "cost_usd": float(event.get("cost_usd") or 0.0),
            "cost_source": str(event.get("cost_source") or "not_billable")[:80],
            "credits": int(event.get("credits") or 0),
            "prompt_tokens": int(event.get("prompt_tokens") or 0),
            "completion_tokens": int(event.get("completion_tokens") or 0),
            "cap_usd": float(event.get("cap_usd") or 0.0),
            "spent_before_usd": float(event.get("spent_before_usd") or 0.0),
            "spent_after_usd": float(event.get("spent_after_usd") or 0.0),
            "cap_state": _provider_cost_cap_state(event),
            "event_doc": {
                "evidence": str(event.get("evidence") or ""),
                "generation_id": str(event.get("generation_id") or "")[:200],
                "tracking_reason": str(event.get("tracking_reason") or "")[:200],
                "trace_seq": entry.get("seq"),
            },
            "anchored_hash": anchored_hash,
        }
        if all(telemetry_ids):
            row.update(
                {
                    "scoring_id": telemetry_ids[0],
                    "scoring_run_id": telemetry_ids[1],
                    "icp_execution_id": telemetry_ids[2],
                }
            )
        try:
            await insert_row("research_lab_provider_cost_events", row)
        except Exception as exc:  # noqa: BLE001 - telemetry must not affect scoring
            if _is_provider_cost_duplicate_error(exc):
                logger.debug(
                    "research_lab_provider_cost_event_duplicate run_type=%s icp_ref=%s provider=%s",
                    run_type,
                    compact_ref(icp_ref),
                    provider,
                )
                continue
            logger.warning(
                "research_lab_provider_cost_event_insert_failed run_type=%s icp_ref=%s provider=%s",
                run_type,
                compact_ref(icp_ref),
                provider,
                exc_info=True,
            )
            all_persisted = False
    return all_persisted


def _baseline_max_unresolved_icps() -> int:
    try:
        return max(0, int(os.getenv("RESEARCH_LAB_BASELINE_MAX_UNRESOLVED_ICPS", "2")))
    except ValueError:
        return 2


DEFAULT_BASELINE_MAX_DAY_JUMP_POINTS = 15.0


def _baseline_max_day_jump_points() -> float | None:
    """Day-over-day quarantine threshold, enforced by default.

    A baseline that swings more than this many points against the previous
    day is far more likely to be a provider outage or measurement failure
    than a real model change, and must not become the promotion reference
    without an operator raising the limit. Explicit "0"/"off"/"none"
    disables enforcement (warn-only).
    """
    raw = os.getenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", "").strip()
    if not raw:
        return DEFAULT_BASELINE_MAX_DAY_JUMP_POINTS
    if raw.lower() in {"0", "0.0", "off", "none", "disabled"}:
        return None
    try:
        return abs(float(raw))
    except ValueError:
        logger.warning(
            "research_lab_baseline_day_jump_threshold_invalid value=%r default=%s",
            raw,
            DEFAULT_BASELINE_MAX_DAY_JUMP_POINTS,
        )
        return DEFAULT_BASELINE_MAX_DAY_JUMP_POINTS


def _baseline_min_utc_day_delay_seconds() -> int:
    raw = os.getenv(
        "RESEARCH_LAB_BASELINE_START_UTC_OFFSET_SECONDS",
        os.getenv(
            "RESEARCH_LAB_BASELINE_MIN_UTC_DAY_DELAY_SECONDS",
            str(DEFAULT_BASELINE_START_UTC_OFFSET_SECONDS),
        ),
    ).strip()
    try:
        return max(0, min(86399, int(raw)))
    except ValueError:
        return DEFAULT_BASELINE_START_UTC_OFFSET_SECONDS


def _candidate_scoring_quiet_start_utc_seconds() -> int:
    raw = os.getenv(
        "RESEARCH_LAB_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS",
        str(DEFAULT_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS),
    ).strip()
    try:
        return max(0, min(86399, int(raw)))
    except ValueError:
        return DEFAULT_CANDIDATE_SCORING_QUIET_START_UTC_SECONDS


def _utc_seconds_since_day_start(value: datetime) -> int:
    dt = value.astimezone(timezone.utc)
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def _candidate_baseline_target_date(now: datetime, *, quiet_start_seconds: int) -> str:
    utc_now = now.astimezone(timezone.utc)
    if _utc_seconds_since_day_start(utc_now) >= quiet_start_seconds:
        return (utc_now.date() + timedelta(days=1)).isoformat()
    return utc_now.date().isoformat()


def _build_baseline_health(
    *,
    per_icp_summaries: list[dict[str, Any]],
    retried: int,
    recovered: int,
    max_unresolved_icps: int,
) -> dict[str, Any]:
    """Health summary for a completed baseline run.

    Retry-exhausted provider errors are scored as zero ICPs and recorded here
    for audit. When ``baseline_health_gate_enforced`` is on, a failing
    ``gate_passed`` blocks publication via
    ``_enforce_baseline_publication_gates``.
    """
    return shared_build_baseline_health(
        per_icp_summaries=per_icp_summaries,
        retried=retried,
        recovered=recovered,
        max_unresolved_icps=max_unresolved_icps,
    )


def _enforce_baseline_publication_gates(
    *,
    baseline_health: Mapping[str, Any],
    aggregate_score: float,
    day_jump_points: float | None,
    health_gate_enforced: bool,
    max_day_jump: float | None,
) -> None:
    """Fail-closed publication gates for a completed baseline run.

    A baseline whose own health gate failed (too many retry-exhausted
    provider-error ICPs) or whose aggregate swings implausibly against the
    previous day is a degraded measurement and must not become the day's
    promotion reference.
    """
    if health_gate_enforced and not bool(baseline_health.get("gate_passed")):
        raise BaselineHealthGateFailure(
            "baseline_unresolved_provider_errors_gate_failed: "
            f"unresolved={baseline_health.get('unresolved_provider_errors')} "
            f"max={baseline_health.get('max_unresolved_icps')} "
            f"aggregate={aggregate_score:.4f}; "
            "refusing to record this degraded run as the day's benchmark reference",
            baseline_health=dict(baseline_health),
        )
    if (
        max_day_jump is not None
        and day_jump_points is not None
        and abs(day_jump_points) > max_day_jump
    ):
        raise BaselineHealthGateFailure(
            "baseline_day_over_day_jump_gate_failed: "
            f"jump={day_jump_points:+.4f} max=±{max_day_jump} "
            f"aggregate={aggregate_score:.4f}; "
            "refusing to record this run as the day's benchmark reference",
            baseline_health={**dict(baseline_health), "gate_passed": False},
        )


class CandidateBaselineNotReady(RuntimeError):
    """Raised when candidate scoring must wait for a matching private baseline."""


class CandidateBaselineWindowChanged(CandidateBaselineNotReady):
    """Raised when an in-flight candidate is scoring against an old ICP window."""

    def __init__(
        self,
        *,
        candidate_window_hash: str,
        current_window_hash: str,
        progress: Mapping[str, Any],
    ) -> None:
        self.candidate_window_hash = candidate_window_hash
        self.current_window_hash = current_window_hash
        self.progress = dict(progress)
        completed = int(self.progress.get("completed_icp_count") or 0)
        super().__init__(
            "rolling ICP window changed during candidate scoring: "
            f"candidate_window={compact_ref(candidate_window_hash)} "
            f"current_window={compact_ref(current_window_hash)} "
            f"completed_icps={completed}"
        )


class ClaimLostDuringScoring(RuntimeError):
    """Raised at an ICP boundary when this worker's claim was lost mid-scoring."""


class PrivateBaselineRepoHeadChanged(RuntimeError):
    """Raised at an ICP boundary when repo main advances during a baseline run."""

    def __init__(self, *, expected_git_sha: str, repo_main_sha: str, item_index: int, total_icps: int):
        self.expected_git_sha = expected_git_sha
        self.repo_main_sha = repo_main_sha
        self.item_index = item_index
        self.total_icps = total_icps
        super().__init__(
            "private baseline repo head changed during run: "
            f"expected={compact_ref(expected_git_sha)} "
            f"repo_main={compact_ref(repo_main_sha)} "
            f"before_icp={item_index}/{total_icps}"
        )


# Known-terminal error classes: validation / malformed-candidate style failures
# that will fail identically on every retry. Everything else defaults to
# retryable (bug #10) — the claim-attempt cap remains the loop guard.
_TERMINAL_CANDIDATE_ERROR_CLASSES = (
    CodeEditBuildError,
    CodeEditPatchApplyError,
    ValueError,
    KeyError,
    TypeError,
)


def _candidate_scoring_failure_class(exc: BaseException) -> tuple[str, bool]:
    text = f"{exc.__class__.__name__}: {str(exc)}"
    lowered = text.lower()
    if isinstance(exc, ConditionalValidationRetryableError):
        return "conditional_validation_retryable_failure", True
    if isinstance(exc, CandidateBaselineNotReady) or "matching_completed_private_baseline_required" in lowered:
        return "baseline_not_ready", True
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timed out" in lowered or "timeout" in lowered:
        return "adapter_timeout", True
    if any(
        marker in lowered
        for marker in (
            "docker daemon",
            "no space left on device",
            "failed to prepare",
            "failed to solve",
            "exit status 137",
            "killed",
            "manifest unknown",
        )
    ):
        return "infra_docker_error", True
    diagnostics = _runtime_error_diagnostics(text)
    category = str(diagnostics.get("category") or "")
    provider = str(diagnostics.get("provider") or "unknown")
    status = int(diagnostics.get("status") or 0)
    provider_like = provider != "unknown" or status > 0 or any(
        marker in lowered
        for marker in (
            "provider-backed sourcing failed",
            "scrapingdog",
            "openrouter",
            "exa",
            "internal server error",
            "too many requests",
            "rate limit",
        )
    )
    if not provider_like:
        if isinstance(exc, _TERMINAL_CANDIDATE_ERROR_CLASSES):
            return "candidate_scoring_error", False
        if isinstance(exc, PrivateModelRuntimeError):
            # Model code crashed without provider markers: the candidate's own
            # code is at fault and will crash again.
            return "candidate_runtime_error", False
        # Unknown infra/environment failures (window rotation, KMS/S3 signing,
        # PostgREST resets) default retryable; the attempt cap bounds loops.
        return "candidate_scoring_error", True
    if category in {"provider_http_5xx", "runtime_provider_error"}:
        return category, True
    if category == "provider_http_4xx":
        # Same status-level verdicts as the baseline classifier: 408/429 and
        # Scrapingdog 400s retry; genuine auth/request/410 errors stay terminal.
        return category, _baseline_error_is_retryable(text)
    if isinstance(exc, PrivateModelRuntimeError):
        return "candidate_runtime_error", False
    return "candidate_scoring_error", True


def _candidate_scoring_should_requeue(
    *,
    failure_class: str,
    retryable: bool,
    claim_attempts: int,
    max_attempts: int,
) -> bool:
    if not retryable:
        return False
    if failure_class == "conditional_validation_retryable_failure":
        return True
    return int(claim_attempts) < int(max_attempts)


def _load_candidate_source_diff(candidate: Mapping[str, Any]) -> str:
    build_doc = candidate.get("candidate_build_doc")
    if not isinstance(build_doc, Mapping):
        raise CodeEditBuildError("stale candidate is missing candidate_build_doc")
    uri = str(build_doc.get("source_diff_artifact_uri") or "")
    if not uri:
        raise CodeEditBuildError("stale candidate has no private source diff artifact")
    expected_source_diff_hash = str(
        candidate.get("candidate_source_diff_hash")
        or build_doc.get("source_diff_hash")
        or ""
    )
    payload = _load_private_json_artifact(uri)
    unified_diff = str(payload.get("unified_diff") or "")
    if not unified_diff.strip():
        raise CodeEditBuildError("private source diff artifact is missing unified_diff")
    actual_source_diff_hash = sha256_json({"unified_diff": unified_diff})
    if expected_source_diff_hash and actual_source_diff_hash != expected_source_diff_hash:
        raise CodeEditBuildError(
            "private source diff artifact hash mismatch: "
            f"expected={compact_ref(expected_source_diff_hash)} actual={compact_ref(actual_source_diff_hash)}"
        )
    return unified_diff


def _stale_parent_rebase_depth(candidate: Mapping[str, Any]) -> int:
    build_doc = candidate.get("candidate_build_doc")
    if not isinstance(build_doc, Mapping):
        return 0
    rebase_doc = build_doc.get("stale_parent_rebase")
    if not isinstance(rebase_doc, Mapping):
        return 0
    try:
        return max(1, int(rebase_doc.get("depth") or 1))
    except (TypeError, ValueError):
        return 1


def _stale_parent_progress_doc(progress: Mapping[str, Any]) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "phase": str(progress.get("phase") or "")[:80],
        "completed_icp_count": int(progress.get("completed_icp_count") or 0),
    }
    for key in ("next_icp_index", "last_icp_index"):
        if key in progress:
            try:
                doc[key] = int(progress.get(key) or 0)
            except (TypeError, ValueError):
                pass
    for key in ("icp_ref", "icp_hash"):
        value = str(progress.get(key) or "").strip()
        if value:
            doc[key] = value[:160]
    return doc


def _candidate_baseline_wait_event_doc(exc: BaseException) -> dict[str, Any]:
    if not isinstance(exc, CandidateBaselineWindowChanged):
        return {}
    return {
        "baseline_wait_reason": "rolling_window_changed_during_candidate_scoring",
        "candidate_window_hash": exc.candidate_window_hash,
        "current_window_hash": exc.current_window_hash,
        "stale_window_progress": _stale_parent_progress_doc(exc.progress),
    }


def _load_private_json_artifact(uri: str) -> dict[str, Any]:
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(uri)
        try:
            import boto3  # type: ignore
        except Exception as exc:
            raise CodeEditBuildError("boto3 is required to load private source diff artifacts") from exc
        response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read().decode("utf-8")
    else:
        raw = Path(uri).expanduser().read_text(encoding="utf-8")
    decoded = json.loads(raw)
    if not isinstance(decoded, Mapping):
        raise CodeEditBuildError("private source diff artifact must be a JSON object")
    text = json.dumps(decoded, sort_keys=True).lower()
    if any(marker in text for marker in ("sk-or-", "service_role", "openrouter_api_key", "raw_secret")):
        raise CodeEditBuildError("private source diff artifact contains forbidden secret-like material")
    return dict(decoded)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    raw = str(uri or "")
    if not raw.startswith("s3://"):
        raise CodeEditBuildError("expected s3:// URI")
    without_scheme = raw[5:]
    bucket, sep, key = without_scheme.partition("/")
    if not bucket or not sep or not key:
        raise CodeEditBuildError("invalid s3 URI")
    return bucket, key


def _extract_diff_paths_safe(unified_diff: str) -> set[str]:
    try:
        return extract_unified_diff_paths(unified_diff)
    except Exception:
        return set()


async def _call_operator_openrouter_json(
    *,
    api_key: str,
    model_id: str,
    messages: list[dict[str, str]],
    timeout_seconds: int,
    empty_content_attempts: int = STALE_PARENT_REBASE_REPAIR_EMPTY_CONTENT_ATTEMPTS,
) -> str:
    """Operator stale-parent repair/rebase call, via the shared telemetry
    transport (trajectoryimprovements.md P1: this path used to request
    reasoning, pay for it, then extract only ``message.content`` — the
    reasoning trace is now captured to encrypted S3 instead of discarded)."""
    from research_lab.openrouter_telemetry import (
        OpenRouterTelemetryError,
        call_openrouter_chat_async,
    )

    attempts = max(1, int(empty_content_attempts or 1))
    for attempt in range(1, attempts + 1):
        include_reasoning = attempt == 1
        try:
            result = await call_openrouter_chat_async(
                api_key=api_key,
                model_id=model_id,
                messages=messages,
                channel="research_lab_operator",
                purpose="operator_stale_parent_repair",
                stage="operator_repair" if include_reasoning else "operator_repair_no_reasoning",
                max_tokens=STALE_PARENT_REBASE_REPAIR_MAX_TOKENS,
                temperature=0.1,
                timeout_seconds=max(1, int(timeout_seconds)),
                response_format={"type": "json_object"},
                include_reasoning=include_reasoning,
                extra_body={"reasoning": STALE_PARENT_REBASE_REPAIR_REASONING_BODY}
                if include_reasoning
                else None,
            )
        except OpenRouterTelemetryError as exc:
            raise CodeEditBuildError(f"operator stale-parent repair failed: {exc}") from exc
        content = str(result.content or "").strip()
        if content:
            return content
        if attempt < attempts:
            logger.warning(
                "research_lab_stale_parent_repair_empty_content_retry attempt=%s/%s fallback_no_reasoning=%s",
                attempt,
                attempts,
                attempt == 1,
            )
            continue
    raise CodeEditBuildError(
        f"operator stale-parent repair returned empty content after {attempts} attempt(s)"
    )


def _status_datetime(raw_status_at: object) -> datetime | None:
    if not raw_status_at:
        return None
    text = str(raw_status_at).strip().replace("Z", "+00:00")
    try:
        status_at = datetime.fromisoformat(text)
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
            status_at = datetime.fromisoformat(f"{match.group('prefix')}.{fraction}{suffix}")
        except ValueError:
            return None
    if status_at.tzinfo is None:
        status_at = status_at.replace(tzinfo=timezone.utc)
    return status_at.astimezone(timezone.utc)


def _status_age_seconds(raw_status_at: object) -> float | None:
    status_at = _status_datetime(raw_status_at)
    if status_at is None:
        return None
    return (datetime.now(timezone.utc) - status_at).total_seconds()


def _status_is_stale(raw_status_at: object, stale_after_seconds: int) -> bool:
    age_seconds = _status_age_seconds(raw_status_at)
    return age_seconds is not None and age_seconds > max(60, int(stale_after_seconds))


def _claim_predates_worker_boot(
    row: Mapping[str, Any],
    *,
    worker_ref: str,
    worker_started_at: datetime,
    grace_seconds: int,
) -> bool:
    status_at = _status_datetime(row.get("current_status_at"))
    return (
        row.get("current_evaluator_ref") == worker_ref
        and status_at is not None
        and status_at
        < worker_started_at - timedelta(seconds=max(0, int(grace_seconds)))
    )


def _stale_claim_recovery_owner_index(candidate_id: str, total_workers: int) -> int:
    total = max(1, int(total_workers or 1))
    digest = hashlib.sha256(str(candidate_id).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % total


def _candidate_claim_recovery_reason(
    row: Mapping[str, Any],
    *,
    candidate_id: str,
    worker_ref: str,
    worker_index: int,
    total_workers: int,
    worker_started_at: datetime,
    stale_after_seconds: int,
    restart_orphan_grace_seconds: int,
) -> str | None:
    if not candidate_id:
        return None
    if _status_is_stale(row.get("current_status_at"), stale_after_seconds):
        owner_index = _stale_claim_recovery_owner_index(candidate_id, total_workers)
        return "stale_claim" if int(worker_index) == owner_index else None
    if _claim_predates_worker_boot(
        row,
        worker_ref=worker_ref,
        worker_started_at=worker_started_at,
        grace_seconds=restart_orphan_grace_seconds,
    ):
        return "restart_orphan"
    return None


# Requeue reasons that mean the claim cycle only waited (no scoring happened);
# such cycles must not burn the retry budget (bug #6).
_CLAIM_WAIT_REASONS = frozenset({"baseline_not_ready"})


def _count_claim_attempts(rows: list[Mapping[str, Any]]) -> int:
    """Count genuine scoring attempts from seq-ascending assigned/queued events.

    Each ``assigned`` event opens one attempt. A requeue that closes the cycle
    does not count again (the old counter double-charged stale requeues), and a
    cycle that ended waiting (``baseline_not_ready``) is refunded entirely.
    """
    attempts = 0
    open_cycle = False
    for row in rows:
        event_type = str(row.get("event_type") or "")
        if event_type == "assigned":
            attempts += 1
            open_cycle = True
        elif event_type == "queued" and open_cycle:
            if str(row.get("reason") or "") in _CLAIM_WAIT_REASONS:
                attempts -= 1
            open_cycle = False
    return max(0, attempts)


def _worker_can_claim_candidate_slot(worker_index: int, max_active_claims: int) -> bool:
    """Return whether this scoring worker may start new heavy candidate claims.

    ``max_active_claims=0`` disables the slot gate. Otherwise only the first N
    scoring worker slots can claim new candidates, preventing a large worker
    fleet from stampeding one gateway host with simultaneous Docker evaluations.
    """
    try:
        cap = int(max_active_claims)
    except (TypeError, ValueError):
        cap = 0
    if cap <= 0:
        return True
    try:
        index = int(worker_index)
    except (TypeError, ValueError):
        index = 0
    return index < cap


def _active_claim_capacity_available(active_count: int, max_active_claims: int) -> bool:
    try:
        cap = int(max_active_claims)
    except (TypeError, ValueError):
        cap = 0
    if cap <= 0:
        return True
    try:
        count = int(active_count)
    except (TypeError, ValueError):
        count = 0
    return count < cap


def _scoring_host_pressure_capacity(
    *,
    min_available_memory_mb: int,
    max_load_per_cpu: float,
    available_memory_mb: int | None = None,
    load_per_cpu: float | None = None,
) -> dict[str, Any]:
    """Return whether this host is healthy enough to start another heavy score."""
    try:
        memory_floor = max(0, int(min_available_memory_mb or 0))
    except (TypeError, ValueError):
        memory_floor = 0
    try:
        load_ceiling = max(0.0, float(max_load_per_cpu or 0.0))
    except (TypeError, ValueError):
        load_ceiling = 0.0
    result: dict[str, Any] = {
        "available": True,
        "min_available_memory_mb": memory_floor,
        "max_load_per_cpu": load_ceiling,
    }
    if available_memory_mb is not None:
        result["available_memory_mb"] = int(available_memory_mb)
    if load_per_cpu is not None:
        result["load_per_cpu"] = round(float(load_per_cpu), 3)
    if memory_floor > 0 and available_memory_mb is not None and available_memory_mb < memory_floor:
        return {**result, "available": False, "reason": "host_memory_pressure"}
    if load_ceiling > 0 and load_per_cpu is not None and load_per_cpu > load_ceiling:
        return {**result, "available": False, "reason": "host_load_pressure"}
    return result


def _read_mem_available_mb(meminfo_path: str = "/proc/meminfo") -> int | None:
    try:
        with open(meminfo_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                return int(parts[1]) // 1024
    except Exception:
        return None
    return None


def _load_average_per_cpu() -> float | None:
    try:
        cpu_count = max(1, int(os.cpu_count() or 1))
        return float(os.getloadavg()[0]) / float(cpu_count)
    except Exception:
        return None


class ResearchLabGatewayScoringWorker:
    """Scores Research Lab candidates inside the gateway trust boundary."""

    def __init__(self, config: ResearchLabGatewayConfig, *, worker_ref: str | None = None):
        config.validate_public_benchmark_split()
        self.config = config
        self.worker_ref = worker_ref or config.scoring_worker_id or "research-lab-scoring-worker"
        self.proxy_url = config.scoring_worker_proxy_url or os.getenv("RESEARCH_LAB_SCORING_WORKER_PROXY", "")
        self.proxy_ref_hash = canonical_hash({"proxy_ref": self.proxy_url}) if self.proxy_url else None
        self._baseline_skip_logged = False
        self._baseline_already_logged_date: str | None = None
        self._candidate_start_hold_logged_key: str | None = None
        self._last_candidate_start_gate: dict[str, Any] | None = None
        self._resolved_epoch_cache: tuple[int, float] | None = None
        self._scorer_trace_recorder = _ScorerTraceRecorder(config)
        self._confirmation_trace_scope: dict[str, Any] | None = None
        self._worker_started_at = datetime.now(timezone.utc)
        self._last_private_source_push_reconcile_at = 0.0
        self._stale_parent_overdue_warning_keys: set[str] = set()
        self._active_baseline_context: dict[str, Any] | None = None
        self._baseline_publication_failure_logged_key: str | None = None
        self._baseline_publication_failures_in_process: set[str] = set()

    async def run_forever(self) -> None:
        # trajectoryimprovements.md P5: one structured capture health block at
        # startup; refuses to start in production when capture is degraded.
        from gateway.research_lab.capture_health import enforce_capture_health

        enforce_capture_health(self.config, worker_kind="scoring_worker")
        last_idle_log = 0.0
        last_error_log = 0.0
        idle_log_seconds = _idle_log_seconds()
        error_backoff_seconds = _error_backoff_seconds()
        processed_jobs = 0
        recycle_rss_mb = _worker_recycle_rss_mb()
        recycle_max_jobs = _worker_recycle_max_jobs()
        while True:
            try:
                outcome = await self.run_once()
            except Exception as exc:
                now = time.monotonic()
                if now - last_error_log >= idle_log_seconds:
                    logger.error(
                        format_worker_block(
                            "RESEARCH LAB SCORING WORKER PASS FAILED",
                            (
                                ("Worker", self.worker_ref),
                                ("Error", _short_error(exc)),
                            ),
                        )
                    )
                    last_error_log = now
                await asyncio.sleep(max(self.config.scoring_worker_poll_seconds, error_backoff_seconds))
                continue
            if outcome.get("processed") or outcome.get("status") != "idle":
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB SCORING WORKER PASS",
                        (
                            ("Worker", self.worker_ref),
                            ("Status", outcome.get("status")),
                            ("Candidates", len(outcome.get("candidate_ids") or [])),
                            (
                                "Baseline status",
                                (outcome.get("baseline") or {}).get("status")
                                if isinstance(outcome.get("baseline"), Mapping)
                                else None,
                            ),
                        ),
                    )
                )
            elif time.monotonic() - last_idle_log >= idle_log_seconds:
                logger.info(
                    "Research Lab scoring worker idle: worker_ref=%s poll_seconds=%s",
                    self.worker_ref,
                    self.config.scoring_worker_poll_seconds,
                )
                last_idle_log = time.monotonic()
            # Recycle between passes (never mid-job): scoring passes leave the
            # interpreter's RSS at its high-water mark, so a worker that has
            # finished a heavy bundle can hold gigabytes it will never reuse.
            # Exiting cleanly hands the slot to a fresh ~60 MB process via the
            # supervisor's existing restart-on-exit path.
            if outcome.get("processed"):
                processed_jobs += 1
            rss_mb = _read_own_rss_mb()
            recycle_reason = ""
            if recycle_rss_mb > 0 and rss_mb is not None and rss_mb >= recycle_rss_mb:
                recycle_reason = f"rss {rss_mb}MB >= {recycle_rss_mb}MB"
            elif recycle_max_jobs > 0 and processed_jobs >= recycle_max_jobs:
                recycle_reason = f"processed_jobs {processed_jobs} >= {recycle_max_jobs}"
            if recycle_reason:
                logger.warning(
                    "research_lab_scoring_worker_recycle worker_ref=%s reason=%s "
                    "rss_mb=%s processed_jobs=%s",
                    self.worker_ref,
                    recycle_reason,
                    rss_mb,
                    processed_jobs,
                )
                return
            await asyncio.sleep(max(1, self.config.scoring_worker_poll_seconds))

    async def run_once(self) -> dict[str, Any]:
        if not self.config.scoring_worker_enabled:
            return {"processed": False, "status": "disabled"}
        if not self.config.production_writes_enabled or not self.config.evaluation_bundles_enabled:
            return {"processed": False, "status": "writes_or_eval_disabled"}
        if self.config.scoring_worker_require_proxy and not self.proxy_url:
            try:
                require_worker_proxy_profile_v2(
                    execution_role="gateway_scoring",
                    worker_index=int(self.config.scoring_worker_index or 0),
                )
            except ProviderProfileV2Error as exc:
                logger.warning(
                    "research_lab_scoring_worker_proxy_profile_invalid "
                    "worker_ref=%s worker_index=%s error=%s",
                    self.worker_ref,
                    int(self.config.scoring_worker_index or 0),
                    _short_error(exc),
                )
                return {
                    "processed": False,
                    "status": "scoring_worker_proxy_required",
                }
        maintenance_state = await get_scoring_maintenance_state()
        if bool(maintenance_state.get("paused")) and not str(
            maintenance_state.get("reason") or ""
        ).startswith(PREFLIGHT_REASON_PREFIX):
            # Operator/manual pauses stop the pass outright. A pause carrying
            # the preflight marker instead falls through to the preflight gate
            # below, which auto-resumes once providers recover.
            return {"processed": False, "status": "maintenance_paused"}

        await self._recover_stale_candidate_claims()
        await self._alert_stuck_candidates()

        # Provider preflight: never start a baseline or claim candidates when
        # ScrapingDog/Exa are out of credits or persistently unreachable —
        # every measurement would be a provider-outage zero.
        preflight = await preflight_gate(
            scope="scoring",
            actor_ref=self.worker_ref,
            is_paused=get_scoring_maintenance_state,
            set_paused=set_scoring_maintenance_paused,
            worker_index=self.config.scoring_worker_index,
        )
        if not preflight.get("proceed"):
            return {
                "status": "provider_preflight_unhealthy",
                "preflight": preflight.get("verdicts"),
            }
        # Providers are healthy at this moment: requeue any candidates whose
        # scores were quarantined during an outage for a clean rescore.
        await self._requeue_quarantined_candidates()

        baseline_result = None
        if self.config.private_baseline_rebenchmark_enabled and self._is_private_baseline_owner():
            baseline_result = await self._run_private_baseline_contained()
        elif self.config.private_baseline_rebenchmark_enabled and not self._baseline_skip_logged:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE SKIPPED",
                    (
                        ("Worker", self.worker_ref),
                        ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                        ("Owner worker index", 1),
                        ("Proxy ref", self.proxy_ref_hash),
                    ),
                )
            )
            self._baseline_skip_logged = True

        # §5.2-2: run at most one pending confirmation measurement per pass.
        # Any worker may pick these up (leased per candidate via
        # confirmation_rerun_started events). Best-effort: a confirmation
        # failure must never take down the scoring pass.
        confirmation_result: dict[str, Any] | None = None
        if promotion_confirmation_rerun_enabled():
            try:
                confirmation_result = await self._maybe_run_pending_confirmation()
            except Exception as exc:  # noqa: BLE001 - never fail the worker pass
                logger.warning(
                    "research_lab_confirmation_pass_failed worker_ref=%s error=%s",
                    self.worker_ref,
                    _short_error(exc),
                )

        private_source_reconcile_result: dict[str, Any] | None = None
        if self.config.scoring_worker_index == 0 and self.config.auto_promotion_enabled:
            interval = max(1, private_source_push_retry_seconds())
            now = time.monotonic()
            if now - self._last_private_source_push_reconcile_at >= interval:
                self._last_private_source_push_reconcile_at = now
                try:
                    private_source_reconcile_result = await reconcile_failed_private_source_pushes(
                        self.config,
                        worker_ref=self.worker_ref,
                        dry_run=False,
                    )
                except Exception as exc:  # noqa: BLE001 - never fail the scoring pass
                    logger.warning(
                        "research_lab_private_source_push_reconcile_pass_failed worker_ref=%s error=%s",
                        self.worker_ref,
                        _short_error(exc),
                    )
                stale_rebase_ids = [
                    str(item.get("candidate_id") or "")
                    for item in (private_source_reconcile_result or {}).get("results", [])
                    if item.get("stale_parent_rebase_eligible")
                ]
                stale_rebase_ids = [candidate_id for candidate_id in stale_rebase_ids if candidate_id]
                if stale_rebase_ids:
                    try:
                        from .recovery import rebase_stale_parent_candidates as rebase_reconciled_stale_parents

                        rebase_result = await rebase_reconciled_stale_parents(
                            candidate_ids=stale_rebase_ids,
                            dry_run=False,
                            max_batch_size=len(stale_rebase_ids),
                            actor_ref=self.worker_ref,
                        )
                        private_source_reconcile_result["stale_parent_rebase_result"] = rebase_result
                    except Exception as exc:  # noqa: BLE001 - never fail the scoring pass
                        private_source_reconcile_result["stale_parent_rebase_error"] = _short_error(exc)
                        logger.warning(
                            "research_lab_private_source_push_stale_parent_rebase_failed worker_ref=%s candidates=%s error=%s",
                            self.worker_ref,
                            ",".join(compact_ref(candidate_id) for candidate_id in stale_rebase_ids),
                            _short_error(exc),
                        )

        processed: list[str] = []
        claim_capacity: dict[str, Any] = {"available": True}
        self._last_candidate_start_gate = None
        for _ in range(max(1, self.config.scoring_worker_max_candidates)):
            claim_capacity = await self._candidate_claim_capacity()
            if not claim_capacity.get("available"):
                reason = str(claim_capacity.get("reason") or "")
                if reason == "active_claim_capacity_full":
                    logger.info(
                        "research_lab_candidate_claim_capacity_full worker_ref=%s active_claims=%s max_active_claims=%s",
                        self.worker_ref,
                        claim_capacity.get("active_claims"),
                        claim_capacity.get("max_active_claims"),
                    )
                elif reason in {"host_memory_pressure", "host_load_pressure"}:
                    logger.info(
                        "research_lab_candidate_claim_host_pressure worker_ref=%s reason=%s available_memory_mb=%s min_available_memory_mb=%s load_per_cpu=%s max_load_per_cpu=%s",
                        self.worker_ref,
                        reason,
                        claim_capacity.get("available_memory_mb"),
                        claim_capacity.get("min_available_memory_mb"),
                        claim_capacity.get("load_per_cpu"),
                        claim_capacity.get("max_load_per_cpu"),
                    )
                break
            if global_icp_queue.global_icp_queue_enabled():
                # Global (candidate, icp) queue path: enqueue queued candidates'
                # jobs and score from one shared job pool so the pool stays
                # saturated (no per-candidate tail-gap idling). The
                # per-candidate path below is untouched and unused while on.
                queue_processed = await self._run_global_icp_queue_pass()
                processed.extend(queue_processed)
                break
            candidate = await self._claim_next_candidate()
            if not candidate:
                break
            await self._score_candidate(candidate)
            processed.append(str(candidate["candidate_id"]))

        baseline_completed = (
            isinstance(baseline_result, Mapping)
            and str(baseline_result.get("status") or "") == "completed"
        )
        confirmation_processed = (
            isinstance(confirmation_result, Mapping)
            and bool(confirmation_result.get("processed"))
        )
        private_source_reconcile_processed = (
            isinstance(private_source_reconcile_result, Mapping)
            and (
                int(private_source_reconcile_result.get("retried") or 0) > 0
                or int(private_source_reconcile_result.get("finalized") or 0) > 0
            )
        )
        if processed:
            status = "processed"
        elif baseline_completed:
            status = "baseline_completed"
        elif confirmation_processed:
            status = "confirmation_processed"
        elif private_source_reconcile_processed:
            status = "private_source_reconciled"
        elif not claim_capacity.get("available"):
            status = str(claim_capacity.get("reason") or "candidate_claim_capacity_limited")
        elif self._last_candidate_start_gate and not self._last_candidate_start_gate.get("available"):
            status = str(
                self._last_candidate_start_gate.get("reason")
                or "candidate_scoring_daily_baseline_hold"
            )
        else:
            status = "idle"
        return {
            "processed": bool(
                processed
                or baseline_completed
                or confirmation_processed
                or private_source_reconcile_processed
            ),
            "status": status,
            "candidate_ids": processed,
            "baseline": baseline_result,
            "confirmation": confirmation_result,
            "private_source_reconcile": private_source_reconcile_result,
            "candidate_claim_capacity": claim_capacity,
            "candidate_start_gate": self._last_candidate_start_gate,
        }

    async def _candidate_claim_capacity(self) -> dict[str, Any]:
        host_pressure = _scoring_host_pressure_capacity(
            min_available_memory_mb=getattr(self.config, "scoring_worker_min_available_memory_mb", 0),
            max_load_per_cpu=getattr(self.config, "scoring_worker_max_load_per_cpu", 0.0),
            available_memory_mb=_read_mem_available_mb(),
            load_per_cpu=_load_average_per_cpu(),
        )
        if not host_pressure.get("available"):
            return host_pressure
        max_active = int(getattr(self.config, "scoring_worker_max_active_claims", 0) or 0)
        if max_active <= 0:
            return {
                "available": True,
                "max_active_claims": 0,
                "cap_disabled": True,
                "host_pressure": host_pressure,
            }
        if not _worker_can_claim_candidate_slot(self.config.scoring_worker_index, max_active):
            return {
                "available": False,
                "reason": "claim_slot_disabled",
                "worker_index": self.config.scoring_worker_index,
                "max_active_claims": max_active,
                "host_pressure": host_pressure,
            }
        active = await select_many(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id,current_candidate_status,current_status_at,current_evaluator_ref",
            filters=(("current_candidate_status", "in", ("assigned", "evaluating")),),
            order_by=(("current_status_at", False),),
            limit=max_active,
        )
        active_count = len(active)
        return {
            "available": _active_claim_capacity_available(active_count, max_active),
            "reason": "" if active_count < max_active else "active_claim_capacity_full",
            "active_claims": active_count,
            "max_active_claims": max_active,
            "worker_index": self.config.scoring_worker_index,
            "host_pressure": host_pressure,
        }

    async def _build_queue_candidate_context(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        """Per-candidate scoring context for the global-queue path.

        Mirrors the setup block in ``_score_candidate`` (artifact/patch/window/
        gate/runner/run_context) and additionally splits the window into public,
        private, and optional conditional ICP items using the frozen gate refs. Cached per
        candidate within one queue pass.
        """
        candidate_id = str(candidate["candidate_id"])
        if str(candidate.get("candidate_kind") or "") != "image_build":
            raise RuntimeError("global queue requires image_build candidate_kind")
        evaluation_epoch = await self._resolve_evaluation_epoch()
        artifact = PrivateModelArtifactManifest.from_mapping(candidate["private_model_manifest_doc"])
        patch = candidate["candidate_patch_manifest"]
        candidate_manifest_doc = candidate.get("candidate_model_manifest_doc")
        if not isinstance(candidate_manifest_doc, Mapping):
            raise RuntimeError("image_build candidate missing candidate_model_manifest_doc")
        candidate_artifact = PrivateModelArtifactManifest.from_mapping(candidate_manifest_doc)
        window, gate = await self._daily_candidate_scoring_window_and_gate(artifact=artifact)
        _validate_candidate_conditional_policy_stamp(
            candidate,
            gate,
            active_policy=self.config.conditional_validation_policy().to_dict(),
        )
        await create_rolling_icp_window(window)
        benchmark = SealedBenchmarkSet(
            benchmark_id=window.benchmark_id,
            icp_set_hash=window.window_hash,
            split_ref=window.split_ref,
            item_refs=window.item_refs,
            scoring_version="qualification-company-scorer:v1",
            hidden_plaintext_available=True,
        )
        runner = AttestedPrivateModelRunnerV2(
            artifact=candidate_artifact,
            spec=DockerPrivateModelSpec(
                image_digest=candidate_artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                env_passthrough=self._private_model_env_passthrough(),
                extra_env=self._with_provider_cost_evaluation_scope(
                    self._private_scoring_env(),
                    run_type="candidate_scoring",
                    rolling_window_hash=window.window_hash,
                    artifact_hash=candidate_artifact.model_artifact_hash,
                    candidate_id=candidate_id,
                    run_id=str(candidate.get("run_id") or ""),
                    ticket_id=str(candidate.get("ticket_id") or ""),
                    evaluation_epoch=evaluation_epoch,
                    started_at=time.time(),
                ),
            ),
            model_kind="candidate",
            worker_index=self.config.scoring_worker_index,
            epoch_id=evaluation_epoch,
            parent_graphs=await _attested_model_parent_graphs(
                model_kind="candidate",
                artifact=candidate_artifact,
                candidate_id=candidate_id,
                epoch_id=evaluation_epoch,
            ),
        )
        run_context = self._candidate_run_context(
            candidate, window_hash=window.window_hash, evaluation_epoch=evaluation_epoch
        )
        evaluation_policy = _stored_daily_baseline_evaluation_policy(
            self._evaluation_policy(),
            gate,
        )
        scoring_config_hash = scoring_configuration_hash()
        public_refs = {str(r) for r in (gate.get("public_icp_refs") or ()) if str(r).strip()}
        conditional_required = bool(gate.get("conditional_validation_required"))
        private_refs = {str(r) for r in (gate.get("private_icp_refs") or ()) if str(r).strip()}
        conditional_refs = {
            str(r) for r in (gate.get("conditional_icp_refs") or ()) if str(r).strip()
        }

        def _ref(item: Mapping[str, Any]) -> str:
            return str(item.get("icp_ref") or item.get("icp_hash") or "")

        items = list(window.benchmark_items)
        public_items = [it for it in items if _ref(it) in public_refs]
        if conditional_required:
            private_items = [it for it in items if _ref(it) in private_refs]
            conditional_items = [it for it in items if _ref(it) in conditional_refs]
            assigned_refs = public_refs | private_refs | conditional_refs
            if assigned_refs != {_ref(item) for item in items}:
                raise RuntimeError("conditional global queue assignment does not cover the window")
            if (
                public_refs & private_refs
                or public_refs & conditional_refs
                or private_refs & conditional_refs
            ):
                raise RuntimeError("conditional global queue assignment overlaps")
        else:
            private_items = [it for it in items if _ref(it) not in public_refs]
            conditional_items = []
        return {
            "candidate": dict(candidate),
            "candidate_id": candidate_id,
            "artifact": artifact,
            "patch": patch,
            "candidate_artifact": candidate_artifact,
            "window": window,
            "gate": gate,
            "benchmark": benchmark,
            "runner": runner,
            "run_context": run_context,
            "evaluation_policy": evaluation_policy,
            "scoring_configuration_hash": scoring_config_hash,
            "scorer": QualificationStyleCompanyScorer(
                attested_epoch_id=evaluation_epoch,
                attested_purpose="research_lab.candidate_score.v1",
            ),
            "public_items": public_items,
            "private_items": private_items,
            "conditional_items": conditional_items,
            "items_by_ref": {_ref(it): it for it in items},
            "baseline_public_score": float(gate.get("baseline_public_score") or 0.0),
            "baseline_preliminary_score": float(
                gate.get("baseline_preliminary_score") or 0.0
            ),
            "threshold_points": float(gate.get("threshold_points") or 0.0),
            "baseline_benchmark_bundle_id": str(
                gate.get("baseline_benchmark_bundle_id") or ""
            ),
            "baseline_benchmark_hash": str(
                gate.get("baseline_benchmark_hash") or ""
            ),
            "category_assignment_hash": str(
                gate.get("category_assignment_hash") or ""
            ),
            "conditional_policy_hash": str(
                gate.get("conditional_validation_policy_hash") or ""
            ),
            "trace_sink": self._candidate_incontainer_trace_sink(
                candidate_id,
                persist_costs=not scoring_telemetry_enabled(self.config),
            ),
        }

    async def _queue_assemble_candidate(
        self, candidate_id: str, docs: Mapping[str, Any], ctx: Mapping[str, Any]
    ) -> None:
        """Assemble a fully-scored candidate into a signed score bundle and
        write its scored events. Uses the same shared assembler and bundle
        builder as the per-candidate path, so the bundle is byte-identical."""
        _results, gate_result = build_holdout_gate_result(
            public_results=docs.get("public") or (),
            private_results=docs.get("private") or (),
            conditional_results=docs.get("conditional") or (),
            public_icp_count=len(ctx["public_items"]),
            private_icp_count=len(ctx["private_items"]),
            conditional_icp_count=len(ctx.get("conditional_items") or ()),
            gate=ctx["gate"],
        )
        if bool(ctx["gate"].get("conditional_validation_required")):
            conditional_docs = list(docs.get("conditional") or ())
            conditional_expected = len(ctx.get("conditional_items") or ())
            if (
                str(gate_result.get("decision") or "")
                == "conditional_validation_required"
                or (
                    conditional_docs
                    and len(conditional_docs) != conditional_expected
                )
            ):
                raise ConditionalValidationRetryableError(
                    "conditional_validation_incomplete:result_count_mismatch"
                )
            expected_conditional_refs = {
                str(item.get("icp_ref") or item.get("icp_hash") or "")
                for item in (ctx.get("conditional_items") or ())
            }
            actual_conditional_refs = {
                str(item.get("icp_ref") or item.get("icp_hash") or "")
                for item in conditional_docs
            }
            if conditional_docs and actual_conditional_refs != expected_conditional_refs:
                raise ConditionalValidationRetryableError(
                    "conditional_validation_incomplete:result_identity_mismatch"
                )
            conditional_failures = _critical_measurement_failures(
                conditional_docs
            )
            if conditional_failures:
                raise ConditionalValidationRetryableError(
                    "conditional_validation_incomplete:"
                    + ",".join(conditional_failures)
                )
        score_bundle = build_score_bundle_from_scored_icps(
            artifact_manifest=ctx["artifact"],
            benchmark=ctx["benchmark"],
            patch_manifest=ctx["patch"],
            candidate_artifact_manifest=ctx["candidate_artifact"].to_dict(),
            per_icp_results=_results,
            run_context={**ctx["run_context"], "signature_ref": "pending"},
            policy=ctx["evaluation_policy"],
            extra_bundle_fields={"private_holdout_gate": gate_result},
        )
        await _compare_candidate_score_bundle_in_enclave(
            evaluation_epoch=int(ctx["run_context"]["evaluation_epoch"]),
            artifact=ctx["artifact"],
            benchmark=ctx["benchmark"],
            patch=ctx["patch"],
            candidate_artifact=ctx["candidate_artifact"],
            per_icp_results=_results,
            run_context={**ctx["run_context"], "signature_ref": "pending"},
            policy=ctx["evaluation_policy"],
            private_holdout_gate=gate_result,
            expected_score_bundle=score_bundle,
            parent_receipts=_queue_attested_parent_receipts(
                docs,
                ctx["runner"],
                ctx["scorer"],
            ),
        )
        scoring_health_gate = self._scoring_health_gate_result(score_bundle)
        signature_ref = await asyncio.to_thread(
            sign_digest_with_kms,
            key_id=self.config.score_bundle_kms_key_id,
            digest_hash=str(score_bundle["score_bundle_hash"]),
            signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
        )
        score_bundle = {**score_bundle, "signature_ref": signature_ref}
        candidate = ctx["candidate"]
        bundle, _bundle_event = await create_score_bundle(
            ResearchLabScoreBundleCreateRequest(
                bundle_status="scored",
                receipt_id=candidate.get("receipt_id") or None,
                score_bundle=score_bundle,
            )
        )
        await _persist_conditional_finalization_events(
            gate_result,
            candidate_id=candidate_id,
            source_score_bundle_id=str(bundle["score_bundle_id"]),
            rolling_window_hash=ctx["window"].window_hash,
            queue_generation_id=str(ctx.get("queue_generation_id") or "") or None,
        )
        await _persist_candidate_category_results(
            gate_result,
            source_bundle_ref=str(bundle["score_bundle_id"]),
            rolling_window_hash=ctx["window"].window_hash,
            candidate_id=candidate_id,
            scoring_run_id=(
                str(ctx["telemetry_session"].run.scoring_run_id)
                if isinstance(ctx.get("telemetry_session"), ScoringTelemetrySession)
                and ctx["telemetry_session"].run is not None
                else ""
            ),
        )
        if isinstance(ctx, dict):
            ctx["score_bundle_id"] = str(bundle["score_bundle_id"])
        await self._create_scored_evaluation_event(
            candidate=candidate,
            candidate_id=candidate_id,
            score_bundle_id=str(bundle["score_bundle_id"]),
            event_doc={
                "score_bundle_hash": score_bundle["score_bundle_hash"],
                "rolling_window_hash": ctx["window"].window_hash,
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
                "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                "scoring_health_gate": scoring_health_gate,
                "serving_model_version": _event_serving_model_version(score_bundle),
                "scored_via": "global_icp_queue",
            },
        )
        await create_scoring_dispatch_event(
            dispatch_type="candidate_scoring",
            dispatch_status="scored",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            candidate_id=candidate_id,
            run_id=str(candidate.get("run_id") or ""),
            ticket_id=str(candidate.get("ticket_id") or ""),
            scoring_id=(
                ctx["telemetry_session"].run.scoring_id
                if isinstance(ctx.get("telemetry_session"), ScoringTelemetrySession)
                and ctx["telemetry_session"].run is not None
                else None
            ),
            scoring_run_id=(
                ctx["telemetry_session"].run.scoring_run_id
                if isinstance(ctx.get("telemetry_session"), ScoringTelemetrySession)
                and ctx["telemetry_session"].run is not None
                else None
            ),
        )
        try:
            private_holdout_rejected = (
                str(gate_result.get("decision") or "")
                == "rejected_before_private_holdout"
            )
            if private_holdout_rejected:
                promotion_result = await self._record_public_holdout_rejected(
                    candidate=candidate,
                    score_bundle_row=bundle,
                    score_bundle=score_bundle,
                    gate_result=gate_result,
                )
            elif scoring_health_gate.get("decision") == "quarantine":
                promotion_result = await self._record_scoring_health_quarantined(
                    candidate=candidate,
                    score_bundle_row=bundle,
                    score_bundle=score_bundle,
                    scoring_health_gate=scoring_health_gate,
                )
            else:
                promotion_result = await self._maybe_promote_scored_candidate(
                    candidate=candidate,
                    score_bundle_row=bundle,
                    score_bundle=score_bundle,
                )
            evaluation_epoch = int(ctx["run_context"]["evaluation_epoch"])
            if promotion_result.get("status") == "stale_parent_needs_rescore":
                promotion_result = await self._queue_stale_parent_rebase(
                    candidate,
                    active_artifact=(
                        await load_active_private_model(
                            self.config,
                            register_bootstrap=True,
                        )
                    ).artifact,
                    candidate_parent=str(candidate.get("parent_artifact_hash") or ""),
                    evaluation_epoch=evaluation_epoch,
                    elapsed_seconds=0.0,
                    stage="after_global_queue_scoring_parent_changed",
                )
            await self._maybe_record_score_backfill(
                candidate=candidate,
                score_bundle_row=bundle,
                score_bundle=score_bundle,
                promotion_result=promotion_result,
            )
            await self._maybe_finalize_candidate_receipt(candidate)
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=(
                    f"candidate_scored:{candidate_id}:{bundle['score_bundle_id']}"
                ),
                reason="gateway_qualification_worker_scored_candidate",
                config=self.config,
            )
            await self._write_audit_bundle(evaluation_epoch)
        except Exception as exc:
            await self._record_scored_candidate_side_effect_failure(
                candidate=candidate,
                candidate_id=candidate_id,
                score_bundle_id=str(bundle["score_bundle_id"]),
                error=exc,
                elapsed_seconds=0.0,
            )

    async def _run_global_icp_queue_pass(self) -> list[str]:
        """One worker pass over the global (candidate, icp) queue.

        Enqueues any queued candidate's public jobs (and parks its private jobs
        held for the gate), then claims and scores jobs from the shared pool,
        deciding gates and assembling finished candidates. Returns the ids of
        candidates assembled in this pass.
        """
        ctx_cache: dict[str, dict[str, Any]] = {}

        def _apply_generation_commitments(
            ctx: dict[str, Any],
            generation_row: Mapping[str, Any],
        ) -> dict[str, Any]:
            if int(generation_row.get("conditional_total") or 0) <= 0:
                return ctx
            expected_strings = {
                "candidate_id": str(ctx["candidate_id"]),
                "window_hash": str(ctx["window"].window_hash),
                "baseline_benchmark_bundle_id": str(
                    ctx["baseline_benchmark_bundle_id"]
                ),
                "baseline_benchmark_hash": str(ctx["baseline_benchmark_hash"]),
                "category_assignment_hash": str(ctx["category_assignment_hash"]),
                "conditional_policy_hash": str(ctx["conditional_policy_hash"]),
                "candidate_artifact_hash": str(
                    ctx["candidate_artifact"].model_artifact_hash
                ),
                "candidate_parent_artifact_hash": str(
                    ctx["artifact"].model_artifact_hash
                ),
                "scoring_configuration_hash": str(
                    ctx["scoring_configuration_hash"]
                ),
            }
            for field, expected in expected_strings.items():
                if str(generation_row.get(field) or "") != expected:
                    raise ConditionalValidationRetryableError(
                        "conditional_queue_commitment_mismatch:" + field
                    )
            expected_counts = {
                "public_total": len(ctx["public_items"]),
                "private_total": len(ctx["private_items"]),
                "conditional_total": len(ctx["conditional_items"]),
            }
            for field, expected in expected_counts.items():
                if int(generation_row.get(field) or 0) != int(expected):
                    raise ConditionalValidationRetryableError(
                        "conditional_queue_commitment_mismatch:" + field
                    )
            expected_scores = {
                "baseline_public_score": float(ctx["baseline_public_score"]),
                "baseline_preliminary_score": float(
                    ctx["baseline_preliminary_score"]
                ),
                "threshold_points": float(ctx["threshold_points"]),
            }
            for field, expected in expected_scores.items():
                actual = float(generation_row.get(field) or 0.0)
                if not math.isfinite(actual) or abs(actual - expected) > 1e-9:
                    raise ConditionalValidationRetryableError(
                        "conditional_queue_commitment_mismatch:" + field
                    )
            proof = generation_row.get("preliminary_gate_proof")
            proof_doc = dict(proof) if isinstance(proof, Mapping) else {}
            preliminary_status = str(
                generation_row.get("preliminary_gate_status") or ""
            )
            if preliminary_status == "passed" and not proof_doc:
                raise ConditionalValidationRetryableError(
                    "conditional_queue_preliminary_proof_missing"
                )
            if proof_doc:
                ctx["gate"] = {
                    **dict(ctx["gate"]),
                    "preliminary_promotion_gate": proof_doc,
                }
            return ctx

        async def _ctx(
            candidate_id: str,
            queue_generation_id: str,
            scoring_run_id: str = "",
        ) -> dict[str, Any] | None:
            cache_key = queue_generation_id or candidate_id
            generation_row: Mapping[str, Any] | None = None
            if queue_generation_id:
                generation_row = await select_one(
                    global_icp_queue.CANDIDATE_TABLE,
                    filters=(("queue_generation_id", queue_generation_id),),
                )
                if generation_row is None:
                    return None
                candidate_id = candidate_id or str(
                    generation_row.get("candidate_id") or ""
                )
                scoring_run_id = scoring_run_id or str(
                    generation_row.get("scoring_run_id") or ""
                )
            if cache_key in ctx_cache:
                cached = ctx_cache[cache_key]
                return (
                    _apply_generation_commitments(cached, generation_row)
                    if generation_row is not None
                    else cached
                )
            row = await select_one(
                "research_lab_candidate_evaluation_current",
                filters=(("candidate_id", candidate_id),),
            )
            if row is None:
                return None
            ctx = await self._build_queue_candidate_context(dict(row))
            telemetry_session = await load_scoring_session(scoring_run_id)
            ctx["telemetry_session"] = telemetry_session
            ctx["queue_generation_id"] = queue_generation_id
            if generation_row is not None:
                _apply_generation_commitments(ctx, generation_row)
            if scoring_telemetry_enabled(self.config) and telemetry_session is None:
                # Allocation/hydration failed: preserve legacy provider-cost
                # capture rather than silently dropping spend rows.
                ctx["trace_sink"] = self._candidate_incontainer_trace_sink(
                    candidate_id,
                    persist_costs=True,
                )
            ctx_cache[cache_key] = ctx
            return ctx

        # 1. Enqueue queued candidates not already in the job queue.
        queued = await select_many(
            "research_lab_candidate_evaluation_current",
            columns="*",
            filters=(("current_candidate_status", "queued"),),
            order_by=(("current_status_at", False),),
            limit=50,
        )
        for offset, row in enumerate(queued):
            cid = str(row.get("candidate_id") or "")
            try:
                ctx = await self._build_queue_candidate_context(dict(row))
            except Exception:
                logger.warning(
                    "research_lab_queue_enqueue_context_failed candidate_id=%s",
                    compact_ref(cid),
                    exc_info=True,
                )
                continue
            active_generation = await select_one(
                global_icp_queue.CANDIDATE_TABLE,
                filters=(
                    ("candidate_id", cid),
                    ("window_hash", ctx["window"].window_hash),
                    ("assembly_status", "in", ("pending", "assembling")),
                ),
            )
            if active_generation is not None:
                # Re-fill idempotently so a transient crash between the
                # generation row and its job inserts cannot wedge this
                # candidate forever.
                await global_icp_queue.enqueue_candidate(
                    candidate_id=cid,
                    window_hash=ctx["window"].window_hash,
                    public_items=ctx["public_items"],
                    private_items=ctx["private_items"],
                    conditional_items=ctx["conditional_items"],
                    baseline_public_score=ctx["baseline_public_score"],
                    baseline_preliminary_score=ctx["baseline_preliminary_score"],
                    threshold_points=ctx["threshold_points"],
                    baseline_benchmark_bundle_id=ctx["baseline_benchmark_bundle_id"],
                    baseline_benchmark_hash=ctx["baseline_benchmark_hash"],
                    category_assignment_hash=ctx["category_assignment_hash"],
                    conditional_policy_hash=ctx["conditional_policy_hash"],
                    candidate_artifact_hash=(
                        ctx["candidate_artifact"].model_artifact_hash
                    ),
                    candidate_parent_artifact_hash=ctx["artifact"].model_artifact_hash,
                    scoring_configuration_hash=ctx["scoring_configuration_hash"],
                    worker_ref=self.worker_ref,
                    seq_base=offset * 10_000,
                    scoring_run_id=str(active_generation.get("scoring_run_id") or ""),
                )
                continue
            telemetry_session: ScoringTelemetrySession | None = None
            if scoring_telemetry_enabled(self.config):
                telemetry_run = await allocate_scoring_run(
                    identity_doc={
                        "run_type": "candidate_scoring",
                        "candidate_id": cid,
                        "source_run_id": str(row.get("run_id") or ""),
                        "rolling_window_hash": ctx["window"].window_hash,
                        "reference_artifact_hash": ctx["artifact"].model_artifact_hash,
                        "candidate_artifact_hash": ctx["candidate_artifact"].model_artifact_hash,
                        "evaluation_epoch": int(ctx["run_context"].get("evaluation_epoch") or 0),
                        "scheduler_type": "global_icp_queue",
                    },
                    run_type="candidate_scoring",
                    worker_ref=self.worker_ref,
                    expected_icp_count=(
                        len(ctx["public_items"])
                        + len(ctx["private_items"])
                        + len(ctx["conditional_items"])
                    ),
                    scheduler_type="global_icp_queue",
                    candidate_id=cid,
                    source_run_id=str(row.get("run_id") or ""),
                    ticket_id=str(row.get("ticket_id") or ""),
                    rolling_window_hash=ctx["window"].window_hash,
                    reference_artifact_hash=ctx["artifact"].model_artifact_hash,
                    reference_manifest_hash=ctx["artifact"].manifest_hash,
                    candidate_artifact_hash=ctx["candidate_artifact"].model_artifact_hash,
                    candidate_manifest_hash=ctx["candidate_artifact"].manifest_hash,
                    baseline_benchmark_bundle_id=str(
                        ctx["gate"].get("baseline_benchmark_bundle_id") or ""
                    ),
                    evaluation_epoch=int(ctx["run_context"].get("evaluation_epoch") or 0),
                )
                if telemetry_run is not None:
                    telemetry_session = ScoringTelemetrySession(telemetry_run)
            queue_generation_id = await global_icp_queue.enqueue_candidate(
                candidate_id=cid,
                window_hash=ctx["window"].window_hash,
                public_items=ctx["public_items"],
                private_items=ctx["private_items"],
                conditional_items=ctx["conditional_items"],
                baseline_public_score=ctx["baseline_public_score"],
                baseline_preliminary_score=ctx["baseline_preliminary_score"],
                threshold_points=ctx["threshold_points"],
                baseline_benchmark_bundle_id=ctx["baseline_benchmark_bundle_id"],
                baseline_benchmark_hash=ctx["baseline_benchmark_hash"],
                category_assignment_hash=ctx["category_assignment_hash"],
                conditional_policy_hash=ctx["conditional_policy_hash"],
                candidate_artifact_hash=ctx["candidate_artifact"].model_artifact_hash,
                candidate_parent_artifact_hash=ctx["artifact"].model_artifact_hash,
                scoring_configuration_hash=ctx["scoring_configuration_hash"],
                worker_ref=self.worker_ref,
                seq_base=offset * 10_000,
                scoring_run_id=(
                    telemetry_session.run.scoring_run_id
                    if telemetry_session is not None and telemetry_session.run is not None
                    else ""
                ),
            )
            if queue_generation_id:
                ctx["telemetry_session"] = telemetry_session
                ctx["queue_generation_id"] = queue_generation_id
                # The context is built before telemetry allocation. If that
                # best-effort allocation failed, restore the legacy trace sink
                # so provider spend is still persisted for the fresh queue
                # generation (the cache-hit hydration path does this too).
                if telemetry_session is None:
                    ctx["trace_sink"] = self._candidate_incontainer_trace_sink(
                        cid,
                        persist_costs=True,
                    )
                ctx_cache[queue_generation_id] = ctx
                if telemetry_session is not None:
                    jobs = await select_many(
                        global_icp_queue.JOB_TABLE,
                        filters=(("queue_generation_id", queue_generation_id),),
                        order_by=(("phase", False), ("item_index", False)),
                        limit=1000,
                    )
                    for job in jobs:
                        await telemetry_session.plan(
                            icp_ref=str(job.get("icp_ref") or ""),
                            icp_hash=str(
                                ctx["items_by_ref"]
                                .get(str(job.get("icp_ref") or ""), {})
                                .get("icp_hash")
                                or ""
                            ),
                            icp_ordinal=int(job.get("item_index") or 0),
                            model_role="candidate",
                            phase=str(job.get("phase") or "all"),
                            held=str(job.get("phase") or "") in {"private", "conditional"},
                            source_job_id=str(job.get("job_id") or ""),
                        )
                    await emit_run_event(telemetry_session.run, "assigned")
                    await emit_run_event(telemetry_session.run, "started")
                await create_scoring_dispatch_event(
                    dispatch_type="candidate_scoring",
                    dispatch_status="assigned",
                    worker_ref=self.worker_ref,
                    proxy_ref_hash=self.proxy_ref_hash,
                    candidate_id=cid,
                    run_id=str(row.get("run_id") or ""),
                    ticket_id=str(row.get("ticket_id") or ""),
                    scoring_id=(
                        telemetry_session.run.scoring_id
                        if telemetry_session is not None and telemetry_session.run is not None
                        else None
                    ),
                    scoring_run_id=(
                        telemetry_session.run.scoring_run_id
                        if telemetry_session is not None and telemetry_session.run is not None
                        else None
                    ),
                    event_doc={
                        "queue_generation_id": queue_generation_id,
                        "rolling_window_hash": ctx["window"].window_hash,
                    },
                )
                await create_candidate_evaluation_event(
                    candidate_id=cid,
                    run_id=str(row.get("run_id") or ""),
                    ticket_id=str(row.get("ticket_id") or ""),
                    event_type="evaluating",
                    candidate_status="evaluating",
                    evaluator_ref=self.worker_ref,
                    reason="global_icp_queue_enqueued",
                    event_doc={
                        "worker_ref": self.worker_ref,
                        "scored_via": "global_icp_queue",
                        "queue_generation_id": queue_generation_id,
                    },
                )
            elif telemetry_session is not None:
                await emit_run_event(
                    telemetry_session.run,
                    "cancelled",
                    failure_category="queue_generation_lost",
                )

        assembled: list[str] = []
        queue_heartbeat_tasks: dict[str, tuple[asyncio.Event, asyncio.Task[Any]]] = {}

        async def score_icp(job: Mapping[str, Any]) -> dict[str, Any]:
            candidate_id = str(job.get("candidate_id") or "")
            queue_generation_id = str(job.get("queue_generation_id") or "")
            ctx = await _ctx(
                candidate_id,
                queue_generation_id,
                str(job.get("scoring_run_id") or ""),
            )
            item = _queue_scoring_item(ctx, job)
            if ctx is None or item is None:
                return {}
            telemetry_session = ctx.get("telemetry_session")
            if not isinstance(telemetry_session, ScoringTelemetrySession):
                telemetry_session = None
            queue_attempt_base = max(0, int(job.get("attempt_count") or 1) - 1) * 100

            async def _queue_lifecycle(
                action: str,
                payload: Mapping[str, Any],
            ) -> Mapping[str, Any] | None:
                if telemetry_session is None:
                    return None
                shifted = {
                    **dict(payload),
                    "icp_ref": str(job.get("icp_ref") or ""),
                    "icp_hash": str(item.get("icp_hash") or ""),
                    "icp_ordinal": int(job.get("item_index") or 0),
                    "model_role": "candidate",
                    "phase": str(job.get("phase") or "all"),
                    "source_job_id": str(job.get("job_id") or ""),
                    "retry_round": queue_attempt_base
                    + max(0, int(payload.get("retry_round") or 0)),
                }
                return await telemetry_session.lifecycle(action, shifted)

            async def _queue_attempt_cost_sink(
                icp_ref: str,
                entries: list[dict[str, Any]],
                execution_context: Mapping[str, Any],
            ) -> None:
                persisted = await _persist_provider_cost_events(
                    entries=entries,
                    run_type="candidate_scoring",
                    icp_ref=icp_ref,
                    icp_hash=str(item.get("icp_hash") or ""),
                    runner_role="candidate",
                    candidate_id=candidate_id,
                    epoch_id=int(ctx["run_context"].get("evaluation_epoch") or 0),
                    rolling_window_hash=str(job.get("window_hash") or ""),
                    scoring_id=str(execution_context.get("scoring_id") or ""),
                    scoring_run_id=str(execution_context.get("scoring_run_id") or ""),
                    icp_execution_id=str(execution_context.get("icp_execution_id") or ""),
                )
                if not persisted and telemetry_session is not None:
                    telemetry_session.degraded = True

            if telemetry_session is not None:
                heartbeat_stop = asyncio.Event()
                heartbeat_task = asyncio.create_task(
                    telemetry_session.heartbeat_loop(heartbeat_stop)
                )
                queue_heartbeat_tasks[str(job.get("job_id") or "")] = (
                    heartbeat_stop,
                    heartbeat_task,
                )
            receipt_hashes_before = {
                str(item.get("receipt_hash") or "")
                for item in _attested_receipts_from(ctx["runner"], ctx["scorer"])
                if item.get("receipt_hash")
            }
            scorer_outcome_count_before = _attested_outcome_count(ctx["scorer"])
            results = await score_private_model_pair_items(
                benchmark_items=[item],
                base_runner=None,
                candidate_runner=ctx["runner"],
                company_scorer=ctx["scorer"],
                run_context=ctx["run_context"],
                image_candidate=True,
                runtime_patch=None,
                parent_freshness_check=None,
                trace_sink=ctx["trace_sink"],
                scoring_telemetry_hook=_queue_lifecycle if telemetry_session is not None else None,
                attempt_cost_sink=(
                    _queue_attempt_cost_sink if telemetry_session is not None else None
                ),
            )
            if str(job.get("phase") or "") == "conditional":
                if len(results) != 1:
                    raise ConditionalValidationRetryableError(
                        "conditional_validation_incomplete:result_count_mismatch"
                    )
                result_ref = str(
                    results[0].get("icp_ref") or results[0].get("icp_hash") or ""
                )
                if result_ref != str(job.get("icp_ref") or ""):
                    raise ConditionalValidationRetryableError(
                        "conditional_validation_incomplete:result_identity_mismatch"
                    )
                conditional_failures = _critical_measurement_failures(results)
                if conditional_failures:
                    raise ConditionalValidationRetryableError(
                        "conditional_validation_incomplete:"
                        + ",".join(conditional_failures)
                    )
            result_doc = dict(results[0]) if results else {}
            if not legacy_v1_enabled():
                receipt_hashes_after = {
                    str(item.get("receipt_hash") or "")
                    for item in _attested_receipts_from(
                        ctx["runner"],
                        ctx["scorer"],
                    )
                    if item.get("receipt_hash")
                }
                new_receipt_hashes = _queue_current_attested_receipt_hashes(
                    scorer=ctx["scorer"],
                    scorer_outcome_count_before=scorer_outcome_count_before,
                    receipt_hashes_before=receipt_hashes_before,
                    receipt_hashes_after=receipt_hashes_after,
                )
                result_doc[
                    global_icp_queue.ATTESTED_RECEIPT_HASHES_FIELD
                ] = new_receipt_hashes
            return result_doc

        async def _job_completed(
            job: Mapping[str, Any],
            result_doc: Mapping[str, Any],
            failed: bool,
            committed: bool,
            error: BaseException | None,
        ) -> None:
            heartbeat = queue_heartbeat_tasks.pop(str(job.get("job_id") or ""), None)
            if heartbeat is not None:
                heartbeat[0].set()
                await heartbeat[1]
            ctx = await _ctx(
                str(job.get("candidate_id") or ""),
                str(job.get("queue_generation_id") or ""),
                str(job.get("scoring_run_id") or ""),
            )
            telemetry_session = ctx.get("telemetry_session") if ctx is not None else None
            if not isinstance(telemetry_session, ScoringTelemetrySession):
                return
            execution = telemetry_session.execution_for(
                icp_ref=str(job.get("icp_ref") or ""),
                model_role="candidate",
            )
            if not committed:
                await emit_icp_event(
                    execution,
                    "cancelled",
                    failure_category="queue_result_cas_lost",
                    error=error,
                )
                if execution is not None:
                    telemetry_session.terminal_execution_ids.add(execution.icp_execution_id)
                return
            if failed:
                failure_class, retryable = (
                    _candidate_scoring_failure_class(error)
                    if error is not None
                    else ("queue_job_failed", False)
                )
                await emit_icp_event(
                    execution,
                    "failed",
                    retryable=retryable,
                    failure_category=failure_class,
                    error=error,
                )
                if execution is not None:
                    telemetry_session.terminal_execution_ids.add(execution.icp_execution_id)
                return
            await telemetry_session.complete_result(
                result_doc,
                model_role="candidate",
                checkpoint_ref=opaque_checkpoint_ref(
                    str(job.get("queue_generation_id") or ""),
                    str(job.get("job_id") or ""),
                ),
                checkpoint_hash=canonical_hash(dict(result_doc)),
                checkpoint_persisted=True,
                outcome="global_queue_job_done",
            )

        async def _stale_job_recovered(job: Mapping[str, Any]) -> None:
            telemetry_session = await load_scoring_session(str(job.get("scoring_run_id") or ""))
            if telemetry_session is None:
                return
            execution = telemetry_session.execution_for(
                icp_ref=str(job.get("icp_ref") or ""),
                model_role="candidate",
            )
            await emit_icp_event(
                execution,
                "cancelled",
                failure_category="queue_lease_expired",
            )

        async def _gate_decided(queue_generation_id: str, decision: str) -> None:
            ctx = await _ctx("", queue_generation_id)
            if ctx is None:
                candidate_row = await select_one(
                    global_icp_queue.CANDIDATE_TABLE,
                    filters=(("queue_generation_id", queue_generation_id),),
                )
                if candidate_row is None:
                    return
                ctx = await _ctx(
                    str(candidate_row.get("candidate_id") or ""),
                    queue_generation_id,
                    str(candidate_row.get("scoring_run_id") or ""),
                )
            telemetry_session = ctx.get("telemetry_session") if ctx is not None else None
            if not isinstance(telemetry_session, ScoringTelemetrySession):
                return
            private_jobs = await select_many(
                global_icp_queue.JOB_TABLE,
                filters=(("queue_generation_id", queue_generation_id), ("phase", "private")),
                limit=1000,
            )
            for private_job in private_jobs:
                if decision == "rejected":
                    await telemetry_session.skip_unstarted(
                        icp_ref=str(private_job.get("icp_ref") or ""),
                        model_role="candidate",
                        failure_category="public_gate_rejected",
                    )
                else:
                    await emit_icp_event(
                        telemetry_session.execution_for(
                            icp_ref=str(private_job.get("icp_ref") or ""),
                            model_role="candidate",
                            retry_round=0,
                        ),
                        "queued",
                        event_ordinal=1,
                        event_doc={"released_from_hold": True},
                    )

            conditional_jobs = await select_many(
                global_icp_queue.JOB_TABLE,
                filters=(("queue_generation_id", queue_generation_id), ("phase", "conditional")),
                limit=1000,
            )
            if decision == "rejected":
                for conditional_job in conditional_jobs:
                    await telemetry_session.skip_unstarted(
                        icp_ref=str(conditional_job.get("icp_ref") or ""),
                        model_role="candidate",
                        failure_category="public_gate_rejected",
                    )

        async def _preliminary_gate_authorizer(
            queue_generation_id: str,
            claim: Mapping[str, Any],
            docs: Mapping[str, list],
            preliminary_score: float,
        ) -> Mapping[str, Any]:
            ctx = await _ctx(
                str(claim.get("candidate_id") or ""),
                queue_generation_id,
                str(claim.get("scoring_run_id") or ""),
            )
            if ctx is None:
                raise ConditionalValidationRetryableError(
                    "conditional_queue_preliminary_context_missing"
                )
            preliminary_results, preliminary_gate = build_holdout_gate_result(
                public_results=docs.get("public") or (),
                private_results=docs.get("private") or (),
                conditional_results=(),
                public_icp_count=len(ctx["public_items"]),
                private_icp_count=len(ctx["private_items"]),
                conditional_icp_count=len(ctx["conditional_items"]),
                gate=ctx["gate"],
            )
            measured_score = _safe_float(
                preliminary_gate.get("candidate_preliminary_score"),
                default=-1.0,
            )
            if abs(measured_score - float(preliminary_score)) > 1e-6:
                raise ConditionalValidationRetryableError(
                    "conditional_queue_preliminary_score_commitment_mismatch"
                )
            await self._check_candidate_scoring_freshness(
                parent_artifact=ctx["artifact"],
                candidate_window_hash=ctx["window"].window_hash,
                progress={
                    "phase": "conditional_preliminary_gate",
                    "completed_icp_count": len(preliminary_results),
                },
            )
            return await _authorize_conditional_preliminary_gate(
                config=self.config,
                evaluation_epoch=int(ctx["run_context"]["evaluation_epoch"]),
                candidate=ctx["candidate"],
                artifact=ctx["artifact"],
                benchmark=ctx["benchmark"],
                patch=ctx["patch"],
                candidate_artifact=ctx["candidate_artifact"],
                preliminary_results=preliminary_results,
                run_context={**ctx["run_context"], "signature_ref": "pending"},
                policy=_stored_daily_baseline_evaluation_policy(
                    self._preliminary_evaluation_policy(),
                    ctx["gate"],
                ),
                preliminary_gate=preliminary_gate,
                parent_receipts=_queue_attested_parent_receipts(
                    docs,
                    ctx["runner"],
                    ctx["scorer"],
                ),
            )

        async def _preliminary_gate_error(
            queue_generation_id: str,
            claim: Mapping[str, Any],
            error: BaseException,
        ) -> bool:
            failure_class, _retryable = _candidate_scoring_failure_class(error)
            attempt = int(claim.get("preliminary_gate_attempt_count") or 0)
            if isinstance(error, (StaleParentDuringScoring, CandidateBaselineWindowChanged)):
                cancelled = await global_icp_queue.cancel_preliminary_gate_for_rebase(
                    queue_generation_id=queue_generation_id,
                    expected_claimed_by=self.worker_ref,
                    expected_attempt_count=attempt,
                    failure_class=failure_class,
                )
                if not cancelled:
                    return False
                ctx = await _ctx(
                    str(claim.get("candidate_id") or ""),
                    "",
                    str(claim.get("scoring_run_id") or ""),
                )
                if ctx is None:
                    return True
                candidate = ctx["candidate"]
                if isinstance(error, StaleParentDuringScoring):
                    await self._queue_stale_parent_rebase(
                        candidate,
                        active_artifact=error.active_artifact,
                        candidate_parent=error.candidate_parent,
                        evaluation_epoch=int(ctx["run_context"]["evaluation_epoch"]),
                        elapsed_seconds=0.0,
                        stage="global_queue_preliminary_parent_changed",
                        stale_progress=error.progress,
                    )
                else:
                    await create_candidate_evaluation_event(
                        candidate_id=str(candidate["candidate_id"]),
                        run_id=str(candidate["run_id"]),
                        ticket_id=str(candidate["ticket_id"]),
                        event_type="queued",
                        candidate_status="queued",
                        evaluator_ref=self.worker_ref,
                        reason="baseline_not_ready",
                        event_doc={
                            "queue_generation_id": queue_generation_id,
                            **_candidate_baseline_wait_event_doc(error),
                        },
                    )
                return True

            await create_conditional_validation_event(
                candidate_id=str(claim.get("candidate_id") or ""),
                event_type="retryable_failure",
                assignment_hash=str(claim.get("category_assignment_hash") or ""),
                policy_hash=str(claim.get("conditional_policy_hash") or ""),
                rolling_window_hash=str(claim.get("window_hash") or ""),
                baseline_benchmark_bundle_id=str(
                    claim.get("baseline_benchmark_bundle_id") or ""
                ),
                source_ref=(
                    f"queue:{queue_generation_id}:preliminary_attempt:{attempt}:authority"
                ),
                threshold_points=_safe_float(
                    claim.get("threshold_points"),
                    default=0.0,
                ),
                queue_generation_id=queue_generation_id,
                failure_class=failure_class,
                event_doc={
                    "path": "global_icp_queue",
                    "failure_class": failure_class,
                    "retryable": True,
                    "preliminary_gate_attempt_count": attempt,
                    "conditional_jobs_released": False,
                },
            )
            return False

        async def _preliminary_gate_decided(
            queue_generation_id: str,
            decision: str,
        ) -> None:
            ctx = await _ctx("", queue_generation_id)
            if ctx is None:
                return
            telemetry_session = ctx.get("telemetry_session")
            if isinstance(telemetry_session, ScoringTelemetrySession):
                conditional_jobs = await select_many(
                    global_icp_queue.JOB_TABLE,
                    filters=(
                        ("queue_generation_id", queue_generation_id),
                        ("phase", "conditional"),
                    ),
                    limit=1000,
                )
                for conditional_job in conditional_jobs:
                    icp_ref = str(conditional_job.get("icp_ref") or "")
                    if decision == "rejected":
                        await telemetry_session.skip_unstarted(
                            icp_ref=icp_ref,
                            model_role="candidate",
                            failure_category="preliminary_gate_rejected",
                        )
                    else:
                        await emit_icp_event(
                            telemetry_session.execution_for(
                                icp_ref=icp_ref,
                                model_role="candidate",
                                retry_round=0,
                            ),
                            "queued",
                            event_ordinal=1,
                            event_doc={"released_from_preliminary_hold": True},
                        )
            candidate = ctx.get("candidate") if isinstance(ctx.get("candidate"), Mapping) else {}
            await create_candidate_evaluation_event(
                candidate_id=str(candidate.get("candidate_id") or ""),
                run_id=str(candidate.get("run_id") or ""),
                ticket_id=str(candidate.get("ticket_id") or ""),
                event_type="evaluating",
                candidate_status="evaluating",
                evaluator_ref=self.worker_ref,
                reason=(
                    "conditional_validation_started"
                    if decision == "passed"
                    else "conditional_validation_preliminary_rejected"
                ),
                event_doc={
                    "queue_generation_id": queue_generation_id,
                    "preliminary_gate_status": decision,
                    "category_assignment_hash": str(
                        ctx["gate"].get("category_assignment_hash") or ""
                    ),
                },
            )

        async def _candidate_assembled(queue_generation_id: str, candidate_id: str) -> None:
            ctx = await _ctx(candidate_id, queue_generation_id)
            telemetry_session = ctx.get("telemetry_session") if ctx is not None else None
            if not isinstance(telemetry_session, ScoringTelemetrySession):
                return
            await emit_run_event(
                telemetry_session.run,
                "completed",
                score_bundle_id=str(ctx.get("score_bundle_id") or ""),
                telemetry_degraded=telemetry_session.degraded,
                event_doc={"queue_generation_id": queue_generation_id},
            )

        def compute_public_score(public_docs: Sequence[Mapping[str, Any]]) -> float:
            return float(_queue_benchmark_style_score(public_docs, "candidate_company_scores"))

        def compute_preliminary_score(preliminary_docs: Sequence[Mapping[str, Any]]) -> float:
            return float(
                _queue_benchmark_style_score(
                    preliminary_docs,
                    "candidate_company_scores",
                )
            )

        async def assemble(
            queue_generation_id: str,
            candidate_id: str,
            docs: Mapping[str, Any],
        ) -> None:
            ctx = await _ctx(candidate_id, queue_generation_id)
            if ctx is None:
                raise RuntimeError("assemble context missing")
            await self._queue_assemble_candidate(candidate_id, docs, ctx)
            assembled.append(candidate_id)

        counters = await global_icp_queue.run_queue_scoring_pass(
            worker_ref=self.worker_ref,
            lease_seconds=self.config.scoring_worker_model_timeout_seconds + 60,
            score_icp=score_icp,
            compute_public_score=compute_public_score,
            compute_preliminary_score=compute_preliminary_score,
            preliminary_gate_authorizer=_preliminary_gate_authorizer,
            assemble_candidate=assemble,
            job_completed=_job_completed,
            stale_job_recovered=_stale_job_recovered,
            gate_decided=_gate_decided,
            preliminary_gate_decided=_preliminary_gate_decided,
            preliminary_gate_error=_preliminary_gate_error,
            retryable_job_error=_queue_job_error_is_retryable,
            candidate_assembled=_candidate_assembled,
        )
        logger.info(
            "research_lab_global_icp_queue_pass worker_ref=%s counters=%s assembled=%s",
            self.worker_ref,
            counters,
            len(assembled),
        )
        return assembled

    async def _claim_next_candidate(self) -> dict[str, Any] | None:
        rows = await select_many(
            "research_lab_candidate_evaluation_current",
            columns="*",
            filters=(("current_candidate_status", "queued"),),
            order_by=(("current_status_at", False),),
            limit=50,
        )
        candidate: dict[str, Any] | None = None
        for row in rows:
            reason = str(row.get("current_reason") or "")
            status_at = row.get("current_status_at")
            if reason == "baseline_not_ready" and not _status_is_stale(
                status_at,
                self.config.scoring_worker_baseline_not_ready_retry_seconds,
            ):
                continue
            if reason in {
                "candidate_scoring_retryable_failure",
                "conditional_validation_retryable_failure",
            } and not _status_is_stale(
                status_at,
                self.config.scoring_worker_retryable_failure_retry_seconds,
            ):
                continue
            candidate = dict(row)
            break
        if not candidate:
            if rows:
                logger.info(
                    "research_lab_candidate_claim_deferred worker_ref=%s queued_candidates=%s",
                    self.worker_ref,
                    len(rows),
                )
            return None
        start_gate = await self._candidate_scoring_start_gate()
        self._last_candidate_start_gate = start_gate
        if not start_gate.get("available"):
            hold_key = ":".join(
                str(start_gate.get(key) or "")
                for key in ("reason", "target_benchmark_date", "private_model_manifest_hash")
            )
            if self._candidate_start_hold_logged_key != hold_key:
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE SCORING START HELD",
                        (
                            ("Worker", self.worker_ref),
                            ("Queued candidates", len(rows)),
                            ("Reason", start_gate.get("reason")),
                            ("UTC now", start_gate.get("now_utc")),
                            ("Target baseline date", start_gate.get("target_benchmark_date")),
                            ("Quiet start seconds", start_gate.get("quiet_start_utc_seconds")),
                            ("Private model", compact_ref(start_gate.get("private_model_manifest_hash"))),
                            ("Action", "leaving queued candidates untouched"),
                        ),
                    )
                )
                self._candidate_start_hold_logged_key = hold_key
            return None
        self._candidate_start_hold_logged_key = None
        candidate_id = str(candidate.get("candidate_id") or "")
        fresh = await select_one(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id,current_candidate_status",
            filters=(("candidate_id", candidate_id),),
        )
        if not fresh or fresh.get("current_candidate_status") != "queued":
            return None
        try:
            assigned_event = await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="assigned",
                candidate_status="assigned",
                evaluator_ref=self.worker_ref,
                reason="assigned_to_gateway_qualification_worker",
                event_doc={
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                },
            )
        except Exception as exc:
            if _is_candidate_claim_race_error(exc):
                logger.info(
                    "research_lab_candidate_claim_race candidate_id=%s worker_ref=%s",
                    compact_ref(candidate_id),
                    self.worker_ref,
                )
                return None
            raise
        assigned_current = await select_one(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id,current_candidate_status,current_evaluator_ref,current_event_hash",
            filters=(("candidate_id", candidate_id),),
        )
        if (
            not assigned_current
            or assigned_current.get("current_candidate_status") != "assigned"
            or assigned_current.get("current_evaluator_ref") != self.worker_ref
            or assigned_current.get("current_event_hash") != assigned_event.get("anchored_hash")
        ):
            logger.info(
                "research_lab_candidate_claim_lost candidate_id=%s worker_ref=%s",
                compact_ref(candidate_id),
                self.worker_ref,
            )
            return None
        await safe_project_public_loop_activity(
            str(candidate["ticket_id"]),
            source_ref=f"candidate_assigned:{candidate_id}",
            reason="assigned_to_gateway_qualification_worker",
            config=self.config,
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB CANDIDATE ALLOCATED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Run", compact_ref(candidate.get("run_id"))),
                    ("Ticket", compact_ref(candidate.get("ticket_id"))),
                    ("Proxy ref", self.proxy_ref_hash),
                ),
            )
        )
        return candidate

    # SLA after which a stale-parent candidate with no rebase should be alerted on.
    _STALE_PARENT_REBASE_SLA_SECONDS = 3600

    async def _alert_stuck_candidates(self) -> None:
        """Observability alerts (structured logs) for candidates stuck beyond their
        expected windows. Read-only; never mutates. Best-effort (swallows query errors).
        """
        if self.config.scoring_worker_index != 0:
            return
        try:
            try:
                retry_seconds = int(self.config.scoring_worker_baseline_not_ready_retry_seconds or 900)
            except (TypeError, ValueError):
                retry_seconds = 900
            baseline_alert_after = max(300, retry_seconds * 4)
            baseline_rows = await select_many(
                "research_lab_candidate_evaluation_current",
                columns="candidate_id,current_status_at,current_reason,current_candidate_status",
                filters=(
                    ("current_candidate_status", "queued"),
                    ("current_reason", "baseline_not_ready"),
                ),
                limit=200,
            )
            stuck_baseline = [
                str(r.get("candidate_id") or "")
                for r in baseline_rows
                if _status_is_stale(r.get("current_status_at"), baseline_alert_after)
            ]
            if stuck_baseline:
                logger.warning(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATES STUCK WAITING FOR BASELINE",
                        (
                            ("Worker", self.worker_ref),
                            ("Count", len(stuck_baseline)),
                            ("Beyond", f"{baseline_alert_after}s"),
                            ("Candidates", ", ".join(compact_ref(c) for c in stuck_baseline[:10])),
                        ),
                    )
                )
                logger.warning(
                    "research_lab_candidates_stuck_baseline_not_ready count=%s threshold_seconds=%s",
                    len(stuck_baseline),
                    baseline_alert_after,
                )

            sla_seconds = int(
                getattr(self.config, "stale_parent_rebase_sla_seconds", None)
                or self._STALE_PARENT_REBASE_SLA_SECONDS
            )
            stale_parent_rows = await select_many(
                "research_lab_candidate_evaluation_current",
                columns="candidate_id,ticket_id,current_status_at",
                filters=(
                    ("current_candidate_status", "rejected"),
                    ("current_reason", "stale_parent_needs_rescore"),
                ),
                limit=200,
            )
            overdue: list[str] = []
            for row in stale_parent_rows:
                if not _status_is_stale(row.get("current_status_at"), sla_seconds):
                    continue
                candidate_id = str(row.get("candidate_id") or "")
                if not candidate_id:
                    continue
                existing = await select_many(
                    "research_lab_candidate_promotion_events",
                    columns="promotion_event_id",
                    filters=(("candidate_id", candidate_id), ("event_type", "rebase_queued")),
                    limit=1,
                )
                if existing:
                    continue
                ticket_id = str(row.get("ticket_id") or "")
                if ticket_id:
                    regeneration_events = await select_many(
                        "research_loop_run_queue_events",
                        columns="run_id,event_doc",
                        filters=(
                            ("ticket_id", ticket_id),
                            ("reason", "regenerate_after_rebase_unavailable"),
                        ),
                        limit=200,
                    )
                    if any(
                        str(
                            (
                                event.get("event_doc")
                                if isinstance(event.get("event_doc"), Mapping)
                                else {}
                            ).get("regenerated_from_candidate_id")
                            or ""
                        )
                        == candidate_id
                        for event in regeneration_events
                    ):
                        continue
                overdue.append(candidate_id)
            if overdue:
                overdue_key = sha256_json({"candidates": sorted(overdue), "sla_seconds": sla_seconds})
                if overdue_key not in self._stale_parent_overdue_warning_keys:
                    self._stale_parent_overdue_warning_keys.add(overdue_key)
                    logger.warning(
                        format_worker_block(
                            "RESEARCH LAB STALE-PARENT CANDIDATES NOT REBASED WITHIN SLA",
                            (
                                ("Worker", self.worker_ref),
                                ("Count", len(overdue)),
                                ("SLA", f"{sla_seconds}s"),
                                ("Candidates", ", ".join(compact_ref(c) for c in overdue[:10])),
                            ),
                        )
                    )
                    logger.warning(
                        "research_lab_stale_parent_candidates_overdue count=%s sla_seconds=%s",
                        len(overdue),
                        sla_seconds,
                    )
        except Exception as exc:  # noqa: BLE001 - alerting must never break the worker pass
            logger.warning("research_lab_stuck_candidate_alert_failed error=%s", str(exc)[:200])

    async def _recover_stale_candidate_claims(self) -> int:
        stale_after_seconds = max(120, int(self.config.scoring_worker_model_timeout_seconds or 900) + 60)
        rows: list[dict[str, Any]] = []
        for status in ("assigned", "evaluating"):
            rows.extend(
                await select_many(
                    "research_lab_candidate_evaluation_current",
                    columns=(
                        "candidate_id,run_id,ticket_id,receipt_id,current_candidate_status,current_status_at,"
                        "current_evaluator_ref,current_event_hash,private_model_manifest_doc,"
                        "candidate_model_manifest_doc,candidate_artifact_hash"
                    ),
                    filters=(("current_candidate_status", status),),
                    # Oldest first: under backlog the stalest claims must be
                    # recovered before the limit truncates the scan.
                    order_by=(("current_status_at", False),),
                    limit=50,
                )
            )
        recovered = 0
        for row in rows:
            candidate_id = str(row.get("candidate_id") or "")
            run_id = str(row.get("run_id") or "")
            ticket_id = str(row.get("ticket_id") or "")
            if not candidate_id or not run_id or not ticket_id:
                continue
            recovery_reason = _candidate_claim_recovery_reason(
                row,
                candidate_id=candidate_id,
                worker_ref=self.worker_ref,
                worker_index=self.config.scoring_worker_index,
                total_workers=self.config.scoring_worker_total_workers,
                worker_started_at=self._worker_started_at,
                stale_after_seconds=stale_after_seconds,
                restart_orphan_grace_seconds=_env_int(
                    "RESEARCH_LAB_SCORING_RESTART_ORPHAN_GRACE_SECONDS",
                    5,
                ),
            )
            if recovery_reason is None:
                continue
            claim_attempts = await self._candidate_claim_attempt_count(candidate_id)
            max_attempts = int(self.config.scoring_worker_max_claim_requeues)
            if claim_attempts >= max_attempts:
                progress_summary = await self._candidate_scoring_progress_summary(row)
                completed_icps = _completed_icp_count_from_progress_doc(progress_summary)
                if completed_icps > 0:
                    try:
                        await create_candidate_evaluation_event(
                            candidate_id=candidate_id,
                            run_id=run_id,
                            ticket_id=ticket_id,
                            event_type="queued",
                            candidate_status="queued",
                            evaluator_ref=self.worker_ref,
                            reason="stale_gateway_scoring_progress_requeued",
                            event_doc={
                                "recovering_worker_ref": self.worker_ref,
                                "previous_evaluator_ref": row.get("current_evaluator_ref"),
                                "previous_candidate_status": row.get("current_candidate_status"),
                                "previous_event_hash": row.get("current_event_hash"),
                                "previous_status_at": row.get("current_status_at"),
                                "stale_after_seconds": stale_after_seconds,
                                "worker_started_at": self._worker_started_at.isoformat(),
                                "recovery_reason": recovery_reason,
                                "claim_attempts": claim_attempts,
                                "max_claim_attempts": max_attempts,
                                "retry_budget_preserved": True,
                                "scoring_progress": progress_summary,
                            },
                        )
                        await create_scoring_dispatch_event(
                            dispatch_type="candidate_scoring",
                            dispatch_status="completed",
                            worker_ref=self.worker_ref,
                            proxy_ref_hash=self.proxy_ref_hash,
                            candidate_id=candidate_id,
                            run_id=run_id,
                            ticket_id=ticket_id,
                            event_doc={
                                "reason": "stale_gateway_scoring_progress_requeued",
                                "dispatch_context": "candidate_scoring_recovery",
                                "retry_budget_preserved": True,
                                "recovery_reason": recovery_reason,
                                "claim_attempts": claim_attempts,
                                "max_claim_attempts": max_attempts,
                                "scoring_progress": progress_summary,
                            },
                        )
                        recovered += 1
                    except Exception as exc:
                        logger.warning(
                            "research_lab_stale_candidate_progress_requeue_failed candidate_id=%s error=%s",
                            compact_ref(candidate_id),
                            str(exc)[:240],
                        )
                        continue
                    logger.info(
                        "research_lab_stale_candidate_progress_requeued candidate_id=%s completed_icps=%s claim_attempts=%s/%s",
                        compact_ref(candidate_id),
                        completed_icps,
                        claim_attempts,
                        max_attempts,
                    )
                    continue
                try:
                    await create_candidate_evaluation_event(
                        candidate_id=candidate_id,
                        run_id=run_id,
                        ticket_id=ticket_id,
                        event_type="failed",
                        candidate_status="failed",
                        evaluator_ref=self.worker_ref,
                        reason="stale_gateway_scoring_retry_limit_exceeded",
                        event_doc={
                            "failure_class": "stale_claim_retry_limit_exceeded",
                            "retryable": False,
                            "recovering_worker_ref": self.worker_ref,
                            "previous_evaluator_ref": row.get("current_evaluator_ref"),
                            "previous_candidate_status": row.get("current_candidate_status"),
                            "previous_event_hash": row.get("current_event_hash"),
                            "previous_status_at": row.get("current_status_at"),
                            "stale_after_seconds": stale_after_seconds,
                            "worker_started_at": self._worker_started_at.isoformat(),
                            "recovery_reason": recovery_reason,
                            "claim_attempts": claim_attempts,
                            "max_claim_attempts": max_attempts,
                        },
                    )
                    await create_scoring_dispatch_event(
                        # NOTE: 'candidate_scoring' (not a '_recovery' variant) —
                        # the dispatch_type DB CHECK only allows the base types.
                        dispatch_type="candidate_scoring",
                        dispatch_status="failed",
                        worker_ref=self.worker_ref,
                        proxy_ref_hash=self.proxy_ref_hash,
                        candidate_id=candidate_id,
                        run_id=run_id,
                        ticket_id=ticket_id,
                        event_doc={
                            "reason": "stale_gateway_scoring_retry_limit_exceeded",
                            "dispatch_context": "candidate_scoring_recovery",
                            "failure_class": "stale_claim_retry_limit_exceeded",
                            "retryable": False,
                            "recovery_reason": recovery_reason,
                            "claim_attempts": claim_attempts,
                            "max_claim_attempts": max_attempts,
                        },
                    )
                    recovered += 1
                except Exception as exc:
                    logger.warning(
                        "research_lab_stale_candidate_fail_limit_failed candidate_id=%s error=%s",
                        compact_ref(candidate_id),
                        str(exc)[:240],
                    )
                    continue
                # This terminal path must mirror the normal terminal path:
                # finalize the receipt and project the public card, or the
                # card freezes at `scoring` forever (bug #11).
                try:
                    await self._maybe_finalize_candidate_receipt(row)
                except Exception:
                    logger.exception(
                        "research_lab_stale_candidate_receipt_finalize_failed candidate_id=%s",
                        compact_ref(candidate_id),
                    )
                try:
                    await safe_project_public_loop_activity(
                        ticket_id,
                        source_ref=f"candidate_stale_claim_retry_limit:{candidate_id}",
                        reason="stale_gateway_scoring_retry_limit_exceeded",
                        config=self.config,
                    )
                except Exception:
                    logger.exception(
                        "research_lab_stale_candidate_projection_failed candidate_id=%s",
                        compact_ref(candidate_id),
                    )
                continue
            try:
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=run_id,
                    ticket_id=ticket_id,
                    event_type="queued",
                    candidate_status="queued",
                    evaluator_ref=self.worker_ref,
                    reason="stale_gateway_scoring_requeued",
                    event_doc={
                        "recovering_worker_ref": self.worker_ref,
                        "previous_evaluator_ref": row.get("current_evaluator_ref"),
                        "previous_candidate_status": row.get("current_candidate_status"),
                        "previous_event_hash": row.get("current_event_hash"),
                        "previous_status_at": row.get("current_status_at"),
                        "stale_after_seconds": stale_after_seconds,
                        "worker_started_at": self._worker_started_at.isoformat(),
                        "recovery_reason": recovery_reason,
                    },
                )
                recovered += 1
            except Exception as exc:
                logger.warning(
                    "research_lab_stale_candidate_requeue_failed candidate_id=%s error=%s",
                    compact_ref(candidate_id),
                    str(exc)[:240],
                )
        if recovered:
            logger.info(
                "research_lab_stale_candidates_requeued worker_ref=%s count=%s stale_after_seconds=%s",
                self.worker_ref,
                recovered,
                stale_after_seconds,
            )
        return recovered

    async def _candidate_claim_attempt_count(self, candidate_id: str) -> int:
        rows = await select_many(
            "research_lab_candidate_evaluation_events",
            columns="candidate_id,event_type,candidate_status,reason,seq",
            filters=(
                ("candidate_id", candidate_id),
                # Heartbeats write an `evaluating` event every ~120s; without
                # this filter the seq-ascending page fills with heartbeats and
                # later assigned events fall outside the limit.
                ("event_type", "in", ("assigned", "queued")),
            ),
            order_by=(("seq", False),),
            limit=100,
        )
        return _count_claim_attempts(rows)

    async def _candidate_scoring_progress_summary(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return _safe_scoring_progress_summary(source="none", completed_icp_count=0)
        event_rows = await select_many(
            "research_lab_candidate_evaluation_events",
            columns="candidate_id,event_type,reason,event_doc,seq",
            filters=(
                ("candidate_id", candidate_id),
                ("event_type", "in", ("evaluating", "queued", "failed", "rejected")),
            ),
            order_by=(("seq", True),),
            limit=80,
        )
        from_events = _latest_scoring_progress_from_events(event_rows)
        if _completed_icp_count_from_progress_doc(from_events) > 0:
            return from_events

        private_manifest = candidate.get("private_model_manifest_doc")
        manifest_uri = (
            str(private_manifest.get("manifest_uri") or "")
            if isinstance(private_manifest, Mapping)
            else ""
        )
        location = _scoring_progress_s3_prefix(manifest_uri, candidate_id)
        if location is None:
            return from_events

        candidate_manifest = candidate.get("candidate_model_manifest_doc")
        candidate_artifact_hash = (
            str(candidate_manifest.get("model_artifact_hash") or "")
            if isinstance(candidate_manifest, Mapping)
            else str(candidate.get("candidate_artifact_hash") or "")
        )
        bucket, object_prefix = location
        try:
            from_s3 = await asyncio.to_thread(
                _load_latest_scoring_progress_summary,
                bucket,
                object_prefix,
                candidate_artifact_hash=candidate_artifact_hash,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_scoring_progress_inspect_failed candidate_id=%s error=%s",
                compact_ref(candidate_id),
                str(exc)[:200],
            )
            return from_events
        return from_s3

    async def _candidate_scoring_heartbeat(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        started_at: float,
        claim_lost: asyncio.Event | None = None,
        progress_snapshot: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        try:
            interval = max(
                60.0,
                float(os.environ.get("RESEARCH_LAB_SCORING_HEARTBEAT_SECONDS", "120")),
            )
        except ValueError:
            interval = 120.0
        while True:
            await asyncio.sleep(interval)
            try:
                current = await select_one(
                    "research_lab_candidate_evaluation_current",
                    columns="candidate_id,current_candidate_status,current_evaluator_ref",
                    filters=(("candidate_id", candidate_id),),
                )
                if (
                    not current
                    or current.get("current_candidate_status") != "evaluating"
                    or current.get("current_evaluator_ref") != self.worker_ref
                ):
                    logger.warning(
                        "research_lab_candidate_heartbeat_claim_lost candidate_id=%s worker_ref=%s",
                        compact_ref(candidate_id),
                        self.worker_ref,
                    )
                    if claim_lost is not None:
                        claim_lost.set()
                    return
                event_doc: dict[str, Any] = {
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                }
                if progress_snapshot is not None:
                    try:
                        progress_doc = dict(progress_snapshot())
                    except Exception:
                        progress_doc = {}
                    completed_icps = _completed_icp_count_from_progress_doc(progress_doc)
                    if completed_icps > 0:
                        event_doc["completed_icp_count"] = completed_icps
                        event_doc["scoring_progress"] = _safe_scoring_progress_summary(
                            source="heartbeat",
                            completed_icp_count=completed_icps,
                            rolling_window_hash=str(progress_doc.get("rolling_window_hash") or ""),
                        )
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="evaluating",
                    candidate_status="evaluating",
                    evaluator_ref=self.worker_ref,
                    reason="gateway_qualification_worker_heartbeat",
                    event_doc=event_doc,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "research_lab_candidate_heartbeat_failed candidate_id=%s worker_ref=%s error=%s",
                    compact_ref(candidate_id),
                    self.worker_ref,
                    str(exc)[:240],
                )

    async def _score_candidate(self, candidate: Mapping[str, Any]) -> None:
        candidate_id = str(candidate["candidate_id"])
        start = time.time()
        scored_event_written = False
        scored_score_bundle_id = ""
        telemetry_session: ScoringTelemetrySession | None = None
        assigned_dispatch_event: Mapping[str, Any] = {}
        try:
            evaluation_epoch = await self._resolve_evaluation_epoch()
            stale_result = await self._maybe_rebase_stale_candidate_before_scoring(
                candidate,
                evaluation_epoch=evaluation_epoch,
                elapsed_seconds=lambda: round(time.time() - start, 3),
            )
            if stale_result.get("status") in {
                "legacy_patch_candidate_unsupported",
                "stale_parent_needs_rescore",
                "stale_parent_rebase_failed",
            }:
                await self._maybe_finalize_candidate_receipt(candidate)
                await safe_project_public_loop_activity(
                    str(candidate["ticket_id"]),
                    source_ref=f"candidate_stale_parent_or_legacy_rejected:{candidate_id}",
                    reason=str(stale_result["status"]),
                    config=self.config,
                )
                try:
                    await self._write_audit_bundle(evaluation_epoch)
                except Exception:
                    logger.exception("Research Lab audit bundle write failed after candidate rejection")
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE PRE-SCORING REJECTED",
                        (
                            ("Worker", self.worker_ref),
                            ("Candidate", compact_ref(candidate_id)),
                            ("Status", stale_result.get("status")),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                return
            if stale_result.get("status") == "stale_parent_rebased_to_current":
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE REBASED BEFORE SCORING",
                        (
                            ("Worker", self.worker_ref),
                            ("Candidate", compact_ref(candidate_id)),
                            ("Derived candidate", compact_ref(stale_result.get("derived_candidate_id"))),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                return
            logger.info(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORING STARTED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Ticket", compact_ref(candidate.get("ticket_id"))),
                        ("Evaluation epoch", evaluation_epoch),
                        ("Model timeout", f"{self.config.scoring_worker_model_timeout_seconds}s"),
                        ("Proxy ref", self.proxy_ref_hash),
                    ),
                )
            )
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="evaluating",
                candidate_status="evaluating",
                evaluator_ref=self.worker_ref,
                reason="gateway_qualification_worker_started",
                event_doc={"worker_ref": self.worker_ref, "proxy_ref_hash": self.proxy_ref_hash},
            )
            artifact = PrivateModelArtifactManifest.from_mapping(candidate["private_model_manifest_doc"])
            patch = candidate["candidate_patch_manifest"]
            candidate_kind = str(candidate.get("candidate_kind") or "")
            if candidate_kind != "image_build":
                raise RuntimeError("candidate scoring requires image_build candidate_kind")
            candidate_manifest_doc = candidate.get("candidate_model_manifest_doc")
            if not isinstance(candidate_manifest_doc, Mapping):
                raise RuntimeError("image_build candidate is missing candidate_model_manifest_doc")
            candidate_artifact = PrivateModelArtifactManifest.from_mapping(candidate_manifest_doc)
            reused_bundle_row = await self._find_reusable_scored_bundle(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                candidate_artifact_hash=candidate_artifact.model_artifact_hash,
                evaluation_epoch=evaluation_epoch,
            )
            if reused_bundle_row is not None:
                # Bug #12 rescore path: this candidate's signed bundle already
                # landed but its `scored` event write failed; reuse the bundle
                # instead of producing a divergent second evaluation.
                scored_score_bundle_id = str(reused_bundle_row["score_bundle_id"])
                reused_doc = (
                    reused_bundle_row.get("score_bundle_doc")
                    if isinstance(reused_bundle_row.get("score_bundle_doc"), Mapping)
                    else {}
                )
                reused_window_hash = str(
                    reused_doc.get("icp_set_hash")
                    or reused_bundle_row.get("icp_set_hash")
                    or ""
                )
                if scoring_telemetry_enabled(self.config):
                    reused_results = (
                        list(reused_doc.get("per_icp_results") or ())
                        if isinstance(reused_doc.get("per_icp_results"), list)
                        else []
                    )
                    reused_run = await allocate_scoring_run(
                        identity_doc={
                            "run_type": "candidate_scoring",
                            "candidate_id": candidate_id,
                            "source_run_id": str(candidate.get("run_id") or ""),
                            "rolling_window_hash": reused_window_hash,
                            "reference_artifact_hash": artifact.model_artifact_hash,
                            "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
                            "evaluation_epoch": evaluation_epoch,
                            "scoring_version": str(reused_doc.get("scoring_version") or ""),
                            "evaluator_version": str(reused_doc.get("evaluator_version") or ""),
                        },
                        run_type="candidate_scoring",
                        worker_ref=self.worker_ref,
                        expected_icp_count=len(reused_results),
                        scheduler_type="serial",
                        candidate_id=candidate_id,
                        source_run_id=str(candidate.get("run_id") or ""),
                        ticket_id=str(candidate.get("ticket_id") or ""),
                        rolling_window_hash=reused_window_hash,
                        reference_artifact_hash=artifact.model_artifact_hash,
                        reference_manifest_hash=artifact.manifest_hash,
                        candidate_artifact_hash=candidate_artifact.model_artifact_hash,
                        candidate_manifest_hash=candidate_artifact.manifest_hash,
                        source_score_bundle_id=scored_score_bundle_id,
                        evaluation_epoch=evaluation_epoch,
                    )
                    telemetry_session = ScoringTelemetrySession(reused_run)
                    await emit_run_event(reused_run, "assigned")
                    await emit_run_event(reused_run, "started")
                    for item_index, result_row in enumerate(reused_results):
                        if not isinstance(result_row, Mapping):
                            continue
                        item_ref = str(
                            result_row.get("icp_ref") or result_row.get("icp_hash") or ""
                        )
                        await telemetry_session.plan(
                            icp_ref=item_ref,
                            icp_hash=str(result_row.get("icp_hash") or ""),
                            icp_ordinal=item_index,
                            model_role="candidate",
                            execution_kind="checkpoint_reuse",
                        )
                        await telemetry_session.complete_result(
                            result_row,
                            model_role="candidate",
                            checkpoint_ref=opaque_checkpoint_ref(
                                "score_bundle",
                                scored_score_bundle_id,
                            ),
                            checkpoint_hash=str(reused_doc.get("score_bundle_hash") or ""),
                            checkpoint_persisted=True,
                            outcome="reused_score_bundle",
                        )
                reuse_dispatch_ids = (
                    telemetry_session.run.linked_ids()
                    if telemetry_session is not None and telemetry_session.run is not None
                    else {}
                )
                await create_scoring_dispatch_event(
                    dispatch_type="candidate_scoring",
                    dispatch_status="assigned",
                    worker_ref=self.worker_ref,
                    proxy_ref_hash=self.proxy_ref_hash,
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    rolling_window_hash=reused_window_hash or None,
                    scoring_id=reuse_dispatch_ids.get("scoring_id"),
                    scoring_run_id=reuse_dispatch_ids.get("scoring_run_id"),
                    event_doc={"reused_signed_score_bundle": True},
                )
                await self._complete_candidate_from_reused_bundle(
                    candidate,
                    candidate_id=candidate_id,
                    bundle_row=reused_bundle_row,
                    evaluation_epoch=evaluation_epoch,
                    start=start,
                    telemetry_session=telemetry_session,
                )
                await emit_run_event(
                    telemetry_session.run if telemetry_session is not None else None,
                    "completed",
                    score_bundle_id=scored_score_bundle_id,
                    event_doc={"outcome": "reused_score_bundle"},
                )
                scored_event_written = True
                return
            window, private_holdout_gate = await self._daily_candidate_scoring_window_and_gate(
                artifact=artifact,
            )
            _validate_candidate_conditional_policy_stamp(
                candidate,
                private_holdout_gate,
                active_policy=self.config.conditional_validation_policy().to_dict(),
            )
            await create_rolling_icp_window(window)
            if scoring_telemetry_enabled(self.config):
                scheduler_type = "serial"
                try:
                    candidate_concurrency = max(
                        1,
                        int(os.getenv("RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY", "1")),
                    )
                except ValueError:
                    candidate_concurrency = 1
                if candidate_concurrency > 1:
                    scheduler_type = (
                        "work_conserving"
                        if _env_flag("RESEARCH_LAB_EVAL_WORK_CONSERVING")
                        else "fixed_wave"
                    )
                telemetry_run = await allocate_scoring_run(
                    identity_doc={
                        "run_type": "candidate_scoring",
                        "candidate_id": candidate_id,
                        "source_run_id": str(candidate.get("run_id") or ""),
                        "rolling_window_hash": window.window_hash,
                        "reference_artifact_hash": artifact.model_artifact_hash,
                        "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
                        "baseline_benchmark_bundle_id": str(
                            private_holdout_gate.get("baseline_benchmark_bundle_id") or ""
                        ),
                        "evaluation_epoch": evaluation_epoch,
                        "scoring_version": "qualification-company-scorer:v1",
                        "evaluator_version": "leadpoet-gateway-qualification-worker:research-lab:v1",
                    },
                    run_type="candidate_scoring",
                    worker_ref=self.worker_ref,
                    expected_icp_count=len(window.benchmark_items),
                    scheduler_type=scheduler_type,
                    candidate_id=candidate_id,
                    source_run_id=str(candidate.get("run_id") or ""),
                    ticket_id=str(candidate.get("ticket_id") or ""),
                    rolling_window_hash=window.window_hash,
                    reference_artifact_hash=artifact.model_artifact_hash,
                    reference_manifest_hash=artifact.manifest_hash,
                    candidate_artifact_hash=candidate_artifact.model_artifact_hash,
                    candidate_manifest_hash=candidate_artifact.manifest_hash,
                    baseline_benchmark_bundle_id=str(
                        private_holdout_gate.get("baseline_benchmark_bundle_id") or ""
                    ),
                    evaluation_epoch=evaluation_epoch,
                )
                telemetry_session = ScoringTelemetrySession(telemetry_run)
                await emit_run_event(telemetry_run, "assigned")
                await emit_run_event(telemetry_run, "started")
            dispatch_ids = telemetry_session.run.linked_ids() if telemetry_session and telemetry_session.run else {}
            assigned_dispatch_event = await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="assigned",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                rolling_window_hash=window.window_hash,
                scoring_id=dispatch_ids.get("scoring_id"),
                scoring_run_id=dispatch_ids.get("scoring_run_id"),
            )
            benchmark = SealedBenchmarkSet(
                benchmark_id=window.benchmark_id,
                icp_set_hash=window.window_hash,
                split_ref=window.split_ref,
                item_refs=window.item_refs,
                scoring_version="qualification-company-scorer:v1",
                hidden_plaintext_available=True,
            )
            candidate_runner = AttestedPrivateModelRunnerV2(
                artifact=candidate_artifact,
                spec=DockerPrivateModelSpec(
                    image_digest=candidate_artifact.image_digest,
                    timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                    env_passthrough=self._private_model_env_passthrough(),
                    extra_env=self._with_provider_cost_evaluation_scope(
                        self._private_scoring_env(),
                        run_type="candidate_scoring",
                        rolling_window_hash=window.window_hash,
                        artifact_hash=candidate_artifact.model_artifact_hash,
                        candidate_id=candidate_id,
                        run_id=str(candidate.get("run_id") or ""),
                        ticket_id=str(candidate.get("ticket_id") or ""),
                        dispatch_event_id=str(assigned_dispatch_event.get("dispatch_event_id") or ""),
                        evaluation_epoch=evaluation_epoch,
                        started_at=start,
                    ),
                ),
                model_kind="candidate",
                worker_index=self.config.scoring_worker_index,
                epoch_id=evaluation_epoch,
                parent_graphs=await _attested_model_parent_graphs(
                    model_kind="candidate",
                    artifact=candidate_artifact,
                    candidate_id=candidate_id,
                ),
            )
            run_context = self._candidate_run_context(
                candidate,
                window_hash=window.window_hash,
                evaluation_epoch=evaluation_epoch,
            )
            evaluation_policy = _stored_daily_baseline_evaluation_policy(
                self._evaluation_policy(),
                private_holdout_gate,
            )
            scoring_config_hash = scoring_configuration_hash()
            checkpoint_commitment_hash = ""
            if bool(private_holdout_gate.get("conditional_validation_required")):
                checkpoint_commitment_hash = canonical_hash(
                    {
                        "candidate_id": candidate_id,
                        "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
                        "active_parent_artifact_hash": artifact.model_artifact_hash,
                        "rolling_window_hash": window.window_hash,
                        "category_assignment_hash": str(
                            private_holdout_gate.get("category_assignment_hash") or ""
                        ),
                        "baseline_benchmark_bundle_id": str(
                            private_holdout_gate.get("baseline_benchmark_bundle_id") or ""
                        ),
                        "baseline_benchmark_hash": str(
                            private_holdout_gate.get("baseline_benchmark_hash") or ""
                        ),
                        "conditional_validation_policy_hash": str(
                            private_holdout_gate.get(
                                "conditional_validation_policy_hash"
                            )
                            or ""
                        ),
                        "threshold_points": _safe_float(
                            private_holdout_gate.get("threshold_points"),
                            default=0.0,
                        ),
                        "scoring_configuration_hash": scoring_config_hash,
                        "evaluation_policy": evaluation_policy,
                    }
                )
            unsigned_run_context = {**run_context, "signature_ref": "pending"}
            # §5.4 scorer-judge traces: wrap the qualification scorer so every
            # per-company judgment is captured (pointer-only in event docs) at
            # the evaluator's scoring boundary; behavior-identical to the
            # default scorer the evaluator would otherwise construct.
            scorer_trace_pointers: dict[str, dict[str, str]] = {}
            candidate_company_scorer = _TraceCapturingCompanyScorer(
                recorder=self._get_scorer_trace_recorder(),
                context_ref=candidate_id,
                manifest_uri=artifact.manifest_uri,
                benchmark_items=window.benchmark_items,
                pointer_map=scorer_trace_pointers,
                candidate_id=candidate_id,
                candidate_model_manifest_hash=getattr(artifact, "manifest_hash", None),
                run_id=str(candidate.get("run_id") or ""),
                ticket_id=str(candidate.get("ticket_id") or ""),
                attested_epoch_id=evaluation_epoch,
                attested_purpose="research_lab.candidate_score.v1",
            )
            last_freshness_check_at = 0.0
            claim_lost_event = asyncio.Event()

            async def parent_freshness_check(progress: Mapping[str, Any]) -> None:
                nonlocal last_freshness_check_at
                if claim_lost_event.is_set() and _env_flag("RESEARCH_LAB_SCORING_ABORT_ON_CLAIM_LOSS"):
                    raise ClaimLostDuringScoring(
                        f"scoring claim lost for candidate {compact_ref(candidate_id)}; "
                        "aborting in-flight evaluation at ICP boundary"
                    )
                now = time.time()
                phase = str(progress.get("phase") or "")
                if (
                    phase != "before_icp"
                    and last_freshness_check_at
                    and now - last_freshness_check_at < self.config.stale_parent_check_interval_seconds
                ):
                    return
                last_freshness_check_at = now
                await self._check_candidate_scoring_freshness(
                    parent_artifact=artifact,
                    candidate_window_hash=window.window_hash,
                    progress=progress,
                )

            async def holdout_transition(
                action: str,
                payload: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                if action != "conditional_validation_started":
                    raise RuntimeError(f"unsupported holdout transition: {action}")
                preliminary_results_raw = payload.get("_preliminary_results")
                if not isinstance(preliminary_results_raw, list) or any(
                    not isinstance(item, Mapping) for item in preliminary_results_raw
                ):
                    raise ConditionalValidationRetryableError(
                        "conditional_preliminary_results_missing"
                    )
                await parent_freshness_check(
                    {
                        "phase": "conditional_preliminary_gate",
                        "completed_icp_count": len(preliminary_results_raw),
                    }
                )
                proof = await _authorize_conditional_preliminary_gate(
                    config=self.config,
                    evaluation_epoch=evaluation_epoch,
                    candidate=candidate,
                    artifact=artifact,
                    benchmark=benchmark,
                    patch=patch,
                    candidate_artifact=candidate_artifact,
                    preliminary_results=[dict(item) for item in preliminary_results_raw],
                    run_context=unsigned_run_context,
                    policy=_stored_daily_baseline_evaluation_policy(
                        self._preliminary_evaluation_policy(),
                        private_holdout_gate,
                    ),
                    preliminary_gate=payload,
                    parent_receipts=_attested_receipts_from(
                        candidate_runner,
                        candidate_company_scorer,
                    ),
                )
                lifecycle_source_ref = (
                    f"direct:{checkpoint_commitment_hash}"
                    if checkpoint_commitment_hash
                    else f"direct:{candidate_id}:{window.window_hash}"
                )
                lifecycle_args = {
                    "candidate_id": candidate_id,
                    "assignment_hash": str(
                        payload.get("category_assignment_hash") or ""
                    ),
                    "policy_hash": str(
                        payload.get("conditional_validation_policy_hash") or ""
                    ),
                    "rolling_window_hash": window.window_hash,
                    "baseline_benchmark_bundle_id": str(
                        payload.get("baseline_benchmark_bundle_id") or ""
                    ),
                    "source_ref": lifecycle_source_ref,
                    "decision_score": _safe_float(
                        payload.get("candidate_preliminary_score"),
                        default=0.0,
                    ),
                    "threshold_points": _safe_float(
                        payload.get("threshold_points"),
                        default=0.0,
                    ),
                }
                await create_conditional_validation_event(
                    event_type="preliminary_gate_passed",
                    event_doc={
                        "path": "direct_candidate_scoring",
                        "preliminary_promotion_gate": proof,
                    },
                    **lifecycle_args,
                )
                await create_conditional_validation_event(
                    event_type="conditional_started",
                    event_doc={
                        "path": "direct_candidate_scoring",
                        "preliminary_event": "persisted_before_conditional_execution",
                    },
                    **lifecycle_args,
                )
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="evaluating",
                    candidate_status="evaluating",
                    evaluator_ref=self.worker_ref,
                    reason="conditional_validation_started",
                    event_doc={
                        "worker_ref": self.worker_ref,
                        "rolling_window_hash": window.window_hash,
                        "checkpoint_commitment_hash": checkpoint_commitment_hash,
                        "private_holdout_gate": _candidate_gate_event_doc(payload),
                        "preliminary_promotion_gate_proof_hash": proof["proof_hash"],
                        "promotion_decision_receipt_hash": proof[
                            "promotion_decision_receipt_hash"
                        ],
                    },
                )
                return {"preliminary_promotion_gate": proof}

            # Bug #31: resume already-completed ICPs from the persisted progress
            # artifact and checkpoint each new ICP result, so a requeue at ICP
            # 19/20 no longer re-runs the whole evaluation. Best-effort: any
            # checkpoint failure only loses resumability.
            resume_results: list[dict[str, Any]] | None = None
            icp_checkpoint = None
            progress_rows: list[dict[str, Any]] = []
            telemetry_completed_refs: set[str] = set()
            progress_location = (
                _scoring_progress_s3_location(
                    artifact.manifest_uri, candidate_id, window.window_hash
                )
                if _per_icp_checkpoint_enabled()
                else None
            )
            if progress_location:
                progress_bucket, progress_key = progress_location
                try:
                    resume_results = await asyncio.to_thread(
                        _load_scoring_progress,
                        progress_bucket,
                        progress_key,
                        window_hash=window.window_hash,
                        candidate_artifact_hash=candidate_artifact.model_artifact_hash,
                        commitment_hash=checkpoint_commitment_hash,
                    )
                except Exception as exc:
                    logger.warning(
                        "research_lab_scoring_progress_load_failed candidate_id=%s error=%s",
                        compact_ref(candidate_id),
                        str(exc)[:200],
                    )
                    resume_results = []
                if resume_results:
                    logger.info(
                        "research_lab_scoring_progress_resumed candidate_id=%s completed_icps=%s",
                        compact_ref(candidate_id),
                        len(resume_results),
                    )
                conditional_refs = {
                    str(ref)
                    for ref in (private_holdout_gate.get("conditional_icp_refs") or ())
                    if str(ref)
                }
                progress_rows = [
                    dict(row)
                    for row in resume_results or []
                    if not (
                        str(row.get("icp_ref") or row.get("icp_hash") or "")
                        in conditional_refs
                        and _retryable_measurement_checkpoint_row(row)
                    )
                ]
                resume_results = list(progress_rows)

                public_refs = {
                    str(ref)
                    for ref in (private_holdout_gate.get("public_icp_refs") or ())
                    if str(ref)
                }
                resumed_by_ref = _progress_rows_by_icp_ref(progress_rows)
                if telemetry_session is not None:
                    checkpoint_ref = opaque_checkpoint_ref(progress_bucket, progress_key)
                    for item_index, item in enumerate(window.benchmark_items):
                        item_ref = str(item.get("icp_ref") or item.get("icp_hash") or "")
                        resumed_row = resumed_by_ref.get(item_ref)
                        await telemetry_session.plan(
                            icp_ref=item_ref,
                            icp_hash=str(item.get("icp_hash") or ""),
                            icp_ordinal=item_index,
                            model_role="candidate",
                            phase=(
                                "public"
                                if item_ref in public_refs
                                else "conditional"
                                if item_ref in conditional_refs
                                else "private"
                            ),
                            held=item_ref not in public_refs and resumed_row is None,
                            execution_kind=(
                                "checkpoint_reuse" if resumed_row is not None else "model_invocation"
                            ),
                        )
                        if resumed_row is not None:
                            await telemetry_session.complete_result(
                                resumed_row,
                                model_role="candidate",
                                checkpoint_ref=checkpoint_ref,
                                checkpoint_persisted=True,
                                outcome="checkpoint_reuse",
                            )
                            telemetry_completed_refs.add(item_ref)

                async def _checkpoint_completed_icp(row: Mapping[str, Any]) -> None:
                    progress_rows.append(dict(row))
                    checkpoint_persisted = False
                    checkpoint_hash = ""
                    try:
                        telemetry_index = checkpoint_telemetry_index(
                            telemetry_session,
                            progress_rows,
                            model_role="candidate",
                        )
                        checkpoint_hash = await asyncio.to_thread(
                            _store_scoring_progress,
                            progress_bucket,
                            progress_key,
                            candidate_id=candidate_id,
                            window_hash=window.window_hash,
                            candidate_artifact_hash=candidate_artifact.model_artifact_hash,
                            rows=progress_rows,
                            telemetry_index=telemetry_index,
                            commitment_hash=checkpoint_commitment_hash,
                        )
                        checkpoint_persisted = True
                    except Exception as exc:
                        logger.warning(
                            "research_lab_scoring_progress_store_failed candidate_id=%s error=%s",
                            compact_ref(candidate_id),
                            str(exc)[:200],
                        )
                        row_ref = str(row.get("icp_ref") or row.get("icp_hash") or "")
                        if (
                            bool(private_holdout_gate.get("conditional_validation_required"))
                            and row_ref in conditional_refs
                        ):
                            raise ConditionalValidationRetryableError(
                                "conditional_validation_checkpoint_persist_failed:"
                                f"{compact_ref(row_ref)}"
                            ) from exc
                    if telemetry_session is not None:
                        failure_reason = str(row.get("failure_reason") or "")
                        latch_skipped = "candidate_model_runtime_skipped_after_" in failure_reason
                        outcome = (
                            "latch_skip"
                            if latch_skipped
                            else "scoreable_failure_zero"
                            if failure_reason and not row.get("candidate_company_scores")
                            else "scored"
                        )
                        await telemetry_session.complete_result(
                            row,
                            model_role="candidate",
                            checkpoint_ref=opaque_checkpoint_ref(progress_bucket, progress_key),
                            checkpoint_hash=checkpoint_hash,
                            checkpoint_persisted=checkpoint_persisted,
                            outcome=outcome,
                            terminal_event="skipped" if latch_skipped else "completed",
                        )
                        telemetry_completed_refs.add(
                            str(row.get("icp_ref") or row.get("icp_hash") or "")
                        )

                icp_checkpoint = _checkpoint_completed_icp

            elif telemetry_session is not None:
                public_refs = {
                    str(ref)
                    for ref in (private_holdout_gate.get("public_icp_refs") or ())
                    if str(ref)
                }
                conditional_refs = {
                    str(ref)
                    for ref in (private_holdout_gate.get("conditional_icp_refs") or ())
                    if str(ref)
                }
                for item_index, item in enumerate(window.benchmark_items):
                    item_ref = str(item.get("icp_ref") or item.get("icp_hash") or "")
                    await telemetry_session.plan(
                        icp_ref=item_ref,
                        icp_hash=str(item.get("icp_hash") or ""),
                        icp_ordinal=item_index,
                        model_role="candidate",
                        phase=(
                            "public"
                            if item_ref in public_refs
                            else "conditional"
                            if item_ref in conditional_refs
                            else "private"
                        ),
                        held=item_ref not in public_refs,
                    )

            def _heartbeat_progress_snapshot() -> Mapping[str, Any]:
                return {
                    "completed_icp_count": len(progress_rows),
                    "rolling_window_hash": window.window_hash,
                }

            async def _candidate_attempt_cost_sink(
                icp_ref: str,
                entries: list[dict[str, Any]],
                execution_context: Mapping[str, Any],
            ) -> None:
                persisted = await _persist_provider_cost_events(
                    entries=entries,
                    run_type="candidate_scoring",
                    icp_ref=icp_ref,
                    runner_role="candidate",
                    candidate_id=candidate_id,
                    scoring_id=str(execution_context.get("scoring_id") or ""),
                    scoring_run_id=str(execution_context.get("scoring_run_id") or ""),
                    icp_execution_id=str(execution_context.get("icp_execution_id") or ""),
                )
                if not persisted and telemetry_session is not None:
                    telemetry_session.degraded = True

            heartbeat_task = asyncio.create_task(
                self._candidate_scoring_heartbeat(
                    candidate=candidate,
                    candidate_id=candidate_id,
                    started_at=start,
                    claim_lost=claim_lost_event,
                    progress_snapshot=_heartbeat_progress_snapshot,
                )
            )
            telemetry_heartbeat_stop = asyncio.Event()
            telemetry_heartbeat_task = (
                asyncio.create_task(telemetry_session.heartbeat_loop(telemetry_heartbeat_stop))
                if telemetry_session is not None
                else None
            )
            langfuse_obs = None
            langfuse_trace_id = ""
            # End-to-end continuity: the eval span joins the loop engine's
            # deterministic run trace (run_trace_id(run_id)), and the sampling
            # decision is seeded by run_id so a sampled run keeps its scoring
            # span too — never a fragment of one without the other.
            langfuse_run_id = str(candidate["run_id"])
            candidate_loop_node_id = (
                str((candidate.get("candidate_build_doc") or {}).get("loop_node_id") or "")
                if isinstance(candidate.get("candidate_build_doc"), Mapping)
                else ""
            )
            try:
                with observation(
                    "research_lab.private_eval_pair",
                    as_type="span",
                    trace_id=langfuse_run_trace_id(langfuse_run_id),
                    sample_seed=langfuse_run_id or None,
                    metadata={
                        "run_id": str(candidate["run_id"]),
                        "ticket_id": str(candidate["ticket_id"]),
                        "candidate_id": candidate_id,
                        **({"loop_node_id": candidate_loop_node_id} if candidate_loop_node_id else {}),
                        "miner_hotkey_hash": miner_hotkey_hash(str(candidate.get("miner_hotkey") or "")),
                        "evaluation_epoch": evaluation_epoch,
                        "parent_artifact_hash": artifact.model_artifact_hash,
                        "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
                        "candidate_patch_hash": str(candidate.get("candidate_patch_hash") or ""),
                        "icp_set_hash": window.window_hash,
                        "benchmark_split_ref": window.split_ref,
                        "worker_ref": self.worker_ref,
                    },
                ) as langfuse_obs:
                    try:
                        score_bundle = await evaluate_private_model_pair(
                            artifact_manifest=artifact,
                            benchmark=benchmark,
                            patch_manifest=patch,
                            candidate_artifact_manifest=candidate_artifact.to_dict(),
                            benchmark_items=window.benchmark_items,
                            base_runner=None,
                            candidate_runner=candidate_runner,
                            company_scorer=candidate_company_scorer,
                            run_context=unsigned_run_context,
                            policy=evaluation_policy,
                            private_holdout_gate=private_holdout_gate,
                            parent_freshness_check=parent_freshness_check,
                            icp_checkpoint=icp_checkpoint,
                            resume_results=resume_results,
                            # In-container traces upload keyed by candidate; the
                            # evaluator puts the returned refs into per-ICP rows.
                            trace_sink=self._candidate_incontainer_trace_sink(
                                candidate_id,
                                persist_costs=telemetry_session is None,
                            ),
                            scoring_telemetry_hook=(
                                telemetry_session.lifecycle if telemetry_session is not None else None
                            ),
                            attempt_cost_sink=(
                                _candidate_attempt_cost_sink
                                if telemetry_session is not None
                                else None
                            ),
                            holdout_transition_hook=holdout_transition,
                        )
                        langfuse_trace_id = finish_score_bundle_observation(langfuse_obs, score_bundle)
                    finally:
                        heartbeat_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await heartbeat_task
                        if telemetry_heartbeat_task is not None:
                            telemetry_heartbeat_stop.set()
                            await telemetry_heartbeat_task
            except StaleParentDuringScoring as stale_exc:
                stale_result = await self._queue_stale_parent_rebase(
                    candidate,
                    active_artifact=stale_exc.active_artifact,
                    candidate_parent=stale_exc.candidate_parent,
                    evaluation_epoch=evaluation_epoch,
                    elapsed_seconds=round(time.time() - start, 3),
                    stage="during_scoring_parent_changed",
                    stale_progress=stale_exc.progress,
                )
                await self._maybe_finalize_candidate_receipt(candidate)
                await safe_project_public_loop_activity(
                    str(candidate["ticket_id"]),
                    source_ref=f"candidate_stale_parent_during_scoring:{candidate_id}",
                    reason=str(stale_result.get("status") or "stale_parent_during_scoring"),
                    config=self.config,
                )
                try:
                    await self._write_audit_bundle(evaluation_epoch)
                except Exception:
                    logger.exception("Research Lab audit bundle write failed after stale parent during scoring")
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB CANDIDATE STALE DURING SCORING",
                        (
                            ("Worker", self.worker_ref),
                            ("Candidate", compact_ref(candidate_id)),
                            ("Status", stale_result.get("status")),
                            ("Derived candidate", compact_ref(stale_result.get("derived_candidate_id"))),
                            ("Completed ICPs", stale_exc.progress.get("completed_icp_count", 0)),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                if telemetry_session is not None:
                    await telemetry_session.cancel_active(
                        failure_category="stale_parent_superseded",
                        error=stale_exc,
                    )
                    await emit_run_event(
                        telemetry_session.run,
                        "cancelled",
                        failure_category="stale_parent_superseded",
                        error=stale_exc,
                        event_doc={"outcome": "superseded"},
                    )
                return
            if telemetry_session is not None:
                aggregates = (
                    score_bundle.get("aggregates")
                    if isinstance(score_bundle.get("aggregates"), Mapping)
                    else {}
                )
                for result_row in aggregates.get("per_icp_results") or ():
                    if not isinstance(result_row, Mapping):
                        continue
                    result_ref = str(
                        result_row.get("icp_ref") or result_row.get("icp_hash") or ""
                    )
                    if result_ref in telemetry_completed_refs:
                        continue
                    failure_reason = str(result_row.get("failure_reason") or "")
                    latch_skipped = "candidate_model_runtime_skipped_after_" in failure_reason
                    await telemetry_session.complete_result(
                        result_row,
                        model_role="candidate",
                        checkpoint_persisted=False,
                        outcome=(
                            "latch_skip"
                            if latch_skipped
                            else "scoreable_failure_zero"
                            if failure_reason and not result_row.get("candidate_company_scores")
                            else "scored"
                        ),
                        terminal_event="skipped" if latch_skipped else "completed",
                    )
                    telemetry_completed_refs.add(result_ref)
            gate_result = score_bundle.get("private_holdout_gate")
            aggregate_doc = score_bundle.get("aggregates")
            per_icp_results = (
                aggregate_doc.get("per_icp_results")
                if isinstance(aggregate_doc, Mapping)
                else None
            )
            if not isinstance(per_icp_results, list):
                raise RuntimeError("score bundle is missing per_icp_results for V2 authority")
            if not isinstance(gate_result, Mapping):
                raise RuntimeError("score bundle is missing private_holdout_gate for V2 authority")
            await _compare_candidate_score_bundle_in_enclave(
                evaluation_epoch=evaluation_epoch,
                artifact=artifact,
                benchmark=benchmark,
                patch=patch,
                candidate_artifact=candidate_artifact,
                per_icp_results=per_icp_results,
                run_context=unsigned_run_context,
                policy=evaluation_policy,
                private_holdout_gate=gate_result,
                expected_score_bundle=score_bundle,
                parent_receipts=_attested_receipts_from(
                    candidate_runner,
                    candidate_company_scorer,
                ),
            )
            private_holdout_rejected = (
                isinstance(gate_result, Mapping)
                and str(gate_result.get("decision") or "") == "rejected_before_private_holdout"
            )
            scoring_health_gate = self._scoring_health_gate_result(score_bundle)
            unsigned_hash = str(score_bundle["score_bundle_hash"])
            signature_ref = await asyncio.to_thread(
                sign_digest_with_kms,
                key_id=self.config.score_bundle_kms_key_id,
                digest_hash=unsigned_hash,
                signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
            )
            score_bundle = {**score_bundle, "signature_ref": signature_ref}
            score_bundle_request = ResearchLabScoreBundleCreateRequest(
                bundle_status="scored",
                receipt_id=candidate.get("receipt_id") or None,
                score_bundle=score_bundle,
            )
            bundle, _bundle_event = await create_score_bundle(score_bundle_request)
            scored_score_bundle_id = str(bundle["score_bundle_id"])
            await _persist_conditional_finalization_events(
                gate_result,
                candidate_id=candidate_id,
                source_score_bundle_id=scored_score_bundle_id,
                rolling_window_hash=window.window_hash,
            )
            await _persist_candidate_category_results(
                gate_result,
                source_bundle_ref=scored_score_bundle_id,
                rolling_window_hash=window.window_hash,
                candidate_id=candidate_id,
                scoring_run_id=(
                    str(telemetry_session.run.scoring_run_id)
                    if telemetry_session is not None and telemetry_session.run is not None
                    else ""
                ),
            )
            if langfuse_trace_id:
                try:
                    await insert_row(
                        "engine_trace_mappings",
                        {
                            "execution_trace_ref": str(score_bundle.get("execution_trace_ref") or f"score_bundle:{score_bundle['score_bundle_hash']}"),
                            "langfuse_trace_id": langfuse_trace_id,
                            "langfuse_project": os.getenv("LANGFUSE_PROJECT", "leadpoet-lab-prod-redacted"),
                            "score_bundle_hash": score_bundle["score_bundle_hash"],
                            "run_id": str(candidate["run_id"]),
                            "ticket_id": str(candidate["ticket_id"]),
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "research_lab_langfuse_trace_mapping_write_failed candidate_id=%s error=%s",
                        compact_ref(candidate_id),
                        str(exc)[:200],
                    )
            await self._create_scored_evaluation_event(
                candidate=candidate,
                candidate_id=candidate_id,
                score_bundle_id=scored_score_bundle_id,
                event_doc={
                    "score_bundle_hash": score_bundle["score_bundle_hash"],
                    "rolling_window_hash": window.window_hash,
                    "elapsed_seconds": round(time.time() - start, 3),
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                    "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                    "scoring_health_gate": scoring_health_gate,
                    "serving_model_version": _event_serving_model_version(score_bundle),
                    # §5.4 pointers only ({icp_ref: {s3_ref, sha256}}) — never
                    # judgment content (audit-scan poison).
                    **(
                        {"scorer_trace_refs": dict(scorer_trace_pointers)}
                        if scorer_trace_pointers
                        else {}
                    ),
                    **({"langfuse_trace_id": langfuse_trace_id} if langfuse_trace_id else {}),
                },
            )
            scored_event_written = True
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="scored",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                rolling_window_hash=window.window_hash,
                score_bundle_id=str(bundle["score_bundle_id"]),
                scoring_id=(
                    telemetry_session.run.scoring_id
                    if telemetry_session is not None and telemetry_session.run is not None
                    else None
                ),
                scoring_run_id=(
                    telemetry_session.run.scoring_run_id
                    if telemetry_session is not None and telemetry_session.run is not None
                    else None
                ),
                event_doc={
                    "elapsed_seconds": round(time.time() - start, 3),
                    "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                    "scoring_health_gate": scoring_health_gate,
                    "serving_model_version": _event_serving_model_version(score_bundle),
                },
            )
            await emit_run_event(
                telemetry_session.run if telemetry_session is not None else None,
                "completed",
                score_bundle_id=str(bundle["score_bundle_id"]),
                telemetry_degraded=bool(telemetry_session and telemetry_session.degraded),
                event_doc={
                    "outcome": (
                        "public_gate_rejected"
                        if private_holdout_rejected
                        else (
                            "scoring_health_quarantined"
                            if scoring_health_gate.get("decision") == "quarantine"
                            else "scored"
                        )
                    )
                },
            )
            if private_holdout_rejected:
                promotion_result = await self._record_public_holdout_rejected(
                    candidate=candidate,
                    score_bundle_row=bundle,
                    score_bundle=score_bundle,
                    gate_result=gate_result,
                )
            elif scoring_health_gate.get("decision") == "quarantine":
                promotion_result = await self._record_scoring_health_quarantined(
                    candidate=candidate,
                    score_bundle_row=bundle,
                    score_bundle=score_bundle,
                    scoring_health_gate=scoring_health_gate,
                )
            else:
                promotion_result = await self._maybe_promote_scored_candidate(
                    candidate=candidate,
                    score_bundle_row=bundle,
                    score_bundle=score_bundle,
                )
            if promotion_result.get("status") == "stale_parent_needs_rescore":
                promotion_result = await self._queue_stale_parent_rebase(
                    candidate,
                    active_artifact=(await load_active_private_model(self.config, register_bootstrap=True)).artifact,
                    candidate_parent=str(candidate.get("parent_artifact_hash") or ""),
                    evaluation_epoch=evaluation_epoch,
                    elapsed_seconds=round(time.time() - start, 3),
                    stage="after_scoring_parent_changed",
                )
            await self._maybe_record_score_backfill(
                candidate=candidate,
                score_bundle_row=bundle,
                score_bundle=score_bundle,
                promotion_result=promotion_result,
            )
            await self._maybe_finalize_candidate_receipt(candidate)
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_scored:{candidate_id}:{bundle['score_bundle_id']}",
                reason="gateway_qualification_worker_scored_candidate",
                config=self.config,
            )
            await self._write_audit_bundle(int(run_context["evaluation_epoch"]))
            logger.info(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Score bundle", compact_ref(bundle["score_bundle_id"])),
                        ("Rolling window", compact_ref(window.window_hash)),
                        ("Private holdout gate", (gate_result or {}).get("decision") if isinstance(gate_result, Mapping) else "-"),
                        ("Promotion", promotion_result.get("status")),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )
        except Exception as exc:
            if scored_event_written:
                await self._record_scored_candidate_side_effect_failure(
                    candidate=candidate,
                    candidate_id=candidate_id,
                    score_bundle_id=scored_score_bundle_id,
                    error=exc,
                    elapsed_seconds=round(time.time() - start, 3),
                )
                return
            failure_class, retryable = _candidate_scoring_failure_class(exc)
            claim_attempts = await self._candidate_claim_attempt_count(candidate_id)
            max_attempts = int(self.config.scoring_worker_max_claim_requeues)
            conditional_retry_hold = (
                failure_class == "conditional_validation_retryable_failure"
            )
            if failure_class == "baseline_not_ready":
                retry_after_seconds = int(self.config.scoring_worker_baseline_not_ready_retry_seconds)
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="queued",
                    candidate_status="queued",
                    evaluator_ref=self.worker_ref,
                    reason="baseline_not_ready",
                    event_doc={
                        "failure_class": failure_class,
                        "retryable": True,
                        "retry_after_seconds": retry_after_seconds,
                        "error_diagnostics": _event_error_diagnostics(exc),
                        "elapsed_seconds": round(time.time() - start, 3),
                        "worker_ref": self.worker_ref,
                        "proxy_ref_hash": self.proxy_ref_hash,
                        "claim_attempts": claim_attempts,
                        **_candidate_baseline_wait_event_doc(exc),
                    },
                )
                logger.warning(
                    "research_lab_candidate_baseline_not_ready_requeued candidate_id=%s retry_after_seconds=%s error=%s",
                    compact_ref(candidate_id),
                    retry_after_seconds,
                    str(exc)[:240],
                )
                if telemetry_session is not None:
                    await telemetry_session.cancel_active(
                        failure_category="baseline_not_ready",
                        error=exc,
                    )
                    await emit_run_event(
                        telemetry_session.run,
                        "cancelled",
                        retryable=True,
                        failure_category="baseline_not_ready",
                        error=exc,
                        event_doc={"outcome": "paused_for_baseline"},
                    )
                return
            if conditional_retry_hold:
                gate_for_failure = locals().get("private_holdout_gate")
                window_for_failure = locals().get("window")
                if isinstance(gate_for_failure, Mapping) and window_for_failure is not None:
                    try:
                        await create_conditional_validation_event(
                            candidate_id=candidate_id,
                            event_type="retryable_failure",
                            assignment_hash=str(
                                gate_for_failure.get("category_assignment_hash") or ""
                            ),
                            policy_hash=str(
                                gate_for_failure.get(
                                    "conditional_validation_policy_hash"
                                )
                                or ""
                            ),
                            rolling_window_hash=str(window_for_failure.window_hash),
                            baseline_benchmark_bundle_id=str(
                                gate_for_failure.get(
                                    "baseline_benchmark_bundle_id"
                                )
                                or ""
                            ),
                            source_ref=f"direct:claim:{claim_attempts}",
                            threshold_points=_safe_float(
                                gate_for_failure.get("threshold_points"),
                                default=0.0,
                            ),
                            failure_class=failure_class,
                            event_doc={
                                "claim_attempt": claim_attempts,
                                "retryable_hold": True,
                            },
                        )
                    except Exception:
                        logger.warning(
                            "research_lab_conditional_retry_event_write_failed "
                            "candidate_id=%s",
                            compact_ref(candidate_id),
                            exc_info=True,
                        )
            if _candidate_scoring_should_requeue(
                failure_class=failure_class,
                retryable=retryable,
                claim_attempts=claim_attempts,
                max_attempts=max_attempts,
            ):
                retry_after_seconds = int(self.config.scoring_worker_retryable_failure_retry_seconds)
                retry_reason = (
                    "conditional_validation_retryable_failure"
                    if failure_class == "conditional_validation_retryable_failure"
                    else "candidate_scoring_retryable_failure"
                )
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="queued",
                    candidate_status="queued",
                    evaluator_ref=self.worker_ref,
                    reason=retry_reason,
                    event_doc={
                        "failure_class": failure_class,
                        "retryable": True,
                        "retry_after_seconds": retry_after_seconds,
                        "error_diagnostics": _event_error_diagnostics(exc),
                        "elapsed_seconds": round(time.time() - start, 3),
                        "worker_ref": self.worker_ref,
                        "proxy_ref_hash": self.proxy_ref_hash,
                        "claim_attempts": claim_attempts,
                        "max_claim_attempts": max_attempts,
                        "terminal_attempt_cap_bypassed": conditional_retry_hold,
                    },
                )
                logger.warning(
                    "research_lab_candidate_retryable_failure_requeued candidate_id=%s failure_class=%s claim_attempts=%s/%s error=%s",
                    compact_ref(candidate_id),
                    failure_class,
                    claim_attempts,
                    max_attempts,
                    str(exc)[:240],
                )
                if telemetry_session is not None:
                    await telemetry_session.cancel_active(
                        failure_category=failure_class,
                        error=exc,
                    )
                    await emit_run_event(
                        telemetry_session.run,
                        "failed",
                        retryable=True,
                        failure_category=failure_class,
                        error=exc,
                    )
                return
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="failed",
                candidate_status="failed",
                evaluator_ref=self.worker_ref,
                reason=f"candidate_scoring_{failure_class}",
                event_doc={
                    "failure_class": failure_class,
                    "retryable": bool(retryable),
                    "claim_attempts": claim_attempts,
                    "max_claim_attempts": max_attempts,
                    "error_diagnostics": _event_error_diagnostics(exc),
                    "elapsed_seconds": round(time.time() - start, 3),
                    "worker_ref": self.worker_ref,
                    "proxy_ref_hash": self.proxy_ref_hash,
                },
            )
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                scoring_id=(
                    telemetry_session.run.scoring_id
                    if telemetry_session is not None and telemetry_session.run is not None
                    else None
                ),
                scoring_run_id=(
                    telemetry_session.run.scoring_run_id
                    if telemetry_session is not None and telemetry_session.run is not None
                    else None
                ),
                event_doc={
                    "failure_class": failure_class,
                    "retryable": bool(retryable),
                    "claim_attempts": claim_attempts,
                    "max_claim_attempts": max_attempts,
                    "error_diagnostics": _event_error_diagnostics(exc),
                },
            )
            if telemetry_session is not None:
                await telemetry_session.cancel_active(
                    failure_category=failure_class,
                    error=exc,
                )
                await emit_run_event(
                    telemetry_session.run,
                    "failed",
                    retryable=bool(retryable),
                    failure_category=failure_class,
                    error=exc,
                )
            await self._maybe_finalize_candidate_receipt(candidate)
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_failed:{candidate_id}",
                reason=f"candidate_scoring_{failure_class}",
                config=self.config,
            )
            try:
                await self._write_audit_bundle(await self._resolve_evaluation_epoch())
            except Exception:
                logger.exception("Research Lab audit bundle write failed after candidate failure")
            logger.exception(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORING FAILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Error", str(exc)[:300]),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )

    async def _create_scored_evaluation_event(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        score_bundle_id: str,
        event_doc: dict[str, Any],
    ) -> None:
        """Write the terminal ``scored`` event with bounded retries (bug #12).

        A signed score bundle already exists when this runs; losing the scored
        event orphans that bundle and re-scores the candidate from scratch on
        the next claim, so transient write failures retry and a persistent
        failure leaves a reconcilable marker before propagating.
        """
        try:
            attempts = max(1, int(os.getenv("RESEARCH_LAB_SCORED_EVENT_WRITE_ATTEMPTS", "3")))
        except ValueError:
            attempts = 3
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(candidate["run_id"]),
                    ticket_id=str(candidate["ticket_id"]),
                    event_type="scored",
                    candidate_status="scored",
                    evaluator_ref=self.worker_ref,
                    reason="gateway_qualification_worker_scored_candidate",
                    score_bundle_id=score_bundle_id,
                    event_doc=event_doc,
                )
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "research_lab_scored_event_write_retry candidate_id=%s attempt=%s/%s error=%s",
                    compact_ref(candidate_id),
                    attempt,
                    attempts,
                    _short_error(exc),
                )
                if attempt < attempts:
                    await asyncio.sleep(min(30.0, 2.0 * attempt))
        try:
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                score_bundle_id=score_bundle_id,
                event_doc={
                    "reason": "scored_event_write_failed_for_signed_bundle",
                    "reconcile_marker": "scored_event_missing_for_signed_bundle",
                    "error_diagnostics": _event_error_diagnostics(last_exc),
                },
            )
        except Exception:
            logger.exception("research_lab_scored_event_marker_write_failed")
        logger.error(
            "research_lab_scored_event_write_failed_permanently candidate_id=%s score_bundle_id=%s attempts=%s",
            compact_ref(candidate_id),
            compact_ref(score_bundle_id),
            attempts,
        )
        assert last_exc is not None
        raise last_exc

    async def _find_reusable_scored_bundle(
        self,
        *,
        candidate_id: str,
        run_id: str,
        candidate_artifact_hash: str,
        evaluation_epoch: int,
    ) -> dict[str, Any] | None:
        """Locate a signed score bundle already produced for this candidate+epoch.

        Bug #12 rescore path: a candidate whose bundle landed but whose scored
        event write failed gets re-claimed; without this it re-runs the full
        evaluation and produces a divergent second signed bundle.
        """
        if _env_flag("RESEARCH_LAB_DISABLE_SCORED_BUNDLE_REUSE"):
            return None
        try:
            rows = await select_many(
                "research_evaluation_score_bundle_current",
                columns="*",
                filters=(
                    ("run_id", run_id),
                    ("candidate_artifact_hash", candidate_artifact_hash),
                    ("evaluation_epoch", int(evaluation_epoch)),
                    ("bundle_status", "scored"),
                ),
                order_by=(("created_at", True),),
                limit=5,
            )
        except Exception as exc:
            # Best-effort side check: never fail the scoring pass over it.
            logger.warning(
                "research_lab_reusable_bundle_lookup_failed candidate_id=%s error=%s",
                compact_ref(candidate_id),
                str(exc)[:200],
            )
            return None
        trace_suffix = f":{candidate_id}"
        for row in rows:
            doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else None
            if not doc:
                continue
            if not str(doc.get("execution_trace_ref") or "").endswith(trace_suffix):
                continue
            if not str(row.get("signature_ref") or doc.get("signature_ref") or ""):
                continue
            if not str(doc.get("score_bundle_hash") or ""):
                continue
            if self._scoring_health_gate_result(doc).get("decision") == "quarantine":
                # A degraded measurement is not reusable evidence: a
                # provider-recovery requeue must re-evaluate fresh instead of
                # re-quarantining the same bundle forever.
                logger.warning(
                    "research_lab_reusable_bundle_skipped_quarantined candidate_id=%s score_bundle_id=%s",
                    compact_ref(candidate_id),
                    compact_ref(str(row.get("score_bundle_id") or "")),
                )
                continue
            return dict(row)
        return None

    async def _complete_candidate_from_reused_bundle(
        self,
        candidate: Mapping[str, Any],
        *,
        candidate_id: str,
        bundle_row: Mapping[str, Any],
        evaluation_epoch: int,
        start: float,
        telemetry_session: ScoringTelemetrySession | None = None,
    ) -> None:
        """Finish a candidate from its already-signed bundle (bug #12): write the
        scored event, then mirror the normal post-scored side-effect sequence."""
        score_bundle = dict(bundle_row.get("score_bundle_doc") or {})
        score_bundle_id = str(bundle_row["score_bundle_id"])
        rolling_window_hash = str(score_bundle.get("icp_set_hash") or bundle_row.get("icp_set_hash") or "")
        gate_result = score_bundle.get("private_holdout_gate")
        private_holdout_rejected = (
            isinstance(gate_result, Mapping)
            and str(gate_result.get("decision") or "") == "rejected_before_private_holdout"
        )
        scoring_health_gate = self._scoring_health_gate_result(score_bundle)
        if isinstance(gate_result, Mapping):
            await _persist_conditional_finalization_events(
                gate_result,
                candidate_id=candidate_id,
                source_score_bundle_id=score_bundle_id,
                rolling_window_hash=rolling_window_hash,
            )
            await _persist_candidate_category_results(
                gate_result,
                source_bundle_ref=score_bundle_id,
                rolling_window_hash=rolling_window_hash,
                candidate_id=candidate_id,
                scoring_run_id=(
                    str(telemetry_session.run.scoring_run_id)
                    if telemetry_session is not None and telemetry_session.run is not None
                    else ""
                ),
            )
        logger.warning(
            format_worker_block(
                "RESEARCH LAB CANDIDATE REUSING SIGNED SCORE BUNDLE",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Run", compact_ref(candidate.get("run_id"))),
                    ("Score bundle", compact_ref(score_bundle_id)),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Reason", "signed bundle exists for this candidate+epoch; skipping re-evaluation"),
                ),
            )
        )
        await self._create_scored_evaluation_event(
            candidate=candidate,
            candidate_id=candidate_id,
            score_bundle_id=score_bundle_id,
            event_doc={
                "score_bundle_hash": str(score_bundle.get("score_bundle_hash") or ""),
                "rolling_window_hash": rolling_window_hash,
                "elapsed_seconds": round(time.time() - start, 3),
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
                "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                "scoring_health_gate": scoring_health_gate,
                "serving_model_version": _event_serving_model_version(score_bundle),
                "reused_signed_score_bundle": True,
            },
        )
        # From here the candidate is scored: side-effect failures must not undo
        # that, mirroring the main path's post-scored handling.
        try:
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="scored",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                rolling_window_hash=rolling_window_hash or None,
                score_bundle_id=score_bundle_id,
                scoring_id=(
                    telemetry_session.run.scoring_id
                    if telemetry_session is not None and telemetry_session.run is not None
                    else None
                ),
                scoring_run_id=(
                    telemetry_session.run.scoring_run_id
                    if telemetry_session is not None and telemetry_session.run is not None
                    else None
                ),
                event_doc={
                    "elapsed_seconds": round(time.time() - start, 3),
                    "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                    "scoring_health_gate": scoring_health_gate,
                    "serving_model_version": _event_serving_model_version(score_bundle),
                    "reused_signed_score_bundle": True,
                },
            )
            if private_holdout_rejected:
                promotion_result = await self._record_public_holdout_rejected(
                    candidate=candidate,
                    score_bundle_row=bundle_row,
                    score_bundle=score_bundle,
                    gate_result=gate_result,
                )
            elif scoring_health_gate.get("decision") == "quarantine":
                promotion_result = await self._record_scoring_health_quarantined(
                    candidate=candidate,
                    score_bundle_row=bundle_row,
                    score_bundle=score_bundle,
                    scoring_health_gate=scoring_health_gate,
                )
            else:
                promotion_result = await self._maybe_promote_scored_candidate(
                    candidate=candidate,
                    score_bundle_row=bundle_row,
                    score_bundle=score_bundle,
                )
            if promotion_result.get("status") == "stale_parent_needs_rescore":
                promotion_result = await self._queue_stale_parent_rebase(
                    candidate,
                    active_artifact=(await load_active_private_model(self.config, register_bootstrap=True)).artifact,
                    candidate_parent=str(candidate.get("parent_artifact_hash") or ""),
                    evaluation_epoch=evaluation_epoch,
                    elapsed_seconds=round(time.time() - start, 3),
                    stage="after_scoring_parent_changed",
                )
            await self._maybe_record_score_backfill(
                candidate=candidate,
                score_bundle_row=bundle_row,
                score_bundle=score_bundle,
                promotion_result=promotion_result,
            )
            await self._maybe_finalize_candidate_receipt(candidate)
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_scored:{candidate_id}:{score_bundle_id}",
                reason="gateway_qualification_worker_scored_candidate",
                config=self.config,
            )
            await self._write_audit_bundle(evaluation_epoch)
            logger.info(
                format_worker_block(
                    "RESEARCH LAB CANDIDATE SCORED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Run", compact_ref(candidate.get("run_id"))),
                        ("Score bundle", compact_ref(score_bundle_id)),
                        ("Rolling window", compact_ref(rolling_window_hash)),
                        ("Private holdout gate", (gate_result or {}).get("decision") if isinstance(gate_result, Mapping) else "-"),
                        ("Promotion", promotion_result.get("status")),
                        ("Reused signed bundle", True),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )
        except Exception as exc:
            await self._record_scored_candidate_side_effect_failure(
                candidate=candidate,
                candidate_id=candidate_id,
                score_bundle_id=score_bundle_id,
                error=exc,
                elapsed_seconds=round(time.time() - start, 3),
            )

    async def _record_scored_candidate_side_effect_failure(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        score_bundle_id: str,
        error: BaseException,
        elapsed_seconds: float,
    ) -> None:
        event_doc = {
            "error_diagnostics": _event_error_diagnostics(error),
            "elapsed_seconds": elapsed_seconds,
            "worker_ref": self.worker_ref,
            "proxy_ref_hash": self.proxy_ref_hash,
            "score_bundle_id": score_bundle_id,
            "candidate_status_preserved": "scored",
        }
        try:
            # Deliberately NOT promotion_failed/failed: scoring (and possibly
            # promotion) succeeded — only a post-score side effect failed. The
            # failed labels outrank `scored` in the public card derivation and
            # corrupted all 21 historical scored candidates (chain B).
            await create_candidate_promotion_event(
                candidate_id=candidate_id,
                source_score_bundle_id=score_bundle_id or None,
                event_type="post_score_side_effect_failed",
                promotion_status="post_score_side_effect_failed",
                active_parent_artifact_hash=str(candidate.get("parent_artifact_hash") or ""),
                candidate_parent_artifact_hash=str(candidate.get("parent_artifact_hash") or ""),
                worker_ref=self.worker_ref,
                event_doc={
                    **event_doc,
                    "reason": "post_score_side_effect_failed",
                },
            )
        except Exception:
            logger.exception("research_lab_post_score_side_effect_event_write_failed")
        try:
            await create_scoring_dispatch_event(
                # 'candidate_scoring' — the dispatch_type DB CHECK only allows
                # the base types, so the old side_effect variant never landed.
                dispatch_type="candidate_scoring",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                score_bundle_id=score_bundle_id or None,
                event_doc=event_doc,
            )
        except Exception:
            logger.exception("research_lab_scored_candidate_side_effect_dispatch_failed")
        # The candidate genuinely scored: finalize the receipt and project the
        # true scored state so the public card does not freeze at `scoring`.
        try:
            await self._maybe_finalize_candidate_receipt(candidate)
        except Exception:
            logger.exception("research_lab_scored_candidate_side_effect_receipt_finalize_failed")
        try:
            await safe_project_public_loop_activity(
                str(candidate["ticket_id"]),
                source_ref=f"candidate_scored_side_effect_failed:{candidate_id}",
                reason="gateway_qualification_worker_scored_candidate",
                config=self.config,
            )
        except Exception:
            logger.exception("research_lab_scored_candidate_side_effect_projection_failed")
        logger.exception(
            format_worker_block(
                "RESEARCH LAB CANDIDATE POST-SCORE SIDE EFFECT FAILED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Run", compact_ref(candidate.get("run_id"))),
                    ("Score bundle", compact_ref(score_bundle_id)),
                    ("Error", str(error)[:300]),
                    ("Candidate state", "scored"),
                ),
            )
        )

    async def _record_public_holdout_rejected(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        gate_result: Any,
    ) -> dict[str, Any]:
        aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
        metric = promotion_improvement_metric(score_bundle)
        improvement_points = float(metric.improvement_points)
        await compare_attested_promotion_metric(
            epoch_id=(
                score_bundle.get("evaluation_epoch")
                or getattr(self.config, "evaluation_epoch", 0)
                or 0
            ),
            score_bundle=score_bundle,
            expected_improvement_points=improvement_points,
            expected_event_doc=metric.event_doc(),
        )
        delta_lcb = float(aggregates.get("delta_lcb") or 0.0)
        candidate_parent = str(candidate.get("parent_artifact_hash") or score_bundle.get("parent_artifact_hash") or "")
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row.get("score_bundle_id") or ""),
            event_type="promotion_checked",
            promotion_status="checked",
            active_parent_artifact_hash=candidate_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=str(score_bundle.get("icp_set_hash") or ""),
            improvement_points=improvement_points,
            threshold_points=float(self.config.improvement_threshold_points),
            worker_ref=self.worker_ref,
            event_doc={
                "delta_lcb": round(delta_lcb, 6),
                "auto_commit_enabled": self.config.auto_commit_enabled,
                "candidate_kind": str(candidate.get("candidate_kind") or ""),
                "decision_path": "public_holdout_rejected",
                "promotion_metric": metric.event_doc(),
                "serving_model_version": _event_serving_model_version(score_bundle),
            },
        )
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row.get("score_bundle_id") or ""),
            event_type="public_holdout_rejected",
            promotion_status="rejected",
            active_parent_artifact_hash=candidate_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=str(score_bundle.get("icp_set_hash") or ""),
            improvement_points=improvement_points,
            threshold_points=float(self.config.improvement_threshold_points),
            worker_ref=self.worker_ref,
            event_doc={
                "reason": "rejected_before_private_holdout",
                "private_holdout_gate": _candidate_gate_event_doc(gate_result),
                "mean_delta": round(improvement_points, 6),
                "delta_lcb": round(delta_lcb, 6),
                "candidate_kind": str(candidate.get("candidate_kind") or ""),
                "promotion_metric": metric.event_doc(),
                "serving_model_version": _event_serving_model_version(score_bundle),
            },
        )
        return {"status": "rejected_public_holdout_gate"}

    async def _record_scoring_health_quarantined(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        scoring_health_gate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Fail-closed divert: record the quarantine, withhold promotion.

        The signed score bundle and the scored evaluation event are already
        written (the measurement is preserved as evidence), but a bundle whose
        scoring health violated the gate thresholds must not drive promotion.
        The recovery reconcile requeues quarantined candidates for a clean
        rescore once providers are healthy again.
        """
        candidate_parent = str(candidate.get("parent_artifact_hash") or score_bundle.get("parent_artifact_hash") or "")
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row.get("score_bundle_id") or ""),
            event_type="scoring_health_quarantined",
            promotion_status="rejected",
            active_parent_artifact_hash=candidate_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=str(score_bundle.get("icp_set_hash") or ""),
            improvement_points=0.0,
            threshold_points=float(self.config.improvement_threshold_points),
            worker_ref=self.worker_ref,
            event_doc={
                "reason": "scoring_health_gate_violation",
                "decision_path": "scoring_health_quarantined",
                "scoring_health_gate": dict(scoring_health_gate),
                "candidate_kind": str(candidate.get("candidate_kind") or ""),
                "serving_model_version": _event_serving_model_version(score_bundle),
            },
        )
        logger.warning(
            "research_lab_candidate_scoring_health_quarantined candidate_id=%s violations=%s",
            compact_ref(str(candidate.get("candidate_id") or "")),
            json.dumps(scoring_health_gate.get("violations") or [])[:400],
        )
        return {"status": "scoring_health_quarantined"}

    async def _requeue_quarantined_candidates(self) -> int:
        """Provider-recovery rescore: requeue quarantined candidates.

        Runs only after the provider preflight reported healthy. A candidate
        whose latest promotion event is ``scoring_health_quarantined`` and
        whose evaluation is terminally ``scored`` gets a fresh ``queued``
        evaluation event (reason ``provider_recovery_rescore``), bounded by a
        per-candidate attempt cap so a flapping provider cannot spin rescores.
        Returns the number of candidates requeued.
        """
        if not self.config.scoring_health_gate_enabled:
            return 0
        interval = float(_env_int("RESEARCH_LAB_QUARANTINE_RECOVERY_INTERVAL_SECONDS", 300))
        now = time.monotonic()
        # None sentinel: a 0.0 default would wrongly throttle the first pass
        # on hosts whose monotonic clock is younger than the interval.
        last = getattr(self, "_last_quarantine_recovery_at", None)
        if last is not None and now - last < interval:
            return 0
        self._last_quarantine_recovery_at = now
        max_attempts = _env_int("RESEARCH_LAB_QUARANTINE_RECOVERY_MAX_ATTEMPTS", 2)
        per_pass_cap = _env_int("RESEARCH_LAB_QUARANTINE_RECOVERY_PER_PASS", 10)
        try:
            quarantine_events = await select_many(
                "research_lab_candidate_promotion_events",
                columns="candidate_id,created_at",
                filters=(("event_type", "scoring_health_quarantined"),),
                order_by=(("created_at", True),),
                limit=100,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_quarantine_recovery_scan_failed error=%s", str(exc)[:200]
            )
            return 0
        requeued = 0
        seen: set[str] = set()
        for event_row in quarantine_events:
            if requeued >= max(1, per_pass_cap):
                break
            candidate_id = str(event_row.get("candidate_id") or "")
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            try:
                current = await select_one(
                    "research_lab_candidate_evaluation_current",
                    filters=(("candidate_id", candidate_id),),
                )
                if not isinstance(current, Mapping):
                    continue
                if str(current.get("current_candidate_status") or "") != "scored":
                    # queued/evaluating means a rescore is already in flight;
                    # failed/rejected/tombstoned are terminal for other reasons.
                    continue
                latest_promotions = await select_many(
                    "research_lab_candidate_promotion_events",
                    columns="event_type,created_at",
                    filters=(("candidate_id", candidate_id),),
                    order_by=(("created_at", True),),
                    limit=1,
                )
                latest_promotion = latest_promotions[0] if latest_promotions else None
                if (
                    not isinstance(latest_promotion, Mapping)
                    or str(latest_promotion.get("event_type") or "") != "scoring_health_quarantined"
                ):
                    # A later promotion decision superseded the quarantine.
                    continue
                prior_recoveries = await select_many(
                    "research_lab_candidate_evaluation_events",
                    columns="event_id",
                    filters=(
                        ("candidate_id", candidate_id),
                        ("event_type", "queued"),
                        ("reason", "provider_recovery_rescore"),
                    ),
                    limit=max_attempts + 1,
                )
                if len(prior_recoveries) >= max(1, max_attempts):
                    logger.warning(
                        "research_lab_quarantine_recovery_attempts_exhausted candidate_id=%s attempts=%d",
                        compact_ref(candidate_id),
                        len(prior_recoveries),
                    )
                    continue
                await create_candidate_evaluation_event(
                    candidate_id=candidate_id,
                    run_id=str(current.get("run_id") or ""),
                    ticket_id=str(current.get("ticket_id") or ""),
                    event_type="queued",
                    candidate_status="queued",
                    evaluator_ref=self.worker_ref,
                    reason="provider_recovery_rescore",
                    event_doc={
                        "recovering_worker_ref": self.worker_ref,
                        "quarantined_at": str(event_row.get("created_at") or ""),
                        "previous_candidate_status": current.get("current_candidate_status"),
                        "recovery_attempt": len(prior_recoveries) + 1,
                        "max_recovery_attempts": max_attempts,
                    },
                )
                requeued += 1
                logger.warning(
                    "research_lab_quarantined_candidate_requeued candidate_id=%s attempt=%d",
                    compact_ref(candidate_id),
                    len(prior_recoveries) + 1,
                )
            except Exception as exc:  # noqa: BLE001 - per-candidate containment
                logger.warning(
                    "research_lab_quarantine_recovery_requeue_failed candidate_id=%s error=%s",
                    compact_ref(candidate_id),
                    str(exc)[:200],
                )
        return requeued

    def _scoring_health_gate_result(self, score_bundle: Mapping[str, Any]) -> dict[str, Any]:
        health = score_bundle.get("scoring_health") if isinstance(score_bundle.get("scoring_health"), Mapping) else {}
        thresholds = {
            "reference_runtime_failure_rate": getattr(
                self.config, "scoring_health_max_reference_runtime_failure_rate", 0.25
            ),
            "candidate_runtime_failure_rate": getattr(
                self.config, "scoring_health_max_candidate_runtime_failure_rate", 0.25
            ),
            "reference_zero_company_rate": getattr(
                self.config, "scoring_health_max_reference_zero_company_rate", 1.0
            ),
            "candidate_zero_company_rate": getattr(
                self.config, "scoring_health_max_candidate_zero_company_rate", 1.0
            ),
            "provider_error_rate": getattr(
                self.config, "scoring_health_max_provider_error_rate", 0.10
            ),
            "timeout_rate": getattr(self.config, "scoring_health_max_timeout_rate", 0.10),
        }
        observed = {
            "reference_runtime_failure_rate": _failure_rate_from_success(
                health.get("reference_runtime_success_rate")
            ),
            "candidate_runtime_failure_rate": _failure_rate_from_success(
                health.get("candidate_runtime_success_rate")
            ),
            "reference_zero_company_rate": _safe_float(health.get("reference_zero_company_rate")),
            "candidate_zero_company_rate": _safe_float(health.get("candidate_zero_company_rate")),
            "provider_error_rate": _safe_float(health.get("provider_error_rate")),
            "timeout_rate": _safe_float(health.get("timeout_rate")),
        }
        violations: list[dict[str, Any]] = []
        for metric, threshold in thresholds.items():
            value = float(observed.get(metric, 0.0))
            if value <= float(threshold):
                continue
            violations.append(
                {
                    "metric": metric,
                    "observed": round(value, 6),
                    "threshold": round(float(threshold), 6),
                }
            )
        configured_enabled = bool(getattr(self.config, "scoring_health_gate_enabled", False))
        would_quarantine = bool(violations)
        # Fail-closed: when the gate is enabled, threshold violations divert
        # the candidate to quarantine (score recorded, promotion withheld,
        # rescored after providers recover) instead of treating a degraded
        # measurement as an authoritative result.
        decision = "quarantine" if configured_enabled and would_quarantine else "observe_only"
        return {
            "schema_version": "1.0",
            "enabled": configured_enabled,
            "configured_enabled": configured_enabled,
            "decision": decision,
            "would_quarantine": would_quarantine,
            "violations": violations,
            "thresholds": {key: round(float(value), 6) for key, value in thresholds.items()},
            "observed": {key: round(float(value), 6) for key, value in observed.items()},
        }

    async def _maybe_promote_scored_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.config.auto_promotion_enabled:
            return {"status": "disabled"}
        return await ResearchLabPromotionController(
            self.config,
            worker_ref=self.worker_ref,
        ).process_scored_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
        )

    async def _maybe_record_score_backfill(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        promotion_result: Mapping[str, Any] | None,
    ) -> None:
        """Phase-5 score backfill (flag default OFF): reconcile the realized
        bundle against the hypothesis's predicted_delta + dev score in an
        append-only calibration row. Best-effort; never blocks scoring."""
        candidate_id = str(candidate.get("candidate_id") or "")
        try:
            from gateway.research_lab.score_backfill import record_score_backfill

            result = await record_score_backfill(
                candidate=candidate,
                score_bundle_row=score_bundle_row,
                score_bundle=score_bundle,
                promotion_result=promotion_result,
                created_by=self.worker_ref,
            )
            if result.get("status") == "recorded":
                logger.info(
                    "research_lab_score_backfill_recorded candidate=%s bundle=%s outcome=%s",
                    compact_ref(candidate_id),
                    compact_ref(result.get("score_bundle_id")),
                    str(result.get("outcome") or "")[:80],
                )
        except Exception as exc:
            logger.warning(
                "research_lab_score_backfill_failed candidate=%s error=%s",
                compact_ref(candidate_id),
                str(exc)[:200],
            )

    # --- §5.2-2 confirmation re-run: worker-side measurement ---------------
    # A candidate held `held_pending_confirmation` cleared every merge gate on
    # its first measurement. The worker re-measures BOTH sides fresh — a full
    # champion (baseline) run AND a candidate run over the same rolling window
    # — as a side measurement (never recorded as the day's benchmark
    # reference), then re-drives the promotion decision. All state derives
    # from promotion events; a crash at any point resumes from the events.

    async def _maybe_run_pending_confirmation(self) -> dict[str, Any] | None:
        holds = await self._find_pending_confirmation_holds()
        if not holds:
            return None
        lease_seconds = _confirmation_lease_seconds()
        budget = confirmation_attempt_budget(self.config)
        for hold in holds:
            candidate_id = str(hold.get("candidate_id") or "")
            score_bundle_id = str(hold.get("source_score_bundle_id") or "")
            if not candidate_id or not score_bundle_id:
                continue
            try:
                state = await load_confirmation_state(
                    candidate_id=candidate_id,
                    score_bundle_id=score_bundle_id,
                )
            except Exception as exc:  # noqa: BLE001 - skip unreadable state, next pass retries
                logger.warning(
                    "research_lab_confirmation_state_load_failed candidate_id=%s error=%s",
                    compact_ref(candidate_id),
                    str(exc)[:200],
                )
                continue
            latest_reason = str(state.get("latest_reason") or "")
            if latest_reason == CONFIRMATION_CLOSED_REASON:
                # A terminal promotion decision already landed for this
                # candidate+bundle; nothing left to drive.
                continue
            try:
                if await candidate_already_promoted(candidate_id):
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "research_lab_confirmation_promoted_check_failed candidate_id=%s error=%s",
                    compact_ref(candidate_id),
                    str(exc)[:200],
                )
                continue
            if latest_reason == CONFIRMATION_RESULT_REASON:
                # Measurement recorded but the decision never landed (crash
                # between record and re-drive): finish the state machine
                # without re-measuring.
                result = await self._redrive_confirmation_decision(
                    candidate_id=candidate_id,
                    score_bundle_id=score_bundle_id,
                    context="recorded_confirmation_pending_decision",
                )
                if result is not None:
                    return {
                        "processed": True,
                        "status": "confirmation_decision_redriven",
                        "candidate_id": candidate_id,
                        "promotion_status": str(result.get("status") or ""),
                    }
                continue
            if latest_reason == CONFIRMATION_STARTED_REASON:
                started = state.get("latest_event") or {}
                age = _status_age_seconds(started.get("created_at"))
                if age is not None and age <= lease_seconds:
                    # In flight elsewhere (or our own crashed predecessor —
                    # wait out the lease either way).
                    continue
            if int(state.get("attempts") or 0) >= budget:
                # Budget exhausted: re-drive so the gate records the terminal
                # rejected_confirmation_failed decision.
                result = await self._redrive_confirmation_decision(
                    candidate_id=candidate_id,
                    score_bundle_id=score_bundle_id,
                    context="confirmation_attempts_exhausted",
                )
                if result is not None:
                    return {
                        "processed": True,
                        "status": "confirmation_decision_redriven",
                        "candidate_id": candidate_id,
                        "promotion_status": str(result.get("status") or ""),
                    }
                continue
            return await self._run_confirmation_attempt(hold=hold, state=state)
        return None

    async def _find_pending_confirmation_holds(self) -> list[dict[str, Any]]:
        """Newest-first held_pending_confirmation events, deduped per
        candidate+bundle. Primary query filters on event_doc->>reason (a plain
        PostgREST JSON filter); falls back to a bounded promotion_checked scan
        filtered in Python if the JSON filter is rejected."""
        columns = (
            "promotion_event_id,candidate_id,source_score_bundle_id,rolling_window_hash,"
            "improvement_points,threshold_points,worker_ref,event_doc,created_at"
        )
        try:
            rows = await select_many(
                "research_lab_candidate_promotion_events",
                columns=columns,
                filters=(
                    ("event_type", "promotion_checked"),
                    ("event_doc->>reason", CONFIRMATION_HOLD_REASON),
                ),
                order_by=(("created_at", True),),
                limit=50,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to a plain-column scan
            logger.warning(
                "research_lab_confirmation_hold_scan_json_filter_failed error=%s",
                str(exc)[:200],
            )
            try:
                rows = await select_many(
                    "research_lab_candidate_promotion_events",
                    columns=columns,
                    filters=(("event_type", "promotion_checked"),),
                    order_by=(("created_at", True),),
                    limit=200,
                )
            except Exception as fallback_exc:  # noqa: BLE001
                logger.warning(
                    "research_lab_confirmation_hold_scan_failed error=%s",
                    str(fallback_exc)[:200],
                )
                return []
            rows = [
                row
                for row in rows
                if isinstance(row.get("event_doc"), Mapping)
                and str(row["event_doc"].get("reason") or "") == CONFIRMATION_HOLD_REASON
            ]
        seen: set[tuple[str, str]] = set()
        holds: list[dict[str, Any]] = []
        for row in rows:
            key = (
                str(row.get("candidate_id") or ""),
                str(row.get("source_score_bundle_id") or ""),
            )
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            holds.append(dict(row))
        return holds

    async def _redrive_confirmation_decision(
        self,
        *,
        candidate_id: str,
        score_bundle_id: str,
        context: str,
    ) -> dict[str, Any] | None:
        """Re-run the promotion decision for a held candidate. The promotion
        controller re-checks every gate (including the recorded confirmation)
        from events, so this is idempotent and crash-safe."""
        candidate = await select_one(
            "research_lab_candidate_evaluation_current",
            filters=(("candidate_id", candidate_id),),
        )
        if not candidate:
            logger.warning(
                "research_lab_confirmation_redrive_candidate_missing candidate_id=%s",
                compact_ref(candidate_id),
            )
            return None
        bundle_row = await select_one(
            "research_evaluation_score_bundle_current",
            filters=(("score_bundle_id", score_bundle_id),),
        )
        score_bundle = (
            bundle_row.get("score_bundle_doc") if isinstance(bundle_row, Mapping) else None
        )
        if not isinstance(score_bundle, Mapping):
            logger.warning(
                "research_lab_confirmation_redrive_bundle_missing candidate_id=%s score_bundle_id=%s",
                compact_ref(candidate_id),
                compact_ref(score_bundle_id),
            )
            return None
        redrive_health_gate = self._scoring_health_gate_result(dict(score_bundle))
        if redrive_health_gate.get("decision") == "quarantine":
            # Fail-closed also on re-driven stored bundles: a measurement that
            # violated the health gate must not promote from the redrive path.
            result = await self._record_scoring_health_quarantined(
                candidate=candidate,
                score_bundle_row=bundle_row,
                score_bundle=dict(score_bundle),
                scoring_health_gate=redrive_health_gate,
            )
        else:
            result = await self._maybe_promote_scored_candidate(
                candidate=candidate,
                score_bundle_row=bundle_row,
                score_bundle=dict(score_bundle),
            )
        if str(result.get("status") or "") == "stale_parent_needs_rescore":
            # Mirror _score_candidate's post-promotion handling: the champion
            # moved while the candidate was held, so queue the rebase (deduped)
            # instead of stranding a scored candidate at rebase_required.
            rebase_result = await self._queue_confirmation_stale_parent_rebase(candidate)
            if rebase_result is not None:
                result = rebase_result
        logger.info(
            format_worker_block(
                "RESEARCH LAB CONFIRMATION DECISION",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Score bundle", compact_ref(score_bundle_id)),
                    ("Context", context),
                    ("Promotion", result.get("status")),
                ),
            )
        )
        status = str(result.get("status") or "")
        if status not in CONFIRMATION_NON_CLOSING_STATUSES:
            # A terminal decision landed: close the confirmation so future
            # passes stop re-driving this candidate+bundle. Best-effort — a
            # lost close only costs one redundant re-drive next pass.
            try:
                await create_candidate_promotion_event(
                    candidate_id=candidate_id,
                    source_score_bundle_id=score_bundle_id,
                    event_type="promotion_checked",
                    promotion_status="checked",
                    worker_ref=self.worker_ref,
                    event_doc={
                        "reason": CONFIRMATION_CLOSED_REASON,
                        "decision_path": "confirmation_rerun_closed",
                        "decision_status": status,
                        "context": context,
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception("research_lab_confirmation_close_event_write_failed")
        return result

    async def _queue_confirmation_stale_parent_rebase(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Queue the stale-parent rebase for a held candidate whose champion
        moved. Deduped on an existing rebase_queued event (the confirmation
        re-drive can repeat after a crash); best-effort — a failure leaves the
        recorded rebase_required promotion event for the operator."""
        candidate_id = str(candidate.get("candidate_id") or "")
        try:
            existing = await select_many(
                "research_lab_candidate_promotion_events",
                columns="promotion_event_id",
                filters=(("candidate_id", candidate_id), ("event_type", "rebase_queued")),
                limit=1,
            )
            if existing:
                return {"status": "stale_parent_rebase_already_queued"}
            active = await load_active_private_model(self.config, register_bootstrap=True)
            evaluation_epoch = await self._resolve_evaluation_epoch()
            return await self._queue_stale_parent_rebase(
                candidate,
                active_artifact=active.artifact,
                candidate_parent=str(candidate.get("parent_artifact_hash") or ""),
                evaluation_epoch=evaluation_epoch,
                elapsed_seconds=0.0,
                stage="confirmation_redrive_parent_changed",
            )
        except Exception as exc:  # noqa: BLE001 - best-effort side effect
            logger.warning(
                "research_lab_confirmation_stale_parent_rebase_failed candidate_id=%s error=%s",
                compact_ref(candidate_id),
                str(exc)[:240],
            )
            return None

    async def _run_confirmation_attempt(
        self,
        *,
        hold: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        candidate_id = str(hold.get("candidate_id") or "")
        score_bundle_id = str(hold.get("source_score_bundle_id") or "")
        candidate = await select_one(
            "research_lab_candidate_evaluation_current",
            filters=(("candidate_id", candidate_id),),
        )
        if not candidate:
            return None
        bundle_row = await select_one(
            "research_evaluation_score_bundle_current",
            filters=(("score_bundle_id", score_bundle_id),),
        )
        score_bundle = (
            bundle_row.get("score_bundle_doc") if isinstance(bundle_row, Mapping) else None
        )
        if not isinstance(score_bundle, Mapping):
            return None
        score_bundle_serving_version = _event_serving_model_version(score_bundle)
        attempt = int(state.get("attempts") or 0) + 1
        candidate_parent = str(candidate.get("parent_artifact_hash") or "")
        first_pass_points = _safe_float(hold.get("improvement_points"), default=0.0)
        threshold_points = _safe_float(
            hold.get("threshold_points"),
            default=float(self.config.improvement_threshold_points),
        )
        started_event = await create_candidate_promotion_event(
            candidate_id=candidate_id,
            source_score_bundle_id=score_bundle_id,
            event_type="promotion_checked",
            promotion_status="checked",
            active_parent_artifact_hash=candidate_parent or None,
            candidate_parent_artifact_hash=candidate_parent or None,
            rolling_window_hash=str(hold.get("rolling_window_hash") or "") or None,
            improvement_points=first_pass_points,
            threshold_points=threshold_points,
            worker_ref=self.worker_ref,
            event_doc={
                "reason": CONFIRMATION_STARTED_REASON,
                "decision_path": "confirmation_rerun_attempt",
                "attempt": attempt,
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
                "serving_model_version": score_bundle_serving_version,
            },
        )
        # Post-write claim confirm (write-then-verify, like
        # _claim_next_candidate): events are append-only with no unique claim
        # constraint, so two workers racing past the pre-check both land a
        # started event. The OLDEST unexpired claim in the contiguous head run
        # of started events wins (ties break on promotion_event_id); the later
        # writer backs off before measuring. Expired claims (dead workers)
        # are ignored, so a stale claim can always be reclaimed.
        fresh = await load_confirmation_state(
            candidate_id=candidate_id,
            score_bundle_id=score_bundle_id,
        )
        if str(fresh.get("latest_reason") or "") != CONFIRMATION_STARTED_REASON:
            # Someone recorded/failed an attempt between our write and this
            # read — the state moved on without us.
            logger.info(
                "research_lab_confirmation_claim_lost candidate_id=%s worker_ref=%s (state advanced)",
                compact_ref(candidate_id),
                self.worker_ref,
            )
            return {"processed": False, "status": "confirmation_claim_lost", "candidate_id": candidate_id}
        lease_seconds = _confirmation_lease_seconds()
        open_claims = []
        for claim in fresh.get("open_claim_events") or []:
            age = _status_age_seconds(claim.get("created_at"))
            if age is not None and age > lease_seconds:
                continue
            open_claims.append(claim)
        my_event_id = str(started_event.get("promotion_event_id") or "")
        winner = min(
            open_claims,
            key=lambda row: (str(row.get("created_at") or ""), str(row.get("promotion_event_id") or "")),
            default=None,
        )
        if winner is None or str(winner.get("promotion_event_id") or "") != my_event_id:
            logger.info(
                "research_lab_confirmation_claim_lost candidate_id=%s worker_ref=%s winner_worker=%s",
                compact_ref(candidate_id),
                self.worker_ref,
                (winner or {}).get("worker_ref"),
            )
            return {"processed": False, "status": "confirmation_claim_lost", "candidate_id": candidate_id}
        start = time.time()
        logger.info(
            format_worker_block(
                "RESEARCH LAB CONFIRMATION RERUN STARTED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Score bundle", compact_ref(score_bundle_id)),
                    ("Attempt", f"{attempt}/{confirmation_attempt_budget(self.config)}"),
                    ("First-pass delta", f"{first_pass_points:.4f}"),
                ),
            )
        )
        try:
            measurement = await self._run_confirmation_measurement(
                candidate=candidate,
                score_bundle=score_bundle,
                hold=hold,
                attempt=attempt,
                start=start,
            )
        except Exception as exc:  # noqa: BLE001 - infra failure re-holds, bounded by budget
            telemetry_session = getattr(self, "_confirmation_telemetry_session", None)
            if isinstance(telemetry_session, ScoringTelemetrySession):
                await telemetry_session.cancel_active(
                    failure_category="confirmation_measurement_failed",
                    error=exc,
                )
                await emit_run_event(
                    telemetry_session.run,
                    "failed",
                    retryable=True,
                    failure_category="confirmation_measurement_failed",
                    error=exc,
                    telemetry_degraded=telemetry_session.degraded,
                )
            self._confirmation_telemetry_session = None
            try:
                await create_candidate_promotion_event(
                    candidate_id=candidate_id,
                    source_score_bundle_id=score_bundle_id,
                    event_type="promotion_checked",
                    promotion_status="checked",
                    active_parent_artifact_hash=candidate_parent or None,
                    candidate_parent_artifact_hash=candidate_parent or None,
                    rolling_window_hash=str(hold.get("rolling_window_hash") or "") or None,
                    improvement_points=first_pass_points,
                    threshold_points=threshold_points,
                    worker_ref=self.worker_ref,
                    event_doc={
                        "reason": CONFIRMATION_ATTEMPT_FAILED_REASON,
                        "decision_path": "confirmation_rerun_attempt",
                        "attempt": attempt,
                        "retryable": True,
                        "unhealthy_measurement": isinstance(exc, ConfirmationMeasurementUnhealthy),
                        "error_diagnostics": _event_error_diagnostics(exc),
                        "elapsed_seconds": round(time.time() - start, 3),
                        "serving_model_version": score_bundle_serving_version,
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception("research_lab_confirmation_attempt_failed_event_write_failed")
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB CONFIRMATION RERUN ATTEMPT FAILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Candidate", compact_ref(candidate_id)),
                        ("Attempt", attempt),
                        ("Error", str(exc)[:300]),
                        ("Elapsed", f"{time.time() - start:.1f}s"),
                    ),
                )
            )
            return {
                "processed": True,
                "status": "confirmation_attempt_failed",
                "candidate_id": candidate_id,
                "attempt": attempt,
            }
        if measurement.get("status") == "skipped_stale_parent":
            # The champion moved while the candidate was held: the fresh delta
            # would be against the wrong parent. Re-drive so the stale-parent
            # path records the proper rebase decision.
            result = await self._redrive_confirmation_decision(
                candidate_id=candidate_id,
                score_bundle_id=score_bundle_id,
                context="confirmation_skipped_stale_parent",
            )
            return {
                "processed": True,
                "status": "confirmation_skipped_stale_parent",
                "candidate_id": candidate_id,
                "promotion_status": str((result or {}).get("status") or ""),
            }
        confirmation_doc = dict(measurement.get("confirmation") or {})
        result_event = await create_candidate_promotion_event(
            candidate_id=candidate_id,
            source_score_bundle_id=score_bundle_id,
            event_type="promotion_checked",
            promotion_status="checked",
            active_parent_artifact_hash=candidate_parent or None,
            candidate_parent_artifact_hash=candidate_parent or None,
            rolling_window_hash=str(confirmation_doc.get("rolling_window_hash") or "") or None,
            improvement_points=first_pass_points,
            threshold_points=threshold_points,
            worker_ref=self.worker_ref,
            event_doc={
                "reason": CONFIRMATION_RESULT_REASON,
                "decision_path": "confirmation_rerun_result",
                "attempt": attempt,
                "confirmation": confirmation_doc,
                # P18: the confirmation pair is calibration gold — two
                # measurements of the SAME artifact. Surface both aggregates
                # and their disagreement as flat, queryable fields (the
                # empirical same-model noise any reward-model training needs).
                "measurement_pair": {
                    "first_pass_delta": round(float(first_pass_points), 6),
                    "confirmation_delta": round(
                        _safe_float(confirmation_doc.get("confirmation_delta"), default=0.0), 6
                    ),
                    "delta_of_deltas": round(
                        _safe_float(confirmation_doc.get("confirmation_delta"), default=0.0)
                        - float(first_pass_points),
                        6,
                    ),
                },
                "serving_model_version": score_bundle_serving_version,
            },
        )
        telemetry_session = getattr(self, "_confirmation_telemetry_session", None)
        if isinstance(telemetry_session, ScoringTelemetrySession):
            await emit_run_event(
                telemetry_session.run,
                "completed",
                promotion_event_id=str(result_event.get("promotion_event_id") or ""),
                telemetry_degraded=telemetry_session.degraded,
                event_doc={"confirmation_attempt": attempt},
            )
        self._confirmation_telemetry_session = None
        logger.info(
            format_worker_block(
                "RESEARCH LAB CONFIRMATION RERUN RECORDED",
                (
                    ("Worker", self.worker_ref),
                    ("Candidate", compact_ref(candidate_id)),
                    ("Attempt", attempt),
                    ("First-pass delta", f"{first_pass_points:.4f}"),
                    ("Confirmation delta", f"{_safe_float(confirmation_doc.get('confirmation_delta'), default=0.0):.4f}"),
                    ("Window match", confirmation_doc.get("window_match")),
                    ("Excluded ICPs", len(confirmation_doc.get("provider_excluded_icp_refs") or [])),
                    ("Elapsed", f"{time.time() - start:.1f}s"),
                ),
            )
        )
        result = await self._redrive_confirmation_decision(
            candidate_id=candidate_id,
            score_bundle_id=score_bundle_id,
            context="confirmation_recorded",
        )
        return {
            "processed": True,
            "status": "confirmation_recorded",
            "candidate_id": candidate_id,
            "attempt": attempt,
            "confirmation_delta": confirmation_doc.get("confirmation_delta"),
            "promotion_status": str((result or {}).get("status") or ""),
        }

    async def _run_confirmation_measurement(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        hold: Mapping[str, Any],
        attempt: int,
        start: float,
    ) -> dict[str, Any]:
        """Fresh baseline + fresh candidate evaluation over the same window.

        A side measurement only: nothing here writes a benchmark bundle or a
        public report, so the day's promotion reference is never replaced.
        Provider-errored ICPs (after the batch retry rounds) are excluded
        SYMMETRICALLY from both sides; too many exclusions make the attempt
        unhealthy (raise) instead of producing a skewed verdict.
        """
        active = await load_active_private_model(self.config, register_bootstrap=True)
        candidate_parent = str(candidate.get("parent_artifact_hash") or "")
        if active.artifact.model_artifact_hash != candidate_parent:
            return {"status": "skipped_stale_parent"}
        candidate_manifest_doc = candidate.get("candidate_model_manifest_doc")
        if not isinstance(candidate_manifest_doc, Mapping):
            raise RuntimeError("confirmation requires candidate_model_manifest_doc")
        candidate_artifact = PrivateModelArtifactManifest.from_mapping(candidate_manifest_doc)

        original_window_hash = str(
            score_bundle.get("icp_set_hash") or hold.get("rolling_window_hash") or ""
        )
        window = await fetch_rolling_icp_window(
            days=self.config.lab_champion_eval_days,
            icps_per_day=self.config.lab_champion_icps_per_day,
            **_rolling_window_fetch_kwargs(self.config),
            allow_partial=self.config.scoring_worker_allow_partial_icp_window,
        )
        window_match = bool(original_window_hash) and window.window_hash == original_window_hash
        if original_window_hash and not window_match:
            reconstructed = await self._reconstruct_rolling_window(original_window_hash)
            if reconstructed is not None:
                window = reconstructed
                window_match = True
        # Idempotent; also satisfies the promotion-event FK when the
        # confirmation ran on a newly rotated window.
        await create_rolling_icp_window(window)

        confirmation_telemetry_session: ScoringTelemetrySession | None = None
        if scoring_telemetry_enabled(self.config):
            telemetry_run = await allocate_scoring_run(
                identity_doc={
                    "run_type": "promotion_confirmation",
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "source_score_bundle_id": str(hold.get("source_score_bundle_id") or ""),
                    "attempt": int(attempt),
                    "rolling_window_hash": window.window_hash,
                    "reference_artifact_hash": active.artifact.model_artifact_hash,
                    "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
                },
                run_type="promotion_confirmation",
                worker_ref=self.worker_ref,
                # Progress is projected per model_role; both sides share the
                # same ICP set, so the denominator is N (not 2N).
                expected_icp_count=len(window.benchmark_items),
                scheduler_type="confirmation_pair",
                minimum_run_attempt=attempt,
                candidate_id=str(candidate.get("candidate_id") or ""),
                source_score_bundle_id=str(hold.get("source_score_bundle_id") or ""),
                rolling_window_hash=window.window_hash,
                reference_artifact_hash=active.artifact.model_artifact_hash,
                reference_manifest_hash=active.artifact.manifest_hash,
                candidate_artifact_hash=candidate_artifact.model_artifact_hash,
                candidate_manifest_hash=candidate_artifact.manifest_hash,
                evaluation_epoch=int(score_bundle.get("evaluation_epoch") or 0),
            )
            if telemetry_run is not None:
                confirmation_telemetry_session = ScoringTelemetrySession(telemetry_run)
                for item_index, item in enumerate(window.benchmark_items):
                    item_ref = str(item.get("icp_ref") or item.get("icp_hash") or "")
                    for model_role in ("reference", "candidate"):
                        await confirmation_telemetry_session.plan(
                            icp_ref=item_ref,
                            icp_hash=str(item.get("icp_hash") or ""),
                            icp_ordinal=item_index,
                            model_role=model_role,
                        )
                await emit_run_event(confirmation_telemetry_session.run, "assigned")
                await emit_run_event(confirmation_telemetry_session.run, "started")
        self._confirmation_telemetry_session = confirmation_telemetry_session

        # Trace scope (§5.4 + in-container capture): keyed per candidate,
        # attempt, and side so confirmation docs never collide across attempts
        # or candidates. Carried via an instance attribute (not a new call
        # kwarg) so the side runner's call signature stays stable.
        self._confirmation_trace_scope = {
            "candidate_id": str(candidate.get("candidate_id") or "candidate"),
            "attempt": int(attempt),
            "manifest_uri": str(active.artifact.manifest_uri or ""),
            "evaluation_epoch": int(score_bundle.get("evaluation_epoch") or 0),
        }
        confirmation_heartbeat_stop = asyncio.Event()
        confirmation_heartbeat_task = (
            asyncio.create_task(
                confirmation_telemetry_session.heartbeat_loop(confirmation_heartbeat_stop)
            )
            if confirmation_telemetry_session is not None
            else None
        )
        try:
            champion_summaries, champion_stats = await self._run_confirmation_side(
                artifact=active.artifact,
                window=window,
                mode_label="confirmation_baseline",
                run_start=start,
            )
            candidate_summaries, candidate_stats = await self._run_confirmation_side(
                artifact=candidate_artifact,
                window=window,
                mode_label="confirmation_candidate",
                run_start=start,
            )
        finally:
            self._confirmation_trace_scope = None
            if confirmation_heartbeat_task is not None:
                confirmation_heartbeat_stop.set()
                await confirmation_heartbeat_task
        baseline_scores, baseline_unresolved, baseline_nonempty = _collect_confirmation_scores(
            champion_summaries
        )
        candidate_scores, candidate_unresolved, candidate_nonempty = _collect_confirmation_scores(
            candidate_summaries
        )
        if confirmation_telemetry_session is not None:
            for summary in champion_summaries:
                if not summary.get("_runtime_error"):
                    await confirmation_telemetry_session.complete_result(
                        summary,
                        model_role="reference",
                        checkpoint_persisted=False,
                        outcome="confirmation_measurement",
                    )
            for summary in candidate_summaries:
                if not summary.get("_runtime_error"):
                    await confirmation_telemetry_session.complete_result(
                        summary,
                        model_role="candidate",
                        checkpoint_persisted=False,
                        outcome="confirmation_measurement",
                    )
        if baseline_nonempty <= 0:
            raise PrivateModelRuntimeError(
                f"confirmation baseline returned zero companies across all {len(window.benchmark_items)} ICPs"
            )
        if candidate_nonempty <= 0:
            raise PrivateModelRuntimeError(
                f"confirmation candidate returned zero companies across all {len(window.benchmark_items)} ICPs"
            )
        excluded = sorted(set(baseline_unresolved) | set(candidate_unresolved))
        max_unresolved = _baseline_max_unresolved_icps()
        if len(excluded) > max_unresolved:
            raise ConfirmationMeasurementUnhealthy(
                "confirmation_measurement_unresolved_provider_errors: "
                f"unresolved={len(excluded)} max={max_unresolved} "
                f"baseline_unresolved={len(baseline_unresolved)} candidate_unresolved={len(candidate_unresolved)}"
            )
        excluded_set = set(excluded)
        included_refs = [
            ref
            for ref in baseline_scores
            if ref in candidate_scores and ref not in excluded_set
        ]
        if not included_refs:
            raise ConfirmationMeasurementUnhealthy(
                "confirmation_measurement_no_icps_left_after_exclusions"
            )
        baseline_aggregate = _average([baseline_scores[ref] for ref in included_refs])
        candidate_total = _average([candidate_scores[ref] for ref in included_refs])
        confirmation_delta = candidate_total - baseline_aggregate
        confirmation = {
            "schema_version": "1.0",
            "measurement_type": "confirmation_rerun_side_measurement",
            "rolling_window_hash": window.window_hash,
            "original_window_hash": original_window_hash,
            "window_match": window_match,
            "selected_icp_count": len(window.item_refs),
            "included_icp_count": len(included_refs),
            "provider_excluded_icp_refs": excluded,
            "baseline_aggregate_score": round(baseline_aggregate, 6),
            "candidate_total_score": round(candidate_total, 6),
            "confirmation_delta": round(confirmation_delta, 6),
            "per_icp_baseline_scores": {
                ref: round(baseline_scores[ref], 6) for ref in sorted(baseline_scores)
            },
            "per_icp_candidate_scores": {
                ref: round(candidate_scores[ref], 6) for ref in sorted(candidate_scores)
            },
            "baseline_retry": champion_stats,
            "candidate_retry": candidate_stats,
            "champion_artifact_hash": active.artifact.model_artifact_hash,
            "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
            "attempt": attempt,
            "worker_ref": self.worker_ref,
            "elapsed_seconds": round(time.time() - start, 3),
        }
        return {"status": "measured", "confirmation": confirmation}

    async def _run_confirmation_side(
        self,
        *,
        artifact: PrivateModelArtifactManifest,
        window: Any,
        mode_label: str,
        run_start: float,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """One side (champion or candidate) of a confirmation measurement,
        through the same batch machinery as the parallel baseline — provider
        retry rounds, benchmark Exa isolation, and §0-N6 scorer isolation
        included, so both sides run under the identical execution regime."""
        confirmation_scope = getattr(self, "_confirmation_trace_scope", None)
        confirmation_scope = confirmation_scope if isinstance(confirmation_scope, Mapping) else {}
        confirmation_attempt = None
        try:
            if confirmation_scope.get("attempt") is not None:
                confirmation_attempt = int(confirmation_scope.get("attempt"))
        except (TypeError, ValueError):
            confirmation_attempt = None
        confirmation_candidate_id = str(confirmation_scope.get("candidate_id") or "")
        telemetry_session = getattr(self, "_confirmation_telemetry_session", None)
        if not isinstance(telemetry_session, ScoringTelemetrySession):
            telemetry_session = None
        telemetry_model_role = (
            "reference" if mode_label == "confirmation_baseline" else "candidate"
        )
        confirmation_model_kind = (
            "private" if mode_label == "confirmation_baseline" else "candidate"
        )
        runner = AttestedPrivateModelRunnerV2(
            artifact=artifact,
            spec=DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                env_passthrough=self._private_model_env_passthrough(),
                extra_env=self._with_provider_cost_evaluation_scope(
                    self._private_baseline_scoring_env(),
                    run_type="promotion_confirmation",
                    rolling_window_hash=window.window_hash,
                    artifact_hash=artifact.model_artifact_hash,
                    candidate_id=confirmation_candidate_id,
                    confirmation_attempt=confirmation_attempt,
                    side=mode_label,
                    started_at=run_start,
                ),
            ),
            model_kind=confirmation_model_kind,
            worker_index=self.config.scoring_worker_index,
            epoch_id=int(confirmation_scope.get("evaluation_epoch") or 0),
            parent_graphs=await _attested_model_parent_graphs(
                model_kind=confirmation_model_kind,
                artifact=artifact,
                candidate_id=confirmation_candidate_id,
                epoch_id=int(confirmation_scope.get("evaluation_epoch") or 0),
            ),
        )
        retry_runner = AttestedPrivateModelRunnerV2(
            artifact=artifact,
            spec=DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                env_passthrough=self._private_model_env_passthrough(),
                extra_env=self._with_provider_cost_evaluation_scope(
                    self._private_baseline_retry_scoring_env(),
                    run_type="promotion_confirmation",
                    rolling_window_hash=window.window_hash,
                    artifact_hash=artifact.model_artifact_hash,
                    candidate_id=confirmation_candidate_id,
                    confirmation_attempt=confirmation_attempt,
                    side=mode_label,
                    started_at=run_start,
                ),
                # The first-pass runner already pulled this digest.
                pull_before_run=False,
            ),
            model_kind=confirmation_model_kind,
            worker_index=self.config.scoring_worker_index,
            epoch_id=int(confirmation_scope.get("evaluation_epoch") or 0),
            parent_graphs=runner.parent_graphs,
        )
        scorer = QualificationStyleCompanyScorer(
            attested_epoch_id=int(confirmation_scope.get("evaluation_epoch") or 0),
            attested_purpose="research_lab.rebenchmark.v1",
            attested_provider_profile="benchmark_scorer",
        )
        summaries, stats = await self._run_baseline_batch(
            runner=runner,
            retry_runner=retry_runner,
            scorer=scorer,
            window=window,
            run_start=run_start,
            mode_label=mode_label,
            trace_context=self._confirmation_side_trace_context(mode_label=mode_label),
            telemetry_session=telemetry_session,
            telemetry_model_role=telemetry_model_role,
        )
        return summaries, {
            "retried": int(stats.get("retried") or 0),
            "recovered": int(stats.get("recovered") or 0),
            "unresolved": int(stats.get("unresolved") or 0),
        }

    def _confirmation_side_trace_context(self, *, mode_label: str) -> dict[str, Any] | None:
        """Trace scope for one confirmation side, derived from the pending
        confirmation's candidate/attempt (set by _run_confirmation_measurement).
        None outside a confirmation measurement — capture then stays off."""
        scope = getattr(self, "_confirmation_trace_scope", None)
        if not isinstance(scope, Mapping):
            return None
        side = "champion" if mode_label == "confirmation_baseline" else "candidate"
        candidate_ref = _trace_path_segment(scope.get("candidate_id"), fallback="candidate")
        return self._baseline_trace_context(
            context_ref=f"confirmation-{candidate_ref}-a{int(scope.get('attempt') or 0)}-{side}",
            manifest_uri=str(scope.get("manifest_uri") or ""),
            run_type=f"promotion_confirmation_{side}",
        )

    async def _reconstruct_rolling_window(self, window_hash: str) -> Any | None:
        """Best-effort re-materialization of a stored rolling window.

        The window selection is deterministic over its set rows, so re-fetching
        the original set_ids reproduces the identical window unless the set
        contents changed — verified by requiring the reconstructed hash to
        match. Any failure falls back to the caller's current window (the
        confirmation stays a same-window paired comparison either way)."""
        try:
            row = await select_one(
                "research_lab_rolling_icp_windows",
                filters=(("rolling_window_hash", window_hash),),
            )
            doc = row.get("window_doc") if isinstance(row, Mapping) else None
            if not isinstance(doc, Mapping):
                return None
            set_ids = [
                int(entry.get("set_id"))
                for entry in (doc.get("sets") or [])
                if isinstance(entry, Mapping) and entry.get("set_id") is not None
            ]
            if not set_ids:
                return None
            rows = await select_many(
                "qualification_private_icp_sets",
                columns="set_id,icps,icp_set_hash,active_from,active_until,is_active",
                filters=(("set_id", "in", tuple(set_ids)),),
                limit=max(len(set_ids), 10),
            )
            if len(rows) < len(set_ids):
                return None
            window = reconstruct_icp_window_from_doc(rows, doc)
            if window.window_hash != window_hash:
                logger.info(
                    "research_lab_confirmation_window_reconstruct_hash_mismatch expected=%s got=%s",
                    compact_ref(window_hash),
                    compact_ref(window.window_hash),
                )
                return None
            return window
        except Exception as exc:  # noqa: BLE001 - best-effort; caller keeps the current window
            logger.warning(
                "research_lab_confirmation_window_reconstruct_failed error=%s",
                str(exc)[:200],
            )
            return None

    async def _maybe_rebase_stale_candidate_before_scoring(
        self,
        candidate: Mapping[str, Any],
        *,
        evaluation_epoch: int,
        elapsed_seconds: Any,
    ) -> dict[str, Any]:
        active = await load_active_private_model(self.config, register_bootstrap=True)
        active_parent = active.artifact.model_artifact_hash
        candidate_parent = str(candidate.get("parent_artifact_hash") or "")
        candidate_id = str(candidate["candidate_id"])
        candidate_kind = str(candidate.get("candidate_kind") or "patch")
        if candidate_kind != "image_build":
            base_event_doc = {
                "action": "legacy_patch_candidate_rejected_before_scoring",
                "candidate_kind": candidate_kind,
                "active_parent_artifact_hash": active_parent,
                "candidate_parent_artifact_hash": candidate_parent,
                "evaluation_epoch": int(evaluation_epoch),
                "worker_ref": self.worker_ref,
                "proxy_ref_hash": self.proxy_ref_hash,
            }
            await create_candidate_promotion_event(
                candidate_id=candidate_id,
                epoch_id=int(evaluation_epoch),
                event_type="unsupported_candidate_kind",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                worker_ref=self.worker_ref,
                event_doc={**base_event_doc, "stage": "before_scoring"},
            )
            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_type="rejected",
                candidate_status="rejected",
                evaluator_ref=self.worker_ref,
                reason="legacy_patch_candidate_unsupported",
                event_doc={**base_event_doc, "elapsed_seconds": elapsed_seconds()},
            )
            await create_scoring_dispatch_event(
                dispatch_type="candidate_scoring",
                dispatch_status="rejected",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                candidate_id=candidate_id,
                run_id=str(candidate["run_id"]),
                ticket_id=str(candidate["ticket_id"]),
                event_doc={**base_event_doc, "reason": "legacy_patch_candidate_unsupported"},
            )
            return {"status": "legacy_patch_candidate_unsupported"}

        if candidate_parent == active_parent:
            return {"status": "current_parent"}

        return await self._queue_stale_parent_rebase(
            candidate,
            active_artifact=active.artifact,
            candidate_parent=candidate_parent,
            evaluation_epoch=evaluation_epoch,
            elapsed_seconds=elapsed_seconds(),
            stage="before_scoring_parent_changed",
        )

    async def _queue_stale_parent_rebase(
        self,
        candidate: Mapping[str, Any],
        *,
        active_artifact: PrivateModelArtifactManifest,
        candidate_parent: str,
        evaluation_epoch: int,
        elapsed_seconds: float,
        stage: str,
        stale_progress: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_parent = active_artifact.model_artifact_hash
        candidate_id = str(candidate["candidate_id"])
        rebase_depth = _stale_parent_rebase_depth(candidate)
        next_rebase_depth = rebase_depth + 1
        base_event_doc = {
            "action": "image_build_candidate_parent_changed",
            "active_parent_artifact_hash": active_parent,
            "candidate_parent_artifact_hash": candidate_parent,
            "evaluation_epoch": int(evaluation_epoch),
            "worker_ref": self.worker_ref,
            "proxy_ref_hash": self.proxy_ref_hash,
            "stage": stage,
            "rebase_depth": rebase_depth,
            "next_rebase_depth": next_rebase_depth,
            "reimbursement_preserved": True,
            "reimbursement_source": "hosted_loop_completion",
        }
        if stale_progress:
            base_event_doc["stale_progress"] = _stale_parent_progress_doc(stale_progress)
        if not self.config.stale_parent_rebase_enabled:
            await self._reject_stale_parent_candidate(
                candidate,
                base_event_doc=base_event_doc,
                reason="stale_parent_needs_rescore",
                elapsed_seconds=elapsed_seconds,
            )
            return {"status": "stale_parent_needs_rescore"}
        if rebase_depth >= self.config.stale_parent_rebase_max_depth:
            await self._reject_stale_parent_candidate(
                candidate,
                base_event_doc={
                    **base_event_doc,
                    "failure_class": "stale_parent_rebase_depth_exceeded",
                    "max_rebase_depth": self.config.stale_parent_rebase_max_depth,
                },
                reason="stale_parent_rebase_failed",
                elapsed_seconds=elapsed_seconds,
            )
            recovery_result = await self._recover_stale_parent_rebase_failed_candidate(candidate)
            return {
                "status": "stale_parent_rebase_failed",
                "error": "stale_parent_rebase_depth_exceeded",
                "recovery_result": recovery_result,
            }

        try:
            draft = await asyncio.to_thread(
                self._draft_from_stale_candidate,
                candidate,
            )
            parent_graphs: tuple[dict[str, Any], ...] = ()
            if not legacy_v1_enabled():
                candidate_manifest = candidate.get("candidate_model_manifest")
                if not isinstance(candidate_manifest, Mapping):
                    raise CodeEditBuildError(
                        "stale candidate is missing its model manifest"
                    )
                candidate_artifact = PrivateModelArtifactManifest.from_mapping(
                    candidate_manifest
                )
                parent_graphs = await _attested_model_parent_graphs(
                    model_kind="candidate",
                    artifact=candidate_artifact,
                    candidate_id=candidate_id,
                )
                if len(parent_graphs) != 1:
                    raise CodeEditBuildError(
                        "stale candidate has no unique measured ancestry"
                    )
        except Exception as exc:
            await self._reject_stale_parent_candidate(
                candidate,
                base_event_doc={
                    **base_event_doc,
                    "failure_class": "stale_parent_rebase_failed",
                    "error_diagnostics": _event_error_diagnostics(exc),
                    "error_hash": sha256_json({"error": str(exc)}),
                },
                reason="stale_parent_rebase_failed",
                elapsed_seconds=elapsed_seconds,
            )
            recovery_result = await self._recover_stale_parent_rebase_failed_candidate(candidate)
            return {
                "status": "stale_parent_rebase_failed",
                "error": str(exc)[:300],
                "recovery_result": recovery_result,
            }

        try:
            if legacy_v1_enabled():
                try:
                    build = await asyncio.to_thread(
                        CodeEditCandidateBuilder(self.config).build,
                        draft=draft,
                        parent_artifact=active_artifact,
                        run_id=str(candidate["run_id"]),
                        candidate_index=await self._next_rebase_candidate_index(
                            str(candidate["run_id"])
                        ),
                    )
                    repair_used = False
                except CodeEditPatchApplyError as exc:
                    draft, build = await self._repair_and_build_stale_candidate(
                        candidate,
                        active_artifact=active_artifact,
                        original_error=exc,
                        run_id=str(candidate["run_id"]),
                    )
                    repair_used = True
                rebase_authority = None
            else:
                rebase_authority = await attest_stale_parent_rebase_v2(
                    candidate=candidate,
                    original_draft=draft,
                    active_artifact=active_artifact,
                    candidate_receipt_graph=parent_graphs[0],
                    epoch_id=int(evaluation_epoch),
                    worker_index=int(self.config.scoring_worker_index or 0),
                    require_egress_proxy=bool(
                        self.config.scoring_worker_require_proxy
                    ),
                    source_bundle_timeout_seconds=(
                        self.config.code_edit_build_timeout_seconds
                    ),
                )
                draft = rebase_authority.draft
                repair_used = rebase_authority.repair_used
                build = await asyncio.to_thread(
                    CodeEditCandidateBuilder(self.config).build,
                    draft=draft,
                    parent_artifact=active_artifact,
                    run_id=str(candidate["run_id"]),
                    candidate_index=await self._next_rebase_candidate_index(
                        str(candidate["run_id"])
                    ),
                )
        except Exception as repair_exc:
            await self._reject_stale_parent_candidate(
                candidate,
                base_event_doc={
                    **base_event_doc,
                    "failure_class": "stale_parent_rebase_repair_failed",
                    "error_diagnostics": _event_error_diagnostics(repair_exc),
                    "error_hash": sha256_json({"error": str(repair_exc)}),
                },
                reason="stale_parent_rebase_failed",
                elapsed_seconds=elapsed_seconds,
            )
            recovery_result = await self._recover_stale_parent_rebase_failed_candidate(
                candidate
            )
            return {
                "status": "stale_parent_rebase_failed",
                "error": str(repair_exc)[:300],
                "recovery_result": recovery_result,
            }

        rebase_build_doc = {
            **build.build_doc,
            **(
                {
                    "conditional_validation_policy": dict(
                        (candidate.get("candidate_build_doc") or {}).get(
                            "conditional_validation_policy"
                        )
                    )
                }
                if isinstance(candidate.get("candidate_build_doc"), Mapping)
                and isinstance(
                    (candidate.get("candidate_build_doc") or {}).get(
                        "conditional_validation_policy"
                    ),
                    Mapping,
                )
                else {}
            ),
            # Preserve the source candidate's loop-node linkage so the rebased
            # candidate's score bundle keeps the deterministic
            # execution_trace:<uuid5> ref (bundle→trace join survives rebases).
            **(
                {"loop_node_id": str((candidate.get("candidate_build_doc") or {}).get("loop_node_id") or "")}
                if isinstance(candidate.get("candidate_build_doc"), Mapping)
                and (candidate.get("candidate_build_doc") or {}).get("loop_node_id")
                else {}
            ),
            "stale_parent_rebase": {
                "schema_version": "1.0",
                "source_candidate_id": candidate_id,
                "source_parent_artifact_hash": candidate_parent,
                "rebased_parent_artifact_hash": active_parent,
                "repair_used": repair_used,
                "stage": stage,
                "depth": next_rebase_depth,
                "max_depth": self.config.stale_parent_rebase_max_depth,
            },
        }
        request = ResearchLabCandidateArtifactCreateRequest(
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            receipt_id=str(candidate.get("receipt_id") or "") or None,
            miner_hotkey=str(candidate["miner_hotkey"]),
            island=str(candidate.get("island") or "generalist"),
            candidate_kind="image_build",
            private_model_manifest=active_artifact.to_dict(),
            candidate_patch_manifest=build.code_edit_manifest,
            candidate_model_manifest=build.candidate_model_manifest.to_dict(),
            candidate_source_diff_hash=build.source_diff_hash,
            candidate_build_doc=rebase_build_doc,
            hypothesis_doc=dict(candidate.get("hypothesis_doc") or {}),
            redacted_public_summary=str(candidate.get("redacted_public_summary") or draft.redacted_summary or ""),
        )
        derived_candidate, _event = await create_candidate_artifact(request)
        derived_candidate_id = str(derived_candidate["candidate_id"])
        if not legacy_v1_enabled():
            if rebase_authority is None:
                raise CodeEditBuildError(
                    "stale-parent rebase authority is unavailable after build"
                )
            authority_receipt = (
                rebase_authority.authority.get("execution_receipt")
                or rebase_authority.authority.get("receipt")
            )
            if not isinstance(authority_receipt, Mapping):
                raise CodeEditBuildError(
                    "stale-parent rebase receipt is unavailable after build"
                )
            from gateway.research_lab.attested_v2_store import (
                persist_business_artifact_links_v2,
            )

            await persist_business_artifact_links_v2(
                receipt_hash=str(authority_receipt.get("receipt_hash") or ""),
                artifacts=(
                    {
                        "artifact_kind": "candidate_model",
                        "artifact_ref": derived_candidate_id,
                        "artifact_hash": build.candidate_model_manifest.manifest_hash,
                    },
                    {
                        "artifact_kind": "candidate_patch",
                        "artifact_ref": derived_candidate_id,
                        "artifact_hash": sha256_json(dict(build.code_edit_manifest)),
                    },
                    {
                        "artifact_kind": "candidate_source",
                        "artifact_ref": derived_candidate_id,
                        "artifact_hash": build.source_diff_hash,
                    },
                ),
            )
        await create_candidate_promotion_event(
            candidate_id=candidate_id,
            derived_candidate_id=derived_candidate_id,
            event_type="rebase_queued",
            promotion_status="rebenchmarking",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            worker_ref=self.worker_ref,
            event_doc={
                **base_event_doc,
                "derived_candidate_id": derived_candidate_id,
                "derived_candidate_artifact_hash": build.candidate_model_manifest.model_artifact_hash,
                "derived_source_diff_hash": build.source_diff_hash,
                "repair_used": repair_used,
            },
        )
        await create_candidate_evaluation_event(
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_type="rejected",
            candidate_status="rejected",
            evaluator_ref=self.worker_ref,
            reason="stale_parent_rebased_to_current",
            event_doc={
                **base_event_doc,
                "derived_candidate_id": derived_candidate_id,
                "derived_source_diff_hash": build.source_diff_hash,
                "elapsed_seconds": elapsed_seconds,
            },
        )
        await create_scoring_dispatch_event(
            dispatch_type="candidate_scoring",
            dispatch_status="rejected",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_doc={
                **base_event_doc,
                "derived_candidate_id": derived_candidate_id,
                "reason": "stale_parent_rebased_to_current",
            },
        )
        return {
            "status": "stale_parent_rebased_to_current",
            "derived_candidate_id": derived_candidate_id,
            "repair_used": repair_used,
        }

    async def _recover_stale_parent_rebase_failed_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return {"ok": False, "status": "candidate_missing_id"}
        try:
            from .recovery import recover_rebase_failed_candidates

            result = await recover_rebase_failed_candidates(
                candidate_ids=[candidate_id],
                dry_run=False,
                max_batch_size=1,
                regenerate=True,
                actor_ref=self.worker_ref,
            )
        except Exception as exc:  # noqa: BLE001 - recovery must not hide the original rebase failure
            logger.warning(
                "research_lab_stale_parent_rebase_failed_auto_recovery_failed candidate_id=%s error=%s",
                compact_ref(candidate_id),
                _short_error(exc),
            )
            return {"ok": False, "status": "auto_recovery_exception", "error": _short_error(exc)}
        if int(result.get("regenerated") or 0):
            logger.info(
                "research_lab_stale_parent_rebase_failed_auto_regenerated candidate_id=%s regenerated=%s",
                compact_ref(candidate_id),
                result.get("regenerated"),
            )
        return result

    async def _reject_stale_parent_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        base_event_doc: Mapping[str, Any],
        reason: str,
        elapsed_seconds: float,
    ) -> None:
        candidate_id = str(candidate["candidate_id"])
        active_parent = str(base_event_doc.get("active_parent_artifact_hash") or "")
        candidate_parent = str(base_event_doc.get("candidate_parent_artifact_hash") or "")
        event_doc = {
            "reimbursement_preserved": True,
            "reimbursement_source": "hosted_loop_completion",
            **dict(base_event_doc),
        }
        await create_candidate_promotion_event(
            candidate_id=candidate_id,
            event_type="stale_parent_detected",
            promotion_status="rejected" if reason == "stale_parent_rebase_failed" else "rebase_required",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            worker_ref=self.worker_ref,
            event_doc=event_doc,
        )
        await create_candidate_evaluation_event(
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_type="rejected",
            candidate_status="rejected",
            evaluator_ref=self.worker_ref,
            reason=reason,
            event_doc={**event_doc, "elapsed_seconds": elapsed_seconds},
        )
        await create_scoring_dispatch_event(
            dispatch_type="candidate_scoring",
            dispatch_status="rejected",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            candidate_id=candidate_id,
            run_id=str(candidate["run_id"]),
            ticket_id=str(candidate["ticket_id"]),
            event_doc={**event_doc, "reason": reason},
        )

    def _draft_from_stale_candidate(self, candidate: Mapping[str, Any]) -> CodeEditDraft:
        unified_diff = _load_candidate_source_diff(candidate)
        patch = candidate.get("candidate_patch_manifest")
        patch_doc = patch.get("patch_doc") if isinstance(patch, Mapping) else {}
        if not isinstance(patch_doc, Mapping):
            patch_doc = {}
        hypothesis = candidate.get("hypothesis_doc") if isinstance(candidate.get("hypothesis_doc"), Mapping) else {}
        patch_summary = str(patch.get("redacted_summary") or "") if isinstance(patch, Mapping) else ""
        target_files = patch_doc.get("target_files")
        if not isinstance(target_files, list) or not target_files:
            target_files = sorted(_extract_diff_paths_safe(unified_diff))
        return CodeEditDraft(
            failure_mode=str(hypothesis.get("failure_mode") or "Previously generated miner code edit")[:700],
            mechanism=str(hypothesis.get("mechanism") or patch_doc.get("expected_improvement") or "")[:1000],
            expected_improvement=str(hypothesis.get("expected_improvement") or patch_doc.get("expected_improvement") or "")[:1000],
            risk=str(hypothesis.get("risk") or patch_doc.get("risk") or "")[:700],
            lane=str(patch_doc.get("lane") or "stale_parent_rebase")[:80],
            target_files=tuple(str(path) for path in target_files),
            unified_diff=unified_diff,
            redacted_summary=str(candidate.get("redacted_public_summary") or patch_summary)[:1200],
            test_plan=str(patch_doc.get("test_plan") or "Run the standard Research Lab private test command.")[:1200],
            rollback_plan=str(patch_doc.get("rollback_plan") or "Discard the rebased candidate image.")[:1200],
            predicted_delta=float(hypothesis.get("predicted_delta") or 1.0),
        )

    async def _next_rebase_candidate_index(self, run_id: str) -> int:
        rows = await select_many(
            "research_lab_candidate_evaluation_current",
            columns="candidate_id",
            filters=(("run_id", str(run_id)),),
            limit=1000,
        )
        return 1000 + len(rows)

    async def _candidate_private_holdout_gate(
        self,
        *,
        artifact: PrivateModelArtifactManifest,
        window_hash: str,
    ) -> dict[str, Any]:
        conditional_policy = self.config.conditional_validation_policy()
        expected_policy_hash = (
            str(conditional_policy.to_dict()["policy_hash"])
            if conditional_policy.enabled
            else ""
        )
        rows = await select_many(
            "research_lab_private_model_benchmark_current",
            columns=(
                "benchmark_bundle_id,private_model_manifest_hash,rolling_window_hash,"
                "benchmark_quality,evaluation_epoch,score_summary_doc,current_benchmark_status,created_at"
            ),
            filters=(
                ("private_model_manifest_hash", artifact.manifest_hash),
                ("rolling_window_hash", window_hash),
                ("current_benchmark_status", "completed"),
            ),
            order_by=(("created_at", True),),
            limit=10,
        )
        for row in rows:
            if (
                not _private_benchmark_row_is_valid(row)
                or not _private_benchmark_matches_policy(
                    row,
                    expected_policy_hash=expected_policy_hash,
                )
            ):
                continue
            gate = _private_holdout_gate_from_baseline_row(row)
            if gate:
                return gate
        raise CandidateBaselineNotReady(
            "matching_completed_private_baseline_required_before_candidate_private_holdout: "
            f"manifest={compact_ref(artifact.manifest_hash)} window={compact_ref(window_hash)}"
        )

    async def _daily_candidate_scoring_window_and_gate(
        self,
        *,
        artifact: PrivateModelArtifactManifest,
        now: datetime | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return the UTC day's frozen baseline window and private holdout gate.

        Candidate scoring must compete against the private baseline that owns
        the UTC day, not a freshly reselected "latest" ICP window. That keeps the
        daily baseline stable even if ICP rows are inserted or repaired after
        midnight.
        """
        today = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()
        conditional_policy = self.config.conditional_validation_policy()
        expected_policy_hash = (
            str(conditional_policy.to_dict()["policy_hash"])
            if conditional_policy.enabled
            else ""
        )
        rows = await select_many(
            "research_lab_private_model_benchmark_current",
            columns=(
                "benchmark_bundle_id,private_model_manifest_hash,rolling_window_hash,"
                "benchmark_quality,evaluation_epoch,score_summary_doc,current_benchmark_status,created_at"
            ),
            filters=(
                ("benchmark_date", today),
                ("private_model_manifest_hash", artifact.manifest_hash),
                ("current_benchmark_status", "completed"),
            ),
            order_by=(("created_at", True),),
            limit=25,
        )
        for row in rows:
            if (
                not _private_benchmark_row_is_valid(row)
                or not _private_benchmark_matches_policy(
                    row,
                    expected_policy_hash=expected_policy_hash,
                )
            ):
                continue
            gate = _private_holdout_gate_from_baseline_row(row)
            window_hash = str(row.get("rolling_window_hash") or "")
            if not gate or not window_hash:
                continue
            window = await self._reconstruct_rolling_window(window_hash)
            if window is None:
                raise CandidateBaselineNotReady(
                    "same_day_private_baseline_window_reconstruction_required_before_candidate_scoring: "
                    f"manifest={compact_ref(artifact.manifest_hash)} window={compact_ref(window_hash)}"
                )
            return window, gate
        raise CandidateBaselineNotReady(
            "same_day_completed_private_baseline_required_before_candidate_scoring: "
            f"date={today} manifest={compact_ref(artifact.manifest_hash)}"
        )

    async def _candidate_scoring_start_gate(self, *, now: datetime | None = None) -> dict[str, Any]:
        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        quiet_start = int(
            getattr(
                self.config,
                "candidate_scoring_quiet_start_utc_seconds",
                _candidate_scoring_quiet_start_utc_seconds(),
            )
            or 0
        )
        quiet_start = max(0, min(86399, quiet_start))
        target_date = _candidate_baseline_target_date(now_utc, quiet_start_seconds=quiet_start)
        target_now = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
        seconds_since_midnight = _utc_seconds_since_day_start(now_utc)
        quiet_window_active = seconds_since_midnight >= quiet_start
        try:
            active = await load_active_private_model(self.config, register_bootstrap=True)
            artifact = active.artifact
        except Exception as exc:  # noqa: BLE001 - fail closed before assigning a candidate
            return {
                "available": False,
                "reason": "candidate_scoring_active_model_unavailable",
                "now_utc": now_utc.isoformat(),
                "target_benchmark_date": target_date,
                "quiet_start_utc_seconds": quiet_start,
                "quiet_window_active": quiet_window_active,
                "error": _short_error(exc),
            }
        try:
            window, gate = await self._daily_candidate_scoring_window_and_gate(
                artifact=artifact,
                now=target_now,
            )
        except CandidateBaselineNotReady as exc:
            return {
                "available": False,
                "reason": (
                    "candidate_scoring_next_daily_baseline_not_ready"
                    if quiet_window_active
                    else "candidate_scoring_daily_baseline_not_ready"
                ),
                "now_utc": now_utc.isoformat(),
                "target_benchmark_date": target_date,
                "quiet_start_utc_seconds": quiet_start,
                "quiet_window_active": quiet_window_active,
                "private_model_manifest_hash": artifact.manifest_hash,
                "private_model_artifact_hash": artifact.model_artifact_hash,
                "error": _short_error(exc),
            }
        except Exception as exc:  # noqa: BLE001 - fail closed before assigning a candidate
            return {
                "available": False,
                "reason": "candidate_scoring_daily_baseline_gate_unavailable",
                "now_utc": now_utc.isoformat(),
                "target_benchmark_date": target_date,
                "quiet_start_utc_seconds": quiet_start,
                "quiet_window_active": quiet_window_active,
                "private_model_manifest_hash": artifact.manifest_hash,
                "private_model_artifact_hash": artifact.model_artifact_hash,
                "error": _short_error(exc),
            }
        return {
            "available": True,
            "reason": "",
            "now_utc": now_utc.isoformat(),
            "target_benchmark_date": target_date,
            "quiet_start_utc_seconds": quiet_start,
            "quiet_window_active": quiet_window_active,
            "private_model_manifest_hash": artifact.manifest_hash,
            "private_model_artifact_hash": artifact.model_artifact_hash,
            "rolling_window_hash": getattr(window, "window_hash", ""),
            "baseline_benchmark_bundle_id": str(gate.get("baseline_benchmark_bundle_id") or ""),
        }

    async def _check_candidate_scoring_freshness(
        self,
        *,
        parent_artifact: PrivateModelArtifactManifest,
        candidate_window_hash: str,
        progress: Mapping[str, Any],
    ) -> None:
        active = await load_active_private_model(self.config, register_bootstrap=True)
        active_parent = active.artifact.model_artifact_hash
        if active_parent != parent_artifact.model_artifact_hash:
            raise StaleParentDuringScoring(
                active_artifact=active.artifact,
                candidate_parent=parent_artifact.model_artifact_hash,
                progress=progress,
            )
        try:
            current_window, _gate = await self._daily_candidate_scoring_window_and_gate(
                artifact=parent_artifact,
            )
        except Exception as exc:
            if isinstance(exc, CandidateBaselineNotReady):
                raise
            raise CandidateBaselineNotReady(
                "candidate_daily_baseline_unavailable_during_candidate_scoring: "
                f"candidate_window={compact_ref(candidate_window_hash)}"
            ) from exc
        current_window_hash = str(getattr(current_window, "window_hash", "") or "")
        if current_window_hash and current_window_hash != candidate_window_hash:
            raise CandidateBaselineWindowChanged(
                candidate_window_hash=candidate_window_hash,
                current_window_hash=current_window_hash,
                progress=progress,
            )

    async def _run_private_baseline_contained(self) -> dict[str, Any] | None:
        """Contain failures that occur after the baseline computation guard."""

        try:
            return await self._maybe_run_private_baseline()
        except Exception as exc:  # noqa: BLE001 - final publication must be terminal
            if self._active_baseline_context is None:
                raise
            return await self._contain_private_baseline_publication_failure(exc)
        finally:
            self._active_baseline_context = None

    async def _contain_private_baseline_publication_failure(self, exc: BaseException) -> dict[str, Any]:
        context = dict(self._active_baseline_context or {})
        telemetry_session = context.get("telemetry_session")
        if not isinstance(telemetry_session, ScoringTelemetrySession):
            telemetry_session = None
        telemetry_heartbeat_stop = context.get("telemetry_heartbeat_stop")
        if isinstance(telemetry_heartbeat_stop, asyncio.Event):
            telemetry_heartbeat_stop.set()
        telemetry_heartbeat_task = context.get("telemetry_heartbeat_task")
        if isinstance(telemetry_heartbeat_task, asyncio.Task):
            await telemetry_heartbeat_task
        today = str(context.get("benchmark_date") or "")
        window_hash = str(context.get("rolling_window_hash") or "")
        manifest_hash = str(context.get("private_model_manifest_hash") or "")
        benchmark_attempt = int(context.get("benchmark_attempt") or 0)
        scope_key = f"{today}:{window_hash}:{manifest_hash}"
        self._baseline_publication_failures_in_process.add(scope_key)
        event_doc = {
            "benchmark_date": today,
            "benchmark_attempt": benchmark_attempt,
            "selected_icp_count": int(context.get("selected_icp_count") or 0),
            "private_model_manifest_hash": manifest_hash,
            "failure_phase": "publication",
            "failure_stage": str(context.get("publication_stage") or "unknown"),
            "terminal_no_automatic_retry": True,
            "scoring_worker_source_hash": _scoring_worker_source_hash(),
            "publication_retry_token_hash": _baseline_publication_retry_token_hash(),
            "error_diagnostics": _event_error_diagnostics(exc),
            "baseline_health": (
                dict(context["baseline_health"])
                if isinstance(context.get("baseline_health"), Mapping)
                else None
            ),
            "elapsed_seconds": round(time.time() - float(context.get("started_at") or time.time()), 3),
        }
        try:
            await create_scoring_dispatch_event(
                dispatch_type="private_baseline_rebenchmark",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                rolling_window_hash=window_hash or None,
                benchmark_bundle_id=str(context.get("benchmark_bundle_id") or "") or None,
                scoring_id=(
                    telemetry_session.run.scoring_id
                    if telemetry_session is not None and telemetry_session.run is not None
                    else None
                ),
                scoring_run_id=(
                    telemetry_session.run.scoring_run_id
                    if telemetry_session is not None and telemetry_session.run is not None
                    else None
                ),
                event_doc=event_doc,
            )
        except Exception as dispatch_exc:  # noqa: BLE001 - preserve the in-process terminal latch
            logger.exception(
                "research_lab_baseline_publication_failed_dispatch_write_failed "
                "benchmark_date=%s attempt=%s error=%s",
                today,
                benchmark_attempt,
                _short_error(dispatch_exc),
            )
        if telemetry_session is not None:
            await telemetry_session.cancel_active(
                failure_category="baseline_publication_failed",
                error=exc,
            )
            await emit_run_event(
                telemetry_session.run,
                "failed",
                retryable=False,
                failure_category="baseline_publication_failed",
                error=exc,
                benchmark_bundle_id=str(context.get("benchmark_bundle_id") or ""),
                telemetry_degraded=telemetry_session.degraded,
                event_doc={"failure_stage": event_doc["failure_stage"]},
            )
        logger.exception(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE PUBLICATION FAILED TERMINALLY",
                (
                    ("Worker", self.worker_ref),
                    ("Benchmark date", today),
                    ("Rolling window", compact_ref(window_hash)),
                    ("Attempt", benchmark_attempt),
                    ("Publication stage", event_doc["failure_stage"]),
                    ("Action", "automatic reruns blocked until worker source or retry token changes"),
                    ("Error", _short_error(exc)),
                ),
            )
        )
        return {
            "status": "baseline_publication_failed_terminal",
            "benchmark_date": today,
            "rolling_window_hash": window_hash,
            "benchmark_attempt": benchmark_attempt,
            "failure_stage": event_doc["failure_stage"],
            "error": _short_error(exc),
        }

    async def _maybe_run_private_baseline(self) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        baseline_start_offset = int(
            getattr(
                self.config,
                "baseline_start_utc_offset_seconds",
                _baseline_min_utc_day_delay_seconds(),
            )
            or 0
        )
        baseline_start_offset = max(0, min(86399, baseline_start_offset))
        min_start_at = utc_day_start(now) + timedelta(seconds=baseline_start_offset)
        if now < min_start_at:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE WAITING FOR DAILY ICP ACTIVATION",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Scheduled start", min_start_at.isoformat()),
                        ("Start offset seconds", baseline_start_offset),
                        ("Action", "deferring baseline so the UTC day's ICP set can activate"),
                    ),
                )
            )
            return {
                "status": "waiting_for_daily_icp_activation",
                "benchmark_date": today,
                "scheduled_start_at": min_start_at.isoformat(),
                "earliest_start_at": min_start_at.isoformat(),
                "start_offset_seconds": baseline_start_offset,
            }
        start = time.time()
        evaluation_epoch = await self._resolve_evaluation_epoch()
        conditional_policy = self.config.conditional_validation_policy()
        expected_policy_hash = (
            str(conditional_policy.to_dict()["policy_hash"])
            if conditional_policy.enabled
            else ""
        )
        expected_icp_count = (
            conditional_policy.total_icps
            if conditional_policy.enabled
            else self.config.lab_champion_eval_days
            * self.config.lab_champion_icps_per_day
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE ALLOCATED",
                (
                    ("Worker", self.worker_ref),
                    ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                    ("Proxy ref", self.proxy_ref_hash),
                    ("Benchmark date", today),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Eval days", self.config.lab_champion_eval_days),
                    ("ICPs per day", self.config.lab_champion_icps_per_day),
                    ("Expected ICPs", expected_icp_count),
                    ("Conditional validation", conditional_policy.mode),
                ),
            )
        )
        repo_head_sync = await sync_active_model_to_repo_head(
            self.config,
            actor_ref=self.worker_ref,
            dry_run=False,
            wait_for_repo_head=False,
        )
        if not bool(repo_head_sync.get("ok")):
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE BLOCKED: REPO HEAD MANIFEST NOT READY",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Status", repo_head_sync.get("status")),
                        ("Repo main SHA", str(repo_head_sync.get("repo_main_sha") or "")[:12] or "-"),
                        (
                            "current.json SHA",
                            str(repo_head_sync.get("current_json_git_sha") or "")[:12] or "-",
                        ),
                        (
                            "Active model SHA",
                            str(repo_head_sync.get("active_model_git_sha") or "")[:12] or "-",
                        ),
                        ("Action", "deferring daily benchmark; stale active lineage will not be benchmarked"),
                    ),
                )
            )
            return {
                "status": str(repo_head_sync.get("status") or "repo_head_sync_failed"),
                "benchmark_date": today,
                "repo_head_sync": repo_head_sync,
            }
        expected_fresh_set_id = utc_set_id_for_datetime(now)
        try:
            window = await fetch_rolling_icp_window(
                days=self.config.lab_champion_eval_days,
                icps_per_day=self.config.lab_champion_icps_per_day,
                **_rolling_window_fetch_kwargs(self.config),
                allow_partial=self.config.scoring_worker_allow_partial_icp_window,
                required_fresh_set_id=expected_fresh_set_id,
                require_fresh_set_active_at=now,
            )
        except RollingIcpWindowUnavailable as exc:
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE BLOCKED: DAILY ICP WINDOW NOT READY",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Expected fresh set", expected_fresh_set_id),
                        ("Reason", str(exc)[:240]),
                        ("Action", "retrying later; not falling back to a prior UTC day's ICP window"),
                    ),
                )
            )
            return {
                "status": "daily_icp_window_not_ready",
                "benchmark_date": today,
                "expected_fresh_set_id": expected_fresh_set_id,
                "error": str(exc),
            }
        active = await load_active_private_model(self.config, register_bootstrap=True)
        artifact = active.artifact
        baseline_repo_main_sha = str(repo_head_sync.get("repo_main_sha") or "")
        existing = await select_many(
            "research_lab_private_model_benchmark_current",
            columns="*",
            filters=(
                ("benchmark_date", today),
                ("rolling_window_hash", window.window_hash),
                ("private_model_manifest_hash", artifact.manifest_hash),
            ),
            order_by=(("created_at", True),),
            limit=25,
        )
        valid_existing = [
            row
            for row in existing
            if _private_benchmark_row_is_valid(row)
            and _private_benchmark_matches_policy(
                row,
                expected_policy_hash=expected_policy_hash,
            )
        ]
        if valid_existing:
            if conditional_policy.enabled:
                try:
                    await _repair_baseline_category_results_from_row(
                        valid_existing[0],
                        expected_policy_hash=expected_policy_hash,
                    )
                except Exception as exc:
                    logger.warning(
                        "research_lab_conditional_baseline_tracking_repair_failed "
                        "benchmark_bundle_id=%s error=%s",
                        compact_ref(valid_existing[0].get("benchmark_bundle_id")),
                        _short_error(exc),
                    )
                    return {
                        "status": "conditional_baseline_tracking_repair_failed",
                        "benchmark_date": today,
                        "rolling_window_hash": window.window_hash,
                        "error": _short_error(exc),
                    }
            already_key = f"{today}:{window.window_hash}:{artifact.manifest_hash}"
            if self._baseline_already_logged_date != already_key:
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ALREADY BENCHMARKED",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Rolling window", compact_ref(window.window_hash)),
                            ("Private model", compact_ref(artifact.model_artifact_hash)),
                            ("Selected ICPs", len(window.item_refs)),
                            ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                        ),
                    )
                )
                self._baseline_already_logged_date = already_key
            return {
                "status": "already_benchmarked",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "private_model_manifest_hash": artifact.manifest_hash,
            }
        same_day_reference = await self._reusable_same_day_benchmark(
            today=today,
            manifest_hash=artifact.manifest_hash,
            expected_policy_hash=expected_policy_hash,
        )
        if same_day_reference is not None:
            if conditional_policy.enabled:
                try:
                    await _repair_baseline_category_results_from_row(
                        same_day_reference,
                        expected_policy_hash=expected_policy_hash,
                    )
                except Exception as exc:
                    logger.warning(
                        "research_lab_same_day_conditional_baseline_incompatible "
                        "benchmark_bundle_id=%s error=%s",
                        compact_ref(same_day_reference.get("benchmark_bundle_id")),
                        _short_error(exc),
                    )
                    return {
                        "status": "same_day_conditional_baseline_incompatible",
                        "benchmark_date": today,
                        "benchmark_bundle_id": str(
                            same_day_reference.get("benchmark_bundle_id") or ""
                        ),
                        "error": _short_error(exc),
                    }
            reuse_key = f"{today}:reuse:{artifact.manifest_hash}"
            if self._baseline_already_logged_date != reuse_key:
                logger.warning(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE SAME-DAY REFERENCE REUSED",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Existing bundle", compact_ref(same_day_reference.get("benchmark_bundle_id"))),
                            ("Existing window", compact_ref(same_day_reference.get("rolling_window_hash"))),
                            ("Requested window", compact_ref(window.window_hash)),
                            ("Private model", compact_ref(artifact.model_artifact_hash)),
                            (
                                "Action",
                                "NOT re-running: a same-day re-run silently replaces the day's promotion "
                                "reference (same-day deltas of 3+ points observed). Set "
                                "RESEARCH_LAB_BASELINE_ALLOW_SAMEDAY_REPLACE=true to replace deliberately.",
                            ),
                        ),
                    )
                )
                self._baseline_already_logged_date = reuse_key
            return {
                "status": "reused_same_day_benchmark",
                "benchmark_date": today,
                "benchmark_bundle_id": str(same_day_reference.get("benchmark_bundle_id") or ""),
                "rolling_window_hash": str(same_day_reference.get("rolling_window_hash") or ""),
                "private_model_manifest_hash": artifact.manifest_hash,
            }
        try:
            dispatch_history = await self._baseline_dispatch_history(
                today=today,
                window_hash=window.window_hash,
                manifest_hash=artifact.manifest_hash,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed before provider spend
            logger.warning(
                "research_lab_baseline_dispatch_history_unavailable "
                "benchmark_date=%s window=%s error=%s",
                today,
                compact_ref(window.window_hash),
                _short_error(exc),
            )
            return {
                "status": "baseline_dispatch_history_unavailable",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "error": _short_error(exc),
            }
        publication_scope_key = f"{today}:{window.window_hash}:{artifact.manifest_hash}"
        publication_retry_blocked, retry_authorization = _baseline_publication_retry_decision(
            dispatch_history,
            scope_key=publication_scope_key,
            in_process_failures=self._baseline_publication_failures_in_process,
        )
        if publication_retry_blocked:
            if self._baseline_publication_failure_logged_key != publication_scope_key:
                logger.error(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE PUBLICATION RETRY BLOCKED",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Rolling window", compact_ref(window.window_hash)),
                            ("Private model", compact_ref(artifact.model_artifact_hash)),
                            ("Action", "waiting for changed worker source or a new publication retry token"),
                        ),
                    )
                )
                self._baseline_publication_failure_logged_key = publication_scope_key
            return {
                "status": "baseline_publication_failed_terminal",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "private_model_manifest_hash": artifact.manifest_hash,
                "automatic_retry_blocked": True,
            }
        if retry_authorization:
            logger.warning(
                "research_lab_baseline_publication_retry_authorized "
                "benchmark_date=%s window=%s reason=%s",
                today,
                compact_ref(window.window_hash),
                retry_authorization,
            )
        if _env_flag("RESEARCH_LAB_BASELINE_ANY_WORKER") and await self._baseline_leased_by_other_worker(today):
            return {"status": "baseline_leased_elsewhere", "benchmark_date": today}
        benchmark_attempt = _next_benchmark_attempt([*existing, *dispatch_history])
        await create_rolling_icp_window(window)
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE STARTED",
                (
                    ("Worker", self.worker_ref),
                    ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                    ("Proxy ref", self.proxy_ref_hash),
                    ("Rolling window", compact_ref(window.window_hash)),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Selected sets", len(window.set_ids)),
                    ("Selected ICPs", len(window.item_refs)),
                    ("Private model", compact_ref(artifact.model_artifact_hash)),
                    ("Active model SHA", str(repo_head_sync.get("active_model_git_sha") or "")[:12] or "-"),
                    ("Repo main SHA", str(repo_head_sync.get("repo_main_sha") or "")[:12] or "-"),
                    ("current.json SHA", str(repo_head_sync.get("current_json_git_sha") or "")[:12] or "-"),
                    ("Active is repo head", repo_head_sync.get("active_is_repo_head")),
                    ("Concurrency", self.config.private_baseline_concurrency),
                    (
                        "Benchmark Exa key",
                        "dedicated"
                        if _provider_profile_has_override(
                            BENCHMARK_MODEL_PROFILE,
                            "exa",
                        )
                        else "inherited",
                    ),
                    ("Exa RPS per container", self.config.benchmark_exa_max_rps or "inherited"),
                ),
            )
        )
        runner = AttestedPrivateModelRunnerV2(
            artifact=artifact,
            spec=DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                env_passthrough=self._private_model_env_passthrough(),
                extra_env=self._with_provider_cost_evaluation_scope(
                    self._private_baseline_scoring_env(),
                    run_type="private_baseline_rebenchmark",
                    rolling_window_hash=window.window_hash,
                    artifact_hash=artifact.model_artifact_hash,
                    benchmark_date=today,
                    benchmark_attempt=benchmark_attempt,
                    evaluation_epoch=evaluation_epoch,
                    started_at=start,
                ),
            ),
            model_kind="private",
            worker_index=self.config.scoring_worker_index,
            epoch_id=evaluation_epoch,
            parent_graphs=await _attested_model_parent_graphs(
                model_kind="private",
                artifact=artifact,
                epoch_id=evaluation_epoch,
            ),
        )
        scorer = QualificationStyleCompanyScorer(
            attested_epoch_id=evaluation_epoch,
            attested_purpose="research_lab.rebenchmark.v1",
            attested_provider_profile="benchmark_scorer",
        )
        # Trace scope (§5.4 + in-container capture): keyed per day, attempt,
        # and window so a deliberate same-day replacement never overwrites the
        # prior attempt's docs.
        baseline_window_tag = str(window.window_hash or "").removeprefix("sha256:")[:16] or "window"
        baseline_trace_context = self._baseline_trace_context(
            context_ref=f"daily-{today}-a{benchmark_attempt}-{baseline_window_tag}",
            manifest_uri=artifact.manifest_uri,
            run_type="private_baseline_rebenchmark",
            benchmark_date=today,
            rolling_window_hash=window.window_hash,
        )
        baseline_telemetry_session: ScoringTelemetrySession | None = None
        if scoring_telemetry_enabled(self.config):
            telemetry_run = await allocate_scoring_run(
                identity_doc={
                    "run_type": "private_baseline_rebenchmark",
                    "benchmark_date": today,
                    "rolling_window_hash": window.window_hash,
                    "reference_artifact_hash": artifact.model_artifact_hash,
                    "reference_manifest_hash": artifact.manifest_hash,
                    "evaluation_epoch": evaluation_epoch,
                    "scoring_worker_source_hash": _scoring_worker_source_hash(),
                },
                run_type="private_baseline_rebenchmark",
                worker_ref=self.worker_ref,
                expected_icp_count=len(window.benchmark_items),
                scheduler_type=(
                    "fixed_wave" if self.config.private_baseline_concurrency > 1 else "serial"
                ),
                minimum_run_attempt=benchmark_attempt,
                benchmark_id=private_baseline_benchmark_id(
                    benchmark_date=today,
                    rolling_window_hash=window.window_hash,
                    reference_artifact_hash=artifact.model_artifact_hash,
                ),
                benchmark_date=today,
                rolling_window_hash=window.window_hash,
                reference_artifact_hash=artifact.model_artifact_hash,
                reference_manifest_hash=artifact.manifest_hash,
                evaluation_epoch=evaluation_epoch,
            )
            if telemetry_run is not None:
                baseline_telemetry_session = ScoringTelemetrySession(telemetry_run)
        per_icp_summaries: list[dict[str, Any]] = []
        nonempty_output_count = 0
        retried_total = 0
        recovered_total = 0
        baseline_progress_rows: list[dict[str, Any]] = []
        baseline_progress_location = (
            _baseline_progress_s3_location(
                artifact.manifest_uri,
                benchmark_date=today,
                window_hash=window.window_hash,
                private_model_artifact_hash=artifact.model_artifact_hash,
            )
            if _per_icp_checkpoint_enabled()
            else None
        )
        if baseline_progress_location is not None:
            await self._ensure_private_baseline_repo_head_unchanged(
                expected_git_sha=baseline_repo_main_sha,
                benchmark_date=today,
                item_index=1,
                total_icps=max(1, len(window.benchmark_items)),
            )
            progress_bucket, progress_key = baseline_progress_location
            baseline_progress_rows = await asyncio.to_thread(
                _load_baseline_scoring_progress,
                progress_bucket,
                progress_key,
                benchmark_date=today,
                window_hash=window.window_hash,
                private_model_artifact_hash=artifact.model_artifact_hash,
                repo_git_sha=baseline_repo_main_sha,
                manifest_hash=str(artifact.manifest_hash or ""),
            )
            if baseline_progress_rows:
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE PROGRESS RESUMED",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Rolling window", compact_ref(window.window_hash)),
                            ("Private model", compact_ref(artifact.model_artifact_hash)),
                            ("Completed ICPs", len(baseline_progress_rows)),
                            ("Action", "skipping already-checkpointed ICPs after restart"),
                        ),
                    )
                )

        if baseline_telemetry_session is not None:
            resumed_by_ref = _progress_rows_by_icp_ref(baseline_progress_rows)
            checkpoint_ref = (
                opaque_checkpoint_ref(*baseline_progress_location)
                if baseline_progress_location is not None
                else ""
            )
            for item_index, item in enumerate(window.benchmark_items):
                item_ref = str(item.get("icp_ref") or item.get("icp_hash") or "")
                resumed_row = resumed_by_ref.get(item_ref)
                await baseline_telemetry_session.plan(
                    icp_ref=item_ref,
                    icp_hash=str(item.get("icp_hash") or ""),
                    icp_ordinal=item_index,
                    model_role="reference",
                    execution_kind=(
                        "checkpoint_reuse" if resumed_row is not None else "model_invocation"
                    ),
                )
                if resumed_row is not None:
                    await baseline_telemetry_session.complete_result(
                        resumed_row,
                        model_role="reference",
                        checkpoint_ref=checkpoint_ref,
                        checkpoint_persisted=True,
                        outcome="checkpoint_reuse",
                    )

        baseline_progress_lock = asyncio.Lock()

        async def _checkpoint_completed_baseline_icp(row: Mapping[str, Any]) -> None:
            if not _baseline_summary_checkpointable(row):
                return
            if baseline_progress_location is None:
                if baseline_telemetry_session is not None:
                    await baseline_telemetry_session.complete_result(
                        row,
                        model_role="reference",
                        checkpoint_persisted=False,
                        outcome="scored_without_checkpoint",
                    )
                return
            async with baseline_progress_lock:
                progress_bucket, progress_key = baseline_progress_location
                public_row = _baseline_progress_public_row(row)
                by_ref = _progress_rows_by_icp_ref(baseline_progress_rows)
                ref = str(public_row.get("icp_ref") or public_row.get("icp_hash") or "")
                if not ref:
                    return
                by_ref[ref] = public_row
                baseline_progress_rows[:] = list(by_ref.values())
                telemetry_index = checkpoint_telemetry_index(
                    baseline_telemetry_session,
                    baseline_progress_rows,
                    model_role="reference",
                )
                checkpoint_hash = await asyncio.to_thread(
                    _store_baseline_scoring_progress,
                    progress_bucket,
                    progress_key,
                    benchmark_date=today,
                    window_hash=window.window_hash,
                    private_model_artifact_hash=artifact.model_artifact_hash,
                    rows=baseline_progress_rows,
                    telemetry_index=telemetry_index,
                    repo_git_sha=baseline_repo_main_sha,
                    manifest_hash=str(artifact.manifest_hash or ""),
                )
            if baseline_telemetry_session is not None:
                await baseline_telemetry_session.complete_result(
                    row,
                    model_role="reference",
                    checkpoint_ref=opaque_checkpoint_ref(*baseline_progress_location),
                    checkpoint_hash=checkpoint_hash,
                    checkpoint_persisted=True,
                    outcome="scored",
                )

        self._active_baseline_context = {
            "benchmark_date": today,
            "benchmark_attempt": benchmark_attempt,
            "rolling_window_hash": window.window_hash,
            "private_model_manifest_hash": artifact.manifest_hash,
            "selected_icp_count": len(window.item_refs),
            "started_at": start,
            "publication_stage": "computation",
            "telemetry_session": baseline_telemetry_session,
        }
        baseline_telemetry_heartbeat_stop: asyncio.Event | None = None
        baseline_telemetry_heartbeat_task: asyncio.Task[Any] | None = None
        retry_runner: AttestedPrivateModelRunnerV2 | None = None

        async def _stop_baseline_telemetry_heartbeat() -> None:
            if baseline_telemetry_heartbeat_stop is not None:
                baseline_telemetry_heartbeat_stop.set()
            if baseline_telemetry_heartbeat_task is not None:
                await baseline_telemetry_heartbeat_task

        try:
            await emit_run_event(
                baseline_telemetry_session.run if baseline_telemetry_session is not None else None,
                "assigned",
            )
            lease_event = await create_scoring_dispatch_event(
                dispatch_type="private_baseline_rebenchmark",
                dispatch_status="assigned",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                rolling_window_hash=window.window_hash,
                scoring_id=(
                    baseline_telemetry_session.run.scoring_id
                    if baseline_telemetry_session is not None and baseline_telemetry_session.run is not None
                    else None
                ),
                scoring_run_id=(
                    baseline_telemetry_session.run.scoring_run_id
                    if baseline_telemetry_session is not None and baseline_telemetry_session.run is not None
                    else None
                ),
                event_doc={
                    "benchmark_date": today,
                    "benchmark_attempt": benchmark_attempt,
                    "selected_icp_count": len(window.item_refs),
                    "private_model_manifest_hash": artifact.manifest_hash,
                    "scoring_worker_source_hash": _scoring_worker_source_hash(),
                    "publication_retry_token_hash": _baseline_publication_retry_token_hash(),
                },
            )
            # Post-write lease confirm (claim-guard pattern, like
            # _claim_next_candidate's write-then-verify): two workers can both
            # pass the pre-check before either's `assigned` event lands. The
            # earliest unexpired open lease wins; the loser backs off silently
            # (writing a failed/completed release here would clear the
            # winner's lease for third-party observers). Only meaningful when
            # any-worker baselines are enabled — worker 0 has no peers.
            if _env_flag("RESEARCH_LAB_BASELINE_ANY_WORKER") and not await self._confirm_baseline_lease(
                today=today,
                lease_event=lease_event,
            ):
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE LEASE LOST",
                        (
                            ("Worker", self.worker_ref),
                            ("Benchmark date", today),
                            ("Action", "another worker holds the earlier lease; backing off"),
                        ),
                    )
                )
                if baseline_telemetry_session is not None:
                    await baseline_telemetry_session.cancel_active(
                        failure_category="baseline_lease_lost"
                    )
                    await emit_run_event(
                        baseline_telemetry_session.run,
                        "cancelled",
                        failure_category="baseline_lease_lost",
                    )
                return {"status": "baseline_leased_elsewhere", "benchmark_date": today}
            await emit_run_event(
                baseline_telemetry_session.run if baseline_telemetry_session is not None else None,
                "started",
            )
            baseline_telemetry_heartbeat_stop = asyncio.Event()
            baseline_telemetry_heartbeat_task = (
                asyncio.create_task(
                    baseline_telemetry_session.heartbeat_loop(
                        baseline_telemetry_heartbeat_stop
                    )
                )
                if baseline_telemetry_session is not None
                else None
            )
            if self._active_baseline_context is not None:
                self._active_baseline_context.update(
                    {
                        "telemetry_heartbeat_stop": baseline_telemetry_heartbeat_stop,
                        "telemetry_heartbeat_task": baseline_telemetry_heartbeat_task,
                    }
                )
            total_icps = len(window.benchmark_items)
            parallel_mode = self.config.private_baseline_concurrency > 1
            if parallel_mode:
                retry_runner = AttestedPrivateModelRunnerV2(
                    artifact=artifact,
                    spec=DockerPrivateModelSpec(
                        image_digest=artifact.image_digest,
                        timeout_seconds=self.config.scoring_worker_model_timeout_seconds,
                        env_passthrough=self._private_model_env_passthrough(),
                        extra_env=self._with_provider_cost_evaluation_scope(
                            self._private_baseline_retry_scoring_env(),
                            run_type="private_baseline_rebenchmark",
                            rolling_window_hash=window.window_hash,
                            artifact_hash=artifact.model_artifact_hash,
                            benchmark_date=today,
                            benchmark_attempt=benchmark_attempt,
                            evaluation_epoch=evaluation_epoch,
                            started_at=start,
                        ),
                        # The first-pass runner already pulled this digest.
                        pull_before_run=False,
                    ),
                    model_kind="private",
                    worker_index=self.config.scoring_worker_index,
                    epoch_id=evaluation_epoch,
                    parent_graphs=runner.parent_graphs,
                )
                batch_summaries, retry_stats = await self._run_baseline_batch(
                    runner=runner,
                    retry_runner=retry_runner,
                    scorer=scorer,
                    window=window,
                    run_start=start,
                    trace_context=baseline_trace_context,
                    resume_results=baseline_progress_rows,
                    icp_checkpoint=_checkpoint_completed_baseline_icp,
                    expected_repo_main_git_sha=baseline_repo_main_sha,
                    benchmark_date=today,
                    telemetry_session=baseline_telemetry_session,
                    telemetry_model_role="reference",
                )
                retried_total = int(retry_stats.get("retried") or 0)
                recovered_total = int(retry_stats.get("recovered") or 0)
                for item_summary in batch_summaries:
                    if item_summary.pop("_nonempty", False):
                        nonempty_output_count += 1
                    item_summary.pop("_item_index", None)
                    item_summary.pop("_retryable", None)
                    item_summary.pop("_runtime_error", None)
                    item_summary.pop("_retry_backoff_seconds", None)
                    per_icp_summaries.append(item_summary)
            baseline_resume_by_ref = _progress_rows_by_icp_ref(baseline_progress_rows)
            for item_index, item in enumerate([] if parallel_mode else window.benchmark_items, start=1):
                item_start = time.time()
                label = str(item.get("icp_ref") or item.get("icp_hash") or "unknown_icp")
                resumed_summary = baseline_resume_by_ref.get(label)
                if resumed_summary is not None:
                    logger.info(
                        format_worker_block(
                            "RESEARCH LAB PRIVATE BASELINE ICP RESUMED FROM CHECKPOINT",
                            (
                                ("Worker", self.worker_ref),
                                ("ICP", f"{item_index}/{total_icps}"),
                                ("ICP ref", compact_ref(label)),
                                ("Rolling window", compact_ref(window.window_hash)),
                            ),
                        )
                    )
                    if _baseline_summary_nonempty(resumed_summary):
                        nonempty_output_count += 1
                    per_icp_summaries.append(dict(resumed_summary))
                    continue
                await self._ensure_private_baseline_repo_head_unchanged(
                    expected_git_sha=baseline_repo_main_sha,
                    benchmark_date=today,
                    item_index=item_index,
                    total_icps=total_icps,
                )
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ICP STARTED",
                        (
                            ("Worker", self.worker_ref),
                            ("ICP", f"{item_index}/{total_icps}"),
                            ("ICP ref", compact_ref(label)),
                            ("ICP hash", compact_ref(item.get("icp_hash"))),
                            ("Set", item.get("set_id")),
                            ("Day", item.get("day_index")),
                            ("Day rank", item.get("day_rank")),
                        ),
                    )
                )
                if baseline_telemetry_session is not None:
                    await baseline_telemetry_session.attempt_started(
                        icp_ref=label,
                        icp_hash=str(item.get("icp_hash") or ""),
                        icp_ordinal=max(0, item_index - 1),
                        model_role="reference",
                        retry_round=0,
                    )
                runtime_error = ""
                # In-container trace collection (§9.1): asyncio.to_thread copies
                # the contextvars context, so the runner thread sees the
                # collector installed here.
                serial_trace_entries: list[dict[str, Any]] | None = None
                serial_trace_token: contextvars.Token | None = None
                if incontainer_trace_capture_enabled():
                    serial_trace_entries, serial_trace_token = begin_incontainer_trace_collection()
                try:
                    outputs = ensure_private_model_outputs(
                        await asyncio.to_thread(runner, item["icp"], {"mode": "private_baseline"}),
                        context_label=f"private baseline for {label}",
                        require_non_empty=False,
                    )
                except PrivateModelRuntimeError as exc:
                    outputs = []
                    runtime_error = _short_error(exc)
                    logger.warning(
                        format_worker_block(
                            "RESEARCH LAB PRIVATE BASELINE ICP RUNTIME ERROR",
                            (
                                ("Worker", self.worker_ref),
                                ("ICP", f"{item_index}/{total_icps}"),
                                ("ICP ref", compact_ref(label)),
                                ("ICP hash", compact_ref(item.get("icp_hash"))),
                                ("Error", runtime_error),
                            ),
                        )
                    )
                finally:
                    if serial_trace_token is not None:
                        end_incontainer_trace_collection(serial_trace_token)
                item_elapsed = time.time() - item_start
                if baseline_telemetry_session is not None and not runtime_error:
                    await baseline_telemetry_session.lifecycle(
                        "sourcing_completed",
                        {
                            "icp_ref": label,
                            "model_role": "reference",
                            "retry_round": 0,
                            "sourced_company_count": len(outputs),
                        },
                    )
                if outputs:
                    nonempty_output_count += 1
                score_breakdowns: list[dict[str, Any]] = []
                scorer_error = ""
                if outputs:
                    if baseline_telemetry_session is not None:
                        await baseline_telemetry_session.lifecycle(
                            "scoring_started",
                            {
                                "icp_ref": label,
                                "model_role": "reference",
                                "retry_round": 0,
                            },
                        )
                    try:
                        score_breakdowns = await self._score_baseline_outputs(
                            scorer=scorer,
                            outputs=outputs,
                            icp=item["icp"],
                            apply_isolation=True,
                        )
                    except Exception as scorer_exc:  # noqa: BLE001 - §0-N6: non-fatal per ICP
                        scorer_error = _short_error(scorer_exc)
                        logger.warning(
                            format_worker_block(
                                "RESEARCH LAB PRIVATE BASELINE ICP SCORER ERROR",
                                (
                                    ("Worker", self.worker_ref),
                                    ("ICP", f"{item_index}/{total_icps}"),
                                    ("ICP ref", compact_ref(label)),
                                    ("ICP hash", compact_ref(item.get("icp_hash"))),
                                    ("Error", scorer_error),
                                ),
                            )
                        )
                scores = [float(row.get("final_score", 0.0) or 0.0) for row in score_breakdowns]
                # Shared flag-aware per-ICP score: with the capped flag on this
                # is sum(top-N)/N for the ICP's requested company count, so the
                # baseline pays for unfilled slots exactly like candidates do.
                icp_score = benchmark_icp_score_from_company_scores(
                    scores,
                    requested_count=_icp_company_goal(item["icp"]),
                    fp_penalty_total=fp_penalty_total_from_breakdowns(
                        score_breakdowns, item["icp"]
                    ),
                )
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ICP SCORED",
                        (
                            ("Worker", self.worker_ref),
                            ("ICP", f"{item_index}/{total_icps}"),
                            ("ICP ref", compact_ref(label)),
                            ("ICP hash", compact_ref(item.get("icp_hash"))),
                            ("Set", item.get("set_id")),
                            ("Day", item.get("day_index")),
                            ("Day rank", item.get("day_rank")),
                            ("Score", f"{icp_score:.4f}"),
                            ("Companies", len(scores)),
                            ("Non-empty output", bool(outputs)),
                            ("Runtime error", runtime_error or "-"),
                            ("Scorer error", scorer_error or "-"),
                            ("ICP runtime", f"{item_elapsed:.1f}s"),
                            ("Elapsed", f"{time.time() - start:.1f}s"),
                        ),
                    )
                )
                item_summary = sanitize_benchmark_item_summary(
                    item=item,
                    score=icp_score,
                    company_count=len(scores),
                    score_breakdowns=score_breakdowns,
                    # Model output count BEFORE the scorer's employee-bucket
                    # pre-filter, so the funnel's first stage is the true
                    # "companies discovered" number.
                    sourced_count=len(outputs),
                )
                if runtime_error or scorer_error:
                    diagnostics = dict(item_summary.get("diagnostics") or {})
                    runtime_diagnostics = _runtime_error_diagnostics(runtime_error or scorer_error)
                    categories = set(diagnostics.get("failure_categories") or [])
                    if runtime_error:
                        categories.add("runtime_provider_error")
                    if scorer_error:
                        categories.add("scorer_provider_error")
                    categories.add(str(runtime_diagnostics["category"]))
                    diagnostics["failure_categories"] = sorted(categories)
                    diagnostics["runtime_error"] = runtime_diagnostics
                    item_summary["diagnostics"] = diagnostics
                await self._record_baseline_icp_traces(
                    item=item,
                    item_summary=item_summary,
                    outputs=outputs,
                    score_breakdowns=score_breakdowns,
                    trace_entries=serial_trace_entries,
                    trace_context=baseline_trace_context,
                    telemetry_session=baseline_telemetry_session,
                    telemetry_model_role="reference",
                )
                _apply_provider_cost_baseline_outcome(item_summary)
                if (
                    item_index >= PRIVATE_BASELINE_FAST_EMPTY_ABORT_AFTER
                    and nonempty_output_count <= 0
                    and time.time() - start < PRIVATE_BASELINE_FAST_EMPTY_ABORT_SECONDS
                ):
                    raise PrivateModelRuntimeError(
                        "private baseline fast-empty guard tripped: "
                        f"first {item_index} ICPs returned zero companies in {time.time() - start:.1f}s. "
                        "The private model is not executing the full provider-backed sourcing path; "
                        "check Docker env passthrough, provider keys, proxy connectivity, and ICP canonicalization."
                    )
                if baseline_telemetry_session is not None and (runtime_error or scorer_error):
                    await baseline_telemetry_session.lifecycle(
                        "attempt_failed",
                        {
                            "icp_ref": label,
                            "model_role": "reference",
                            "retry_round": 0,
                            "retryable": False,
                            "failure_category": (
                                "runtime_provider_error"
                                if runtime_error
                                else "scorer_provider_error"
                            ),
                            "error": runtime_error or scorer_error,
                        },
                    )
                await _checkpoint_completed_baseline_icp(item_summary)
                per_icp_summaries.append(item_summary)
            if nonempty_output_count <= 0:
                raise PrivateModelRuntimeError(
                    f"private baseline returned zero companies across all {len(window.benchmark_items)} ICPs"
                )
            baseline_health = _build_baseline_health(
                per_icp_summaries=per_icp_summaries,
                retried=retried_total,
                recovered=recovered_total,
                max_unresolved_icps=_baseline_max_unresolved_icps(),
            )
            aggregate_score = _average([summary["score"] for summary in per_icp_summaries])
            day_jump_points = await self._baseline_day_jump_points(
                today=today,
                aggregate_score=aggregate_score,
            )
            if day_jump_points is not None:
                baseline_health["day_jump_points"] = round(day_jump_points, 4)
            if self._active_baseline_context is not None:
                self._active_baseline_context["baseline_health"] = dict(baseline_health)
            _enforce_baseline_publication_gates(
                baseline_health=baseline_health,
                aggregate_score=aggregate_score,
                day_jump_points=day_jump_points,
                health_gate_enforced=bool(self.config.baseline_health_gate_enforced),
                max_day_jump=_baseline_max_day_jump_points(),
            )
        except Exception as exc:
            await _stop_baseline_telemetry_heartbeat()
            await create_scoring_dispatch_event(
                dispatch_type="private_baseline_rebenchmark",
                dispatch_status="failed",
                worker_ref=self.worker_ref,
                proxy_ref_hash=self.proxy_ref_hash,
                rolling_window_hash=window.window_hash,
                scoring_id=(
                    baseline_telemetry_session.run.scoring_id
                    if baseline_telemetry_session is not None and baseline_telemetry_session.run is not None
                    else None
                ),
                scoring_run_id=(
                    baseline_telemetry_session.run.scoring_run_id
                    if baseline_telemetry_session is not None and baseline_telemetry_session.run is not None
                    else None
                ),
                event_doc={
                    "benchmark_date": today,
                    "benchmark_attempt": benchmark_attempt,
                    "selected_icp_count": len(window.item_refs),
                    "private_model_manifest_hash": artifact.manifest_hash,
                    "failure_phase": "computation",
                    "terminal_no_automatic_retry": False,
                    "scoring_worker_source_hash": _scoring_worker_source_hash(),
                    "publication_retry_token_hash": _baseline_publication_retry_token_hash(),
                    "error_diagnostics": _event_error_diagnostics(exc),
                    "baseline_health": (
                        dict(exc.baseline_health)
                        if isinstance(exc, BaselineHealthGateFailure)
                        else None
                    ),
                    "elapsed_seconds": round(time.time() - start, 3),
                },
            )
            if baseline_telemetry_session is not None:
                await baseline_telemetry_session.cancel_active(
                    failure_category="baseline_computation_failed",
                    error=exc,
                )
                await emit_run_event(
                    baseline_telemetry_session.run,
                    "failed",
                    retryable=True,
                    failure_category="baseline_computation_failed",
                    error=exc,
                    telemetry_degraded=baseline_telemetry_session.degraded,
                )
            logger.exception(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE FAILED",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Rolling window", compact_ref(window.window_hash)),
                        ("Evaluation epoch", evaluation_epoch),
                        ("Attempt", benchmark_attempt),
                        ("Error", str(exc)[:300]),
                    ),
                )
            )
            return {
                "status": "failed",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "error": str(exc)[:300],
            }
        await _stop_baseline_telemetry_heartbeat()
        if self._active_baseline_context is not None:
            self._active_baseline_context["publication_stage"] = "pre_record_conflict_check"
        pre_record_conflict = await self._same_day_reference_recorded_while_running(
            today=today,
            window_hash=window.window_hash,
            manifest_hash=artifact.manifest_hash,
            expected_policy_hash=expected_policy_hash,
        )
        if pre_record_conflict is not None:
            if conditional_policy.enabled:
                try:
                    await _repair_baseline_category_results_from_row(
                        pre_record_conflict,
                        expected_policy_hash=expected_policy_hash,
                    )
                except Exception as exc:
                    logger.warning(
                        "research_lab_concurrent_conditional_baseline_incompatible "
                        "benchmark_bundle_id=%s error=%s",
                        compact_ref(pre_record_conflict.get("benchmark_bundle_id")),
                        _short_error(exc),
                    )
                    return {
                        "status": "concurrent_conditional_baseline_incompatible",
                        "benchmark_date": today,
                        "benchmark_bundle_id": str(
                            pre_record_conflict.get("benchmark_bundle_id") or ""
                        ),
                        "error": _short_error(exc),
                    }
            if baseline_telemetry_session is not None:
                await baseline_telemetry_session.cancel_active(
                    failure_category="baseline_reference_conflict"
                )
                await emit_run_event(
                    baseline_telemetry_session.run,
                    "cancelled",
                    failure_category="baseline_reference_conflict",
                )
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE COMPLETED BUT REFERENCE ALREADY RECORDED",
                    (
                        ("Worker", self.worker_ref),
                        ("Benchmark date", today),
                        ("Existing bundle", compact_ref(pre_record_conflict.get("benchmark_bundle_id"))),
                        ("Action", "discarding this run's recording to keep the day's reference stable"),
                    ),
                )
            )
            return {
                "status": "already_benchmarked",
                "benchmark_date": today,
                "rolling_window_hash": window.window_hash,
                "private_model_manifest_hash": artifact.manifest_hash,
            }
        if self._active_baseline_context is not None:
            self._active_baseline_context["publication_stage"] = "score_summary_build"
        baseline_summary_payload = {
            "artifact_manifest": artifact.to_dict(),
            "benchmark_date": today,
            "benchmark_attempt": benchmark_attempt,
            "rolling_window_hash": window.window_hash,
            "evaluation_epoch": evaluation_epoch,
            "benchmark_items": list(window.benchmark_items),
            "per_icp_summaries": list(per_icp_summaries),
            "public_icps_per_day": self.config.public_benchmark_public_icps_per_day,
            "public_weak_per_day": self.config.public_benchmark_public_weak_per_day,
            "public_total_icps": self.config.public_benchmark_public_total_icps,
            "public_weak_total": self.config.public_benchmark_public_weak_total,
            "retried": retried_total,
            "recovered": recovered_total,
            "max_unresolved_icps": _baseline_max_unresolved_icps(),
            "day_jump_points": day_jump_points,
            # The old path captured elapsed time after constructing the other
            # summary fields. Build once before taking the timestamp to retain
            # that observable ordering, then bind the exact value below.
            "elapsed_seconds": 0.0,
        }
        if conditional_policy.enabled:
            baseline_summary_payload["conditional_validation_policy"] = (
                conditional_policy.to_dict()
            )
        baseline_summary_result = build_baseline_score_summary(**baseline_summary_payload)
        baseline_summary_payload["elapsed_seconds"] = round(time.time() - start, 3)
        score_summary_doc = {
            **baseline_summary_result["score_summary_doc"],
            "elapsed_seconds": baseline_summary_payload["elapsed_seconds"],
        }
        baseline_summary_result = {
            **baseline_summary_result,
            "score_summary_doc": score_summary_doc,
        }
        if baseline_summary_result["aggregate_score"] != aggregate_score:
            raise RuntimeError("shared baseline aggregate diverged from existing calculation")
        if baseline_summary_result["baseline_health"] != baseline_health:
            raise RuntimeError("shared baseline health diverged from existing calculation")
        aggregate_score = baseline_summary_result["aggregate_score"]
        baseline_health = baseline_summary_result["baseline_health"]
        serving_model_version = baseline_summary_result["serving_model_version"]
        per_icp_summaries = baseline_summary_result["per_icp_summaries"]
        visibility_split = baseline_summary_result["visibility_split"]
        category_assignment = baseline_summary_result.get("category_assignment")
        noise_budget = baseline_summary_result["daily_noise_budget"]
        attested_baseline_outcome = await compare_attested_baseline_score_summary(
            epoch_id=evaluation_epoch,
            build_payload=baseline_summary_payload,
            expected_result=baseline_summary_result,
            parent_receipts=_attested_receipts_from(
                runner,
                retry_runner,
                scorer,
            ),
        )
        bundle_hash = canonical_hash(score_summary_doc)
        if self._active_baseline_context is not None:
            self._active_baseline_context["publication_stage"] = "kms_signature"
        signature_ref = await asyncio.to_thread(
            sign_digest_with_kms,
            key_id=self.config.score_bundle_kms_key_id,
            digest_hash=bundle_hash,
            signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
        )
        if self._active_baseline_context is not None:
            self._active_baseline_context["publication_stage"] = "private_bundle_insert"
        bundle, _event = await create_private_model_benchmark_bundle(
            benchmark_date=today,
            private_model_artifact_hash=artifact.model_artifact_hash,
            private_model_manifest_hash=artifact.manifest_hash,
            rolling_window_hash=window.window_hash,
            evaluation_epoch=evaluation_epoch,
            benchmark_attempt=benchmark_attempt,
            benchmark_quality="passed",
            aggregate_score=aggregate_score,
            scoring_worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            signature_ref=signature_ref,
            score_summary_doc=score_summary_doc,
        )
        if isinstance(category_assignment, Mapping):
            await _persist_baseline_category_results(
                category_assignment,
                source_bundle_ref=str(bundle["benchmark_bundle_id"]),
                rolling_window_hash=window.window_hash,
                scoring_run_id=(
                    str(baseline_telemetry_session.run.scoring_run_id)
                    if baseline_telemetry_session is not None
                    and baseline_telemetry_session.run is not None
                    else ""
                ),
            )
        await persist_attested_outcome_artifact_links(
            attested_baseline_outcome,
            artifact_links=[
                {
                    "artifact_kind": "benchmark_score_summary",
                    "artifact_ref": str(bundle["benchmark_bundle_id"]),
                    "artifact_hash": bundle_hash,
                },
                {
                    "artifact_kind": "benchmark_bundle",
                    "artifact_ref": str(bundle["benchmark_bundle_id"]),
                    "artifact_hash": str(bundle["benchmark_bundle_hash"]),
                },
            ],
        )
        if self._active_baseline_context is not None:
            self._active_baseline_context["benchmark_bundle_id"] = str(bundle["benchmark_bundle_id"])
            self._active_baseline_context["publication_stage"] = "completed_dispatch_insert"
        await create_scoring_dispatch_event(
            dispatch_type="private_baseline_rebenchmark",
            dispatch_status="completed",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            rolling_window_hash=window.window_hash,
            benchmark_bundle_id=str(bundle["benchmark_bundle_id"]),
            scoring_id=(
                baseline_telemetry_session.run.scoring_id
                if baseline_telemetry_session is not None and baseline_telemetry_session.run is not None
                else None
            ),
            scoring_run_id=(
                baseline_telemetry_session.run.scoring_run_id
                if baseline_telemetry_session is not None and baseline_telemetry_session.run is not None
                else None
            ),
            event_doc={
                "benchmark_date": today,
                "benchmark_attempt": benchmark_attempt,
                "elapsed_seconds": round(time.time() - start, 3),
                "selected_icp_count": len(window.item_refs),
                "public_icp_count": int(visibility_split.get("public_count") or 0),
                "private_holdout_icp_count": int(visibility_split.get("private_count") or 0),
                "conditional_holdout_icp_count": int(
                    visibility_split.get("conditional_count") or 0
                ),
                "private_model_manifest_hash": artifact.manifest_hash,
                "scoring_worker_source_hash": _scoring_worker_source_hash(),
                "publication_retry_token_hash": _baseline_publication_retry_token_hash(),
            },
        )
        if self._active_baseline_context is not None:
            self._active_baseline_context["publication_stage"] = "public_report_build"
        public_report_doc = build_public_benchmark_report(
            benchmark_date=today,
            rolling_window_hash=window.window_hash,
            aggregate_score=aggregate_score,
            per_icp_summaries=per_icp_summaries,
            benchmark_items=window.benchmark_items,
            public_icps_per_day=self.config.public_benchmark_public_icps_per_day,
            public_weak_per_day=self.config.public_benchmark_public_weak_per_day,
            public_total_icps=self.config.public_benchmark_public_total_icps,
            public_weak_total=self.config.public_benchmark_public_weak_total,
            category_assignment=category_assignment,
        )
        public_report_doc = {
            **public_report_doc,
            "serving_model_version": _public_serving_model_version_doc(serving_model_version),
            "daily_noise_budget": noise_budget,
        }
        public_report_doc["report_public_hash"] = sha256_json(
            {key: value for key, value in public_report_doc.items() if key != "report_public_hash"}
        )
        if self._active_baseline_context is not None:
            self._active_baseline_context["publication_stage"] = "public_report_insert"
        public_report, _report_event = await create_public_benchmark_report(
            benchmark_date=today,
            benchmark_bundle_id=str(bundle["benchmark_bundle_id"]),
            private_model_artifact_hash=artifact.model_artifact_hash,
            private_model_manifest_hash=artifact.manifest_hash,
            rolling_window_hash=window.window_hash,
            aggregate_score=aggregate_score,
            benchmark_attempt=benchmark_attempt,
            benchmark_quality="passed",
            report_doc=public_report_doc,
        )
        if self._active_baseline_context is not None:
            self._active_baseline_context["publication_stage"] = "audit_bundle_write"
        await self._write_audit_bundle(evaluation_epoch)
        await emit_run_event(
            baseline_telemetry_session.run if baseline_telemetry_session is not None else None,
            "completed",
            benchmark_bundle_id=str(bundle["benchmark_bundle_id"]),
            telemetry_degraded=(
                baseline_telemetry_session.degraded
                if baseline_telemetry_session is not None
                else False
            ),
            event_doc={
                "public_report_id": str(public_report["report_id"]),
                "selected_icp_count": len(window.item_refs),
            },
        )
        self._baseline_publication_failures_in_process.discard(publication_scope_key)
        self._baseline_publication_failure_logged_key = None
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE COMPLETED",
                (
                    ("Worker", self.worker_ref),
                    ("Worker index", f"{self.config.scoring_worker_index + 1}/{self.config.scoring_worker_total_workers}"),
                    ("Benchmark bundle", compact_ref(bundle["benchmark_bundle_id"])),
                    ("Public report", compact_ref(public_report["report_id"])),
                    ("Rolling window", compact_ref(window.window_hash)),
                    ("Evaluation epoch", evaluation_epoch),
                    ("Selected ICPs", len(window.item_refs)),
                    ("Attempt", benchmark_attempt),
                    ("Public ICPs", visibility_split.get("public_count")),
                    ("Private holdout ICPs", visibility_split.get("private_count")),
                    ("Public strength", visibility_split.get("public_strength_counts")),
                    ("Private strength", visibility_split.get("private_strength_counts")),
                    ("Aggregate score", f"{aggregate_score:.4f}"),
                    ("Unresolved provider errors", baseline_health.get("unresolved_provider_errors")),
                    ("Baseline health", "healthy" if baseline_health.get("gate_passed") else "degraded"),
                    ("Health decision", baseline_health.get("decision") or "observe_only"),
                    ("Day jump", f"{day_jump_points:+.4f}" if day_jump_points is not None else "-"),
                    ("Elapsed", f"{time.time() - start:.1f}s"),
                ),
            )
        )
        return {
            "status": "completed",
            "benchmark_date": today,
            "benchmark_bundle_id": str(bundle["benchmark_bundle_id"]),
            "public_report_id": str(public_report["report_id"]),
            "rolling_window_hash": window.window_hash,
        }

    def _get_scorer_trace_recorder(self) -> _ScorerTraceRecorder:
        recorder = getattr(self, "_scorer_trace_recorder", None)
        if recorder is None:
            recorder = _ScorerTraceRecorder(self.config)
            self._scorer_trace_recorder = recorder
        return recorder

    def _baseline_trace_context(
        self,
        *,
        context_ref: str,
        manifest_uri: str,
        run_type: str = "private_baseline_rebenchmark",
        benchmark_date: str = "",
        rolling_window_hash: str = "",
    ) -> dict[str, Any]:
        """Shared trace scope for one baseline/champion batch: keys the §5.4
        scorer-trace docs and the in-container trace docs, and carries the
        batch's log-once drop state for unconfigured S3."""
        return {
            "context_ref": str(context_ref),
            "manifest_uri": str(manifest_uri or ""),
            "run_type": str(run_type or "private_baseline_rebenchmark"),
            "benchmark_date": str(benchmark_date or ""),
            "rolling_window_hash": str(rolling_window_hash or ""),
            "incontainer_drop_state": {"logged": False, "dropped_entries": 0},
        }

    async def _ensure_private_baseline_repo_head_unchanged(
        self,
        *,
        expected_git_sha: str,
        benchmark_date: str,
        item_index: int,
        total_icps: int,
    ) -> None:
        expected = str(expected_git_sha or "").strip()
        if not expected:
            return
        try:
            status = await private_repo_head_alignment_status(self.config)
        except Exception as exc:  # noqa: BLE001 - do not kill a healthy run on transient git status failure
            logger.warning(
                "research_lab_private_baseline_repo_head_check_unavailable worker_ref=%s "
                "benchmark_date=%s icp=%s/%s error=%s",
                self.worker_ref,
                benchmark_date,
                item_index,
                total_icps,
                _short_error(exc),
            )
            return
        repo_main_sha = str(status.get("repo_main_sha") or "").strip()
        if not repo_main_sha or repo_main_sha.lower() == expected.lower():
            return
        try:
            await sync_active_model_to_repo_head(
                self.config,
                actor_ref=self.worker_ref,
                dry_run=False,
                wait_for_repo_head=False,
            )
        except Exception:  # noqa: BLE001 - raising the repo-change exception is the control path
            logger.warning(
                "research_lab_private_baseline_repo_head_sync_after_change_failed worker_ref=%s "
                "benchmark_date=%s old_repo_main=%s new_repo_main=%s",
                self.worker_ref,
                benchmark_date,
                compact_ref(expected),
                compact_ref(repo_main_sha),
                exc_info=True,
            )
        logger.warning(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE RESTART REQUIRED: REPO HEAD CHANGED",
                (
                    ("Worker", self.worker_ref),
                    ("Benchmark date", benchmark_date),
                    ("ICP boundary", f"{item_index}/{total_icps}"),
                    ("Started repo main", compact_ref(expected)),
                    ("Current repo main", compact_ref(repo_main_sha)),
                    ("Action", "failing this attempt so the next pass benchmarks the new artifact"),
                ),
            )
        )
        raise PrivateBaselineRepoHeadChanged(
            expected_git_sha=expected,
            repo_main_sha=repo_main_sha,
            item_index=item_index,
            total_icps=total_icps,
        )

    async def _record_baseline_icp_traces(
        self,
        *,
        item: Mapping[str, Any],
        item_summary: dict[str, Any],
        outputs: list[Any],
        score_breakdowns: list[dict[str, Any]],
        trace_entries: list[dict[str, Any]] | None,
        trace_context: Mapping[str, Any] | None,
        telemetry_session: ScoringTelemetrySession | None = None,
        telemetry_model_role: str = "reference",
    ) -> None:
        """Best-effort §5.4 scorer-trace capture plus in-container trace
        publication for one baseline/champion ICP.

        Mutates ONLY ``item_summary['diagnostics']`` with pointer fields
        (``scorer_trace_ref``/``scorer_trace_sha256`` and the
        ``incontainer_trace_*`` trio) — never content — and never raises.
        With capture flags off (or ``trace_context`` unset) the summary stays
        byte-identical to the pre-capture shape.
        """
        if trace_context is None:
            return
        try:
            label = str(item.get("icp_ref") or item.get("icp_hash") or "unknown_icp")
            context_ref = str(trace_context.get("context_ref") or "baseline")
            diagnostics_updates: dict[str, Any] = {}
            if score_breakdowns:
                pointer = self._get_scorer_trace_recorder().capture(
                    context_ref=context_ref,
                    icp_ref=label,
                    icp_hash=str(item.get("icp_hash") or ""),
                    outputs=outputs,
                    breakdowns=score_breakdowns,
                    is_reference_model=True,
                    manifest_uri=str(trace_context.get("manifest_uri") or ""),
                )
                if pointer:
                    diagnostics_updates["scorer_trace_ref"] = pointer["s3_ref"]
                    diagnostics_updates["scorer_trace_sha256"] = pointer["sha256"]
                await _persist_company_label_examples(
                    context_ref=context_ref,
                    icp_ref=label,
                    icp_hash=str(item.get("icp_hash") or ""),
                    is_reference_model=True,
                    outputs=outputs,
                    breakdowns=score_breakdowns,
                    scorer_trace_pointer=pointer,
                    model_manifest_hash=str(trace_context.get("private_model_manifest_hash") or ""),
                )
                # Persist model-sourced companies the harness rejected, for later
                # false-rejection analysis (best-effort; never blocks scoring).
                await _persist_rejected_companies(
                    context_ref=context_ref,
                    icp_ref=label,
                    icp_hash=str(item.get("icp_hash") or ""),
                    is_reference_model=True,
                    outputs=outputs,
                    breakdowns=score_breakdowns,
                )
            if trace_entries:
                drop_state = trace_context.get("incontainer_drop_state")
                execution = (
                    telemetry_session.execution_for(
                        icp_ref=label,
                        model_role=telemetry_model_role,
                    )
                    if telemetry_session is not None
                    else None
                )
                persisted = await _persist_provider_cost_events(
                    entries=trace_entries,
                    run_type=str(trace_context.get("run_type") or "private_baseline_rebenchmark"),
                    icp_ref=label,
                    icp_hash=str(item.get("icp_hash") or ""),
                    runner_role="baseline",
                    benchmark_date=str(trace_context.get("benchmark_date") or ""),
                    rolling_window_hash=str(trace_context.get("rolling_window_hash") or ""),
                    scoring_id=str(execution.scoring_id if execution is not None else ""),
                    scoring_run_id=str(execution.scoring_run_id if execution is not None else ""),
                    icp_execution_id=str(
                        execution.icp_execution_id if execution is not None else ""
                    ),
                )
                if not persisted and telemetry_session is not None:
                    telemetry_session.degraded = True
                diagnostics_updates.update(
                    await _publish_baseline_incontainer_trace(
                        context_ref=context_ref,
                        icp_ref=label,
                        entries=trace_entries,
                        drop_state=drop_state if isinstance(drop_state, dict) else {},
                    )
                )
            if diagnostics_updates:
                diagnostics = dict(item_summary.get("diagnostics") or {})
                diagnostics.update(diagnostics_updates)
                item_summary["diagnostics"] = diagnostics
        except Exception:  # noqa: BLE001 - capture can never affect the batch
            logger.warning(
                "research_lab_baseline_trace_record_failed icp_ref=%s",
                compact_ref(str(item.get("icp_ref") or "")),
                exc_info=True,
            )

    def _candidate_incontainer_trace_sink(
        self,
        candidate_id: str,
        *,
        persist_costs: bool = True,
    ) -> Any:
        """Candidate-scoped in-container trace/cost sink for the evaluator.

        Provider-cost events must be persisted even when full in-container
        trace upload is disabled. When an S3 prefix is configured, uploads key
        by candidate (``{prefix}/{candidate_id}/{icp_ref}.json``) instead of
        the default run-scoped ref, so one candidate's traces stay enumerable
        under a single prefix; the evaluator places the returned uri into the
        row's ``incontainer_trace_ref``. When no encrypted S3 sink is
        available, this still persists cost events and returns an empty ref so
        the evaluator records dropped trace counts plus cost summaries. The
        ``RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE`` kill switch still overrides
        injected sinks inside the evaluator.
        """
        prefix = str(os.getenv(INCONTAINER_TRACE_S3_PREFIX_ENV) or "").strip().rstrip("/")
        kms_key_id = str(os.getenv(INCONTAINER_TRACE_KMS_KEY_ENV) or "").strip()
        upload_enabled = prefix.startswith("s3://") and bool(kms_key_id)
        if prefix.startswith("s3://") and not kms_key_id:
            # P5/P13: prefix-on/key-off must never write unencrypted; fall back
            # to dropped trace refs while still writing provider-cost events.
            logger.error(
                "research_lab_incontainer_trace_sink_refused candidate=%s reason=missing_kms_key "
                "(set %s so candidate in-container traces are written SSE-KMS encrypted)",
                compact_ref(candidate_id),
                INCONTAINER_TRACE_KMS_KEY_ENV,
            )
        safe_candidate = _trace_path_segment(candidate_id, fallback="candidate")

        async def _sink(icp_ref: str, entries: list[dict[str, Any]]) -> str:
            if persist_costs:
                await _persist_provider_cost_events(
                    entries=entries,
                    run_type="candidate_scoring",
                    icp_ref=str(icp_ref),
                    runner_role="candidate",
                    candidate_id=candidate_id,
                )
            if not upload_enabled:
                return ""
            return await asyncio.to_thread(
                _upload_incontainer_trace_doc,
                prefix,
                safe_candidate,
                str(icp_ref),
                list(entries),
                kms_key_id,
            )

        return _sink

    async def _run_baseline_icp(
        self,
        *,
        runner: AttestedPrivateModelRunnerV2,
        scorer: QualificationStyleCompanyScorer,
        item: Mapping[str, Any],
        item_index: int,
        total_icps: int,
        run_start: float,
        executor: concurrent.futures.Executor,
        scorer_semaphore: asyncio.Semaphore | None = None,
        mode_label: str = "private_baseline",
        trace_context: Mapping[str, Any] | None = None,
        expected_repo_main_git_sha: str = "",
        benchmark_date: str = "",
        telemetry_session: ScoringTelemetrySession | None = None,
        telemetry_model_role: str = "reference",
        retry_round: int = 0,
    ) -> dict[str, Any]:
        """Run one benchmark ICP (docker sourcing + scoring) and build its summary.

        Mirrors the serial baseline loop body, minus the fast-empty guard (its
        sequential-completion assumption does not hold under concurrency). The
        blocking docker wait runs on the dedicated baseline executor instead of
        asyncio.to_thread so many concurrent ~10-minute subprocess waits cannot
        starve the loop's default executor (which the KMS signing call uses).

        A scorer-side provider failure is NOT fatal for the batch (§0-N6): the
        scorer call is retried once inline, then the ICP is marked unresolved
        (retryable per the runner-error classifier) so retry rounds and
        observe-only baseline health diagnostics see it instead of the whole
        batch dying.

        The returned summary carries underscore-prefixed orchestration fields
        (_item_index, _retryable, _nonempty, _runtime_error,
        _retry_backoff_seconds) that MUST be popped before the summary enters
        score_summary_doc.
        """
        loop = asyncio.get_running_loop()
        item_start = time.time()
        label = str(item.get("icp_ref") or item.get("icp_hash") or "unknown_icp")
        if telemetry_session is not None:
            await telemetry_session.attempt_started(
                icp_ref=label,
                icp_hash=str(item.get("icp_hash") or ""),
                icp_ordinal=max(0, item_index - 1),
                model_role=telemetry_model_role,
                retry_round=retry_round,
            )
        await self._ensure_private_baseline_repo_head_unchanged(
            expected_git_sha=expected_repo_main_git_sha,
            benchmark_date=benchmark_date,
            item_index=item_index,
            total_icps=total_icps,
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE ICP STARTED",
                (
                    ("Worker", self.worker_ref),
                    ("ICP", f"{item_index}/{total_icps}"),
                    ("ICP ref", compact_ref(label)),
                    ("ICP hash", compact_ref(item.get("icp_hash"))),
                    ("Set", item.get("set_id")),
                    ("Day", item.get("day_index")),
                    ("Day rank", item.get("day_rank")),
                ),
            )
        )
        runtime_error = ""
        scorer_error = ""
        retryable = False
        retry_backoff_seconds = 0.0
        # In-container trace collection (§9.1): install a per-task collector so
        # the runner's stderr trace markers are published instead of dropped.
        # run_in_executor does NOT copy contextvars (asyncio.to_thread does), so
        # the context is copied explicitly AFTER the collector is installed —
        # the executor thread then appends to the same list object.
        trace_entries: list[dict[str, Any]] | None = None
        trace_token: contextvars.Token | None = None
        if trace_context is not None and incontainer_trace_capture_enabled():
            trace_entries, trace_token = begin_incontainer_trace_collection()
        try:
            if inspect.iscoroutinefunction(getattr(runner, "__call__", None)):
                raw_outputs = await runner(item["icp"], {"mode": mode_label})
            else:
                runner_call: Any = functools.partial(
                    runner,
                    item["icp"],
                    {"mode": mode_label},
                )
                if trace_token is not None:
                    runner_call = functools.partial(
                        contextvars.copy_context().run,
                        runner_call,
                    )
                raw_outputs = await loop.run_in_executor(executor, runner_call)
            outputs = ensure_private_model_outputs(
                raw_outputs,
                context_label=f"{mode_label} for {label}",
                require_non_empty=False,
            )
        except PrivateModelRuntimeError as exc:
            outputs = []
            runtime_error = _short_error(exc)
            # Classify from the full exception text: _short_error truncates to
            # 300 chars and can drop the status marker the classifier needs.
            retryable = _baseline_error_is_retryable(str(exc))
            retry_backoff_seconds = max(
                retry_backoff_seconds,
                _baseline_429_retry_backoff_seconds(str(exc)),
            )
            logger.warning(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE ICP RUNTIME ERROR",
                    (
                        ("Worker", self.worker_ref),
                        ("ICP", f"{item_index}/{total_icps}"),
                        ("ICP ref", compact_ref(label)),
                        ("ICP hash", compact_ref(item.get("icp_hash"))),
                        ("Retryable", retryable),
                        ("Error", runtime_error),
                    ),
                )
            )
        finally:
            if trace_token is not None:
                end_incontainer_trace_collection(trace_token)
        item_elapsed = time.time() - item_start
        if telemetry_session is not None and not runtime_error:
            await telemetry_session.lifecycle(
                "sourcing_completed",
                {
                    "icp_ref": label,
                    "model_role": telemetry_model_role,
                    "retry_round": retry_round,
                    "sourced_company_count": len(outputs),
                },
            )
        score_breakdowns: list[dict[str, Any]] = []
        if outputs:
            if telemetry_session is not None:
                await telemetry_session.lifecycle(
                    "scoring_started",
                    {
                        "icp_ref": label,
                        "model_role": telemetry_model_role,
                        "retry_round": retry_round,
                    },
                )
            try:
                score_breakdowns = await self._score_baseline_outputs(
                    scorer=scorer,
                    outputs=outputs,
                    icp=item["icp"],
                    scorer_semaphore=scorer_semaphore,
                )
            except Exception as scorer_exc:  # noqa: BLE001 - §0-N6: non-fatal for the batch
                scorer_error = _short_error(scorer_exc)
                retryable = retryable or _baseline_error_is_retryable(str(scorer_exc))
                retry_backoff_seconds = max(
                    retry_backoff_seconds,
                    _baseline_429_retry_backoff_seconds(str(scorer_exc)),
                )
                logger.warning(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE ICP SCORER ERROR",
                        (
                            ("Worker", self.worker_ref),
                            ("ICP", f"{item_index}/{total_icps}"),
                            ("ICP ref", compact_ref(label)),
                            ("ICP hash", compact_ref(item.get("icp_hash"))),
                            ("Retryable", retryable),
                            ("Error", scorer_error),
                        ),
                    )
                )
        scores = [float(row.get("final_score", 0.0) or 0.0) for row in score_breakdowns]
        # Shared flag-aware per-ICP score (see the isolation-path comment):
        # capped mode divides by the ICP's requested company count.
        icp_score = benchmark_icp_score_from_company_scores(
            scores,
            requested_count=_icp_company_goal(item["icp"]),
            fp_penalty_total=fp_penalty_total_from_breakdowns(
                score_breakdowns, item["icp"]
            ),
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE ICP SCORED",
                (
                    ("Worker", self.worker_ref),
                    ("ICP", f"{item_index}/{total_icps}"),
                    ("ICP ref", compact_ref(label)),
                    ("ICP hash", compact_ref(item.get("icp_hash"))),
                    ("Set", item.get("set_id")),
                    ("Day", item.get("day_index")),
                    ("Day rank", item.get("day_rank")),
                    ("Score", f"{icp_score:.4f}"),
                    ("Companies", len(scores)),
                    ("Non-empty output", bool(outputs)),
                    ("Runtime error", runtime_error or "-"),
                    ("Scorer error", scorer_error or "-"),
                    ("ICP runtime", f"{item_elapsed:.1f}s"),
                    ("Elapsed", f"{time.time() - run_start:.1f}s"),
                ),
            )
        )
        item_summary = sanitize_benchmark_item_summary(
            item=item,
            score=icp_score,
            company_count=len(scores),
            score_breakdowns=score_breakdowns,
            # Model output count BEFORE the scorer's employee-bucket
            # pre-filter, so the funnel's first stage is the true
            # "companies discovered" number.
            sourced_count=len(outputs),
        )
        if runtime_error or scorer_error:
            diagnostics = dict(item_summary.get("diagnostics") or {})
            runtime_diagnostics = _runtime_error_diagnostics(runtime_error or scorer_error)
            categories = set(diagnostics.get("failure_categories") or [])
            if runtime_error:
                categories.add("runtime_provider_error")
            if scorer_error:
                categories.add("scorer_provider_error")
            categories.add(str(runtime_diagnostics["category"]))
            diagnostics["failure_categories"] = sorted(categories)
            diagnostics["runtime_error"] = runtime_diagnostics
            item_summary["diagnostics"] = diagnostics
        await self._record_baseline_icp_traces(
            item=item,
            item_summary=item_summary,
            outputs=outputs,
            score_breakdowns=score_breakdowns,
            trace_entries=trace_entries,
            trace_context=trace_context,
            telemetry_session=telemetry_session,
            telemetry_model_role=telemetry_model_role,
        )
        _apply_provider_cost_baseline_outcome(item_summary)
        diagnostics = item_summary.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            categories = {str(value) for value in diagnostics.get("failure_categories") or []}
            if categories.intersection({"provider_cost_cap_blocked", "provider_cost_tracking_failed"}):
                retryable = False
        item_summary["_item_index"] = item_index
        item_summary["_retryable"] = retryable
        item_summary["_nonempty"] = bool(outputs)
        item_summary["_runtime_error"] = runtime_error or scorer_error
        item_summary["_retry_backoff_seconds"] = retry_backoff_seconds
        if telemetry_session is not None and (runtime_error or scorer_error):
            await telemetry_session.lifecycle(
                "attempt_failed",
                {
                    "icp_ref": label,
                    "model_role": telemetry_model_role,
                    "retry_round": retry_round,
                    "retryable": retryable,
                    "failure_category": (
                        "runtime_provider_error" if runtime_error else "scorer_provider_error"
                    ),
                    "error": runtime_error or scorer_error,
                },
            )
        return item_summary

    async def _score_baseline_outputs(
        self,
        *,
        scorer: QualificationStyleCompanyScorer,
        outputs: list[Any],
        icp: Mapping[str, Any],
        scorer_semaphore: asyncio.Semaphore | None = None,
        apply_isolation: bool = False,
    ) -> list[dict[str, Any]]:
        """Baseline-batch scorer call with the §0-N6 semantics.

        * optional benchmark scorer-key isolation (``apply_isolation`` — the
          serial loop applies it per call; the parallel batch applies ONE
          batch-wide scope instead, because per-call enter/exit under
          concurrency would restore prod keys mid-flight for sibling tasks);
        * optional burst semaphore (RESEARCH_LAB_BENCHMARK_SCORER_MAX_CONCURRENCY);
        * one inline retry for transient scorer provider errors (classified by
          the same rules as runner errors).
        """
        # max_companies is scorer-enforced: when the ICP pins a goal, at most
        # that many companies are scored/counted (model output is best-first,
        # so the head keeps its best). A model ignoring the field cannot earn
        # credit for extra companies.
        goal = _icp_company_goal(icp)
        if goal is not None and len(outputs) > goal:
            logger.warning(
                "research_lab_outputs_capped_to_goal goal=%s submitted=%s",
                goal, len(outputs),
            )
            outputs = list(outputs)[:goal]

        async def _call() -> list[dict[str, Any]]:
            if scorer_semaphore is None:
                return await scorer.score_with_breakdowns(outputs, icp, True)
            async with scorer_semaphore:
                return await scorer.score_with_breakdowns(outputs, icp, True)

        async def _guarded() -> list[dict[str, Any]]:
            try:
                return await _call()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not _baseline_error_is_retryable(str(exc)):
                    raise
                logger.warning(
                    "research_lab_baseline_scorer_retry worker_ref=%s error=%s",
                    self.worker_ref,
                    _short_error(exc),
                )
                backoff_seconds = _baseline_429_retry_backoff_seconds(str(exc))
                if backoff_seconds > 0:
                    logger.warning(
                        "research_lab_baseline_scorer_rate_limit_backoff worker_ref=%s backoff_seconds=%.1f error=%s",
                        self.worker_ref,
                        backoff_seconds,
                        _short_error(exc),
                    )
                    await asyncio.sleep(backoff_seconds)
                else:
                    await asyncio.sleep(2.0)
                return await _call()

        if not apply_isolation:
            return await _guarded()
        with _benchmark_scorer_isolation():
            return await _guarded()

    async def _run_baseline_batch(
        self,
        *,
        runner: AttestedPrivateModelRunnerV2,
        retry_runner: AttestedPrivateModelRunnerV2,
        scorer: QualificationStyleCompanyScorer,
        window: Any,
        run_start: float,
        mode_label: str = "private_baseline",
        trace_context: Mapping[str, Any] | None = None,
        resume_results: list[dict[str, Any]] | None = None,
        icp_checkpoint: Callable[[dict[str, Any]], Any] | None = None,
        expected_repo_main_git_sha: str = "",
        benchmark_date: str = "",
        telemetry_session: ScoringTelemetrySession | None = None,
        telemetry_model_role: str = "reference",
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Run all benchmark ICPs concurrently, then retry transient failures.

        The whole batch (first pass + retry rounds) runs inside ONE benchmark
        scorer-key isolation scope (§0-N6): scorer calls burst on the dedicated
        benchmark Scrapingdog/OpenRouter keys when configured, container runs
        keep their construction-time prod keys via extra_env.
        """
        with _benchmark_scorer_isolation():
            return await self._run_baseline_batch_inner(
                runner=runner,
                retry_runner=retry_runner,
                scorer=scorer,
                window=window,
                run_start=run_start,
                mode_label=mode_label,
                trace_context=trace_context,
                resume_results=resume_results,
                icp_checkpoint=icp_checkpoint,
                expected_repo_main_git_sha=expected_repo_main_git_sha,
                benchmark_date=benchmark_date,
                telemetry_session=telemetry_session,
                telemetry_model_role=telemetry_model_role,
            )

    async def _run_baseline_batch_inner(
        self,
        *,
        runner: AttestedPrivateModelRunnerV2,
        retry_runner: AttestedPrivateModelRunnerV2,
        scorer: QualificationStyleCompanyScorer,
        window: Any,
        run_start: float,
        mode_label: str = "private_baseline",
        trace_context: Mapping[str, Any] | None = None,
        resume_results: list[dict[str, Any]] | None = None,
        icp_checkpoint: Callable[[dict[str, Any]], Any] | None = None,
        expected_repo_main_git_sha: str = "",
        benchmark_date: str = "",
        telemetry_session: ScoringTelemetrySession | None = None,
        telemetry_model_role: str = "reference",
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Run all benchmark ICPs concurrently, then retry transient failures.

        Returns the ordered per-ICP summaries plus retry stats
        (``retried``/``recovered``/``unresolved``) for baseline health diagnostics.

        First pass fans out at private_baseline_concurrency. Provider/infra
        errors classified retryable are re-run for up to
        private_baseline_provider_retry_rounds at the lower retry concurrency
        (with the aggregate Exa budget re-spread by the retry runner env), so a
        transient 429 never leaves a valid ICP scored 0. ICPs that still fail
        keep their error and score 0 — aggregate semantics are unchanged.

        Results are reassembled in benchmark_items order: score_summary_doc is
        canonically hashed and the promotion gate consumes that hash, so
        ordering must be deterministic regardless of completion order.
        """
        concurrency = self.config.private_baseline_concurrency
        items = list(enumerate(window.benchmark_items, start=1))
        total_icps = len(items)
        resumed_by_ref = _progress_rows_by_icp_ref(resume_results)
        results: dict[int, dict[str, Any]] = {}
        run_items: list[tuple[int, Mapping[str, Any]]] = []
        for item_index, item in items:
            ref = _benchmark_item_ref_for_progress(item)
            resumed = resumed_by_ref.get(ref)
            if resumed is None:
                run_items.append((item_index, item))
                continue
            restored = dict(resumed)
            restored["_item_index"] = item_index
            restored["_retryable"] = False
            restored["_nonempty"] = _baseline_summary_nonempty(restored)
            restored["_runtime_error"] = ""
            restored["_retry_backoff_seconds"] = 0.0
            results[item_index] = restored
        if resumed_by_ref:
            logger.info(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE PARALLEL RESUME",
                    (
                        ("Worker", self.worker_ref),
                        ("Resumed ICPs", len(results)),
                        ("Remaining ICPs", len(run_items)),
                        ("Total ICPs", total_icps),
                    ),
                )
            )
        scorer_limit = _benchmark_scorer_max_concurrency()
        scorer_semaphore = asyncio.Semaphore(scorer_limit) if scorer_limit > 0 else None
        executor = concurrent.futures.ThreadPoolExecutor(
            # Retry rounds reuse this pool — size for whichever phase is wider.
            max_workers=max(concurrency, self.config.private_baseline_retry_concurrency),
            thread_name_prefix="baseline-icp",
        )
        try:
            semaphore = asyncio.Semaphore(concurrency)

            async def run_one(item_index: int, item: Mapping[str, Any]) -> dict[str, Any]:
                async with semaphore:
                    entry = await self._run_baseline_icp(
                        runner=runner,
                        scorer=scorer,
                        item=item,
                        item_index=item_index,
                        total_icps=total_icps,
                        run_start=run_start,
                        executor=executor,
                        scorer_semaphore=scorer_semaphore,
                        mode_label=mode_label,
                        trace_context=trace_context,
                        expected_repo_main_git_sha=expected_repo_main_git_sha,
                        benchmark_date=benchmark_date,
                        telemetry_session=telemetry_session,
                        telemetry_model_role=telemetry_model_role,
                        retry_round=0,
                    )
                    if icp_checkpoint is not None and _baseline_summary_checkpointable(entry):
                        await icp_checkpoint(entry)
                    return entry

            # Settle-then-raise: blocking docker waits cannot be cancelled, so a
            # fail-fast gather would leave containers racing the failure path.
            # Wait for every task to finish (bounded by the per-container
            # timeout), then surface the first fatal error.
            settled = await asyncio.gather(
                *(run_one(item_index, item) for item_index, item in run_items),
                return_exceptions=True,
            )
            fatal = [entry for entry in settled if isinstance(entry, BaseException)]
            if fatal:
                raise fatal[0]
            for entry in settled:
                results[entry["_item_index"]] = entry

            retried_total = 0
            recovered_total = 0
            for round_no in range(1, self.config.private_baseline_provider_retry_rounds + 1):
                pending = sorted(
                    item_index for item_index, entry in results.items() if entry.get("_retryable")
                )
                if not pending:
                    break
                logger.info(
                    format_worker_block(
                        "RESEARCH LAB PRIVATE BASELINE RETRY ROUND",
                        (
                            ("Worker", self.worker_ref),
                            ("Round", f"{round_no}/{self.config.private_baseline_provider_retry_rounds}"),
                            ("Retrying ICPs", len(pending)),
                            ("ICP indexes", ", ".join(str(item_index) for item_index in pending)),
                            ("Retry concurrency", self.config.private_baseline_retry_concurrency),
                        ),
                    )
                )
                retry_semaphore = asyncio.Semaphore(self.config.private_baseline_retry_concurrency)
                round_retry_runner = _retry_runner_with_provider_cost_scope(
                    retry_runner,
                    retry_round=round_no,
                )

                async def retry_one(item_index: int) -> dict[str, Any]:
                    async with retry_semaphore:
                        backoff_seconds = float(
                            results.get(item_index, {}).get("_retry_backoff_seconds") or 0.0
                        )
                        if backoff_seconds > 0:
                            logger.warning(
                                "research_lab_baseline_icp_rate_limit_backoff worker_ref=%s round=%s icp_index=%s backoff_seconds=%.1f",
                                self.worker_ref,
                                round_no,
                                item_index,
                                backoff_seconds,
                            )
                            await asyncio.sleep(backoff_seconds)
                        entry = await self._run_baseline_icp(
                            runner=round_retry_runner,
                            scorer=scorer,
                            item=window.benchmark_items[item_index - 1],
                            item_index=item_index,
                            total_icps=total_icps,
                            run_start=run_start,
                            executor=executor,
                            scorer_semaphore=scorer_semaphore,
                            mode_label=mode_label,
                            trace_context=trace_context,
                            expected_repo_main_git_sha=expected_repo_main_git_sha,
                            benchmark_date=benchmark_date,
                            telemetry_session=telemetry_session,
                            telemetry_model_role=telemetry_model_role,
                            retry_round=round_no,
                        )
                        if icp_checkpoint is not None and _baseline_summary_checkpointable(entry):
                            await icp_checkpoint(entry)
                        return entry

                retried = await asyncio.gather(
                    *(retry_one(item_index) for item_index in pending),
                    return_exceptions=True,
                )
                fatal = [entry for entry in retried if isinstance(entry, BaseException)]
                if fatal:
                    raise fatal[0]
                recovered = 0
                for entry in retried:
                    if not entry.get("_runtime_error"):
                        recovered += 1
                    # Replace on success AND on repeat failure: the fresher
                    # attempt carries the more current diagnostics.
                    results[entry["_item_index"]] = entry
                retried_total += len(pending)
                recovered_total += recovered
            unresolved = sorted(
                item_index for item_index, entry in results.items() if entry.get("_runtime_error")
            )
            logger.info(
                format_worker_block(
                    "RESEARCH LAB PRIVATE BASELINE PARALLEL SUMMARY",
                    (
                        ("Worker", self.worker_ref),
                        ("ICPs", total_icps),
                        ("Concurrency", concurrency),
                        ("Retried", retried_total),
                        ("Recovered", recovered_total),
                        ("Unresolved provider errors", len(unresolved)),
                        ("Unresolved ICP indexes", ", ".join(str(item_index) for item_index in unresolved) or "-"),
                        ("Elapsed", f"{time.time() - run_start:.1f}s"),
                    ),
                )
            )
            return (
                [results[item_index] for item_index in sorted(results)],
                {
                    "retried": retried_total,
                    "recovered": recovered_total,
                    "unresolved": len(unresolved),
                },
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    async def _reusable_same_day_benchmark(
        self,
        *,
        today: str,
        manifest_hash: str,
        expected_policy_hash: str = "",
    ) -> dict[str, Any] | None:
        """Any valid benchmark already recorded today for this model (any window).

        A restart used to silently replace the day's promotion reference
        mid-day (§0.3). Reuse therefore requires the active policy generation;
        an actual policy cutover must produce its own baseline.
        """
        if _env_flag("RESEARCH_LAB_BASELINE_ALLOW_SAMEDAY_REPLACE"):
            return None
        rows = await select_many(
            "research_lab_private_model_benchmark_current",
            columns="*",
            filters=(
                ("benchmark_date", today),
                ("private_model_manifest_hash", manifest_hash),
            ),
            order_by=(("created_at", True),),
            limit=25,
        )
        for row in rows:
            if (
                _private_benchmark_row_is_valid(row)
                and _private_benchmark_matches_policy(
                    row,
                    expected_policy_hash=expected_policy_hash,
                )
            ):
                return dict(row)
        return None

    async def _baseline_dispatch_history(
        self,
        *,
        today: str,
        window_hash: str,
        manifest_hash: str,
    ) -> list[dict[str, Any]]:
        rows = await select_many(
            "research_lab_scoring_dispatch_events",
            columns=(
                "dispatch_event_id,dispatch_status,worker_ref,rolling_window_hash,"
                "event_doc,created_at"
            ),
            filters=(
                ("dispatch_type", "private_baseline_rebenchmark"),
                ("rolling_window_hash", window_hash),
            ),
            order_by=(("created_at", True),),
            limit=250,
        )
        matching: list[dict[str, Any]] = []
        for row in rows:
            doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
            if str(doc.get("benchmark_date") or "") != today:
                continue
            row_manifest_hash = str(doc.get("private_model_manifest_hash") or "")
            if row_manifest_hash and row_manifest_hash != manifest_hash:
                continue
            matching.append(dict(row))
        return matching

    async def _same_day_reference_recorded_while_running(
        self,
        *,
        today: str,
        window_hash: str,
        manifest_hash: str,
        expected_policy_hash: str = "",
    ) -> dict[str, Any] | None:
        """Re-check just before recording: another worker (or a restart) may have
        recorded the day's reference while this multi-hour run was in flight."""
        if _env_flag("RESEARCH_LAB_BASELINE_ALLOW_SAMEDAY_REPLACE"):
            return None
        try:
            rows = await select_many(
                "research_lab_private_model_benchmark_current",
                columns="*",
                filters=(
                    ("benchmark_date", today),
                    ("private_model_manifest_hash", manifest_hash),
                ),
                order_by=(("created_at", True),),
                limit=25,
            )
        except Exception as exc:
            logger.warning("research_lab_baseline_pre_record_check_failed error=%s", str(exc)[:200])
            return None
        for row in rows:
            if (
                _private_benchmark_row_is_valid(row)
                and _private_benchmark_matches_policy(
                    row,
                    expected_policy_hash=expected_policy_hash,
                )
            ):
                return dict(row)
        return None

    async def _baseline_day_jump_points(
        self,
        *,
        today: str,
        aggregate_score: float,
    ) -> float | None:
        """Aggregate jump vs the most recent valid pre-today benchmark.

        Best-effort: a lookup failure never fails the baseline. Always logs the
        jump loudly — day-over-day drift of 3-5 points has been observed live,
        several times the 1.0-point promotion threshold (§0-N1).
        """
        try:
            rows = await select_many(
                "research_lab_private_model_benchmark_current",
                columns="*",
                filters=(("benchmark_date", "lt", today),),
                order_by=(("benchmark_date", True), ("created_at", True)),
                limit=10,
            )
        except Exception as exc:
            logger.warning("research_lab_baseline_day_jump_lookup_failed error=%s", str(exc)[:200])
            return None
        previous = next((row for row in rows if _private_benchmark_row_is_valid(row)), None)
        if previous is None:
            return None
        previous_aggregate = _safe_float(previous.get("aggregate_score"), default=0.0)
        day_jump = float(aggregate_score) - previous_aggregate
        logger.warning(
            format_worker_block(
                "RESEARCH LAB PRIVATE BASELINE DAY-OVER-DAY JUMP",
                (
                    ("Worker", self.worker_ref),
                    ("Benchmark date", today),
                    ("Aggregate score", f"{aggregate_score:.4f}"),
                    ("Previous date", previous.get("benchmark_date")),
                    ("Previous aggregate", f"{previous_aggregate:.4f}"),
                    ("Previous model", compact_ref(previous.get("private_model_manifest_hash"))),
                    ("Jump", f"{day_jump:+.4f}"),
                    ("Quarantine threshold", _baseline_max_day_jump_points() or "warn-only"),
                ),
            )
        )
        return day_jump

    async def _baseline_leased_by_other_worker(self, today: str) -> bool:
        """Best-effort lease via dispatch events so any worker can run baselines
        (RESEARCH_LAB_BASELINE_ANY_WORKER) without doubling the day's run."""
        try:
            lease_seconds = max(600, int(os.getenv("RESEARCH_LAB_BASELINE_LEASE_SECONDS", "5400")))
        except ValueError:
            lease_seconds = 5400
        try:
            rows = await select_many(
                "research_lab_scoring_dispatch_events",
                columns="dispatch_event_id,dispatch_status,worker_ref,event_doc,created_at",
                filters=(("dispatch_type", "private_baseline_rebenchmark"),),
                order_by=(("created_at", True),),
                limit=50,
            )
        except Exception as exc:
            logger.warning("research_lab_baseline_lease_check_failed error=%s", str(exc)[:200])
            return False
        for row in rows:
            doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
            if str(doc.get("benchmark_date") or "") != today:
                continue
            # Rows are newest-first: the first event for today decides.
            status = str(row.get("dispatch_status") or "")
            if status in ("completed", "failed"):
                return False
            if status == "assigned" and str(row.get("worker_ref") or "") != self.worker_ref:
                age = _status_age_seconds(row.get("created_at"))
                if age is not None and age <= lease_seconds:
                    logger.info(
                        "research_lab_baseline_leased_elsewhere worker_ref=%s owner=%s age_seconds=%.0f",
                        self.worker_ref,
                        row.get("worker_ref"),
                        age,
                    )
                    return True
            return False
        return False

    async def _confirm_baseline_lease(
        self,
        *,
        today: str,
        lease_event: Mapping[str, Any] | None,
    ) -> bool:
        """Ownership confirm after writing our `assigned` lease marker.

        The winner is the earliest unexpired open lease for today: per worker,
        only its LATEST private_baseline_rebenchmark event counts (a later
        completed/failed closes that worker's lease), leases older than
        RESEARCH_LAB_BASELINE_LEASE_SECONDS are expired (dead worker), and ties
        break on dispatch_event_id. Read failures confirm optimistically — the
        same-day-reuse guard and the pre-record conflict check backstop
        duplicate recordings; this confirm only avoids wasting a duplicate
        multi-hour run.
        """
        try:
            lease_seconds = max(600, int(os.getenv("RESEARCH_LAB_BASELINE_LEASE_SECONDS", "5400")))
        except ValueError:
            lease_seconds = 5400
        try:
            rows = await select_many(
                "research_lab_scoring_dispatch_events",
                columns="dispatch_event_id,dispatch_status,worker_ref,event_doc,created_at",
                filters=(("dispatch_type", "private_baseline_rebenchmark"),),
                order_by=(("created_at", True),),
                limit=100,
            )
        except Exception as exc:
            logger.warning("research_lab_baseline_lease_confirm_failed error=%s", str(exc)[:200])
            return True
        latest_by_worker: dict[str, dict[str, Any]] = {}
        for row in rows:
            doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
            if str(doc.get("benchmark_date") or "") != today:
                continue
            worker = str(row.get("worker_ref") or "")
            if worker and worker not in latest_by_worker:
                # Rows are newest-first: the first row per worker is its
                # current lease state for today.
                latest_by_worker[worker] = row
        open_leases: list[dict[str, Any]] = []
        for row in latest_by_worker.values():
            if str(row.get("dispatch_status") or "") != "assigned":
                continue
            age = _status_age_seconds(row.get("created_at"))
            if age is None or age > lease_seconds:
                continue
            open_leases.append(row)
        if not open_leases:
            return True
        winner = min(
            open_leases,
            key=lambda row: (str(row.get("created_at") or ""), str(row.get("dispatch_event_id") or "")),
        )
        if str(winner.get("worker_ref") or "") == self.worker_ref:
            return True
        lease_event_id = str((lease_event or {}).get("dispatch_event_id") or "")
        if lease_event_id and str(winner.get("dispatch_event_id") or "") == lease_event_id:
            return True
        logger.info(
            "research_lab_baseline_lease_lost worker_ref=%s winner_worker=%s winner_event=%s",
            self.worker_ref,
            winner.get("worker_ref"),
            compact_ref(winner.get("dispatch_event_id")),
        )
        return False

    def _is_private_baseline_owner(self) -> bool:
        if self.config.scoring_worker_index == 0:
            return True
        # Opt-in: let any worker run the daily baseline (leased via dispatch
        # events) instead of being hostage to worker 0.
        return _env_flag("RESEARCH_LAB_BASELINE_ANY_WORKER")

    async def _resolve_evaluation_epoch(self) -> int:
        # Always enter the shared resolver before consulting any chain-value
        # cache.  It force-refreshes the durable cutover lifecycle first, so a
        # long-running legacy worker cannot reuse an old ordinal after the
        # singleton flips to stateful_active.  The shared resolver retains its
        # own legacy chain-value cache only after that fresh lifecycle gate.
        self._resolved_epoch_cache = None
        epoch, block, source = await resolve_research_lab_evaluation_epoch(
            self.config.evaluation_epoch
        )

        if epoch <= 0:
            raise RuntimeError(
                "Research Lab evaluation epoch resolved to 0; refusing to write epoch-0 score/audit bundles"
            )
        self._resolved_epoch_cache = None
        logger.info(
            "Research Lab scoring worker resolved evaluation epoch: epoch=%s block=%s source=%s",
            epoch,
            block,
            source,
        )
        return epoch

    async def _write_audit_bundle(self, epoch: int) -> None:
        """Best-effort audit bundle write: it runs on the scoring hot path after
        every scored candidate/baseline, so a failure here must never fail the
        pass (bug #9/#36 — it used to mislabel every scored candidate)."""
        try:
            await self._write_audit_bundle_inner(epoch)
        except Exception as exc:
            logger.warning(
                "research_lab_audit_bundle_write_failed epoch=%s error=%s",
                epoch,
                _short_error(exc),
            )
            try:
                await create_scoring_dispatch_event(
                    dispatch_type="audit_bundle_build",
                    dispatch_status="failed",
                    worker_ref=self.worker_ref,
                    proxy_ref_hash=self.proxy_ref_hash,
                    event_doc={
                        "evaluation_epoch": int(epoch),
                        "error_diagnostics": _event_error_diagnostics(exc),
                    },
                )
            except Exception:
                logger.exception("research_lab_audit_bundle_failure_event_write_failed")

    async def _write_audit_bundle_inner(self, epoch: int) -> None:
        # The five append-only event tables grow without bound and overflow the
        # select_all cap; scope them to a recent window like score bundles are
        # scoped by epoch (bug #9).
        event_window = self._audit_event_window_filters()
        ticket_rows = await self._audit_select_all("research_loop_ticket_current", current_view=True)
        queue_rows = await self._audit_select_all("research_loop_run_queue_current", current_view=True)
        receipt_rows = await self._audit_select_all("research_loop_receipt_current", current_view=True)
        candidate_rows = await self._audit_select_all("research_lab_candidate_evaluation_current", current_view=True)
        candidate_event_rows = await self._audit_select_all("research_lab_candidate_evaluation_events", filters=event_window)
        loop_event_rows = await self._audit_select_all("research_lab_auto_research_loop_events", filters=event_window)
        dispatch_event_rows = await self._audit_select_all("research_lab_scoring_dispatch_events", filters=event_window)
        rolling_window_rows = await self._audit_select_all("research_lab_rolling_icp_windows")
        benchmark_rows = await self._audit_select_all("research_lab_private_model_benchmark_current", current_view=True)
        private_model_version_rows = await self._audit_select_all("research_lab_private_model_version_current", current_view=True)
        promotion_event_rows = await self._audit_select_all("research_lab_candidate_promotion_events", filters=event_window)
        private_repo_commit_event_rows = await self._audit_select_all("research_lab_private_repo_commit_events", filters=event_window)
        public_benchmark_report_rows = await self._audit_select_all(
            "research_lab_public_benchmark_report_current",
            current_view=True,
        )
        score_bundle_rows = await self._audit_select_all(
            "research_evaluation_score_bundle_current",
            filters=(("evaluation_epoch", epoch),),
            current_view=True,
        )
        bundle_doc = build_research_lab_audit_bundle(
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
        audit_hash = canonical_hash(bundle_doc)
        signature_ref = await asyncio.to_thread(
            sign_digest_with_kms,
            key_id=self.config.score_bundle_kms_key_id,
            digest_hash=audit_hash,
            signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
        )
        bundle, _event = await create_signed_audit_bundle(
            epoch=epoch,
            bundle_doc=bundle_doc,
            signature_ref=signature_ref,
        )
        await create_scoring_dispatch_event(
            dispatch_type="audit_bundle_build",
            dispatch_status="completed",
            worker_ref=self.worker_ref,
            proxy_ref_hash=self.proxy_ref_hash,
            event_doc={
                "audit_bundle_id": str(bundle["audit_bundle_id"]),
                "audit_bundle_hash": str(bundle["audit_bundle_hash"]),
            },
        )
        logger.info(
            format_worker_block(
                "RESEARCH LAB AUDIT BUNDLE WRITTEN",
                (
                    ("Worker", self.worker_ref),
                    ("Epoch", epoch),
                    ("Audit bundle", compact_ref(bundle["audit_bundle_id"])),
                    ("Audit hash", compact_ref(bundle["audit_bundle_hash"])),
                ),
            )
        )

    def _audit_event_window_filters(self) -> tuple[tuple[Any, ...], ...]:
        try:
            days = float(os.getenv("RESEARCH_LAB_AUDIT_EVENT_WINDOW_DAYS", "3"))
        except ValueError:
            days = 3.0
        if days <= 0:
            # Explicit opt-out restores unscoped full-history fetches.
            return ()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return (("created_at", "gte", cutoff.isoformat()),)

    def _audit_select_limits(self) -> tuple[int, int]:
        max_rows = max(1000, _env_int("RESEARCH_LAB_AUDIT_SELECT_MAX_ROWS", 10000))
        batch_size = max(100, min(max_rows, _env_int("RESEARCH_LAB_AUDIT_SELECT_BATCH_SIZE", 500)))
        return max_rows, batch_size

    async def _audit_select_all(
        self,
        table: str,
        *,
        filters: tuple[tuple[Any, ...], ...] = (),
        current_view: bool = False,
    ) -> list[dict[str, Any]]:
        primary_order = (("current_status_at", True),) if current_view else (("created_at", True),)
        max_rows, batch_size = self._audit_select_limits()
        try:
            return await select_all(
                table,
                filters=filters,
                order_by=primary_order,
                max_rows=max_rows,
                batch_size=batch_size,
                allow_partial=True,
            )
        except Exception as exc:
            logger.warning(
                "research_lab_audit_select_order_fallback table=%s order=%s error=%s",
                table,
                primary_order,
                str(exc)[:200],
            )
            return await select_all(
                table,
                filters=filters,
                max_rows=max_rows,
                batch_size=batch_size,
                allow_partial=True,
            )

    async def _maybe_finalize_candidate_receipt(self, candidate: Mapping[str, Any]) -> bool:
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
        has_scored_candidate = status_counts.get("scored", 0) > 0
        event_doc = {
            "run_id": str(candidate["run_id"]),
            "candidate_status_counts": status_counts,
            "score_bundle_ids": score_bundle_ids,
            "finalization_source": "gateway_qualification_worker_results",
        }
        try:
            await create_receipt_event(
                receipt_id=str(receipt_id),
                ticket_id=str(candidate["ticket_id"]),
                event_type="completed" if has_scored_candidate else "failed",
                receipt_status="completed" if has_scored_candidate else "failed",
                event_doc=event_doc,
            )
        except Exception as exc:
            if not _is_event_sequence_race_error(exc):
                raise
            latest_receipt = await select_one(
                "research_loop_receipt_current",
                filters=(("receipt_id", str(receipt_id)),),
            )
            if latest_receipt and latest_receipt.get("current_receipt_status") != "queued":
                logger.info(
                    "research_lab_receipt_finalization_race_lost receipt_id=%s status=%s",
                    compact_ref(receipt_id),
                    latest_receipt.get("current_receipt_status"),
                )
                return False
            raise
        try:
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
        except Exception as exc:
            if not _is_event_sequence_race_error(exc):
                raise
            logger.warning(
                "research_lab_ticket_finalization_race_lost ticket_id=%s receipt_id=%s error=%s",
                compact_ref(candidate["ticket_id"]),
                compact_ref(receipt_id),
                str(exc)[:240],
            )
        logger.info(
            format_worker_block(
                "RESEARCH LAB RECEIPT FINALIZED",
                (
                    ("Worker", self.worker_ref),
                    ("Receipt", compact_ref(receipt_id)),
                    ("Run", compact_ref(candidate["run_id"])),
                    ("Status", "completed" if has_scored_candidate else "failed"),
                    ("Candidates scored", status_counts.get("scored", 0)),
                    ("Candidates failed", status_counts.get("failed", 0)),
                    ("Score bundles", len(score_bundle_ids)),
                ),
            )
        )
        return True

    def _candidate_run_context(
        self,
        candidate: Mapping[str, Any],
        *,
        window_hash: str,
        evaluation_epoch: int,
    ) -> dict[str, Any]:
        # Bundle→trace forward linkage: when the candidate row carries its loop
        # node id (worker-side annotation in candidate_build_doc), stamp the
        # SAME deterministic execution_trace:<uuid5> ref the trajectory
        # projector writes for that node, so score bundles, engine_trace_
        # mappings, and execution_traces all join on one key. Candidates
        # without node linkage (pre-annotation rows) keep the legacy
        # worker-scoped ref; readers must tolerate both formats — historical
        # bundles are immutable.
        loop_node_id = ""
        build_doc_for_node = candidate.get("candidate_build_doc")
        if isinstance(build_doc_for_node, Mapping):
            loop_node_id = str(build_doc_for_node.get("loop_node_id") or "")
        if loop_node_id:
            execution_trace_ref = (
                f"execution_trace:{execution_trace_id_for_node(str(candidate['run_id']), loop_node_id)}"
            )
        else:
            execution_trace_ref = (
                f"gateway_qualification_worker:{self.worker_ref}:{candidate['candidate_id']}"
            )
        context = {
            "run_id": str(candidate["run_id"]),
            "ticket_id": str(candidate["ticket_id"]),
            "candidate_id": str(candidate["candidate_id"]),
            "miner_hotkey": str(candidate["miner_hotkey"]),
            "island": str(candidate.get("island") or "generalist"),
            "evaluation_epoch": int(evaluation_epoch),
            "evaluator_version": "leadpoet-gateway-qualification-worker:research-lab:v1",
            "run_scope": "candidate_scoring",
            "rolling_window_hash": str(window_hash),
            "provider_cache_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "private_model_version_id": str(candidate.get("private_model_version_id") or ""),
            "evidence_bundle_refs": [f"research_lab_rolling_icp_window:{window_hash}"],
            "execution_trace_ref": execution_trace_ref,
            "cost_ledger_ref": "cost_ledger:" + canonical_hash(
                {
                    "candidate_id": candidate["candidate_id"],
                    "worker_ref": self.worker_ref,
                    "rolling_window_hash": window_hash,
                }
            ).split(":", 1)[1],
        }
        if str(candidate.get("candidate_kind") or "") == "image_build":
            if candidate.get("candidate_source_diff_hash"):
                context["candidate_source_diff_hash"] = str(candidate["candidate_source_diff_hash"])
            build_doc = candidate.get("candidate_build_doc")
            if isinstance(build_doc, Mapping):
                context["candidate_build_ref"] = str(
                    build_doc.get("build_doc_hash")
                    or canonical_hash(build_doc)
                )
        return context

    def _evaluation_policy(self) -> dict[str, Any]:
        conditional_policy = self.config.conditional_validation_policy()
        expected_icp_count = (
            conditional_policy.total_icps
            if conditional_policy.enabled
            else self.config.lab_champion_eval_days * self.config.lab_champion_icps_per_day
        )
        return {
            "min_delta": float(
                os.environ.get(
                    "RESEARCH_LAB_MIN_DELTA",
                    str(self.config.improvement_threshold_points),
                )
            ),
            "min_successful_icps": int(
                os.environ.get(
                    "RESEARCH_LAB_MIN_SUCCESSFUL_ICPS",
                    str(expected_icp_count),
                )
            ),
            "max_hard_failures": int(os.environ.get("RESEARCH_LAB_MAX_HARD_FAILURES", "0")),
            "min_candidate_score": float(os.environ.get("RESEARCH_LAB_MIN_CANDIDATE_SCORE", "0")),
            "observed_cost_usd": 0.0,
            # FP penalty knobs ride in the bundle policy so the public
            # verifier recomputes per-ICP scores with the exact values the
            # builder used (defaults 0 = off keep historical scores). The
            # evaluator helpers are the single source of truth: falsified
            # intent inherits the main FP penalty unless overridden.
            "fp_penalty_points": _fp_penalty_points(),
            "fp_unverified_primary_penalty_points": (
                _fp_unverified_primary_penalty_points()
            ),
        }

    def _preliminary_evaluation_policy(self) -> dict[str, Any]:
        """Reproduce the pre-conditional policy for the frozen 20-ICP gate."""

        policy = self._evaluation_policy()
        if "RESEARCH_LAB_MIN_SUCCESSFUL_ICPS" not in os.environ:
            conditional_policy = self.config.conditional_validation_policy()
            if conditional_policy.enabled:
                policy["min_successful_icps"] = (
                    conditional_policy.public_total_icps
                    + conditional_policy.private_total_icps
                )
        return policy

    def _private_scoring_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for name in (
            "EXA_MAX_RPS",
            "SOURCING_DEEPLINE_FALLBACK",
            "SOURCING_DEEPLINE_TIMEOUT_S",
        ):
            value = os.getenv(name)
            if value:
                env[name] = value
        # Provider evidence replay: when the worker carries a recorded per-ICP
        # cache directory, candidate runs replay the recorded baseline
        # evidence for identical provider requests.
        evidence_cache_dir = os.getenv("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR")
        if evidence_cache_dir:
            env["RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR"] = evidence_cache_dir
        # Every run records its provider I/O at full fidelity: cached hits
        # replay recorded evidence, and any live call (cache miss or fresh
        # surface) is captured completely for audit and future replay.
        env["RESEARCH_LAB_PROVIDER_EVIDENCE_RECORD"] = "1"
        evidence_proxy_url = os.getenv("RESEARCH_LAB_EVIDENCE_PROXY_URL")
        if evidence_proxy_url:
            env["RESEARCH_LAB_EVIDENCE_PROXY_URL"] = evidence_proxy_url
        env["RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP"] = str(
            self.config.provider_cost_cap_usd_per_icp
        )
        env["RESEARCH_LAB_SCRAPINGDOG_COST_PER_CREDIT_USD"] = str(
            self.config.scrapingdog_cost_per_credit_usd
        )
        env["RESEARCH_LAB_SCRAPINGDOG_UNKNOWN_ENDPOINT_CREDITS"] = str(
            self.config.scrapingdog_unknown_endpoint_credits
        )
        env["RESEARCH_LAB_PROVIDER_COST_UNKNOWN_ENDPOINT_POLICY"] = str(
            self.config.provider_cost_unknown_endpoint_policy
        )
        scoring_cache_dir = os.getenv("RESEARCH_LAB_SCORING_CACHE_DIR")
        if scoring_cache_dir:
            env["RESEARCH_LAB_SCORING_CACHE_DIR"] = scoring_cache_dir
        if self.proxy_url and self.config.private_model_docker_global_proxy_enabled:
            env.update(
                {
                    "HTTP_PROXY": self.proxy_url,
                    "HTTPS_PROXY": self.proxy_url,
                    "http_proxy": self.proxy_url,
                    "https_proxy": self.proxy_url,
                }
            )
        no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy")
        if no_proxy:
            env["NO_PROXY"] = no_proxy
            env["no_proxy"] = no_proxy
        # P13 corpus-mode capture budgets: with an S3 persistence prefix set,
        # containers get the larger byte caps so tool-call payloads are not
        # silently truncated (explicit operator caps always win).
        from research_lab.eval.private_runtime import incontainer_trace_corpus_env

        env.update(incontainer_trace_corpus_env())
        return env

    def _with_provider_cost_evaluation_scope(
        self,
        env: Mapping[str, str],
        *,
        run_type: str,
        rolling_window_hash: str = "",
        artifact_hash: str = "",
        candidate_id: str = "",
        run_id: str = "",
        ticket_id: str = "",
        dispatch_event_id: str = "",
        benchmark_date: str = "",
        benchmark_attempt: int | None = None,
        confirmation_attempt: int | None = None,
        side: str = "",
        evaluation_epoch: int | None = None,
        started_at: float | None = None,
    ) -> dict[str, str]:
        scoped = dict(env)
        scope_doc: dict[str, Any] = {
            "schema_version": "research_lab_provider_cost_evaluation_scope.v1",
            "run_type": str(run_type or "unknown"),
            "worker_ref": self.worker_ref,
        }
        optional_values: dict[str, Any] = {
            "rolling_window_hash": rolling_window_hash,
            "artifact_hash": artifact_hash,
            "candidate_id": candidate_id,
            "run_id": run_id,
            "ticket_id": ticket_id,
            "dispatch_event_id": dispatch_event_id,
            "benchmark_date": benchmark_date,
            "side": side,
        }
        for key, value in optional_values.items():
            text = str(value or "").strip()
            if text:
                scope_doc[key] = text
        if benchmark_attempt is not None:
            scope_doc["benchmark_attempt"] = int(benchmark_attempt)
        if confirmation_attempt is not None:
            scope_doc["confirmation_attempt"] = int(confirmation_attempt)
        if evaluation_epoch is not None:
            scope_doc["evaluation_epoch"] = int(evaluation_epoch)
        if started_at is not None:
            scope_doc["started_at_ms"] = int(float(started_at) * 1000)
        scoped[PROVIDER_COST_EVALUATION_SCOPE_ENV] = sha256_json(scope_doc)
        return scoped

    def _private_baseline_scoring_env(self) -> dict[str, str]:
        """Candidate scoring env with the benchmark's dedicated Exa budget.

        The daily baseline can run many model containers at once; EXA_MAX_RPS is
        enforced per container, so the aggregate burst must be isolated from the
        prod Exa key and split across the concurrent containers. Both overrides
        are opt-in — unset config falls back to the prod values.
        """
        env = self._private_scoring_env()
        # The baseline run seeds the replay cache and must observe live
        # providers only, so it never receives a cache to consume (recording
        # is inherited from the base scoring env).
        env.pop("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR", None)
        if legacy_v1_enabled():
            benchmark_exa_key = os.getenv(
                "RESEARCH_LAB_BENCHMARK_EXA_API_KEY", ""
            ).strip()
            if benchmark_exa_key:
                env["EXA_API_KEY"] = benchmark_exa_key
        else:
            env[V2_PROVIDER_PROFILE_ENV] = "benchmark_model"
        if self.config.benchmark_exa_max_rps > 0:
            env["EXA_MAX_RPS"] = str(self.config.benchmark_exa_max_rps)
        return env

    def _private_baseline_retry_scoring_env(self) -> dict[str, str]:
        # Retry rounds run fewer containers; re-spread the SAME aggregate Exa
        # budget across them so retried ICPs finish faster instead of idling at
        # the first-pass per-container rate.
        env = self._private_baseline_scoring_env()
        if self.config.benchmark_exa_max_rps > 0:
            aggregate = self.config.benchmark_exa_max_rps * self.config.private_baseline_concurrency
            env["EXA_MAX_RPS"] = str(
                round(aggregate / max(1, self.config.private_baseline_retry_concurrency), 3)
            )
        return env

    def _private_model_env_passthrough(self) -> tuple[str, ...]:
        return private_model_env_passthrough(
            include_proxy=self.config.private_model_docker_global_proxy_enabled
        )


def _average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _next_benchmark_attempt(rows: list[Mapping[str, Any]]) -> int:
    attempts: list[int] = []
    for row in rows:
        value: Any = row.get("benchmark_attempt")
        if value is None:
            event_doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
            value = event_doc.get("benchmark_attempt")
        try:
            attempts.append(int(value or 0))
        except (TypeError, ValueError):
            attempts.append(0)
    return (max(attempts) + 1) if attempts else 0


def _private_benchmark_row_is_valid(row: Mapping[str, Any]) -> bool:
    status = str(row.get("current_benchmark_status") or row.get("benchmark_status") or "")
    if status and status != "completed":
        return False
    doc = row.get("score_summary_doc") if isinstance(row.get("score_summary_doc"), Mapping) else {}
    summaries = doc.get("per_icp_summaries") if isinstance(doc, Mapping) else None
    if not isinstance(summaries, list) or not summaries:
        return False
    if not any(_benchmark_summary_has_companies(item) for item in summaries):
        return False
    if str(row.get("benchmark_quality") or "") == "passed":
        return True
    try:
        return int(row.get("evaluation_epoch") or 0) > 0
    except (TypeError, ValueError):
        return False


def _private_benchmark_matches_policy(
    row: Mapping[str, Any],
    *,
    expected_policy_hash: str,
) -> bool:
    doc = (
        row.get("score_summary_doc")
        if isinstance(row.get("score_summary_doc"), Mapping)
        else {}
    )
    assignment = (
        doc.get("category_assignment")
        if isinstance(doc.get("category_assignment"), Mapping)
        else {}
    )
    observed_policy_hash = str(assignment.get("policy_hash") or "")
    return observed_policy_hash == expected_policy_hash


def _validate_candidate_conditional_policy_stamp(
    candidate: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    active_policy: Mapping[str, Any],
) -> None:
    """Prevent an in-progress candidate from crossing policy generations."""

    build_doc = (
        candidate.get("candidate_build_doc")
        if isinstance(candidate.get("candidate_build_doc"), Mapping)
        else {}
    )
    stamp = (
        build_doc.get("conditional_validation_policy")
        if isinstance(build_doc.get("conditional_validation_policy"), Mapping)
        else None
    )
    if stamp is None:
        # Historical candidates created before migration 97 retain their
        # existing behavior. Every new Git-tree candidate carries the stamp.
        return
    stamped_mode = str(stamp.get("mode") or "").strip().lower()
    active_mode = str(active_policy.get("mode") or "").strip().lower()
    if stamped_mode not in {"off", "enforce"}:
        raise CandidateBaselineNotReady("candidate_conditional_policy_stamp_invalid")
    if stamped_mode != active_mode:
        raise CandidateBaselineNotReady(
            "candidate_conditional_policy_changed_after_creation:"
            f"stamped={stamped_mode}:active={active_mode}"
        )
    gate_requires_conditional = bool(gate.get("conditional_validation_required"))
    if stamped_mode == "off":
        if gate_requires_conditional:
            raise CandidateBaselineNotReady(
                "candidate_stamped_legacy_but_daily_baseline_requires_conditional_validation"
            )
        return
    stamped_hash = str(stamp.get("policy_hash") or "")
    gate_hash = str(gate.get("conditional_validation_policy_hash") or "")
    if not gate_requires_conditional or not stamped_hash or stamped_hash != gate_hash:
        raise CandidateBaselineNotReady(
            "candidate_conditional_policy_commitment_mismatch:"
            f"stamped={compact_ref(stamped_hash)}:baseline={compact_ref(gate_hash)}"
        )


def _rolling_window_fetch_kwargs(config: Any) -> dict[str, Any]:
    conditional_policy_factory = getattr(config, "conditional_validation_policy", None)
    if callable(conditional_policy_factory):
        conditional_policy = conditional_policy_factory()
        if conditional_policy.enabled:
            return {
                "window_mode": "hybrid_fresh_retained",
                "fresh_icp_count": conditional_policy.fresh_icp_count,
                "retained_icp_count": conditional_policy.retained_icp_count,
                "min_new_icp_count": conditional_policy.fresh_icp_count,
                "required_total_icps": conditional_policy.total_icps,
                "require_unique_icps": True,
            }
    total = max(
        1,
        _safe_int(getattr(config, "lab_champion_eval_days", 10), default=10)
        * _safe_int(getattr(config, "lab_champion_icps_per_day", 2), default=2),
    )
    if total == 20:
        default_fresh = 10
    else:
        default_fresh = max(1, total // 2)
    default_retained = max(1, total - default_fresh)
    fresh = max(
        1,
        _safe_int(getattr(config, "lab_champion_fresh_icp_count", default_fresh), default=default_fresh),
    )
    retained = max(
        1,
        _safe_int(
            getattr(config, "lab_champion_retained_icp_count", default_retained),
            default=default_retained,
        ),
    )
    return {
        "window_mode": str(
            getattr(config, "lab_champion_window_mode", "hybrid_fresh_retained")
            or "hybrid_fresh_retained"
        ).strip().lower(),
        "fresh_icp_count": fresh,
        "retained_icp_count": retained,
        "min_new_icp_count": fresh,
    }


def _benchmark_summary_has_companies(item: Any) -> bool:
    if not isinstance(item, Mapping):
        return False
    try:
        return int(item.get("company_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def _private_holdout_gate_from_baseline_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    doc = row.get("score_summary_doc") if isinstance(row.get("score_summary_doc"), Mapping) else {}
    assignment = (
        doc.get("category_assignment")
        if isinstance(doc.get("category_assignment"), Mapping)
        else {}
    )
    assignment_items = (
        assignment.get("items")
        if isinstance(assignment.get("items"), list)
        else []
    )
    if assignment_items:
        categories: dict[str, list[Mapping[str, Any]]] = {
            "public": [],
            "private": [],
            "conditional": [],
        }
        for item in assignment_items:
            if not isinstance(item, Mapping):
                return None
            category = str(item.get("category") or "")
            if category not in categories:
                return None
            categories[category].append(item)
        if not all(categories.values()):
            return None
        preliminary_items = [*categories["public"], *categories["private"]]
        policy = assignment.get("policy") if isinstance(assignment.get("policy"), Mapping) else {}
        return {
            "schema_version": "1.1",
            "promotion_metric_version": PAIRED_LCB_PROMOTION_METRIC_VERSION,
            "gate_type": "public_private_then_conditional_validation",
            "baseline_benchmark_bundle_id": str(row.get("benchmark_bundle_id") or ""),
            "baseline_benchmark_hash": canonical_hash(doc) if doc else "",
            "baseline_aggregate_score": _safe_float(
                doc.get("aggregate_score"),
                default=_average(
                    [_safe_float(item.get("score"), default=0.0) for item in assignment_items]
                ),
            ),
            "baseline_preliminary_score": _average(
                [_safe_float(item.get("score"), default=0.0) for item in preliminary_items]
            ),
            "baseline_public_score": _average(
                [_safe_float(item.get("score"), default=0.0) for item in categories["public"]]
            ),
            "baseline_private_score": _average(
                [_safe_float(item.get("score"), default=0.0) for item in categories["private"]]
            ),
            "baseline_conditional_score": _average(
                [_safe_float(item.get("score"), default=0.0) for item in categories["conditional"]]
            ),
            "baseline_public_icp_count": len(categories["public"]),
            "baseline_private_holdout_icp_count": len(categories["private"]),
            "baseline_conditional_holdout_icp_count": len(categories["conditional"]),
            "rolling_window_hash": str(row.get("rolling_window_hash") or ""),
            "private_model_manifest_hash": str(row.get("private_model_manifest_hash") or ""),
            "public_icp_refs": [str(item.get("icp_ref") or "") for item in categories["public"]],
            "private_icp_refs": [str(item.get("icp_ref") or "") for item in categories["private"]],
            "conditional_icp_refs": [
                str(item.get("icp_ref") or "") for item in categories["conditional"]
            ],
            "conditional_validation_required": True,
            "category_assignment_hash": str(assignment.get("assignment_hash") or ""),
            "conditional_validation_policy_hash": str(policy.get("policy_hash") or ""),
            "threshold_points": _safe_float(policy.get("threshold_points"), default=1.0),
            "baseline_per_icp_scores": {
                str(item.get("icp_ref") or ""): _safe_float(item.get("score"), default=0.0)
                for item in assignment_items
                if str(item.get("icp_ref") or "").strip()
            },
        }
    split = doc.get("visibility_split") if isinstance(doc.get("visibility_split"), Mapping) else {}
    items = split.get("items") if isinstance(split.get("items"), list) else []
    public_items = [
        item for item in items
        if isinstance(item, Mapping) and str(item.get("visibility") or "") == "public"
    ]
    private_count = _safe_int(split.get("private_count"), default=0)
    if private_count <= 0:
        private_count = sum(
            1
            for item in items
            if isinstance(item, Mapping) and str(item.get("visibility") or "") == "private"
        )
    public_refs = [
        str(item.get("icp_ref") or "")
        for item in public_items
        if str(item.get("icp_ref") or "").strip()
    ]
    if not public_refs or private_count <= 0:
        return None
    public_scores = [_safe_float(item.get("score"), default=0.0) for item in public_items]
    private_scores = [
        _safe_float(item.get("score"), default=0.0)
        for item in items
        if isinstance(item, Mapping) and str(item.get("visibility") or "") == "private"
    ]
    all_scores = [
        _safe_float(item.get("score"), default=0.0)
        for item in items
        if isinstance(item, Mapping)
    ]
    baseline_aggregate_score = _safe_float(
        doc.get("aggregate_score"),
        default=_average(all_scores),
    )
    return {
        "schema_version": "1.0",
        "promotion_metric_version": PAIRED_LCB_PROMOTION_METRIC_VERSION,
        "gate_type": "public_score_before_private_holdout",
        "baseline_benchmark_bundle_id": str(row.get("benchmark_bundle_id") or ""),
        "baseline_benchmark_hash": canonical_hash(doc) if doc else "",
        "baseline_aggregate_score": baseline_aggregate_score,
        "baseline_public_score": _average(public_scores),
        "baseline_private_score": _average(private_scores),
        "baseline_public_icp_count": len(public_refs),
        "baseline_private_holdout_icp_count": private_count,
        "rolling_window_hash": str(row.get("rolling_window_hash") or ""),
        "private_model_manifest_hash": str(row.get("private_model_manifest_hash") or ""),
        "public_icp_refs": public_refs,
        # Per-ICP baseline scores let verifiers recompute the true advisory
        # delta (candidate-per-ICP minus stored-baseline-per-ICP) instead of
        # falling back to a not_applicable verdict (§0-N2).
        "baseline_per_icp_scores": {
            str(item.get("icp_ref") or ""): _safe_float(item.get("score"), default=0.0)
            for item in items
            if isinstance(item, Mapping) and str(item.get("icp_ref") or "").strip()
        },
    }


def _candidate_gate_event_doc(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "promotion_metric_version": str(
            value.get("promotion_metric_version") or ""
        ),
        "gate_type": str(value.get("gate_type") or ""),
        "decision": str(value.get("decision") or ""),
        "baseline_benchmark_bundle_id": str(value.get("baseline_benchmark_bundle_id") or ""),
        "baseline_benchmark_hash": str(value.get("baseline_benchmark_hash") or ""),
        "baseline_aggregate_score": _safe_float(value.get("baseline_aggregate_score"), default=0.0),
        "baseline_public_score": _safe_float(value.get("baseline_public_score"), default=0.0),
        "baseline_private_score": _safe_float(value.get("baseline_private_score"), default=0.0),
        "baseline_conditional_score": _safe_float(
            value.get("baseline_conditional_score"),
            default=0.0,
        ),
        "baseline_preliminary_score": _safe_float(
            value.get("baseline_preliminary_score"),
            default=0.0,
        ),
        "candidate_public_score": _safe_float(value.get("candidate_public_score"), default=0.0),
        "candidate_private_score": _safe_float(value.get("candidate_private_score"), default=0.0),
        "candidate_conditional_score": _safe_float(
            value.get("candidate_conditional_score"),
            default=0.0,
        ),
        "candidate_preliminary_score": _safe_float(
            value.get("candidate_preliminary_score"),
            default=0.0,
        ),
        "candidate_preliminary_delta": _safe_float(
            value.get("candidate_preliminary_delta"),
            default=0.0,
        ),
        "paired_base_public_score": _safe_float(value.get("paired_base_public_score"), default=0.0),
        "candidate_total_score": _safe_float(value.get("candidate_total_score"), default=0.0),
        "paired_base_total_score": _safe_float(value.get("paired_base_total_score"), default=0.0),
        "candidate_delta_vs_daily_baseline": _safe_float(
            value.get("candidate_delta_vs_daily_baseline"),
            default=0.0,
        ),
        "reference_evaluation_mode": str(value.get("reference_evaluation_mode") or ""),
        "public_icp_count": _safe_int(value.get("public_icp_count"), default=0),
        "private_holdout_icp_count": _safe_int(value.get("private_holdout_icp_count"), default=0),
        "conditional_holdout_icp_count": _safe_int(
            value.get("conditional_holdout_icp_count"),
            default=0,
        ),
        "private_holdout_evaluated": bool(value.get("private_holdout_evaluated")),
        "conditional_holdout_evaluated": bool(value.get("conditional_holdout_evaluated")),
        "conditional_validation_required": bool(value.get("conditional_validation_required")),
        "preliminary_decision": str(value.get("preliminary_decision") or ""),
        "final_decision": str(value.get("final_decision") or ""),
        "threshold_points": _safe_float(value.get("threshold_points"), default=0.0),
        "category_assignment_hash": str(value.get("category_assignment_hash") or ""),
        "conditional_validation_policy_hash": str(
            value.get("conditional_validation_policy_hash") or ""
        ),
    }


async def _persist_baseline_category_results(
    assignment: Mapping[str, Any],
    *,
    source_bundle_ref: str,
    rolling_window_hash: str,
    scoring_run_id: str = "",
) -> None:
    counts = (
        assignment.get("category_counts")
        if isinstance(assignment.get("category_counts"), Mapping)
        else {}
    )
    scores = (
        assignment.get("category_scores")
        if isinstance(assignment.get("category_scores"), Mapping)
        else {}
    )
    assignment_hash = str(assignment.get("assignment_hash") or "")
    policy_hash = str(assignment.get("policy_hash") or "")
    if not assignment_hash or not policy_hash:
        raise RuntimeError("conditional baseline category commitments are missing")
    for category in ("public", "private", "conditional"):
        await create_scoring_category_result(
            source_kind="baseline",
            source_bundle_ref=source_bundle_ref,
            category=category,
            assignment_hash=assignment_hash,
            policy_hash=policy_hash,
            rolling_window_hash=rolling_window_hash,
            icp_count=_safe_int(counts.get(category), default=0),
            aggregate_score=_safe_float(scores.get(category), default=0.0),
            scoring_run_id=scoring_run_id or None,
        )
    await create_scoring_category_result(
        source_kind="baseline",
        source_bundle_ref=source_bundle_ref,
        category="overall",
        assignment_hash=assignment_hash,
        policy_hash=policy_hash,
        rolling_window_hash=rolling_window_hash,
        icp_count=sum(_safe_int(counts.get(name), default=0) for name in ("public", "private", "conditional")),
        aggregate_score=_safe_float(assignment.get("aggregate_score"), default=0.0),
        scoring_run_id=scoring_run_id or None,
    )


async def _repair_baseline_category_results_from_row(
    row: Mapping[str, Any],
    *,
    expected_policy_hash: str,
) -> None:
    doc = (
        row.get("score_summary_doc")
        if isinstance(row.get("score_summary_doc"), Mapping)
        else {}
    )
    assignment = (
        doc.get("category_assignment")
        if isinstance(doc.get("category_assignment"), Mapping)
        else None
    )
    if assignment is None:
        raise RuntimeError("conditional baseline has no category assignment")
    policy_hash = str(assignment.get("policy_hash") or "")
    if policy_hash != str(expected_policy_hash or ""):
        raise RuntimeError(
            "conditional baseline policy mismatch:"
            f"stored={compact_ref(policy_hash)}:expected={compact_ref(expected_policy_hash)}"
        )
    source_bundle_ref = str(row.get("benchmark_bundle_id") or "")
    rolling_window_hash = str(row.get("rolling_window_hash") or "")
    if not source_bundle_ref or not rolling_window_hash:
        raise RuntimeError("conditional baseline row lacks immutable bundle/window identity")
    await _persist_baseline_category_results(
        assignment,
        source_bundle_ref=source_bundle_ref,
        rolling_window_hash=rolling_window_hash,
    )


async def _persist_conditional_finalization_events(
    gate: Mapping[str, Any],
    *,
    candidate_id: str,
    source_score_bundle_id: str,
    rolling_window_hash: str,
    queue_generation_id: str | None = None,
) -> None:
    if not bool(gate.get("conditional_validation_required")):
        return
    assignment_hash = str(gate.get("category_assignment_hash") or "")
    policy_hash = str(gate.get("conditional_validation_policy_hash") or "")
    baseline_bundle_id = str(gate.get("baseline_benchmark_bundle_id") or "")
    if not assignment_hash or not policy_hash or not baseline_bundle_id:
        raise RuntimeError("conditional finalization commitments are missing")
    source_ref = str(source_score_bundle_id)
    threshold = _safe_float(gate.get("threshold_points"), default=0.0)
    common = {
        "candidate_id": candidate_id,
        "assignment_hash": assignment_hash,
        "policy_hash": policy_hash,
        "rolling_window_hash": rolling_window_hash,
        "baseline_benchmark_bundle_id": baseline_bundle_id,
        "source_ref": source_ref,
        "threshold_points": threshold,
        "queue_generation_id": queue_generation_id,
        "source_score_bundle_id": source_score_bundle_id,
    }
    decision = str(gate.get("decision") or "")
    if decision == "rejected_before_private_holdout":
        return
    if decision == "rejected_before_conditional_validation":
        await create_conditional_validation_event(
            event_type="preliminary_gate_failed",
            decision_score=_safe_float(
                gate.get("candidate_preliminary_score"),
                default=0.0,
            ),
            event_doc={
                "preliminary_decision": decision,
                "public_icp_count": _safe_int(gate.get("public_icp_count"), default=0),
                "private_icp_count": _safe_int(
                    gate.get("private_holdout_icp_count"),
                    default=0,
                ),
            },
            **common,
        )
        return
    if decision not in {
        "conditional_validation_approved",
        "rejected_after_conditional_validation",
    } or not bool(gate.get("conditional_holdout_evaluated")):
        raise RuntimeError(
            "conditional score bundle lacks a terminal conditional decision:"
            f"{decision or '<empty>'}"
        )
    final_score = _safe_float(gate.get("candidate_total_score"), default=0.0)
    await create_conditional_validation_event(
        event_type="conditional_completed",
        decision_score=_safe_float(
            gate.get("candidate_conditional_score"),
            default=0.0,
        ),
        event_doc={
            "conditional_icp_count": _safe_int(
                gate.get("conditional_holdout_icp_count"),
                default=0,
            ),
            "final_decision": decision,
        },
        **common,
    )
    await create_conditional_validation_event(
        event_type=(
            "final_pass"
            if decision == "conditional_validation_approved"
            else "final_fail"
        ),
        decision_score=final_score,
        event_doc={
            "final_decision": decision,
            "candidate_delta_vs_daily_baseline": _safe_float(
                gate.get("candidate_delta_vs_daily_baseline"),
                default=0.0,
            ),
            "total_icp_count": (
                _safe_int(gate.get("public_icp_count"), default=0)
                + _safe_int(gate.get("private_holdout_icp_count"), default=0)
                + _safe_int(gate.get("conditional_holdout_icp_count"), default=0)
            ),
        },
        **common,
    )


async def _persist_candidate_category_results(
    gate: Mapping[str, Any],
    *,
    source_bundle_ref: str,
    rolling_window_hash: str,
    candidate_id: str,
    scoring_run_id: str = "",
) -> None:
    if not bool(gate.get("conditional_validation_required")):
        return
    assignment_hash = str(gate.get("category_assignment_hash") or "")
    policy_hash = str(gate.get("conditional_validation_policy_hash") or "")
    if not assignment_hash or not policy_hash:
        raise RuntimeError("conditional candidate category commitments are missing")
    categories: list[tuple[str, str, str, int]] = [
        (
            "public",
            "candidate_public_score",
            "baseline_public_score",
            _safe_int(gate.get("public_icp_count"), default=0),
        )
    ]
    if bool(gate.get("private_holdout_evaluated")):
        categories.extend(
            [
                (
                    "private",
                    "candidate_private_score",
                    "baseline_private_score",
                    _safe_int(gate.get("private_holdout_icp_count"), default=0),
                ),
                (
                    "preliminary",
                    "candidate_preliminary_score",
                    "baseline_preliminary_score",
                    _safe_int(gate.get("public_icp_count"), default=0)
                    + _safe_int(gate.get("private_holdout_icp_count"), default=0),
                ),
            ]
        )
    if bool(gate.get("conditional_holdout_evaluated")):
        categories.extend(
            [
                (
                    "conditional",
                    "candidate_conditional_score",
                    "baseline_conditional_score",
                    _safe_int(gate.get("conditional_holdout_icp_count"), default=0),
                ),
                (
                    "overall",
                    "candidate_total_score",
                    "baseline_aggregate_score",
                    _safe_int(gate.get("public_icp_count"), default=0)
                    + _safe_int(gate.get("private_holdout_icp_count"), default=0)
                    + _safe_int(gate.get("conditional_holdout_icp_count"), default=0),
                ),
            ]
        )
    for category, candidate_field, baseline_field, icp_count in categories:
        candidate_score = _safe_float(gate.get(candidate_field), default=0.0)
        baseline_score = _safe_float(gate.get(baseline_field), default=0.0)
        await create_scoring_category_result(
            source_kind="candidate",
            source_bundle_ref=source_bundle_ref,
            category=category,
            assignment_hash=assignment_hash,
            policy_hash=policy_hash,
            rolling_window_hash=rolling_window_hash,
            icp_count=icp_count,
            aggregate_score=candidate_score,
            scoring_run_id=scoring_run_id or None,
            candidate_id=candidate_id,
            delta_vs_baseline=candidate_score - baseline_score,
        )


def _compact_scoring_health_doc(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allowed = {
        "schema_version",
        "health_status",
        "icp_count",
        "reference_runtime_success_rate",
        "candidate_runtime_success_rate",
        "reference_zero_company_rate",
        "candidate_zero_company_rate",
        "provider_error_rate",
        "timeout_rate",
        "invalid_output_rate",
        "skipped_candidate_rate",
        "public_holdout_decision",
        "baseline_bundle_id",
        "baseline_bundle_hash",
    }
    return {key: value[key] for key in allowed if key in value}


def _failure_rate_from_success(value: Any) -> float:
    return max(0.0, min(1.0, 1.0 - _safe_float(value, default=1.0)))


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stored_daily_baseline_evaluation_policy(
    policy: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind score-bundle arithmetic to the frozen daily baseline per ICP."""

    effective = dict(policy)
    effective["reference_evaluation_mode"] = "stored_daily_baseline"
    raw_scores = gate.get("baseline_per_icp_scores")
    if isinstance(raw_scores, Mapping):
        baseline_scores: dict[str, float] = {}
        for key, value in raw_scores.items():
            ref = str(key or "").strip()
            if not ref:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                baseline_scores[ref] = score
        if baseline_scores:
            effective["baseline_per_icp_scores"] = baseline_scores
    excluded = gate.get("provider_excluded_icp_ids")
    if (
        isinstance(excluded, Sequence)
        and not isinstance(excluded, (str, bytes, bytearray))
    ):
        effective["provider_excluded_icp_ids"] = sorted(
            {str(item).strip() for item in excluded if str(item).strip()}
        )
    return effective


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_candidate_claim_race_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_candidate_claim_conflict" in message
        or "research_lab_candidate_eval_events_candidate_seq_key" in message
        or "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
    )


def _is_event_sequence_race_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "duplicate key" in message
        or "unique constraint" in message
        or "23505" in message
    )
