"""Measured V2 adapter around the unchanged code-edit autoresearch loop.

The executor owns every calculation, prompt, validation, and candidate
decision made by ``CodeEditLoopEngine``.  Operations that inherently require
the parent (database append, S3 compatibility artifact, Docker/ECR build, or
maintenance state) cross the signed host-operation channel and are committed
to the final execution receipt.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import fields
import json
import math
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from gateway.research_lab.code_build import (
    CodeEditBuildResult,
    CodeEditPatchApplyError,
    CodeEditCandidateBuilder,
    ParentImageSourceContext,
    _copy_source_tree,
    _editable_runtime_files,
    _initialize_temporary_git_repo,
    _run_git_apply,
    _source_file_previews,
    _top_level_paths,
    _prepare_parent_image_workspace,
    _run,
    resolve_source_inspection_requests,
)
from gateway.research_lab.code_loop_engine import (
    BuiltCodeEditCandidate,
    CodeEditLoopEngine,
    CodeEditLoopResult,
)
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.git_tree_models import (
    GitTreeContractError,
    TreeChildSlot,
    TreeNode,
    TreePolicy,
    derive_tree_id,
    generation_operation_id,
)
from gateway.research_lab.provider_capabilities import (
    load_effective_provider_capabilities_sync,
)
from gateway.research_lab.provider_evidence_proxy import (
    ProviderRegistryEntry,
    seed_provider_registry,
    validate_provider_registry_entries,
)
from gateway.research_lab.source_add_catalog import (
    probe_endpoints_from_provisioned_rows,
)
from gateway.research_lab.key_vault import (
    preflight_openrouter_key,
    strict_openrouter_provider_policy,
    verify_openrouter_workspace_privacy,
)
from gateway.research_lab.autoresearch_runtime import (
    AutoResearchLoopEvent,
    AutoResearchRuntimeSettings,
    OpenRouterCallResult,
)
from gateway.research_lab.source_symbol_index import build_source_symbol_index
from gateway.research_lab.store import canonical_hash
from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionResultV2,
)
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.provider_evidence_v2 import (
    REQUEST_SCHEMA_VERSION as PROVIDER_EVIDENCE_REQUEST_SCHEMA_VERSION,
    validate_signed_provider_evidence_record,
)
from gateway.tee.provider_outcome_v2 import (
    validate_provider_outcome_snapshot_v2,
)
from gateway.tee.source_bundle_v2 import extract_source_bundle_v2
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_runtime_catalog_v2,
    source_add_route_for_provider_v2,
    source_add_runtime_retry_hashes_v2,
    validate_source_add_runtime_catalog_v2,
)
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_bytes,
    sha256_json,
    validate_receipt_graph,
)
from research_lab.code_editing import (
    CodeEditDraft,
    CodeEditSourceInspectionRequest,
    build_code_edit_repair_messages,
    parse_code_edit_repair_response,
)
from research_lab.eval import (
    CandidatePatchManifest,
    PrivateModelArtifactManifest,
    validate_candidate_patch_manifest,
    validate_private_model_artifact_manifest,
)
from research_lab.eval.private_runtime import compute_private_source_tree_hash
from research_lab.probe_catalog import (
    ProviderProbeEndpoint,
    default_probe_catalog,
    validate_probe_catalog,
)


AUTORESEARCH_REQUEST_SCHEMA_VERSION = "leadpoet.autoresearch_request.v2"
AUTORESEARCH_RESULT_SCHEMA_VERSION = "leadpoet.autoresearch_result.v2"
OPENROUTER_GUARD_REQUEST_SCHEMA_VERSION = "leadpoet.openrouter_guard_request.v2"
OPENROUTER_GUARD_RESULT_SCHEMA_VERSION = "leadpoet.openrouter_guard_result.v2"
STALE_PARENT_REPAIR_REQUEST_SCHEMA_VERSION = (
    "leadpoet.stale_parent_repair_request.v2"
)
STALE_PARENT_REPAIR_RESULT_SCHEMA_VERSION = (
    "leadpoet.stale_parent_repair_result.v2"
)
HOST_EVENT_RESULT_SCHEMA_VERSION = "leadpoet.autoresearch_event_result.v2"
HOST_BUILD_RESULT_SCHEMA_VERSION = "leadpoet.autoresearch_build_result.v2"
HOST_ARTIFACT_RESULT_SCHEMA_VERSION = "leadpoet.autoresearch_artifact_result.v2"
HOST_PAUSE_RESULT_SCHEMA_VERSION = "leadpoet.autoresearch_pause_result.v2"
HOST_DEV_EVAL_RESULT_SCHEMA_VERSION = "leadpoet.autoresearch_dev_eval_result.v2"
HOST_DEV_EVAL_COHORT_RESULT_SCHEMA_VERSION = (
    "leadpoet.autoresearch_dev_eval_cohort_result.v3"
)
HOST_GIT_TREE_RESULT_SCHEMA_VERSION = "leadpoet.autoresearch_git_tree_result.v2"

OP_RUN_CODE_EDIT_LOOP = "run_code_edit_loop"
OP_VERIFY_OPENROUTER_GUARD = "verify_openrouter_guard"
OP_REPAIR_STALE_PARENT = "repair_stale_parent"

AUTORESEARCH_OPERATIONS_V2 = {
    OP_RUN_CODE_EDIT_LOOP: frozenset({"research_lab.candidate_decision.v2"}),
    OP_VERIFY_OPENROUTER_GUARD: frozenset(
        {"research_lab.openrouter_guard.v2"}
    ),
    OP_REPAIR_STALE_PARENT: frozenset(
        {"research_lab.stale_parent_repair.v2"}
    ),
}

HOST_APPEND_EVENT = "autoresearch_append_event"
HOST_BUILD_CANDIDATE = "autoresearch_build_candidate"
HOST_READ_ARTIFACT = "autoresearch_read_artifact"
HOST_WRITE_ARTIFACT = "autoresearch_write_artifact"
HOST_CHECK_PAUSE = "autoresearch_check_pause"
HOST_DEV_EVALUATE = "autoresearch_dev_evaluate"
HOST_RECORD_PRIVACY = "autoresearch_record_privacy_proof"
HOST_GIT_TREE = "autoresearch_git_tree"

AUTORESEARCH_HOST_OPERATIONS_V2 = frozenset(
    {
        HOST_APPEND_EVENT,
        HOST_BUILD_CANDIDATE,
        HOST_READ_ARTIFACT,
        HOST_WRITE_ARTIFACT,
        HOST_CHECK_PAUSE,
        HOST_DEV_EVALUATE,
        HOST_RECORD_PRIVACY,
        HOST_GIT_TREE,
    }
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_OPENROUTER_KEY_REF_RE = re.compile(r"^encrypted_ref:openrouter:[0-9a-f]{32}$")
_RUNTIME_KEY_PLACEHOLDER = "sk-or-v1-" + ("r" * 32)
_MANAGEMENT_KEY_PLACEHOLDER = "sk-or-v1-" + ("m" * 32)

_REQUEST_FIELDS = {
    "schema_version",
    "run_id",
    "ticket",
    "artifact",
    "component_registry",
    "component_registry_evidence",
    "active_model_evidence",
    "provider_catalog_evidence",
    "provider_outcome_evidence",
    "benchmark_public_summary",
    "model_id",
    "model_doc",
    "budget_context",
    "requested_loop_count",
    "resume_state",
    "loop_settings",
    "source_bundle",
    "probe_private_window_term_hashes",
    "provider_outcome_digest",
    "dev_evaluator_enabled",
    "openrouter_context",
    "expected_event_state_hash",
}

_EVENT_PURPOSES = {
    "source_inspection_requested": "research_lab.source_inspection.v2",
    "source_inspection_resolved": "research_lab.source_inspection.v2",
    "source_inspection_failed": "research_lab.source_inspection.v2",
    "loop_direction_planned": "research_lab.research_plan.v2",
    "allocator_decision": "research_lab.research_plan.v2",
    "plan_alignment_judged": "research_lab.research_plan.v2",
    "code_edit_drafted": "research_lab.patch_draft.v2",
    "candidate_generation_fallback_drafted": "research_lab.patch_draft.v2",
    "code_edit_repair_drafted": "research_lab.patch_draft.v2",
    "code_edit_validation_passed": "research_lab.patch_validation.v2",
    "code_edit_validation_failed": "research_lab.patch_validation.v2",
    "code_edit_alignment_rejected": "research_lab.patch_validation.v2",
    "code_edit_repair_requested": "research_lab.patch_validation.v2",
    "code_edit_repair_failed": "research_lab.patch_validation.v2",
    "candidate_patch_parse_failed": "research_lab.patch_validation.v2",
    "candidate_repair_exhausted": "research_lab.patch_validation.v2",
    "candidate_build_started": "research_lab.candidate_build.v2",
    "candidate_build_passed": "research_lab.candidate_build.v2",
    "candidate_build_failed": "research_lab.candidate_build.v2",
    "checkpoint_saved": "research_lab.checkpoint.v2",
    "candidate_selected": "research_lab.candidate_decision.v2",
    "loop_completed": "research_lab.candidate_decision.v2",
    "loop_failed": "research_lab.candidate_decision.v2",
    "no_viable_patch": "research_lab.candidate_decision.v2",
}


class AutoresearchExecutorV2Error(ValueError):
    """A V2 loop request or host response is incomplete or unverifiable."""


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise AutoresearchExecutorV2Error("%s is invalid" % field)
    return normalized


def _mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutoresearchExecutorV2Error("%s must be an object" % field)
    return dict(value)


def _event_document(event: AutoResearchLoopEvent) -> Dict[str, Any]:
    return {
        "event_type": str(event.event_type),
        "loop_status": str(event.loop_status),
        "elapsed_seconds": float(event.elapsed_seconds),
        "node_id": event.node_id,
        "candidate_artifact_hash": event.candidate_artifact_hash,
        "candidate_patch_hash": event.candidate_patch_hash,
        "provider_usage": [dict(item) for item in event.provider_usage],
        "cost_ledger": dict(event.cost_ledger),
        "event_doc": dict(event.event_doc),
    }


def _draft_from_mapping(value: Mapping[str, Any]) -> CodeEditDraft:
    allowed = {item.name for item in fields(CodeEditDraft)}
    unknown = set(value) - allowed - {"unified_diff_hash"}
    if unknown or not allowed.issubset(set(value) | {"predicted_delta", "plan_path_id", "plan_alignment", "expected_metric_effect"}):
        raise AutoresearchExecutorV2Error("code-edit draft fields are invalid")
    kwargs = {name: value[name] for name in allowed if name in value}
    kwargs["target_files"] = tuple(str(item) for item in value.get("target_files", ()))
    for name in ("plan_alignment", "expected_metric_effect"):
        if name in kwargs:
            kwargs[name] = _mapping(kwargs[name], name)
    return CodeEditDraft(**kwargs)


def _build_result_document(value: CodeEditBuildResult) -> Dict[str, Any]:
    return {
        "candidate_model_manifest": value.candidate_model_manifest.to_dict(),
        "code_edit_manifest": dict(value.code_edit_manifest),
        "source_diff_hash": str(value.source_diff_hash),
        "build_doc": dict(value.build_doc),
    }


def _validate_build_result(
    value: Mapping[str, Any],
    *,
    draft: CodeEditDraft,
    parent_artifact: PrivateModelArtifactManifest,
    expected_candidate_artifact_hash: str,
) -> CodeEditBuildResult:
    if set(value) != {
        "candidate_model_manifest",
        "code_edit_manifest",
        "source_diff_hash",
        "build_doc",
    }:
        raise AutoresearchExecutorV2Error("candidate build result fields are invalid")
    candidate = PrivateModelArtifactManifest.from_mapping(
        _mapping(value["candidate_model_manifest"], "candidate_model_manifest")
    )
    errors = validate_private_model_artifact_manifest(candidate)
    if errors:
        raise AutoresearchExecutorV2Error(
            "candidate model manifest is invalid: " + "; ".join(errors)
        )
    source_diff_hash = _hash(value["source_diff_hash"], "source_diff_hash")
    if source_diff_hash != sha256_json({"unified_diff": draft.unified_diff}):
        raise AutoresearchExecutorV2Error("candidate source diff commitment differs")
    patch = CandidatePatchManifest.from_mapping(
        _mapping(value["code_edit_manifest"], "code_edit_manifest")
    )
    patch_errors = validate_candidate_patch_manifest(
        patch,
        expected_parent_artifact_hash=parent_artifact.model_artifact_hash,
    )
    if patch_errors:
        raise AutoresearchExecutorV2Error(
            "candidate patch manifest is invalid: " + "; ".join(patch_errors)
        )
    if patch.candidate_artifact_hash != candidate.model_artifact_hash:
        raise AutoresearchExecutorV2Error("candidate manifest ancestry differs")
    if candidate.model_artifact_hash != expected_candidate_artifact_hash:
        raise AutoresearchExecutorV2Error(
            "candidate model source differs from the measured patch result"
        )
    build_doc = _mapping(value["build_doc"], "build_doc")
    declared_build_hash = _hash(build_doc.get("build_doc_hash"), "build_doc_hash")
    if declared_build_hash != sha256_json(
        {key: item for key, item in build_doc.items() if key != "build_doc_hash"}
    ):
        raise AutoresearchExecutorV2Error("candidate build document hash differs")
    return CodeEditBuildResult(
        candidate_model_manifest=candidate,
        code_edit_manifest=dict(value["code_edit_manifest"]),
        source_diff_hash=source_diff_hash,
        build_doc=build_doc,
    )


def _candidate_document(candidate: BuiltCodeEditCandidate) -> Dict[str, Any]:
    return {
        "draft": candidate.draft.to_dict(),
        "build": _build_result_document(candidate.build),
        "node_id": candidate.node_id,
        "iteration": candidate.iteration,
        "rehydration_artifact_uri": candidate.rehydration_artifact_uri,
        "rehydration_artifact_hash": candidate.rehydration_artifact_hash,
        "dev_score": candidate.dev_score,
        "dev_score_version": candidate.dev_score_version,
        "dev_evaluation": dict(candidate.dev_evaluation),
        "dev_feedback": dict(candidate.dev_feedback),
        "dev_feedback_hash": candidate.dev_feedback_hash,
        "tree_id": candidate.tree_id,
        "tree_parent_node_id": candidate.tree_parent_node_id,
        "tree_root_branch_id": candidate.tree_root_branch_id,
        "tree_depth": candidate.tree_depth,
        "tree_child_slot": candidate.tree_child_slot,
        "tree_branch_objective_path_id": (
            candidate.tree_branch_objective_path_id
        ),
        "tree_branch_objective_hash": candidate.tree_branch_objective_hash,
        "tree_generation_attempt_count": (
            candidate.tree_generation_attempt_count
        ),
        "tree_git_commit": candidate.tree_git_commit,
        "tree_root_artifact_hash": candidate.tree_root_artifact_hash,
        "tree_parent_artifact_hash": candidate.tree_parent_artifact_hash,
        "tree_parent_dev_score": candidate.tree_parent_dev_score,
        "tree_parent_feedback_hash": candidate.tree_parent_feedback_hash,
        "tree_incremental_source_diff_hash": candidate.tree_incremental_source_diff_hash,
        "tree_cumulative_source_diff_hash": candidate.tree_cumulative_source_diff_hash,
        "tree_composition": dict(candidate.tree_composition),
        "tree_settled_cost_microusd": candidate.tree_settled_cost_microusd,
    }


def _result_document(result: CodeEditLoopResult) -> Dict[str, Any]:
    return {
        "schema_version": AUTORESEARCH_RESULT_SCHEMA_VERSION,
        "selected_candidates": [
            _candidate_document(candidate) for candidate in result.selected_candidates
        ],
        "iterations_completed": result.iterations_completed,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "estimated_cost_usd": result.estimated_cost_usd,
        "actual_openrouter_cost_usd": result.actual_openrouter_cost_usd,
        "actual_openrouter_cost_microusd": result.actual_openrouter_cost_microusd,
        "openrouter_call_count": result.openrouter_call_count,
        "tree_result": result.tree_result.to_dict(),
        "provider_usage": [dict(item) for item in result.provider_usage],
        "status": result.status,
        "checkpoint_doc": (
            dict(result.checkpoint_doc) if result.checkpoint_doc is not None else None
        ),
    }


class _AttestedRawTraceRecorder:
    """Compatibility pointer backed by coordinator request/response artifacts."""

    def capture(
        self,
        *,
        run_id: str,
        stage: str,
        request_doc: Mapping[str, Any],
        response_doc: Any,
        outcome: str,
    ) -> Dict[str, str]:
        trace_hash = sha256_json(
            {
                "run_id": str(run_id),
                "stage": str(stage),
                "request_doc": dict(request_doc),
                "response_doc": response_doc,
                "outcome": str(outcome),
            }
        )
        return {
            "artifact_ref": "attested-v2:%s" % trace_hash.split(":", 1)[1],
            "sha256": trace_hash,
        }


class _HostArtifactIO:
    def __init__(self, context: ExecutionContextV2) -> None:
        self._context = context

    async def read_json(self, *, uri: str, content_hash: str) -> Mapping[str, Any]:
        expected = _hash(content_hash, "artifact content hash")
        response = await asyncio.to_thread(
            self._context.execute_host_operation,
            operation=HOST_READ_ARTIFACT,
            payload={"uri": str(uri), "content_hash": expected},
            expected_state_hash=expected,
            timeout_seconds=120,
            response_validator=lambda value: self._validate_read(
                value,
                uri=str(uri),
                content_hash=expected,
            ),
        )
        return dict(response["document"])

    async def write_json(
        self,
        *,
        uri: str,
        document: Mapping[str, Any],
        content_hash: str,
        artifact_kind: str,
    ) -> Mapping[str, Any]:
        expected = _hash(content_hash, "artifact content hash")
        normalized_kind = str(artifact_kind or "")
        hash_field = {
            "code_edit_failure_diagnostic": "diagnostic_hash",
            "loop_candidate_rehydration": "loop_candidate_artifact_hash",
        }.get(normalized_kind)
        if not hash_field:
            raise AutoresearchExecutorV2Error("artifact write kind is invalid")
        normalized_document = dict(document)
        if (
            normalized_document.get(hash_field) != expected
            or sha256_json(
                {
                    key: value
                    for key, value in normalized_document.items()
                    if key != hash_field
                }
            )
            != expected
        ):
            raise AutoresearchExecutorV2Error("artifact write commitment differs")
        return await asyncio.to_thread(
            self._context.execute_host_operation,
            operation=HOST_WRITE_ARTIFACT,
            payload={
                "uri": str(uri),
                "content_hash": expected,
                "artifact_kind": normalized_kind,
                "document": normalized_document,
            },
            expected_state_hash=expected,
            timeout_seconds=120,
            response_validator=lambda value: self._validate_write(
                value,
                uri=str(uri),
                content_hash=expected,
            ),
        )

    @staticmethod
    def _validate_read(
        value: Mapping[str, Any], *, uri: str, content_hash: str
    ) -> Dict[str, Any]:
        if set(value) != {"schema_version", "uri", "content_hash", "document"}:
            raise AutoresearchExecutorV2Error("artifact read response fields are invalid")
        if value.get("schema_version") != HOST_ARTIFACT_RESULT_SCHEMA_VERSION:
            raise AutoresearchExecutorV2Error("artifact read response schema is invalid")
        document = _mapping(value.get("document"), "artifact document")
        if value.get("uri") != uri or value.get("content_hash") != content_hash:
            raise AutoresearchExecutorV2Error("artifact read response binding differs")
        embedded_hash = str(
            document.get("loop_candidate_artifact_hash")
            or document.get("diagnostic_hash")
            or ""
        )
        hash_fields = {
            "loop_candidate_artifact_hash",
            "diagnostic_hash",
        }.intersection(document)
        if (
            len(hash_fields) != 1
            or embedded_hash != content_hash
            or sha256_json(
                {
                    key: item
                    for key, item in document.items()
                    if key not in hash_fields
                }
            )
            != content_hash
        ):
            raise AutoresearchExecutorV2Error("artifact read response hash differs")
        return {**dict(value), "document": document}

    @staticmethod
    def _validate_write(
        value: Mapping[str, Any], *, uri: str, content_hash: str
    ) -> Dict[str, Any]:
        if set(value) != {"schema_version", "uri", "content_hash", "persisted"}:
            raise AutoresearchExecutorV2Error("artifact write response fields are invalid")
        if (
            value.get("schema_version") != HOST_ARTIFACT_RESULT_SCHEMA_VERSION
            or value.get("uri") != uri
            or value.get("content_hash") != content_hash
            or value.get("persisted") is not True
        ):
            raise AutoresearchExecutorV2Error("artifact write response binding differs")
        return dict(value)


class _HostGitTreeRepository:
    """Signed proxy for the host's private SHA-256 Git object database."""

    def __init__(self, context: ExecutionContextV2, *, tree_id: str) -> None:
        self._context = context
        self._tree_id = _hash(tree_id, "Git-tree id")

    def initialize(
        self,
        *,
        source_root: Path,
        root_artifact_hash: str,
        policy_hash: str,
        run_id: str = "",
        root_manifest_hash: str = "",
        root_image_digest: str = "",
        evaluator_commitment_hash: str = "",
        tree_doc: Mapping[str, Any] | None = None,
    ) -> str:
        root_source_tree_hash = compute_private_source_tree_hash(Path(source_root))
        payload = {
            "action": "initialize",
            "tree_id": self._tree_id,
            "root_artifact_hash": _hash(
                root_artifact_hash, "Git-tree root artifact hash"
            ),
            "policy_hash": _hash(policy_hash, "Git-tree policy hash"),
            "root_source_tree_hash": _hash(
                root_source_tree_hash, "Git-tree root source hash"
            ),
            "run_id": str(run_id),
            "root_manifest_hash": _hash(
                root_manifest_hash, "Git-tree root manifest hash"
            ),
            "root_image_digest": _hash(
                root_image_digest, "Git-tree root image digest"
            ),
            "evaluator_commitment_hash": _hash(
                evaluator_commitment_hash,
                "Git-tree evaluator commitment hash",
            ),
            "tree_doc": dict(tree_doc or {}),
        }
        state_hash = sha256_json(payload)
        response = self._context.execute_host_operation(
            operation=HOST_GIT_TREE,
            payload=payload,
            expected_state_hash=state_hash,
            timeout_seconds=1800,
            response_validator=lambda value: self._validate_response(
                value, action="initialize", state_hash=state_hash
            ),
        )
        result = _mapping(response.get("result"), "Git-tree initialize result")
        if set(result) != {
            "tree_id",
            "root_git_commit",
            "root_source_tree_hash",
        }:
            raise AutoresearchExecutorV2Error(
                "Git-tree initialize result fields are invalid"
            )
        if (
            result.get("tree_id") != self._tree_id
            or result.get("root_source_tree_hash") != root_source_tree_hash
            or not _HEX_HASH_RE.fullmatch(str(result.get("root_git_commit") or ""))
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree initialize result binding differs"
            )
        return str(result["root_git_commit"])

    def plan_slot(
        self,
        *,
        slot: TreeChildSlot,
        request_hash: str,
        operation_id: str,
        node_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if operation_id != generation_operation_id(slot):
            raise AutoresearchExecutorV2Error(
                "Git-tree generation operation identity differs"
            )
        return self._execute_control(
            {
                "action": "plan_slot",
                "tree_id": self._tree_id,
                "slot": slot.to_dict(),
                "request_hash": _hash(
                    request_hash, "Git-tree generation request"
                ),
                "operation_id": _hash(
                    operation_id, "Git-tree generation operation"
                ),
                "node_doc": dict(node_doc),
            },
            action="plan_slot",
        )

    def settle_operation(
        self,
        *,
        operation_id: str,
        operation_status: str,
        request_hash: str,
        result_hash: str,
        settled_cost_microusd: int,
        provider_call_count: int,
        settlement_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._execute_control(
            {
                "action": "settle_operation",
                "tree_id": self._tree_id,
                "operation_id": _hash(operation_id, "Git-tree operation"),
                "operation_status": str(operation_status),
                "request_hash": _hash(
                    request_hash, "Git-tree operation request"
                ),
                "result_hash": _hash(
                    result_hash, "Git-tree operation result"
                ),
                "settled_cost_microusd": max(
                    0, int(settled_cost_microusd)
                ),
                "provider_call_count": max(0, int(provider_call_count)),
                "settlement_doc": dict(settlement_doc),
            },
            action="settle_operation",
        )

    def reserve_operation(
        self,
        *,
        operation_id: str,
        operation_kind: str,
        request_hash: str,
        node_id: str = "",
        reservation_doc: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._execute_control(
            {
                "action": "reserve_operation",
                "tree_id": self._tree_id,
                "operation_id": _hash(operation_id, "Git-tree operation"),
                "operation_kind": str(operation_kind),
                "request_hash": _hash(
                    request_hash, "Git-tree operation request"
                ),
                "node_id": str(node_id or ""),
                "reservation_doc": dict(reservation_doc or {}),
            },
            action="reserve_operation",
        )

    def inspect_operation(self, *, operation_id: str) -> Mapping[str, Any]:
        return self._execute_control(
            {
                "action": "inspect_operation",
                "tree_id": self._tree_id,
                "operation_id": _hash(operation_id, "Git-tree operation"),
            },
            action="inspect_operation",
        )

    def operation_settlement_commitment(self) -> Mapping[str, Any]:
        result = self._execute_control(
            {
                "action": "operation_settlement_commitment",
                "tree_id": self._tree_id,
            },
            action="operation_settlement_commitment",
        )
        if set(result) != {
            "tree_id",
            "action",
            "operation_count",
            "settled_cost_microusd",
            "provider_call_count",
            "operation_settlement_hash",
        }:
            raise AutoresearchExecutorV2Error(
                "Git-tree operation commitment fields are invalid"
            )
        _hash(
            result.get("operation_settlement_hash"),
            "Git-tree operation settlement hash",
        )
        for field in (
            "operation_count",
            "settled_cost_microusd",
            "provider_call_count",
        ):
            value = result.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AutoresearchExecutorV2Error(
                    "Git-tree operation commitment counts are invalid"
                )
        return dict(result)

    def publish_bundle(self) -> Mapping[str, Any]:
        result = self._execute_control(
            {
                "action": "publish_bundle",
                "tree_id": self._tree_id,
            },
            action="publish_bundle",
        )
        uri = str(result.get("bundle_uri") or "")
        if (
            not uri.startswith("s3://")
            or not result.get("readback_verified")
            or int(result.get("bundle_size_bytes") or 0) <= 0
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree bundle result is invalid"
            )
        _hash(result.get("bundle_hash"), "Git-tree bundle hash")
        return dict(result)

    def record_node(self, *, node_doc: Mapping[str, Any]) -> Mapping[str, Any]:
        node = TreeNode.from_mapping(node_doc)
        if node.tree_id != self._tree_id:
            raise AutoresearchExecutorV2Error(
                "Git-tree node belongs to another tree"
            )
        return self._execute_control(
            {
                "action": "record_node",
                "tree_id": self._tree_id,
                "node": node.to_dict(),
            },
            action="record_node",
        )

    def commit_checkpoint(
        self, *, checkpoint_hash: str, checkpoint_doc: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if sha256_json(dict(checkpoint_doc)) != checkpoint_hash:
            raise AutoresearchExecutorV2Error("Git-tree checkpoint hash differs")
        return self._execute_control(
            {
                "action": "commit_checkpoint",
                "tree_id": self._tree_id,
                "checkpoint_hash": checkpoint_hash,
                "checkpoint": dict(checkpoint_doc),
            },
            action="commit_checkpoint",
        )

    def select_final(
        self, *, selection_hash: str, selection_doc: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if sha256_json(dict(selection_doc)) != selection_hash:
            raise AutoresearchExecutorV2Error(
                "Git-tree final selection hash differs"
            )
        return self._execute_control(
            {
                "action": "select_final",
                "tree_id": self._tree_id,
                "selection_hash": selection_hash,
                "selection": dict(selection_doc),
            },
            action="select_final",
        )

    def fail_tree(
        self, *, failure_hash: str, failure_doc: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if sha256_json(dict(failure_doc)) != failure_hash:
            raise AutoresearchExecutorV2Error("Git-tree failure hash differs")
        return self._execute_control(
            {
                "action": "fail_tree",
                "tree_id": self._tree_id,
                "failure_hash": failure_hash,
                "failure": dict(failure_doc),
            },
            action="fail_tree",
        )

    def _execute_control(
        self, payload: Mapping[str, Any], *, action: str
    ) -> Mapping[str, Any]:
        state_hash = sha256_json(dict(payload))
        response = self._context.execute_host_operation(
            operation=HOST_GIT_TREE,
            payload=dict(payload),
            expected_state_hash=state_hash,
            timeout_seconds=180,
            response_validator=lambda value: self._validate_response(
                value, action=action, state_hash=state_hash
            ),
        )
        result = _mapping(response.get("result"), f"Git-tree {action} result")
        if result.get("tree_id") != self._tree_id or result.get("action") != action:
            raise AutoresearchExecutorV2Error(
                f"Git-tree {action} result binding differs"
            )
        return dict(result)

    def commit_child(
        self,
        *,
        slot: TreeChildSlot,
        draft: CodeEditDraft,
        expected_parent_source_tree_hash: str,
    ) -> Dict[str, Any]:
        if slot.tree_id != self._tree_id:
            raise AutoresearchExecutorV2Error(
                "Git-tree child slot belongs to another tree"
            )
        payload = {
            "action": "commit_child",
            "tree_id": self._tree_id,
            "slot": slot.to_dict(),
            "draft": draft.to_dict(),
            "expected_parent_source_tree_hash": _hash(
                expected_parent_source_tree_hash,
                "Git-tree parent source hash",
            ),
        }
        state_hash = sha256_json(payload)
        response = self._context.execute_host_operation(
            operation=HOST_GIT_TREE,
            payload=payload,
            expected_state_hash=state_hash,
            timeout_seconds=300,
            response_validator=lambda value: self._validate_response(
                value, action="commit_child", state_hash=state_hash
            ),
        )
        result = _mapping(response.get("result"), "Git-tree commit result")
        required = {
            "schema_version",
            "tree_id",
            "node_id",
            "parent_node_id",
            "root_branch_id",
            "depth",
            "slot_index",
            "git_commit",
            "parent_git_commit",
            "source_tree_hash",
            "draft_patch_hash",
            "incremental_patch_hash",
            "cumulative_patch_hash",
            "changed_files",
            "incremental_patch",
            "cumulative_patch",
        }
        if set(result) != required:
            raise AutoresearchExecutorV2Error(
                "Git-tree commit result fields are invalid"
            )
        if any(
            (
                result.get("tree_id") != slot.tree_id,
                result.get("node_id") != slot.node_id,
                result.get("parent_node_id") != slot.parent_node_id,
                result.get("root_branch_id") != slot.root_branch_id,
                int(result.get("depth") or 0) != slot.depth,
                int(result.get("slot_index") or -1) != slot.slot_index,
                result.get("draft_patch_hash")
                != sha256_json({"unified_diff": draft.unified_diff}),
            )
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree commit result topology differs"
            )
        incremental_patch = str(result.get("incremental_patch") or "")
        cumulative_patch = str(result.get("cumulative_patch") or "")
        if (
            not _HEX_HASH_RE.fullmatch(str(result.get("git_commit") or ""))
            or not _HEX_HASH_RE.fullmatch(
                str(result.get("parent_git_commit") or "")
            )
            or not incremental_patch.startswith("diff --git ")
            or not cumulative_patch.startswith("diff --git ")
            or result.get("incremental_patch_hash")
            != sha256_json({"unified_diff": incremental_patch})
            or result.get("cumulative_patch_hash")
            != sha256_json({"unified_diff": cumulative_patch})
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree commit result commitment differs"
            )
        _hash(result.get("source_tree_hash"), "Git-tree child source hash")
        changed_files = result.get("changed_files")
        if (
            not isinstance(changed_files, list)
            or not changed_files
            or any(not isinstance(item, str) or not item for item in changed_files)
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree commit changed-file set is invalid"
            )
        return dict(result)

    def verify_node_identity(
        self,
        *,
        node_id: str,
        git_commit: str,
        parent_node_id: str,
    ) -> None:
        payload = {
            "action": "verify_node_identity",
            "tree_id": self._tree_id,
            "node_id": str(node_id),
            "git_commit": str(git_commit),
            "parent_node_id": str(parent_node_id),
        }
        state_hash = sha256_json(payload)
        response = self._context.execute_host_operation(
            operation=HOST_GIT_TREE,
            payload=payload,
            expected_state_hash=state_hash,
            timeout_seconds=120,
            response_validator=lambda value: self._validate_response(
                value, action="verify_node_identity", state_hash=state_hash
            ),
        )
        result = _mapping(response.get("result"), "Git-tree verify result")
        if result != {
            "tree_id": self._tree_id,
            "node_id": str(node_id),
            "git_commit": str(git_commit),
            "parent_node_id": str(parent_node_id),
            "verified": True,
        }:
            raise AutoresearchExecutorV2Error(
                "Git-tree verification result binding differs"
            )

    @staticmethod
    def _validate_response(
        value: Mapping[str, Any], *, action: str, state_hash: str
    ) -> Dict[str, Any]:
        if set(value) != {"schema_version", "action", "state_hash", "result"}:
            raise AutoresearchExecutorV2Error(
                "Git-tree host response fields are invalid"
            )
        if (
            value.get("schema_version") != HOST_GIT_TREE_RESULT_SCHEMA_VERSION
            or value.get("action") != action
            or value.get("state_hash") != state_hash
            or not isinstance(value.get("result"), Mapping)
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree host response binding differs"
            )
        return dict(value)


class _HostCandidateBuilder:
    def __init__(
        self,
        *,
        config: ResearchLabGatewayConfig,
        source_context: ParentImageSourceContext,
        source_bundle_hash: str,
        execution_context: ExecutionContextV2,
    ) -> None:
        self.config = config
        self._local = CodeEditCandidateBuilder(config)
        self._source_context = source_context
        self._source_contexts = {
            source_context.source_tree_hash: source_context,
        }
        self._source_context_lock = threading.RLock()
        self._derived_source_root = source_context.source_root.parent / "git-tree-candidates"
        self._derived_source_root.mkdir(parents=True, exist_ok=True)
        self._source_bundle_hash = source_bundle_hash
        self._context = execution_context

    def enabled(self) -> bool:
        return self._local.enabled()

    def prepare_parent_source_context(
        self,
        *,
        parent_artifact: PrivateModelArtifactManifest,
        workspace_dir: Path,
    ) -> ParentImageSourceContext:
        del workspace_dir
        with self._source_context_lock:
            context = self._source_contexts.get(parent_artifact.model_artifact_hash)
        if context is None:
            raise AutoresearchExecutorV2Error(
                "source context differs from known Git-tree parent artifacts"
            )
        return context

    def validate_draft_against_source_context(self, *args: Any, **kwargs: Any):
        return self._local.validate_draft_against_source_context(*args, **kwargs)

    def check_patch_applies(self, *args: Any, **kwargs: Any) -> None:
        self._local.check_patch_applies(*args, **kwargs)

    def restore_rehydrated_candidate_source_context(
        self,
        *,
        candidate: BuiltCodeEditCandidate,
    ) -> ParentImageSourceContext:
        """Rebuild a measured tree node from its committed cumulative root patch."""

        artifact = candidate.build.candidate_model_manifest
        artifact_hash = artifact.model_artifact_hash
        with self._source_context_lock:
            existing = self._source_contexts.get(artifact_hash)
            if existing is not None:
                return existing

            root_hash = self._source_context.source_tree_hash
            draft_hash = sha256_json(
                {"unified_diff": candidate.draft.unified_diff}
            )
            if (
                candidate.tree_depth <= 0
                or candidate.tree_root_artifact_hash != root_hash
                or candidate.tree_cumulative_source_diff_hash != draft_hash
                or candidate.build.source_diff_hash != draft_hash
                or not re.fullmatch(r"[0-9a-f]{64}", candidate.tree_git_commit)
                or str(
                    candidate.build.code_edit_manifest.get("parent_artifact_hash")
                    or ""
                )
                != root_hash
            ):
                raise AutoresearchExecutorV2Error(
                    "rehydrated Git-tree candidate commitment differs"
                )
            source_errors = self._local.validate_draft_against_source_context(
                candidate.draft,
                self._source_context,
            )
            if source_errors:
                raise AutoresearchExecutorV2Error(
                    "rehydrated Git-tree candidate source binding differs"
                )

            destination = (
                self._derived_source_root / artifact_hash.split(":", 1)[-1]
            )
            with tempfile.TemporaryDirectory(
                prefix="restore-",
                dir=self._derived_source_root,
            ) as tmp:
                staging = Path(tmp) / "source"
                _copy_source_tree(self._source_context.source_root, staging)
                _initialize_temporary_git_repo(staging)
                diff_path = Path(tmp) / "candidate.diff"
                diff_path.write_text(
                    candidate.draft.unified_diff,
                    encoding="utf-8",
                )
                _run_git_apply(
                    diff_path,
                    cwd=staging,
                    timeout_seconds=120,
                    check=True,
                )
                _run_git_apply(
                    diff_path,
                    cwd=staging,
                    timeout_seconds=120,
                    check=False,
                )
                measured_hash = compute_private_source_tree_hash(staging)
                if measured_hash != artifact_hash:
                    raise AutoresearchExecutorV2Error(
                        "rehydrated Git-tree candidate source hash differs"
                    )
                _copy_source_tree(staging, destination)

            context = _source_context(
                source_root=destination,
                artifact=artifact,
                config=self.config,
            )
            if context.source_tree_hash != artifact_hash:
                raise AutoresearchExecutorV2Error(
                    "rehydrated candidate source context commitment differs"
                )
            self._source_contexts[artifact_hash] = context
            return context

    def build(
        self,
        *,
        draft: CodeEditDraft,
        parent_artifact: PrivateModelArtifactManifest,
        run_id: str,
        candidate_index: int,
        source_context: ParentImageSourceContext | None = None,
    ) -> CodeEditBuildResult:
        context = source_context or self._source_context
        self._local.check_patch_applies(
            draft=draft,
            parent_artifact=parent_artifact,
            source_context=context,
        )
        source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
        with tempfile.TemporaryDirectory(
            prefix="leadpoet-autoresearch-patched-source-v2-"
        ) as tmp:
            measured_repo = Path(tmp) / "repo"
            _prepare_parent_image_workspace(
                image_digest=parent_artifact.image_digest,
                repo_dir=measured_repo,
                timeout_seconds=self.config.code_edit_build_timeout_seconds,
                source_context=context,
            )
            diff_path = Path(tmp) / "candidate.diff"
            diff_path.write_text(draft.unified_diff, encoding="utf-8")
            _run(
                ["git", "apply", "--recount", str(diff_path)],
                cwd=measured_repo,
                timeout_seconds=120,
            )
            expected_candidate_artifact_hash = compute_private_source_tree_hash(
                measured_repo
            )
            derived_source_root = (
                self._derived_source_root
                / expected_candidate_artifact_hash.split(":", 1)[-1]
            )
            with self._source_context_lock:
                if not derived_source_root.is_dir():
                    _copy_source_tree(measured_repo, derived_source_root)
        expected_state_hash = sha256_json(
            {
                "run_id": str(run_id),
                "candidate_index": int(candidate_index),
                "parent_artifact_hash": parent_artifact.model_artifact_hash,
                "source_bundle_hash": self._source_bundle_hash,
                "source_diff_hash": source_diff_hash,
                "expected_candidate_artifact_hash": expected_candidate_artifact_hash,
            }
        )

        def validate(value: Mapping[str, Any]) -> Dict[str, Any]:
            if set(value) != {"schema_version", "build_result"}:
                raise AutoresearchExecutorV2Error("candidate build response fields are invalid")
            if value.get("schema_version") != HOST_BUILD_RESULT_SCHEMA_VERSION:
                raise AutoresearchExecutorV2Error("candidate build response schema is invalid")
            result = _validate_build_result(
                _mapping(value.get("build_result"), "build_result"),
                draft=draft,
                parent_artifact=parent_artifact,
                expected_candidate_artifact_hash=expected_candidate_artifact_hash,
            )
            return {
                "schema_version": HOST_BUILD_RESULT_SCHEMA_VERSION,
                "build_result": _build_result_document(result),
            }

        response = self._context.execute_host_operation(
            operation=HOST_BUILD_CANDIDATE,
            payload={
                "run_id": str(run_id),
                "candidate_index": int(candidate_index),
                "parent_artifact": parent_artifact.to_dict(),
                "draft": draft.to_dict(),
                "source_bundle_hash": self._source_bundle_hash,
                "source_diff_hash": source_diff_hash,
                "expected_candidate_artifact_hash": expected_candidate_artifact_hash,
            },
            expected_state_hash=expected_state_hash,
            timeout_seconds=max(120, int(self.config.code_edit_build_timeout_seconds) + 120),
            response_validator=validate,
        )
        build_result = _validate_build_result(
            response["build_result"],
            draft=draft,
            parent_artifact=parent_artifact,
            expected_candidate_artifact_hash=expected_candidate_artifact_hash,
        )
        candidate_source_context = _source_context(
            source_root=derived_source_root,
            artifact=build_result.candidate_model_manifest,
            config=self.config,
        )
        if candidate_source_context.source_tree_hash != expected_candidate_artifact_hash:
            raise AutoresearchExecutorV2Error(
                "derived candidate source context commitment differs"
            )
        with self._source_context_lock:
            self._source_contexts[expected_candidate_artifact_hash] = (
                candidate_source_context
            )
        return build_result


def _source_context(
    *,
    source_root: Path,
    artifact: PrivateModelArtifactManifest,
    config: ResearchLabGatewayConfig,
) -> ParentImageSourceContext:
    editable_files = _editable_runtime_files(
        source_root,
        allowed_prefixes=config.code_edit_allowed_path_prefixes(),
        allowed_exact_paths=config.code_edit_allowed_exact_paths(),
        allowed_suffixes=config.code_edit_allowed_suffixes(),
    )
    if not editable_files:
        raise AutoresearchExecutorV2Error("source bundle has no editable runtime files")
    parent_image_digest_hash = canonical_hash({"image_digest": artifact.image_digest})
    planner_source_index: Dict[str, Any] = {}
    if bool(getattr(config, "planner_symbol_index_enabled", True)):
        planner_source_index = build_source_symbol_index(
            source_root=source_root,
            editable_files=editable_files,
            source_tree_hash=artifact.model_artifact_hash,
            parent_image_digest_hash=parent_image_digest_hash,
        )
    return ParentImageSourceContext(
        source_root=source_root,
        # Preserve the exact prompt/event value used by the host engine.  The
        # V2 receipt records the attested bundle separately; changing this
        # product-facing field would alter loop prompts and checkpoints.
        source_mode="parent_image_extract",
        parent_image_digest_hash=parent_image_digest_hash,
        source_tree_hash=artifact.model_artifact_hash,
        top_level_paths=tuple(_top_level_paths(source_root)),
        editable_files=tuple(editable_files),
        file_previews=tuple(_source_file_previews(source_root, editable_files)),
        planner_source_index=planner_source_index,
    )


class AutoresearchExecutorV2:
    """Run the current loop in the measured autoresearch role."""

    def __init__(
        self,
        *,
        provider_execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hashes: Mapping[str, str],
        config_supplier: Callable[[], ResearchLabGatewayConfig] = ResearchLabGatewayConfig,
        engine_factory: Callable[..., CodeEditLoopEngine] = CodeEditLoopEngine,
        scoring_graph_verifier: Optional[Callable[[Mapping[str, Any]], None]] = None,
        probe_execute: Optional[
            Callable[[Mapping[str, Any]], Mapping[str, Any]]
        ] = None,
        coordinator_boot_verifier: Optional[
            Callable[[Mapping[str, Any]], Any]
        ] = None,
        artifact_seal: Optional[Callable[..., Mapping[str, Any]]] = None,
    ) -> None:
        if artifact_seal is None:
            raise AutoresearchExecutorV2Error(
                "autoresearch hidden-artifact sealing is required"
            )
        self._provider_execute = provider_execute
        self._retry_policy_hashes = dict(retry_policy_hashes)
        self._config_supplier = config_supplier
        self._engine_factory = engine_factory
        self._scoring_graph_verifier = scoring_graph_verifier
        self._probe_execute = probe_execute
        self._coordinator_boot_verifier = coordinator_boot_verifier
        self._artifact_seal = artifact_seal
        self._transport = BrokeredProviderTransportV2(self._provider_execute)
        self._transport.install()

    def close(self) -> None:
        self._transport.restore()

    async def _seal_hidden_artifact(
        self,
        *,
        context: ExecutionContextV2,
        plaintext: bytes,
    ) -> Dict[str, Any]:
        payload = bytes(plaintext)
        if not payload:
            raise AutoresearchExecutorV2Error(
                "autoresearch hidden artifact is empty"
            )
        descriptor = dict(
            await asyncio.to_thread(
                self._artifact_seal,
                plaintext=payload,
                job_id=context.job_id,
                purpose=context.purpose,
                artifact_kind="autoresearch_hidden_artifact",
            )
        )
        if (
            descriptor.get("status") != "sealed"
            or descriptor.get("job_id") != context.job_id
            or descriptor.get("purpose") != context.purpose
            or descriptor.get("artifact_kind")
            != "autoresearch_hidden_artifact"
            or descriptor.get("plaintext_hash") != sha256_bytes(payload)
        ):
            raise AutoresearchExecutorV2Error(
                "autoresearch hidden artifact commitment differs"
            )
        for field in (
            "artifact_id",
            "plaintext_hash",
            "ciphertext_hash",
            "encryption_context_hash",
        ):
            context.record_artifact(_hash(descriptor.get(field), field))
        return descriptor

    def _probe_resolver(
        self,
        context: ExecutionContextV2,
        *,
        dynamic_provider_catalog: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]]:
        if self._probe_execute is None:
            return None
        if self._coordinator_boot_verifier is None:
            raise AutoresearchExecutorV2Error(
                "provider probe coordinator verifier is unavailable"
            )
        if dynamic_provider_catalog is None:
            dynamic_provider_catalog = {
                "schema_version": "leadpoet.source_add_runtime_catalog.v2",
                "routes": [],
                "catalog_hash": sha256_json(
                    {
                        "schema_version": (
                            "leadpoet.source_add_runtime_catalog.v2"
                        ),
                        "routes": [],
                    }
                ),
            }
        normalized_dynamic_catalog = validate_source_add_runtime_catalog_v2(
            dynamic_provider_catalog
        )

        def resolve(value: Mapping[str, Any]) -> Mapping[str, Any]:
            endpoint = value.get("endpoint")
            provider_id = (
                str(endpoint.get("provider_id") or "")
                if isinstance(endpoint, Mapping)
                else ""
            )
            dynamic_route = source_add_route_for_provider_v2(
                normalized_dynamic_catalog,
                provider_id,
            )
            request = {
                "schema_version": PROVIDER_EVIDENCE_REQUEST_SCHEMA_VERSION,
                "caller_job_id": context.job_id,
                "purpose": context.purpose,
                **dict(value),
            }
            if dynamic_route is not None:
                request["dynamic_route"] = dynamic_route
            result = dict(self._probe_execute(request))
            required = {
                "status",
                "body_b64",
                "evidence",
                "transport_attempts",
                "evidence_artifact_hashes",
                "record",
                "source_record",
                "source_boot_identity",
                "coordinator_boot_identity",
            }
            if set(result) != required:
                raise AutoresearchExecutorV2Error(
                    "provider probe coordinator result fields are invalid"
                )
            boot = _mapping(
                result["coordinator_boot_identity"],
                "coordinator_boot_identity",
            )
            self._coordinator_boot_verifier(boot)
            record = validate_signed_provider_evidence_record(
                _mapping(result["record"], "provider_evidence_record"),
                boot_identity=boot,
            )
            if record["request_hash"] != sha256_json(request):
                raise AutoresearchExecutorV2Error(
                    "provider probe record request commitment differs"
                )
            try:
                body = base64.b64decode(str(result["body_b64"]), validate=True)
            except Exception as exc:
                raise AutoresearchExecutorV2Error(
                    "provider probe response body is invalid"
                ) from exc
            if record["body_hash"] != sha256_bytes(body):
                raise AutoresearchExecutorV2Error(
                    "provider probe response commitment differs"
                )
            if (
                int(result["status"]) != record["status"]
                or str(result["evidence"]) != record["evidence"]
            ):
                raise AutoresearchExecutorV2Error(
                    "provider probe terminal differs from signed record"
                )
            source = result["source_record"]
            source_boot = result["source_boot_identity"]
            if record["evidence"] == "hit":
                normalized_source_boot = _mapping(
                    source_boot,
                    "provider_evidence_source_boot_identity",
                )
                self._coordinator_boot_verifier(normalized_source_boot)
                source_record = validate_signed_provider_evidence_record(
                    _mapping(source, "provider_evidence_source_record"),
                    boot_identity=normalized_source_boot,
                )
                if (
                    source_record["record_hash"] != record["source_record_hash"]
                    or source_record["evidence"] not in {"recorded", "restored"}
                    or source_record["request_fingerprint"]
                    != record["request_fingerprint"]
                    or source_record["body_hash"] != record["body_hash"]
                    or source_record["status"] != record["status"]
                ):
                    raise AutoresearchExecutorV2Error(
                        "provider probe cache ancestry differs"
                    )
                if (
                    source_record["evidence"] == "recorded"
                    and source_record["source_record_hash"]
                ) or (
                    source_record["evidence"] == "restored"
                    and not source_record["source_record_hash"]
                ):
                    raise AutoresearchExecutorV2Error(
                        "provider probe cache source lineage is invalid"
                    )
            elif (
                source is not None
                or source_boot is not None
                or record["source_record_hash"]
            ):
                raise AutoresearchExecutorV2Error(
                    "provider probe unexpectedly carries cache ancestry"
                )
            attempts = result["transport_attempts"]
            if not isinstance(attempts, list):
                raise AutoresearchExecutorV2Error(
                    "provider probe transport ledger is invalid"
                )
            attempt_hashes = set()
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    raise AutoresearchExecutorV2Error(
                        "provider probe transport attempt is invalid"
                    )
                attempt_hash = str(attempt.get("attempt_hash") or "")
                if attempt_hash in attempt_hashes:
                    raise AutoresearchExecutorV2Error(
                        "provider probe transport attempt is duplicated"
                    )
                attempt_hashes.add(attempt_hash)
                context.record_transport(attempt)
            if record["transport_attempt_hash"] and (
                record["transport_attempt_hash"] not in attempt_hashes
            ):
                raise AutoresearchExecutorV2Error(
                    "provider probe transport attempt is missing"
                )
            artifacts = result["evidence_artifact_hashes"]
            if not isinstance(artifacts, list) or record["record_hash"] not in artifacts:
                raise AutoresearchExecutorV2Error(
                    "provider probe evidence artifacts are incomplete"
                )
            if record["evidence"] == "hit":
                required_cache_artifacts = {
                    source_record["record_hash"],
                    normalized_source_boot["boot_identity_hash"],
                }
                if source_record["evidence"] == "restored":
                    required_cache_artifacts.add(source_record["source_record_hash"])
                if not required_cache_artifacts.issubset(set(artifacts)):
                    raise AutoresearchExecutorV2Error(
                        "provider probe cache artifacts are incomplete"
                    )
            for artifact_hash in artifacts:
                context.record_artifact(str(artifact_hash))
            return {
                "status": record["status"],
                "body": body,
                "evidence": record["evidence"],
                "transport_attempts": [dict(item) for item in attempts],
                "evidence_artifact_hashes": [str(item) for item in artifacts],
            }

        return resolve

    async def __call__(
        self,
        operation: str,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if operation == OP_VERIFY_OPENROUTER_GUARD:
            return await self._verify_openrouter_guard(payload, context)
        if operation == OP_REPAIR_STALE_PARENT:
            return await self._repair_stale_parent(payload, context)
        if operation != OP_RUN_CODE_EDIT_LOOP:
            raise AutoresearchExecutorV2Error("unsupported V2 autoresearch operation")
        request = self._validate_request(payload)
        if (
            request["openrouter_context"]["privacy_receipt_hash"]
            not in context.parent_receipt_hashes
        ):
            raise AutoresearchExecutorV2Error(
                "OpenRouter guard receipt is missing from autoresearch ancestry"
            )
        if (
            request["component_registry_receipt_hash"]
            not in context.parent_receipt_hashes
        ):
            raise AutoresearchExecutorV2Error(
                "component registry receipt is missing from autoresearch ancestry"
            )
        if (
            request["provider_catalog_receipt_hash"]
            not in context.parent_receipt_hashes
        ):
            raise AutoresearchExecutorV2Error(
                "provider catalog receipt is missing from autoresearch ancestry"
            )
        if (
            request["provider_outcome_receipt_hash"]
            not in context.parent_receipt_hashes
        ):
            raise AutoresearchExecutorV2Error(
                "provider outcome receipt is missing from autoresearch ancestry"
            )
        config = self._config_supplier()
        artifact = PrivateModelArtifactManifest.from_mapping(request["artifact"])
        artifact_errors = validate_private_model_artifact_manifest(artifact)
        if artifact_errors:
            raise AutoresearchExecutorV2Error(
                "parent artifact is invalid: " + "; ".join(artifact_errors)
            )

        with tempfile.TemporaryDirectory(prefix="leadpoet-autoresearch-v2-") as tmp:
            source_root = Path(tmp) / "source"
            source_evidence = extract_source_bundle_v2(
                request["source_bundle"],
                destination=source_root,
                expected_source_tree_hash=artifact.model_artifact_hash,
            )
            source_archive = base64.b64decode(
                str(request["source_bundle"]["archive_b64"]), validate=True
            )
            await self._seal_hidden_artifact(
                context=context,
                plaintext=source_archive,
            )
            source_context = _source_context(
                source_root=source_root,
                artifact=artifact,
                config=config,
            )
            tree_runtime_policy = _mapping(
                request["budget_context"].get("tree_policy"),
                "Git-tree runtime policy",
            )
            tree_policy = TreePolicy.from_mapping(
                _mapping(tree_runtime_policy.get("policy"), "Git-tree policy")
            )
            tree_id = derive_tree_id(
                run_id=request["run_id"],
                root_artifact_hash=artifact.model_artifact_hash,
                policy=tree_policy,
            )
            (
                provider_entries,
                provider_capabilities,
                provider_probe_catalog,
                dynamic_provider_catalog,
            ) = self._provider_execution_inputs(request, context)
            artifact_io = _HostArtifactIO(context)
            builder = _HostCandidateBuilder(
                config=config,
                source_context=source_context,
                source_bundle_hash=str(source_evidence["archive_sha256"]),
                execution_context=context,
            )
            event_sink = self._event_sink(
                context=context,
                expected_event_state_hash=request["expected_event_state_hash"],
            )
            pause_checker = self._pause_checker(context)
            dev_evaluator = (
                self._dev_evaluator(context)
                if request["dev_evaluator_enabled"]
                else None
            )
            loop_caller = self._loop_model_caller(
                context=context,
                config=config,
                run_id=request["run_id"],
                model_id=request["model_id"],
                model_doc=request["model_doc"],
                openrouter_context=request["openrouter_context"],
            )
            settings = AutoResearchRuntimeSettings(**request["loop_settings"])
            engine = self._engine_factory(
                settings=settings,
                call_openrouter=loop_caller,
                event_sink=event_sink,
                builder=builder,
                dev_evaluator=dev_evaluator,
                probe_private_window_term_hashes=frozenset(
                    request["probe_private_window_term_hashes"]
                ),
                probe_evidence_resolver=self._probe_resolver(
                    context,
                    dynamic_provider_catalog=dynamic_provider_catalog,
                ),
                provider_outcome_digest=(request["provider_outcome_digest"] or None),
                artifact_io=artifact_io,
                tree_repository=_HostGitTreeRepository(
                    context,
                    tree_id=tree_id,
                ),
                provider_registry_loader=lambda: (
                    list(provider_entries),
                    provider_capabilities,
                ),
                provider_probe_catalog_loader=lambda: list(
                    provider_probe_catalog
                ),
            )
            retry_policy_hashes = {
                **self._retry_policy_hashes,
                **source_add_runtime_retry_hashes_v2(
                    dynamic_provider_catalog
                ),
            }
            with self._transport.scope(
                job_id=context.job_id,
                purpose=context.purpose,
                logical_operation_id=context.job_id,
                retry_policy_hashes=retry_policy_hashes,
                terminal_sink=context.record_transport,
                artifact_sink=context.record_artifact,
                dynamic_provider_catalog=dynamic_provider_catalog,
            ):
                result = await engine.run(
                    run_id=request["run_id"],
                    ticket=request["ticket"],
                    artifact=artifact,
                    component_registry=request["component_registry"],
                    benchmark_public_summary=request["benchmark_public_summary"],
                    model_id=request["model_id"],
                    budget_context=request["budget_context"],
                    requested_loop_count=request["requested_loop_count"],
                    resume_state=(request["resume_state"] or None),
                    should_pause=pause_checker,
                )
            output = _result_document(result)
            await self._seal_hidden_artifact(
                context=context,
                plaintext=canonical_json(output).encode("utf-8"),
            )
            context.record_artifact(source_evidence["archive_sha256"])
            context.record_artifact(artifact.model_artifact_hash)
            return ExecutionResultV2(
                output=output,
                artifact_hashes=(
                    source_evidence["archive_sha256"],
                    artifact.model_artifact_hash,
                ),
            )

    async def _verify_openrouter_guard(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        required = {
            "schema_version",
            "key_ref",
            "key_ref_hash",
            "miner_hotkey_hash",
            "runtime_credential_value_hash",
            "management_credential_value_hash",
            "stage",
            "request_policy",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise AutoresearchExecutorV2Error(
                "OpenRouter guard request fields are invalid"
            )
        if payload.get("schema_version") != OPENROUTER_GUARD_REQUEST_SCHEMA_VERSION:
            raise AutoresearchExecutorV2Error(
                "OpenRouter guard request schema is invalid"
            )
        key_ref = str(payload.get("key_ref") or "")
        if not _OPENROUTER_KEY_REF_RE.fullmatch(key_ref):
            raise AutoresearchExecutorV2Error("OpenRouter guard key ref is invalid")
        key_ref_hash = _hash(payload.get("key_ref_hash"), "OpenRouter key ref hash")
        if key_ref_hash != sha256_bytes(key_ref.encode("utf-8")):
            raise AutoresearchExecutorV2Error(
                "OpenRouter guard key ref commitment differs"
            )
        runtime_hash = _hash(
            payload.get("runtime_credential_value_hash"),
            "runtime credential value hash",
        )
        management_hash = _hash(
            payload.get("management_credential_value_hash"),
            "management credential value hash",
        )
        miner_hash = _hash(
            payload.get("miner_hotkey_hash"),
            "miner hotkey hash",
        )
        stage = str(payload.get("stage") or "")
        if not stage or "\x00" in stage:
            raise AutoresearchExecutorV2Error("OpenRouter guard stage is invalid")
        request_policy = _mapping(payload.get("request_policy"), "request_policy")
        if request_policy != strict_openrouter_provider_policy():
            raise AutoresearchExecutorV2Error(
                "OpenRouter guard request policy differs"
            )

        preflight_status = "passed"
        preflight_error_type = ""
        preflight_doc: Dict[str, Any] = {}
        with self._transport.scope(
            job_id=context.job_id,
            purpose=context.purpose,
            logical_operation_id=context.job_id,
            retry_policy_hashes=self._retry_policy_hashes,
            terminal_sink=context.record_transport,
            artifact_sink=context.record_artifact,
        ):
            try:
                preflight_doc = dict(
                    await asyncio.to_thread(
                        preflight_openrouter_key,
                        _RUNTIME_KEY_PLACEHOLDER,
                    )
                )
            except Exception as exc:
                # Preserve the existing best-effort credit preflight.  The
                # mandatory workspace privacy check below still fails closed
                # on an unusable key or unauthenticated transport.
                preflight_status = "skipped"
                preflight_error_type = type(exc).__name__
            privacy_doc = dict(
                await asyncio.to_thread(
                    verify_openrouter_workspace_privacy,
                    runtime_key=_RUNTIME_KEY_PLACEHOLDER,
                    management_key=_MANAGEMENT_KEY_PLACEHOLDER,
                    stage=stage,
                    request_policy=request_policy,
                )
            )

        # The exact key never leaves the coordinator.  Replace the placeholder
        # hash with the KMS-envelope commitment that the coordinator verified
        # while leasing the job credential.
        privacy_doc["management_key_hash"] = management_hash.split(":", 1)[1]
        privacy_doc["proof_hash"] = canonical_hash(
            {
                key: value
                for key, value in privacy_doc.items()
                if key != "proof_hash"
            }
        )
        remaining = preflight_doc.get("limit_remaining")
        try:
            credit_depleted = remaining is not None and float(remaining) <= 0.0
        except (TypeError, ValueError):
            credit_depleted = False
        output = {
            "schema_version": OPENROUTER_GUARD_RESULT_SCHEMA_VERSION,
            "key_ref_hash": key_ref_hash,
            "miner_hotkey_hash": miner_hash,
            "runtime_credential_value_hash": runtime_hash,
            "management_credential_value_hash": management_hash,
            "preflight_status": preflight_status,
            "preflight_error_type": preflight_error_type,
            "credit_depleted": bool(credit_depleted),
            "credit_limit_remaining": remaining,
            "privacy_proof_doc": privacy_doc,
        }
        await self._seal_hidden_artifact(
            context=context,
            plaintext=canonical_json(output).encode("utf-8"),
        )
        return ExecutionResultV2(
            output=output,
            artifact_hashes=(
                key_ref_hash,
                miner_hash,
                runtime_hash,
                management_hash,
                sha256_json(privacy_doc),
            ),
        )

    async def _repair_stale_parent(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        required = {
            "schema_version",
            "run_id",
            "candidate_id",
            "active_artifact",
            "source_bundle",
            "original_draft",
            "original_source_diff_hash",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise AutoresearchExecutorV2Error(
                "stale-parent repair request fields are invalid"
            )
        if (
            payload.get("schema_version")
            != STALE_PARENT_REPAIR_REQUEST_SCHEMA_VERSION
        ):
            raise AutoresearchExecutorV2Error(
                "stale-parent repair request schema is invalid"
            )
        run_id = str(payload.get("run_id") or "")
        candidate_id = str(payload.get("candidate_id") or "")
        if not run_id or not candidate_id or "\x00" in run_id + candidate_id:
            raise AutoresearchExecutorV2Error(
                "stale-parent repair identity is invalid"
            )
        if not context.parent_receipt_hashes:
            raise AutoresearchExecutorV2Error(
                "stale-parent repair lacks candidate receipt ancestry"
            )
        if (
            context.provider_credential_profile != "stale_parent_repair"
            or "openrouter" not in context.provider_credential_ref_hashes
        ):
            raise AutoresearchExecutorV2Error(
                "stale-parent repair provider authority is unavailable"
            )

        artifact = PrivateModelArtifactManifest.from_mapping(
            _mapping(payload.get("active_artifact"), "active_artifact")
        )
        artifact_errors = validate_private_model_artifact_manifest(artifact)
        if artifact_errors:
            raise AutoresearchExecutorV2Error(
                "stale-parent active artifact is invalid: "
                + "; ".join(artifact_errors)
            )
        original_draft = _draft_from_mapping(
            _mapping(payload.get("original_draft"), "original_draft")
        )
        original_source_diff_hash = _hash(
            payload.get("original_source_diff_hash"),
            "original_source_diff_hash",
        )
        if original_source_diff_hash != sha256_json(
            {"unified_diff": original_draft.unified_diff}
        ):
            raise AutoresearchExecutorV2Error(
                "stale-parent original source diff commitment differs"
            )

        config = self._config_supplier()
        source_bundle = _mapping(payload.get("source_bundle"), "source_bundle")
        with tempfile.TemporaryDirectory(
            prefix="leadpoet-stale-parent-repair-v2-"
        ) as tmp:
            source_root = Path(tmp) / "source"
            source_evidence = extract_source_bundle_v2(
                source_bundle,
                destination=source_root,
                expected_source_tree_hash=artifact.model_artifact_hash,
            )
            source_archive = base64.b64decode(
                str(source_bundle["archive_b64"]), validate=True
            )
            await self._seal_hidden_artifact(
                context=context,
                plaintext=source_archive,
            )
            source_context = _source_context(
                source_root=source_root,
                artifact=artifact,
                config=config,
            )
            builder = CodeEditCandidateBuilder(config)
            repair_used = False
            draft = original_draft
            try:
                builder.check_patch_applies(
                    draft=original_draft,
                    parent_artifact=artifact,
                    source_context=source_context,
                )
            except CodeEditPatchApplyError as original_error:
                if not config.stale_parent_rebase_repair_enabled:
                    raise
                model_id = str(
                    config.stale_parent_rebase_repair_model or ""
                ).strip()
                if not model_id:
                    raise AutoresearchExecutorV2Error(
                        "stale-parent repair model is not configured"
                    )
                read_batch = resolve_source_inspection_requests(
                    source_context,
                    [
                        CodeEditSourceInspectionRequest(
                            operation="read_file",
                            path=path,
                            rationale=(
                                "repair stale parent code-edit diff against "
                                "current model source"
                            ),
                        )
                        for path in original_draft.target_files
                    ],
                    already_read_paths=(),
                    max_files=max(
                        len(original_draft.target_files),
                        config.code_edit_source_inspection_max_files,
                    ),
                    max_file_bytes=(
                        config.code_edit_source_inspection_file_bytes
                    ),
                    max_total_bytes=(
                        config.code_edit_source_inspection_total_bytes
                    ),
                    max_search_matches=(
                        config.code_edit_source_inspection_search_matches
                    ),
                )
                # Keep the exact current request builder, retry behavior, and
                # response parser.  Only its transport and execution boundary
                # move into the measured autoresearch role.
                from gateway.research_lab.scoring_worker import (
                    _call_operator_openrouter_json,
                )

                with self._transport.scope(
                    job_id=context.job_id,
                    purpose=context.purpose,
                    logical_operation_id=context.job_id,
                    retry_policy_hashes=self._retry_policy_hashes,
                    terminal_sink=context.record_transport,
                    artifact_sink=context.record_artifact,
                ):
                    raw = await _call_operator_openrouter_json(
                        api_key=_RUNTIME_KEY_PLACEHOLDER,
                        model_id=model_id,
                        messages=build_code_edit_repair_messages(
                            draft=original_draft,
                            apply_error=str(original_error),
                            source_inspection_context=read_batch.model_context,
                            runtime_source_context=source_context.prompt_context(),
                            budget_context={
                                "repair_context": "stale_parent_rebase",
                                "operator_funded": True,
                            },
                            repair_attempt=1,
                            max_candidates=1,
                        ),
                        timeout_seconds=(
                            config.stale_parent_rebase_repair_timeout_seconds
                        ),
                    )
                draft = parse_code_edit_repair_response(
                    raw,
                    original_draft=original_draft,
                )[0]
                source_errors = builder.validate_draft_against_source_context(
                    draft,
                    source_context,
                    read_paths=read_batch.read_paths,
                    require_read=True,
                )
                if source_errors:
                    raise AutoresearchExecutorV2Error(
                        "; ".join(source_errors)
                    )
                builder.check_patch_applies(
                    draft=draft,
                    parent_artifact=artifact,
                    source_context=source_context,
                )
                repair_used = True

        result_source_diff_hash = sha256_json(
            {"unified_diff": draft.unified_diff}
        )
        output = {
            "schema_version": STALE_PARENT_REPAIR_RESULT_SCHEMA_VERSION,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "draft": draft.to_dict(),
            "repair_used": repair_used,
            "original_source_diff_hash": original_source_diff_hash,
            "result_source_diff_hash": result_source_diff_hash,
            "active_artifact_hash": artifact.model_artifact_hash,
            "source_bundle_hash": str(source_evidence["archive_sha256"]),
        }
        await self._seal_hidden_artifact(
            context=context,
            plaintext=canonical_json(output).encode("utf-8"),
        )
        return ExecutionResultV2(
            output=output,
            artifact_hashes=(
                artifact.model_artifact_hash,
                artifact.manifest_hash,
                str(source_evidence["archive_sha256"]),
                original_source_diff_hash,
                result_source_diff_hash,
            ),
        )

    def _provider_execution_inputs(
        self,
        request: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> tuple[list[ProviderRegistryEntry], Any, list[Any], Dict[str, Any]]:
        result = _mapping(
            request.get("provider_catalog_result"),
            "provider catalog result",
        )
        source_rows = [
            dict(item) for item in result.get("provisioned_sources") or ()
        ]
        private_rows = [
            dict(item) for item in result.get("private_registry_rows") or ()
        ]
        runtime_catalog = validate_source_add_runtime_catalog_v2(
            _mapping(result.get("runtime_catalog"), "runtime catalog")
        )
        credential_refs = dict(context.provider_credential_ref_hashes)
        builtin_slots = {
            "deepline": "deepline",
            "exa": "exa",
            "openrouter": "openrouter",
            "or": "openrouter",
            "scrapingdog": "scrapingdog",
            "sd": "scrapingdog",
            "truelist": "truelist",
        }

        def credential_ready(provider: Mapping[str, Any]) -> bool | None:
            provider_id = str(provider.get("id") or "")
            if str(provider.get("auth_kind") or "none").lower() == "none":
                return True
            route = source_add_route_for_provider_v2(
                runtime_catalog,
                provider_id,
            )
            if route is not None:
                return credential_refs.get(provider_id) == route.get(
                    "credential_value_hash"
                )
            slot = builtin_slots.get(provider_id, provider_id)
            if slot in credential_refs:
                return bool(credential_refs[slot])
            return None

        static_docs = [entry.to_dict() for entry in seed_provider_registry()]
        capabilities = load_effective_provider_capabilities_sync(
            static_docs,
            strict_remote=True,
            private_row_loader=lambda: list(private_rows),
            source_row_loader=lambda: list(source_rows),
            credential_ready_resolver=credential_ready,
        )
        entries = [
            ProviderRegistryEntry.from_mapping(item)
            for item in capabilities.providers
        ]
        entry_errors = validate_provider_registry_entries(entries)
        if entry_errors:
            raise AutoresearchExecutorV2Error(
                "authenticated provider registry is invalid: "
                + "; ".join(entry_errors[:5])
            )
        if capabilities.source_add_provider_count != len(
            runtime_catalog["routes"]
        ):
            raise AutoresearchExecutorV2Error(
                "authenticated SOURCE_ADD capability set is incomplete"
            )
        if capabilities.private_snapshot_loaded:
            probe_catalog = [
                ProviderProbeEndpoint.from_mapping(item)
                for entry in entries
                for item in entry.probe_endpoints
            ]
        else:
            probe_catalog = list(default_probe_catalog())
            existing_ids = {item.endpoint_id for item in probe_catalog}
            probe_catalog.extend(
                item
                for item in probe_endpoints_from_provisioned_rows(source_rows)
                if item.endpoint_id not in existing_ids
            )
        probe_errors = validate_probe_catalog(probe_catalog)
        if probe_errors:
            raise AutoresearchExecutorV2Error(
                "authenticated provider probe catalog is invalid: "
                + "; ".join(probe_errors[:5])
            )
        return entries, capabilities, probe_catalog, runtime_catalog

    def _validate_request(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != _REQUEST_FIELDS:
            raise AutoresearchExecutorV2Error("autoresearch request fields are invalid")
        if payload.get("schema_version") != AUTORESEARCH_REQUEST_SCHEMA_VERSION:
            raise AutoresearchExecutorV2Error("autoresearch request schema is invalid")
        run_id = str(payload.get("run_id") or "")
        model_id = str(payload.get("model_id") or "")
        if not run_id or not model_id or "\x00" in run_id + model_id:
            raise AutoresearchExecutorV2Error("autoresearch request identity is invalid")
        requested_loop_count = payload.get("requested_loop_count")
        if not isinstance(requested_loop_count, int) or requested_loop_count < 1:
            raise AutoresearchExecutorV2Error("requested_loop_count is invalid")
        settings = _mapping(payload.get("loop_settings"), "loop_settings")
        settings_fields = {item.name for item in fields(AutoResearchRuntimeSettings)}
        if set(settings) != settings_fields:
            raise AutoresearchExecutorV2Error("loop settings fields are invalid")
        AutoResearchRuntimeSettings(**settings).normalized()
        hashes = payload.get("probe_private_window_term_hashes")
        if not isinstance(hashes, list) or any(
            not _HEX_HASH_RE.fullmatch(str(item or "")) for item in hashes
        ):
            raise AutoresearchExecutorV2Error("probe guard hashes are invalid")
        if len(set(hashes)) != len(hashes):
            raise AutoresearchExecutorV2Error("probe guard hashes are duplicated")
        if not isinstance(payload.get("dev_evaluator_enabled"), bool):
            raise AutoresearchExecutorV2Error("dev evaluator flag is invalid")
        openrouter = _mapping(payload.get("openrouter_context"), "openrouter_context")
        if set(openrouter) != {
            "key_ref",
            "miner_hotkey",
            "privacy_proof_doc",
            "privacy_receipt_hash",
            "runtime_credential_value_hash",
            "management_credential_value_hash",
        }:
            raise AutoresearchExecutorV2Error("OpenRouter context fields are invalid")
        privacy_receipt_hash = _hash(
            openrouter.get("privacy_receipt_hash"), "privacy receipt hash"
        )
        openrouter["privacy_receipt_hash"] = privacy_receipt_hash
        for field in (
            "runtime_credential_value_hash",
            "management_credential_value_hash",
        ):
            openrouter[field] = _hash(openrouter.get(field), field)
        privacy_doc = _mapping(openrouter.get("privacy_proof_doc"), "privacy_proof_doc")
        if not privacy_doc:
            raise AutoresearchExecutorV2Error("OpenRouter privacy proof is empty")
        component_registry = _mapping(
            payload.get("component_registry"), "component_registry"
        )
        component_evidence = _mapping(
            payload.get("component_registry_evidence"),
            "component_registry_evidence",
        )
        if set(component_evidence) != {
            "result",
            "receipt_graph",
            "root_receipt_hash",
        }:
            raise AutoresearchExecutorV2Error(
                "component registry evidence fields are invalid"
            )
        component_result = _mapping(
            component_evidence.get("result"), "component registry result"
        )
        component_graph = _mapping(
            component_evidence.get("receipt_graph"),
            "component registry receipt graph",
        )
        component_root = _hash(
            component_evidence.get("root_receipt_hash"),
            "component registry receipt hash",
        )
        validate_receipt_graph(
            component_graph,
            required_purposes=("research_lab.private_model_run.v2",),
            require_boot_attestation_verification=(
                self._scoring_graph_verifier is not None
            ),
            boot_attestation_verifier=(
                (lambda identity: self._scoring_graph_verifier(identity))
                if self._scoring_graph_verifier is not None
                else None
            ),
        )
        if component_graph.get("root_receipt_hash") != component_root:
            raise AutoresearchExecutorV2Error(
                "component registry receipt graph root differs"
            )
        if (
            component_result.get("operation") != "metadata"
            or component_result.get("output") != component_registry
            or component_result.get("output_hash") != sha256_json(component_registry)
        ):
            raise AutoresearchExecutorV2Error(
                "component registry differs from measured model metadata"
            )
        expected_result_root = sha256_bytes(
            canonical_json(component_result).encode("utf-8")
        )
        if not any(
            receipt.get("purpose") == "research_lab.private_model_run.v2"
            and receipt.get("output_root") == expected_result_root
            for receipt in component_graph.get("receipts", [])
        ):
            raise AutoresearchExecutorV2Error(
                "component registry result is not committed by its receipt graph"
            )
        active_model_evidence = _mapping(
            payload.get("active_model_evidence"),
            "active_model_evidence",
        )
        if set(active_model_evidence) != {
            "result",
            "receipt_graph",
            "root_receipt_hash",
        }:
            raise AutoresearchExecutorV2Error(
                "active model evidence fields are invalid"
            )
        active_model_result = _mapping(
            active_model_evidence.get("result"),
            "active model result",
        )
        active_model_graph = _mapping(
            active_model_evidence.get("receipt_graph"),
            "active model receipt graph",
        )
        active_model_root = _hash(
            active_model_evidence.get("root_receipt_hash"),
            "active model receipt hash",
        )
        validate_receipt_graph(
            active_model_graph,
            required_purposes=("research_lab.active_private_model.v2",),
            require_boot_attestation_verification=(
                self._scoring_graph_verifier is not None
            ),
            boot_attestation_verifier=(
                (lambda identity: self._scoring_graph_verifier(identity))
                if self._scoring_graph_verifier is not None
                else None
            ),
        )
        if (
            active_model_graph.get("root_receipt_hash") != active_model_root
            or active_model_result.get("artifact") != payload.get("artifact")
            or not any(
                receipt.get("receipt_hash") == active_model_root
                and receipt.get("purpose")
                == "research_lab.active_private_model.v2"
                and receipt.get("output_root") == sha256_json(active_model_result)
                for receipt in active_model_graph.get("receipts", [])
            )
        ):
            raise AutoresearchExecutorV2Error(
                "active model evidence differs from autoresearch artifact"
            )
        catalog_evidence = _mapping(
            payload.get("provider_catalog_evidence"),
            "provider catalog evidence",
        )
        if set(catalog_evidence) != {
            "result",
            "receipt_graph",
            "root_receipt_hash",
        }:
            raise AutoresearchExecutorV2Error(
                "provider catalog evidence fields are invalid"
            )
        catalog_result = _mapping(
            catalog_evidence.get("result"),
            "provider catalog result",
        )
        if set(catalog_result) != {
            "schema_version",
            "provisioned_sources",
            "provisioned_sources_hash",
            "private_registry_rows",
            "private_registry_rows_hash",
            "runtime_catalog",
            "runtime_catalog_hash",
        } or catalog_result.get("schema_version") != (
            "leadpoet.source_add_catalog_snapshot.v2"
        ):
            raise AutoresearchExecutorV2Error(
                "provider catalog result fields are invalid"
            )
        source_rows = catalog_result.get("provisioned_sources")
        private_rows = catalog_result.get("private_registry_rows")
        if (
            not isinstance(source_rows, list)
            or any(not isinstance(item, Mapping) for item in source_rows)
            or not isinstance(private_rows, list)
            or any(not isinstance(item, Mapping) for item in private_rows)
            or catalog_result.get("provisioned_sources_hash")
            != sha256_json([dict(item) for item in source_rows])
            or catalog_result.get("private_registry_rows_hash")
            != sha256_json([dict(item) for item in private_rows])
        ):
            raise AutoresearchExecutorV2Error(
                "provider catalog snapshot commitments differ"
            )
        try:
            runtime_catalog = validate_source_add_runtime_catalog_v2(
                _mapping(
                    catalog_result.get("runtime_catalog"),
                    "SOURCE_ADD runtime catalog",
                )
            )
            independently_derived_catalog = build_source_add_runtime_catalog_v2(
                [dict(item) for item in source_rows]
            )
        except Exception as exc:
            raise AutoresearchExecutorV2Error(
                "SOURCE_ADD runtime catalog is invalid"
            ) from exc
        if (
            runtime_catalog != independently_derived_catalog
            or catalog_result.get("runtime_catalog_hash")
            != runtime_catalog["catalog_hash"]
        ):
            raise AutoresearchExecutorV2Error(
                "SOURCE_ADD runtime catalog differs from authenticated rows"
            )
        catalog_graph = _mapping(
            catalog_evidence.get("receipt_graph"),
            "provider catalog receipt graph",
        )
        catalog_root = _hash(
            catalog_evidence.get("root_receipt_hash"),
            "provider catalog receipt hash",
        )
        validate_receipt_graph(
            catalog_graph,
            required_purposes=(
                "research_lab.source_add_catalog_snapshot.v2",
            ),
            require_boot_attestation_verification=(
                self._coordinator_boot_verifier is not None
            ),
            boot_attestation_verifier=(
                (lambda identity: self._coordinator_boot_verifier(identity))
                if self._coordinator_boot_verifier is not None
                else None
            ),
        )
        if catalog_graph.get("root_receipt_hash") != catalog_root:
            raise AutoresearchExecutorV2Error(
                "provider catalog receipt graph root differs"
            )
        catalog_root_receipts = [
            receipt
            for receipt in catalog_graph.get("receipts", ())
            if isinstance(receipt, Mapping)
            and receipt.get("receipt_hash") == catalog_root
        ]
        if (
            len(catalog_root_receipts) != 1
            or catalog_root_receipts[0].get("role") != "gateway_coordinator"
            or catalog_root_receipts[0].get("purpose")
            != "research_lab.source_add_catalog_snapshot.v2"
            or catalog_root_receipts[0].get("output_root")
            != sha256_json(catalog_result)
        ):
            raise AutoresearchExecutorV2Error(
                "provider catalog result is not committed by its receipt graph"
            )
        provider_outcome_evidence = _mapping(
            payload.get("provider_outcome_evidence"),
            "provider outcome evidence",
        )
        if set(provider_outcome_evidence) != {
            "result",
            "receipt_graph",
            "root_receipt_hash",
        }:
            raise AutoresearchExecutorV2Error(
                "provider outcome evidence fields are invalid"
            )
        try:
            provider_outcome_result = validate_provider_outcome_snapshot_v2(
                _mapping(
                    provider_outcome_evidence.get("result"),
                    "provider outcome result",
                )
            )
        except Exception as exc:
            raise AutoresearchExecutorV2Error(
                "provider outcome snapshot is invalid"
            ) from exc
        provider_outcome_graph = _mapping(
            provider_outcome_evidence.get("receipt_graph"),
            "provider outcome receipt graph",
        )
        provider_outcome_root = _hash(
            provider_outcome_evidence.get("root_receipt_hash"),
            "provider outcome receipt hash",
        )
        validate_receipt_graph(
            provider_outcome_graph,
            required_purposes=(
                "research_lab.provider_outcome_snapshot.v2",
            ),
            require_boot_attestation_verification=(
                self._coordinator_boot_verifier is not None
            ),
            boot_attestation_verifier=(
                (lambda identity: self._coordinator_boot_verifier(identity))
                if self._coordinator_boot_verifier is not None
                else None
            ),
        )
        if provider_outcome_graph.get("root_receipt_hash") != provider_outcome_root:
            raise AutoresearchExecutorV2Error(
                "provider outcome receipt graph root differs"
            )
        provider_outcome_root_receipts = [
            receipt
            for receipt in provider_outcome_graph.get("receipts", ())
            if isinstance(receipt, Mapping)
            and receipt.get("receipt_hash") == provider_outcome_root
        ]
        if (
            len(provider_outcome_root_receipts) != 1
            or provider_outcome_root_receipts[0].get("role")
            != "gateway_coordinator"
            or provider_outcome_root_receipts[0].get("purpose")
            != "research_lab.provider_outcome_snapshot.v2"
            or provider_outcome_root_receipts[0].get("status") != "succeeded"
            or provider_outcome_root_receipts[0].get("output_root")
            != sha256_json(provider_outcome_result)
        ):
            raise AutoresearchExecutorV2Error(
                "provider outcome result is not committed by its receipt graph"
            )
        if payload.get("provider_outcome_digest") != provider_outcome_result.get(
            "provider_outcome_digest"
        ):
            raise AutoresearchExecutorV2Error(
                "provider outcome digest differs from measured snapshot"
            )
        budget_context = _mapping(
            payload.get("budget_context"), "budget_context"
        )
        tree_runtime_policy = _mapping(
            budget_context.get("tree_policy"), "Git-tree runtime policy"
        )
        if set(tree_runtime_policy) != {
            "schema_version",
            "policy",
            "evaluator_enabled",
            "evaluator_commitment",
            "prior_evaluation_provider_call_count",
            "prior_evaluation_cost_microusd",
            "snapshot_age_seconds",
        } or tree_runtime_policy.get("schema_version") != (
            "research_lab.git_tree_runtime_policy.v1"
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree runtime policy fields are invalid"
            )
        tree_policy = TreePolicy.from_mapping(
            _mapping(tree_runtime_policy.get("policy"), "Git-tree policy")
        )
        if tree_policy.mode != "active":
            raise AutoresearchExecutorV2Error(
                "measured Git-tree policy is not active"
            )
        evaluator_commitment = _mapping(
            tree_runtime_policy.get("evaluator_commitment"),
            "Git-tree evaluator commitment",
        )
        if set(evaluator_commitment) != {
            "schema_version",
            "snapshot_manifest_hash",
            "snapshot_ready_hash",
            "dev_set_hash",
            "dev_set_size",
            "snapshot_bank_hash",
            "snapshot_bank_size",
            "daily_bank_hash",
            "selection_manifest_hash",
            "selection_seed_hash",
            "miner_direction_hash",
            "benchmark_date",
            "benchmark_bundle_id",
            "benchmark_bundle_hash",
            "rolling_window_hash",
            "private_model_manifest_hash",
            "champion_image_digest",
            "source_commit",
            "model_config_hash",
            "provider_model_ids",
            "miss_policy",
            "score_version",
            "evaluation_timeout_seconds",
            "live_max_icps_per_node",
            "live_max_provider_calls",
            "live_cap_microusd",
            "minimum_evidence_retention_days",
        }:
            raise AutoresearchExecutorV2Error(
                "Git-tree evaluator commitment fields are invalid"
            )
        for field in (
            "snapshot_manifest_hash",
            "snapshot_ready_hash",
            "dev_set_hash",
            "snapshot_bank_hash",
            "daily_bank_hash",
            "selection_manifest_hash",
            "selection_seed_hash",
            "miner_direction_hash",
            "benchmark_bundle_hash",
            "rolling_window_hash",
            "private_model_manifest_hash",
            "model_config_hash",
        ):
            _hash(evaluator_commitment.get(field), field)
        provider_model_ids = evaluator_commitment.get("provider_model_ids")
        if (
            evaluator_commitment.get("schema_version")
            != "research_lab.git_tree_evaluator_commitment.v2"
            or evaluator_commitment.get("miss_policy") != "strict"
            or int(evaluator_commitment.get("dev_set_size") or 0)
            != tree_policy.live_max_icps_per_node
            or int(evaluator_commitment.get("snapshot_bank_size") or 0)
            < tree_policy.live_max_icps_per_node
            or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}",
                str(evaluator_commitment.get("benchmark_date") or ""),
            )
            or not re.fullmatch(
                r"private_benchmark:[0-9a-f]{64}",
                str(evaluator_commitment.get("benchmark_bundle_id") or ""),
            )
            or not str(evaluator_commitment.get("champion_image_digest") or "")
            or not str(evaluator_commitment.get("source_commit") or "")
            or not str(evaluator_commitment.get("score_version") or "")
            or not isinstance(provider_model_ids, list)
            or any(not isinstance(item, str) or not item for item in provider_model_ids)
            or int(evaluator_commitment.get("evaluation_timeout_seconds") or 0) < 30
            or int(evaluator_commitment.get("live_max_icps_per_node") or 0)
            != tree_policy.live_max_icps_per_node
            or int(evaluator_commitment.get("live_max_provider_calls") or 0)
            != tree_policy.live_max_provider_calls
            or int(evaluator_commitment.get("live_cap_microusd") or 0)
            != tree_policy.live_cap_microusd
            or int(
                evaluator_commitment.get("minimum_evidence_retention_days")
                or 0
            )
            != tree_policy.evidence_retention_days
            or tree_runtime_policy.get("evaluator_enabled") is not True
            or payload.get("dev_evaluator_enabled") is not True
            or int(settings.get("max_candidates") or 0) != tree_policy.max_nodes
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree evaluator commitment is not production eligible"
            )
        evaluation_timeout_seconds = int(
            evaluator_commitment.get("evaluation_timeout_seconds") or 0
        )
        try:
            required_final_context_seconds = (
                tree_policy.required_final_context_seconds(
                    evaluation_timeout_seconds
                )
            )
        except GitTreeContractError as exc:
            raise AutoresearchExecutorV2Error(
                "Git-tree deadline cannot contain final evaluation and handoff"
            ) from exc
        if required_final_context_seconds >= int(settings.get("max_seconds") or 0):
            raise AutoresearchExecutorV2Error(
                "Git-tree runtime max_seconds cannot contain final evaluation "
                "and handoff"
            )
        prior_evaluation_provider_call_count = tree_runtime_policy.get(
            "prior_evaluation_provider_call_count"
        )
        prior_evaluation_cost_microusd = tree_runtime_policy.get(
            "prior_evaluation_cost_microusd"
        )
        if (
            isinstance(prior_evaluation_provider_call_count, bool)
            or not isinstance(prior_evaluation_provider_call_count, int)
            or prior_evaluation_provider_call_count < 0
            or prior_evaluation_provider_call_count
            > tree_policy.live_max_provider_calls
            or isinstance(prior_evaluation_cost_microusd, bool)
            or not isinstance(prior_evaluation_cost_microusd, int)
            or prior_evaluation_cost_microusd < 0
            or prior_evaluation_cost_microusd > tree_policy.live_cap_microusd
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree prior evaluation usage is invalid"
            )
        try:
            snapshot_age_seconds = float(
                tree_runtime_policy.get("snapshot_age_seconds")
            )
        except (TypeError, ValueError) as exc:
            raise AutoresearchExecutorV2Error(
                "Git-tree snapshot age is invalid"
            ) from exc
        if (
            not math.isfinite(snapshot_age_seconds)
            or snapshot_age_seconds < 0
            or snapshot_age_seconds > 14 * 86400
        ):
            raise AutoresearchExecutorV2Error(
                "Git-tree snapshot is stale or invalid"
            )
        return {
            "schema_version": AUTORESEARCH_REQUEST_SCHEMA_VERSION,
            "run_id": run_id,
            "ticket": _mapping(payload.get("ticket"), "ticket"),
            "artifact": _mapping(payload.get("artifact"), "artifact"),
            "component_registry": component_registry,
            "component_registry_receipt_hash": component_root,
            "provider_catalog_result": catalog_result,
            "provider_catalog_receipt_hash": catalog_root,
            "provider_outcome_receipt_hash": provider_outcome_root,
            "benchmark_public_summary": _mapping(
                payload.get("benchmark_public_summary"), "benchmark_public_summary"
            ),
            "model_id": model_id,
            "model_doc": _mapping(payload.get("model_doc"), "model_doc"),
            "budget_context": budget_context,
            "requested_loop_count": requested_loop_count,
            "resume_state": _mapping(payload.get("resume_state"), "resume_state"),
            "loop_settings": settings,
            "source_bundle": _mapping(payload.get("source_bundle"), "source_bundle"),
            "probe_private_window_term_hashes": sorted(str(item) for item in hashes),
            "provider_outcome_digest": dict(
                provider_outcome_result["provider_outcome_digest"]
            ),
            "dev_evaluator_enabled": bool(payload["dev_evaluator_enabled"]),
            "openrouter_context": {**openrouter, "privacy_proof_doc": privacy_doc},
            "expected_event_state_hash": _hash(
                payload.get("expected_event_state_hash"),
                "expected_event_state_hash",
            ),
        }

    def _event_sink(
        self,
        *,
        context: ExecutionContextV2,
        expected_event_state_hash: str,
    ) -> Callable[[AutoResearchLoopEvent], Any]:
        state = {"hash": expected_event_state_hash, "sequence": 0}
        lock = threading.Lock()

        async def sink(event: AutoResearchLoopEvent) -> None:
            event_doc = _event_document(event)
            event_hash = sha256_json(event_doc)

            def validate(value: Mapping[str, Any]) -> Dict[str, Any]:
                if set(value) != {
                    "schema_version",
                    "event_hash",
                    "event_sequence",
                    "next_state_hash",
                }:
                    raise AutoresearchExecutorV2Error("event append response fields are invalid")
                if value.get("schema_version") != HOST_EVENT_RESULT_SCHEMA_VERSION:
                    raise AutoresearchExecutorV2Error("event append response schema is invalid")
                if value.get("event_hash") != event_hash:
                    raise AutoresearchExecutorV2Error("event append response hash differs")
                if value.get("event_sequence") != state["sequence"]:
                    raise AutoresearchExecutorV2Error("event append sequence differs")
                return {
                    **dict(value),
                    "next_state_hash": _hash(
                        value.get("next_state_hash"), "next event state hash"
                    ),
                }

            with lock:
                current_state = state["hash"]
                response = await asyncio.to_thread(
                    context.execute_host_operation,
                    operation=HOST_APPEND_EVENT,
                    payload={
                        "event": event_doc,
                        "event_hash": event_hash,
                        "event_sequence": state["sequence"],
                    },
                    expected_state_hash=current_state,
                    timeout_seconds=120,
                    response_validator=validate,
                )
                stage_purpose = _EVENT_PURPOSES.get(event.event_type)
                if stage_purpose:
                    context.record_stage(
                        purpose=stage_purpose,
                        input_root=current_state,
                        output_root=event_hash,
                    )
                state["hash"] = response["next_state_hash"]
                state["sequence"] += 1

        return sink

    def _pause_checker(
        self, context: ExecutionContextV2
    ) -> Callable[[], Any]:
        sequence = {"value": 0}

        async def check() -> bool:
            expected = sha256_json(
                {"job_id": context.job_id, "pause_check": sequence["value"]}
            )

            def validate(value: Mapping[str, Any]) -> Dict[str, Any]:
                if set(value) != {"schema_version", "paused", "state_hash"}:
                    raise AutoresearchExecutorV2Error("pause response fields are invalid")
                if value.get("schema_version") != HOST_PAUSE_RESULT_SCHEMA_VERSION:
                    raise AutoresearchExecutorV2Error("pause response schema is invalid")
                if not isinstance(value.get("paused"), bool):
                    raise AutoresearchExecutorV2Error("pause response value is invalid")
                return {**dict(value), "state_hash": _hash(value.get("state_hash"), "pause state hash")}

            response = await asyncio.to_thread(
                context.execute_host_operation,
                operation=HOST_CHECK_PAUSE,
                payload={"check_sequence": sequence["value"]},
                expected_state_hash=expected,
                timeout_seconds=30,
                response_validator=validate,
            )
            sequence["value"] += 1
            return bool(response["paused"])

        return check

    def _dev_evaluator(
        self, context: ExecutionContextV2
    ) -> Callable[[BuiltCodeEditCandidate], Any]:
        async def evaluate(candidate: BuiltCodeEditCandidate) -> Mapping[str, Any]:
            candidate_doc = _candidate_document(candidate)
            candidate_hash = sha256_json(candidate_doc)

            def validate(value: Mapping[str, Any]) -> Dict[str, Any]:
                if set(value) != {
                    "schema_version",
                    "candidate_hash",
                    "result",
                    "receipt_graph",
                }:
                    raise AutoresearchExecutorV2Error("dev-eval response fields are invalid")
                if value.get("schema_version") != HOST_DEV_EVAL_RESULT_SCHEMA_VERSION:
                    raise AutoresearchExecutorV2Error("dev-eval response schema is invalid")
                if value.get("candidate_hash") != candidate_hash:
                    raise AutoresearchExecutorV2Error("dev-eval candidate binding differs")
                result = _mapping(value.get("result"), "dev-eval result")
                graph = _mapping(value.get("receipt_graph"), "dev-eval receipt graph")
                validate_receipt_graph(
                    graph,
                    required_purposes=("research_lab.candidate_test.v2",),
                    require_boot_attestation_verification=(
                        self._scoring_graph_verifier is not None
                    ),
                    boot_attestation_verifier=(
                        (lambda identity: self._scoring_graph_verifier(identity))
                        if self._scoring_graph_verifier is not None
                        else None
                    ),
                )
                root = next(
                    item
                    for item in graph["receipts"]
                    if item["receipt_hash"] == graph["root_receipt_hash"]
                )
                if root.get("role") != "gateway_scoring":
                    raise AutoresearchExecutorV2Error("dev-eval receipt role is invalid")
                if root.get("output_root") != sha256_bytes(
                    canonical_json(result).encode("utf-8")
                ):
                    raise AutoresearchExecutorV2Error("dev-eval result commitment differs")
                context.record_external_receipt_graph(graph)
                return {
                    "schema_version": HOST_DEV_EVAL_RESULT_SCHEMA_VERSION,
                    "candidate_hash": candidate_hash,
                    "result": result,
                    "receipt_graph": graph,
                }

            response = await asyncio.to_thread(
                context.execute_host_operation,
                operation=HOST_DEV_EVALUATE,
                payload={"candidate": candidate_doc, "candidate_hash": candidate_hash},
                expected_state_hash=candidate_hash,
                timeout_seconds=1800,
                response_validator=validate,
            )
            return {
                **dict(response["result"]),
                "receipt_root": str(
                    response["receipt_graph"].get("root_receipt_hash") or ""
                ),
            }

        async def evaluate_cohort(
            candidates: Sequence[BuiltCodeEditCandidate],
            *,
            remaining_tree_budget_microusd: int | None = None,
        ) -> Mapping[str, Any]:
            ordered = sorted(tuple(candidates), key=lambda item: str(item.node_id))
            if not ordered:
                raise AutoresearchExecutorV2Error("dev-eval cohort is empty")
            candidate_docs = [_candidate_document(item) for item in ordered]
            candidate_hashes = [sha256_json(item) for item in candidate_docs]
            if (
                remaining_tree_budget_microusd is None
                or isinstance(remaining_tree_budget_microusd, bool)
                or int(remaining_tree_budget_microusd) < 0
            ):
                raise AutoresearchExecutorV2Error(
                    "dev-eval cohort remaining tree budget is invalid"
                )
            remaining_budget = int(remaining_tree_budget_microusd)
            request_hash = sha256_json(
                {
                    "schema_version": "leadpoet.autoresearch_dev_eval_cohort_request.v4",
                    "candidate_hashes": candidate_hashes,
                    "remaining_tree_budget_microusd": remaining_budget,
                }
            )

            def validate(value: Mapping[str, Any]) -> Dict[str, Any]:
                required = {
                    "schema_version",
                    "cohort_request_hash",
                    "cohort_hash",
                    "evaluation_mode",
                    "overlay_hash",
                    "provider_call_count",
                    "settled_cost_microusd",
                    "results",
                }
                if set(value) != required:
                    raise AutoresearchExecutorV2Error(
                        "dev-eval cohort response fields are invalid"
                    )
                if (
                    value.get("schema_version")
                    != HOST_DEV_EVAL_COHORT_RESULT_SCHEMA_VERSION
                    or value.get("cohort_request_hash") != request_hash
                ):
                    raise AutoresearchExecutorV2Error(
                        "dev-eval cohort response binding differs"
                    )
                cohort_hash = _hash(value.get("cohort_hash"), "cohort hash")
                overlay_hash = _hash(value.get("overlay_hash"), "overlay hash")
                mode = str(value.get("evaluation_mode") or "")
                if mode not in {"replay", "hybrid"}:
                    raise AutoresearchExecutorV2Error(
                        "dev-eval cohort mode is invalid"
                    )
                provider_call_count = value.get("provider_call_count")
                settled_cost = value.get("settled_cost_microusd")
                if (
                    isinstance(provider_call_count, bool)
                    or not isinstance(provider_call_count, int)
                    or provider_call_count < 0
                    or isinstance(settled_cost, bool)
                    or not isinstance(settled_cost, int)
                    or settled_cost < 0
                ):
                    raise AutoresearchExecutorV2Error(
                        "dev-eval cohort accounting is invalid"
                    )
                rows = value.get("results")
                if not isinstance(rows, list) or len(rows) != len(ordered):
                    raise AutoresearchExecutorV2Error(
                        "dev-eval cohort result count differs"
                    )
                expected_by_node = {
                    str(candidate.node_id): candidate_hashes[index]
                    for index, candidate in enumerate(ordered)
                }
                normalized_rows: list[dict[str, Any]] = []
                seen_nodes: set[str] = set()
                required_purpose = (
                    "research_lab.candidate_hybrid_test.v2"
                    if mode == "hybrid"
                    else "research_lab.candidate_test.v2"
                )
                for row in rows:
                    if not isinstance(row, Mapping) or set(row) != {
                        "node_id",
                        "candidate_hash",
                        "result",
                        "receipt_graph",
                        "evaluation_metadata",
                    }:
                        raise AutoresearchExecutorV2Error(
                            "dev-eval cohort row fields are invalid"
                        )
                    node_id = str(row.get("node_id") or "")
                    if (
                        node_id not in expected_by_node
                        or node_id in seen_nodes
                        or row.get("candidate_hash") != expected_by_node[node_id]
                    ):
                        raise AutoresearchExecutorV2Error(
                            "dev-eval cohort candidate binding differs"
                        )
                    seen_nodes.add(node_id)
                    result = _mapping(row.get("result"), "dev-eval cohort result")
                    graph = _mapping(
                        row.get("receipt_graph"), "dev-eval cohort receipt graph"
                    )
                    metadata = _mapping(
                        row.get("evaluation_metadata"),
                        "dev-eval cohort metadata",
                    )
                    if set(metadata) != {
                        "evaluation_mode",
                        "overlay_hash",
                        "cohort_hash",
                        "provider_call_count",
                        "settled_cost_microusd",
                        "evaluation_plan",
                    } or any(
                        metadata.get(name) != expected
                        for name, expected in (
                            ("evaluation_mode", mode),
                            ("overlay_hash", overlay_hash),
                            ("cohort_hash", cohort_hash),
                            ("provider_call_count", provider_call_count),
                            ("settled_cost_microusd", settled_cost),
                        )
                    ):
                        raise AutoresearchExecutorV2Error(
                            "dev-eval cohort metadata differs"
                        )
                    if not isinstance(metadata.get("evaluation_plan"), Mapping):
                        raise AutoresearchExecutorV2Error(
                            "dev-eval cohort plan is invalid"
                        )
                    validate_receipt_graph(
                        graph,
                        required_purposes=(required_purpose,),
                        require_boot_attestation_verification=(
                            self._scoring_graph_verifier is not None
                        ),
                        boot_attestation_verifier=(
                            (lambda identity: self._scoring_graph_verifier(identity))
                            if self._scoring_graph_verifier is not None
                            else None
                        ),
                    )
                    root = next(
                        item
                        for item in graph["receipts"]
                        if item["receipt_hash"] == graph["root_receipt_hash"]
                    )
                    if (
                        root.get("role") != "gateway_scoring"
                        or root.get("output_root")
                        != sha256_bytes(canonical_json(result).encode("utf-8"))
                    ):
                        raise AutoresearchExecutorV2Error(
                            "dev-eval cohort result commitment differs"
                        )
                    context.record_external_receipt_graph(graph)
                    normalized_rows.append(
                        {
                            "node_id": node_id,
                            "candidate_hash": expected_by_node[node_id],
                            "result": result,
                            "receipt_graph": graph,
                            "evaluation_metadata": metadata,
                        }
                    )
                if seen_nodes != set(expected_by_node):
                    raise AutoresearchExecutorV2Error(
                        "dev-eval cohort omitted a candidate"
                    )
                return {
                    "schema_version": HOST_DEV_EVAL_COHORT_RESULT_SCHEMA_VERSION,
                    "cohort_request_hash": request_hash,
                    "cohort_hash": cohort_hash,
                    "evaluation_mode": mode,
                    "overlay_hash": overlay_hash,
                    "provider_call_count": provider_call_count,
                    "settled_cost_microusd": settled_cost,
                    "results": normalized_rows,
                }

            response = await asyncio.to_thread(
                context.execute_host_operation,
                operation=HOST_DEV_EVALUATE,
                payload={
                    "candidates": candidate_docs,
                    "candidate_hashes": candidate_hashes,
                    "cohort_request_hash": request_hash,
                    "remaining_tree_budget_microusd": remaining_budget,
                },
                expected_state_hash=request_hash,
                timeout_seconds=1800,
                response_validator=validate,
            )
            return {
                "schema_version": "research_lab.git_tree_eval_cohort_result.v1",
                "cohort_hash": response["cohort_hash"],
                "evaluation_mode": response["evaluation_mode"],
                "overlay_hash": response["overlay_hash"],
                "provider_call_count": response["provider_call_count"],
                "settled_cost_microusd": response["settled_cost_microusd"],
                "results": [
                    {
                        "node_id": row["node_id"],
                        "candidate_hash": row["candidate_hash"],
                        "result": {
                            **dict(row["result"]),
                            **dict(row["evaluation_metadata"]),
                            "receipt_root": str(
                                row["receipt_graph"].get("root_receipt_hash") or ""
                            ),
                        },
                    }
                    for row in response["results"]
                ],
            }

        setattr(evaluate, "evaluate_cohort", evaluate_cohort)
        return evaluate

    def _loop_model_caller(
        self,
        *,
        context: ExecutionContextV2,
        config: ResearchLabGatewayConfig,
        run_id: str,
        model_id: str,
        model_doc: Mapping[str, Any],
        openrouter_context: Mapping[str, Any],
    ) -> Callable[..., Any]:
        # Import the unchanged provider implementation only after the worker
        # module and V2 bridge have finished loading.  A module-level import
        # creates worker -> authority -> bridge -> executor -> worker and makes
        # both the hosted worker and enclave service unimportable.
        from gateway.research_lab.worker import (
            CreditBlockedHostedRunError,
            HostedResearchLabWorkerError,
            ResearchLabHostedWorker,
            _resolve_code_edit_loop_stage_model_request,
        )

        worker = ResearchLabHostedWorker(config, worker_ref="enclave:autoresearch-v2")
        worker._raw_trace_recorder = _AttestedRawTraceRecorder()

        def privacy_verifier(**kwargs: Any) -> Mapping[str, Any]:
            proof = dict(
                verify_openrouter_workspace_privacy(
                    runtime_key=str(kwargs.get("runtime_key") or ""),
                    management_key=str(kwargs.get("management_key") or ""),
                    stage=str(kwargs.get("stage") or "openrouter_call"),
                    request_policy=_mapping(
                        kwargs.get("request_policy"), "request_policy"
                    ),
                )
            )
            proof["management_key_hash"] = str(
                openrouter_context["management_credential_value_hash"]
            ).split(":", 1)[1]
            guard_proof = _mapping(
                openrouter_context["privacy_proof_doc"], "guard privacy proof"
            )
            for field in (
                "workspace_id_hash",
                "runtime_key_hash",
                "runtime_key_label_hash",
                "runtime_key_creator_user_id_hash",
                "management_key_hash",
            ):
                if proof.get(field) != guard_proof.get(field):
                    raise AutoresearchExecutorV2Error(
                        "OpenRouter privacy proof changed after guard"
                    )
            proof["proof_hash"] = canonical_hash(
                {key: value for key, value in proof.items() if key != "proof_hash"}
            )
            return proof

        def privacy_recorder(**kwargs: Any) -> None:
            proof = _mapping(kwargs.get("proof_doc"), "privacy proof event")
            proof_hash = sha256_json(proof)

            def validate(value: Mapping[str, Any]) -> Dict[str, Any]:
                if set(value) != {"schema_version", "proof_hash", "persisted"}:
                    raise AutoresearchExecutorV2Error("privacy event response fields are invalid")
                if (
                    value.get("schema_version") != HOST_ARTIFACT_RESULT_SCHEMA_VERSION
                    or value.get("proof_hash") != proof_hash
                    or value.get("persisted") is not True
                ):
                    raise AutoresearchExecutorV2Error("privacy event response binding differs")
                return dict(value)

            context.execute_host_operation(
                operation=HOST_RECORD_PRIVACY,
                payload={
                    "key_ref": str(kwargs.get("key_ref") or ""),
                    "miner_hotkey": str(kwargs.get("miner_hotkey") or ""),
                    "run_id": kwargs.get("run_id"),
                    "stage": str(kwargs.get("stage") or ""),
                    "proof_status": str(kwargs.get("proof_status") or ""),
                    "proof_doc": proof,
                    "proof_hash": proof_hash,
                },
                expected_state_hash=proof_hash,
                timeout_seconds=120,
                response_validator=validate,
            )

        async def call(
            messages: Sequence[Mapping[str, str]],
            timeout_seconds: int,
            max_tokens: int,
            call_stage: str = "code_edit_draft",
        ) -> OpenRouterCallResult:
            options = _resolve_code_edit_loop_stage_model_request(
                config,
                stage=call_stage,
                model_id=model_id,
                model_doc=model_doc,
                requested_max_tokens=max_tokens,
            )
            stage = str(options["stage"])
            stage_models = tuple(
                str(item)
                for item in options.get("model_ids", (options["model_id"],))
                if str(item).strip()
            ) or (str(options["model_id"]),)
            if stage in {"loop_planner", "plan_alignment_judge"}:
                effective_max_tokens = max(1, int(options["max_tokens"] or 0))
            else:
                effective_max_tokens = worker._auto_research_max_tokens_for_call(
                    requested_max_tokens=int(options["max_tokens"] or 0),
                    model_doc=model_doc,
                )
            last_exc: Optional[Exception] = None
            fallback_usage = []
            for index, attempt_model in enumerate(stage_models):
                try:
                    result = await worker._call_openrouter(
                        messages=messages,
                        api_key=_RUNTIME_KEY_PLACEHOLDER,
                        model_id=attempt_model,
                        reasoning_effort=str(options.get("reasoning_effort") or ""),
                        timeout_seconds=timeout_seconds,
                        max_tokens=effective_max_tokens,
                        temperature=float(options.get("temperature") or 0.0),
                        allow_non_zdr=bool(options.get("allow_non_zdr")),
                        capture_run_id=run_id,
                        capture_stage=stage,
                        privacy_key_ref=str(openrouter_context["key_ref"]),
                        privacy_miner_hotkey=str(openrouter_context["miner_hotkey"]),
                        privacy_management_key=_MANAGEMENT_KEY_PLACEHOLDER,
                        privacy_verifier=privacy_verifier,
                        privacy_event_recorder=privacy_recorder,
                        raw_trace_recorder=worker._raw_trace_recorder,
                    )
                    if fallback_usage:
                        usage = dict(result.provider_usage or {})
                        usage["model_fallback_attempts"] = fallback_usage
                        usage["model_fallback_attempt_count"] = len(fallback_usage)
                        result = OpenRouterCallResult(
                            content=result.content,
                            provider_usage=usage,
                            cost_microusd=result.cost_microusd,
                        )
                    return result
                except CreditBlockedHostedRunError:
                    raise
                except HostedResearchLabWorkerError as exc:
                    last_exc = exc
                    if index >= len(stage_models) - 1:
                        raise
                    fallback_usage.append(
                        {
                            "stage": stage,
                            "model_ref": attempt_model,
                            "error_hash": sha256_json({"error": str(exc)}),
                            "next_model_ref": stage_models[index + 1],
                        }
                    )
            if last_exc is not None:
                raise last_exc
            raise HostedResearchLabWorkerError(
                "OpenRouter stage model resolution failed"
            )

        return call
