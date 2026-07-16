"""Code-edit image candidate generation loop for hosted Research Lab runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields, replace
import inspect
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union

from gateway.research_lab.code_build import (
    CodeEditBuildError,
    CodeEditInfraFailureError,
    CodeEditImageBuildError,
    CodeEditPatchApplyError,
    CodeEditPrivateTestError,
    CodeEditBuildResult,
    CodeEditCandidateBuilder,
    attach_git_tree_lineage,
    resolve_source_inspection_requests,
)
from gateway.research_lab.git_tree_models import (
    GitTreeContractError,
    TreeChildSlot,
    TreeCheckpoint,
    TreeEvaluation,
    TreeNode,
    TreePolicy,
    TreeResult,
    build_operation_id,
    cohort_evaluation_operation_id,
    derive_tree_id,
    generation_operation_id,
    select_finalist,
    summarize_tree_evaluations,
)
from gateway.research_lab.git_tree_scheduler import (
    GitTreeScheduler,
    GitTreeSchedulerError,
    sanitized_branch_context,
)
from gateway.research_lab.autoresearch_runtime import (
    AutoResearchLoopEvent,
    AutoResearchRuntimeSettings,
    OpenRouterCallResult,
    budget_limit_microusd as _budget_limit_microusd,
    coerce_call_result as _coerce_call_result,
    estimated_call_microusd as _estimated_call_microusd,
    running_cost_ledger as _running_cost_ledger,
    runtime_settings_doc as _settings_doc,
    safe_budget_doc as _safe_budget_doc,
    would_exceed_budget as _would_exceed_budget,
)
from research_lab.axis_provenance import call_episode
from gateway.research_lab.logging_utils import safe_event_error_text
from research_lab.canonical import sha256_json
from research_lab.code_editing import (
    FORBIDDEN_CODE_EDIT_TERMS,
    CodeEditDraft,
    CodeEditSourceInspectionRequest,
    build_code_edit_auto_research_messages,
    build_code_edit_fallback_messages,
    build_code_edit_repair_messages,
    build_code_edit_source_inspection_messages,
    build_loop_direction_planner_messages,
    build_loop_direction_reference_repair_messages,
    build_plan_alignment_judge_messages,
    code_edit_plan_alignment_errors,
    loop_direction_plan_contract_errors,
    loop_direction_plan_from_mapping,
    parse_code_edit_no_viable_patch_response,
    parse_loop_direction_plan_response,
    parse_plan_alignment_judge_response,
    parse_code_edit_repair_response,
    parse_code_edit_response,
    parse_code_edit_source_inspection_response,
)
from gateway.research_lab.source_slice_context import plan_read_requests
from gateway.research_lab.source_symbol_index import (
    bind_source_references_exact,
    resolve_source_references,
    unresolved_references_from_context,
)
from gateway.research_lab.provider_evidence_proxy import (
    load_provider_registry as load_provider_registry_entries,
    load_provider_registry_with_capabilities,
)
from gateway.research_lab.provider_capabilities import (
    summary_mentions_private_capability,
    validate_candidate_provider_diff,
)
from gateway.research_lab.provider_probe import (
    ProbeBudgetState,
    resolve_provider_probe,
)
from research_lab.engine_v1 import ReflectionRecord
from research_lab.eval import PrivateModelArtifactManifest
from research_lab.eval.snapshot_store import SnapshotMiss
from research_lab.probe_catalog import ProviderProbeEndpoint, load_probe_catalog, validate_probe_catalog
from research_lab.observability.langfuse_client import (
    observation as langfuse_observation,
    run_trace_id as langfuse_run_trace_id,
    update_observation as langfuse_update_observation,
)
from research_lab.trajectory_corpus import PROTECTED_CORPUS_MARKERS


CodeEditOpenRouterCaller = Callable[
    [Sequence[Mapping[str, str]], int, int, str],
    Awaitable[Union[str, OpenRouterCallResult]],
]

logger = logging.getLogger(__name__)

_ENGINE_FLAG_TRUTHY = {"1", "true", "yes", "on"}

# Exception class names (checked by name to avoid a circular import with
# worker.py) that must always escape bug-17 stage containment: they change the
# run's funding or ownership, not just the current stage.
_STAGE_PROPAGATE_ERROR_CLASS_NAMES = frozenset(
    {
        "CreditBlockedHostedRunError",
        "HostedResearchLabClaimLost",
    }
)


def _engine_env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in _ENGINE_FLAG_TRUTHY


def _resume_restore_selected_enabled() -> bool:
    """Bug 5 kill switch: restore already-built candidates from the checkpoint on resume."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_RESUME_RESTORE_SELECTED", "true")


def _loop_provider_probes_enabled(config: Any) -> bool:
    """W4 probe_provider flag: builder config first, env fallback, default off."""

    attr = getattr(config, "loop_provider_probes_enabled", None)
    if attr is not None:
        return bool(attr)
    return _engine_env_flag("RESEARCH_LAB_LOOP_PROVIDER_PROBES", "false")


def _probe_window_guard_required() -> bool:
    """Fail-closed default: no private-window term hashes → no probes."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_PROBE_REQUIRE_WINDOW_GUARD", "true")


def _probe_snapshot_overlay_uri() -> str:
    return str(os.getenv("RESEARCH_LAB_PROBE_SNAPSHOT_OVERLAY_URI") or "").strip()


def _planner_parse_retry_enabled() -> bool:
    """Bug 16 kill switch: retry the planner once on parse failure, then fall back to plan-less mode."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_PLANNER_PARSE_RETRY", "true")


def _stage_error_containment_enabled() -> bool:
    """Bug 17 kill switch: a failed stage LLM call skips that stage/iteration instead of failing the run."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT", "true")


_UNIMPLEMENTABLE_PLAN_MARKERS = (
    "outside the allowed edit scope",
    "not listed in editable_files",
    "not in editable_files",
    "not in editable source",
    "not present in editable",
    "not present in the visible",
    "not present in visible",
    "not present in current",
    "no editable",
    "no visible",
    "no source path",
    "source path does not exist",
    "required file is not present",
    "required path is not present",
    "no sonar provider",
    "no sonar client",
    "no sonar parse",
    "no sonar response",
    "no discover_events_via_sonar",
    "no call_sonar",
    "no parse_sonar_json",
)


def _binding_plan_unimplementable_reason(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    return bool(text and any(marker in text for marker in _UNIMPLEMENTABLE_PLAN_MARKERS))


def _planner_reference_repair_enabled(config: Any) -> bool:
    return bool(getattr(config, "planner_reference_repair_enabled", True)) and int(
        getattr(config, "planner_reference_repair_max_attempts", 1) or 0
    ) > 0


def _is_editable_test_path(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/").strip().lower()
    basename = normalized.rsplit("/", 1)[-1]
    return (
        "/tests/" in f"/{normalized}"
        or basename.startswith("test_") and basename.endswith(".py")
        or basename.endswith("_test.py")
    )


def _candidate_edit_constraints(
    source_context: Any,
    *,
    config: Any,
    dev_evaluator_configured: bool,
) -> dict[str, Any]:
    editable_files = tuple(str(item) for item in getattr(source_context, "editable_files", ()) if item)
    editable_test_paths = tuple(sorted(path for path in editable_files if _is_editable_test_path(path)))
    validation_modes = ["runtime_checks"]
    if editable_test_paths:
        validation_modes.append("existing_test_files")
    return {
        "schema_version": "1.0",
        "new_files_allowed": False,
        "editable_test_path_count": len(editable_test_paths),
        "editable_test_paths": list(editable_test_paths),
        "allowed_validation_modes": validation_modes,
        "default_validation_mode": "runtime_checks",
        "runtime_checks": {
            "py_compile_changed_python_files": True,
            "private_test_command_configured": bool(str(getattr(config, "private_test_cmd", "") or "").strip()),
            "image_build_required": True,
            "dev_eval_when_enabled": bool(dev_evaluator_configured),
        },
    }


@dataclass(frozen=True)
class _PlanSourceBindingResult:
    plan_doc: dict[str, Any] | None
    errors: tuple[str, ...] = ()
    missing_references: tuple[str, ...] = ()
    invalid_references: tuple[str, ...] = ()
    ambiguous_references: tuple[str, ...] = ()


def _bind_loop_direction_plan(
    plan_doc: Mapping[str, Any] | None,
    *,
    source_context: Any,
    candidate_edit_constraints: Mapping[str, Any],
) -> _PlanSourceBindingResult:
    if not isinstance(plan_doc, Mapping):
        return _PlanSourceBindingResult(plan_doc=None)
    try:
        plan = loop_direction_plan_from_mapping(plan_doc)
    except Exception as exc:
        return _PlanSourceBindingResult(
            plan_doc=dict(plan_doc),
            errors=(f"loop_direction_plan_invalid:{safe_event_error_text(exc)}",),
        )
    errors = list(loop_direction_plan_contract_errors(plan))
    canonical_doc = plan.to_dict()
    if str(plan.schema_version or "1.0") != "1.1" or plan.no_new_safe_path:
        return _PlanSourceBindingResult(
            plan_doc=canonical_doc,
            errors=tuple(dict.fromkeys(errors)),
        )

    reference_binding = bind_source_references_exact(
        index_doc=getattr(source_context, "planner_source_index", {}),
        source_root=source_context.source_root,
        editable_files=getattr(source_context, "editable_files", ()),
        references=plan.must_inspect,
    )
    ranked_paths: list[dict[str, Any]] = []
    for raw_path in canonical_doc.get("ranked_paths", []):
        path_doc = dict(raw_path) if isinstance(raw_path, Mapping) else {}
        if _ranked_path_id(path_doc) == plan.selected_path_id:
            path_doc["must_inspect"] = list(reference_binding.normalized_references)
        ranked_paths.append(path_doc)
    canonical_doc["ranked_paths"] = ranked_paths
    canonical_doc["must_inspect"] = list(reference_binding.normalized_references)
    canonical_doc = loop_direction_plan_from_mapping(canonical_doc).to_dict()

    for reference in reference_binding.missing_references:
        errors.append(f"loop_direction_plan_reference_missing:{reference}")
    for reference in reference_binding.invalid_references:
        errors.append(f"loop_direction_plan_reference_invalid:{reference}")
    for reference in reference_binding.ambiguous_references:
        errors.append(f"loop_direction_plan_reference_ambiguous:{reference}")

    editable_files = set(str(item) for item in getattr(source_context, "editable_files", ()) if item)
    editable_test_paths = {path for path in editable_files if _is_editable_test_path(path)}
    missing_references = list(reference_binding.missing_references)
    invalid_references = list(reference_binding.invalid_references)
    if plan.validation_mode == "existing_test_files":
        if not editable_test_paths:
            errors.append("loop_direction_plan_existing_tests_unavailable")
        for path in plan.validation_paths:
            if path not in editable_files:
                errors.append(f"loop_direction_plan_validation_path_unavailable:{path}")
                if path not in missing_references:
                    missing_references.append(path)
            elif path not in editable_test_paths:
                errors.append(f"loop_direction_plan_validation_path_not_test_file:{path}")
                if path not in invalid_references:
                    invalid_references.append(path)
    elif plan.validation_paths:
        errors.append("loop_direction_plan_runtime_checks_must_not_require_validation_paths")
    return _PlanSourceBindingResult(
        plan_doc=canonical_doc,
        errors=tuple(dict.fromkeys(errors)),
        missing_references=tuple(missing_references),
        invalid_references=tuple(invalid_references),
        ambiguous_references=reference_binding.ambiguous_references,
    )


def _plan_source_feasibility_errors(
    plan_doc: Mapping[str, Any] | None,
    *,
    source_context: Any,
    candidate_edit_constraints: Mapping[str, Any],
) -> tuple[list[str], tuple[str, ...]]:
    binding = _bind_loop_direction_plan(
        plan_doc,
        source_context=source_context,
        candidate_edit_constraints=candidate_edit_constraints,
    )
    return list(binding.errors), binding.missing_references


def _ranked_path_id(path: Mapping[str, Any] | None) -> str:
    if not isinstance(path, Mapping):
        return ""
    return str(path.get("path_id") or path.get("id") or path.get("selected_path_id") or "").strip()


def _loop_plan_selected_path_id(plan_doc: Mapping[str, Any] | None) -> str:
    if not isinstance(plan_doc, Mapping):
        return ""
    return str(plan_doc.get("selected_path_id") or "").strip()


def _tree_branch_direction_plan(
    base_plan_doc: Mapping[str, Any] | None,
    *,
    root_slot_index: int,
) -> dict[str, Any] | None:
    """Bind one immutable ranked objective to one root branch.

    The planner emits an ordered set of independently viable paths. A tree
    branch owns the path at its root slot for its entire ancestry; a sibling's
    refusal or score can never mutate this document.
    """

    if not isinstance(base_plan_doc, Mapping):
        return None
    ranked_paths = base_plan_doc.get("ranked_paths")
    if not isinstance(ranked_paths, (list, tuple)):
        return None
    index = int(root_slot_index)
    if index < 0 or index >= len(ranked_paths):
        return None
    raw_path = ranked_paths[index]
    if not isinstance(raw_path, Mapping):
        return None
    path_id = _ranked_path_id(raw_path)
    required_lane = str(
        raw_path.get("lane") or raw_path.get("required_lane") or ""
    ).strip()
    required_mechanism = str(
        raw_path.get("mechanism")
        or raw_path.get("required_mechanism")
        or raw_path.get("hypothesis")
        or ""
    ).strip()
    if not path_id or not required_lane or not required_mechanism:
        return None
    candidate = dict(base_plan_doc)
    candidate["no_new_safe_path"] = False
    candidate["ranked_paths"] = [dict(raw_path)]
    candidate["selected_path_id"] = path_id
    candidate["required_lane"] = required_lane
    candidate["required_mechanism"] = required_mechanism
    for field_name in (
        "target_behavior",
        "must_inspect",
        "allowed_lanes",
        "disallowed_lanes",
        "must_not_try",
        "success_criteria",
        "novelty_requirements",
        "anti_overfit_checks",
        "validation_paths",
    ):
        candidate[field_name] = raw_path.get(field_name, [])
    for field_name in ("generalization_claim", "novelty_contrast"):
        candidate[field_name] = raw_path.get(
            field_name, base_plan_doc.get(field_name, "")
        )
    candidate["validation_mode"] = raw_path.get(
        "validation_mode", "runtime_checks"
    )
    candidate.pop("plan_hash", None)
    candidate["plan_hash"] = sha256_json(candidate)
    try:
        return loop_direction_plan_from_mapping(candidate).to_dict()
    except Exception as exc:
        logger.warning(
            "research_lab_git_tree_branch_plan_invalid "
            "selected_path_id=%s plan_hash=%s error=%s",
            path_id[:120],
            str(candidate.get("plan_hash") or "")[:80],
            safe_event_error_text(exc),
        )
        return None


def _fallback_max_target_files(config: Any) -> int:
    attr = getattr(config, "code_edit_fallback_max_target_files", None)
    if attr is not None:
        try:
            return max(1, int(attr))
        except (TypeError, ValueError):
            pass
    try:
        return max(1, int(os.getenv("RESEARCH_LAB_CODE_EDIT_FALLBACK_MAX_TARGET_FILES", "3")))
    except ValueError:
        return 3


def _judge_parse_soft_skip_enabled() -> bool:
    """Bug 21 kill switch: retry the alignment judge once on parse failure, then accept neutrally
    instead of recording a confident rejection."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_JUDGE_PARSE_SOFT_SKIP", "true")


def _within_run_memory_enabled() -> bool:
    """Feed this run's prior rejections into later iterations and dedupe rejected diff hashes."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_WITHIN_RUN_MEMORY", "true")


def _min_runtime_skip_when_selected_enabled() -> bool:
    """Skip the post-loop minimum-runtime sleep when candidates are already selected."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_MIN_RUNTIME_SKIP_WHEN_SELECTED", "true")


def _build_heartbeat_enabled() -> bool:
    """Run the docker build off the event loop and emit heartbeat loop events during it."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_BUILD_HEARTBEAT", "true")


def _reflection_emission_enabled() -> bool:
    """§9.1 item 4 kill switch: emit a mechanical ``reflection_recorded`` loop event
    after each iteration's judge/build outcome (pure capture, no extra LLM call)."""

    return _engine_env_flag("RESEARCH_LAB_REFLECTION_EMISSION_ENABLED", "true")


def _dev_eval_enabled() -> bool:
    """§6.3-1 L1 dev-eval rung flag (default ON when a dev evaluator is wired):
    score built candidates through the wired ``dev_evaluator`` seam.
    Dev scores are ranking-only within a run and strictly best-effort."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED", "true")


def _dev_snapshot_uri() -> str:
    """The frozen provider-snapshot set URI dev-eval replays against
    (``research_lab.eval.snapshot_store.SNAPSHOT_URI_ENV``). Empty = no set."""

    from gateway.research_lab.config import DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI

    return (
        os.environ.get(
            "RESEARCH_LAB_DEV_SNAPSHOT_URI",
            DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI,
        ).strip()
    )


@dataclass(frozen=True)
class BuiltCodeEditCandidate:
    draft: CodeEditDraft
    build: CodeEditBuildResult
    node_id: str
    iteration: int
    rehydration_artifact_uri: str = ""
    rehydration_artifact_hash: str = ""
    # §6.3-1: ranking-only L1 dev-eval score (None = never dev-evaluated). The
    # defaults keep pre-dev-eval checkpoints/rehydration artifacts loadable.
    dev_score: float | None = None
    dev_score_version: str = ""
    # Compact, ranking-only evaluator evidence. This deliberately excludes
    # per-ICP outputs; checkpoints need the commitment and coverage facts, not
    # hidden development examples or provider payloads.
    dev_evaluation: Mapping[str, Any] = field(default_factory=dict)
    # Private, anonymized test feedback for direct children. It never
    # contains ICP refs, hashes, company payloads, provider output, or prompts.
    dev_feedback: Mapping[str, Any] = field(default_factory=dict)
    dev_feedback_hash: str = ""
    # Tree ancestry is separate from the official promotion parent. Every
    # official candidate remains rooted at the approved active model.
    tree_id: str = ""
    tree_parent_node_id: str = "root"
    tree_root_branch_id: str = ""
    tree_depth: int = 0
    tree_child_slot: int = 0
    tree_branch_objective_path_id: str = ""
    tree_branch_objective_hash: str = ""
    tree_generation_attempt_count: int = 0
    tree_git_commit: str = ""
    tree_root_artifact_hash: str = ""
    tree_parent_artifact_hash: str = ""
    tree_parent_dev_score: float | None = None
    tree_parent_feedback_hash: str = ""
    tree_incremental_source_diff_hash: str = ""
    tree_cumulative_source_diff_hash: str = ""
    tree_composition: Mapping[str, Any] = field(default_factory=dict)
    tree_settled_cost_microusd: int = 0


def _candidate_dev_evaluation_matches_policy(
    candidate: BuiltCodeEditCandidate,
    policy: Mapping[str, Any],
) -> bool:
    evidence = candidate.dev_evaluation
    if not isinstance(evidence, Mapping) or not evidence:
        return False
    expected = policy.get("evaluator_commitment")
    actual = evidence.get("evaluator_commitment")
    commitment_matches = bool(
        isinstance(expected, Mapping)
        and isinstance(actual, Mapping)
        and dict(actual) == dict(expected)
    )
    if not commitment_matches:
        return False
    if not bool(evidence.get("eligible")):
        return candidate.dev_score is None
    score = candidate.dev_score
    return bool(score is not None and math.isfinite(float(score)))


def _finite_feedback_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _score_band(score: float | None) -> str:
    if score is None:
        return "unavailable"
    if score >= 80:
        return "strong"
    if score >= 60:
        return "adequate"
    if score > 0:
        return "weak"
    return "zero"


def _build_tree_dev_feedback(
    *,
    result: Mapping[str, Any],
    candidate: BuiltCodeEditCandidate,
    score: float,
    max_examples: int,
) -> dict[str, Any]:
    """Create bounded feedback without hidden example identities or payloads."""

    raw_rows = result.get("per_icp")
    rows = (
        list(raw_rows)
        if isinstance(raw_rows, Sequence)
        and not isinstance(raw_rows, (str, bytes, bytearray))
        else []
    )
    examples: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows[: max(0, int(max_examples))], start=1):
        row = dict(raw_row) if isinstance(raw_row, Mapping) else {}
        example_score = _finite_feedback_number(
            row.get("dev_score", row.get("score"))
        )
        failure = bool(row.get("failure_reason") or row.get("failure_class"))
        snapshot_miss = bool(row.get("snapshot_miss"))
        zero_output = bool(row.get("zero_output"))
        status = (
            "snapshot_miss"
            if snapshot_miss
            else "failed"
            if failure
            else "zero_output"
            if zero_output
            else "completed"
        )
        examples.append(
            {
                "example_number": index,
                "status": status,
                "quality_score": (
                    round(example_score, 6) if example_score is not None else None
                ),
                "quality_band": _score_band(example_score),
                "result_count": max(0, int(row.get("company_count") or 0)),
                "scored_result_count": max(
                    0, int(row.get("scored_company_count") or 0)
                ),
            }
        )
    weak_count = sum(
        1 for item in examples if item["quality_band"] in {"weak", "zero"}
    )
    zero_count = sum(1 for item in examples if item["status"] == "zero_output")
    guidance = [
        "Preserve behavior that produced adequate or strong examples.",
        "Improve the weakest anonymized examples without hard-coding example-specific values.",
    ]
    if zero_count:
        guidance.append(
            "Address empty-result behavior while preserving fit and intent constraints."
        )
    if weak_count and not zero_count:
        guidance.append(
            "Improve result quality and ranking before broadening result volume."
        )
    parent_score = _finite_feedback_number(candidate.tree_parent_dev_score)
    payload = {
        "schema_version": "research_lab.git_tree_dev_feedback.v1",
        "candidate_node_id": str(candidate.node_id)[:80],
        "tree_id": str(candidate.tree_id),
        "parent_node_id": str(candidate.tree_parent_node_id),
        "root_branch_id": str(candidate.tree_root_branch_id),
        "depth": max(0, int(candidate.tree_depth)),
        "aggregate_score": round(float(score), 6),
        "parent_score": round(parent_score, 6) if parent_score is not None else None,
        "score_change_from_parent": (
            round(float(score) - parent_score, 6)
            if parent_score is not None
            else None
        ),
        "example_count": len(examples),
        "weak_example_count": weak_count,
        "zero_output_example_count": zero_count,
        "examples": examples,
        "guidance": guidance,
        "rules": [
            "Use this only for direct children on this branch.",
            "Do not infer hidden example identities or add example-specific branches.",
            "The approved scoring and promotion workflow remains authoritative.",
        ],
    }
    return {**payload, "feedback_hash": sha256_json(payload)}


def _candidate_tree_node(
    candidate: BuiltCodeEditCandidate,
    *,
    status: str | None = None,
) -> TreeNode:
    evidence = dict(candidate.dev_evaluation or {})
    score = candidate.dev_score
    eligible = bool(evidence.get("eligible")) and score is not None
    normalized_status = str(status or ("eligible" if eligible else "ineligible"))
    if normalized_status == "evaluating":
        evaluation = None
    else:
        evaluation = TreeEvaluation(
            score=float(score) if eligible else None,
            eligible=eligible,
            reason=str(evidence.get("eligibility_reason") or "evaluation_missing"),
            execution_coverage=float(evidence.get("execution_coverage") or 0.0),
            snapshot_miss_count=max(
                0, int(evidence.get("snapshot_miss_count") or 0)
            ),
            true_miss_count=max(0, int(evidence.get("true_miss_count") or 0)),
            failure_count=max(0, int(evidence.get("failure_count") or 0)),
            zero_output_count=max(0, int(evidence.get("zero_output_count") or 0)),
            snapshot_hash=str(evidence.get("snapshot_manifest_hash") or ""),
            dev_set_hash=str(evidence.get("dev_set_hash") or ""),
            policy=str(evidence.get("miss_policy") or "strict"),
            score_version=str(candidate.dev_score_version or ""),
            receipt_root=str(evidence.get("receipt_root") or ""),
            context_hash=str(evidence.get("context_hash") or ""),
            parent_delta=(
                float(evidence["parent_delta"])
                if isinstance(evidence.get("parent_delta"), (int, float))
                and not isinstance(evidence.get("parent_delta"), bool)
                else None
            ),
            settled_cost_microusd=max(
                0, int(evidence.get("settled_cost_microusd") or 0)
            ),
            provider_call_count=max(
                0, int(evidence.get("provider_call_count") or 0)
            ),
            evaluation_mode=str(evidence.get("evaluation_mode") or "replay"),
            unclassified_error=bool(evidence.get("unclassified_error")),
            feedback=dict(candidate.dev_feedback or {}),
        )
    return TreeNode(
        tree_id=candidate.tree_id,
        node_id=candidate.node_id,
        parent_node_id=candidate.tree_parent_node_id,
        root_branch_id=candidate.tree_root_branch_id,
        depth=candidate.tree_depth,
        slot_index=candidate.tree_child_slot,
        status=normalized_status,
        branch_objective_path_id=candidate.tree_branch_objective_path_id,
        branch_objective_hash=candidate.tree_branch_objective_hash,
        generation_attempt_count=candidate.tree_generation_attempt_count,
        git_commit=candidate.tree_git_commit,
        source_tree_hash=(
            candidate.build.candidate_model_manifest.model_artifact_hash
        ),
        incremental_patch_hash=candidate.tree_incremental_source_diff_hash,
        cumulative_patch_hash=candidate.tree_cumulative_source_diff_hash,
        candidate_artifact_hash=(
            candidate.build.candidate_model_manifest.model_artifact_hash
        ),
        lineage_hash=sha256_json(
            dict(candidate.build.build_doc.get("git_tree") or {})
        ),
        complexity=(
            len(candidate.draft.target_files)
            + sum(
                1
                for line in candidate.draft.unified_diff.splitlines()
                if line.startswith(("+", "-"))
                and not line.startswith(("+++", "---"))
            )
        ),
        settled_cost_microusd=max(
            0, int(candidate.tree_settled_cost_microusd)
        ),
        evaluation=evaluation,
    )


def _apply_tree_cohort_settlement(
    candidates: Sequence[BuiltCodeEditCandidate],
) -> tuple[list[BuiltCodeEditCandidate], int, int]:
    """Allocate one shared cohort charge once across its candidate nodes."""

    ordered = sorted(tuple(candidates), key=lambda item: str(item.node_id))
    if not ordered:
        return [], 0, 0
    reported_costs = {
        max(0, int(item.dev_evaluation.get("settled_cost_microusd") or 0))
        for item in ordered
    }
    reported_calls = {
        max(0, int(item.dev_evaluation.get("provider_call_count") or 0))
        for item in ordered
    }
    if len(reported_costs) != 1 or len(reported_calls) != 1:
        raise GitTreeSchedulerError(
            "tree cohort candidates reported inconsistent shared accounting"
        )
    settled_cost_microusd = reported_costs.pop()
    provider_call_count = reported_calls.pop()
    per_node_cost, remainder = divmod(settled_cost_microusd, len(ordered))
    allocated = [
        replace(
            item,
            tree_settled_cost_microusd=(
                max(0, int(item.tree_settled_cost_microusd))
                + per_node_cost
                + (1 if index < remainder else 0)
            ),
        )
        for index, item in enumerate(ordered)
    ]
    return allocated, settled_cost_microusd, provider_call_count


class ContainedStageFailure(str):
    """Contained-failure error text that keeps the failed call's telemetry.

    trajectoryimprovements.md P3: the raw trace was already captured at the
    HTTP layer before containment; this ``str`` subclass rides the existing
    error-text plumbing unchanged while carrying ``provider_usage`` /
    ``cost_microusd`` so failure events keep the pointer and the cost ledger
    is incremented. Failed generations are high-value negative examples.
    """

    provider_usage: dict[str, Any] | None
    cost_microusd: int
    stage: str

    def __new__(
        cls,
        error_text: str,
        *,
        stage: str = "",
        provider_usage: Mapping[str, Any] | None = None,
        cost_microusd: int = 0,
    ) -> "ContainedStageFailure":
        obj = super().__new__(cls, error_text)
        obj.provider_usage = dict(provider_usage) if provider_usage else None
        obj.cost_microusd = max(0, int(cost_microusd or 0))
        obj.stage = str(stage or "")
        return obj

    def failure_usage_entries(self) -> list[dict[str, Any]]:
        """Provider-usage entries for the failure event doc (may be empty)."""
        if not self.provider_usage:
            return []
        return [
            {
                **self.provider_usage,
                "call_stage": self.provider_usage.get("call_stage") or self.stage,
                "call_outcome": "contained_failure",
            }
        ]


@dataclass(frozen=True)
class CodeEditLoopResult:
    selected_candidates: tuple[BuiltCodeEditCandidate, ...]
    iterations_completed: int
    stop_reason: str
    elapsed_seconds: float
    estimated_cost_usd: float
    actual_openrouter_cost_usd: float
    actual_openrouter_cost_microusd: int
    openrouter_call_count: int
    tree_result: TreeResult
    provider_usage: tuple[dict[str, Any], ...] = ()
    status: str = "completed"
    checkpoint_doc: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        status = str(self.status or "").strip().lower()
        if status not in {"completed", "paused", "failed"}:
            raise GitTreeContractError("code-edit loop result status is invalid")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "selected_candidates", tuple(self.selected_candidates))
        if self.tree_result.status != status:
            raise GitTreeContractError("loop and tree result statuses differ")
        if not isinstance(self.checkpoint_doc, Mapping):
            raise GitTreeContractError("loop result requires a tree checkpoint")
        if (
            self.checkpoint_doc.get("git_tree_checkpoint")
            != self.tree_result.checkpoint.to_dict()
        ):
            raise GitTreeContractError(
                "loop result checkpoint differs from its tree result"
            )
        if status == "completed":
            if len(self.selected_candidates) != 1:
                raise GitTreeContractError(
                    "completed loop requires exactly one selected candidate"
                )
            selected = self.selected_candidates[0]
            selected_nodes = [
                node
                for node in self.tree_result.nodes
                if node.node_id == self.tree_result.selected_node_id
            ]
            if (
                selected.node_id != self.tree_result.selected_node_id
                or selected.tree_id != self.tree_result.tree_id
                or len(selected_nodes) != 1
                or _candidate_tree_node(selected) != selected_nodes[0]
            ):
                raise GitTreeContractError(
                    "selected candidate differs from the selected tree node"
                )
        elif self.selected_candidates or self.tree_result.selected_node_id:
            raise GitTreeContractError(
                "paused or failed loop cannot expose a selected candidate"
            )

    def cost_ledger(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "status": self.status if self.status in {"paused", "completed", "failed"} else "completed",
            "total_usd": round(self.actual_openrouter_cost_usd, 6),
            "actual_openrouter_cost_usd": round(self.actual_openrouter_cost_usd, 6),
            "actual_openrouter_cost_microusd": int(self.actual_openrouter_cost_microusd),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "openrouter_call_count": self.openrouter_call_count,
            "iterations_completed": self.iterations_completed,
            "stop_reason": self.stop_reason,
        }


@dataclass
class CodeEditLoopEngine:
    settings: AutoResearchRuntimeSettings
    call_openrouter: CodeEditOpenRouterCaller
    event_sink: Any
    builder: CodeEditCandidateBuilder
    # §6.3-1 dev-eval seam: an optional caller-wired evaluator that receives one
    # built candidate and returns a ``DevEvalResult.to_dict()``-shaped mapping
    # (the engine reads ``aggregate_dev_score``/``dev_score`` +
    # ``dev_score_version``). None (the default) means built candidates stay
    # unscored even when ``RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED`` is on: the
    # engine deliberately does not construct a docker dev runner itself — the
    # worker wires a container runner (``snapshot_store.container_replay_env``
    # + ``dev_eval.evaluate_dev``) here in a later wave.
    dev_evaluator: Callable[[BuiltCodeEditCandidate], Awaitable[Mapping[str, Any]]] | None = None
    # W4 probe query guard: hashed private-window ICP/company terms (see
    # provider_probe.hash_private_window_terms). Plaintext window terms are
    # never held on the engine; empty means only the forbidden-term screen
    # applies. The worker wires this per run alongside the private window.
    probe_private_window_term_hashes: frozenset[str] = frozenset()
    # V2 supplies a measured coordinator resolver.  Legacy host execution
    # leaves this unset and retains the existing evidence-proxy path.
    probe_evidence_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    # W2: sanitized per-parent/day provider-outcome digest
    # (provider_outcome_digest.build_provider_outcome_digest), wired by the
    # worker from already-recorded truth. None keeps prompts byte-identical
    # to pre-W2 behavior.
    provider_outcome_digest: Mapping[str, Any] | None = None
    # V2 enclave seam for the existing private S3 artifact operations. None
    # preserves the original boto3 behavior; the measured role supplies a
    # signed host-operation adapter.
    artifact_io: Any | None = None
    # Git is a host primitive. The measured executor receives only committed
    # results through this adapter; it never shells out to Git inside the EIF.
    tree_repository: Any | None = None
    # V2 supplies authenticated, receipt-bound provider inputs. None preserves
    # the existing host loaders and their exact continuity behavior.
    provider_registry_loader: Callable[[], tuple[list[Any], Any]] | None = None
    provider_probe_catalog_loader: Callable[[], list[Any]] | None = None
    # Set by run() so stage/build spans can attach to the run's deterministic
    # Langfuse trace (run_trace_id(run_id)) without threading run_id through
    # every stage-call signature. One engine instance serves one run at a time.
    _langfuse_run_id: str = field(default="", init=False, repr=False)
    _tree_policy_doc: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _active_tree_scheduler: GitTreeScheduler | None = field(
        default=None, init=False, repr=False
    )
    _active_tree_policy: TreePolicy | None = field(default=None, init=False, repr=False)
    _active_tree_id: str = field(default="", init=False, repr=False)
    _active_tree_root_git_commit: str = field(default="", init=False, repr=False)

    async def _tree_repository_call(self, method_name: str, **kwargs: Any) -> Any:
        repository = self.tree_repository
        if repository is None:
            raise GitTreeSchedulerError("measured Git-tree repository is not wired")
        method = getattr(repository, method_name, None)
        if method is None or not callable(method):
            raise GitTreeSchedulerError(
                f"measured Git-tree repository lacks {method_name}"
            )
        if inspect.iscoroutinefunction(method):
            return await method(**kwargs)
        return await asyncio.to_thread(method, **kwargs)

    async def _maybe_dev_eval_candidate(
        self,
        candidate: BuiltCodeEditCandidate,
        *,
        run_id: str,
        remaining_seconds: float | None = None,
        precomputed_response: Mapping[str, Any] | None = None,
    ) -> BuiltCodeEditCandidate:
        """§6.3-1: attach a ranking-only dev score to a just-built candidate.

        Runs only when ``RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED`` is on AND a
        frozen snapshot set is configured (``RESEARCH_LAB_DEV_SNAPSHOT_URI``)
        AND a ``dev_evaluator`` seam is wired. Dev scores rank candidates
        within this run and are never promotion evidence. Strictly
        best-effort: any evaluator failure or malformed result logs and
        returns the candidate unscored — dev-eval must never fail a run that
        already built an image.
        """
        policy = dict(self._tree_policy_doc or {})
        commitment = dict(policy.get("evaluator_commitment") or {})

        def _ineligible(reason: str, **details: Any) -> BuiltCodeEditCandidate:
            evidence = {
                "schema_version": "research_lab.git_tree_evaluation.v1",
                "eligible": False,
                "eligibility_reason": str(reason)[:160],
                "evaluator_commitment": commitment,
                **details,
            }
            return replace(
                candidate,
                dev_score=None,
                dev_score_version="",
                dev_evaluation=evidence,
                dev_feedback={},
                dev_feedback_hash="",
            )

        try:
            if not bool(policy.get("evaluator_enabled")):
                return candidate
            if not _dev_eval_enabled():
                return _ineligible("dev_eval_kill_switch_disabled")
            if not _dev_snapshot_uri():
                return _ineligible("snapshot_uri_missing")
            if self.dev_evaluator is None:
                logger.warning(
                    "research_lab_loop_dev_eval_unwired run_id=%s node_id=%s",
                    run_id,
                    str(candidate.node_id)[:80],
                )
                return _ineligible("dev_evaluator_unwired")

            measured_policy = (
                dict(policy.get("policy") or {})
                if isinstance(policy.get("policy"), Mapping)
                else {}
            )
            reserve = max(
                0,
                int(
                    measured_policy.get("finalization_reserve_seconds")
                    or 120
                ),
            )
            configured_timeout = max(
                1,
                int(commitment.get("evaluation_timeout_seconds") or 300),
            )
            timeout = configured_timeout
            if remaining_seconds is not None and precomputed_response is None:
                available = float(remaining_seconds) - reserve
                if available <= 0:
                    return _ineligible(
                        "insufficient_deadline_for_evaluation",
                        remaining_seconds=round(float(remaining_seconds), 3),
                        finalization_reserve_seconds=reserve,
                    )
                timeout = max(1, min(configured_timeout, int(available)))

            response = (
                dict(precomputed_response)
                if precomputed_response is not None
                else await asyncio.wait_for(self.dev_evaluator(candidate), timeout=timeout)
            )
            if not isinstance(response, Mapping):
                raise ValueError("dev_evaluator returned a non-mapping result")
            envelope = dict(response)
            result = envelope.get("result") if isinstance(envelope.get("result"), Mapping) else envelope
            result = dict(result)
            graph = envelope.get("receipt_graph")
            receipt_root = (
                str(graph.get("root_receipt_hash") or "")
                if isinstance(graph, Mapping)
                else str(result.get("receipt_root") or "")
            )
            raw_score = result.get("aggregate_dev_score", result.get("dev_score"))
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError("dev_evaluator result carried no numeric aggregate_dev_score")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError("dev_evaluator returned a non-finite aggregate_dev_score")
            version = str(result.get("dev_score_version") or "")
            result_manifest_hash = str(result.get("snapshot_manifest_hash") or "")
            result_dev_set_hash = str(result.get("dev_set_hash") or "")
            expected_manifest_hash = str(
                commitment.get("snapshot_manifest_hash") or ""
            )
            expected_dev_set_hash = str(commitment.get("dev_set_hash") or "")
            expected_icp_count = int(commitment.get("dev_set_size") or 0)
            measured_icp_count = int(
                measured_policy.get("live_max_icps_per_node") or 0
            )
            evaluation_mode = str(result.get("evaluation_mode") or "replay")
            overlay_hash = str(result.get("overlay_hash") or "")
            cohort_hash = str(result.get("cohort_hash") or "")
            score_commitment_doc = {
                "schema_version": (
                    "research_lab.git_tree_dev_score_commitment.v1"
                ),
                "dev_score_version": version,
                "dev_set_hash": result_dev_set_hash,
                "snapshot_manifest_hash": result_manifest_hash,
                "miss_policy": str(result.get("miss_policy") or ""),
                "evaluation_mode": evaluation_mode,
                "overlay_hash": overlay_hash,
                "cohort_hash": cohort_hash,
            }
            expected_score_commitment = sha256_json(score_commitment_doc)
            dev_feedback = _build_tree_dev_feedback(
                result=result,
                candidate=candidate,
                score=score,
                max_examples=expected_icp_count,
            )
            checks = {
                "evaluator_reported_eligible": bool(result.get("eligible")),
                "dev_set_size_matches_config": expected_icp_count > 0
                and expected_icp_count == measured_icp_count
                and int(result.get("icp_count") or 0) == expected_icp_count,
                "full_execution_coverage": float(result.get("execution_coverage") or 0.0) == 1.0,
                "all_icps_scored": int(result.get("scored_icp_count") or 0)
                == expected_icp_count,
                "no_snapshot_misses": int(result.get("snapshot_miss_count") or 0) == 0,
                "no_true_misses": int(result.get("true_miss_count") or 0) == 0,
                "no_failures": int(result.get("failure_count") or 0) == 0,
                "strict_miss_policy": str(result.get("miss_policy") or "") == "strict",
                "evaluation_mode_valid": evaluation_mode in {"replay", "hybrid"},
                "overlay_commitment_valid": bool(
                    re.fullmatch(r"sha256:[0-9a-f]{64}", overlay_hash)
                )
                and (
                    evaluation_mode == "hybrid"
                    or overlay_hash == sha256_json({})
                ),
                "cohort_commitment_present": bool(
                    re.fullmatch(r"sha256:[0-9a-f]{64}", cohort_hash)
                ),
                "snapshot_manifest_matches": bool(expected_manifest_hash)
                and result_manifest_hash == expected_manifest_hash,
                "dev_set_matches": bool(expected_dev_set_hash)
                and result_dev_set_hash == expected_dev_set_hash,
                "score_version_present": bool(version),
                "score_commitment_matches": str(
                    result.get("score_commitment") or ""
                ) == expected_score_commitment,
                "configured_anonymized_feedback_examples": int(
                    dev_feedback.get("example_count") or 0
                ) == expected_icp_count,
            }
            evaluation_doc = {
                "schema_version": "research_lab.git_tree_evaluation.v1",
                "eligible": all(checks.values()),
                "eligibility_reason": (
                    "eligible"
                    if all(checks.values())
                    else next(name for name, passed in checks.items() if not passed)
                ),
                "execution_coverage": float(result.get("execution_coverage") or 0.0),
                "icp_count": int(result.get("icp_count") or 0),
                "scored_icp_count": int(result.get("scored_icp_count") or 0),
                "snapshot_miss_count": int(result.get("snapshot_miss_count") or 0),
                "true_miss_count": int(result.get("true_miss_count") or 0),
                "failure_count": int(result.get("failure_count") or 0),
                "zero_output_count": int(result.get("zero_output_count") or 0),
                "miss_policy": str(result.get("miss_policy") or ""),
                "snapshot_manifest_hash": result_manifest_hash,
                "dev_set_hash": result_dev_set_hash,
                "score_commitment": str(result.get("score_commitment") or ""),
                "receipt_root": receipt_root,
                "evaluation_mode": evaluation_mode,
                "overlay_hash": overlay_hash,
                "cohort_hash": cohort_hash,
                "evaluation_plan": dict(result.get("evaluation_plan") or {}),
                "provider_call_count": max(
                    0, int(result.get("provider_call_count") or 0)
                ),
                "settled_cost_microusd": max(
                    0, int(result.get("settled_cost_microusd") or 0)
                ),
                "context_hash": sha256_json(
                    {
                        "schema_version": "research_lab.git_tree_eval_context.v1",
                        "evaluator_commitment": commitment,
                        "snapshot_manifest_hash": result_manifest_hash,
                        "dev_set_hash": result_dev_set_hash,
                        "score_version": version,
                        "miss_policy": str(result.get("miss_policy") or ""),
                        "evaluation_mode": evaluation_mode,
                        "overlay_hash": overlay_hash,
                        "cohort_hash": cohort_hash,
                    }
                ),
                "parent_delta": dev_feedback.get("score_change_from_parent"),
                "evaluator_commitment": commitment,
                "checks": checks,
            }
            if not evaluation_doc["eligible"]:
                logger.warning(
                    "research_lab_loop_dev_eval_ineligible run_id=%s node_id=%s reason=%s",
                    run_id,
                    str(candidate.node_id)[:80],
                    evaluation_doc["eligibility_reason"],
                )
                return replace(
                    candidate,
                    dev_score=None,
                    dev_score_version=version,
                    dev_evaluation=evaluation_doc,
                    dev_feedback={},
                    dev_feedback_hash="",
                )
            logger.info(
                "research_lab_loop_dev_eval_scored run_id=%s node_id=%s dev_score=%s dev_score_version=%s",
                run_id,
                str(candidate.node_id)[:80],
                round(score, 6),
                version[:80],
            )
            return replace(
                candidate,
                dev_score=score,
                dev_score_version=version,
                dev_evaluation=evaluation_doc,
                dev_feedback=dev_feedback,
                dev_feedback_hash=str(dev_feedback.get("feedback_hash") or ""),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(exc, SnapshotMiss):
                reason = "snapshot_miss"
                unclassified = False
            elif isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                reason = "evaluation_timeout"
                unclassified = False
            elif type(exc).__name__ in {
                "DevEvalRunnerError",
                "DevEvalError",
                "DevSnapshotStoreError",
            }:
                reason = f"evaluator_failure:{type(exc).__name__}"
                unclassified = False
            else:
                reason = f"evaluator_error:{type(exc).__name__}"
                unclassified = True
            logger.warning(
                "research_lab_loop_dev_eval_failed run_id=%s node_id=%s error=%s",
                run_id,
                str(candidate.node_id)[:80],
                str(exc)[:200],
            )
            return _ineligible(
                reason,
                unclassified_error=unclassified,
                error_hash=sha256_json({"error_type": type(exc).__name__, "error": str(exc)}),
            )

    async def _maybe_dev_eval_cohort(
        self,
        candidates: Sequence[BuiltCodeEditCandidate],
        *,
        run_id: str,
        remaining_seconds: float | None = None,
        remaining_tree_budget_microusd: int | None = None,
        post_evaluation_reserve_seconds: int | None = None,
    ) -> list[BuiltCodeEditCandidate]:
        """Evaluate one complete scheduler round under one frozen context."""

        ordered = sorted(tuple(candidates), key=lambda item: str(item.node_id))
        if not ordered:
            return []
        policy = dict(self._tree_policy_doc or {})
        commitment = dict(policy.get("evaluator_commitment") or {})

        def failed_candidate(
            candidate: BuiltCodeEditCandidate,
            *,
            reason: str,
            error: Exception | None = None,
            details: Mapping[str, Any] | None = None,
        ) -> BuiltCodeEditCandidate:
            evidence = {
                "schema_version": "research_lab.git_tree_evaluation.v1",
                "eligible": False,
                "eligibility_reason": str(reason)[:160],
                "evaluator_commitment": commitment,
                "cohort_evaluation": True,
                **dict(details or {}),
            }
            if error is not None:
                evidence.update(
                    {
                        "unclassified_error": type(error).__name__
                        not in {
                            "DevEvalRunnerError",
                            "DevEvalError",
                            "DevSnapshotStoreError",
                            "TimeoutError",
                        },
                        "error_hash": sha256_json(
                            {
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        ),
                    }
                )
            return replace(
                candidate,
                dev_score=None,
                dev_score_version="",
                dev_evaluation=evidence,
                dev_feedback={},
                dev_feedback_hash="",
            )

        if (
            not bool(policy.get("evaluator_enabled"))
            or not _dev_eval_enabled()
            or not _dev_snapshot_uri()
            or self.dev_evaluator is None
        ):
            return [
                await self._maybe_dev_eval_candidate(
                    item,
                    run_id=run_id,
                    remaining_seconds=remaining_seconds,
                )
                for item in ordered
            ]
        evaluate_cohort = getattr(self.dev_evaluator, "evaluate_cohort", None)
        if not callable(evaluate_cohort):
            logger.warning(
                "research_lab_tree_dev_eval_cohort_unwired run_id=%s node_count=%s",
                run_id,
                len(ordered),
            )
            return [
                failed_candidate(item, reason="dev_evaluator_cohort_unwired")
                for item in ordered
            ]
        measured_policy = (
            dict(policy.get("policy") or {})
            if isinstance(policy.get("policy"), Mapping)
            else {}
        )
        reserve = max(
            0,
            int(
                post_evaluation_reserve_seconds
                if post_evaluation_reserve_seconds is not None
                else measured_policy.get("finalization_reserve_seconds") or 120
            ),
        )
        configured_timeout = max(
            1, int(commitment.get("evaluation_timeout_seconds") or 300)
        )
        timeout = configured_timeout
        if remaining_seconds is not None:
            available = float(remaining_seconds) - reserve
            if available <= 0:
                return [
                    failed_candidate(
                        item,
                        reason="insufficient_deadline_for_cohort_evaluation",
                        details={
                            "remaining_seconds": round(
                                float(remaining_seconds), 3
                            ),
                            "post_evaluation_reserve_seconds": reserve,
                            "configured_evaluation_timeout_seconds": (
                                configured_timeout
                            ),
                        },
                    )
                    for item in ordered
                ]
            timeout = max(1, min(configured_timeout, int(available)))
        try:
            response = await asyncio.wait_for(
                evaluate_cohort(
                    ordered,
                    remaining_tree_budget_microusd=(
                        max(0, int(remaining_tree_budget_microusd))
                        if remaining_tree_budget_microusd is not None
                        else None
                    ),
                ),
                timeout=timeout,
            )
            if not isinstance(response, Mapping):
                raise ValueError("dev-evaluator cohort returned a non-mapping result")
            rows = response.get("results")
            if not isinstance(rows, Sequence) or isinstance(
                rows, (str, bytes, bytearray)
            ):
                raise ValueError("dev-evaluator cohort carried no result rows")
            rows_by_node: dict[str, Mapping[str, Any]] = {}
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("dev-evaluator cohort row is invalid")
                node_id = str(row.get("node_id") or "")
                if not node_id or node_id in rows_by_node:
                    raise ValueError("dev-evaluator cohort node binding is invalid")
                rows_by_node[node_id] = row
            if set(rows_by_node) != {str(item.node_id) for item in ordered}:
                raise ValueError("dev-evaluator cohort omitted or added a node")
            evaluated: list[BuiltCodeEditCandidate] = []
            for item in ordered:
                row = rows_by_node[str(item.node_id)]
                raw_result = row.get("result")
                if not isinstance(raw_result, Mapping):
                    raise ValueError("dev-evaluator cohort candidate result is invalid")
                metadata = row.get("evaluation_metadata")
                merged_result = {
                    **dict(raw_result),
                    **(
                        dict(metadata)
                        if isinstance(metadata, Mapping)
                        else {}
                    ),
                }
                graph = row.get("receipt_graph")
                precomputed = (
                    {
                        "result": merged_result,
                        "receipt_graph": dict(graph),
                    }
                    if isinstance(graph, Mapping)
                    else merged_result
                )
                evaluated.append(
                    await self._maybe_dev_eval_candidate(
                        item,
                        run_id=run_id,
                        remaining_seconds=remaining_seconds,
                        precomputed_response=precomputed,
                    )
                )
            context_hashes = {
                str(item.dev_evaluation.get("context_hash") or "")
                for item in evaluated
                if item.dev_evaluation.get("eligible")
            }
            if len(context_hashes) > 1:
                raise ValueError("dev-evaluator cohort contexts differ")
            return evaluated
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "research_lab_tree_dev_eval_cohort_failed run_id=%s node_count=%s error=%s",
                run_id,
                len(ordered),
                str(exc)[:200],
            )
            raise

    async def _rehydrate_candidate_reference(
        self,
        *,
        uri: str,
        expected_hash: str,
        run_id: str,
    ) -> BuiltCodeEditCandidate:
        if not str(uri).startswith("s3://"):
            raise ValueError("rehydration_artifact_uri_invalid")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(expected_hash or "")):
            raise ValueError("rehydration_artifact_hash_missing")
        bucket, object_key = _parse_s3_uri(uri)
        if self.artifact_io is not None:
            payload = await _artifact_io_read_json(
                self.artifact_io,
                uri,
                content_hash=expected_hash,
            )
        else:
            def _get() -> dict[str, Any]:
                import boto3  # type: ignore

                body = boto3.client("s3").get_object(
                    Bucket=bucket, Key=object_key
                )["Body"].read()
                return json.loads(body.decode("utf-8"))

            payload = await asyncio.to_thread(_get)
        stored_hash = str(payload.get("loop_candidate_artifact_hash") or "")
        if stored_hash != expected_hash:
            raise ValueError("rehydration_artifact_hash_mismatch")
        candidate = replace(
            _rehydrated_candidate_from_artifact_payload(payload),
            rehydration_artifact_uri=uri,
            rehydration_artifact_hash=expected_hash,
        )
        if candidate.tree_depth <= 0:
            raise ValueError("rehydrated candidate is not a Git-tree node")
        restore_source_context = getattr(
            self.builder,
            "restore_rehydrated_candidate_source_context",
            None,
        )
        if callable(restore_source_context):
            try:
                await asyncio.to_thread(
                    restore_source_context,
                    candidate=candidate,
                )
            except Exception as exc:
                logger.warning(
                    "research_lab_tree_resume_source_restore_failed "
                    "run_id=%s node_id=%s error=%s",
                    run_id,
                    candidate.node_id[:80],
                    type(exc).__name__,
                )
                raise
        return candidate

    async def _restore_selected_from_resume(
        self,
        *,
        resume: Mapping[str, Any],
        run_id: str,
        artifact: PrivateModelArtifactManifest,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
    ) -> list[BuiltCodeEditCandidate]:
        """Rehydrate already-built candidates from checkpoint artifacts (bug #5).

        Each committed build must have a content-addressed private artifact.
        Missing or tampered state is fatal to resume; substituting another node
        or the root would change the measured search tree.
        """
        del openrouter_calls, estimated_cost, actual_cost_microusd  # event-free restore
        required_node_ids = {
            node.node_id
            for node in (
                self._active_tree_scheduler.nodes
                if self._active_tree_scheduler is not None
                else ()
            )
            if node.status in {"evaluating", "eligible", "ineligible"}
        }
        summaries = resume.get("selected_candidates")
        if not isinstance(summaries, Sequence):
            if required_node_ids:
                raise GitTreeSchedulerError(
                    "tree checkpoint is missing candidate artifact summaries"
                )
            return []
        restored: list[BuiltCodeEditCandidate] = []
        for summary in summaries:
            if not isinstance(summary, Mapping):
                continue
            uri = str(summary.get("rehydration_artifact_uri") or "")
            expected_hash = str(summary.get("rehydration_artifact_hash") or "")
            if not uri.startswith("s3://"):
                if str(summary.get("node_id") or "") in required_node_ids:
                    raise GitTreeSchedulerError(
                        "tree checkpoint candidate artifact URI is missing"
                    )
                continue
            try:
                candidate = await self._rehydrate_candidate_reference(
                    uri=uri,
                    expected_hash=expected_hash,
                    run_id=run_id,
                )
                expected_node = next(
                    (
                        node
                        for node in (
                            self._active_tree_scheduler.nodes
                            if self._active_tree_scheduler is not None
                            else ()
                        )
                        if node.node_id == candidate.node_id
                    ),
                    None,
                )
                if expected_node is None:
                    raise ValueError("rehydrated candidate tree node is missing")
                if expected_node.status == "evaluating":
                    if candidate.dev_evaluation or candidate.dev_score is not None:
                        raise ValueError(
                            "pending tree candidate contains uncommitted evaluation"
                        )
                    restored_node = _candidate_tree_node(
                        candidate, status="evaluating"
                    )
                else:
                    if not _candidate_dev_evaluation_matches_policy(
                        candidate, self._tree_policy_doc
                    ):
                        raise ValueError(
                            "rehydrated candidate evaluator commitment differs"
                        )
                    restored_node = _candidate_tree_node(candidate)
                if restored_node != expected_node:
                    raise ValueError(
                        "rehydrated candidate differs from tree checkpoint"
                    )
                restored.append(candidate)
            except Exception as exc:
                logger.warning(
                    "research_lab_loop_candidate_restore_failed run_id=%s node_id=%s error=%s",
                    run_id,
                    str(summary.get("node_id") or "")[:80],
                    str(exc)[:200],
                )
                raise GitTreeSchedulerError(
                    "tree checkpoint candidate could not be restored"
                ) from exc
        restored_ids = {candidate.node_id for candidate in restored}
        if restored_ids != required_node_ids:
            raise GitTreeSchedulerError(
                "tree checkpoint candidate artifacts are incomplete"
            )
        if restored:
            logger.info(
                "research_lab_loop_candidates_restored run_id=%s count=%s parent=%s",
                run_id,
                len(restored),
                artifact.model_artifact_hash[:24],
            )
        return restored

    async def _call_stage_contained(
        self,
        messages: Sequence[Mapping[str, str]],
        timeout_seconds: int,
        max_tokens: int,
        stage: str,
    ) -> tuple[OpenRouterCallResult | None, "ContainedStageFailure | None"]:
        """Run one stage LLM call with bug-17 error containment.

        Returns ``(result, None)`` on success and ``(None, failure)`` on a
        contained failure — the caller skips the stage/iteration and the run
        keeps whatever it already built. Credit blocks and claim losses always
        propagate (they change run ownership/funding, not just this stage), as
        does everything when containment is disabled.

        P3: the failure is a ``str`` subclass still carrying the failed call's
        ``provider_usage`` / ``cost_microusd`` when the exception had them, so
        failure events keep the raw-trace pointer instead of dropping it on
        the containment floor.

        Langfuse mirror: each stage call emits one ``generation`` observation
        on the run's deterministic trace — stage name, outcome, and cost only
        (never prompt/response content; the canonical raw trace already lives
        in SSE-KMS S3). A Langfuse failure yields a None span and the stage
        proceeds untouched.
        """
        run_id = self._langfuse_run_id
        with langfuse_observation(
            f"research_lab.loop_stage.{stage}",
            as_type="generation",
            metadata={"run_id": run_id, "stage": stage},
            trace_id=langfuse_run_trace_id(run_id),
            sample_seed=run_id or None,
        ) as stage_obs:
            try:
                raw = await self.call_openrouter(messages, timeout_seconds, max_tokens, stage)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if any(
                    base.__name__ in _STAGE_PROPAGATE_ERROR_CLASS_NAMES
                    for base in type(exc).__mro__
                ):
                    raise
                if not _stage_error_containment_enabled():
                    raise
                lost_cost_microusd = max(0, int(getattr(exc, "cost_microusd", 0) or 0))
                failure_usage = getattr(exc, "provider_usage", None)
                logger.warning(
                    "research_lab_loop_stage_call_contained stage=%s lost_cost_microusd=%s error=%s",
                    stage,
                    lost_cost_microusd,
                    str(exc)[:200],
                )
                langfuse_update_observation(
                    stage_obs,
                    output={
                        "call_outcome": "contained_failure",
                        "error_type": type(exc).__name__,
                        "lost_cost_microusd": lost_cost_microusd,
                    },
                )
                return None, ContainedStageFailure(
                    _diagnostic_text(f"{type(exc).__name__}: {exc}", limit=300),
                    stage=stage,
                    provider_usage=failure_usage if isinstance(failure_usage, Mapping) else None,
                    cost_microusd=lost_cost_microusd,
                )
            result = _coerce_call_result(raw)
            langfuse_update_observation(
                stage_obs,
                output={
                    "call_outcome": "ok",
                    "cost_microusd": int(getattr(result, "cost_microusd", 0) or 0),
                    "content_length": len(result.content or ""),
                },
            )
            return result, None

    async def _build_candidate_with_heartbeat(
        self,
        *,
        draft: CodeEditDraft,
        artifact: PrivateModelArtifactManifest,
        run_id: str,
        candidate_index: int,
        source_context: Any,
        node_id: str,
        iteration: int,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        build_timeout_seconds: int | None = None,
    ) -> CodeEditBuildResult:
        """Run the docker build off the event loop with liveness heartbeats.

        The synchronous build used to block the event loop for up to the full
        build timeout with no loop events — exactly the no-loop-event window
        the stale-claim guard requeues (Chain E). Heartbeat events keep the run
        visibly alive during long builds.

        Langfuse mirror: one ``candidate_build`` span on the run trace whose
        duration is the wall-clock build; metadata is node/iteration refs
        only. Build exceptions propagate through it unchanged.
        """
        with langfuse_observation(
            "research_lab.loop_stage.candidate_build",
            metadata={
                "run_id": run_id,
                "node_id": node_id,
                "iteration": iteration,
                "candidate_index": candidate_index,
            },
            trace_id=langfuse_run_trace_id(run_id),
            sample_seed=run_id or None,
        ) as build_obs:
            build_result = await self._build_candidate_inner(
                draft=draft,
                artifact=artifact,
                run_id=run_id,
                candidate_index=candidate_index,
                source_context=source_context,
                node_id=node_id,
                iteration=iteration,
                elapsed=elapsed,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                build_timeout_seconds=build_timeout_seconds,
            )
            langfuse_update_observation(
                build_obs,
                output={
                    "call_outcome": "ok",
                    "source_diff_hash": getattr(build_result, "source_diff_hash", None),
                },
            )
            return build_result

    async def _build_candidate_inner(
        self,
        *,
        draft: CodeEditDraft,
        artifact: PrivateModelArtifactManifest,
        run_id: str,
        candidate_index: int,
        source_context: Any,
        node_id: str,
        iteration: int,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        build_timeout_seconds: int | None = None,
    ) -> CodeEditBuildResult:
        build_kwargs = {
            "draft": draft,
            "parent_artifact": artifact,
            "run_id": run_id,
            "candidate_index": candidate_index,
            "source_context": source_context,
        }
        try:
            build_parameters = inspect.signature(self.builder.build).parameters
        except (TypeError, ValueError):
            build_parameters = {}
        if build_timeout_seconds is not None and (
            "timeout_seconds" in build_parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in build_parameters.values()
            )
        ):
            build_kwargs["timeout_seconds"] = max(
                1, int(build_timeout_seconds)
            )
        if not _build_heartbeat_enabled():
            return self.builder.build(**build_kwargs)
        source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="candidate_build_started",
                loop_status="running",
                elapsed_seconds=elapsed(),
                node_id=node_id,
                cost_ledger=_running_cost_ledger(
                    openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_started"
                ),
                event_doc={
                    "iteration": iteration,
                    "candidate_index": candidate_index,
                    "status": "started",
                    "source_diff_hash": source_diff_hash,
                },
            )
        )
        task = asyncio.create_task(
            asyncio.to_thread(
                self.builder.build,
                **build_kwargs,
            )
        )
        heartbeat_index = 0
        try:
            while not task.done():
                done, _pending = await asyncio.wait({task}, timeout=30.0)
                if done:
                    break
                heartbeat_index += 1
                try:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_started",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "candidate_build_heartbeat",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "candidate_index": candidate_index,
                                "status": "heartbeat",
                                "heartbeat_index": heartbeat_index,
                                "source_diff_hash": source_diff_hash,
                            },
                        )
                    )
                except Exception as exc:
                    # A failed heartbeat write must never fail the build itself.
                    logger.warning(
                        "research_lab_git_tree_build_heartbeat_emit_failed "
                        "node_id=%s heartbeat_index=%s error=%s",
                        str(node_id or "")[:120],
                        heartbeat_index,
                        safe_event_error_text(exc),
                    )
            return await task
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _prepare_parent_source_context_with_heartbeat(
        self,
        *,
        run_id: str,
        artifact: PrivateModelArtifactManifest,
        workspace_dir: Path,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
    ) -> Any:
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="source_inspection_requested",
                loop_status="running",
                elapsed_seconds=elapsed(),
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "parent_image_source_prepare_started",
                ),
                event_doc={
                    "operation": "parent_image_source_prepare",
                    "status": "started",
                    "run_id": run_id,
                    "parent_image_digest_hash": sha256_json({"image_digest": artifact.image_digest}),
                },
            )
        )
        task = asyncio.create_task(
            asyncio.to_thread(
                self.builder.prepare_parent_source_context,
                parent_artifact=artifact,
                workspace_dir=workspace_dir,
            )
        )
        heartbeat_index = 0
        try:
            while not task.done():
                done, _pending = await asyncio.wait({task}, timeout=30.0)
                if done:
                    break
                heartbeat_index += 1
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="source_inspection_requested",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "parent_image_source_prepare_heartbeat",
                        ),
                        event_doc={
                            "operation": "parent_image_source_prepare",
                            "status": "heartbeat",
                            "heartbeat_index": heartbeat_index,
                            "run_id": run_id,
                            "parent_image_digest_hash": sha256_json({"image_digest": artifact.image_digest}),
                        },
                    )
                )
            source_context = await task
        except Exception as exc:
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="source_inspection_failed",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "parent_image_source_prepare_failed",
                    ),
                    event_doc={
                        "operation": "parent_image_source_prepare",
                        "status": "failed",
                        "run_id": run_id,
                        "parent_image_digest_hash": sha256_json({"image_digest": artifact.image_digest}),
                        "error": safe_event_error_text(exc),
                        "error_hash": sha256_json({"error": str(exc)}),
                    },
                )
            )
            raise
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="source_inspection_resolved",
                loop_status="running",
                elapsed_seconds=elapsed(),
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "parent_image_source_prepare_completed",
                ),
                event_doc={
                    "operation": "parent_image_source_prepare",
                    "status": "completed",
                    "run_id": run_id,
                    "source_mode": source_context.source_mode,
                    "source_tree_hash": source_context.source_tree_hash,
                    "parent_image_digest_hash": source_context.parent_image_digest_hash,
                    "extracted_top_level_paths": list(source_context.top_level_paths),
                    "editable_file_count": len(source_context.editable_files),
                    "symbol_index_hash": str(source_context.planner_source_index.get("index_hash") or ""),
                    "symbol_count": int(source_context.planner_source_index.get("symbol_count") or 0),
                    "import_count": int(source_context.planner_source_index.get("import_count") or 0),
                    "symbol_index_truncated": bool(source_context.planner_source_index.get("truncated")),
                },
            )
        )
        return source_context

    async def run(
        self,
        *,
        run_id: str,
        ticket: Mapping[str, Any],
        artifact: PrivateModelArtifactManifest,
        component_registry: Mapping[str, Any],
        benchmark_public_summary: Mapping[str, Any],
        model_id: str,
        budget_context: Mapping[str, Any],
        requested_loop_count: int,
        resume_state: Mapping[str, Any] | None = None,
        should_pause: Any | None = None,
    ) -> CodeEditLoopResult:
        start = time.monotonic()
        settings = self.settings.normalized()
        root_artifact = artifact
        raw_tree_policy_doc = budget_context.get("tree_policy")
        if not isinstance(raw_tree_policy_doc, Mapping):
            raise GitTreeSchedulerError("authoritative tree policy is missing")
        self._tree_policy_doc = dict(raw_tree_policy_doc)
        measured_policy = self._tree_policy_doc.get("policy")
        if not isinstance(measured_policy, Mapping):
            raise GitTreeSchedulerError("measured tree policy is missing")
        tree_policy = TreePolicy.from_mapping(measured_policy)
        if tree_policy.mode != "active":
            raise GitTreeSchedulerError("tree engine cannot run while tree mode is off")
        evaluator_commitment = dict(
            self._tree_policy_doc.get("evaluator_commitment") or {}
        )
        evaluation_timeout_seconds = max(
            1,
            int(evaluator_commitment.get("evaluation_timeout_seconds") or 300),
        )
        final_context_reserve_seconds = (
            tree_policy.required_final_context_seconds(
                evaluation_timeout_seconds
            )
        )
        settings = replace(
            settings,
            max_iterations=tree_policy.max_nodes,
            max_candidates=tree_policy.max_nodes,
            max_seconds=min(settings.max_seconds, tree_policy.deadline_seconds),
        ).normalized()
        if final_context_reserve_seconds >= settings.max_seconds:
            raise GitTreeSchedulerError(
                "tree runtime cannot contain final evaluation and handoff"
            )
        tree_id = derive_tree_id(
            run_id=run_id,
            root_artifact_hash=root_artifact.model_artifact_hash,
            policy=tree_policy,
        )
        selected: list[BuiltCodeEditCandidate] = []
        resume = dict(resume_state or {})
        raw_tree_checkpoint = resume.get("git_tree_checkpoint")
        if isinstance(raw_tree_checkpoint, Mapping):
            checkpoint = TreeCheckpoint.from_mapping(raw_tree_checkpoint)
            if (
                checkpoint.tree_id != tree_id
                or checkpoint.root_artifact_hash
                != root_artifact.model_artifact_hash
                or checkpoint.policy != tree_policy
            ):
                raise GitTreeSchedulerError(
                    "resume checkpoint tree authority differs from this run"
                )
            tree_scheduler = GitTreeScheduler.restore(
                tree_id=tree_id,
                policy=tree_policy,
                nodes=checkpoint.nodes,
                planned_slots=checkpoint.planned_slots,
            )
        else:
            tree_scheduler = GitTreeScheduler(tree_id=tree_id, policy=tree_policy)
        self._active_tree_scheduler = tree_scheduler
        self._active_tree_policy = tree_policy
        self._active_tree_id = tree_id
        iteration = max(0, int(resume.get("iterations_completed") or 0))
        # Langfuse continuity: every observation this run emits (stage calls,
        # builds, and later the scoring worker's private-eval span) attaches
        # to the deterministic trace run_trace_id(run_id). The marker span
        # below names the trace and records start/resume; it closes
        # immediately so pauses/requeues never strand an open span.
        self._langfuse_run_id = str(run_id or "")
        with langfuse_observation(
            "research_lab.loop_run",
            metadata={
                "run_id": str(run_id or ""),
                "ticket_id": str(ticket.get("ticket_id") or ""),
                "resumed": bool(resume),
                "iteration": iteration,
            },
            trace_id=langfuse_run_trace_id(str(run_id or "")),
            sample_seed=str(run_id or "") or None,
        ):
            pass
        openrouter_calls = max(0, int(resume.get("openrouter_call_count") or 0))
        estimated_cost = max(0.0, float(resume.get("estimated_cost_usd") or 0.0))
        actual_cost_microusd = max(0, int(resume.get("actual_openrouter_cost_microusd") or 0))
        provider_usage: list[dict[str, Any]] = [
            dict(item) for item in resume.get("provider_usage", []) if isinstance(item, Mapping)
        ]
        elapsed_offset = max(0.0, float(resume.get("elapsed_seconds") or 0.0))
        elapsed = lambda: elapsed_offset + (time.monotonic() - start)
        budget_limit_microusd = _budget_limit_microusd(budget_context)
        budget_limit_microusd = tree_policy.effective_billable_cap(
            budget_limit_microusd
        )
        if budget_limit_microusd <= 0:
            raise GitTreeSchedulerError(
                "Git-tree autoresearch requires a positive funded budget"
            )
        tree_evaluation_cost_microusd = max(
            0,
            int(
                self._tree_policy_doc.get(
                    "prior_evaluation_cost_microusd", 0
                )
                or 0
            ),
        )
        tree_evaluation_provider_call_count = max(
            0,
            int(
                self._tree_policy_doc.get(
                    "prior_evaluation_provider_call_count", 0
                )
                or 0
            ),
        )
        if tree_evaluation_cost_microusd > budget_limit_microusd:
            raise GitTreeSchedulerError(
                "settled tree evaluation cost exceeds the funded budget"
            )
        budget_limit_microusd -= tree_evaluation_cost_microusd
        built_candidate_total = max(0, int(resume.get("built_candidate_count") or 0))
        finalization_reserve_reached = False
        provider_capabilities = None
        provider_capability_summary: dict[str, Any] | None = None
        effective_provider_entries: list[Any] = []
        if bool(getattr(self.builder.config, "provider_capability_catalog_enabled", True)):
            try:
                registry_loader = (
                    self.provider_registry_loader
                    or load_provider_registry_with_capabilities
                )
                effective_provider_entries, provider_capabilities = (
                    await asyncio.to_thread(registry_loader)
                )
                if provider_capabilities.private_snapshot_loaded:
                    provider_capability_summary = provider_capabilities.prompt_summary()
            except Exception as exc:
                logger.warning(
                    "research_lab_provider_capability_loop_load_failed run_id=%s error=%s",
                    str(run_id or ""),
                    str(exc)[:200],
                )
        # W4: per-run provider-probe state. Caps persist across iterations and
        # resume (probe accounting restored from the resume checkpoint).
        probes_enabled = _loop_provider_probes_enabled(self.builder.config)
        if probes_enabled and not self.probe_private_window_term_hashes and _probe_window_guard_required():
            # Fail closed: without the hashed private-window term set the query
            # guard can only screen forbidden terms — not good enough to let a
            # loop reach providers. The worker wires the hashes per run.
            logger.warning(
                "research_lab_probe_guard_window_terms_missing run_id=%s probes disabled for this run",
                str(run_id or ""),
            )
            probes_enabled = False
        probe_catalog: list[Any] = []
        probe_budget = None
        if probes_enabled:
            try:
                if self.provider_probe_catalog_loader is not None:
                    probe_catalog = await asyncio.to_thread(
                        self.provider_probe_catalog_loader
                    )
                elif provider_capabilities is not None and provider_capabilities.private_snapshot_loaded:
                    probe_catalog = [
                        ProviderProbeEndpoint.from_mapping(item)
                        for entry in effective_provider_entries
                        for item in entry.probe_endpoints
                    ]
                    probe_errors = validate_probe_catalog(probe_catalog) if probe_catalog else []
                    if probe_errors:
                        raise ValueError("invalid private probe catalog: " + "; ".join(probe_errors[:5]))
                else:
                    probe_catalog = load_probe_catalog()
                    try:
                        from gateway.research_lab.source_add_catalog import (
                            load_provisioned_source_rows_sync,
                            probe_endpoints_from_provisioned_rows,
                        )

                        provisioned_probes = await asyncio.to_thread(
                            lambda: probe_endpoints_from_provisioned_rows(load_provisioned_source_rows_sync())
                        )
                        existing_probe_ids = {entry.endpoint_id for entry in probe_catalog}
                        probe_catalog.extend(
                            entry for entry in provisioned_probes if entry.endpoint_id not in existing_probe_ids
                        )
                    except Exception as exc:
                        logger.warning("research_lab_source_add_probe_catalog_extend_failed error=%s", str(exc)[:200])
            except Exception as exc:
                logger.warning("research_lab_probe_catalog_load_failed error=%s", str(exc)[:200])
                probes_enabled = False
            if probes_enabled:
                probe_budget = ProbeBudgetState(
                    max_probes=max(0, int(getattr(self.builder.config, "loop_probe_max_probes", 4))),
                    max_cost_microusd=max(
                        0, int(getattr(self.builder.config, "loop_probe_max_cost_microusd", 250_000))
                    ),
                    probes_used=max(0, int(resume.get("probe_count") or 0)),
                    cost_used_microusd=max(0, int(resume.get("probe_cost_microusd") or 0)),
                )
        probe_registry_base_urls: dict[str, str] = {}
        if probes_enabled:
            try:
                registry_entries = effective_provider_entries or load_provider_registry_entries()
                probe_registry_base_urls = {
                    entry.id: entry.base_url for entry in registry_entries
                }
            except Exception as exc:
                logger.warning("research_lab_probe_registry_load_failed error=%s", str(exc)[:200])
        # Tree generation always carries only branch-local attempts and parent
        # feedback. Exact diff hashes may be deduplicated globally without
        # exposing one sibling's source or evaluator details to another.
        within_run_memory_active = True
        all_rejected_diff_hashes: set[str] = set()
        branch_rejected_diff_hashes: dict[str, set[str]] = {}
        branch_rejections: dict[str, list[dict[str, Any]]] = {}
        branch_dev_scores: dict[str, list[dict[str, Any]]] = {}
        branch_dev_best_scores: dict[str, float] = {}
        current_branch_id = ""
        current_branch_context_doc: dict[str, Any] | None = None
        # §9.5 / §9.4 cross-run context (both flag-gated OFF by default; filled
        # best-effort after the loop_started emission, injected into prompts only).
        retrieved_lessons_doc: dict[str, Any] | None = None
        cell_yield_priors_doc: dict[str, Any] | None = None

        def _record_within_run_rejection(
            *,
            stage: str,
            reason: str,
            iteration_index: int,
            draft: CodeEditDraft | None = None,
            diff_hash: str | None = None,
        ) -> None:
            if not within_run_memory_active:
                return
            resolved_hash = diff_hash or (
                sha256_json({"unified_diff": draft.unified_diff}) if draft is not None else ""
            )
            if resolved_hash:
                all_rejected_diff_hashes.add(resolved_hash)
                branch_rejected_diff_hashes.setdefault(current_branch_id, set()).add(
                    resolved_hash
                )
            rejections = branch_rejections.setdefault(current_branch_id, [])
            rejections.append(
                {
                    "iteration": int(iteration_index),
                    "stage": str(stage)[:80],
                    "reason": _memory_safe_text(reason),
                    "lane": str(draft.lane if draft is not None else "")[:80],
                    "plan_path_id": str(draft.plan_path_id if draft is not None else "")[:120],
                    "target_files": list(draft.target_files)[:10] if draft is not None else [],
                    "unified_diff_hash": resolved_hash,
                    "status": "rejected",
                }
            )
            del rejections[:-25]

        # Ranking-only development scores are kept per branch for generation
        # context. The scheduler, not an environment-controlled plateau rule,
        # decides when the committed tree frontier is exhausted.
        def _record_dev_score(
            *, iteration_index: int, node_id: str, score: float, version: str
        ) -> None:
            prior_best = branch_dev_best_scores.get(current_branch_id)
            branch_dev_best_scores[current_branch_id] = (
                score if prior_best is None else max(prior_best, score)
            )
            records = branch_dev_scores.setdefault(current_branch_id, [])
            records.append(
                {
                    "iteration": int(iteration_index),
                    "node_id": str(node_id)[:80],
                    "dev_score": round(float(score), 6),
                    "dev_score_version": str(version)[:120],
                }
            )
            del records[:-25]

        def _within_run_memory_doc() -> dict[str, Any] | None:
            if not within_run_memory_active:
                return None
            within_run_rejections = branch_rejections.get(current_branch_id, [])
            rejected_diff_hashes = branch_rejected_diff_hashes.get(
                current_branch_id, set()
            )
            dev_score_records = branch_dev_scores.get(current_branch_id, [])
            branch_best_score = branch_dev_best_scores.get(current_branch_id)
            if (
                not within_run_rejections
                and not dev_score_records
                and current_branch_context_doc is None
            ):
                return None
            memory_doc = {
                "schema_version": "1.0",
                "note": (
                    "Rejections recorded earlier in this run. Do not propose a diff identical to a "
                    "rejected one; address the recorded rejection reason instead."
                ),
                "rejected_attempt_count": len(within_run_rejections),
                "rejected_diff_hashes": sorted(rejected_diff_hashes)[:50],
                "recent_rejections": [dict(item) for item in within_run_rejections[-10:]],
            }
            if dev_score_records:
                # §6.3-1: dev scores are ranking-only feedback for later drafts in
                # THIS run; they are never promotion evidence.
                memory_doc["dev_scores"] = {
                    "note": (
                        "Ranking-only dev scores (frozen snapshot-replay eval) for candidates "
                        "already built in this run; higher is better. Aim the next draft at "
                        "beating best_dev_score. Never promotion evidence."
                    ),
                    "best_dev_score": (
                        round(float(branch_best_score), 6)
                        if branch_best_score is not None
                        else None
                    ),
                    "recent_scores": [dict(item) for item in dev_score_records[-10:]],
                }
            if current_branch_context_doc is not None:
                memory_doc["git_tree_branch"] = dict(current_branch_context_doc)
            return memory_doc

        def _memory_budget_context(
            base: dict[str, Any], *, include_lessons: bool = False
        ) -> dict[str, Any]:
            memory_doc = _within_run_memory_doc()
            merged = dict(base)
            if memory_doc is not None:
                merged["within_run_memory"] = memory_doc
            if include_lessons and retrieved_lessons_doc is not None:
                merged["retrieved_lessons"] = retrieved_lessons_doc
            return merged

        source_tmp = tempfile.TemporaryDirectory(prefix="research-lab-parent-image-source-")

        def _cleanup_source_tmp() -> None:
            nonlocal source_tmp
            if source_tmp is None:
                return
            source_tmp.cleanup()
            source_tmp = None

        try:
            source_context = await self._prepare_parent_source_context_with_heartbeat(
                run_id=run_id,
                artifact=artifact,
                workspace_dir=Path(source_tmp.name),
                elapsed=elapsed,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
            )
        except Exception:
            _cleanup_source_tmp()
            raise
        candidate_edit_constraints = _candidate_edit_constraints(
            source_context,
            config=self.builder.config,
            dev_evaluator_configured=self.dev_evaluator is not None,
        )
        root_source_context = source_context
        tree_authority_doc = {
            "schema_version": "research_lab.git_tree_authority.v1",
            "run_id": str(run_id),
            "policy": tree_policy.to_dict(),
            "evaluator_commitment": evaluator_commitment,
        }
        root_git_commit = await self._tree_repository_call(
            "initialize",
            source_root=root_source_context.source_root,
            root_artifact_hash=root_artifact.model_artifact_hash,
            policy_hash=tree_policy.policy_hash,
            run_id=str(run_id),
            root_manifest_hash=root_artifact.manifest_hash,
            root_image_digest=sha256_json(
                {"image_digest": root_artifact.image_digest}
            ),
            evaluator_commitment_hash=sha256_json(evaluator_commitment),
            tree_doc=tree_authority_doc,
        )
        if not re.fullmatch(r"[0-9a-f]{64}", str(root_git_commit or "")):
            _cleanup_source_tmp()
            raise GitTreeSchedulerError(
                "Git-tree repository returned an invalid SHA-256 root commit"
            )
        self._active_tree_root_git_commit = str(root_git_commit)
        for checkpoint_node in tree_scheduler.nodes:
            if checkpoint_node.git_commit:
                await self._tree_repository_call(
                    "verify_node_identity",
                    node_id=checkpoint_node.node_id,
                    git_commit=checkpoint_node.git_commit,
                    parent_node_id=checkpoint_node.parent_node_id,
                )
        tree_source_contexts: dict[str, Any] = {
            root_artifact.model_artifact_hash: root_source_context
        }
        candidates_by_node_id: dict[str, BuiltCodeEditCandidate] = {}
        tree_commits_by_node_id: dict[str, dict[str, Any]] = {}
        generation_request_hashes: dict[str, str] = {}
        generation_start_costs: dict[str, int] = {}
        generation_start_calls: dict[str, int] = {}
        generation_attempt_counts: dict[str, int] = {}
        branch_objective_path_ids: dict[str, str] = {}
        branch_objective_hashes: dict[str, str] = {}

        def _unbuilt_tree_node(
            slot: TreeChildSlot,
            *,
            status: str,
        ) -> TreeNode:
            commit_doc = tree_commits_by_node_id.get(str(slot.node_id), {})
            return TreeNode(
                tree_id=tree_id,
                node_id=str(slot.node_id),
                parent_node_id=str(slot.parent_node_id),
                root_branch_id=str(slot.root_branch_id),
                depth=int(slot.depth),
                slot_index=int(slot.slot_index),
                status=status,
                branch_objective_path_id=branch_objective_path_ids.get(
                    str(slot.node_id), ""
                ),
                branch_objective_hash=branch_objective_hashes.get(
                    str(slot.node_id), ""
                ),
                generation_attempt_count=generation_attempt_counts.get(
                    str(slot.node_id), 0
                ),
                git_commit=str(commit_doc.get("git_commit") or ""),
                source_tree_hash=str(commit_doc.get("source_tree_hash") or ""),
                incremental_patch_hash=str(
                    commit_doc.get("incremental_patch_hash") or ""
                ),
                cumulative_patch_hash=str(
                    commit_doc.get("cumulative_patch_hash") or ""
                ),
                settled_cost_microusd=max(
                    0,
                    int(actual_cost_microusd)
                    - generation_start_costs.get(str(slot.node_id), 0),
                ),
            )

        async def _inspect_tree_operation(
            operation_id: str, *, allow_missing: bool = False
        ) -> dict[str, Any] | None:
            inspected = await self._tree_repository_call(
                "inspect_operation", operation_id=operation_id
            )
            if not isinstance(inspected, Mapping) or inspected.get(
                "operation_id"
            ) != operation_id:
                raise GitTreeSchedulerError(
                    "tree operation could not be reconciled"
                )
            if inspected.get("exists") is not True:
                if allow_missing and inspected.get("exists") is False:
                    return None
                raise GitTreeSchedulerError(
                    "tree operation could not be reconciled"
                )
            if not isinstance(inspected.get("operation"), Mapping):
                raise GitTreeSchedulerError(
                    "tree operation could not be reconciled"
                )
            operation = dict(inspected["operation"])
            if operation.get("tree_id") != tree_id:
                raise GitTreeSchedulerError(
                    "tree operation belongs to another tree"
                )
            return operation

        async def _resolve_tree_parent(slot: Any) -> tuple[
            BuiltCodeEditCandidate | None,
            PrivateModelArtifactManifest,
            Any,
        ]:
            nonlocal current_branch_id, current_branch_context_doc
            current_branch_id = str(slot.node_id)
            parent = (
                None
                if slot.parent_node_id == "root"
                else candidates_by_node_id.get(str(slot.parent_node_id))
            )
            if slot.parent_node_id != "root" and parent is None:
                raise GitTreeSchedulerError("committed tree parent candidate is missing")
            if parent is None:
                parent_artifact = root_artifact
                parent_context = root_source_context
            else:
                parent_artifact = parent.build.candidate_model_manifest
                parent_context = tree_source_contexts.get(
                    parent_artifact.model_artifact_hash
                )
            parent_hash = parent_artifact.model_artifact_hash
            if parent_context is None:
                try:
                    parent_context = (
                        await self._prepare_parent_source_context_with_heartbeat(
                            run_id=run_id,
                            artifact=parent_artifact,
                            workspace_dir=(
                                Path(source_tmp.name)
                                / (
                                    "tree-parent-"
                                    + re.sub(r"[^0-9a-zA-Z]", "", parent_hash)[-24:]
                                )
                            ),
                            elapsed=elapsed,
                            openrouter_calls=openrouter_calls,
                            estimated_cost=estimated_cost,
                            actual_cost_microusd=actual_cost_microusd,
                        )
                    )
                    tree_source_contexts[parent_hash] = parent_context
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "research_lab_tree_parent_context_failed "
                        "run_id=%s node_id=%s error=%s",
                        run_id,
                        str(slot.parent_node_id)[:80],
                        type(exc).__name__,
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=str(slot.node_id),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "tree_parent_context_failed",
                            ),
                            event_doc={
                                "stage": "tree_parent_context_failed",
                                "error_class": type(exc).__name__,
                                "error_hash": sha256_json(
                                    {
                                        "error_type": type(exc).__name__,
                                        "error": str(exc),
                                    }
                                ),
                                "tree_id": tree_id,
                                "tree_parent_node_id": str(slot.parent_node_id),
                                "tree_parent_artifact_hash": parent_hash,
                                "recovery": "checkpoint_and_requeue",
                            },
                        )
                    )
                    raise GitTreeSchedulerError(
                        "tree parent source could not be reconstructed"
                    ) from exc
            current_branch_context_doc = sanitized_branch_context(
                slot=slot,
                parent=_candidate_tree_node(parent) if parent is not None else None,
                ancestors=tuple(_candidate_tree_node(item) for item in selected),
            )
            return parent, parent_artifact, parent_context

        async def _persist_generation_outcome(
            *, slot: Any, node: TreeNode, reason: str
        ) -> None:
            request_hash = generation_request_hashes.get(str(slot.node_id))
            if not request_hash:
                raise GitTreeSchedulerError(
                    "tree generation reservation is missing for terminal node"
                )
            result_doc = {
                "schema_version": "research_lab.git_tree_generation_result.v1",
                "tree_id": tree_id,
                "node_id": str(slot.node_id),
                "status": node.status,
                "reason": str(reason)[:240],
                "node_hash": sha256_json(node.to_dict()),
                "node": node.to_dict(),
            }
            candidate = candidates_by_node_id.get(str(slot.node_id))
            if candidate is not None:
                result_doc.update(
                    {
                        "rehydration_artifact_uri": (
                            candidate.rehydration_artifact_uri
                        ),
                        "rehydration_artifact_hash": (
                            candidate.rehydration_artifact_hash
                        ),
                    }
                )
            await self._tree_repository_call(
                "settle_operation",
                operation_id=generation_operation_id(slot),
                operation_status=(
                    "succeeded"
                    if node.status in {"evaluating", "eligible", "ineligible"}
                    else "indeterminate"
                    if node.status == "indeterminate"
                    else "failed"
                ),
                request_hash=request_hash,
                result_hash=sha256_json(result_doc),
                settled_cost_microusd=max(
                    0,
                    int(actual_cost_microusd)
                    - generation_start_costs.get(str(slot.node_id), 0),
                ),
                provider_call_count=max(
                    0,
                    int(openrouter_calls)
                    - generation_start_calls.get(str(slot.node_id), 0),
                ),
                settlement_doc={
                    **result_doc,
                    "operation_kind": "generation",
                },
            )
            await self._tree_repository_call(
                "record_node", node_doc=node.to_dict()
            )

        async def _settle_build_operation(
            *,
            slot: TreeChildSlot,
            request_hash: str,
            operation_status: str,
            node: TreeNode,
            reason: str,
            candidate: BuiltCodeEditCandidate | None = None,
        ) -> None:
            result_doc: dict[str, Any] = {
                "schema_version": "research_lab.git_tree_build_result.v1",
                "tree_id": tree_id,
                "node_id": slot.node_id,
                "operation_kind": "build",
                "status": operation_status,
                "reason": str(reason)[:240],
                "build_request_hash": request_hash,
                "generation_request_hash": generation_request_hashes.get(
                    slot.node_id, ""
                ),
                "generation_settled_cost_microusd": max(
                    0,
                    int(actual_cost_microusd)
                    - generation_start_costs.get(slot.node_id, 0),
                ),
                "generation_provider_call_count": max(
                    0,
                    int(openrouter_calls)
                    - generation_start_calls.get(slot.node_id, 0),
                ),
                "node_hash": sha256_json(node.to_dict()),
                "node": node.to_dict(),
            }
            if candidate is not None:
                result_doc.update(
                    {
                        "rehydration_artifact_uri": (
                            candidate.rehydration_artifact_uri
                        ),
                        "rehydration_artifact_hash": (
                            candidate.rehydration_artifact_hash
                        ),
                        "candidate_artifact_hash": (
                            candidate.build.candidate_model_manifest.model_artifact_hash
                        ),
                        "source_diff_hash": candidate.build.source_diff_hash,
                    }
                )
            await self._tree_repository_call(
                "settle_operation",
                operation_id=build_operation_id(slot),
                operation_status=operation_status,
                request_hash=request_hash,
                result_hash=sha256_json(result_doc),
                settled_cost_microusd=0,
                provider_call_count=0,
                settlement_doc=result_doc,
            )

        async def _recover_generation_operation(
            *, slot: TreeChildSlot, request_hash: str
        ) -> bool:
            operation_id = generation_operation_id(slot)
            operation = await _inspect_tree_operation(operation_id)
            assert operation is not None
            if (
                operation.get("operation_kind") != "generation"
                or operation.get("request_hash") != request_hash
                or str(operation.get("node_id") or "") != slot.node_id
            ):
                raise GitTreeSchedulerError(
                    "tree generation operation commitment differs"
                )
            status = str(operation.get("status") or "")
            settlement = dict(operation.get("settlement_doc") or {})
            node_doc = settlement.get("node")
            node = (
                TreeNode.from_mapping(node_doc)
                if isinstance(node_doc, Mapping)
                else TreeNode(
                    tree_id=tree_id,
                    node_id=slot.node_id,
                    parent_node_id=slot.parent_node_id,
                    root_branch_id=slot.root_branch_id,
                    depth=slot.depth,
                    slot_index=slot.slot_index,
                    status=(
                        "failed" if status == "failed" else "indeterminate"
                    ),
                    branch_objective_path_id=branch_objective_path_ids.get(
                        slot.node_id, ""
                    ),
                    branch_objective_hash=branch_objective_hashes.get(
                        slot.node_id, ""
                    ),
                    generation_attempt_count=generation_attempt_counts.get(
                        slot.node_id, 0
                    ),
                )
            )
            if (
                node.tree_id != tree_id
                or node.node_id != slot.node_id
                or node.parent_node_id != slot.parent_node_id
                or node.root_branch_id != slot.root_branch_id
                or node.depth != slot.depth
                or node.slot_index != slot.slot_index
            ):
                raise GitTreeSchedulerError(
                    "tree generation recovery topology differs"
                )

            if status == "succeeded":
                uri = str(settlement.get("rehydration_artifact_uri") or "")
                artifact_hash = str(
                    settlement.get("rehydration_artifact_hash") or ""
                )
                candidate = await self._rehydrate_candidate_reference(
                    uri=uri,
                    expected_hash=artifact_hash,
                    run_id=run_id,
                )
                restored_node = _candidate_tree_node(
                    candidate,
                    status="evaluating" if node.status == "evaluating" else None,
                )
                if restored_node != node:
                    raise GitTreeSchedulerError(
                        "tree generation recovery artifact differs"
                    )
                _replace_selected_candidate(candidate)
            elif status == "reserved":
                build_id = build_operation_id(slot)
                build_operation = await _inspect_tree_operation(
                    build_id, allow_missing=True
                )
                recovered_candidate: BuiltCodeEditCandidate | None = None
                recovered_status = "indeterminate"
                recovered_reason = "reserved_generation_found_after_restart"
                generation_cost = int(
                    operation.get("settled_cost_microusd") or 0
                )
                generation_calls = int(
                    operation.get("provider_call_count") or 0
                )
                if build_operation is not None:
                    if (
                        build_operation.get("operation_kind") != "build"
                        or str(build_operation.get("node_id") or "")
                        != slot.node_id
                    ):
                        raise GitTreeSchedulerError(
                            "tree build operation commitment differs"
                        )
                    build_request_hash = str(
                        build_operation.get("request_hash") or ""
                    )
                    build_reservation = dict(
                        build_operation.get("settlement_doc")
                        or build_operation.get("reservation_doc")
                        or {}
                    )
                    if (
                        build_reservation.get("generation_request_hash")
                        != request_hash
                        or str(
                            build_reservation.get("build_request_hash")
                            or build_request_hash
                        )
                        != build_request_hash
                        or build_reservation.get("tree_id") != tree_id
                        or build_reservation.get("node_id") != slot.node_id
                    ):
                        raise GitTreeSchedulerError(
                            "tree build reservation differs from generation"
                        )
                    generation_cost = int(
                        build_reservation.get(
                            "generation_settled_cost_microusd",
                            generation_cost,
                        )
                        or 0
                    )
                    generation_calls = int(
                        build_reservation.get(
                            "generation_provider_call_count",
                            generation_calls,
                        )
                        or 0
                    )
                    build_status = str(build_operation.get("status") or "")
                    build_settlement = dict(
                        build_operation.get("settlement_doc") or {}
                    )
                    build_node_doc = build_settlement.get("node")
                    if build_status == "succeeded":
                        if not isinstance(build_node_doc, Mapping):
                            raise GitTreeSchedulerError(
                                "successful tree build is missing its node"
                            )
                        node = TreeNode.from_mapping(build_node_doc)
                        uri = str(
                            build_settlement.get("rehydration_artifact_uri")
                            or ""
                        )
                        artifact_hash = str(
                            build_settlement.get("rehydration_artifact_hash")
                            or ""
                        )
                        recovered_candidate = (
                            await self._rehydrate_candidate_reference(
                                uri=uri,
                                expected_hash=artifact_hash,
                                run_id=run_id,
                            )
                        )
                        if (
                            _candidate_tree_node(
                                recovered_candidate, status="evaluating"
                            )
                            != node
                        ):
                            raise GitTreeSchedulerError(
                                "recovered tree build artifact differs"
                            )
                        _replace_selected_candidate(recovered_candidate)
                        recovered_status = "succeeded"
                        recovered_reason = (
                            "build_settled_before_generation_restart"
                        )
                    elif build_status == "failed":
                        node = (
                            TreeNode.from_mapping(build_node_doc)
                            if isinstance(build_node_doc, Mapping)
                            else _unbuilt_tree_node(slot, status="failed")
                        )
                        recovered_status = "failed"
                        recovered_reason = "build_failed_before_generation_settlement"
                    elif build_status in {"reserved", "indeterminate"}:
                        node = (
                            TreeNode.from_mapping(build_node_doc)
                            if isinstance(build_node_doc, Mapping)
                            else _unbuilt_tree_node(slot, status="indeterminate")
                        )
                        if build_status == "reserved":
                            await _settle_build_operation(
                                slot=slot,
                                request_hash=build_request_hash,
                                operation_status="indeterminate",
                                node=node,
                                reason="reserved_build_found_after_restart",
                            )
                        recovered_reason = "build_state_indeterminate_after_restart"
                    else:
                        raise GitTreeSchedulerError(
                            "tree build operation status is invalid"
                        )
                else:
                    node = _unbuilt_tree_node(slot, status="indeterminate")

                result_doc = {
                    "schema_version": "research_lab.git_tree_generation_result.v1",
                    "tree_id": tree_id,
                    "node_id": slot.node_id,
                    "operation_kind": "generation",
                    "status": node.status,
                    "reason": recovered_reason,
                    "node_hash": sha256_json(node.to_dict()),
                    "node": node.to_dict(),
                }
                if recovered_candidate is not None:
                    result_doc.update(
                        {
                            "rehydration_artifact_uri": (
                                recovered_candidate.rehydration_artifact_uri
                            ),
                            "rehydration_artifact_hash": (
                                recovered_candidate.rehydration_artifact_hash
                            ),
                        }
                    )
                await self._tree_repository_call(
                    "settle_operation",
                    operation_id=operation_id,
                    operation_status=recovered_status,
                    request_hash=request_hash,
                    result_hash=sha256_json(result_doc),
                    settled_cost_microusd=max(0, generation_cost),
                    provider_call_count=max(0, generation_calls),
                    settlement_doc=result_doc,
                )
            elif status not in {"failed", "indeterminate"}:
                raise GitTreeSchedulerError(
                    "tree generation operation status is invalid"
                )

            if (
                node.tree_id != tree_id
                or node.node_id != slot.node_id
                or node.parent_node_id != slot.parent_node_id
                or node.root_branch_id != slot.root_branch_id
                or node.depth != slot.depth
                or node.slot_index != slot.slot_index
            ):
                raise GitTreeSchedulerError(
                    "reconciled tree generation topology differs"
                )
            tree_scheduler.record_node(node)
            await self._tree_repository_call(
                "record_node", node_doc=node.to_dict()
            )
            logger.warning(
                "research_lab_tree_generation_reconciled "
                "run_id=%s node_id=%s status=%s",
                run_id,
                slot.node_id,
                status,
            )
            return True

        async def _terminalize_unbuilt_slot(
            slot: TreeChildSlot,
            reason: str,
            *,
            status: str = "failed",
        ) -> None:
            if any(node.node_id == slot.node_id for node in tree_scheduler.nodes):
                return
            node = _unbuilt_tree_node(slot, status=status)
            tree_scheduler.record_node(node)
            await _persist_generation_outcome(slot=slot, node=node, reason=reason)
            logger.warning(
                "research_lab_tree_node_failed run_id=%s node_id=%s reason=%s",
                run_id,
                str(slot.node_id)[:80],
                str(reason)[:160],
            )

        async def _persist_candidate_rehydration(
            candidate: BuiltCodeEditCandidate,
        ) -> BuiltCodeEditCandidate:
            artifact_doc = await _write_private_loop_candidate_artifact(
                artifact=root_artifact,
                run_id=run_id,
                node_id=candidate.node_id,
                iteration=candidate.iteration,
                draft=candidate.draft,
                build=candidate.build,
                dev_score=candidate.dev_score,
                dev_score_version=candidate.dev_score_version,
                dev_evaluation=candidate.dev_evaluation,
                dev_feedback=candidate.dev_feedback,
                dev_feedback_hash=candidate.dev_feedback_hash,
                tree_settled_cost_microusd=(
                    candidate.tree_settled_cost_microusd
                ),
                git_tree=dict(candidate.build.build_doc.get("git_tree") or {}),
                artifact_io=self.artifact_io,
            )
            uri = str(artifact_doc.get("loop_candidate_artifact_uri") or "")
            content_hash = str(
                artifact_doc.get("loop_candidate_artifact_hash") or ""
            )
            if not uri.startswith("s3://") or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", content_hash
            ):
                raise GitTreeSchedulerError(
                    "tree candidate rehydration artifact was not durably verified"
                )
            return replace(
                candidate,
                rehydration_artifact_uri=uri,
                rehydration_artifact_hash=content_hash,
            )

        def _replace_selected_candidate(candidate: BuiltCodeEditCandidate) -> None:
            for index, existing in enumerate(selected):
                if existing.node_id == candidate.node_id:
                    selected[index] = candidate
                    break
            else:
                selected.append(candidate)
            candidates_by_node_id[candidate.node_id] = candidate

        async def _mark_evaluation_indeterminate(
            candidates: Sequence[BuiltCodeEditCandidate], *, reason: str
        ) -> None:
            for candidate in candidates:
                previous = next(
                    (
                        node
                        for node in tree_scheduler.nodes
                        if node.node_id == candidate.node_id
                    ),
                    None,
                )
                if previous is None:
                    raise GitTreeSchedulerError(
                        "tree evaluation node is missing during recovery"
                    )
                terminal = replace(
                    previous,
                    status="indeterminate",
                    evaluation=None,
                )
                tree_scheduler.replace_node(terminal)
                await self._tree_repository_call(
                    "record_node", node_doc=terminal.to_dict()
                )
            logger.warning(
                "research_lab_tree_evaluation_indeterminate "
                "run_id=%s node_count=%s reason=%s",
                run_id,
                len(candidates),
                str(reason)[:160],
            )

        async def _recover_evaluation_operation(
            *,
            candidates: Sequence[BuiltCodeEditCandidate],
            operation_id: str,
            request_hash: str,
            stage: str,
            operation: Mapping[str, Any] | None = None,
        ) -> list[BuiltCodeEditCandidate]:
            state = dict(
                operation
                or await _inspect_tree_operation(operation_id)
                or {}
            )
            if (
                state.get("operation_kind") != "evaluation"
                or state.get("request_hash") != request_hash
                or str(state.get("node_id") or "")
            ):
                raise GitTreeSchedulerError(
                    "tree evaluation operation commitment differs"
                )
            status = str(state.get("status") or "")
            settlement = dict(state.get("settlement_doc") or {})
            ordered = sorted(tuple(candidates), key=lambda item: item.node_id)
            if status == "succeeded":
                rows = settlement.get("nodes")
                if not isinstance(rows, Sequence):
                    raise GitTreeSchedulerError(
                        "tree evaluation settlement artifacts are missing"
                    )
                rows_by_node = {
                    str(row.get("node_id") or ""): dict(row)
                    for row in rows
                    if isinstance(row, Mapping)
                }
                if set(rows_by_node) != {item.node_id for item in ordered}:
                    raise GitTreeSchedulerError(
                        "tree evaluation settlement node set differs"
                    )
                restored: list[BuiltCodeEditCandidate] = []
                for candidate in ordered:
                    row = rows_by_node[candidate.node_id]
                    restored_candidate = await self._rehydrate_candidate_reference(
                        uri=str(row.get("artifact_uri") or ""),
                        expected_hash=str(row.get("artifact_hash") or ""),
                        run_id=run_id,
                    )
                    if (
                        restored_candidate.node_id != candidate.node_id
                        or not _candidate_dev_evaluation_matches_policy(
                            restored_candidate, self._tree_policy_doc
                        )
                    ):
                        raise GitTreeSchedulerError(
                            "tree evaluation recovery artifact differs"
                        )
                    expected_node_doc = row.get("node")
                    expected_node = (
                        TreeNode.from_mapping(expected_node_doc)
                        if isinstance(expected_node_doc, Mapping)
                        else _candidate_tree_node(restored_candidate)
                    )
                    restored_node = _candidate_tree_node(restored_candidate)
                    if restored_node != expected_node:
                        raise GitTreeSchedulerError(
                            "tree evaluation recovery node differs"
                        )
                    tree_scheduler.replace_node(restored_node)
                    _replace_selected_candidate(restored_candidate)
                    await self._tree_repository_call(
                        "record_node", node_doc=restored_node.to_dict()
                    )
                    restored.append(restored_candidate)
                return restored

            if status == "reserved":
                result_doc = {
                    "schema_version": "research_lab.git_tree_evaluation_result.v1",
                    "tree_id": tree_id,
                    "stage": stage,
                    "operation_kind": "evaluation",
                    "node_id": "",
                    "operation_id": operation_id,
                    "request_hash": request_hash,
                    "status": "indeterminate",
                    "reason": "reserved_operation_found_after_restart",
                    "nodes": [
                        {"node_id": item.node_id} for item in ordered
                    ],
                }
                await self._tree_repository_call(
                    "settle_operation",
                    operation_id=operation_id,
                    operation_status="indeterminate",
                    request_hash=request_hash,
                    result_hash=sha256_json(result_doc),
                    settled_cost_microusd=int(
                        state.get("settled_cost_microusd") or 0
                    ),
                    provider_call_count=int(
                        state.get("provider_call_count") or 0
                    ),
                    settlement_doc=result_doc,
                )
            elif status not in {"failed", "indeterminate"}:
                raise GitTreeSchedulerError(
                    "tree evaluation operation status is invalid"
                )
            await _mark_evaluation_indeterminate(
                ordered,
                reason=f"operation_{status}",
            )
            return []

        async def _evaluate_tree_cohort(
            candidates: Sequence[BuiltCodeEditCandidate],
            *,
            stage: str,
        ) -> list[BuiltCodeEditCandidate]:
            nonlocal current_branch_id
            nonlocal budget_limit_microusd
            nonlocal tree_evaluation_cost_microusd
            nonlocal tree_evaluation_provider_call_count
            ordered = sorted(tuple(candidates), key=lambda item: item.node_id)
            if not ordered:
                return []
            operation_id = cohort_evaluation_operation_id(
                tree_id=tree_id,
                node_ids=[candidate.node_id for candidate in ordered],
                stage=stage,
            )
            request_doc = {
                "schema_version": "research_lab.git_tree_evaluation_request.v1",
                "tree_id": tree_id,
                "stage": stage,
                "operation_id": operation_id,
                "policy_hash": tree_policy.policy_hash,
                "evaluator_commitment_hash": sha256_json(evaluator_commitment),
                "candidates": [
                    {
                        "node_id": candidate.node_id,
                        "artifact_hash": candidate.build.candidate_model_manifest.model_artifact_hash,
                        "manifest_hash": candidate.build.candidate_model_manifest.manifest_hash,
                        "source_diff_hash": candidate.build.source_diff_hash,
                    }
                    for candidate in ordered
                ],
            }
            request_hash = sha256_json(request_doc)
            existing_operation = await _inspect_tree_operation(
                operation_id, allow_missing=True
            )
            if existing_operation is not None:
                return await _recover_evaluation_operation(
                    candidates=ordered,
                    operation_id=operation_id,
                    request_hash=request_hash,
                    stage=stage,
                    operation=existing_operation,
                )
            try:
                reservation = await self._tree_repository_call(
                    "reserve_operation",
                    operation_id=operation_id,
                    operation_kind="evaluation",
                    request_hash=request_hash,
                    reservation_doc=request_doc,
                )
            except Exception:
                raced_operation = await _inspect_tree_operation(
                    operation_id, allow_missing=True
                )
                if raced_operation is None:
                    raise
                return await _recover_evaluation_operation(
                    candidates=ordered,
                    operation_id=operation_id,
                    request_hash=request_hash,
                    stage=stage,
                    operation=raced_operation,
                )
            if (
                not isinstance(reservation, Mapping)
                or reservation.get("operation_status") != "reserved"
            ):
                raise GitTreeSchedulerError(
                    "tree evaluation reservation response is invalid"
                )
            if reservation.get("created") is not True:
                return await _recover_evaluation_operation(
                    candidates=ordered,
                    operation_id=operation_id,
                    request_hash=request_hash,
                    stage=stage,
                )
            try:
                evaluated = await self._maybe_dev_eval_cohort(
                    ordered,
                    run_id=run_id,
                    remaining_seconds=max(
                        0.0, settings.max_seconds - elapsed()
                    ),
                    remaining_tree_budget_microusd=max(
                        0,
                        budget_limit_microusd - actual_cost_microusd,
                    ),
                    post_evaluation_reserve_seconds=(
                        tree_policy.finalization_reserve_seconds
                        if stage == "final_shortlist"
                        else final_context_reserve_seconds
                    ),
                )
                if {item.node_id for item in evaluated} != {
                    item.node_id for item in ordered
                }:
                    raise GitTreeSchedulerError(
                        "tree cohort evaluation returned different nodes"
                    )
                (
                    evaluated,
                    settled_cost_microusd,
                    provider_call_count,
                ) = _apply_tree_cohort_settlement(evaluated)
                if settled_cost_microusd > max(
                    0, budget_limit_microusd - actual_cost_microusd
                ):
                    raise GitTreeSchedulerError(
                        "tree evaluation exceeded the remaining funded budget"
                    )
                persisted: list[BuiltCodeEditCandidate] = []
                for candidate in evaluated:
                    candidate = await _persist_candidate_rehydration(candidate)
                    persisted.append(candidate)
                    _replace_selected_candidate(candidate)
                    node = _candidate_tree_node(candidate)
                    tree_scheduler.replace_node(node)
                    await self._tree_repository_call(
                        "record_node", node_doc=node.to_dict()
                    )
                    if candidate.dev_score is not None:
                        previous_branch = current_branch_id
                        try:
                            current_branch_id = candidate.node_id
                            _record_dev_score(
                                iteration_index=candidate.iteration,
                                node_id=candidate.node_id,
                                score=candidate.dev_score,
                                version=candidate.dev_score_version,
                            )
                        finally:
                            current_branch_id = previous_branch
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type=(
                                "dev_check_passed" if node.eligible else "dev_check_failed"
                            ),
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=candidate.node_id,
                            candidate_artifact_hash=(
                                candidate.build.candidate_model_manifest.model_artifact_hash
                            ),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "tree_candidate_evaluated",
                            ),
                            event_doc={
                                "tree_id": tree_id,
                                "stage": stage,
                                "eligible": node.eligible,
                                "eligibility_reason": str(
                                    candidate.dev_evaluation.get(
                                        "eligibility_reason"
                                    )
                                    or ""
                                )[:160],
                                "evaluation_mode": str(
                                    candidate.dev_evaluation.get(
                                        "evaluation_mode"
                                    )
                                    or ""
                                ),
                                "context_hash": str(
                                    candidate.dev_evaluation.get("context_hash")
                                    or ""
                                ),
                                "receipt_root": str(
                                    candidate.dev_evaluation.get("receipt_root")
                                    or ""
                                ),
                            },
                        )
                    )
                result_doc = {
                    "schema_version": "research_lab.git_tree_evaluation_result.v1",
                    "tree_id": tree_id,
                    "stage": stage,
                    "operation_kind": "evaluation",
                    "node_id": "",
                    "operation_id": operation_id,
                    "request_hash": request_hash,
                    "nodes": [
                        {
                            "node_id": item.node_id,
                            "artifact_uri": item.rehydration_artifact_uri,
                            "artifact_hash": item.rehydration_artifact_hash,
                            "evaluation_hash": sha256_json(
                                dict(item.dev_evaluation)
                            ),
                            "node": _candidate_tree_node(item).to_dict(),
                        }
                        for item in persisted
                    ],
                    "provider_call_count": provider_call_count,
                    "settled_cost_microusd": settled_cost_microusd,
                }
                await self._tree_repository_call(
                    "settle_operation",
                    operation_id=operation_id,
                    operation_status="succeeded",
                    request_hash=request_hash,
                    result_hash=sha256_json(result_doc),
                    settled_cost_microusd=settled_cost_microusd,
                    provider_call_count=provider_call_count,
                    settlement_doc=result_doc,
                )
                budget_limit_microusd -= settled_cost_microusd
                tree_evaluation_cost_microusd += settled_cost_microusd
                tree_evaluation_provider_call_count += provider_call_count
                return persisted
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "research_lab_tree_cohort_evaluation_unsettled "
                    "run_id=%s stage=%s operation_id=%s error=%s",
                    run_id,
                    stage,
                    operation_id,
                    type(exc).__name__,
                )
                failure_doc = {
                    "schema_version": "research_lab.git_tree_evaluation_result.v1",
                    "tree_id": tree_id,
                    "stage": stage,
                    "operation_kind": "evaluation",
                    "node_id": "",
                    "operation_id": operation_id,
                    "request_hash": request_hash,
                    "status": "indeterminate",
                    "reason": f"evaluation_interrupted:{type(exc).__name__}",
                    "cost_treatment": "deferred_to_attested_provider_ledger",
                    "error_hash": sha256_json(
                        {
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ),
                    "nodes": [
                        {"node_id": item.node_id} for item in ordered
                    ],
                }
                await self._tree_repository_call(
                    "settle_operation",
                    operation_id=operation_id,
                    operation_status="indeterminate",
                    request_hash=request_hash,
                    result_hash=sha256_json(failure_doc),
                    settled_cost_microusd=0,
                    provider_call_count=0,
                    settlement_doc=failure_doc,
                )
                await _mark_evaluation_indeterminate(
                    ordered,
                    reason=failure_doc["reason"],
                )
                raise GitTreeSchedulerError(
                    "tree evaluation became indeterminate; automatic retry is forbidden"
                ) from exc

        async def _evaluate_pending_round() -> bool:
            if tree_scheduler.planned_slots:
                return False
            pending_nodes = tuple(
                node for node in tree_scheduler.nodes if node.status == "evaluating"
            )
            if not pending_nodes:
                return False
            # Re-score the existing eligible contenders with every new sibling
            # cohort. The scheduler may only compare nodes sharing this frozen
            # round context; historical scores from another overlay are never
            # used to choose the next expandable parent.
            comparable_nodes = tuple(
                node
                for node in tree_scheduler.nodes
                if node.status == "evaluating" or node.eligible
            )
            pending_candidates = []
            for node in comparable_nodes:
                candidate = candidates_by_node_id.get(node.node_id)
                if candidate is None:
                    raise GitTreeSchedulerError(
                        "tree evaluation candidate artifact is missing"
                    )
                pending_candidates.append(candidate)
            await _evaluate_tree_cohort(pending_candidates, stage="round")
            return True

        # A tree resume is fail-closed: every committed node must restore with
        # identical ancestry and evaluator commitments before expansion resumes.
        restored_candidate_count = 0
        if resume and _resume_restore_selected_enabled():
            selected = await self._restore_selected_from_resume(
                resume=resume,
                run_id=run_id,
                artifact=artifact,
                elapsed=elapsed,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
            )
            restored_candidate_count = len(selected)
            built_candidate_total = max(built_candidate_total, restored_candidate_count)
            # §6.3-1: re-seed dev-score memory from restored candidates so their
            # ranking-only scores stay visible to later drafts. Plateau counting
            # never spans a pause: restored candidates arrive in ranked (not
            # build) order, so recomputing staleness over them would be spurious
            # — resume conservatively with a fresh improvement window.
            for restored_candidate in selected:
                if restored_candidate.tree_id != tree_id:
                    raise GitTreeSchedulerError(
                        "restored candidate belongs to another tree"
                    )
                candidates_by_node_id[restored_candidate.node_id] = restored_candidate
                current_branch_id = restored_candidate.node_id
                if restored_candidate.dev_score is not None:
                    _record_dev_score(
                        iteration_index=restored_candidate.iteration,
                        node_id=restored_candidate.node_id,
                        score=restored_candidate.dev_score,
                        version=restored_candidate.dev_score_version,
                    )
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="loop_resumed" if resume else "loop_started",
                loop_status="running",
                elapsed_seconds=elapsed_offset,
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "code_edit_loop_resumed" if resume else "code_edit_loop_started",
                ),
                event_doc={
                    "run_id": run_id,
                    "candidate_kind": "image_build",
                    "requested_loop_count": int(requested_loop_count),
                    "settings": _settings_doc(settings),
                    "budget_context": _safe_budget_doc(budget_context),
                    "git_tree_policy": dict(self._tree_policy_doc),
                    "resumed_from_checkpoint": bool(resume),
                    "restored_selected_candidate_count": restored_candidate_count,
                    "checkpoint_hash": resume.get("checkpoint_hash"),
                    "source_mode": source_context.source_mode,
                    "source_tree_hash": source_context.source_tree_hash,
                    "parent_image_digest_hash": source_context.parent_image_digest_hash,
                    "extracted_top_level_paths": list(source_context.top_level_paths),
                    "editable_file_count": len(source_context.editable_files),
                    "editable_file_sample": list(source_context.editable_files[:25]),
                    "file_preview_count": len(source_context.file_previews),
                    "symbol_index_hash": str(source_context.planner_source_index.get("index_hash") or ""),
                    "symbol_count": int(source_context.planner_source_index.get("symbol_count") or 0),
                    "import_count": int(source_context.planner_source_index.get("import_count") or 0),
                    "provider_capability_hash": (
                        provider_capabilities.capability_hash
                        if provider_capabilities is not None
                        and provider_capabilities.private_snapshot_loaded
                        else ""
                    ),
                    "provider_capability_count": (
                        int(provider_capability_summary.get("provider_count") or 0)
                        if provider_capability_summary is not None
                        else 0
                    ),
                },
            )
        )

        prior_attempts = _prior_attempts_from_budget_context(budget_context)
        # §9.5 lesson retrieval (flag default OFF): compact cross-run lessons for the
        # planner + draft prompt context. Strictly best-effort — a retrieval failure
        # must never fail a paid run.
        try:
            from gateway.research_lab.lesson_store import (
                build_lesson_prompt_context,
                lesson_retrieval_enabled,
            )

            if lesson_retrieval_enabled():
                retrieved_lessons_doc = await build_lesson_prompt_context(
                    lane=None,
                    components=(),
                    active_parent_hash=artifact.model_artifact_hash,
                )
        except Exception as exc:
            logger.warning(
                "research_lab_lesson_retrieval_failed run_id=%s error=%s",
                run_id,
                str(exc)[:200],
            )
        # §9.4 meta-allocator cell-yield priors (flag default OFF): deterministic
        # seeded-Thompson ordering/weight hint for the planner context. Context
        # reordering only — never a funding or promotion decision. Best-effort.
        try:
            from gateway.research_lab.allocator_priors import (
                allocator_priors_enabled,
                load_cell_yield_priors,
            )

            if allocator_priors_enabled():
                # Prefers the persisted nightly selection record (identical
                # hint for every run that day); falls back to computing from
                # the ledger before the first nightly pass lands.
                cell_yield_priors_doc = await load_cell_yield_priors()
                # P18: the priors injected into the planner prompt used to
                # survive only inside the captured request body (S3-only,
                # unqueryable). One pointer-scale allocator_decision event per
                # run start records WHICH cell priors aimed this run — the
                # meta-allocator's own future training data. scripts/64 extends
                # the loop-event CHECK for this event type.
                if cell_yield_priors_doc is not None and _engine_env_flag(
                    "RESEARCH_LAB_ALLOCATOR_DECISION_EVENTS_ENABLED", "true"
                ):
                    priors_rows = (
                        cell_yield_priors_doc.get("priors")
                        if isinstance(cell_yield_priors_doc, Mapping)
                        else None
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="allocator_decision",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "allocator_decision",
                            ),
                            event_doc={
                                "schema_version": "1.0",
                                "run_id": run_id,
                                "priors_doc_hash": sha256_json(cell_yield_priors_doc),
                                "prior_count": (
                                    len(priors_rows)
                                    if isinstance(priors_rows, (list, tuple))
                                    else 0
                                ),
                                "cell_yield_priors": cell_yield_priors_doc,
                            },
                        )
                    )
        except Exception as exc:
            logger.warning(
                "research_lab_allocator_priors_failed run_id=%s error=%s",
                run_id,
                str(exc)[:200],
            )
        loop_direction_plan_doc: dict[str, Any] | None = None
        if isinstance(resume.get("loop_direction_plan"), Mapping):
            try:
                resume_binding = _bind_loop_direction_plan(
                    resume["loop_direction_plan"],
                    source_context=source_context,
                    candidate_edit_constraints=candidate_edit_constraints,
                )
                loop_direction_plan_doc = resume_binding.plan_doc
            except Exception as exc:
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "resume_loop_direction_plan_parse_failed",
                        ),
                        event_doc={
                            "stage": "resume_loop_direction_plan_parse",
                            "error": safe_event_error_text(exc),
                            "checkpoint_hash": resume.get("checkpoint_hash"),
                        },
                    )
                )
                loop_direction_plan_doc = None
        planner_terminal_without_candidate = False
        binding_plan_terminal_without_candidate = False
        last_checkpoint: dict[str, Any] | None = None
        stop_reason = "max_iterations"
        reference_repair_attempted = bool(resume.get("planner_reference_repair_attempted"))
        reference_repair_status = str(resume.get("planner_reference_repair_status") or "")

        async def _attempt_planner_reference_repair(
            *,
            trigger: str,
            reason: str,
            plan_doc: Mapping[str, Any],
            explicit_references: Sequence[Any] = (),
            feasibility_errors: Sequence[str] = (),
        ) -> dict[str, Any] | None:
            nonlocal loop_direction_plan_doc
            nonlocal planner_terminal_without_candidate
            nonlocal binding_plan_terminal_without_candidate
            nonlocal stop_reason
            nonlocal reference_repair_attempted
            nonlocal reference_repair_status
            nonlocal openrouter_calls
            nonlocal estimated_cost
            nonlocal actual_cost_microusd
            nonlocal last_checkpoint

            if reference_repair_attempted or not _planner_reference_repair_enabled(self.builder.config):
                return None
            requested_references = unresolved_references_from_context(explicit=explicit_references)
            exact_binding = bind_source_references_exact(
                index_doc=source_context.planner_index(),
                source_root=source_context.source_root,
                editable_files=source_context.editable_files,
                references=requested_references,
            )
            references = exact_binding.missing_references
            if not references:
                reference_repair_status = "no_safe_references"
                return None
            if elapsed() >= settings.max_seconds:
                reference_repair_status = "skipped_time_budget"
                return None
            if _would_exceed_budget(
                actual_cost_microusd,
                _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                budget_limit_microusd,
            ):
                reference_repair_status = "skipped_compute_budget"
                return None

            reference_repair_attempted = True
            reference_repair_status = "attempted"
            # Persist the one-shot state before the paid call. If the process
            # exits during that call, resume must not purchase another repair.
            last_checkpoint = await self._emit_checkpoint(
                run_id=run_id,
                settings=settings,
                artifact=artifact,
                model_id=model_id,
                budget_context=budget_context,
                iterations_completed=iteration,
                elapsed_seconds=elapsed(),
                selected=selected,
                provider_usage=provider_usage,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                stage="before_planner_reference_repair",
                loop_direction_plan=loop_direction_plan_doc,
                built_candidate_count=built_candidate_total,
                probe_budget=probe_budget,
                planner_reference_repair_attempted=True,
                planner_reference_repair_status=reference_repair_status,
            )
            resolution = await asyncio.to_thread(
                resolve_source_references,
                index_doc=source_context.planner_index(),
                source_root=source_context.source_root,
                references=references,
            )
            prior_plan_hash = str(plan_doc.get("plan_hash") or sha256_json(dict(plan_doc)))
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="candidate_generation_fallback_requested",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "planner_reference_repair_requested",
                    ),
                    event_doc={
                        "schema_version": "1.0",
                        "stage": "planner_reference_repair",
                        "trigger": str(trigger or "")[:120],
                        "prior_plan_hash": prior_plan_hash,
                        "symbol_index_hash": str(resolution.get("symbol_index_hash") or ""),
                        "reference_count": int(resolution.get("reference_count") or 0),
                        "resolved_reference_count": int(resolution.get("resolved_reference_count") or 0),
                        "resolution_hash": str(resolution.get("resolution_hash") or ""),
                        "feasibility_error_count": len(feasibility_errors),
                    },
                )
            )
            remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
            repair_result, repair_error = await self._call_stage_contained(
                build_loop_direction_reference_repair_messages(
                    ticket={
                        "brief_public_summary": _ticket_doc_value(ticket, "brief_public_summary"),
                        "focus_signature_hash": _focus_signature_hash(ticket),
                    },
                    original_plan=plan_doc,
                    reference_resolution=resolution,
                    candidate_edit_constraints=candidate_edit_constraints,
                    feasibility_errors=feasibility_errors,
                ),
                min(settings.draft_timeout_seconds, remaining_call_seconds),
                self.builder.config.loop_planner_max_tokens,
                "loop_planner_reference_repair",
            )
            if repair_result is None:
                failure_usage = (
                    repair_error.failure_usage_entries()
                    if isinstance(repair_error, ContainedStageFailure)
                    else []
                )
                if isinstance(repair_error, ContainedStageFailure):
                    actual_cost_microusd += repair_error.cost_microusd
                    provider_usage.extend(failure_usage)
                reference_repair_status = "call_failed"
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        provider_usage=failure_usage,
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "planner_reference_repair_call_failed",
                        ),
                        event_doc={
                            "stage": "planner_reference_repair",
                            "status": reference_repair_status,
                            "prior_plan_hash": prior_plan_hash,
                            "resolution_hash": str(resolution.get("resolution_hash") or ""),
                            "failure_hash": sha256_json(
                                {"failure": str(repair_error or "planner_reference_repair_call_failed")}
                            ),
                        },
                    )
                )
                return None

            openrouter_calls += 1
            estimated_cost += settings.estimated_iteration_cost_usd
            actual_cost_microusd += max(0, int(repair_result.cost_microusd))
            if repair_result.provider_usage:
                provider_usage.append(
                    {**repair_result.provider_usage, "call_stage": "loop_planner_reference_repair"}
                )
            try:
                repaired_plan = parse_loop_direction_plan_response(repair_result.content)
                repaired_binding = _bind_loop_direction_plan(
                    repaired_plan.to_dict(),
                    source_context=source_context,
                    candidate_edit_constraints=candidate_edit_constraints,
                )
                repaired_doc = dict(repaired_binding.plan_doc or repaired_plan.to_dict())
                repaired_plan = loop_direction_plan_from_mapping(repaired_doc)
            except Exception:
                reference_repair_status = "parse_failed"
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "planner_reference_repair_parse_failed",
                        ),
                        event_doc={
                            "stage": "planner_reference_repair",
                            "status": reference_repair_status,
                            "prior_plan_hash": prior_plan_hash,
                            "resolution_hash": str(resolution.get("resolution_hash") or ""),
                            "raw_response_hash": sha256_json({"raw_response": repair_result.content}),
                        },
                    )
                )
                return None

            repaired_feasibility_errors = list(repaired_binding.errors)
            loop_direction_plan_doc = repaired_doc
            planner_terminal_without_candidate = bool(
                repaired_plan.no_new_safe_path or repaired_feasibility_errors
            )
            binding_plan_terminal_without_candidate = bool(repaired_feasibility_errors)
            stop_reason = (
                "binding_plan_unimplementable"
                if repaired_feasibility_errors
                else "loop_direction_no_new_safe_path"
                if repaired_plan.no_new_safe_path
                else "max_iterations"
            )
            reference_repair_status = (
                "still_infeasible"
                if repaired_feasibility_errors
                else "still_unresolved"
                if repaired_plan.no_new_safe_path
                else "repaired"
            )
            last_checkpoint = await self._emit_checkpoint(
                run_id=run_id,
                settings=settings,
                artifact=artifact,
                model_id=model_id,
                budget_context=budget_context,
                iterations_completed=iteration,
                elapsed_seconds=elapsed(),
                selected=selected,
                provider_usage=provider_usage,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                stage="after_planner_reference_repair",
                loop_direction_plan=repaired_doc,
                built_candidate_count=built_candidate_total,
                probe_budget=probe_budget,
                planner_reference_repair_attempted=True,
                planner_reference_repair_status=reference_repair_status,
            )
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="loop_direction_planned",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    provider_usage=([provider_usage[-1]] if provider_usage else []),
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "planner_reference_repair_completed",
                    ),
                    event_doc={
                        "schema_version": "1.0",
                        "stage": "planner_reference_repair",
                        "status": reference_repair_status,
                        "prior_plan_hash": prior_plan_hash,
                        "repaired_plan_hash": str(repaired_doc.get("plan_hash") or ""),
                        "symbol_index_hash": str(resolution.get("symbol_index_hash") or ""),
                        "reference_count": int(resolution.get("reference_count") or 0),
                        "resolved_reference_count": int(resolution.get("resolved_reference_count") or 0),
                        "resolution_hash": str(resolution.get("resolution_hash") or ""),
                        "feasibility_error_count": len(repaired_feasibility_errors),
                    },
                )
            )
            return None if repaired_feasibility_errors else repaired_doc

        if (
            self.builder.config.loop_planner_enabled
            and loop_direction_plan_doc is None
            and len(tree_scheduler.nodes) < tree_policy.max_nodes
        ):
            if _would_exceed_budget(
                actual_cost_microusd,
                _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                budget_limit_microusd,
            ):
                stop_reason = "compute_budget_exhausted_before_loop_planner"
                planner_terminal_without_candidate = True
            else:
                # Bug 16: the planner used to be single-shot — one malformed planner response
                # terminally failed the paid run. Retry the call once on failure, then fall
                # back to the existing plan-less mode instead of a terminal failure.
                planner_attempt_limit = 2 if _planner_parse_retry_enabled() else 1
                for planner_attempt in range(1, planner_attempt_limit + 1):
                    if planner_attempt > 1 and _would_exceed_budget(
                        actual_cost_microusd,
                        _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                        budget_limit_microusd,
                    ):
                        break
                    remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
                    raw_plan = ""
                    planner_result, planner_call_error = await self._call_stage_contained(
                        build_loop_direction_planner_messages(
                            ticket={
                                "ticket_id": str(ticket.get("ticket_id") or ""),
                                "run_id": run_id,
                                "miner_hotkey": ticket.get("miner_hotkey"),
                                "island": ticket.get("island"),
                                "brief_sanitized_ref": ticket.get("brief_sanitized_ref"),
                                "brief_public_summary": _ticket_doc_value(ticket, "brief_public_summary"),
                                "requested_loop_count": requested_loop_count,
                                "focus_signature_hash": _focus_signature_hash(ticket),
                            },
                            artifact_manifest=root_artifact.to_dict(),
                            component_registry=dict(component_registry),
                            benchmark_public_summary=benchmark_public_summary,
                            runtime_source_index=source_context.planner_index(),
                            budget_context={
                                **dict(budget_context),
                                "candidate_kind": "image_build",
                                "focus_signature_hash": _focus_signature_hash(ticket),
                                **(
                                    {"retrieved_lessons": retrieved_lessons_doc}
                                    if retrieved_lessons_doc is not None
                                    else {}
                                ),
                                **(
                                    {"cell_yield_priors": cell_yield_priors_doc}
                                    if cell_yield_priors_doc is not None
                                    else {}
                                ),
                            },
                            prior_attempts=prior_attempts,
                            provider_outcome_digest=self.provider_outcome_digest,
                            provider_capability_summary=provider_capability_summary,
                            candidate_edit_constraints=candidate_edit_constraints,
                            branch_factor=tree_policy.branch_factor,
                        ),
                        min(settings.draft_timeout_seconds, remaining_call_seconds),
                        self.builder.config.loop_planner_max_tokens,
                        "loop_planner",
                    )
                    if planner_result is None:
                        # P3: keep the failed call's telemetry — the raw trace
                        # already exists; propagate the pointer and the cost.
                        failure_usage = (
                            planner_call_error.failure_usage_entries()
                            if isinstance(planner_call_error, ContainedStageFailure)
                            else []
                        )
                        if isinstance(planner_call_error, ContainedStageFailure):
                            actual_cost_microusd += planner_call_error.cost_microusd
                            provider_usage.extend(failure_usage)
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="code_edit_validation_failed",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "loop_direction_planner_call_failed",
                                ),
                                provider_usage=failure_usage,
                                event_doc={
                                    "stage": "loop_direction_planner",
                                    "planner_attempt": planner_attempt,
                                    "error": planner_call_error or "loop_direction_planner_call_failed",
                                    "fallback": (
                                        "plan_less_mode" if planner_attempt >= planner_attempt_limit else "retry"
                                    ),
                                    "focus_signature_hash": _focus_signature_hash(ticket),
                                },
                            )
                        )
                        continue
                    raw_plan = planner_result.content
                    openrouter_calls += 1
                    estimated_cost += settings.estimated_iteration_cost_usd
                    actual_cost_microusd += max(0, int(planner_result.cost_microusd))
                    if planner_result.provider_usage:
                        provider_usage.append({**planner_result.provider_usage, "call_stage": "loop_planner"})
                    try:
                        loop_plan = parse_loop_direction_plan_response(raw_plan)
                        initial_binding = _bind_loop_direction_plan(
                            loop_plan.to_dict(),
                            source_context=source_context,
                            candidate_edit_constraints=candidate_edit_constraints,
                        )
                        loop_direction_plan_doc = dict(initial_binding.plan_doc or loop_plan.to_dict())
                        loop_plan = loop_direction_plan_from_mapping(loop_direction_plan_doc)
                    except Exception as exc:
                        if planner_attempt >= planner_attempt_limit and not _planner_parse_retry_enabled():
                            stop_reason = "loop_direction_plan_parse_failed"
                            planner_terminal_without_candidate = True
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="code_edit_validation_failed",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                provider_usage=([provider_usage[-1]] if provider_usage else []),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "loop_direction_plan_parse_failed",
                                ),
                                event_doc={
                                    "stage": "loop_direction_planner",
                                    "planner_attempt": planner_attempt,
                                    "error": safe_event_error_text(exc),
                                    "raw_response_hash": sha256_json({"raw_response": raw_plan}),
                                    "fallback": (
                                        "terminal"
                                        if planner_terminal_without_candidate
                                        else (
                                            "plan_less_mode"
                                            if planner_attempt >= planner_attempt_limit
                                            else "retry"
                                        )
                                    ),
                                    "focus_signature_hash": _focus_signature_hash(ticket),
                                },
                            )
                        )
                        continue
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="loop_direction_planned",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "loop_direction_planned",
                            ),
                            event_doc={
                                "focus_signature_hash": _focus_signature_hash(ticket),
                                "loop_direction_plan": loop_direction_plan_doc,
                                "prior_attempt_count": len(prior_attempts),
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    if loop_plan.no_new_safe_path:
                        if loop_plan.unresolved_references:
                            repaired_doc = await _attempt_planner_reference_repair(
                                trigger="loop_direction_no_new_safe_path",
                                reason=loop_plan.reason or "planner returned no_new_safe_path",
                                plan_doc=loop_direction_plan_doc,
                                explicit_references=loop_plan.unresolved_references,
                            )
                            if repaired_doc is not None:
                                loop_plan = loop_direction_plan_from_mapping(repaired_doc)
                                loop_direction_plan_doc = repaired_doc
                                if not loop_plan.no_new_safe_path:
                                    break
                        stop_reason = "loop_direction_no_new_safe_path"
                        planner_terminal_without_candidate = True
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="no_viable_patch",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                provider_usage=([provider_usage[-1]] if provider_usage else []),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "no_viable_patch",
                                ),
                                event_doc={
                                    "reason": loop_plan.reason or "planner returned no_new_safe_path",
                                    "failure_class": (
                                        "binding_plan_unimplementable"
                                        if loop_plan.unresolved_references
                                        or _binding_plan_unimplementable_reason(loop_plan.reason)
                                        else "no_safe_patch"
                                    ),
                                    "missing_references": list(loop_plan.unresolved_references),
                                    "loop_direction_plan_hash": loop_direction_plan_doc.get("plan_hash"),
                                    "focus_signature_hash": _focus_signature_hash(ticket),
                                },
                            )
                        )
                    break
        elif not self.builder.config.loop_planner_enabled:
            loop_direction_plan_doc = None
        if (
            isinstance(loop_direction_plan_doc, Mapping)
            and not planner_terminal_without_candidate
            and not binding_plan_terminal_without_candidate
        ):
            plan_binding = _bind_loop_direction_plan(
                loop_direction_plan_doc,
                source_context=source_context,
                candidate_edit_constraints=candidate_edit_constraints,
            )
            loop_direction_plan_doc = dict(plan_binding.plan_doc or loop_direction_plan_doc)
            plan_feasibility_errors = list(plan_binding.errors)
            plan_missing_references = plan_binding.missing_references
            if plan_feasibility_errors:
                feasibility_reason = "; ".join(plan_feasibility_errors)[:700]
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "loop_direction_plan_feasibility_failed",
                        ),
                        event_doc={
                            "stage": "loop_direction_plan_feasibility",
                            "error": feasibility_reason,
                            "feasibility_error_count": len(plan_feasibility_errors),
                            "loop_direction_plan_hash": loop_direction_plan_doc.get("plan_hash"),
                        },
                    )
                )
                repaired_doc = await _attempt_planner_reference_repair(
                    trigger="loop_direction_plan_feasibility",
                    reason=feasibility_reason,
                    plan_doc=loop_direction_plan_doc,
                    explicit_references=plan_missing_references,
                    feasibility_errors=plan_feasibility_errors,
                )
                if repaired_doc is not None and not bool(repaired_doc.get("no_new_safe_path")):
                    planner_terminal_without_candidate = False
                    binding_plan_terminal_without_candidate = False
                    stop_reason = "max_iterations"
                else:
                    planner_terminal_without_candidate = True
                    binding_plan_terminal_without_candidate = True
                    stop_reason = "binding_plan_unimplementable"
        tree_base_plan_doc = dict(loop_direction_plan_doc or {})
        valid_root_objectives = [
            _tree_branch_direction_plan(
                tree_base_plan_doc,
                root_slot_index=index,
            )
            for index in range(tree_policy.branch_factor)
        ]
        if any(item is None for item in valid_root_objectives):
            planner_terminal_without_candidate = True
            binding_plan_terminal_without_candidate = True
            stop_reason = "tree_branch_objectives_incomplete"
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="no_viable_patch",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "tree_branch_objective_validation",
                    ),
                    event_doc={
                        "stage": "tree_branch_objective_validation",
                        "failure_class": "tree_branch_objectives_incomplete",
                        "expected_branch_count": tree_policy.branch_factor,
                        "valid_branch_count": sum(
                            item is not None for item in valid_root_objectives
                        ),
                        "loop_direction_plan_hash": str(
                            tree_base_plan_doc.get("plan_hash") or ""
                        ),
                        "recovery": "generate_a_complete_independent_path_set",
                    },
                )
            )

        def _root_slot_index(slot: TreeChildSlot) -> int:
            if slot.parent_node_id == "root":
                return int(slot.slot_index)
            root_node = next(
                (
                    node
                    for node in tree_scheduler.nodes
                    if node.node_id == slot.root_branch_id
                    and node.parent_node_id == "root"
                ),
                None,
            )
            if root_node is None:
                raise GitTreeSchedulerError(
                    "tree branch root is missing for child objective"
                )
            return int(root_node.slot_index)

        def _direction_plan_for_slot(
            slot: TreeChildSlot,
        ) -> tuple[dict[str, Any] | None, str, str]:
            plan_doc = _tree_branch_direction_plan(
                tree_base_plan_doc,
                root_slot_index=_root_slot_index(slot),
            )
            if not isinstance(plan_doc, Mapping):
                return None, "", ""
            path_id = _loop_plan_selected_path_id(plan_doc)
            plan_hash = str(plan_doc.get("plan_hash") or "")
            if not path_id or not re.fullmatch(r"sha256:[0-9a-f]{64}", plan_hash):
                raise GitTreeSchedulerError(
                    "tree branch objective commitment is invalid"
                )
            existing_root = next(
                (
                    node
                    for node in tree_scheduler.nodes
                    if node.node_id == slot.root_branch_id
                ),
                None,
            )
            if existing_root is not None and (
                existing_root.branch_objective_path_id != path_id
                or existing_root.branch_objective_hash != plan_hash
            ):
                raise GitTreeSchedulerError(
                    "tree branch objective changed after root commitment"
                )
            return dict(plan_doc), path_id, plan_hash

        while iteration < settings.max_iterations:
            if planner_terminal_without_candidate or binding_plan_terminal_without_candidate:
                break
            if not tree_scheduler.planned_slots and any(
                node.status == "evaluating" for node in tree_scheduler.nodes
            ):
                if should_pause and await should_pause():
                    last_checkpoint = await self._emit_checkpoint(
                        run_id=run_id,
                        settings=settings,
                        artifact=artifact,
                        model_id=model_id,
                        budget_context=budget_context,
                        iterations_completed=iteration,
                        elapsed_seconds=elapsed(),
                        selected=selected,
                        provider_usage=provider_usage,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                        stage="pause_before_tree_round_evaluation",
                        loop_direction_plan=loop_direction_plan_doc,
                        built_candidate_count=built_candidate_total,
                        probe_budget=probe_budget,
                        planner_reference_repair_attempted=reference_repair_attempted,
                        planner_reference_repair_status=reference_repair_status,
                    )
                    _cleanup_source_tmp()
                    return self._result(
                        selected=selected,
                        status="paused",
                        stop_reason="maintenance_pause_requested",
                        iterations_completed=iteration,
                        elapsed_seconds=elapsed(),
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                        openrouter_calls=openrouter_calls,
                        provider_usage=provider_usage,
                        checkpoint=last_checkpoint,
                    )
                if await _evaluate_pending_round():
                    last_checkpoint = await self._emit_checkpoint(
                        run_id=run_id,
                        settings=settings,
                        artifact=artifact,
                        model_id=model_id,
                        budget_context=budget_context,
                        iterations_completed=iteration,
                        elapsed_seconds=elapsed(),
                        selected=selected,
                        provider_usage=provider_usage,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                        stage="tree_round_evaluated",
                        loop_direction_plan=loop_direction_plan_doc,
                        built_candidate_count=built_candidate_total,
                        probe_budget=probe_budget,
                        planner_reference_repair_attempted=reference_repair_attempted,
                        planner_reference_repair_status=reference_repair_status,
                    )
            if elapsed() >= settings.max_seconds:
                stop_reason = "max_seconds"
                break
            if (
                settings.max_seconds - elapsed()
                <= final_context_reserve_seconds
            ):
                stop_reason = "tree_finalization_reserve"
                break
            if _would_exceed_budget(
                actual_cost_microusd,
                _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                budget_limit_microusd,
            ):
                stop_reason = "compute_budget_exhausted_before_next_code_edit"
                break
            if should_pause and await should_pause():
                last_checkpoint = await self._emit_checkpoint(
                    run_id=run_id,
                    settings=settings,
                    artifact=artifact,
                    model_id=model_id,
                    budget_context=budget_context,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    selected=selected,
                    provider_usage=provider_usage,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    stage="pause_before_next_code_edit",
                    loop_direction_plan=loop_direction_plan_doc,
                    built_candidate_count=built_candidate_total,
                    probe_budget=probe_budget,
                    planner_reference_repair_attempted=reference_repair_attempted,
                    planner_reference_repair_status=reference_repair_status,
                )
                _cleanup_source_tmp()
                return self._result(
                    selected=selected,
                    status="paused",
                    stop_reason="maintenance_pause_requested",
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    openrouter_calls=openrouter_calls,
                    provider_usage=provider_usage,
                    checkpoint=last_checkpoint,
                )

            if not tree_scheduler.planned_slots:
                planned_round = tree_scheduler.plan_round()
                if planned_round:
                    last_checkpoint = await self._emit_checkpoint(
                        run_id=run_id,
                        settings=settings,
                        artifact=artifact,
                        model_id=model_id,
                        budget_context=budget_context,
                        iterations_completed=iteration,
                        elapsed_seconds=elapsed(),
                        selected=selected,
                        provider_usage=provider_usage,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                        stage="tree_round_planned",
                        loop_direction_plan=loop_direction_plan_doc,
                        built_candidate_count=built_candidate_total,
                        probe_budget=probe_budget,
                        planner_reference_repair_attempted=reference_repair_attempted,
                        planner_reference_repair_status=reference_repair_status,
                    )
            tree_slot = tree_scheduler.next_planned()
            if tree_slot is None:
                stop_reason = "tree_frontier_exhausted"
                break
            iteration += 1
            built_nodes_before_iteration = len(selected)
            (
                tree_parent_candidate,
                tree_parent_artifact,
                source_context,
            ) = await _resolve_tree_parent(tree_slot)
            candidate_edit_constraints = _candidate_edit_constraints(
                source_context,
                config=self.builder.config,
                dev_evaluator_configured=self.dev_evaluator is not None,
            )
            (
                branch_direction_plan_doc,
                branch_objective_path_id,
                branch_objective_hash,
            ) = _direction_plan_for_slot(tree_slot)
            branch_binding_errors: tuple[str, ...] = ()
            if isinstance(branch_direction_plan_doc, Mapping):
                branch_binding = _bind_loop_direction_plan(
                    branch_direction_plan_doc,
                    source_context=source_context,
                    candidate_edit_constraints=candidate_edit_constraints,
                )
                branch_binding_errors = tuple(branch_binding.errors)
                branch_direction_plan_doc = dict(
                    branch_binding.plan_doc or branch_direction_plan_doc
                )
                branch_objective_path_id = _loop_plan_selected_path_id(
                    branch_direction_plan_doc
                )
                branch_objective_hash = str(
                    branch_direction_plan_doc.get("plan_hash") or ""
                )
            branch_objective_path_ids[tree_slot.node_id] = (
                branch_objective_path_id
            )
            branch_objective_hashes[tree_slot.node_id] = branch_objective_hash
            current_branch_context_doc = {
                **dict(current_branch_context_doc or {}),
                "branch_objective_path_id": branch_objective_path_id,
                "branch_objective_hash": branch_objective_hash,
            }
            generation_request_doc = {
                "schema_version": "research_lab.git_tree_generation_request.v1",
                "tree_id": tree_id,
                "slot": tree_slot.to_dict(),
                "policy_hash": tree_policy.policy_hash,
                "parent_artifact_hash": (
                    tree_parent_artifact.model_artifact_hash
                ),
                "parent_source_tree_hash": source_context.source_tree_hash,
                "branch_context_hash": sha256_json(
                    dict(current_branch_context_doc or {})
                ),
                "branch_objective_path_id": branch_objective_path_id,
                "branch_objective_hash": branch_objective_hash,
                "evaluator_commitment_hash": sha256_json(
                    evaluator_commitment
                ),
            }
            generation_request_hash = sha256_json(generation_request_doc)
            generation_operation = generation_operation_id(tree_slot)
            active_request_hash = generation_request_hashes.get(tree_slot.node_id)
            if active_request_hash:
                if active_request_hash != generation_request_hash:
                    raise GitTreeSchedulerError(
                        "in-process tree generation commitment changed"
                    )
                # A bounded parser/reference repair may retry this exact slot
                # before the process exits. Keep its one logical reservation
                # open; only a newly constructed engine treats a pre-existing
                # reserved operation as indeterminate crash recovery.
                plan_result = {
                    "created": True,
                    "operation_status": "reserved",
                    "in_process_retry": True,
                }
            else:
                generation_request_hashes[tree_slot.node_id] = generation_request_hash
                generation_start_costs[tree_slot.node_id] = int(
                    actual_cost_microusd
                )
                generation_start_calls[tree_slot.node_id] = int(openrouter_calls)
                plan_result = await self._tree_repository_call(
                    "plan_slot",
                    slot=tree_slot,
                    request_hash=generation_request_hash,
                    operation_id=generation_operation,
                    node_doc={
                        "schema_version": "research_lab.git_tree_node_plan.v1",
                        "tree_id": tree_id,
                        "node_id": tree_slot.node_id,
                        "parent_node_id": tree_slot.parent_node_id,
                        "root_branch_id": tree_slot.root_branch_id,
                        "depth": tree_slot.depth,
                        "child_slot": tree_slot.slot_index,
                        "parent_artifact_hash": (
                            tree_parent_artifact.model_artifact_hash
                        ),
                        "parent_source_tree_hash": source_context.source_tree_hash,
                        "branch_objective_path_id": branch_objective_path_id,
                        "branch_objective_hash": branch_objective_hash,
                        "generation_request_hash": generation_request_hash,
                    },
                )
            if not isinstance(plan_result, Mapping) or plan_result.get(
                "operation_status"
            ) not in {"reserved", "succeeded", "failed", "indeterminate"}:
                raise GitTreeSchedulerError(
                    "tree generation reservation response is invalid"
                )
            if plan_result.get("created") is not True:
                await _recover_generation_operation(
                    slot=tree_slot,
                    request_hash=generation_request_hash,
                )
                last_checkpoint = await self._emit_checkpoint(
                    run_id=run_id,
                    settings=settings,
                    artifact=artifact,
                    model_id=model_id,
                    budget_context=budget_context,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    selected=selected,
                    provider_usage=provider_usage,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    stage="tree_generation_operation_reconciled",
                    loop_direction_plan=loop_direction_plan_doc,
                    built_candidate_count=built_candidate_total,
                    probe_budget=probe_budget,
                    planner_reference_repair_attempted=reference_repair_attempted,
                    planner_reference_repair_status=reference_repair_status,
                )
                continue
            if not branch_objective_path_id or not branch_objective_hash:
                await _terminalize_unbuilt_slot(
                    tree_slot, "tree_branch_objective_unavailable"
                )
                continue
            if branch_binding_errors:
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=tree_slot.node_id,
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "tree_parent_plan_binding_failed",
                        ),
                        event_doc={
                            "stage": "tree_parent_plan_binding_failed",
                            "error_count": len(branch_binding_errors),
                            "error_hash": sha256_json(
                                {"errors": list(branch_binding_errors)}
                            ),
                            "tree_id": tree_id,
                            "tree_parent_node_id": tree_slot.parent_node_id,
                            "branch_objective_path_id": branch_objective_path_id,
                            "branch_objective_hash": branch_objective_hash,
                            "recovery": "prune_node",
                        },
                    )
                )
                await _terminalize_unbuilt_slot(
                    tree_slot, "tree_parent_plan_binding_failed"
                )
                continue
            remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
            source_inspection_context: dict[str, Any] = {
                "schema_version": "1.0",
                "source_tree_hash": source_context.source_tree_hash,
                "read_files": [],
                "results": [],
                "bytes_returned": 0,
            }
            read_paths: set[str] = set()
            read_ranges: dict[str, tuple[tuple[int, int], ...]] = {}
            source_access_v2 = bool(getattr(self.builder.config, "code_edit_source_access_v2", True))
            source_bytes_returned = 0
            # Symbol-slice seeding: the accepted plan's must_inspect carries
            # canonical path::symbol references whose spans the index already
            # knows. Seed those spans as pre-read ranges through the
            # PRODUCTION resolver (same schema, caps, and redaction as the
            # model's own reads) so round one starts sighted. Interactive
            # search/ranged reads stay untouched as the fallback; the seeded
            # share of the byte budget is capped so it can never starve them.
            slice_mode = str(
                getattr(self.builder.config, "code_edit_symbol_slice_mode", "off") or "off"
            ).lower()
            if slice_mode in {"shadow", "on"} and source_access_v2:
                try:
                    plan_references = tuple(
                        str(item)
                        for item in (
                            (branch_direction_plan_doc or {}).get("must_inspect", [])
                            if isinstance(branch_direction_plan_doc, Mapping)
                            else []
                        )
                        if item
                    )
                    slice_requests, slice_unplannable = plan_read_requests(
                        source_root=source_context.source_root,
                        index_doc=source_context.planner_index(),
                        normalized_references=plan_references,
                    )
                    slice_budget_bytes = max(
                        0,
                        int(
                            self.builder.config.code_edit_source_inspection_total_bytes
                            * float(
                                getattr(
                                    self.builder.config,
                                    "code_edit_symbol_slice_budget_share",
                                    0.4,
                                )
                            )
                        ),
                    )
                    seeded_bytes = 0
                    seeded_ranges = 0
                    if slice_mode == "on" and slice_requests:
                        seed_batch = resolve_source_inspection_requests(
                            source_context,
                            [
                                CodeEditSourceInspectionRequest(
                                    operation="read_file",
                                    path=request.path,
                                    start_line=request.start_line,
                                    max_lines=request.max_lines,
                                    rationale=f"seeded_symbol_slice:{request.reason}",
                                )
                                for request in slice_requests
                            ],
                            already_read_paths=(),
                            max_files=self.builder.config.code_edit_source_inspection_max_files,
                            max_file_bytes=self.builder.config.code_edit_source_inspection_file_bytes,
                            max_total_bytes=slice_budget_bytes,
                            max_search_matches=self.builder.config.code_edit_source_inspection_search_matches,
                            source_access_v2=True,
                            already_read_ranges={},
                            max_ranges_per_path=max(
                                int(
                                    getattr(
                                        self.builder.config,
                                        "code_edit_source_inspection_max_ranges_per_path",
                                        3,
                                    )
                                ),
                                len(slice_requests),
                            ),
                        )
                        source_bytes_returned += seed_batch.bytes_returned
                        seeded_bytes = seed_batch.bytes_returned
                        read_paths = set(seed_batch.read_paths)
                        read_ranges = dict(seed_batch.read_ranges)
                        seeded_ranges = sum(len(spans) for spans in read_ranges.values())
                        source_inspection_context = _merge_source_inspection_context(
                            source_inspection_context,
                            seed_batch.model_context,
                            total_bytes=source_bytes_returned,
                            read_paths=read_paths,
                        )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="source_inspection_seeded",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "source_inspection_seeded",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "mode": slice_mode,
                                "planned_requests": len(slice_requests),
                                "unplannable_references": list(slice_unplannable)[:8],
                                "seeded_bytes": seeded_bytes,
                                "seeded_ranges": seeded_ranges,
                                "slice_budget_bytes": slice_budget_bytes,
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                except Exception as exc:
                    # Fail open to today's behavior: seeding is an
                    # optimization and must never abort a paid run.
                    logger.warning(
                        "research_lab_symbol_slice_seed_failed mode=%s error=%s",
                        slice_mode,
                        str(exc)[:200],
                    )
            budget_exhausted_after_source_inspection = False
            max_inspection_rounds = max(
                1, int(self.builder.config.code_edit_source_inspection_rounds)
            )
            last_inspection_round = 0
            for inspection_round in range(1, max_inspection_rounds + 1):
                last_inspection_round = inspection_round
                if elapsed() >= settings.max_seconds:
                    stop_reason = "max_seconds"
                    break
                if _would_exceed_budget(
                    actual_cost_microusd,
                    _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                    budget_limit_microusd,
                ):
                    stop_reason = "compute_budget_exhausted_before_source_inspection"
                    budget_exhausted_after_source_inspection = True
                    break
                remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
                # P11: source-inspection rounds are the one genuinely agentic
                # stage (the model's search/read_file/finish output chooses the
                # next tool call). The episode scope stamps the correlation id
                # onto the persisted raw trace so multi-round inspection
                # reassembles as one axis-A episode.
                with call_episode(
                    run_id=run_id,
                    iteration=iteration,
                    inspection_round=inspection_round,
                ):
                    inspection_result, inspection_call_error = await self._call_stage_contained(
                        build_code_edit_source_inspection_messages(
                            ticket={
                                "ticket_id": str(ticket.get("ticket_id") or ""),
                                "run_id": run_id,
                                "miner_hotkey": ticket.get("miner_hotkey"),
                                "island": ticket.get("island"),
                                "brief_sanitized_ref": ticket.get("brief_sanitized_ref"),
                                "brief_public_summary": _ticket_doc_value(ticket, "brief_public_summary"),
                                "requested_loop_count": requested_loop_count,
                                "loop_iteration": iteration,
                                "inspection_round": inspection_round,
                            },
                            artifact_manifest=tree_parent_artifact.to_dict(),
                            component_registry=dict(component_registry),
                            benchmark_public_summary=benchmark_public_summary,
                            runtime_source_index=source_context.inspection_index(),
                            source_inspection_context=source_inspection_context,
                            loop_direction_plan=branch_direction_plan_doc,
                            budget_context=_memory_budget_context({
                                **dict(budget_context),
                                "loop_iteration": iteration,
                                "inspection_round": inspection_round,
                                "candidate_kind": "image_build",
                                "loop_direction_plan_hash": (
                                    (branch_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(branch_direction_plan_doc, Mapping)
                                    else None
                                ),
                            }),
                            max_requests=4,
                            source_access_v2=source_access_v2,
                            provider_outcome_digest=self.provider_outcome_digest,
                            provider_capability_summary=provider_capability_summary,
                            candidate_edit_constraints=candidate_edit_constraints,
                            inspection_round=inspection_round,
                            max_inspection_rounds=max_inspection_rounds,
                            provider_probe_catalog=(
                                {
                                    "endpoints": [endpoint.prompt_summary() for endpoint in probe_catalog],
                                    "budget": probe_budget.to_context(),
                                }
                                if probes_enabled and probe_budget is not None
                                else None
                            ),
                        ),
                        min(settings.draft_timeout_seconds, remaining_call_seconds),
                        3000,
                        "source_inspection",
                    )
                if inspection_result is None:
                    # Bug 17: a non-retryable LLM error mid-inspection used to abort the run.
                    # Skip the remaining inspection rounds and continue with what was gathered.
                    # P3: keep the failed call's telemetry pointer + cost.
                    failure_usage = (
                        inspection_call_error.failure_usage_entries()
                        if isinstance(inspection_call_error, ContainedStageFailure)
                        else []
                    )
                    if isinstance(inspection_call_error, ContainedStageFailure):
                        actual_cost_microusd += inspection_call_error.cost_microusd
                        provider_usage.extend(failure_usage)
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="source_inspection_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "source_inspection_call_failed",
                            ),
                            provider_usage=failure_usage,
                            event_doc={
                                "iteration": iteration,
                                "inspection_round": inspection_round,
                                "stage": "source_inspection_call_failed",
                                "error": inspection_call_error or "source_inspection_call_failed",
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    break
                raw_inspection = inspection_result.content
                openrouter_calls += 1
                estimated_cost += settings.estimated_iteration_cost_usd
                actual_cost_microusd += max(0, int(inspection_result.cost_microusd))
                if inspection_result.provider_usage:
                    provider_usage.append(
                        {
                            **inspection_result.provider_usage,
                            "loop_iteration": iteration,
                            "inspection_round": inspection_round,
                            "call_stage": "source_inspection",
                        }
                    )
                try:
                    requests = parse_code_edit_source_inspection_response(
                        raw_inspection,
                        max_requests=4,
                        allowed_operations=(
                            ("search", "read_file", "probe_provider", "finish") if probes_enabled else None
                        ),
                    )
                except Exception as exc:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="source_inspection_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "source_inspection_parse_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "inspection_round": inspection_round,
                                "error": safe_event_error_text(exc),
                                "raw_response_hash": sha256_json({"raw_response": raw_inspection}),
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    break
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="source_inspection_requested",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "source_inspection_requested",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "inspection_round": inspection_round,
                            "source_tree_hash": source_context.source_tree_hash,
                            "requests": [request.to_event_doc() for request in requests],
                            "request_hash": sha256_json({"requests": [request.to_event_doc() for request in requests]}),
                        },
                    )
                )
                probe_requests = [request for request in requests if request.operation == "probe_provider"]
                if probe_requests and probes_enabled and probe_budget is not None:
                    proxy_url = str(os.getenv("RESEARCH_LAB_EVIDENCE_PROXY_URL") or "").strip()
                    live_probes = _engine_env_flag("RESEARCH_LAB_LOOP_PROVIDER_PROBES_LIVE", "false")
                    for probe_request in probe_requests:
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="probe_requested",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                event_doc={
                                    "iteration": iteration,
                                    "inspection_round": inspection_round,
                                    **probe_request.to_event_doc(),
                                },
                            )
                        )
                        resolution = await asyncio.to_thread(
                            resolve_provider_probe,
                            probe_request,
                            catalog=probe_catalog,
                            proxy_url=proxy_url,
                            budget=probe_budget,
                            live_enabled=live_probes,
                            private_window_term_hashes=self.probe_private_window_term_hashes,
                            registry_base_urls=probe_registry_base_urls,
                            snapshot_overlay_uri=_probe_snapshot_overlay_uri(),
                            evidence_resolver=self.probe_evidence_resolver,
                        )
                        # Probe spend charges the run's existing microusd ledger.
                        actual_cost_microusd += max(0, int(resolution.cost_microusd))
                        probe_event_type = "probe_blocked" if resolution.outcome == "blocked" else "probe_resolved"
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type=probe_event_type,
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    probe_event_type,
                                ),
                                event_doc={
                                    "iteration": iteration,
                                    "inspection_round": inspection_round,
                                    "outcome": resolution.outcome,
                                    **resolution.event_doc,
                                },
                            )
                        )
                        source_inspection_context = _merge_source_inspection_context(
                            source_inspection_context,
                            {
                                "source_tree_hash": source_context.source_tree_hash,
                                "results": [resolution.model_result],
                            },
                            total_bytes=source_bytes_returned,
                            read_paths=read_paths,
                        )
                if any(request.operation == "finish" for request in requests):
                    break
                source_requests = [request for request in requests if request.operation != "probe_provider"]
                if not source_requests:
                    continue
                try:
                    batch = resolve_source_inspection_requests(
                        source_context,
                        source_requests,
                        already_read_paths=tuple(sorted(read_paths)),
                        max_files=self.builder.config.code_edit_source_inspection_max_files,
                        max_file_bytes=self.builder.config.code_edit_source_inspection_file_bytes,
                        max_total_bytes=max(
                            0,
                            self.builder.config.code_edit_source_inspection_total_bytes - source_bytes_returned,
                        ),
                        max_search_matches=self.builder.config.code_edit_source_inspection_search_matches,
                        source_access_v2=source_access_v2,
                        already_read_ranges=read_ranges,
                        max_ranges_per_path=int(
                            getattr(self.builder.config, "code_edit_source_inspection_max_ranges_per_path", 3)
                        ),
                    )
                except CodeEditBuildError as exc:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="source_inspection_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "source_inspection_resolution_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "inspection_round": inspection_round,
                                "error": safe_event_error_text(exc),
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    break
                source_bytes_returned += batch.bytes_returned
                read_paths = set(batch.read_paths)
                if source_access_v2:
                    read_ranges = dict(batch.read_ranges)
                source_inspection_context = _merge_source_inspection_context(
                    source_inspection_context,
                    batch.model_context,
                    total_bytes=source_bytes_returned,
                    read_paths=read_paths,
                )
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="source_inspection_resolved",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "source_inspection_resolved",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "inspection_round": inspection_round,
                            **batch.event_doc,
                        },
                    )
                )
                if source_bytes_returned >= self.builder.config.code_edit_source_inspection_total_bytes:
                    break
                if actual_cost_microusd >= budget_limit_microusd > 0:
                    stop_reason = "compute_budget_exhausted_after_source_inspection"
                    budget_exhausted_after_source_inspection = True
                    break
            if budget_exhausted_after_source_inspection:
                break
            if not read_paths:
                source_unread_stage = (
                    "source_inspection_exhausted_without_read"
                    if last_inspection_round >= max_inspection_rounds
                    else "code_edit_no_source_files_read"
                )
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            source_unread_stage,
                        ),
                        event_doc={
                            "iteration": iteration,
                            "error": "code_edit_no_source_files_read",
                            "stage": source_unread_stage,
                            "inspection_round": last_inspection_round,
                            "max_inspection_rounds": max_inspection_rounds,
                            "source_tree_hash": source_context.source_tree_hash,
                        },
                    )
                )
                drafts = []
                budget_exhausted_after_call = actual_cost_microusd >= budget_limit_microusd > 0
                raw = ""
            else:
                remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
                if _would_exceed_budget(
                    actual_cost_microusd,
                    _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                    budget_limit_microusd,
                ):
                    stop_reason = "compute_budget_exhausted_before_code_edit"
                    break
                # One provider operation owns one predeclared child slot. A
                # bounded repair/fallback attempt may replace a malformed
                # response for that same slot, but a response can never create
                # siblings or consume multiple node identities.
                draft_parse_limit = 1
                candidate_generation_attempt_count = 0

                async def _attempt_candidate_generation_fallback(
                    *,
                    trigger: str,
                    reason: str,
                ) -> list[CodeEditDraft]:
                    nonlocal openrouter_calls, estimated_cost, actual_cost_microusd
                    nonlocal candidate_generation_attempt_count
                    if (
                        candidate_generation_attempt_count
                        >= tree_policy.generation_attempts
                        or not read_paths
                    ):
                        return []
                    if elapsed() >= settings.max_seconds:
                        return []
                    if _would_exceed_budget(
                        actual_cost_microusd,
                        _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                        budget_limit_microusd,
                    ):
                        return []
                    candidate_generation_attempt_count += 1
                    generation_attempt_counts[tree_slot.node_id] = (
                        candidate_generation_attempt_count
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_generation_fallback_requested",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "candidate_generation_fallback_requested",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "trigger": str(trigger or "")[:120],
                                "reason": safe_event_error_text(reason),
                                "mode": "smaller_same_lane_inspected_files",
                                "generation_attempt": candidate_generation_attempt_count,
                                "generation_attempt_limit": tree_policy.generation_attempts,
                                "max_candidates": 1,
                                "max_target_files": _fallback_max_target_files(self.builder.config),
                                "read_file_count": len(read_paths),
                                "read_files_sample": list(sorted(read_paths))[:8],
                            },
                        )
                    )
                    remaining_fallback_seconds = max(1, int(settings.max_seconds - elapsed()))
                    fallback_result, fallback_call_error = await self._call_stage_contained(
                        build_code_edit_fallback_messages(
                            ticket={
                                "ticket_id": str(ticket.get("ticket_id") or ""),
                                "run_id": run_id,
                                "miner_hotkey": ticket.get("miner_hotkey"),
                                "island": ticket.get("island"),
                                "brief_sanitized_ref": ticket.get("brief_sanitized_ref"),
                                "brief_public_summary": _ticket_doc_value(ticket, "brief_public_summary"),
                                "requested_loop_count": requested_loop_count,
                                "loop_iteration": iteration,
                            },
                            artifact_manifest=tree_parent_artifact.to_dict(),
                            component_registry=dict(component_registry),
                            benchmark_public_summary=benchmark_public_summary,
                            runtime_source_context=source_context.prompt_context(),
                            source_inspection_context=source_inspection_context,
                            loop_direction_plan=branch_direction_plan_doc,
                            budget_context=_memory_budget_context(
                                {
                                    **dict(budget_context),
                                    "loop_iteration": iteration,
                                    "candidate_kind": "image_build",
                                    "fallback_trigger": str(trigger or "")[:120],
                                    "loop_direction_plan_hash": (
                                        (branch_direction_plan_doc or {}).get("plan_hash")
                                        if isinstance(branch_direction_plan_doc, Mapping)
                                        else None
                                    ),
                                },
                                include_lessons=True,
                            ),
                        fallback_reason=reason,
                        max_candidates=1,
                        max_target_files=_fallback_max_target_files(self.builder.config),
                        candidate_edit_constraints=candidate_edit_constraints,
                    ),
                        min(settings.draft_timeout_seconds, remaining_fallback_seconds),
                        3000,
                        "code_edit_fallback",
                    )
                    if fallback_result is None:
                        failure_usage = (
                            fallback_call_error.failure_usage_entries()
                            if isinstance(fallback_call_error, ContainedStageFailure)
                            else []
                        )
                        if isinstance(fallback_call_error, ContainedStageFailure):
                            actual_cost_microusd += fallback_call_error.cost_microusd
                            provider_usage.extend(failure_usage)
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="candidate_generation_fallback_failed",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                provider_usage=failure_usage,
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "candidate_generation_fallback_call_failed",
                                ),
                                event_doc={
                                    "iteration": iteration,
                                    "trigger": str(trigger or "")[:120],
                                    "stage": "candidate_generation_fallback_call_failed",
                                    "error": fallback_call_error or "candidate_generation_fallback_call_failed",
                                },
                            )
                        )
                        return await _attempt_candidate_generation_fallback(
                            trigger=trigger,
                            reason=reason,
                        )
                    openrouter_calls += 1
                    estimated_cost += settings.estimated_iteration_cost_usd
                    actual_cost_microusd += max(0, int(fallback_result.cost_microusd))
                    if fallback_result.provider_usage:
                        provider_usage.append(
                            {
                                **fallback_result.provider_usage,
                                "loop_iteration": iteration,
                                "call_stage": "code_edit_fallback",
                                "fallback_trigger": str(trigger or "")[:120],
                            }
                        )
                    try:
                        fallback_drafts = parse_code_edit_response(
                            fallback_result.content,
                            max_candidates=1,
                            max_target_files=_fallback_max_target_files(self.builder.config),
                        )
                    except Exception as exc:
                        try:
                            no_viable = parse_code_edit_no_viable_patch_response(fallback_result.content)
                        except ValueError:
                            no_viable = None
                        no_viable_reason = no_viable.reason if no_viable is not None else ""
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type=(
                                    "no_viable_patch"
                                    if no_viable_reason
                                    else "candidate_generation_fallback_failed"
                                ),
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                provider_usage=([provider_usage[-1]] if provider_usage else []),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "candidate_generation_fallback_no_viable_patch"
                                    if no_viable_reason
                                    else "candidate_generation_fallback_parse_failed",
                                ),
                                event_doc={
                                    "iteration": iteration,
                                    "trigger": str(trigger or "")[:120],
                                    "stage": (
                                        "candidate_generation_fallback_no_viable_patch"
                                        if no_viable_reason
                                        else "candidate_generation_fallback_parse_failed"
                                    ),
                                    "reason": no_viable_reason,
                                    **(
                                        {
                                            "failure_class": no_viable.failure_class,
                                            "missing_references": list(no_viable.missing_references),
                                        }
                                        if no_viable is not None
                                        else {}
                                    ),
                                    "error": safe_event_error_text(exc),
                                    "raw_response_hash": sha256_json({"raw_response": fallback_result.content}),
                                },
                            )
                        )
                        return await _attempt_candidate_generation_fallback(
                            trigger=trigger,
                            reason=no_viable_reason or safe_event_error_text(exc),
                        )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_generation_fallback_drafted",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "candidate_generation_fallback_drafted",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "trigger": str(trigger or "")[:120],
                                "draft_count": len(fallback_drafts),
                                "target_files": list(fallback_drafts[0].target_files) if fallback_drafts else [],
                                "unified_diff_hash": (
                                    sha256_json({"unified_diff": fallback_drafts[0].unified_diff})
                                    if fallback_drafts
                                    else ""
                                ),
                            },
                        )
                    )
                    return fallback_drafts

                candidate_generation_attempt_count += 1
                generation_attempt_counts[tree_slot.node_id] = (
                    candidate_generation_attempt_count
                )
                draft_result, draft_call_error = await self._call_stage_contained(
                    build_code_edit_auto_research_messages(
                        ticket={
                            "ticket_id": str(ticket.get("ticket_id") or ""),
                            "run_id": run_id,
                            "miner_hotkey": ticket.get("miner_hotkey"),
                            "island": ticket.get("island"),
                            "brief_sanitized_ref": ticket.get("brief_sanitized_ref"),
                            "brief_public_summary": _ticket_doc_value(ticket, "brief_public_summary"),
                            "requested_loop_count": requested_loop_count,
                            "loop_iteration": iteration,
                        },
                        artifact_manifest=tree_parent_artifact.to_dict(),
                        component_registry=dict(component_registry),
                        benchmark_public_summary=benchmark_public_summary,
                        runtime_source_context=source_context.prompt_context(),
                        source_inspection_context=source_inspection_context,
                        loop_direction_plan=branch_direction_plan_doc,
                        budget_context=_memory_budget_context(
                            {
                                **dict(budget_context),
                                "loop_iteration": iteration,
                                "candidate_kind": "image_build",
                                "loop_direction_plan_hash": (
                                    (branch_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(branch_direction_plan_doc, Mapping)
                                    else None
                                ),
                            },
                            include_lessons=True,
                        ),
                        max_candidates=draft_parse_limit,
                        provider_outcome_digest=self.provider_outcome_digest,
                        provider_capability_summary=provider_capability_summary,
                        candidate_edit_constraints=candidate_edit_constraints,
                    ),
                    min(settings.draft_timeout_seconds, remaining_call_seconds),
                    3000,
                    "code_edit_draft",
                )
                if draft_result is None:
                    # Bug 17: a non-retryable LLM error on the draft call used to abort the
                    # run and discard prior iterations/candidates. Skip this iteration instead.
                    # P3: keep the failed call's telemetry pointer + cost.
                    failure_usage = (
                        draft_call_error.failure_usage_entries()
                        if isinstance(draft_call_error, ContainedStageFailure)
                        else []
                    )
                    if isinstance(draft_call_error, ContainedStageFailure):
                        actual_cost_microusd += draft_call_error.cost_microusd
                        provider_usage.extend(failure_usage)
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_draft_call_failed",
                            ),
                            provider_usage=failure_usage,
                            event_doc={
                                "iteration": iteration,
                                "stage": "code_edit_draft_call_failed",
                                "error": draft_call_error or "code_edit_draft_call_failed",
                                "generation_attempt": candidate_generation_attempt_count,
                                "generation_attempt_limit": tree_policy.generation_attempts,
                            },
                        )
                    )
                    drafts = await _attempt_candidate_generation_fallback(
                        trigger="code_edit_draft_call_failed",
                        reason=str(
                            draft_call_error or "code_edit_draft_call_failed"
                        ),
                    )
                    raw = ""
                    budget_exhausted_after_call = actual_cost_microusd >= budget_limit_microusd > 0
                else:
                    raw = draft_result.content
                    openrouter_calls += 1
                    estimated_cost += settings.estimated_iteration_cost_usd
                    actual_cost_microusd += max(0, int(draft_result.cost_microusd))
                    if draft_result.provider_usage:
                        provider_usage.append({**draft_result.provider_usage, "loop_iteration": iteration, "call_stage": "code_edit_draft"})
                    budget_exhausted_after_call = (
                        budget_limit_microusd > 0 and actual_cost_microusd >= budget_limit_microusd
                    )
                    try:
                        drafts = parse_code_edit_response(raw, max_candidates=draft_parse_limit)
                    except Exception as exc:
                        try:
                            no_viable = parse_code_edit_no_viable_patch_response(raw)
                        except ValueError:
                            no_viable = None
                        no_viable_reason = no_viable.reason if no_viable is not None else ""
                        fallback_drafts: list[CodeEditDraft] = []
                        if no_viable_reason:
                            terminal_binding_plan = bool(branch_direction_plan_doc) and bool(
                                no_viable is not None
                                and no_viable.failure_class == "binding_plan_unimplementable"
                            )
                            await self.event_sink(
                                AutoResearchLoopEvent(
                                    event_type="no_viable_patch",
                                    loop_status="running",
                                    elapsed_seconds=elapsed(),
                                    cost_ledger=_running_cost_ledger(
                                        openrouter_calls,
                                        estimated_cost,
                                        actual_cost_microusd,
                                        "no_viable_patch",
                                    ),
                                    # P3: the draft call succeeded (the model
                                    # declined to patch) — carry its usage so
                                    # the raw-trace pointer rides the event.
                                    provider_usage=([provider_usage[-1]] if provider_usage else []),
                                    event_doc={
                                        "iteration": iteration,
                                        "reason": no_viable_reason,
                                        "failure_class": (
                                            no_viable.failure_class if no_viable is not None else "no_safe_patch"
                                        ),
                                        "missing_references": (
                                            list(no_viable.missing_references)
                                            if no_viable is not None
                                            else []
                                        ),
                                        **(
                                            {
                                                "terminal_node": True,
                                                "node_failure_reason": "binding_plan_unimplementable",
                                            }
                                            if terminal_binding_plan
                                            else {}
                                        ),
                                        "raw_response_hash": sha256_json({"raw_response": raw}),
                                        "loop_direction_plan_hash": (
                                            (branch_direction_plan_doc or {}).get("plan_hash")
                                            if isinstance(branch_direction_plan_doc, Mapping)
                                            else None
                                        ),
                                    },
                                )
                            )
                            fallback_drafts = await _attempt_candidate_generation_fallback(
                                trigger="no_viable_patch",
                                reason=no_viable_reason,
                            )
                        else:
                            await self.event_sink(
                                AutoResearchLoopEvent(
                                    event_type="candidate_patch_parse_failed",
                                    loop_status="running",
                                    elapsed_seconds=elapsed(),
                                    cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_patch_parse_failed"),
                                    # P3: parse failures are negative training
                                    # examples — the draft call's usage carries
                                    # the raw-trace pointer to the malformed
                                    # response.
                                    provider_usage=([provider_usage[-1]] if provider_usage else []),
                                    event_doc={
                                        "iteration": iteration,
                                        "error": safe_event_error_text(exc),
                                        "raw_response_hash": sha256_json({"raw_response": raw}),
                                    },
                                )
                            )
                            fallback_drafts = await _attempt_candidate_generation_fallback(
                                trigger="candidate_patch_parse_failed",
                                reason=safe_event_error_text(exc),
                            )
                        drafts = fallback_drafts if fallback_drafts else []
            for draft in drafts[:1]:
                if (
                    provider_capabilities is not None
                    and provider_capabilities.private_snapshot_loaded
                    and (
                        str(draft.lane or "")
                        in {"source_routing", "provider_fallback", "openrouter_model_selection"}
                        or summary_mentions_private_capability(
                            draft.redacted_summary,
                            provider_capabilities,
                        )
                    )
                ):
                    draft = replace(
                        draft,
                        redacted_summary=(
                            "Adjusted an approved provider path inside the existing runtime "
                            "while preserving credential, cost, and output safeguards."
                        ),
                    )
                draft_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
                if draft_diff_hash in all_rejected_diff_hashes:
                    # Within-run memory: skip a draft identical to a diff already rejected
                    # earlier in this run instead of paying to judge/build it again.
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "within_run_duplicate_rejected_diff",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "stage": "within_run_duplicate_rejected_diff",
                                "error": "duplicate_rejected_diff_skipped",
                                "unified_diff_hash": draft_diff_hash,
                            },
                        )
                    )
                    continue
                node_id = tree_slot.node_id
                source_errors = self.builder.validate_draft_against_source_context(
                    draft,
                    source_context,
                    read_paths=tuple(sorted(read_paths)),
                    require_read=True,
                )
                if provider_capabilities is not None and provider_capabilities.private_snapshot_loaded:
                    source_errors.extend(
                        validate_candidate_provider_diff(
                            draft.unified_diff,
                            provider_capabilities,
                        )
                    )
                if source_errors:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_source_context_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "target_files": list(draft.target_files),
                                "error": "; ".join(source_errors)[:500],
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    _record_within_run_rejection(
                        stage="source_context_validation",
                        reason="; ".join(source_errors),
                        iteration_index=iteration,
                        draft=draft,
                        diff_hash=draft_diff_hash,
                    )
                    await self._emit_reflection_recorded(
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        draft=draft,
                        outcome="source_context_validation",
                        detail="; ".join(source_errors),
                        artifact=tree_parent_artifact,
                        component_registry=component_registry,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                    )
                    continue
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_drafted",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "code_edit_drafted"),
                        event_doc={
                            "iteration": iteration,
                            "lane": draft.lane,
                            "plan_path_id": draft.plan_path_id,
                            "loop_direction_plan_hash": (
                                (branch_direction_plan_doc or {}).get("plan_hash")
                                if isinstance(branch_direction_plan_doc, Mapping)
                                else None
                            ),
                            "target_files": list(draft.target_files),
                            "unified_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                            "hypothesis": {
                                "failure_mode": draft.failure_mode,
                                "mechanism": draft.mechanism,
                                "expected_improvement": draft.expected_improvement,
                                "risk": draft.risk,
                                "predicted_delta": draft.predicted_delta,
                            },
                        },
                    )
                )
                (
                    candidate_draft,
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    patch_budget_exhausted,
                ) = await self._ensure_patch_applies_or_repair(
                    draft=draft,
                    run_id=run_id,
                    node_id=node_id,
                    iteration=iteration,
                    settings=settings,
                    artifact=tree_parent_artifact,
                    source_context=source_context,
                    source_inspection_context=source_inspection_context,
                    read_paths=tuple(sorted(read_paths)),
                    budget_context=budget_context,
                    budget_limit_microusd=budget_limit_microusd,
                    elapsed=elapsed,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    provider_usage=provider_usage,
                    within_run_memory=_within_run_memory_doc(),
                )
                if patch_budget_exhausted:
                    budget_exhausted_after_call = True
                    continue
                if candidate_draft is None:
                    _record_within_run_rejection(
                        stage="patch_apply_repair_exhausted",
                        reason="patch did not apply after repair attempts",
                        iteration_index=iteration,
                        draft=draft,
                        diff_hash=draft_diff_hash,
                    )
                    await self._emit_reflection_recorded(
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        draft=draft,
                        outcome="patch_apply_repair_exhausted",
                        detail="patch did not apply after repair attempts",
                        artifact=tree_parent_artifact,
                        component_registry=component_registry,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                    )
                    continue
                (
                    alignment_ok,
                    candidate_draft,
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    alignment_budget_exhausted,
                ) = await self._judge_plan_alignment(
                    draft=candidate_draft,
                    loop_direction_plan=branch_direction_plan_doc,
                    prior_attempts=prior_attempts,
                    node_id=node_id,
                    iteration=iteration,
                    settings=settings,
                    budget_limit_microusd=budget_limit_microusd,
                    elapsed=elapsed,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    provider_usage=provider_usage,
                )
                if alignment_budget_exhausted:
                    budget_exhausted_after_call = True
                    continue
                if not alignment_ok:
                    alignment_doc = dict(candidate_draft.plan_alignment or {})
                    alignment_reason = str(
                        alignment_doc.get("blocking_issue") or alignment_doc.get("reason") or "plan alignment rejected"
                    )
                    _record_within_run_rejection(
                        stage="plan_alignment_rejected",
                        reason=alignment_reason,
                        iteration_index=iteration,
                        draft=candidate_draft,
                    )
                    await self._emit_reflection_recorded(
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        draft=candidate_draft,
                        outcome="plan_alignment_rejected",
                        detail=alignment_reason,
                        artifact=tree_parent_artifact,
                        component_registry=component_registry,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                    )
                    continue
                build_completed = False
                tree_terminal_persisted = False
                build_request_hash = ""
                build_operation_reserved = False
                if (
                    settings.max_seconds - elapsed()
                    <= final_context_reserve_seconds
                ):
                    finalization_reserve_reached = True
                    break
                available_build_seconds = max(
                    1,
                    int(
                        settings.max_seconds
                        - elapsed()
                        - final_context_reserve_seconds
                    ),
                )
                build_timeout_seconds = min(
                    max(
                        1,
                        int(
                            getattr(
                                self.builder.config,
                                "code_edit_build_timeout_seconds",
                                available_build_seconds,
                            )
                        ),
                    ),
                    available_build_seconds,
                )
                try:
                    incremental_draft = candidate_draft
                    drafted_incremental_hash = sha256_json(
                        {"unified_diff": incremental_draft.unified_diff}
                    )
                    tree_commit = await self._tree_repository_call(
                        "commit_child",
                        slot=tree_slot,
                        draft=incremental_draft,
                        expected_parent_source_tree_hash=source_context.source_tree_hash,
                    )
                    commit_doc = (
                        tree_commit.to_dict()
                        if callable(getattr(tree_commit, "to_dict", None))
                        else dict(tree_commit)
                        if isinstance(tree_commit, Mapping)
                        else {}
                    )
                    tree_commits_by_node_id[node_id] = dict(commit_doc)
                    canonical_incremental_patch = str(
                        getattr(tree_commit, "incremental_patch", "")
                        or commit_doc.get("incremental_patch")
                        or ""
                    )
                    canonical_cumulative_patch = str(
                        getattr(tree_commit, "cumulative_patch", "")
                        or commit_doc.get("cumulative_patch")
                        or ""
                    )
                    changed_files = tuple(
                        str(item)
                        for item in (
                            getattr(tree_commit, "changed_files", ())
                            or commit_doc.get("changed_files")
                            or ()
                        )
                    )
                    if (
                        not canonical_incremental_patch.startswith("diff --git ")
                        or not canonical_cumulative_patch.startswith("diff --git ")
                        or not changed_files
                    ):
                        raise CodeEditPatchApplyError(
                            "Git-tree commit returned incomplete canonical patches"
                        )
                    if str(commit_doc.get("draft_patch_hash") or "") != drafted_incremental_hash:
                        raise CodeEditPatchApplyError(
                            "Git-tree draft commitment differs from generated child"
                        )
                    incremental_source_diff_hash = sha256_json(
                        {"unified_diff": canonical_incremental_patch}
                    )
                    submission_draft = replace(
                        incremental_draft,
                        target_files=changed_files,
                        unified_diff=canonical_cumulative_patch,
                    )
                    cumulative_source_diff_hash = sha256_json(
                        {"unified_diff": submission_draft.unified_diff}
                    )
                    if str(commit_doc.get("incremental_patch_hash") or "") != incremental_source_diff_hash:
                        raise CodeEditPatchApplyError(
                            "Git-tree incremental patch commitment mismatch"
                        )
                    if str(commit_doc.get("cumulative_patch_hash") or "") != cumulative_source_diff_hash:
                        raise CodeEditPatchApplyError(
                            "Git-tree cumulative patch commitment mismatch"
                        )
                    tree_git_commit = str(commit_doc.get("git_commit") or "")
                    if not re.fullmatch(r"[0-9a-f]{64}", tree_git_commit):
                        raise CodeEditPatchApplyError(
                            "Git-tree child commit is not SHA-256"
                        )
                    tree_depth = int(tree_slot.depth)
                    tree_parent_node_id = str(tree_slot.parent_node_id)
                    tree_parent_dev_score = (
                        tree_parent_candidate.dev_score
                        if tree_parent_candidate is not None
                        else None
                    )
                    tree_parent_feedback_hash = (
                        tree_parent_candidate.dev_feedback_hash
                        if tree_parent_candidate is not None
                        else ""
                    )
                    composition_doc: dict[str, Any] = {
                        "schema_version": "research_lab.git_tree_composition.v1",
                        "mode": "direct_parent_git_commit",
                        "tree_id": tree_id,
                        "node_id": node_id,
                        "root_git_commit": str(root_git_commit),
                        "parent_git_commit": str(
                            commit_doc.get("parent_git_commit") or ""
                        ),
                        "git_commit": tree_git_commit,
                        "branch_objective_path_id": branch_objective_path_id,
                        "branch_objective_hash": branch_objective_hash,
                        "generation_attempt_count": candidate_generation_attempt_count,
                        "root_source_tree_hash": root_source_context.source_tree_hash,
                        "parent_source_tree_hash": source_context.source_tree_hash,
                        "child_source_tree_hash": str(
                            commit_doc.get("source_tree_hash") or ""
                        ),
                        "draft_patch_hash": drafted_incremental_hash,
                        "incremental_source_diff_hash": incremental_source_diff_hash,
                        "cumulative_source_diff_hash": cumulative_source_diff_hash,
                        "cumulative_changed_files": list(changed_files),
                        "cumulative_apply_verified": True,
                    }
                    build_candidate_index = built_candidate_total
                    build_request_doc = {
                        "schema_version": "research_lab.git_tree_build_request.v1",
                        "tree_id": tree_id,
                        "node_id": node_id,
                        "generation_request_hash": generation_request_hash,
                        "policy_hash": tree_policy.policy_hash,
                        "candidate_index": build_candidate_index,
                        "root_artifact_hash": root_artifact.model_artifact_hash,
                        "parent_artifact_hash": (
                            tree_parent_artifact.model_artifact_hash
                        ),
                        "root_source_tree_hash": (
                            root_source_context.source_tree_hash
                        ),
                        "parent_source_tree_hash": source_context.source_tree_hash,
                        "child_source_tree_hash": str(
                            commit_doc.get("source_tree_hash") or ""
                        ),
                        "git_commit": tree_git_commit,
                        "incremental_source_diff_hash": (
                            incremental_source_diff_hash
                        ),
                        "cumulative_source_diff_hash": (
                            cumulative_source_diff_hash
                        ),
                        "branch_objective_path_id": branch_objective_path_id,
                        "branch_objective_hash": branch_objective_hash,
                        "generation_attempt_count": (
                            candidate_generation_attempt_count
                        ),
                        "build_timeout_seconds": build_timeout_seconds,
                        "generation_settled_cost_microusd": max(
                            0,
                            int(actual_cost_microusd)
                            - generation_start_costs.get(node_id, 0),
                        ),
                        "generation_provider_call_count": max(
                            0,
                            int(openrouter_calls)
                            - generation_start_calls.get(node_id, 0),
                        ),
                    }
                    build_request_hash = sha256_json(build_request_doc)
                    build_reservation = await self._tree_repository_call(
                        "reserve_operation",
                        operation_id=build_operation_id(tree_slot),
                        operation_kind="build",
                        request_hash=build_request_hash,
                        node_id=node_id,
                        reservation_doc=build_request_doc,
                    )
                    if (
                        not isinstance(build_reservation, Mapping)
                        or build_reservation.get("operation_status")
                        not in {
                            "reserved",
                            "succeeded",
                            "failed",
                            "indeterminate",
                        }
                    ):
                        raise GitTreeSchedulerError(
                            "tree build reservation response is invalid"
                        )
                    if build_reservation.get("created") is not True:
                        await _recover_generation_operation(
                            slot=tree_slot,
                            request_hash=generation_request_hash,
                        )
                        tree_terminal_persisted = True
                        break
                    build_operation_reserved = True
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_started",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_started"),
                            event_doc={
                                "iteration": iteration,
                                "source_diff_hash": cumulative_source_diff_hash,
                                "git_tree": {
                                    "tree_id": tree_id,
                                    "node_id": node_id,
                                    "root_branch_id": tree_slot.root_branch_id,
                                    "depth": tree_depth,
                                    "child_slot": tree_slot.slot_index,
                                    "git_commit": tree_git_commit,
                                    "branch_objective_path_id": branch_objective_path_id,
                                    "branch_objective_hash": branch_objective_hash,
                                    "generation_attempt_count": candidate_generation_attempt_count,
                                    "root_artifact_hash": root_artifact.model_artifact_hash,
                                    "parent_artifact_hash": tree_parent_artifact.model_artifact_hash,
                                    "parent_node_id": tree_parent_node_id,
                                    "parent_feedback_hash": tree_parent_feedback_hash,
                                    "incremental_source_diff_hash": incremental_source_diff_hash,
                                    "cumulative_source_diff_hash": cumulative_source_diff_hash,
                                },
                                "loop_direction_plan_hash": (
                                    (branch_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(branch_direction_plan_doc, Mapping)
                                    else None
                                ),
                                "plan_alignment": dict(candidate_draft.plan_alignment or {}),
                                "build_timeout_seconds": build_timeout_seconds,
                            },
                        )
                    )
                    # Bug 20: candidate_index is a monotonically increasing per-run build counter
                    # (previously len(selected), which repeats after the post-cap truncation and
                    # overwrote the persisted S3 source-diff artifact key each iteration).
                    build = await self._build_candidate_with_heartbeat(
                        draft=submission_draft,
                        artifact=root_artifact,
                        run_id=run_id,
                        candidate_index=build_candidate_index,
                        source_context=root_source_context,
                        node_id=node_id,
                        iteration=iteration,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                        build_timeout_seconds=build_timeout_seconds,
                    )
                    built_candidate_total += 1
                    build_completed = True
                    if build.source_diff_hash != cumulative_source_diff_hash:
                        raise CodeEditPatchApplyError(
                            "built image source diff differs from Git-tree commitment"
                        )
                    if (
                        build.candidate_model_manifest.model_artifact_hash
                        != str(commit_doc.get("source_tree_hash") or "")
                    ):
                        raise CodeEditPatchApplyError(
                            "built image source differs from the committed Git tree"
                        )
                    lineage = {
                        "schema_version": "research_lab.git_tree_lineage.v1",
                        "tree_id": tree_id,
                        "node_id": node_id,
                        "parent_node_id": tree_parent_node_id,
                        "root_branch_id": tree_slot.root_branch_id,
                        "depth": tree_depth,
                        "child_slot": tree_slot.slot_index,
                        "git_commit": tree_git_commit,
                        "branch_objective_path_id": branch_objective_path_id,
                        "branch_objective_hash": branch_objective_hash,
                        "generation_attempt_count": candidate_generation_attempt_count,
                        "root_artifact_hash": root_artifact.model_artifact_hash,
                        "parent_artifact_hash": tree_parent_artifact.model_artifact_hash,
                        "parent_dev_score": tree_parent_dev_score,
                        "parent_dev_feedback_hash": tree_parent_feedback_hash,
                        "incremental_source_diff_hash": incremental_source_diff_hash,
                        "cumulative_source_diff_hash": cumulative_source_diff_hash,
                        "composition": dict(composition_doc),
                    }
                    build = attach_git_tree_lineage(
                        build=build,
                        draft=submission_draft,
                        root_artifact_hash=root_artifact.model_artifact_hash,
                        lineage=lineage,
                    )
                    built_candidate = BuiltCodeEditCandidate(
                        draft=submission_draft,
                        build=build,
                        node_id=node_id,
                        iteration=iteration,
                        tree_id=tree_id,
                        tree_parent_node_id=tree_parent_node_id,
                        tree_root_branch_id=tree_slot.root_branch_id,
                        tree_depth=tree_depth,
                        tree_child_slot=tree_slot.slot_index,
                        tree_branch_objective_path_id=branch_objective_path_id,
                        tree_branch_objective_hash=branch_objective_hash,
                        tree_generation_attempt_count=candidate_generation_attempt_count,
                        tree_git_commit=tree_git_commit,
                        tree_root_artifact_hash=root_artifact.model_artifact_hash,
                        tree_parent_artifact_hash=tree_parent_artifact.model_artifact_hash,
                        tree_parent_dev_score=tree_parent_dev_score,
                        tree_parent_feedback_hash=tree_parent_feedback_hash,
                        tree_incremental_source_diff_hash=incremental_source_diff_hash,
                        tree_cumulative_source_diff_hash=cumulative_source_diff_hash,
                        tree_composition=composition_doc,
                        tree_settled_cost_microusd=max(
                            0,
                            int(actual_cost_microusd)
                            - generation_start_costs.get(node_id, 0),
                        ),
                    )
                    # Siblings are built completely before the frozen cohort is
                    # evaluated. The immutable artifact makes this intermediate
                    # state restartable without rebuilding or regenerating.
                    built_candidate = await _persist_candidate_rehydration(
                        built_candidate
                    )
                    _replace_selected_candidate(built_candidate)
                    pending_node = _candidate_tree_node(
                        built_candidate, status="evaluating"
                    )
                    await _settle_build_operation(
                        slot=tree_slot,
                        request_hash=build_request_hash,
                        operation_status="succeeded",
                        node=pending_node,
                        reason="candidate_build_and_artifact_verified",
                        candidate=built_candidate,
                    )
                    build_operation_reserved = False
                    tree_scheduler.record_node(pending_node)
                    await _persist_generation_outcome(
                        slot=tree_slot,
                        node=pending_node,
                        reason="candidate_built_awaiting_cohort_evaluation",
                    )
                    tree_terminal_persisted = True
                    rehydration_doc = {
                        "loop_candidate_artifact_uri": (
                            built_candidate.rehydration_artifact_uri
                        ),
                        "loop_candidate_artifact_hash": (
                            built_candidate.rehydration_artifact_hash
                        ),
                    }
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_passed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            candidate_artifact_hash=build.candidate_model_manifest.model_artifact_hash,
                            candidate_patch_hash=sha256_json(build.code_edit_manifest),
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_passed"),
                            event_doc={
                                "iteration": iteration,
                                "candidate_kind": "image_build",
                                "candidate_model_manifest_hash": build.candidate_model_manifest.manifest_hash,
                                "candidate_source_diff_hash": build.source_diff_hash,
                                "build_doc_hash": build.build_doc.get("build_doc_hash"),
                                "loop_direction_plan_hash": (
                                    (branch_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(branch_direction_plan_doc, Mapping)
                                    else None
                                ),
                                "plan_alignment": dict(candidate_draft.plan_alignment or {}),
                                "git_tree": dict(build.build_doc.get("git_tree") or {}),
                                "evaluation_status": "awaiting_cohort",
                                **{
                                    key: value
                                    for key, value in rehydration_doc.items()
                                    if key.startswith("loop_candidate_artifact")
                                },
                            },
                        )
                    )
                    await self._emit_reflection_recorded(
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        draft=candidate_draft,
                        outcome="candidate_build_passed",
                        detail="image built and private tests passed",
                        artifact=artifact,
                        component_registry=component_registry,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                    )
                except CodeEditInfraFailureError:
                    # Registry/auth/network failures are not candidate-quality
                    # failures. Let the hosted worker requeue the paid run.
                    if build_operation_reserved:
                        await _settle_build_operation(
                            slot=tree_slot,
                            request_hash=build_request_hash,
                            operation_status="indeterminate",
                            node=_unbuilt_tree_node(
                                tree_slot, status="indeterminate"
                            ),
                            reason="candidate_build_infrastructure_failure",
                        )
                        build_operation_reserved = False
                    raise
                except (CodeEditPrivateTestError, CodeEditImageBuildError, CodeEditPatchApplyError) as exc:
                    event_type = str(getattr(exc, "failure_stage", "") or "candidate_build_failed")
                    if build_operation_reserved:
                        await _settle_build_operation(
                            slot=tree_slot,
                            request_hash=build_request_hash,
                            operation_status="failed",
                            node=_unbuilt_tree_node(tree_slot, status="failed"),
                            reason=event_type,
                        )
                        build_operation_reserved = False
                    _record_within_run_rejection(
                        stage=event_type,
                        reason=safe_event_error_text(exc),
                        iteration_index=iteration,
                        draft=candidate_draft,
                    )
                    diagnostic_doc = await _write_private_code_edit_diagnostic(
                        artifact=artifact,
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        stage=event_type,
                        draft=candidate_draft,
                        error=exc,
                        artifact_io=self.artifact_io,
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type=event_type,
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, event_type),
                            event_doc={
                                "iteration": iteration,
                                "target_files": list(candidate_draft.target_files),
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                "error": safe_event_error_text(exc),
                                "error_hash": sha256_json({"error": str(exc)}),
                                **diagnostic_doc,
                            },
                        )
                    )
                    await self._emit_reflection_recorded(
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        draft=candidate_draft,
                        outcome=event_type,
                        detail=str(exc),
                        artifact=artifact,
                        component_registry=component_registry,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                    )
                except CodeEditBuildError as exc:
                    if build_operation_reserved:
                        await _settle_build_operation(
                            slot=tree_slot,
                            request_hash=build_request_hash,
                            operation_status="failed",
                            node=_unbuilt_tree_node(tree_slot, status="failed"),
                            reason="candidate_build_failed",
                        )
                        build_operation_reserved = False
                    _record_within_run_rejection(
                        stage="candidate_build_failed",
                        reason=safe_event_error_text(exc),
                        iteration_index=iteration,
                        draft=candidate_draft,
                    )
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_failed"),
                            event_doc={
                                "iteration": iteration,
                                "target_files": list(candidate_draft.target_files),
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                "error": safe_event_error_text(exc),
                                "error_hash": sha256_json({"error": str(exc)}),
                            },
                        )
                    )
                    await self._emit_reflection_recorded(
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        draft=candidate_draft,
                        outcome="candidate_build_failed",
                        detail=str(exc),
                        artifact=artifact,
                        component_registry=component_registry,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                    )
                except Exception as exc:
                    if build_completed and not tree_terminal_persisted:
                        raise
                    # Bug 17: an unexpected infra error during build/event emission used to
                    # abort the run. Contain it to this candidate; the run keeps whatever it
                    # has already built.
                    if not _stage_error_containment_enabled():
                        raise
                    if not build_completed:
                        if build_operation_reserved:
                            await _settle_build_operation(
                                slot=tree_slot,
                                request_hash=build_request_hash,
                                operation_status="indeterminate",
                                node=_unbuilt_tree_node(
                                    tree_slot, status="indeterminate"
                                ),
                                reason="candidate_build_unexpected_error",
                            )
                            build_operation_reserved = False
                        _record_within_run_rejection(
                            stage="candidate_build_unexpected_error",
                            reason=safe_event_error_text(exc),
                            iteration_index=iteration,
                            draft=candidate_draft,
                        )
                    try:
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="candidate_build_failed" if not build_completed else "code_edit_validation_failed",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                node_id=node_id,
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "candidate_build_unexpected_error",
                                ),
                                event_doc={
                                    "iteration": iteration,
                                    "stage": (
                                        "candidate_build_unexpected_error"
                                        if not build_completed
                                        else "post_build_event_emit_failed"
                                    ),
                                    "target_files": list(candidate_draft.target_files),
                                    "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                    "error": safe_event_error_text(exc),
                                    "error_hash": sha256_json({"error": str(exc)}),
                                },
                            )
                        )
                    except Exception as event_exc:
                        logger.warning(
                            "research_lab_git_tree_build_failure_event_emit_failed "
                            "node_id=%s build_completed=%s error=%s",
                            str(node_id or "")[:120],
                            build_completed,
                            safe_event_error_text(event_exc),
                        )
                    if not build_completed:
                        await self._emit_reflection_recorded(
                            run_id=run_id,
                            node_id=node_id,
                            iteration=iteration,
                            draft=candidate_draft,
                            outcome="candidate_build_unexpected_error",
                            detail=str(exc),
                            artifact=artifact,
                            component_registry=component_registry,
                            elapsed=elapsed,
                            openrouter_calls=openrouter_calls,
                            estimated_cost=estimated_cost,
                            actual_cost_microusd=actual_cost_microusd,
                        )
                        await _terminalize_unbuilt_slot(
                            tree_slot,
                            "candidate_build_unexpected_error",
                            status="indeterminate",
                        )
            if (
                len(selected) == built_nodes_before_iteration
            ):
                await _terminalize_unbuilt_slot(
                    tree_slot, "tree_node_produced_no_build"
                )
            last_checkpoint = await self._emit_checkpoint(
                run_id=run_id,
                settings=settings,
                artifact=artifact,
                model_id=model_id,
                budget_context=budget_context,
                iterations_completed=iteration,
                elapsed_seconds=elapsed(),
                selected=selected,
                provider_usage=provider_usage,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                stage="code_edit_iteration_completed",
                loop_direction_plan=loop_direction_plan_doc,
                built_candidate_count=built_candidate_total,
                probe_budget=probe_budget,
                planner_reference_repair_attempted=reference_repair_attempted,
                planner_reference_repair_status=reference_repair_status,
            )
            if should_pause and await should_pause():
                _cleanup_source_tmp()
                return self._result(
                    selected=selected,
                    status="paused",
                    stop_reason="maintenance_pause_requested",
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    openrouter_calls=openrouter_calls,
                    provider_usage=provider_usage,
                    checkpoint=last_checkpoint,
                )
            if finalization_reserve_reached:
                stop_reason = "tree_finalization_reserve"
                break
            if budget_exhausted_after_call:
                stop_reason = "compute_budget_exhausted_after_code_edit"
                break
        if not tree_scheduler.planned_slots and any(
            node.status == "evaluating" for node in tree_scheduler.nodes
        ):
            if await _evaluate_pending_round():
                last_checkpoint = await self._emit_checkpoint(
                    run_id=run_id,
                    settings=settings,
                    artifact=artifact,
                    model_id=model_id,
                    budget_context=budget_context,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    selected=selected,
                    provider_usage=provider_usage,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    stage="tree_round_evaluated_before_finalization",
                    loop_direction_plan=loop_direction_plan_doc,
                    built_candidate_count=built_candidate_total,
                    probe_budget=probe_budget,
                    planner_reference_repair_attempted=reference_repair_attempted,
                    planner_reference_repair_status=reference_repair_status,
                )
        if selected:
            remaining_minimum = settings.min_seconds - elapsed()
            remaining_before_final_context = (
                settings.max_seconds
                - elapsed()
                - final_context_reserve_seconds
            )
            if _min_runtime_skip_when_selected_enabled():
                # Candidates are already selected; parking the worker slot until min_seconds
                # elapses is pure waste. Proceed straight to finalization.
                remaining_minimum = 0.0
            if remaining_minimum > 0 and remaining_before_final_context > 0:
                sleep_remaining = min(
                    remaining_minimum, remaining_before_final_context
                )
                while sleep_remaining > 0:
                    await asyncio.sleep(min(5.0, sleep_remaining))
                    sleep_remaining = min(
                        settings.min_seconds - elapsed(),
                        settings.max_seconds
                        - elapsed()
                        - final_context_reserve_seconds,
                    )
                    if should_pause and await should_pause():
                        break
            if should_pause and await should_pause():
                last_checkpoint = await self._emit_checkpoint(
                    run_id=run_id,
                    settings=settings,
                    artifact=artifact,
                    model_id=model_id,
                    budget_context=budget_context,
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    selected=selected,
                    provider_usage=provider_usage,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    stage="pause_after_code_edit_minimum_runtime",
                    loop_direction_plan=loop_direction_plan_doc,
                    built_candidate_count=built_candidate_total,
                    probe_budget=probe_budget,
                    planner_reference_repair_attempted=reference_repair_attempted,
                    planner_reference_repair_status=reference_repair_status,
                )
                _cleanup_source_tmp()
                return self._result(
                    selected=selected,
                    status="paused",
                    stop_reason="maintenance_pause_requested",
                    iterations_completed=iteration,
                    elapsed_seconds=elapsed(),
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                    openrouter_calls=openrouter_calls,
                    provider_usage=provider_usage,
                    checkpoint=last_checkpoint,
                )
        shortlist_nodes = tree_scheduler.shortlist()
        shortlist_candidates = [
            candidates_by_node_id[node.node_id]
            for node in shortlist_nodes
            if node.node_id in candidates_by_node_id
        ]
        finalist_pool: tuple[TreeNode, ...] = ()
        if shortlist_candidates:
            final_evaluations = await _evaluate_tree_cohort(
                shortlist_candidates,
                stage="final_shortlist",
            )
            finalist_pool = tuple(
                _candidate_tree_node(candidate)
                for candidate in final_evaluations
            )

        try:
            finalist_node = select_finalist(finalist_pool)
        except ValueError as exc:
            logger.warning(
                "research_lab_tree_final_context_mismatch run_id=%s error=%s",
                run_id,
                safe_event_error_text(exc),
            )
            finalist_node = None
            stop_reason = "tree_final_evaluation_context_mismatch"
        selected = (
            [candidates_by_node_id[finalist_node.node_id]]
            if finalist_node is not None
            and finalist_node.node_id in candidates_by_node_id
            else []
        )
        if (
            not selected
            and tree_scheduler.nodes
            and stop_reason in {"max_iterations", "tree_frontier_exhausted"}
        ):
            stop_reason = "no_eligible_tree_finalist"
        git_bundle: dict[str, Any] = {}
        if selected:
            published_bundle = await self._tree_repository_call("publish_bundle")
            if (
                not isinstance(published_bundle, Mapping)
                or published_bundle.get("readback_verified") is not True
                or not str(published_bundle.get("bundle_uri") or "").startswith(
                    "s3://"
                )
                or not re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(published_bundle.get("bundle_hash") or ""),
                )
            ):
                raise GitTreeSchedulerError(
                    "Git-tree bundle was not durably verified"
                )
            git_bundle = dict(published_bundle)
        evaluation_summary = summarize_tree_evaluations(tree_scheduler.nodes)
        selected_git_tree = (
            dict(selected[0].build.build_doc.get("git_tree") or {})
            if selected
            else {}
        )
        if selected and not selected_git_tree:
            raise GitTreeSchedulerError(
                "selected Git-tree candidate has no committed lineage"
            )
        tree_selection = {
            "schema_version": "research_lab.git_tree_selection.v1",
            "tree_id": tree_id,
            "policy_hash": tree_policy.policy_hash,
            "root_git_commit": str(root_git_commit),
            "node_count": len(tree_scheduler.nodes),
            "built_node_count": int(evaluation_summary["built_node_count"]),
            "eligible_node_count": int(evaluation_summary["eligible_node_count"]),
            "evaluation_summary": evaluation_summary,
            "shortlist_node_ids": [node.node_id for node in shortlist_nodes],
            "selected_node_id": selected[0].node_id if selected else "",
            "selected_candidate_artifact_hash": (
                selected[0].build.candidate_model_manifest.model_artifact_hash
                if selected
                else ""
            ),
            "selected_node_git_commit": (
                selected[0].tree_git_commit if selected else ""
            ),
            "selected_lineage_hash": (
                sha256_json(selected_git_tree) if selected else ""
            ),
            "frontier_hash": tree_scheduler.frontier_hash,
            "paid_finalist_count": len(selected),
            "git_bundle": git_bundle,
        }
        if selected:
            await self._tree_repository_call(
                "select_final",
                selection_hash=sha256_json(tree_selection),
                selection_doc=tree_selection,
            )
        for candidate in selected:
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="candidate_selected",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    node_id=candidate.node_id,
                    candidate_artifact_hash=candidate.build.candidate_model_manifest.model_artifact_hash,
                    candidate_patch_hash=sha256_json(candidate.build.code_edit_manifest),
                    cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_selected"),
                    event_doc={
                        "candidate_index": 0,
                        "iteration": candidate.iteration,
                        "candidate_kind": "image_build",
                        "candidate_model_manifest_hash": candidate.build.candidate_model_manifest.manifest_hash,
                        "candidate_source_diff_hash": candidate.build.source_diff_hash,
                        "redacted_summary": candidate.draft.redacted_summary,
                        "git_tree_selection": dict(tree_selection),
                        "paid_scoring_candidate": True,
                        "git_tree": {
                            "tree_id": candidate.tree_id,
                            "node_id": candidate.node_id,
                            "parent_node_id": candidate.tree_parent_node_id,
                            "root_branch_id": candidate.tree_root_branch_id,
                            "depth": candidate.tree_depth,
                            "child_slot": candidate.tree_child_slot,
                            "branch_objective_path_id": candidate.tree_branch_objective_path_id,
                            "branch_objective_hash": candidate.tree_branch_objective_hash,
                            "generation_attempt_count": candidate.tree_generation_attempt_count,
                            "git_commit": candidate.tree_git_commit,
                            "root_artifact_hash": candidate.tree_root_artifact_hash,
                            "parent_artifact_hash": candidate.tree_parent_artifact_hash,
                            "parent_feedback_hash": candidate.tree_parent_feedback_hash,
                            "incremental_source_diff_hash": candidate.tree_incremental_source_diff_hash,
                            "cumulative_source_diff_hash": candidate.tree_cumulative_source_diff_hash,
                            "settled_cost_microusd": (
                                candidate.tree_settled_cost_microusd
                            ),
                        },
                        # P18: dev-eval ranking scores become queryable from
                        # the event stream (previously only in the S3
                        # rehydration artifact) — the dev-vs-live divergence
                        # calibration signal for the cheap rung.
                        **(
                            {
                                "dev_score": round(float(candidate.dev_score), 6),
                                "dev_score_version": str(candidate.dev_score_version)[:120],
                            }
                            if candidate.dev_score is not None
                            else {}
                        ),
                        "loop_direction_plan_hash": candidate.tree_branch_objective_hash,
                        "selected_path_id": candidate.tree_branch_objective_path_id,
                        "plan_alignment": dict(candidate.draft.plan_alignment or {}),
                        **(
                            {
                                "dev_score": candidate.dev_score,
                                "dev_score_version": candidate.dev_score_version,
                                "dev_score_ranking_only": True,
                            }
                            if candidate.dev_score is not None
                            else {}
                        ),
                    },
                )
            )

        last_checkpoint = await self._emit_checkpoint(
            run_id=run_id,
            settings=settings,
            artifact=artifact,
            model_id=model_id,
            budget_context=budget_context,
            iterations_completed=iteration,
            elapsed_seconds=elapsed(),
            selected=selected,
            provider_usage=provider_usage,
            openrouter_calls=openrouter_calls,
            estimated_cost=estimated_cost,
            actual_cost_microusd=actual_cost_microusd,
            stage="tree_final_selection_committed",
            loop_direction_plan=loop_direction_plan_doc,
            built_candidate_count=built_candidate_total,
            probe_budget=probe_budget,
            planner_reference_repair_attempted=reference_repair_attempted,
            planner_reference_repair_status=reference_repair_status,
        )

        if not selected:
            checkpoint_doc = dict(
                (last_checkpoint or {}).get("git_tree_checkpoint") or {}
            )
            tree_failure = {
                "schema_version": "research_lab.git_tree_failed.v1",
                "tree_id": tree_id,
                "stop_reason": str(stop_reason or "no_eligible_tree_finalist"),
                "selected_node_id": "",
                "paid_finalist_count": 0,
                "node_count": len(tree_scheduler.nodes),
                "eligible_node_count": sum(
                    1 for node in tree_scheduler.nodes if node.eligible
                ),
                "frontier_hash": tree_scheduler.frontier_hash,
                "checkpoint_hash": sha256_json(checkpoint_doc),
            }
            await self._tree_repository_call(
                "fail_tree",
                failure_hash=sha256_json(tree_failure),
                failure_doc=tree_failure,
            )

        result = self._result(
            selected=selected,
            status="completed" if selected else "failed",
            stop_reason=stop_reason,
            iterations_completed=iteration,
            elapsed_seconds=elapsed(),
            estimated_cost=estimated_cost,
            actual_cost_microusd=actual_cost_microusd,
            openrouter_calls=openrouter_calls,
            provider_usage=provider_usage,
            checkpoint=last_checkpoint,
        )
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="loop_completed" if selected else "loop_failed",
                loop_status="completed" if selected else "failed",
                elapsed_seconds=result.elapsed_seconds,
                provider_usage=list(result.provider_usage),
                cost_ledger=result.cost_ledger(),
                event_doc={
                    "candidate_kind": "image_build",
                    "iterations_completed": result.iterations_completed,
                    "selected_candidate_count": len(selected),
                    "stop_reason": result.stop_reason,
                    "git_tree_policy": dict(self._tree_policy_doc),
                    "git_tree_selection": dict(tree_selection),
                    "loop_direction_plan_hash": (
                        (loop_direction_plan_doc or {}).get("plan_hash")
                        if isinstance(loop_direction_plan_doc, Mapping)
                        else None
                    ),
                    "selected_path_id": (
                        selected[0].tree_branch_objective_path_id
                        if selected
                        else None
                    ),
                    "selected_branch_objective_hash": (
                        selected[0].tree_branch_objective_hash
                        if selected
                        else None
                    ),
                    "gateway_scoring_queue_visible_after_this_event": bool(selected),
                    # P14 / v5 §8.3 run-summary contract: every run exit path
                    # terminates with this fixed machine-parseable block; a
                    # terminal event WITHOUT it is mechanically classifiable
                    # as a crash by the projector.
                    "run_summary": {
                        "schema_version": "1.0",
                        "status": "completed" if selected else "failed",
                        "stop_reason": result.stop_reason,
                        "iterations_completed": result.iterations_completed,
                        "selected_candidate_count": len(selected),
                        "wall_clock_seconds": round(float(result.elapsed_seconds), 3),
                        "cost_ledger": result.cost_ledger(),
                        "openrouter_call_count": result.openrouter_call_count,
                    },
                },
            )
        )
        _cleanup_source_tmp()
        return result

    async def _emit_checkpoint(
        self,
        *,
        run_id: str,
        settings: AutoResearchRuntimeSettings,
        artifact: PrivateModelArtifactManifest,
        model_id: str,
        budget_context: Mapping[str, Any],
        iterations_completed: int,
        elapsed_seconds: float,
        selected: Sequence[BuiltCodeEditCandidate],
        provider_usage: Sequence[Mapping[str, Any]],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        stage: str,
        loop_direction_plan: Mapping[str, Any] | None = None,
        built_candidate_count: int = 0,
        probe_budget: ProbeBudgetState | None = None,
        planner_reference_repair_attempted: bool = False,
        planner_reference_repair_status: str = "",
    ) -> dict[str, Any]:
        tree_scheduler = self._active_tree_scheduler
        tree_policy = self._active_tree_policy
        if (
            tree_scheduler is None
            or tree_policy is None
            or not self._active_tree_id
        ):
            raise GitTreeSchedulerError("tree checkpoint authority is unavailable")
        operation_commitment = await self._tree_repository_call(
            "operation_settlement_commitment"
        )
        if (
            not isinstance(operation_commitment, Mapping)
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(
                    operation_commitment.get("operation_settlement_hash")
                    or ""
                ),
            )
        ):
            raise GitTreeSchedulerError(
                "tree operation settlement commitment is invalid"
            )
        operation_settlement_hash = str(
            operation_commitment["operation_settlement_hash"]
        )
        tree_checkpoint = TreeCheckpoint(
            tree_id=self._active_tree_id,
            root_artifact_hash=artifact.model_artifact_hash,
            policy=tree_policy,
            nodes=tree_scheduler.nodes,
            planned_slots=tree_scheduler.planned_slots,
            frontier_hash=tree_scheduler.frontier_hash,
            operation_settlement_hash=operation_settlement_hash,
            selected_node_id=(
                selected[0].node_id
                if stage == "tree_final_selection_committed" and len(selected) == 1
                else ""
            ),
            stop_reason=str(stage),
        )
        tree_checkpoint_doc = tree_checkpoint.to_dict()
        await self._tree_repository_call(
            "commit_checkpoint",
            checkpoint_hash=sha256_json(tree_checkpoint_doc),
            checkpoint_doc=tree_checkpoint_doc,
        )
        payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "stage": stage,
            "candidate_kind": "image_build",
            "model_id": model_id,
            "artifact_hash": artifact.model_artifact_hash,
            "manifest_hash": artifact.manifest_hash,
            "settings": _settings_doc(settings),
            "budget_context": _safe_budget_doc(budget_context),
            "git_tree_id": self._active_tree_id,
            "git_tree_root_commit": self._active_tree_root_git_commit,
            "git_tree_evaluator_commitment": dict(
                self._tree_policy_doc.get("evaluator_commitment") or {}
            ),
            "git_tree_operation_settlements": dict(operation_commitment),
            "git_tree_checkpoint": tree_checkpoint_doc,
            "iterations_completed": int(iterations_completed),
            "next_iteration": int(iterations_completed) + 1,
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "built_candidate_count": max(0, int(built_candidate_count)),
            "selected_candidates": [
                {
                    "node_id": candidate.node_id,
                    "iteration": candidate.iteration,
                    "candidate_artifact_hash": candidate.build.candidate_model_manifest.model_artifact_hash,
                    "candidate_model_manifest_hash": candidate.build.candidate_model_manifest.manifest_hash,
                    "candidate_source_diff_hash": candidate.build.source_diff_hash,
                    "build_doc_hash": candidate.build.build_doc.get("build_doc_hash"),
                    "draft": _redacted_draft_doc(candidate.draft),
                    # Bug 5: private artifact refs (hashes/URIs only — never raw diffs or
                    # manifests) that let resume rehydrate this already-built candidate.
                    "rehydration_artifact_uri": candidate.rehydration_artifact_uri or None,
                    "rehydration_artifact_hash": candidate.rehydration_artifact_hash or None,
                    "source_diff_artifact_uri": candidate.build.build_doc.get("source_diff_artifact_uri"),
                    "source_diff_artifact_hash": candidate.build.build_doc.get("source_diff_artifact_hash"),
                    # §6.3-1: present only when dev-eval scored this candidate, so
                    # dev-eval-off checkpoints keep the exact pre-dev-eval shape.
                    **(
                        {
                            "dev_score": candidate.dev_score,
                            "dev_score_version": candidate.dev_score_version,
                            "dev_evaluation": dict(candidate.dev_evaluation),
                        }
                        if candidate.dev_evaluation
                        else {}
                    ),
                    "git_tree": {
                        "tree_id": candidate.tree_id,
                        "parent_node_id": candidate.tree_parent_node_id,
                        "root_branch_id": candidate.tree_root_branch_id,
                        "depth": candidate.tree_depth,
                        "child_slot": candidate.tree_child_slot,
                        "branch_objective_path_id": candidate.tree_branch_objective_path_id,
                        "branch_objective_hash": candidate.tree_branch_objective_hash,
                        "generation_attempt_count": candidate.tree_generation_attempt_count,
                        "git_commit": candidate.tree_git_commit,
                        "root_artifact_hash": candidate.tree_root_artifact_hash,
                        "parent_artifact_hash": candidate.tree_parent_artifact_hash,
                        "parent_dev_score": candidate.tree_parent_dev_score,
                        "parent_feedback_hash": candidate.tree_parent_feedback_hash,
                        "dev_feedback_hash": candidate.dev_feedback_hash,
                        "incremental_source_diff_hash": candidate.tree_incremental_source_diff_hash,
                        "cumulative_source_diff_hash": candidate.tree_cumulative_source_diff_hash,
                        "settled_cost_microusd": (
                            candidate.tree_settled_cost_microusd
                        ),
                    },
                }
                for candidate in selected
            ],
            "loop_direction_plan": dict(loop_direction_plan or {}),
            "loop_direction_plan_hash": (
                (loop_direction_plan or {}).get("plan_hash")
                if isinstance(loop_direction_plan, Mapping)
                else None
            ),
            "openrouter_call_count": int(openrouter_calls),
            "estimated_cost_usd": round(float(estimated_cost), 6),
            "actual_openrouter_cost_usd": round(int(actual_cost_microusd) / 1_000_000, 6),
            "actual_openrouter_cost_microusd": int(actual_cost_microusd),
            "provider_usage": [dict(item) for item in provider_usage if isinstance(item, Mapping)],
            # W4 probe accounting survives pause/resume (keys absent pre-probe
            # so probe-off checkpoints keep their exact prior shape).
            **(
                {
                    "probe_count": int(probe_budget.probes_used),
                    "probe_cost_microusd": int(probe_budget.cost_used_microusd),
                }
                if probe_budget is not None
                else {}
            ),
            **(
                {
                    "planner_reference_repair_attempted": True,
                    "planner_reference_repair_status": str(
                        planner_reference_repair_status or "attempted"
                    )[:80],
                }
                if planner_reference_repair_attempted
                else {}
            ),
        }
        checkpoint = {**payload, "checkpoint_hash": sha256_json(payload)}
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="checkpoint_saved",
                loop_status="running",
                elapsed_seconds=elapsed_seconds,
                provider_usage=[dict(item) for item in provider_usage if isinstance(item, Mapping)],
                cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "checkpoint_saved"),
                event_doc={"checkpoint": checkpoint},
            )
        )
        return checkpoint

    def _result(
        self,
        *,
        selected: Sequence[BuiltCodeEditCandidate],
        status: str,
        stop_reason: str,
        iterations_completed: int,
        elapsed_seconds: float,
        estimated_cost: float,
        actual_cost_microusd: int,
        openrouter_calls: int,
        provider_usage: Sequence[Mapping[str, Any]],
        checkpoint: dict[str, Any] | None,
    ) -> CodeEditLoopResult:
        normalized_status = str(status or "").strip().lower()
        if normalized_status == "completed":
            if len(selected) != 1:
                raise GitTreeContractError(
                    "completed tree loop requires exactly one finalist"
                )
            result_selected = tuple(selected)
        else:
            result_selected = ()
        if not isinstance(checkpoint, Mapping):
            raise GitTreeContractError("tree loop result checkpoint is missing")
        tree_checkpoint_doc = checkpoint.get("git_tree_checkpoint")
        if not isinstance(tree_checkpoint_doc, Mapping):
            raise GitTreeContractError(
                "tree loop result checkpoint has no tree commitment"
            )
        tree_checkpoint = TreeCheckpoint.from_mapping(tree_checkpoint_doc)
        tree_result = TreeResult(
            tree_id=tree_checkpoint.tree_id,
            status=normalized_status,
            stop_reason=str(stop_reason),
            selected_node_id=(
                result_selected[0].node_id if result_selected else ""
            ),
            nodes=tree_checkpoint.nodes,
            checkpoint=tree_checkpoint,
        )
        # Terminal marker on the run trace: outcome, iteration count, and
        # finalist count — the span the dashboard sorts/filters runs by.
        run_id = self._langfuse_run_id
        with langfuse_observation(
            "research_lab.loop_run_completed",
            metadata={
                "run_id": run_id,
                "loop_status": normalized_status,
                "stop_reason": str(stop_reason)[:120],
                "iterations_completed": int(iterations_completed),
                "candidate_count": len(result_selected),
            },
            trace_id=langfuse_run_trace_id(run_id),
            sample_seed=run_id or None,
        ):
            pass
        return CodeEditLoopResult(
            selected_candidates=result_selected,
            iterations_completed=int(iterations_completed),
            stop_reason=stop_reason,
            elapsed_seconds=round(float(elapsed_seconds), 3),
            estimated_cost_usd=round(float(estimated_cost), 6),
            actual_openrouter_cost_usd=round(int(actual_cost_microusd) / 1_000_000, 6),
            actual_openrouter_cost_microusd=int(actual_cost_microusd),
            openrouter_call_count=int(openrouter_calls),
            tree_result=tree_result,
            provider_usage=tuple(dict(item) for item in provider_usage if isinstance(item, Mapping)),
            status=normalized_status,
            checkpoint_doc=dict(checkpoint),
        )

    async def _judge_plan_alignment(
        self,
        *,
        draft: CodeEditDraft,
        loop_direction_plan: Mapping[str, Any] | None,
        prior_attempts: Sequence[Mapping[str, Any]],
        node_id: str,
        iteration: int,
        settings: AutoResearchRuntimeSettings,
        budget_limit_microusd: int,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        provider_usage: list[dict[str, Any]],
    ) -> tuple[bool, CodeEditDraft, int, float, int, bool]:
        if not loop_direction_plan:
            return True, draft, openrouter_calls, estimated_cost, actual_cost_microusd, False
        heuristic_errors = code_edit_plan_alignment_errors(
            draft,
            loop_direction_plan=loop_direction_plan,
            prior_attempts=prior_attempts,
            strict=bool(self.builder.config.loop_novelty_strict),
        )
        if heuristic_errors:
            verdict_doc = {
                "schema_version": "1.0",
                "verdict": "fail",
                "source": "local_heuristic",
                "reason": "; ".join(heuristic_errors)[:700],
                "detected_lane": draft.lane,
                "detected_mechanism": draft.mechanism,
                "novel": not any("duplicate" in error for error in heuristic_errors),
                "blocking_issue": heuristic_errors[0],
                "confidence": 1.0,
                "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                "selected_path_id": loop_direction_plan.get("selected_path_id"),
            }
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="plan_alignment_judged",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    node_id=node_id,
                    provider_usage=([provider_usage[-1]] if provider_usage else []),
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "plan_alignment_judged",
                    ),
                    event_doc={
                        "iteration": iteration,
                        "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                        "selected_path_id": loop_direction_plan.get("selected_path_id"),
                        "source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                        "verdict": verdict_doc,
                    },
                )
            )
            await self._emit_alignment_rejection(
                draft=draft,
                node_id=node_id,
                iteration=iteration,
                verdict_doc=verdict_doc,
                loop_direction_plan=loop_direction_plan,
                elapsed=elapsed,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                provider_usage=provider_usage,
            )
            return False, replace(draft, plan_alignment=verdict_doc), openrouter_calls, estimated_cost, actual_cost_microusd, False

        if not self.builder.config.loop_alignment_judge_enabled:
            verdict_doc = {
                "schema_version": "1.0",
                "verdict": "pass",
                "source": "local_heuristic",
                "reason": "alignment judge disabled; local heuristic passed",
                "detected_lane": draft.lane,
                "detected_mechanism": draft.mechanism,
                "novel": True,
                "blocking_issue": "",
                "confidence": 0.5,
                "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                "selected_path_id": loop_direction_plan.get("selected_path_id"),
            }
            return True, replace(draft, plan_alignment=verdict_doc), openrouter_calls, estimated_cost, actual_cost_microusd, False

        if elapsed() >= settings.max_seconds:
            return False, draft, openrouter_calls, estimated_cost, actual_cost_microusd, False
        if _would_exceed_budget(
            actual_cost_microusd,
            _estimated_call_microusd(settings.estimated_iteration_cost_usd),
            budget_limit_microusd,
        ):
            await self._emit_alignment_rejection(
                draft=draft,
                node_id=node_id,
                iteration=iteration,
                verdict_doc={
                    "schema_version": "1.0",
                    "verdict": "fail",
                    "source": "budget_guard",
                    "reason": "compute budget exhausted before plan alignment judge",
                    "blocking_issue": "compute_budget_exhausted_before_plan_alignment_judge",
                    "novel": True,
                    "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                    "selected_path_id": loop_direction_plan.get("selected_path_id"),
                },
                loop_direction_plan=loop_direction_plan,
                elapsed=elapsed,
                openrouter_calls=openrouter_calls,
                estimated_cost=estimated_cost,
                actual_cost_microusd=actual_cost_microusd,
                provider_usage=provider_usage,
            )
            return False, draft, openrouter_calls, estimated_cost, actual_cost_microusd, True

        # Bug 21: a judge parse failure used to be recorded as a hard rejection at
        # confidence 1.0. Retry the judge once on a failed/unparseable call; if it is still
        # unusable, accept neutrally on the already-passed local heuristics instead of
        # recording a confident rejection.
        judge_attempt_limit = 2 if _judge_parse_soft_skip_enabled() else 1
        verdict_doc: dict[str, Any] | None = None
        raw_judge = ""
        judge_failure_reason = ""
        for judge_attempt in range(1, judge_attempt_limit + 1):
            if judge_attempt > 1 and (
                elapsed() >= settings.max_seconds
                or _would_exceed_budget(
                    actual_cost_microusd,
                    _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                    budget_limit_microusd,
                )
            ):
                break
            remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
            judge_result, judge_call_error = await self._call_stage_contained(
                build_plan_alignment_judge_messages(
                    loop_direction_plan=loop_direction_plan,
                    draft=draft,
                    prior_attempts=prior_attempts,
                ),
                min(settings.draft_timeout_seconds, remaining_call_seconds),
                self.builder.config.loop_alignment_judge_max_tokens,
                "plan_alignment_judge",
            )
            if judge_result is None:
                judge_failure_reason = judge_call_error or "plan_alignment_judge_call_failed"
                # P3: keep the failed judge call's telemetry pointer + cost.
                if isinstance(judge_call_error, ContainedStageFailure):
                    actual_cost_microusd += judge_call_error.cost_microusd
                    provider_usage.extend(judge_call_error.failure_usage_entries())
                continue
            raw_judge = judge_result.content
            openrouter_calls += 1
            estimated_cost += settings.estimated_iteration_cost_usd
            actual_cost_microusd += max(0, int(judge_result.cost_microusd))
            if judge_result.provider_usage:
                provider_usage.append({**judge_result.provider_usage, "loop_iteration": iteration, "call_stage": "plan_alignment_judge"})
            try:
                verdict = parse_plan_alignment_judge_response(raw_judge)
                verdict_doc = {**verdict.to_dict(), "source": "model_judge"}
                break
            except Exception as exc:
                judge_failure_reason = safe_event_error_text(exc)
        if verdict_doc is None:
            if _judge_parse_soft_skip_enabled():
                verdict_doc = {
                    "schema_version": "1.0",
                    "verdict": "pass",
                    "source": "model_judge_unavailable",
                    "reason": (
                        "plan alignment judge unavailable or unparseable after retry; "
                        "accepted on local heuristics: " + judge_failure_reason
                    )[:700],
                    "detected_lane": draft.lane,
                    "detected_mechanism": draft.mechanism,
                    "novel": True,
                    "blocking_issue": "",
                    "confidence": 0.0,
                    "raw_response_hash": sha256_json({"raw_response": raw_judge}),
                }
            else:
                verdict_doc = {
                    "schema_version": "1.0",
                    "verdict": "fail",
                    "source": "model_judge_parse_failed",
                    "reason": judge_failure_reason,
                    "detected_lane": "",
                    "detected_mechanism": "",
                    "novel": False,
                    "blocking_issue": "plan_alignment_judge_parse_failed",
                    "confidence": 1.0,
                    "raw_response_hash": sha256_json({"raw_response": raw_judge}),
                }
        verdict_doc["loop_direction_plan_hash"] = loop_direction_plan.get("plan_hash")
        verdict_doc["selected_path_id"] = loop_direction_plan.get("selected_path_id")
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="plan_alignment_judged",
                loop_status="running",
                elapsed_seconds=elapsed(),
                node_id=node_id,
                provider_usage=([provider_usage[-1]] if provider_usage else []),
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "plan_alignment_judged",
                ),
                event_doc={
                    "iteration": iteration,
                    "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                    "selected_path_id": loop_direction_plan.get("selected_path_id"),
                    "source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                    "verdict": verdict_doc,
                },
            )
        )
        accepted = verdict_doc.get("verdict") == "pass" and verdict_doc.get("novel") is not False
        judged_draft = replace(draft, plan_alignment=verdict_doc)
        if accepted:
            return True, judged_draft, openrouter_calls, estimated_cost, actual_cost_microusd, False
        await self._emit_alignment_rejection(
            draft=judged_draft,
            node_id=node_id,
            iteration=iteration,
            verdict_doc=verdict_doc,
            loop_direction_plan=loop_direction_plan,
            elapsed=elapsed,
            openrouter_calls=openrouter_calls,
            estimated_cost=estimated_cost,
            actual_cost_microusd=actual_cost_microusd,
            provider_usage=provider_usage,
        )
        return False, judged_draft, openrouter_calls, estimated_cost, actual_cost_microusd, False

    async def _emit_alignment_rejection(
        self,
        *,
        draft: CodeEditDraft,
        node_id: str,
        iteration: int,
        verdict_doc: Mapping[str, Any],
        loop_direction_plan: Mapping[str, Any],
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        provider_usage: Sequence[Mapping[str, Any]],
    ) -> None:
        await self.event_sink(
            AutoResearchLoopEvent(
                event_type="code_edit_alignment_rejected",
                loop_status="running",
                elapsed_seconds=elapsed(),
                node_id=node_id,
                provider_usage=([dict(provider_usage[-1])] if provider_usage else []),
                cost_ledger=_running_cost_ledger(
                    openrouter_calls,
                    estimated_cost,
                    actual_cost_microusd,
                    "code_edit_alignment_rejected",
                ),
                event_doc={
                    "iteration": iteration,
                    "lane": draft.lane,
                    "plan_path_id": draft.plan_path_id,
                    "target_files": list(draft.target_files),
                    "source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                    "loop_direction_plan_hash": loop_direction_plan.get("plan_hash"),
                    "selected_path_id": loop_direction_plan.get("selected_path_id"),
                    "verdict": dict(verdict_doc),
                },
            )
        )

    async def _emit_reflection_recorded(
        self,
        *,
        run_id: str,
        node_id: str | None,
        iteration: int,
        draft: CodeEditDraft | None,
        outcome: str,
        detail: str,
        artifact: PrivateModelArtifactManifest,
        component_registry: Mapping[str, Any],
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
    ) -> None:
        """§9.1 item 4: record a mechanical reflection after a judge/build outcome.

        The reflection is derived mechanically from the judge verdict / build
        error / within-run rejection reason — no extra LLM call — and doubles
        as the §9.5 lesson-store input (`reflection_recorded` is already in the
        scripts/34 event-type allowlist; the §9.1 projector maps it to
        ``NODE_REFLECTED``). Strictly best-effort: any failure logs and the run
        continues untouched.
        """
        if not _reflection_emission_enabled():
            return
        try:
            event_doc = _mechanical_reflection_doc(
                run_id=run_id,
                node_id=str(node_id or ""),
                iteration=iteration,
                outcome=outcome,
                detail=detail,
                draft=draft,
                artifact=artifact,
                component_registry=component_registry,
            )
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="reflection_recorded",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    node_id=node_id,
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "reflection_recorded",
                    ),
                    event_doc=event_doc,
                )
            )
        except Exception as exc:
            logger.warning(
                "research_lab_reflection_emission_failed run_id=%s node_id=%s outcome=%s error=%s",
                run_id,
                str(node_id or "")[:80],
                str(outcome)[:80],
                str(exc)[:200],
            )

    async def _ensure_patch_applies_or_repair(
        self,
        *,
        draft: CodeEditDraft,
        run_id: str,
        node_id: str,
        iteration: int,
        settings: AutoResearchRuntimeSettings,
        artifact: PrivateModelArtifactManifest,
        source_context: Any,
        source_inspection_context: Mapping[str, Any],
        read_paths: Sequence[str],
        budget_context: Mapping[str, Any],
        budget_limit_microusd: int,
        elapsed: Callable[[], float],
        openrouter_calls: int,
        estimated_cost: float,
        actual_cost_microusd: int,
        provider_usage: list[dict[str, Any]],
        within_run_memory: Mapping[str, Any] | None = None,
    ) -> tuple[CodeEditDraft | None, int, float, int, bool]:
        candidate_draft = draft.with_unified_diff(draft.unified_diff)
        max_repairs = max(0, int(self.builder.config.code_edit_patch_repair_attempts))
        for repair_attempt in range(0, max_repairs + 1):
            try:
                self.builder.check_patch_applies(
                    draft=candidate_draft,
                    parent_artifact=artifact,
                    source_context=source_context,
                )
                return candidate_draft, openrouter_calls, estimated_cost, actual_cost_microusd, False
            except CodeEditPatchApplyError as exc:
                failure_stage = str(getattr(exc, "failure_stage", "") or "candidate_patch_apply_failed")
                diagnostic_doc = await _write_private_code_edit_diagnostic(
                    artifact=artifact,
                    run_id=run_id,
                    node_id=node_id,
                    iteration=iteration,
                    stage=failure_stage,
                    draft=candidate_draft,
                    error=exc,
                    artifact_io=self.artifact_io,
                )
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type=failure_stage,
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            failure_stage,
                        ),
                        event_doc={
                            "iteration": iteration,
                            "repair_attempt": repair_attempt,
                            "target_files": list(candidate_draft.target_files),
                            "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                            "error": safe_event_error_text(exc),
                            "error_hash": sha256_json({"error": str(exc)}),
                            "stderr_hash": sha256_json({"stderr": getattr(exc, "stderr", "")}),
                            **diagnostic_doc,
                        },
                    )
                )
                if repair_attempt >= max_repairs:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_repair_exhausted",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "candidate_repair_exhausted",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "repair_attempts": max_repairs,
                                "target_files": list(candidate_draft.target_files),
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                "error": safe_event_error_text(exc),
                                "error_hash": sha256_json({"error": str(exc)}),
                                "last_failure_stage": failure_stage,
                            },
                        )
                    )
                    return None, openrouter_calls, estimated_cost, actual_cost_microusd, False

                if elapsed() >= settings.max_seconds:
                    return None, openrouter_calls, estimated_cost, actual_cost_microusd, False
                if _would_exceed_budget(
                    actual_cost_microusd,
                    _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                    budget_limit_microusd,
                ):
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_repair_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_repair_budget_exhausted",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "repair_attempt": repair_attempt + 1,
                                "error": "compute_budget_exhausted_before_code_edit_repair",
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                            },
                        )
                    )
                    return None, openrouter_calls, estimated_cost, actual_cost_microusd, True

                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_repair_requested",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "code_edit_repair_requested",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "repair_attempt": repair_attempt + 1,
                            "target_files": list(candidate_draft.target_files),
                            "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                            "apply_error_hash": sha256_json({"error": str(exc)}),
                        },
                    )
                )

                remaining_call_seconds = max(1, int(settings.max_seconds - elapsed()))
                repair_result, repair_call_error = await self._call_stage_contained(
                    build_code_edit_repair_messages(
                        draft=candidate_draft,
                        apply_error=str(exc),
                        source_inspection_context=source_inspection_context,
                        runtime_source_context=source_context.prompt_context(),
                        budget_context={
                            **dict(budget_context),
                            "loop_iteration": iteration,
                            "repair_attempt": repair_attempt + 1,
                            "candidate_kind": "image_build",
                        },
                        repair_attempt=repair_attempt + 1,
                        max_candidates=1,
                    ),
                    min(settings.draft_timeout_seconds, remaining_call_seconds),
                    3000,
                    "code_edit_repair",
                )
                if repair_result is None:
                    failure_usage = (
                        repair_call_error.failure_usage_entries()
                        if isinstance(repair_call_error, ContainedStageFailure)
                        else []
                    )
                    if isinstance(repair_call_error, ContainedStageFailure):
                        actual_cost_microusd += repair_call_error.cost_microusd
                        provider_usage.extend(failure_usage)
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_repair_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=failure_usage,
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_repair_call_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "repair_attempt": repair_attempt + 1,
                                "stage": "code_edit_repair_call_failed",
                                "target_files": list(candidate_draft.target_files),
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                "error": repair_call_error or "code_edit_repair_call_failed",
                            },
                        )
                    )
                    if repair_attempt + 1 >= max_repairs:
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="candidate_repair_exhausted",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                node_id=node_id,
                                provider_usage=failure_usage,
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "candidate_repair_exhausted",
                                ),
                                event_doc={
                                    "iteration": iteration,
                                    "repair_attempts": max_repairs,
                                    "stage": "code_edit_repair_call_failed",
                                    "target_files": list(candidate_draft.target_files),
                                    "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                    "error": repair_call_error or "code_edit_repair_call_failed",
                                },
                            )
                        )
                        return None, openrouter_calls, estimated_cost, actual_cost_microusd, False
                    continue
                openrouter_calls += 1
                estimated_cost += settings.estimated_iteration_cost_usd
                actual_cost_microusd += max(0, int(repair_result.cost_microusd))
                if repair_result.provider_usage:
                    provider_usage.append(
                        {
                            **repair_result.provider_usage,
                            "loop_iteration": iteration,
                            "repair_attempt": repair_attempt + 1,
                            "call_stage": "code_edit_repair",
                        }
                    )
                try:
                    repaired_drafts = parse_code_edit_repair_response(
                        repair_result.content,
                        original_draft=draft,
                    )
                except Exception as parse_exc:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_repair_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_repair_parse_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "repair_attempt": repair_attempt + 1,
                                "error": str(parse_exc)[:500],
                                "raw_response_hash": sha256_json({"raw_response": repair_result.content}),
                            },
                        )
                    )
                    if repair_attempt + 1 >= max_repairs:
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="candidate_repair_exhausted",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                node_id=node_id,
                                provider_usage=([provider_usage[-1]] if provider_usage else []),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "candidate_repair_exhausted",
                                ),
                                event_doc={
                                    "iteration": iteration,
                                    "repair_attempts": max_repairs,
                                    "stage": "code_edit_repair_parse_failed",
                                    "error": str(parse_exc)[:500],
                                    "raw_response_hash": sha256_json({"raw_response": repair_result.content}),
                                },
                            )
                        )
                        return None, openrouter_calls, estimated_cost, actual_cost_microusd, False
                    continue
                repaired = repaired_drafts[0].with_unified_diff(repaired_drafts[0].unified_diff)
                source_errors = self.builder.validate_draft_against_source_context(
                    repaired,
                    source_context,
                    read_paths=read_paths,
                    require_read=True,
                )
                if source_errors:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_repair_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            provider_usage=([provider_usage[-1]] if provider_usage else []),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "code_edit_repair_source_context_failed",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "repair_attempt": repair_attempt + 1,
                                "target_files": list(repaired.target_files),
                                "error": "; ".join(source_errors)[:500],
                                "source_diff_hash": sha256_json({"unified_diff": repaired.unified_diff}),
                                "source_tree_hash": source_context.source_tree_hash,
                            },
                        )
                    )
                    if repair_attempt + 1 >= max_repairs:
                        await self.event_sink(
                            AutoResearchLoopEvent(
                                event_type="candidate_repair_exhausted",
                                loop_status="running",
                                elapsed_seconds=elapsed(),
                                node_id=node_id,
                                provider_usage=([provider_usage[-1]] if provider_usage else []),
                                cost_ledger=_running_cost_ledger(
                                    openrouter_calls,
                                    estimated_cost,
                                    actual_cost_microusd,
                                    "candidate_repair_exhausted",
                                ),
                                event_doc={
                                    "iteration": iteration,
                                    "repair_attempts": max_repairs,
                                    "stage": "code_edit_repair_source_context_failed",
                                    "target_files": list(repaired.target_files),
                                    "error": "; ".join(source_errors)[:500],
                                    "source_diff_hash": sha256_json({"unified_diff": repaired.unified_diff}),
                                    "source_tree_hash": source_context.source_tree_hash,
                                },
                            )
                        )
                        return None, openrouter_calls, estimated_cost, actual_cost_microusd, False
                    continue
                candidate_draft = repaired
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_repair_drafted",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        node_id=node_id,
                        provider_usage=([provider_usage[-1]] if provider_usage else []),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "code_edit_repair_drafted",
                        ),
                        event_doc={
                            "iteration": iteration,
                            "repair_attempt": repair_attempt + 1,
                            "target_files": list(candidate_draft.target_files),
                            "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                            "original_source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
                        },
                    )
                )
        return None, openrouter_calls, estimated_cost, actual_cost_microusd, False


async def _write_private_code_edit_diagnostic(
    *,
    artifact: PrivateModelArtifactManifest,
    run_id: str,
    node_id: str,
    iteration: int,
    stage: str,
    draft: CodeEditDraft,
    error: BaseException,
    artifact_io: Any | None = None,
) -> dict[str, Any]:
    manifest_uri = str(getattr(artifact, "manifest_uri", "") or "")
    if not manifest_uri.startswith("s3://"):
        return {"diagnostic_artifact_skipped": "manifest_uri_not_s3"}
    try:
        bucket, key = _parse_s3_uri(manifest_uri)
    except ValueError as exc:
        return {"diagnostic_artifact_error": safe_event_error_text(exc)}
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    safe_node = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(node_id or "node"))[:80]
    safe_stage = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(stage or "failure"))[:80]
    object_key = f"{base_prefix}/candidates/{run_id}/diagnostics/{int(iteration):03d}-{safe_node}-{safe_stage}.json"
    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_code_edit_failure_diagnostic",
        "run_id": str(run_id),
        "node_id": str(node_id),
        "iteration": int(iteration),
        "stage": str(stage),
        "target_files": list(draft.target_files),
        "source_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
        "unified_diff": draft.unified_diff,
        "error": _diagnostic_text(str(error), limit=12000),
        "stderr": _diagnostic_text(str(getattr(error, "stderr", "") or ""), limit=12000),
        "stdout": _diagnostic_text(str(getattr(error, "stdout", "") or ""), limit=12000),
        "exit_code": getattr(error, "exit_code", None),
    }
    payload_hash = sha256_json(payload)

    try:
        stored_payload = {**payload, "diagnostic_hash": payload_hash}
        if artifact_io is not None:
            await _artifact_io_write_json(
                artifact_io,
                uri=f"s3://{bucket}/{object_key}",
                document=stored_payload,
                content_hash=payload_hash,
                artifact_kind="code_edit_failure_diagnostic",
            )
        else:
            def _put() -> None:
                import boto3  # type: ignore

                boto3.client("s3").put_object(
                    Bucket=bucket,
                    Key=object_key,
                    Body=json.dumps(stored_payload, sort_keys=True).encode("utf-8"),
                    ContentType="application/json",
                )

            await asyncio.to_thread(_put)
    except Exception as exc:
        return {
            "diagnostic_artifact_hash": payload_hash,
            "diagnostic_artifact_error": safe_event_error_text(exc),
        }
    return {
        "diagnostic_artifact_uri": f"s3://{bucket}/{object_key}",
        "diagnostic_artifact_hash": payload_hash,
    }


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    raw = str(uri or "")
    if not raw.startswith("s3://"):
        raise ValueError("s3_uri_required")
    rest = raw[5:]
    bucket, sep, key = rest.partition("/")
    if not bucket or not sep or not key:
        raise ValueError("invalid_s3_uri")
    return bucket, key


def _diagnostic_text(value: str, *, limit: int) -> str:
    text = str(value or "")
    text = re.sub(r"sk-or-[A-Za-z0-9._:-]+", "[redacted-openrouter-key]", text)
    text = re.sub(r"sb_secret_[A-Za-z0-9._:-]+", "[redacted-supabase-service-role-key]", text)
    text = re.sub(r"sb_publishable_[A-Za-z0-9._:-]+", "[redacted-supabase-anon-key]", text)
    text = re.sub(r"AKIA[A-Z0-9]{16}", "[redacted-aws-access-key-id]", text)
    text = re.sub(r"https?://[^@\s]+@([^\s/]+)", r"[redacted-proxy-url]@\1", text)
    text = re.sub(r"(?i)(api_key=)[^&\s]+", r"\1[redacted]", text)
    replacements = (
        "service_role",
        "openrouter_api_key",
        "raw_openrouter_key",
        "raw_secret",
        "aws_secret_access_key",
        "password",
        "proxy",
        "webshare",
    )
    lowered = text.lower()
    if any(marker in lowered for marker in replacements):
        return "[redacted secret-like diagnostic text]"
    return text[: max(1, int(limit))]


def _memory_safe_text(value: str) -> str:
    """Sanitized short text for within-run memory records (§6.2-5): rejection
    reasons re-enter later prompts and event docs, so secret-shaped content is
    redacted the same way diagnostics are."""
    return _diagnostic_text(str(value or ""), limit=280)


# §9.1-5 hazard: the trajectory-corpus protected-material scanner fails any
# record whose payload contains markers like "llm response"/"judge prompt"/
# "page content", and the scripts/34 event-doc CHECK rejects terms like
# "judge_prompt"/"hidden_icp". Reflection free text therefore gets the
# diagnostic sanitizer PLUS marker redaction (space and underscore variants).
_REFLECTION_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(re.escape(marker), re.IGNORECASE)
    for marker in sorted(
        set(FORBIDDEN_CODE_EDIT_TERMS)
        | set(PROTECTED_CORPUS_MARKERS)
        | {marker.replace(" ", "_") for marker in PROTECTED_CORPUS_MARKERS},
        key=len,
        reverse=True,
    )
)
_REFLECTION_REDACTED = "[protected-material-redacted]"


def _reflection_safe_text(value: str, *, limit: int = 280) -> str:
    """Sanitize reflection/lesson free text for event docs and prompt reinjection."""
    text = _diagnostic_text(str(value or ""), limit=limit)
    for pattern in _REFLECTION_MARKER_PATTERNS:
        text = pattern.sub(_REFLECTION_REDACTED, text)
    return text


def _reflection_narrative(
    *,
    outcome: str,
    detail: str,
    component: str,
    lane: str,
    champion_base: str,
) -> tuple[str, str, str, str]:
    """Mechanical {worked, failed, why, next_question} per judge/build outcome."""
    lane_label = lane or "code_edit"
    if outcome == "candidate_build_passed":
        return (
            f"Code edit targeting {component} applied cleanly, built an image, and "
            f"passed private tests in the {lane_label} lane.",
            "No failure at the build stage; the live benchmark delta is still unmeasured.",
            "Build and private tests validate structure only; the scored delta versus "
            "the daily baseline decides keep or discard.",
            "Does this candidate improve candidate_delta_vs_daily_baseline over parent "
            f"{champion_base[:24]}?",
        )
    if outcome == "plan_alignment_rejected":
        return (
            "Draft parsed and its patch applied to the parent source tree.",
            f"Plan alignment judge rejected the draft: {detail}",
            "The judge verdict found the diff diverged from the selected plan path or "
            "repeated an already-rejected approach.",
            "What diff would satisfy the selected plan path without repeating a "
            "rejected approach?",
        )
    if outcome == "patch_apply_repair_exhausted":
        return (
            "Draft parsed into a structured code-edit candidate.",
            f"Patch did not apply after repair attempts: {detail}",
            "The unified diff did not match the parent source tree at the targeted hunks.",
            f"Which regions of {component} must be re-read to produce an applying diff?",
        )
    if outcome == "source_context_validation":
        return (
            "Draft parsed into a structured code-edit candidate.",
            f"Source-context validation rejected the draft: {detail}",
            "The draft referenced files or regions outside the inspected source context.",
            f"Which files must be inspected before targeting {component} again?",
        )
    if outcome == "candidate_build_unexpected_error":
        return (
            "Draft parsed and its patch was accepted for building.",
            f"Unexpected infrastructure error during the build stage: {detail}",
            "The failure occurred in build infrastructure, so it is weak evidence "
            "against the candidate diff itself.",
            "Does the same diff build cleanly on retry, or does the error reproduce?",
        )
    # Build-stage failures (candidate_build_failed and typed failure_stage values).
    stage_label = str(outcome or "candidate_build_failed").replace("_", " ")
    return (
        "Draft parsed, its patch applied, and the candidate reached the build stage.",
        f"{stage_label}: {detail}" if detail else f"{stage_label} rejected the candidate.",
        f"The {stage_label} stage rejected the candidate before it could be scored.",
        f"What minimal change to the diff avoids the recorded failure in {component}?",
    )


def _mechanical_reflection_doc(
    *,
    run_id: str,
    node_id: str,
    iteration: int,
    outcome: str,
    detail: str,
    draft: CodeEditDraft | None,
    artifact: PrivateModelArtifactManifest,
    component_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the ``reflection_recorded`` event doc for one judge/build outcome.

    The nested ``reflection`` follows ``engine_v1.build_reflection_record``
    semantics — an engine-authored ``ReflectionRecord`` with ``{worked, failed,
    why, next_question}`` plus provenance ``{champion_base: parent artifact
    hash, component: primary target file/area, eval_version}`` — built
    mechanically (no LLM call). ``basis_patch_seq`` is 0 because live code-edit
    drafts are not typed per-component patches yet (§9.5 defers typed patches);
    staleness is judged downstream by champion_base mismatch, mirroring
    ``mark_lesson_staleness``.
    """
    lane = str(draft.lane if draft is not None else "")[:80]
    target_files = [
        str(path)[:240] for path in (draft.target_files if draft is not None else ())
    ][:10]
    component = str(target_files[0] if target_files else (lane or "code_edit"))[:128]
    champion_base = str(artifact.model_artifact_hash)
    eval_version = str(component_registry.get("eval_version") or "") or "unversioned"
    safe_detail = _reflection_safe_text(detail, limit=200)
    worked, failed, why, next_question = _reflection_narrative(
        outcome=str(outcome),
        detail=safe_detail,
        component=component,
        lane=lane,
        champion_base=champion_base,
    )
    record = ReflectionRecord(
        lesson_id="lesson:"
        + sha256_json(
            {
                "run_id": str(run_id),
                "node_id": str(node_id),
                "iteration": int(iteration),
                "outcome": str(outcome),
            }
        ).split(":", 1)[1][:16],
        node_id=str(node_id),
        worked=_reflection_safe_text(worked, limit=400),
        failed=_reflection_safe_text(failed, limit=400),
        why=_reflection_safe_text(why, limit=400),
        next_question=_reflection_safe_text(next_question, limit=400),
        champion_base=champion_base,
        component=component,
        eval_version=eval_version,
        basis_patch_seq=0,
        stale_basis=False,
        engine_authored=True,
    )
    return {
        "schema_version": "1.0",
        "iteration": int(iteration),
        "outcome": str(outcome)[:80],
        "reflection_source": "mechanical",
        "lane": lane,
        "plan_path_id": str(draft.plan_path_id if draft is not None else "")[:120],
        "target_files": target_files,
        "unified_diff_hash": (
            sha256_json({"unified_diff": draft.unified_diff}) if draft is not None else None
        ),
        "reflection": record.to_dict(),
    }


async def _write_private_loop_candidate_artifact(
    *,
    artifact: PrivateModelArtifactManifest,
    run_id: str,
    node_id: str,
    iteration: int,
    draft: CodeEditDraft,
    build: CodeEditBuildResult,
    dev_score: float | None = None,
    dev_score_version: str = "",
    dev_evaluation: Mapping[str, Any] | None = None,
    dev_feedback: Mapping[str, Any] | None = None,
    dev_feedback_hash: str = "",
    tree_settled_cost_microusd: int = 0,
    git_tree: Mapping[str, Any] | None = None,
    artifact_io: Any | None = None,
) -> dict[str, Any]:
    """Persist a full rehydration doc for a built candidate (bug #5).

    The checkpoint keeps only URI + hash; this S3 artifact carries everything
    needed to reconstruct the ``BuiltCodeEditCandidate`` on resume. Best-effort:
    a failed write only loses restorability for this candidate. The §6.3-1
    ranking-only dev score travels with the doc when present; unscored
    candidates keep the exact pre-dev-eval payload shape (and hash arithmetic).
    """
    manifest_uri = str(getattr(artifact, "manifest_uri", "") or "")
    if not manifest_uri.startswith("s3://"):
        return {"loop_candidate_artifact_skipped": "manifest_uri_not_s3"}
    try:
        bucket, key = _parse_s3_uri(manifest_uri)
    except ValueError as exc:
        return {"loop_candidate_artifact_error": safe_event_error_text(exc)}
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    safe_node = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(node_id or "node"))[:80]
    payload = {
        "schema_version": "research_lab.git_tree_candidate_rehydration.v1",
        "artifact_type": "research_lab_git_tree_candidate_rehydration",
        "run_id": str(run_id),
        "node_id": str(node_id),
        "iteration": int(iteration),
        "draft": draft.to_dict(),
        "candidate_model_manifest": build.candidate_model_manifest.to_dict(),
        "code_edit_manifest": dict(build.code_edit_manifest),
        "source_diff_hash": str(build.source_diff_hash),
        "build_doc": dict(build.build_doc),
        "tree_settled_cost_microusd": max(
            0, int(tree_settled_cost_microusd)
        ),
    }
    if dev_score is not None:
        payload["dev_score"] = float(dev_score)
        payload["dev_score_version"] = str(dev_score_version or "")
    if dev_evaluation:
        payload["dev_evaluation"] = dict(dev_evaluation)
    if dev_feedback:
        payload["dev_feedback"] = dict(dev_feedback)
        payload["dev_feedback_hash"] = str(dev_feedback_hash or "")
    if not git_tree:
        return {"loop_candidate_artifact_error": "git_tree_lineage_missing"}
    payload["git_tree"] = dict(git_tree)
    payload_hash = sha256_json(payload)
    object_key = (
        f"{base_prefix}/candidates/{run_id}/loop-candidates/"
        f"{int(iteration):03d}-{safe_node}-{payload_hash.split(':', 1)[1]}.json"
    )
    artifact_uri = f"s3://{bucket}/{object_key}"

    try:
        stored_payload = {**payload, "loop_candidate_artifact_hash": payload_hash}
        if artifact_io is not None:
            await _artifact_io_write_json(
                artifact_io,
                uri=artifact_uri,
                document=stored_payload,
                content_hash=payload_hash,
                artifact_kind="loop_candidate_rehydration",
            )
            readback = await _artifact_io_read_json(
                artifact_io,
                artifact_uri,
                content_hash=payload_hash,
            )
        else:
            def _put_and_read() -> dict[str, Any]:
                import boto3  # type: ignore

                client = boto3.client("s3")
                client.put_object(
                    Bucket=bucket,
                    Key=object_key,
                    Body=json.dumps(stored_payload, sort_keys=True).encode("utf-8"),
                    ContentType="application/json",
                )
                body = client.get_object(Bucket=bucket, Key=object_key)["Body"].read()
                return json.loads(body.decode("utf-8"))

            readback = await asyncio.to_thread(_put_and_read)
        if (
            not isinstance(readback, Mapping)
            or dict(readback) != stored_payload
            or str(readback.get("loop_candidate_artifact_hash") or "")
            != payload_hash
        ):
            raise ValueError("loop candidate artifact readback differs")
    except Exception as exc:
        return {
            "loop_candidate_artifact_hash": payload_hash,
            "loop_candidate_artifact_error": safe_event_error_text(exc),
        }
    return {
        "loop_candidate_artifact_uri": artifact_uri,
        "loop_candidate_artifact_hash": payload_hash,
    }


async def _artifact_io_read_json(
    artifact_io: Any,
    uri: str,
    *,
    content_hash: str,
) -> dict[str, Any]:
    value = artifact_io.read_json(uri=uri, content_hash=content_hash)
    if inspect.isawaitable(value):
        value = await value
    if not isinstance(value, Mapping):
        raise ValueError("artifact reader returned a non-object")
    return dict(value)


async def _artifact_io_write_json(
    artifact_io: Any,
    *,
    uri: str,
    document: Mapping[str, Any],
    content_hash: str,
    artifact_kind: str,
) -> dict[str, Any]:
    value = artifact_io.write_json(
        uri=uri,
        document=dict(document),
        content_hash=content_hash,
        artifact_kind=artifact_kind,
    )
    if inspect.isawaitable(value):
        value = await value
    if not isinstance(value, Mapping):
        raise ValueError("artifact writer returned a non-object")
    result = dict(value)
    if result.get("uri") != uri or result.get("content_hash") != content_hash:
        raise ValueError("artifact writer acknowledgement differs from request")
    return result


def _rehydrated_candidate_from_artifact_payload(
    payload: Mapping[str, Any],
) -> BuiltCodeEditCandidate:
    """Reconstruct a ``BuiltCodeEditCandidate`` from a rehydration artifact.

    Raises on any shape mismatch — the caller treats a failed candidate as
    unrestorable and degrades to the legacy empty-``selected`` behavior.
    """
    stored = dict(payload)
    expected_hash = str(stored.pop("loop_candidate_artifact_hash", "") or "")
    if expected_hash and sha256_json(stored) != expected_hash:
        raise ValueError("loop_candidate_artifact_hash_mismatch")
    draft_doc = dict(stored.get("draft") or {})
    draft_fields = {f.name for f in fields(CodeEditDraft)}
    draft_kwargs = {name: value for name, value in draft_doc.items() if name in draft_fields}
    draft_kwargs["target_files"] = tuple(draft_kwargs.get("target_files") or ())
    draft_kwargs["plan_alignment"] = dict(draft_kwargs.get("plan_alignment") or {})
    draft = CodeEditDraft(**draft_kwargs)
    build = CodeEditBuildResult(
        candidate_model_manifest=PrivateModelArtifactManifest.from_mapping(
            stored["candidate_model_manifest"]
        ),
        code_edit_manifest=dict(stored.get("code_edit_manifest") or {}),
        source_diff_hash=str(stored.get("source_diff_hash") or ""),
        build_doc=dict(stored.get("build_doc") or {}),
    )
    # §6.3-1: dev fields are optional both ways — pre-dev-eval artifacts lack
    # them (restore as unscored) and dev-scored artifacts restore their score.
    raw_dev_score = stored.get("dev_score")
    dev_score = (
        float(raw_dev_score)
        if isinstance(raw_dev_score, (int, float)) and not isinstance(raw_dev_score, bool)
        else None
    )
    dev_feedback = (
        dict(stored.get("dev_feedback") or {})
        if isinstance(stored.get("dev_feedback"), Mapping)
        else {}
    )
    dev_feedback_hash = str(stored.get("dev_feedback_hash") or "")
    if dev_feedback:
        embedded_feedback_hash = str(dev_feedback.get("feedback_hash") or "")
        if not dev_feedback_hash or embedded_feedback_hash != dev_feedback_hash:
            raise ValueError("loop_candidate_dev_feedback_hash_mismatch")
    tree_doc = (
        dict(stored.get("git_tree") or {})
        if isinstance(stored.get("git_tree"), Mapping)
        else {}
    )
    if tree_doc.get("schema_version") != "research_lab.git_tree_lineage.v1":
        raise ValueError("loop_candidate_git_tree_lineage_missing")
    if (
        str(tree_doc.get("node_id") or "") != str(stored.get("node_id") or "")
        or dict(build.build_doc.get("git_tree") or {}) != tree_doc
        or str(tree_doc.get("cumulative_source_diff_hash") or "")
        != build.source_diff_hash
    ):
        raise ValueError("loop_candidate_git_tree_lineage_mismatch")
    raw_parent_score = tree_doc.get("parent_dev_score")
    tree_parent_dev_score = (
        float(raw_parent_score)
        if isinstance(raw_parent_score, (int, float))
        and not isinstance(raw_parent_score, bool)
        and math.isfinite(float(raw_parent_score))
        else None
    )
    tree_settled_cost_microusd = stored.get(
        "tree_settled_cost_microusd", 0
    )
    if (
        isinstance(tree_settled_cost_microusd, bool)
        or not isinstance(tree_settled_cost_microusd, int)
        or tree_settled_cost_microusd < 0
    ):
        raise ValueError("loop_candidate_tree_settled_cost_invalid")
    return BuiltCodeEditCandidate(
        draft=draft,
        build=build,
        node_id=str(stored.get("node_id") or ""),
        iteration=int(stored.get("iteration") or 0),
        dev_score=dev_score,
        dev_score_version=str(stored.get("dev_score_version") or ""),
        dev_evaluation=(
            dict(stored.get("dev_evaluation") or {})
            if isinstance(stored.get("dev_evaluation"), Mapping)
            else {}
        ),
        dev_feedback=dev_feedback,
        dev_feedback_hash=dev_feedback_hash,
        tree_id=str(tree_doc.get("tree_id") or ""),
        tree_parent_node_id=str(tree_doc.get("parent_node_id") or ""),
        tree_root_branch_id=str(tree_doc.get("root_branch_id") or ""),
        tree_depth=max(0, int(tree_doc.get("depth") or 0)),
        tree_child_slot=max(0, int(tree_doc.get("child_slot") or 0)),
        tree_branch_objective_path_id=str(
            tree_doc.get("branch_objective_path_id") or ""
        ),
        tree_branch_objective_hash=str(
            tree_doc.get("branch_objective_hash") or ""
        ),
        tree_generation_attempt_count=max(
            0, int(tree_doc.get("generation_attempt_count") or 0)
        ),
        tree_git_commit=str(tree_doc.get("git_commit") or ""),
        tree_root_artifact_hash=str(tree_doc.get("root_artifact_hash") or ""),
        tree_parent_artifact_hash=str(tree_doc.get("parent_artifact_hash") or ""),
        tree_parent_dev_score=tree_parent_dev_score,
        tree_parent_feedback_hash=str(
            tree_doc.get("parent_dev_feedback_hash") or ""
        ),
        tree_incremental_source_diff_hash=str(
            tree_doc.get("incremental_source_diff_hash") or ""
        ),
        tree_cumulative_source_diff_hash=str(
            tree_doc.get("cumulative_source_diff_hash") or ""
        ),
        tree_composition=(
            dict(tree_doc.get("composition") or {})
            if isinstance(tree_doc.get("composition"), Mapping)
            else {}
        ),
        tree_settled_cost_microusd=tree_settled_cost_microusd,
    )


def _redacted_draft_doc(draft: CodeEditDraft) -> dict[str, Any]:
    return {
        "failure_mode": draft.failure_mode,
        "mechanism": draft.mechanism,
        "expected_improvement": draft.expected_improvement,
        "risk": draft.risk,
        "lane": draft.lane,
        "plan_path_id": draft.plan_path_id,
        "plan_alignment": dict(draft.plan_alignment or {}),
        "target_files": list(draft.target_files),
        "unified_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
        "redacted_summary": draft.redacted_summary,
        "test_plan": draft.test_plan,
        "rollback_plan": draft.rollback_plan,
        "predicted_delta": draft.predicted_delta,
    }


def _focus_signature_hash(ticket: Mapping[str, Any]) -> str:
    # Delegates to the shared intake helper so ticket-intake refusal dedup
    # and the refusal events stamped here can never drift apart.
    from gateway.research_lab.ticket_intake_validation import focus_signature_hash_for_brief

    focus = _ticket_doc_value(ticket, "brief_public_summary")
    return focus_signature_hash_for_brief(str(focus or ""))


def _prior_attempts_from_budget_context(budget_context: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    memory = budget_context.get("active_parent_outcome_memory")
    if not isinstance(memory, Mapping):
        return ()
    attempts = memory.get("recent_attempts")
    if not isinstance(attempts, list):
        return ()
    cleaned: list[dict[str, Any]] = []
    for item in attempts[:100]:
        if not isinstance(item, Mapping):
            continue
        cleaned.append(
            {
                "candidate_id": str(item.get("candidate_id") or "")[:120],
                "run_id": str(item.get("run_id") or "")[:120],
                "lane": str(item.get("lane") or "")[:120],
                "plan_path_id": str(item.get("plan_path_id") or "")[:120],
                "target_files": [
                    str(path)[:240]
                    for path in (item.get("target_files") or [])
                    if isinstance(path, str)
                ][:20],
                "unified_diff_hash": str(item.get("unified_diff_hash") or "")[:120],
                "candidate_source_diff_hash": str(item.get("candidate_source_diff_hash") or "")[:120],
                "semantic_edit_summary": str(item.get("semantic_edit_summary") or "")[:500],
                "status": str(item.get("status") or "")[:120],
                "reason": str(item.get("reason") or "")[:240],
            }
        )
    return tuple(cleaned)


def _merge_source_inspection_context(
    existing: Mapping[str, Any],
    update: Mapping[str, Any],
    *,
    total_bytes: int,
    read_paths: set[str],
) -> dict[str, Any]:
    existing_results = existing.get("results") if isinstance(existing, Mapping) else []
    update_results = update.get("results") if isinstance(update, Mapping) else []
    results: list[dict[str, Any]] = []
    for item in list(existing_results or []) + list(update_results or []):
        if isinstance(item, Mapping):
            results.append(dict(item))
    return {
        "schema_version": "1.0",
        "source_tree_hash": str(update.get("source_tree_hash") or existing.get("source_tree_hash") or ""),
        "read_files": sorted(read_paths),
        "results": results,
        "bytes_returned": int(total_bytes),
        "context_hash": sha256_json(
            {
                "read_files": sorted(read_paths),
                "result_hashes": [sha256_json(item) for item in results],
                "bytes_returned": int(total_bytes),
            }
        ),
    }


def _ticket_doc_value(ticket: Mapping[str, Any], key: str) -> Any:
    if key in ticket:
        return ticket.get(key)
    doc = ticket.get("ticket_doc")
    if isinstance(doc, Mapping):
        return doc.get(key)
    return None
