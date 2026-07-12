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
    resolve_source_inspection_requests,
)
from gateway.research_lab.loop_engine import (
    AutoResearchLoopEvent,
    AutoResearchLoopResult,
    AutoResearchLoopSettings,
    OpenRouterCallResult,
    _budget_limit_microusd,
    _coerce_call_result,
    _estimated_call_microusd,
    _running_cost_ledger,
    _safe_budget_doc,
    _settings_doc,
    _would_exceed_budget,
)
from research_lab.axis_provenance import call_episode
from gateway.research_lab.logging_utils import safe_event_error_text
from research_lab.canonical import sha256_json
from research_lab.code_editing import (
    FORBIDDEN_CODE_EDIT_TERMS,
    CodeEditDraft,
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


def _ranked_path_fallback_enabled(config: Any) -> bool:
    attr = getattr(config, "ranked_path_fallback_enabled", None)
    if attr is not None:
        return bool(attr)
    return _engine_env_flag("RESEARCH_LAB_RANKED_PATH_FALLBACK_ENABLED", "true")


def _ranked_path_fallback_max_paths(config: Any) -> int:
    attr = getattr(config, "ranked_path_fallback_max_paths", None)
    if attr is None:
        raw = os.getenv("RESEARCH_LAB_RANKED_PATH_FALLBACK_MAX_PATHS", "3")
        try:
            attr = int(raw)
        except (TypeError, ValueError):
            attr = 3
    return max(1, int(attr))


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


def _ranked_path_fallback_plan(
    base_plan_doc: Mapping[str, Any] | None,
    *,
    attempted_path_ids: set[str],
    max_paths: int,
    fallback_index: int,
    refused_lanes: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any] | None:
    if not isinstance(base_plan_doc, Mapping):
        return None
    ranked_paths = base_plan_doc.get("ranked_paths")
    if not isinstance(ranked_paths, (list, tuple)):
        return None
    # Diversity preference: when a lane was refused against the inspected
    # source, prefer the next un-attempted path in a DIFFERENT lane; fall back
    # to same-lane paths only when no alternative lane remains.
    if refused_lanes:
        preferred = _ranked_path_fallback_plan(
            {
                **dict(base_plan_doc),
                "ranked_paths": [
                    raw_path
                    for raw_path in ranked_paths
                    if isinstance(raw_path, Mapping)
                    and str(raw_path.get("lane") or raw_path.get("required_lane") or "")
                    not in refused_lanes
                ],
            },
            attempted_path_ids=attempted_path_ids,
            max_paths=max_paths,
            fallback_index=fallback_index,
        )
        if preferred is not None:
            return preferred
    checked = 0
    for raw_path in ranked_paths:
        if not isinstance(raw_path, Mapping):
            continue
        path_id = _ranked_path_id(raw_path)
        if not path_id or path_id in attempted_path_ids:
            continue
        checked += 1
        if checked > max_paths:
            return None
        required_lane = str(raw_path.get("lane") or raw_path.get("required_lane") or "")
        required_mechanism = str(
            raw_path.get("mechanism")
            or raw_path.get("required_mechanism")
            or raw_path.get("hypothesis")
            or ""
        )
        if not required_lane or not required_mechanism:
            continue
        candidate = dict(base_plan_doc)
        candidate["no_new_safe_path"] = False
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
            candidate[field_name] = raw_path.get(field_name, "")
        candidate["validation_mode"] = raw_path.get("validation_mode", "runtime_checks")
        candidate["ranked_path_fallback"] = {
            "schema_version": "1.0",
            "fallback_index": max(1, int(fallback_index)),
            "path_id": path_id,
            "source_plan_hash": str(base_plan_doc.get("plan_hash") or ""),
        }
        candidate.pop("plan_hash", None)
        candidate["plan_hash"] = sha256_json(candidate)
        return candidate
    return None


def _stop_at_candidate_cap_enabled() -> bool:
    """Bug 20 kill switch: stop iterating/building once max_candidates is reached (no build-and-discard)."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_STOP_AT_CANDIDATE_CAP", "true")


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


def _refusal_lane_advance_enabled() -> bool:
    """After a source-grounded refusal (drafter declined the lane against the
    inspected source, and the bounded fallback declined again), advance to the
    next ranked path — or terminate cheaply — instead of re-asking the same
    lane every iteration until the compute budget dies."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_REFUSAL_LANE_ADVANCE", "true")


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


def _multi_candidate_drafts_enabled() -> bool:
    """§6.2-8 kill switch: ask for and parse up to N>1 candidates per draft call,
    bounded by the remaining candidate slots, instead of a flat
    ``settings.max_candidates``. Inert at prod config: with
    ``hosted_worker_max_candidates=1`` the remaining-slot bound keeps N at 1."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_MULTI_CANDIDATE_DRAFTS", "true")


def _drafts_per_call_limit() -> int:
    """§6.2-8: cap on candidates requested/parsed from one draft call (default 3, min 1)."""

    raw = os.environ.get("RESEARCH_LAB_LOOP_DRAFTS_PER_CALL", "").strip()
    try:
        value = int(raw) if raw else 3
    except ValueError:
        value = 3
    return max(1, value)


def _dev_eval_enabled() -> bool:
    """§6.3-1 L1 dev-eval rung flag (default ON when a dev evaluator is wired):
    score built candidates through the wired ``dev_evaluator`` seam.
    Dev scores are ranking-only within a run and strictly best-effort."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED", "true")


def _dev_snapshot_uri() -> str:
    """The frozen provider-snapshot set URI dev-eval replays against
    (``research_lab.eval.snapshot_store.SNAPSHOT_URI_ENV``). Empty = no set."""

    return os.environ.get("RESEARCH_LAB_DEV_SNAPSHOT_URI", "").strip()


def _dev_plateau_stop_enabled() -> bool:
    """§6.3-4 lite flag (default OFF): stop iterating early when the recent dev
    scores show no improvement over the run's best (requires dev-eval on)."""

    return _engine_env_flag("RESEARCH_LAB_LOOP_DEV_PLATEAU_STOP", "false")


def _dev_plateau_window() -> int:
    """§6.3-4 lite: consecutive non-improving dev scores before stopping (default 2)."""

    raw = os.environ.get("RESEARCH_LAB_LOOP_DEV_PLATEAU_WINDOW", "").strip()
    try:
        value = int(raw) if raw else 2
    except ValueError:
        value = 2
    return max(1, value)


def _dev_plateau_min_delta() -> float:
    """§6.3-4 lite: a dev score must beat the run's best by more than this to
    count as improvement (default 0.5 on the capped-top-5 dev-score scale)."""

    raw = os.environ.get("RESEARCH_LAB_LOOP_DEV_PLATEAU_MIN_DELTA", "").strip()
    try:
        value = float(raw) if raw else 0.5
    except ValueError:
        value = 0.5
    return max(0.0, value)


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


def _rank_selected_by_dev_score(
    candidates: Sequence[BuiltCodeEditCandidate],
) -> list[BuiltCodeEditCandidate]:
    """Intra-run keep-best ordering (§6.3-1).

    Whenever at least one candidate carries a dev score: scored candidates
    come first, ordered by ``dev_score`` desc (stable, so ties keep build
    order), and unscored candidates keep build order after the scored ones.
    A single scored candidate must outrank every unscored one — the per-
    iteration cap truncation keeps only the head of this list, so an
    unscored-first ordering would discard the only build with evidence.
    With zero scored candidates the input order is preserved byte-for-byte
    (dev-eval-off runs keep build order). Dev scores never affect anything
    beyond this intra-run ranking."""

    items = list(candidates)
    scored = [candidate for candidate in items if candidate.dev_score is not None]
    if not scored:
        return items
    unscored = [candidate for candidate in items if candidate.dev_score is None]
    scored.sort(key=lambda candidate: -float(candidate.dev_score or 0.0))
    return scored + unscored


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
    provider_usage: tuple[dict[str, Any], ...] = ()
    status: str = "completed"
    checkpoint_doc: dict[str, Any] | None = None

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
    settings: AutoResearchLoopSettings
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
    # V2 supplies authenticated, receipt-bound provider inputs. None preserves
    # the existing host loaders and their exact continuity behavior.
    provider_registry_loader: Callable[[], tuple[list[Any], Any]] | None = None
    provider_probe_catalog_loader: Callable[[], list[Any]] | None = None
    # Set by run() so stage/build spans can attach to the run's deterministic
    # Langfuse trace (run_trace_id(run_id)) without threading run_id through
    # every stage-call signature. One engine instance serves one run at a time.
    _langfuse_run_id: str = field(default="", init=False, repr=False)

    async def _maybe_dev_eval_candidate(
        self,
        candidate: BuiltCodeEditCandidate,
        *,
        run_id: str,
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
        try:
            if not _dev_eval_enabled():
                return candidate
            if not _dev_snapshot_uri():
                return candidate
            if self.dev_evaluator is None:
                logger.info(
                    "research_lab_loop_dev_eval_unwired run_id=%s node_id=%s "
                    "(RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED is on but no dev_evaluator seam is wired)",
                    run_id,
                    str(candidate.node_id)[:80],
                )
                return candidate
            result = await self.dev_evaluator(candidate)
            if not isinstance(result, Mapping):
                raise ValueError("dev_evaluator returned a non-mapping result")
            raw_score = result.get("aggregate_dev_score", result.get("dev_score"))
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError("dev_evaluator result carried no numeric aggregate_dev_score")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError("dev_evaluator returned a non-finite aggregate_dev_score")
            version = str(result.get("dev_score_version") or "")
            logger.info(
                "research_lab_loop_dev_eval_scored run_id=%s node_id=%s dev_score=%s dev_score_version=%s",
                run_id,
                str(candidate.node_id)[:80],
                round(score, 6),
                version[:80],
            )
            return replace(candidate, dev_score=score, dev_score_version=version)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "research_lab_loop_dev_eval_failed run_id=%s node_id=%s error=%s",
                run_id,
                str(candidate.node_id)[:80],
                str(exc)[:200],
            )
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

        Each checkpoint summary carries a ``rehydration_artifact_uri``/``_hash``
        pointing at the full S3 rehydration doc. Any per-candidate failure is
        logged and skipped; the caller degrades to the legacy empty ``selected``
        on total failure. The ``loop_resumed`` event reports the restored count.
        """
        del elapsed, openrouter_calls, estimated_cost, actual_cost_microusd  # event-free restore
        summaries = resume.get("selected_candidates")
        if not isinstance(summaries, Sequence):
            return []
        restored: list[BuiltCodeEditCandidate] = []
        for summary in summaries:
            if not isinstance(summary, Mapping):
                continue
            uri = str(summary.get("rehydration_artifact_uri") or "")
            expected_hash = str(summary.get("rehydration_artifact_hash") or "")
            if not uri.startswith("s3://"):
                continue
            try:
                bucket, object_key = _parse_s3_uri(uri)

                if self.artifact_io is not None:
                    if not expected_hash:
                        raise ValueError("rehydration_artifact_hash_missing")
                    payload = await _artifact_io_read_json(
                        self.artifact_io,
                        uri,
                        content_hash=expected_hash,
                    )
                else:
                    def _get(bucket: str = bucket, object_key: str = object_key) -> dict[str, Any]:
                        import boto3  # type: ignore

                        body = boto3.client("s3").get_object(Bucket=bucket, Key=object_key)["Body"].read()
                        return json.loads(body.decode("utf-8"))

                    payload = await asyncio.to_thread(_get)
                stored_hash = str(payload.get("loop_candidate_artifact_hash") or "")
                if expected_hash and stored_hash and stored_hash != expected_hash:
                    raise ValueError("rehydration_artifact_hash_mismatch")
                candidate = _rehydrated_candidate_from_artifact_payload(payload)
                restored.append(
                    replace(
                        candidate,
                        rehydration_artifact_uri=uri,
                        rehydration_artifact_hash=expected_hash or stored_hash,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "research_lab_loop_candidate_restore_failed run_id=%s node_id=%s error=%s",
                    run_id,
                    str(summary.get("node_id") or "")[:80],
                    str(exc)[:200],
                )
                continue
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
    ) -> CodeEditBuildResult:
        if not _build_heartbeat_enabled():
            return self.builder.build(
                draft=draft,
                parent_artifact=artifact,
                run_id=run_id,
                candidate_index=candidate_index,
                source_context=source_context,
            )
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
                draft=draft,
                parent_artifact=artifact,
                run_id=run_id,
                candidate_index=candidate_index,
                source_context=source_context,
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
                except Exception:
                    # A failed heartbeat write must never fail the build itself.
                    pass
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
        selected: list[BuiltCodeEditCandidate] = []
        resume = dict(resume_state or {})
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
        built_candidate_total = max(0, int(resume.get("built_candidate_count") or 0))
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
        within_run_memory_active = _within_run_memory_enabled()
        rejected_diff_hashes: set[str] = set()
        within_run_rejections: list[dict[str, Any]] = []
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
                rejected_diff_hashes.add(resolved_hash)
            within_run_rejections.append(
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
            del within_run_rejections[:-25]

        # §6.3-1 dev-eval state (flag default OFF): ranking-only dev scores of
        # candidates built this run, surfaced into within-run memory, plus the
        # §6.3-4-lite plateau tracker — the count of consecutive scored builds
        # that failed to improve the run's best dev score by more than the
        # configured delta.
        dev_score_records: list[dict[str, Any]] = []
        dev_best_score: float | None = None
        dev_scores_since_improvement = 0

        def _record_dev_score(
            *, iteration_index: int, node_id: str, score: float, version: str
        ) -> None:
            nonlocal dev_best_score, dev_scores_since_improvement
            prior_best = dev_best_score
            dev_best_score = score if prior_best is None else max(prior_best, score)
            if prior_best is None or score > prior_best + _dev_plateau_min_delta():
                dev_scores_since_improvement = 0
            else:
                dev_scores_since_improvement += 1
            dev_score_records.append(
                {
                    "iteration": int(iteration_index),
                    "node_id": str(node_id)[:80],
                    "dev_score": round(float(score), 6),
                    "dev_score_version": str(version)[:120],
                }
            )
            del dev_score_records[:-25]

        def _within_run_memory_doc() -> dict[str, Any] | None:
            if not within_run_memory_active:
                return None
            if not within_run_rejections and not dev_score_records:
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
                        round(float(dev_best_score), 6) if dev_best_score is not None else None
                    ),
                    "recent_scores": [dict(item) for item in dev_score_records[-10:]],
                }
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

        # Bug 5: resume used to discard already-built candidates, so a paused/requeued run
        # that had built+pushed an image resumed empty-handed and failed with "no finalists".
        # Restoration is strictly best-effort: any failure degrades to the previous behavior.
        restored_candidate_count = 0
        if resume and _resume_restore_selected_enabled():
            try:
                selected = await self._restore_selected_from_resume(
                    resume=resume,
                    run_id=run_id,
                    artifact=artifact,
                    elapsed=elapsed,
                    openrouter_calls=openrouter_calls,
                    estimated_cost=estimated_cost,
                    actual_cost_microusd=actual_cost_microusd,
                )
            except Exception:
                selected = []
            restored_candidate_count = len(selected)
            built_candidate_total = max(built_candidate_total, restored_candidate_count)
            # §6.3-1: re-seed dev-score memory from restored candidates so their
            # ranking-only scores stay visible to later drafts. Plateau counting
            # never spans a pause: restored candidates arrive in ranked (not
            # build) order, so recomputing staleness over them would be spurious
            # — resume conservatively with a fresh improvement window.
            for restored_candidate in selected:
                if restored_candidate.dev_score is not None:
                    _record_dev_score(
                        iteration_index=restored_candidate.iteration,
                        node_id=restored_candidate.node_id,
                        score=restored_candidate.dev_score,
                        version=restored_candidate.dev_score_version,
                    )
            dev_scores_since_improvement = 0

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
        ranked_path_base_doc: dict[str, Any] = dict(loop_direction_plan_doc or {})
        ranked_path_attempted_ids: set[str] = set()
        selected_path_id = _loop_plan_selected_path_id(loop_direction_plan_doc)
        if selected_path_id:
            ranked_path_attempted_ids.add(selected_path_id)
        ranked_path_fallback_count = 0
        # Lanes the drafter refused against the inspected source this run —
        # the fallback prefers a different lane over re-asking a refused one.
        refused_lane_keys: set[str] = set()
        reference_repair_attempted = bool(resume.get("planner_reference_repair_attempted"))
        reference_repair_status = str(resume.get("planner_reference_repair_status") or "")

        async def _activate_ranked_path_fallback(*, trigger: str, reason: str) -> bool:
            nonlocal loop_direction_plan_doc
            nonlocal planner_terminal_without_candidate
            nonlocal binding_plan_terminal_without_candidate
            nonlocal stop_reason
            nonlocal ranked_path_fallback_count

            if not _ranked_path_fallback_enabled(self.builder.config):
                return False
            if not ranked_path_base_doc:
                return False
            max_ranked_paths = _ranked_path_fallback_max_paths(self.builder.config)
            remaining_path_slots = max_ranked_paths - len(ranked_path_attempted_ids)
            next_index = ranked_path_fallback_count + 1
            previous_path_id = _loop_plan_selected_path_id(loop_direction_plan_doc)
            next_plan = _ranked_path_fallback_plan(
                ranked_path_base_doc,
                attempted_path_ids=ranked_path_attempted_ids,
                max_paths=remaining_path_slots,
                fallback_index=next_index,
                refused_lanes=refused_lane_keys,
            )
            if not next_plan:
                if ranked_path_fallback_count > 0:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="no_viable_patch",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "ranked_path_fallback_exhausted",
                            ),
                            event_doc={
                                "schema_version": "1.0",
                                "stage": "ranked_path_fallback",
                                "trigger": str(trigger or "")[:120],
                                "reason": safe_event_error_text(reason),
                                "failure_class": (
                                    "binding_plan_unimplementable"
                                    if trigger in {
                                        "binding_plan_unimplementable",
                                        "loop_direction_plan_feasibility",
                                    }
                                    or _binding_plan_unimplementable_reason(reason)
                                    else "no_safe_patch"
                                ),
                                "missing_references": [],
                                "ranked_path_fallback_attempted": False,
                                "previous_path_id": previous_path_id,
                                "next_path_id": "",
                                "fallback_index": ranked_path_fallback_count,
                                "terminal_after_ranked_paths_exhausted": True,
                                "source_plan_hash": ranked_path_base_doc.get("plan_hash"),
                                "focus_signature_hash": _focus_signature_hash(ticket),
                            },
                        )
                    )
                return False
            next_path_id = _loop_plan_selected_path_id(next_plan)
            if not next_path_id:
                return False
            next_plan_binding = _bind_loop_direction_plan(
                next_plan,
                source_context=source_context,
                candidate_edit_constraints=candidate_edit_constraints,
            )
            next_plan_errors = list(next_plan_binding.errors)
            if next_plan_errors:
                ranked_path_attempted_ids.add(next_path_id)
                ranked_path_fallback_count = next_index
                await self.event_sink(
                    AutoResearchLoopEvent(
                        event_type="code_edit_validation_failed",
                        loop_status="running",
                        elapsed_seconds=elapsed(),
                        cost_ledger=_running_cost_ledger(
                            openrouter_calls,
                            estimated_cost,
                            actual_cost_microusd,
                            "ranked_path_feasibility_failed",
                        ),
                        event_doc={
                            "schema_version": "1.0",
                            "stage": "ranked_path_feasibility",
                            "trigger": str(trigger or "")[:120],
                            "path_id": next_path_id,
                            "feasibility_error_count": len(next_plan_errors),
                            "error": "; ".join(next_plan_errors)[:700],
                            "ranked_path_fallback_attempted": False,
                            "fallback_index": ranked_path_fallback_count,
                        },
                    )
                )
                return await _activate_ranked_path_fallback(trigger=trigger, reason=reason)
            next_plan = dict(next_plan_binding.plan_doc or next_plan)
            ranked_path_attempted_ids.add(next_path_id)
            ranked_path_fallback_count = next_index
            loop_direction_plan_doc = next_plan
            planner_terminal_without_candidate = False
            binding_plan_terminal_without_candidate = False
            stop_reason = "max_iterations"
            event_doc = {
                "schema_version": "1.0",
                "stage": "ranked_path_fallback",
                "trigger": str(trigger or "")[:120],
                "reason": safe_event_error_text(reason),
                "ranked_path_fallback_attempted": True,
                "previous_path_id": previous_path_id,
                "next_path_id": next_path_id,
                "fallback_index": ranked_path_fallback_count,
                "terminal_after_ranked_paths_exhausted": False,
                "loop_direction_plan_hash": next_plan.get("plan_hash"),
                "source_plan_hash": ranked_path_base_doc.get("plan_hash"),
                "focus_signature_hash": _focus_signature_hash(ticket),
            }
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="candidate_generation_fallback_requested",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "ranked_path_fallback_attempted",
                    ),
                    event_doc=event_doc,
                )
            )
            await self.event_sink(
                AutoResearchLoopEvent(
                    event_type="loop_direction_planned",
                    loop_status="running",
                    elapsed_seconds=elapsed(),
                    cost_ledger=_running_cost_ledger(
                        openrouter_calls,
                        estimated_cost,
                        actual_cost_microusd,
                        "ranked_path_fallback_selected",
                    ),
                    event_doc={
                        **event_doc,
                        "loop_direction_plan": next_plan,
                    },
                )
            )
            return True

        async def _attempt_planner_reference_repair(
            *,
            trigger: str,
            reason: str,
            plan_doc: Mapping[str, Any],
            explicit_references: Sequence[Any] = (),
            feasibility_errors: Sequence[str] = (),
        ) -> dict[str, Any] | None:
            nonlocal loop_direction_plan_doc
            nonlocal ranked_path_base_doc
            nonlocal ranked_path_attempted_ids
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
            ranked_path_base_doc = dict(repaired_doc)
            ranked_path_attempted_ids = set()
            repaired_path_id = _loop_plan_selected_path_id(repaired_doc)
            if repaired_path_id:
                ranked_path_attempted_ids.add(repaired_path_id)
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
            # Bug 5/20: skip the planner call when restored candidates already fill the cap;
            # the loop will finalize immediately, so a plan would be paid for and unused.
            and not (selected and len(selected) >= settings.max_candidates and _stop_at_candidate_cap_enabled())
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
                            artifact_manifest=artifact.to_dict(),
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
                        ranked_path_base_doc = dict(loop_direction_plan_doc)
                        ranked_path_attempted_ids.clear()
                        initial_path_id = _loop_plan_selected_path_id(loop_direction_plan_doc)
                        if initial_path_id:
                            ranked_path_attempted_ids.add(initial_path_id)
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
                        if await _activate_ranked_path_fallback(
                            trigger="loop_direction_no_new_safe_path",
                            reason=loop_plan.reason or "planner returned no_new_safe_path",
                        ):
                            stop_reason = "max_iterations"
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
                    if await _activate_ranked_path_fallback(
                        trigger="loop_direction_plan_feasibility",
                        reason=feasibility_reason,
                    ):
                        planner_terminal_without_candidate = False
                        binding_plan_terminal_without_candidate = False
                        stop_reason = "max_iterations"
        while iteration < settings.max_iterations:
            if planner_terminal_without_candidate or binding_plan_terminal_without_candidate:
                break
            if elapsed() >= settings.max_seconds:
                stop_reason = "max_seconds"
                break
            # Bug 20: iterations past the candidate cap used to build docker images and then
            # discard them (selected is truncated to max_candidates every iteration). Once the
            # cap is filled, stop iterating; minimum runtime is enforced by the post-loop wait.
            if len(selected) >= settings.max_candidates and _stop_at_candidate_cap_enabled():
                stop_reason = "candidate_limit_reached"
                break
            if (
                iteration >= settings.min_iterations
                and elapsed() >= settings.min_seconds
                and len(selected) >= settings.max_candidates
            ):
                stop_reason = "candidate_limit_reached_after_minimum_runtime"
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

            iteration += 1
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
            budget_exhausted_after_source_inspection = False
            retry_iteration_after_reference_repair = False
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
                            artifact_manifest=artifact.to_dict(),
                            component_registry=dict(component_registry),
                            benchmark_public_summary=benchmark_public_summary,
                            runtime_source_index=source_context.inspection_index(),
                            source_inspection_context=source_inspection_context,
                            loop_direction_plan=loop_direction_plan_doc,
                            budget_context=_memory_budget_context({
                                **dict(budget_context),
                                "loop_iteration": iteration,
                                "inspection_round": inspection_round,
                                "candidate_kind": "image_build",
                                "loop_direction_plan_hash": (
                                    (loop_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(loop_direction_plan_doc, Mapping)
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
                # §6.2-8 multi-candidate drafts (flag default ON; inert at prod config
                # because hosted_worker_max_candidates=1 bounds it to one): ask for and
                # parse up to min(remaining candidate slots, RESEARCH_LAB_LOOP_DRAFTS_PER_CALL)
                # candidates from ONE draft call instead of a flat settings.max_candidates,
                # so a single call can fill several slots but never yields drafts past the
                # bug-20 cap (never build-and-discard). The call still counts once in
                # iteration/cost accounting; each build counts per built candidate.
                if _multi_candidate_drafts_enabled():
                    remaining_candidate_slots = max(1, settings.max_candidates - len(selected))
                    draft_parse_limit = min(remaining_candidate_slots, _drafts_per_call_limit())
                else:
                    draft_parse_limit = settings.max_candidates
                candidate_generation_fallback_attempted = False

                async def _attempt_candidate_generation_fallback(
                    *,
                    trigger: str,
                    reason: str,
                ) -> list[CodeEditDraft]:
                    nonlocal openrouter_calls, estimated_cost, actual_cost_microusd
                    nonlocal candidate_generation_fallback_attempted
                    if candidate_generation_fallback_attempted or not read_paths:
                        return []
                    if elapsed() >= settings.max_seconds:
                        return []
                    if _would_exceed_budget(
                        actual_cost_microusd,
                        _estimated_call_microusd(settings.estimated_iteration_cost_usd),
                        budget_limit_microusd,
                    ):
                        return []
                    candidate_generation_fallback_attempted = True
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
                            artifact_manifest=artifact.to_dict(),
                            component_registry=dict(component_registry),
                            benchmark_public_summary=benchmark_public_summary,
                            runtime_source_context=source_context.prompt_context(),
                            source_inspection_context=source_inspection_context,
                            loop_direction_plan=loop_direction_plan_doc,
                            budget_context=_memory_budget_context(
                                {
                                    **dict(budget_context),
                                    "loop_iteration": iteration,
                                    "candidate_kind": "image_build",
                                    "fallback_trigger": str(trigger or "")[:120],
                                    "loop_direction_plan_hash": (
                                        (loop_direction_plan_doc or {}).get("plan_hash")
                                        if isinstance(loop_direction_plan_doc, Mapping)
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
                        return []
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
                        return []
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
                        artifact_manifest=artifact.to_dict(),
                        component_registry=dict(component_registry),
                        benchmark_public_summary=benchmark_public_summary,
                        runtime_source_context=source_context.prompt_context(),
                        source_inspection_context=source_inspection_context,
                        loop_direction_plan=loop_direction_plan_doc,
                        budget_context=_memory_budget_context(
                            {
                                **dict(budget_context),
                                "loop_iteration": iteration,
                                "candidate_kind": "image_build",
                                "loop_direction_plan_hash": (
                                    (loop_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(loop_direction_plan_doc, Mapping)
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
                            },
                        )
                    )
                    drafts = []
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
                            terminal_binding_plan = bool(loop_direction_plan_doc) and bool(
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
                                                "terminal": True,
                                                "stop_reason": "binding_plan_unimplementable",
                                            }
                                            if terminal_binding_plan
                                            else {}
                                        ),
                                        "raw_response_hash": sha256_json({"raw_response": raw}),
                                        "loop_direction_plan_hash": (
                                            (loop_direction_plan_doc or {}).get("plan_hash")
                                            if isinstance(loop_direction_plan_doc, Mapping)
                                            else None
                                        ),
                                    },
                                )
                            )
                            if terminal_binding_plan:
                                stop_reason = "binding_plan_unimplementable"
                                binding_plan_terminal_without_candidate = True
                            repaired_doc = None
                            if (
                                terminal_binding_plan
                                and no_viable is not None
                                and no_viable.missing_references
                            ):
                                repaired_doc = await _attempt_planner_reference_repair(
                                    trigger="binding_plan_unimplementable",
                                    reason=no_viable_reason,
                                    plan_doc=loop_direction_plan_doc or {},
                                    explicit_references=(
                                        no_viable.missing_references if no_viable is not None else ()
                                    ),
                                )
                            if repaired_doc is not None and not bool(repaired_doc.get("no_new_safe_path")):
                                retry_iteration_after_reference_repair = True
                                binding_plan_terminal_without_candidate = False
                                stop_reason = "max_iterations"
                            else:
                                fallback_drafts = await _attempt_candidate_generation_fallback(
                                    trigger="no_viable_patch",
                                    reason=no_viable_reason,
                                )
                                if fallback_drafts:
                                    binding_plan_terminal_without_candidate = False
                                elif terminal_binding_plan:
                                    await _activate_ranked_path_fallback(
                                        trigger="binding_plan_unimplementable",
                                        reason=no_viable_reason,
                                    )
                                elif (
                                    isinstance(loop_direction_plan_doc, Mapping)
                                    and _refusal_lane_advance_enabled()
                                ):
                                    # Source-grounded refusal on a non-terminal
                                    # lane: the drafter (and its bounded
                                    # fallback) declined this lane against the
                                    # inspected source. Re-asking the identical
                                    # lane next iteration is pure spend —
                                    # advance to the next ranked path
                                    # (different lane preferred) or terminate
                                    # cheaply with budget left.
                                    refused_lane = str(
                                        loop_direction_plan_doc.get("required_lane") or ""
                                    )
                                    if refused_lane:
                                        refused_lane_keys.add(refused_lane)
                                    advanced = await _activate_ranked_path_fallback(
                                        trigger="source_grounded_refusal",
                                        reason=no_viable_reason,
                                    )
                                    if not advanced:
                                        stop_reason = "loop_direction_no_new_safe_path"
                                        planner_terminal_without_candidate = True
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
            if retry_iteration_after_reference_repair:
                iteration = max(0, iteration - 1)
                drafts = []
            for draft_index, draft in enumerate(drafts):
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
                if len(selected) >= settings.max_candidates and _stop_at_candidate_cap_enabled():
                    # Bug 20: never start a docker build for a candidate that would be
                    # truncated away by the max_candidates cap.
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="code_edit_validation_failed",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            cost_ledger=_running_cost_ledger(
                                openrouter_calls,
                                estimated_cost,
                                actual_cost_microusd,
                                "candidate_cap_reached",
                            ),
                            event_doc={
                                "iteration": iteration,
                                "stage": "candidate_cap_reached",
                                "error": "candidate_cap_reached_draft_skipped",
                                "selected_candidate_count": len(selected),
                            },
                        )
                    )
                    break
                draft_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
                if within_run_memory_active and draft_diff_hash in rejected_diff_hashes:
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
                node_id = _node_id(run_id, iteration, draft_index, draft)
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
                        artifact=artifact,
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
                                (loop_direction_plan_doc or {}).get("plan_hash")
                                if isinstance(loop_direction_plan_doc, Mapping)
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
                    artifact=artifact,
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
                        artifact=artifact,
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
                    loop_direction_plan=loop_direction_plan_doc,
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
                        artifact=artifact,
                        component_registry=component_registry,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                    )
                    continue
                build_completed = False
                try:
                    await self.event_sink(
                        AutoResearchLoopEvent(
                            event_type="candidate_build_started",
                            loop_status="running",
                            elapsed_seconds=elapsed(),
                            node_id=node_id,
                            cost_ledger=_running_cost_ledger(openrouter_calls, estimated_cost, actual_cost_microusd, "candidate_build_started"),
                            event_doc={
                                "iteration": iteration,
                                "source_diff_hash": sha256_json({"unified_diff": candidate_draft.unified_diff}),
                                "loop_direction_plan_hash": (
                                    (loop_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(loop_direction_plan_doc, Mapping)
                                    else None
                                ),
                                "plan_alignment": dict(candidate_draft.plan_alignment or {}),
                            },
                        )
                    )
                    # Bug 20: candidate_index is a monotonically increasing per-run build counter
                    # (previously len(selected), which repeats after the post-cap truncation and
                    # overwrote the persisted S3 source-diff artifact key each iteration).
                    build = await self._build_candidate_with_heartbeat(
                        draft=candidate_draft,
                        artifact=artifact,
                        run_id=run_id,
                        candidate_index=built_candidate_total,
                        source_context=source_context,
                        node_id=node_id,
                        iteration=iteration,
                        elapsed=elapsed,
                        openrouter_calls=openrouter_calls,
                        estimated_cost=estimated_cost,
                        actual_cost_microusd=actual_cost_microusd,
                    )
                    built_candidate_total += 1
                    build_completed = True
                    built_candidate = BuiltCodeEditCandidate(
                        draft=candidate_draft,
                        build=build,
                        node_id=node_id,
                        iteration=iteration,
                    )
                    # §6.3-1 L1 dev-eval rung (flag default OFF): attach a ranking-only
                    # dev score through the dev_evaluator seam. Best-effort — a dev-eval
                    # failure leaves the candidate unscored and the run untouched.
                    built_candidate = await self._maybe_dev_eval_candidate(
                        built_candidate, run_id=run_id
                    )
                    if built_candidate.dev_score is not None:
                        _record_dev_score(
                            iteration_index=iteration,
                            node_id=node_id,
                            score=built_candidate.dev_score,
                            version=built_candidate.dev_score_version,
                        )
                    # Bug 5: persist a full rehydration doc so a paused/requeued run can restore
                    # this candidate on resume. Best-effort: failure only loses restorability.
                    rehydration_doc = await _write_private_loop_candidate_artifact(
                        artifact=artifact,
                        run_id=run_id,
                        node_id=node_id,
                        iteration=iteration,
                        draft=candidate_draft,
                        build=build,
                        dev_score=built_candidate.dev_score,
                        dev_score_version=built_candidate.dev_score_version,
                        artifact_io=self.artifact_io,
                    )
                    built_candidate = replace(
                        built_candidate,
                        rehydration_artifact_uri=str(rehydration_doc.get("loop_candidate_artifact_uri") or ""),
                        rehydration_artifact_hash=str(rehydration_doc.get("loop_candidate_artifact_hash") or ""),
                    )
                    selected.append(built_candidate)
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
                                    (loop_direction_plan_doc or {}).get("plan_hash")
                                    if isinstance(loop_direction_plan_doc, Mapping)
                                    else None
                                ),
                                "plan_alignment": dict(candidate_draft.plan_alignment or {}),
                                # §6.3-1: emitted only when a dev score was attached, so
                                # dev-eval-off runs keep the exact pre-dev-eval doc shape.
                                **(
                                    {
                                        "dev_score": built_candidate.dev_score,
                                        "dev_score_version": built_candidate.dev_score_version,
                                        "dev_score_ranking_only": True,
                                    }
                                    if built_candidate.dev_score is not None
                                    else {}
                                ),
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
                    raise
                except (CodeEditPrivateTestError, CodeEditImageBuildError, CodeEditPatchApplyError) as exc:
                    event_type = str(getattr(exc, "failure_stage", "") or "candidate_build_failed")
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
                    # Bug 17: an unexpected infra error during build/event emission used to
                    # abort the run. Contain it to this candidate; the run keeps whatever it
                    # has already built.
                    if not _stage_error_containment_enabled():
                        raise
                    if not build_completed:
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
                    except Exception:
                        pass
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
            # §6.3-1 keep-best (bug-20 plumbing): rank by dev score before the cap
            # truncation so the best-scoring builds survive; without 2+ dev scores
            # this preserves build order byte-for-byte.
            selected = _rank_selected_by_dev_score(selected)[: settings.max_candidates]
            try:
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
            except Exception:
                # Bug 17: a transient checkpoint-write failure must not fail a run that may
                # already hold built candidates; the previous checkpoint remains usable.
                if not _stage_error_containment_enabled():
                    raise
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
            if budget_exhausted_after_call:
                stop_reason = "compute_budget_exhausted_after_code_edit"
                break
            if (
                _dev_eval_enabled()
                and _dev_plateau_stop_enabled()
                and dev_scores_since_improvement >= _dev_plateau_window()
            ):
                # §6.3-4 lite: the last N dev-scored builds failed to improve the
                # run's best dev score by more than the configured delta — stop
                # paying for further iterations on a plateaued run.
                stop_reason = "dev_score_plateau"
                break

        if selected:
            remaining_minimum = settings.min_seconds - elapsed()
            remaining_maximum = settings.max_seconds - elapsed()
            if _min_runtime_skip_when_selected_enabled():
                # Candidates are already selected; parking the worker slot until min_seconds
                # elapses is pure waste. Proceed straight to finalization.
                remaining_minimum = 0.0
            if remaining_minimum > 0 and remaining_maximum > 0:
                sleep_remaining = min(remaining_minimum, remaining_maximum)
                while sleep_remaining > 0:
                    await asyncio.sleep(min(5.0, sleep_remaining))
                    sleep_remaining = min(settings.min_seconds - elapsed(), settings.max_seconds - elapsed())
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
        if not selected and stop_reason in {
            "max_iterations",
            "candidate_limit_reached",
            "candidate_limit_reached_after_minimum_runtime",
        }:
            stop_reason = "no_valid_image_build_candidates"

        # §6.3-1: final intra-run ranking — candidate_selected order and the
        # result's selected_candidates agree, with dev-scored builds first
        # (desc). A no-op unless 2+ candidates carry dev scores.
        selected = _rank_selected_by_dev_score(selected)
        for index, candidate in enumerate(selected):
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
                        "candidate_index": index,
                        "iteration": candidate.iteration,
                        "candidate_kind": "image_build",
                        "candidate_model_manifest_hash": candidate.build.candidate_model_manifest.manifest_hash,
                        "candidate_source_diff_hash": candidate.build.source_diff_hash,
                        "redacted_summary": candidate.draft.redacted_summary,
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
                        "loop_direction_plan_hash": (
                            (loop_direction_plan_doc or {}).get("plan_hash")
                            if isinstance(loop_direction_plan_doc, Mapping)
                            else None
                        ),
                        "selected_path_id": (
                            (loop_direction_plan_doc or {}).get("selected_path_id")
                            if isinstance(loop_direction_plan_doc, Mapping)
                            else candidate.draft.plan_path_id
                        ),
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
                    "loop_direction_plan_hash": (
                        (loop_direction_plan_doc or {}).get("plan_hash")
                        if isinstance(loop_direction_plan_doc, Mapping)
                        else None
                    ),
                    "selected_path_id": (
                        (loop_direction_plan_doc or {}).get("selected_path_id")
                        if isinstance(loop_direction_plan_doc, Mapping)
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
        settings: AutoResearchLoopSettings,
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
                        }
                        if candidate.dev_score is not None
                        else {}
                    ),
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
        # Terminal marker on the run trace: outcome, iteration count, and
        # finalist count — the span the dashboard sorts/filters runs by.
        run_id = self._langfuse_run_id
        with langfuse_observation(
            "research_lab.loop_run_completed",
            metadata={
                "run_id": run_id,
                "loop_status": status if status in {"paused", "completed", "failed"} else "completed",
                "stop_reason": str(stop_reason)[:120],
                "iterations_completed": int(iterations_completed),
                "candidate_count": len(selected),
            },
            trace_id=langfuse_run_trace_id(run_id),
            sample_seed=run_id or None,
        ):
            pass
        return CodeEditLoopResult(
            # §6.3-1: rank-then-truncate so the kept candidates are the best
            # dev-scored builds; unchanged ordering without 2+ dev scores.
            selected_candidates=tuple(
                _rank_selected_by_dev_score(selected)[: self.settings.normalized().max_candidates]
            ),
            iterations_completed=int(iterations_completed),
            stop_reason=stop_reason,
            elapsed_seconds=round(float(elapsed_seconds), 3),
            estimated_cost_usd=round(float(estimated_cost), 6),
            actual_openrouter_cost_usd=round(int(actual_cost_microusd) / 1_000_000, 6),
            actual_openrouter_cost_microusd=int(actual_cost_microusd),
            openrouter_call_count=int(openrouter_calls),
            provider_usage=tuple(dict(item) for item in provider_usage if isinstance(item, Mapping)),
            status=status,
            checkpoint_doc=checkpoint,
        )

    async def _judge_plan_alignment(
        self,
        *,
        draft: CodeEditDraft,
        loop_direction_plan: Mapping[str, Any] | None,
        prior_attempts: Sequence[Mapping[str, Any]],
        node_id: str,
        iteration: int,
        settings: AutoResearchLoopSettings,
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
        settings: AutoResearchLoopSettings,
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
    object_key = f"{base_prefix}/candidates/{run_id}/loop-candidates/{int(iteration):03d}-{safe_node}.json"
    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_loop_candidate_rehydration",
        "run_id": str(run_id),
        "node_id": str(node_id),
        "iteration": int(iteration),
        "draft": draft.to_dict(),
        "candidate_model_manifest": build.candidate_model_manifest.to_dict(),
        "code_edit_manifest": dict(build.code_edit_manifest),
        "source_diff_hash": str(build.source_diff_hash),
        "build_doc": dict(build.build_doc),
    }
    if dev_score is not None:
        payload["dev_score"] = float(dev_score)
        payload["dev_score_version"] = str(dev_score_version or "")
    payload_hash = sha256_json(payload)

    try:
        stored_payload = {**payload, "loop_candidate_artifact_hash": payload_hash}
        if artifact_io is not None:
            await _artifact_io_write_json(
                artifact_io,
                uri=f"s3://{bucket}/{object_key}",
                document=stored_payload,
                content_hash=payload_hash,
                artifact_kind="loop_candidate_rehydration",
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
            "loop_candidate_artifact_hash": payload_hash,
            "loop_candidate_artifact_error": safe_event_error_text(exc),
        }
    return {
        "loop_candidate_artifact_uri": f"s3://{bucket}/{object_key}",
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
    return BuiltCodeEditCandidate(
        draft=draft,
        build=build,
        node_id=str(stored.get("node_id") or ""),
        iteration=int(stored.get("iteration") or 0),
        dev_score=dev_score,
        dev_score_version=str(stored.get("dev_score_version") or ""),
    )


def _node_id(run_id: str, iteration: int, candidate_index: int, draft: CodeEditDraft) -> str:
    digest = sha256_json(
        {
            "run_id": run_id,
            "iteration": iteration,
            "candidate_index": candidate_index,
            "draft": _redacted_draft_doc(draft),
        }
    ).split(":", 1)[1]
    return f"node:code-edit:{digest[:16]}"


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
