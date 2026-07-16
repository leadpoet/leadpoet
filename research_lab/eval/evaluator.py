"""Private-model Research Lab evaluator orchestration boundary."""

from __future__ import annotations

import asyncio
import contextlib
from importlib import import_module
import inspect
import logging
import os
import re
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Union

from .global_scoring_pool import get_global_scoring_pool

from leadpoet_verifier.aggregation import per_icp_normalized_score
from leadpoet_verifier.research_evaluation import (
    build_research_evaluation_score_bundle,
    score_bundle_hash,
)
from research_lab.canonical import canonical_json, sha256_json
from research_lab.employee_buckets import normalize_employee_count_bucket

from .artifacts import PrivateModelArtifactManifest, validate_private_model_artifact_manifest
from .benchmark import SealedBenchmarkSet, validate_sealed_benchmark_set
from .patches import (
    CandidatePatchManifest,
    runtime_compatible_candidate_patch_manifest,
    validate_candidate_patch_manifest,
)
from .private_runtime import (
    PrivateModelRuntimeError,
    begin_incontainer_trace_collection,
    canonicalize_private_model_icp,
    employee_count_buckets_for_icp,
    end_incontainer_trace_collection,
    ensure_private_model_outputs,
    incontainer_trace_capture_enabled,
    publish_incontainer_trace_entries,
)
from .provider_costs import summarize_provider_cost_trace_entries

logger = logging.getLogger(__name__)

ModelRunner = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Union[Awaitable[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]],
]
CompanyScorer = Callable[
    [Sequence[Mapping[str, Any]], Mapping[str, Any], bool],
    Union[Awaitable[list[float]], list[float]],
]
ParentFreshnessCheck = Callable[[Mapping[str, Any]], Union[Awaitable[None], None]]
IcpCheckpoint = Callable[[Mapping[str, Any]], Union[Awaitable[None], None]]
# Receives ``(icp_ref, entries)`` per completed ICP with the decoded
# in-container trace entries. May be sync or async; a returned string is used
# as the row's ``incontainer_trace_ref`` pointer. Sink failures are logged and
# swallowed — capture never fails a run.
TraceSink = Callable[[str, "list[dict[str, Any]]"], Union[Awaitable[Any], Any]]
# Optional gateway-supplied observer. It is intentionally a callback instead
# of a gateway import so the open evaluator remains reusable and telemetry
# cannot become a scoring dependency.
# NOTE: Optional[...] not PEP 604 `| None` — this alias is evaluated at
# runtime and the attested scoring workers run Python 3.9.
ScoringTelemetryHook = Callable[
    [str, Mapping[str, Any]],
    Union[Awaitable[Optional[Mapping[str, Any]]], Optional[Mapping[str, Any]]],
]
AttemptCostSink = Callable[
    [str, "list[dict[str, Any]]", Mapping[str, Any]],
    Union[Awaitable[None], None],
]
HoldoutTransitionHook = Callable[
    [str, Mapping[str, Any]],
    Union[
        Awaitable[Optional[Mapping[str, Any]]],
        Optional[Mapping[str, Any]],
    ],
]


async def _emit_scoring_telemetry(
    hook: ScoringTelemetryHook | None,
    action: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if hook is None:
        return None
    try:
        result = hook(action, dict(payload))
        if inspect.isawaitable(result):
            result = await result
        return dict(result) if isinstance(result, Mapping) else None
    except Exception:  # noqa: BLE001 - observation cannot affect scoring
        logger.warning(
            "research_lab_scoring_telemetry_hook_failed action=%s",
            action,
            exc_info=True,
        )
        return None


async def _emit_attempt_costs(
    sink: AttemptCostSink | None,
    icp_ref: str,
    entries: list[dict[str, Any]],
    context: Mapping[str, Any] | None,
) -> None:
    if sink is None or not entries:
        return
    try:
        result = sink(str(icp_ref), list(entries), dict(context or {}))
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001 - cost telemetry cannot affect scoring
        logger.warning(
            "research_lab_scoring_attempt_cost_sink_failed icp_ref=%s",
            str(icp_ref)[:120],
            exc_info=True,
        )


async def _emit_holdout_transition(
    hook: HoldoutTransitionHook | None,
    action: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Persist a gate boundary before beginning conditional provider work.

    Unlike optional telemetry, this callback is part of crash-safe orchestration:
    failures propagate so a worker cannot run the hidden stage without recording
    that it crossed the frozen preliminary gate.
    """

    if hook is None:
        return None
    result = hook(action, dict(payload))
    if inspect.isawaitable(result):
        result = await result
    return dict(result) if isinstance(result, Mapping) else None

# Default trace sink configuration (used when no ``trace_sink`` is injected):
# with the S3 prefix set, one JSON object per ICP is uploaded to
# ``{prefix}/{run_or_candidate_ref}/{icp_ref}.json`` (SSE-KMS when the key id
# is set); without it, entries are counted and dropped (logged once per eval).
INCONTAINER_TRACE_S3_PREFIX_ENV = "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX"
INCONTAINER_TRACE_KMS_KEY_ENV = "RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID"

# Company scores are normalized against the ICP's requested lead budget when
# the capped scoring flag is on; must match the verifier's advisory recompute
# (leadpoet_verifier.research_evaluation.DEFAULT_LEADS_PER_ICP_NORMALIZER).
# This is the fallback budget for ICPs that carry no max_companies goal.
_TOP5_LEADS_PER_ICP = 5
# Same clamp as the goal-seeking sourcing/scoring paths
# (gateway.research_lab.scoring_worker._icp_company_goal).
_MAX_ICP_COMPANY_GOAL = 50
_EVAL_ENV_TRUTHY = {"1", "true", "yes", "on"}
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
    return any(marker in lowered for marker in _PROVIDER_COST_CAP_ERROR_MARKERS) or _provider_error_status_code(lowered) == 402


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in _EVAL_ENV_TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _timeout_latch_legacy_enabled() -> bool:
    """Bug #14 revert switch: latch the candidate skip on the FIRST timeout.

    Default (false) retries a timed-out ICP once and only latches the
    remaining-ICP skip after 2+ consecutive post-retry timeouts.
    """
    return _env_flag("RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY", False)


def _provider_flake_retry_enabled() -> bool:
    """Retry retryable candidate provider errors before scoring them as misses.

    Default ON; flip off only to skip provider retries. Retry-exhausted provider
    failures are never excluded from aggregates in the current scoring contract.
    """
    return _env_flag("RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY", True)


def _capped_top5_score_enabled() -> bool:
    """Bug #8 score-scale switch: per-ICP score becomes the verifier's capped
    sum(top-N company scores)/N instead of the unweighted company mean, where
    N is the ICP's requested company count (max_companies, fallback 5).

    Under the mean, one 80-score company earns the same ICP score as three
    80-score companies, so the model is never rewarded for filling the
    requested quantity. The capped form divides by the requested count, so
    every unfilled slot costs score.

    Default OFF — enabling changes the score scale, so the daily baseline and
    all candidates must flip in the same deploy followed by a fresh baseline
    run (never compare scores across the boundary).
    """
    return _env_flag("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", False)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _fp_penalty_points() -> float:
    """Per-company penalty for deterministic false positives.

    A company zeroed by a model-controllable gate (exclusion list, required
    attribute, missing hard fields, country/duplicate/data-quality
    pre-checks, company re-verification) subtracts this many points from
    the ICP's score sum before normalization, so shipping junk scores
    WORSE than honestly returning fewer companies. Default 0 = off; like
    the capped flag, enable only together with a fresh baseline (never
    compare scores across the boundary).
    """
    return max(0.0, _env_float("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", 0.0))


def _fp_unverified_primary_penalty_points() -> float:
    """Per-company penalty when the ICP's PRIMARY intent failed verification.

    Separate knob from the deterministic-gate penalty because intent
    verification involves providers; keep 0 until the verifier's behavior
    under transient provider failures is proven not to blame the model.
    """
    return max(
        0.0,
        _env_float("RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY", 0.0),
    )


# Failure reasons that count as model-controllable false positives. Matched
# as case-insensitive substrings of LeadScoreBreakdown.failure_reason. Keep
# this list conservative: provider/infra failures must NEVER appear here.
PENALIZABLE_FAILURE_MARKERS: tuple = (
    "exclusion list",             # ICP excluded_companies hit
    "required_attribute",         # attribute claim absent / failed / unbacked
    "missing employee_count",     # hard-field gaps in the model's own output
    "missing company_stage",
    "country mismatch",           # deterministic zero-check gates
    "missing country",
    "duplicate company",
    "data quality issue",
    "missing industry",
    "company verification failed",  # scorer re-verification rejected
)

_NEVER_PENALIZE_MARKERS: tuple = ("error", "timeout", "provider", "429")


def _failure_reason_is_penalizable(reason: Any) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _NEVER_PENALIZE_MARKERS):
        return False
    return any(marker in text for marker in PENALIZABLE_FAILURE_MARKERS)


def count_penalizable_false_positives(
    breakdowns: Sequence[Mapping[str, Any]],
    *,
    icp_has_intent_signals: bool,
) -> tuple[int, int]:
    """(gate_fp_count, unverified_primary_count) for one ICP's breakdowns.

    Gate FPs are zero-score rows whose failure_reason is a deterministic,
    model-controllable rejection. Unverified-primary rows are companies the
    scorer kept (fit points only) after zeroing intent because no verified
    signal matched the ICP's primary intent; they are counted separately
    and penalized by their own knob. Derived from breakdowns only, so the
    cache-hit and cache-miss scoring paths count identically.
    """
    gate_fps = 0
    unverified_primary = 0
    for row in breakdowns:
        if not isinstance(row, Mapping):
            continue
        reason = row.get("failure_reason")
        if reason:
            if _failure_reason_is_penalizable(reason):
                gate_fps += 1
            continue
        if not icp_has_intent_signals:
            continue
        details = row.get("intent_signals_detail")
        if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
            continue
        primary_verified = False
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            try:
                idx = int(detail.get("matched_icp_signal", -1))
            except (TypeError, ValueError):
                continue
            if idx == 0 and float(detail.get("after_decay") or 0.0) > 0.0:
                primary_verified = True
                break
        if details and not primary_verified:
            unverified_primary += 1
    return gate_fps, unverified_primary


def fp_penalty_total_from_breakdowns(
    breakdowns: Sequence[Mapping[str, Any]],
    icp: Any,
) -> float:
    """Pre-multiplied FP penalty points for one ICP under the current knobs.

    Shared by the daily-baseline path (scoring_worker) and any caller that
    holds raw breakdowns, so baseline and candidate scoring penalize
    identically.
    """
    icp_has_intents = bool(
        (icp.get("intent_signals") if isinstance(icp, Mapping) else None)
        or (icp.get("intent_signal") if isinstance(icp, Mapping) else None)
        or getattr(icp, "intent_signals", None)
        or getattr(icp, "intent_signal", None)
    )
    gate, primary = count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=icp_has_intents
    )
    return gate * _fp_penalty_points() + primary * _fp_unverified_primary_penalty_points()


def _icp_company_goal(icp: Any) -> int:
    """The ICP's requested company count (max_companies), clamped to 1..50.

    Falls back to the fixed 5-lead budget when the ICP carries no goal, so
    legacy ICP sets keep the historical normalization exactly.
    """
    raw = icp.get("max_companies") if isinstance(icp, Mapping) else None
    if raw is None:
        return _TOP5_LEADS_PER_ICP
    try:
        goal = int(raw)
    except (TypeError, ValueError):
        return _TOP5_LEADS_PER_ICP
    return max(1, min(goal, _MAX_ICP_COMPANY_GOAL))


def _max_scored_companies_per_icp() -> int:
    """Bug #8 cost control: cap how many companies are LLM-scored per ICP.

    Default 0 = unlimited (legacy). When RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE is
    enabled, keep this cap >= 5 so the top-5 normalization still sees a full
    lead budget. Applied inside QualificationStyleCompanyScorer so the baseline
    and candidate paths are capped identically.
    """
    return max(0, _env_int("RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES", 0))


def _candidate_scoring_concurrency() -> int:
    """Optional ICP-level concurrency for candidate scoring (default 1 = serial,
    the legacy behavior). Results are always assembled in benchmark-item order
    so summary hashing stays deterministic regardless of completion order.
    """
    return max(1, _env_int("RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY", 1))


def _candidate_provider_retry_rounds() -> int:
    """Candidate retry rounds mirror daily-baseline provider retries.

    The shared env keeps candidate and rebenchmark scoring comparable:
    initial attempt + N retry rounds for retryable provider/timeouts.
    """
    return max(0, _env_int("RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS", 2))


def _work_conserving_enabled() -> bool:
    """Opt-in work-conserving ICP scheduling (default off = legacy waves).

    The fixed-size wave loop is a barrier: 4 fast ICPs idle while the wave's
    straggler finishes, so container slots sit unused (observed 5<->10
    oscillation at concurrency 5). The work-conserving pool keeps exactly
    ``concurrency`` ICP jobs in flight and starts the next queued ICP the
    moment one finishes. Results are still assembled in benchmark-item order
    (hash determinism); the timeout latch and freshness checks run in
    completion order, which concurrency>1 already made approximate.
    """
    raw = str(os.getenv("RESEARCH_LAB_EVAL_WORK_CONSERVING") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class RealEvaluatorRequired(RuntimeError):
    """Raised when a production Research Lab evaluation lacks real inputs."""


class ConditionalValidationRetryableError(RuntimeError):
    """Conditional evidence was incomplete and must be retried, not scored."""


async def evaluate_private_model_pair(
    *,
    artifact_manifest: PrivateModelArtifactManifest | Mapping[str, Any],
    benchmark: SealedBenchmarkSet | Mapping[str, Any],
    patch_manifest: CandidatePatchManifest | Mapping[str, Any],
    candidate_artifact_manifest: PrivateModelArtifactManifest | Mapping[str, Any] | None = None,
    benchmark_items: Sequence[Mapping[str, Any]],
    base_runner: ModelRunner | None,
    candidate_runner: ModelRunner | None,
    company_scorer: CompanyScorer | None = None,
    run_context: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
    private_holdout_gate: Mapping[str, Any] | None = None,
    parent_freshness_check: ParentFreshnessCheck | None = None,
    icp_checkpoint: IcpCheckpoint | None = None,
    resume_results: Sequence[Mapping[str, Any]] | None = None,
    trace_sink: TraceSink | None = None,
    scoring_telemetry_hook: ScoringTelemetryHook | None = None,
    attempt_cost_sink: AttemptCostSink | None = None,
    holdout_transition_hook: HoldoutTransitionHook | None = None,
) -> dict[str, Any]:
    """Run a real paired base-vs-candidate evaluation.

    Production callers must pass private model runner callables backed by the
    immutable artifact. This function deliberately has no fallback fixture model.

    ``icp_checkpoint`` (optional) is invoked with each completed per-ICP result
    so callers can persist progress; ``resume_results`` (optional) supplies
    previously completed per-ICP results (matched by icp_ref/icp_hash) that are
    reused instead of re-running their ICPs. Both default to off so existing
    callers are unchanged.

    ``trace_sink`` (optional) receives the decoded in-container trace entries
    per completed ICP (see ``TraceSink``); when ``None`` the default sink is
    used (S3 upload when configured, count-and-drop otherwise). Capture is
    gated by ``RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE`` (default on) and is
    pure observation: rows only ever gain pointer fields, never content.
    """
    artifact = artifact_manifest if isinstance(artifact_manifest, PrivateModelArtifactManifest) else PrivateModelArtifactManifest.from_mapping(artifact_manifest)
    benchmark_set = benchmark if isinstance(benchmark, SealedBenchmarkSet) else SealedBenchmarkSet.from_mapping(benchmark)
    image_candidate = candidate_artifact_manifest is not None
    candidate_artifact = (
        candidate_artifact_manifest
        if isinstance(candidate_artifact_manifest, PrivateModelArtifactManifest)
        else (
            PrivateModelArtifactManifest.from_mapping(candidate_artifact_manifest)
            if candidate_artifact_manifest is not None
            else None
        )
    )
    patch = None if image_candidate else (
        patch_manifest if isinstance(patch_manifest, CandidatePatchManifest) else CandidatePatchManifest.from_mapping(patch_manifest)
    )

    errors = []
    errors.extend(validate_private_model_artifact_manifest(artifact))
    errors.extend(validate_sealed_benchmark_set(benchmark_set))
    if image_candidate:
        if candidate_artifact is None:
            errors.append("candidate_artifact_manifest_required_for_image_candidate")
        else:
            errors.extend(validate_private_model_artifact_manifest(candidate_artifact))
            if candidate_artifact.model_artifact_hash == artifact.model_artifact_hash:
                errors.append("candidate_artifact_hash_must_differ_from_parent")
            patch_parent = str(dict(patch_manifest).get("parent_artifact_hash") or "")
            if patch_parent and patch_parent != artifact.model_artifact_hash:
                errors.append("parent_artifact_hash_mismatch")
    else:
        errors.extend(validate_candidate_patch_manifest(patch, expected_parent_artifact_hash=artifact.model_artifact_hash))
    if errors:
        raise ValueError("; ".join(errors))
    if candidate_runner is None:
        raise RealEvaluatorRequired("private candidate_runner is required")
    if private_holdout_gate is None and base_runner is None:
        raise RealEvaluatorRequired("private base_runner is required for paired evaluation")
    if not benchmark_items:
        raise RealEvaluatorRequired("sealed benchmark items are required")

    scorer = company_scorer or QualificationStyleCompanyScorer()
    runtime_patch = None if image_candidate else runtime_compatible_candidate_patch_manifest(patch)
    # Resolve the trace sink once so the holdout gate's two scoring passes
    # share one sink instance (and one log-once drop notice).
    resolved_trace_sink = _resolve_trace_sink(trace_sink, run_context)
    if private_holdout_gate:
        per_icp_results, gate_result = await _score_with_private_holdout_gate(
            benchmark_items=benchmark_items,
            base_runner=None,
            candidate_runner=candidate_runner,
            scorer=scorer,
            run_context=run_context,
            image_candidate=image_candidate,
            runtime_patch=runtime_patch,
            gate=private_holdout_gate,
            parent_freshness_check=parent_freshness_check,
            icp_checkpoint=icp_checkpoint,
            resume_results=resume_results,
            trace_sink=resolved_trace_sink,
            scoring_telemetry_hook=scoring_telemetry_hook,
            attempt_cost_sink=attempt_cost_sink,
            holdout_transition_hook=holdout_transition_hook,
        )
        return build_score_bundle_from_scored_icps(
            artifact_manifest=artifact,
            benchmark=benchmark_set,
            patch_manifest=patch_manifest,
            candidate_artifact_manifest=candidate_artifact,
            per_icp_results=per_icp_results,
            run_context=run_context,
            policy=policy or {},
            extra_bundle_fields={"private_holdout_gate": gate_result},
        )

    per_icp_results = await score_private_model_pair_items(
        benchmark_items=benchmark_items,
        base_runner=base_runner,
        candidate_runner=candidate_runner,
        company_scorer=scorer,
        run_context=run_context,
        image_candidate=image_candidate,
        runtime_patch=runtime_patch,
        parent_freshness_check=parent_freshness_check,
        icp_checkpoint=icp_checkpoint,
        resume_results=resume_results,
        trace_sink=resolved_trace_sink,
        scoring_telemetry_hook=scoring_telemetry_hook,
        attempt_cost_sink=attempt_cost_sink,
    )
    return build_score_bundle_from_scored_icps(
        artifact_manifest=artifact,
        benchmark=benchmark_set,
        patch_manifest=patch_manifest,
        candidate_artifact_manifest=candidate_artifact,
        per_icp_results=per_icp_results,
        run_context=run_context,
        policy=policy or {},
    )


async def score_private_model_pair_items(
    *,
    benchmark_items: Sequence[Mapping[str, Any]],
    base_runner: ModelRunner | None,
    candidate_runner: ModelRunner,
    company_scorer: CompanyScorer | None = None,
    run_context: Mapping[str, Any],
    image_candidate: bool,
    runtime_patch: CandidatePatchManifest | None = None,
    parent_freshness_check: ParentFreshnessCheck | None = None,
    icp_checkpoint: IcpCheckpoint | None = None,
    resume_results: Sequence[Mapping[str, Any]] | None = None,
    trace_sink: TraceSink | None = None,
    scoring_telemetry_hook: ScoringTelemetryHook | None = None,
    attempt_cost_sink: AttemptCostSink | None = None,
) -> list[dict[str, Any]]:
    """Score a subset of private benchmark items without building a bundle.

    Fairness semantics (bugs #14/#15):
    - A timed-out candidate ICP is retried once; the remaining-ICP skip latch
      only engages after 2+ consecutive post-retry timeouts
      (``RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY=true`` restores first-timeout
      latching with no retry).
    - Retryable candidate provider errors are retried for the same number of
      rounds as daily baseline rebenchmarks; retry-exhausted failures score 0
      and remain in the aggregate.
    - Any candidate ICP that returns zero companies without a recorded runtime
      error is flagged ``sourced_zero_no_error`` (bug #35) so health gates and
      unresolved-error counters can see silent zeros.

    ``icp_checkpoint``/``resume_results`` implement optional per-ICP
    checkpointing (bug #31); resumed rows are reused verbatim, do not re-invoke
    the checkpoint sink, and do not affect the timeout latch.

    ``RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY`` (default 1 = the legacy serial
    behavior) fans ICPs out in fixed-size waves; results are always assembled
    in benchmark-item order so summary hashing stays deterministic. At
    concurrency 1 the serial semantics (latch visibility, freshness-check
    ordering) are preserved exactly; at higher settings ICPs inside one wave
    cannot see a latch raised by a wave-mate.

    ``trace_sink`` (optional) receives decoded in-container trace entries per
    completed ICP; rows that produced entries gain POINTER fields only
    (``incontainer_trace_ref``/``incontainer_trace_sha256``/
    ``incontainer_trace_call_count``). With capture disabled
    (``RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE=false``) or runners that emit no
    trace markers, rows are byte-identical to the pre-capture shape.
    """

    scorer = company_scorer or QualificationStyleCompanyScorer()
    legacy_timeout_latch = _timeout_latch_legacy_enabled()
    provider_flake_retry = _provider_flake_retry_enabled()
    effective_trace_sink = _resolve_trace_sink(trace_sink, run_context)
    concurrency = _candidate_scoring_concurrency()
    resume_rows = _resume_rows_by_ref(resume_results)
    if concurrency > 1 and _work_conserving_enabled():
        return await _score_items_work_conserving(
            benchmark_items=benchmark_items,
            base_runner=base_runner,
            candidate_runner=candidate_runner,
            scorer=scorer,
            run_context=run_context,
            image_candidate=image_candidate,
            runtime_patch=runtime_patch,
            parent_freshness_check=parent_freshness_check,
            icp_checkpoint=icp_checkpoint,
            resume_rows=resume_rows,
            trace_sink=effective_trace_sink,
            concurrency=concurrency,
            legacy_timeout_latch=legacy_timeout_latch,
            provider_flake_retry=provider_flake_retry,
            scoring_telemetry_hook=scoring_telemetry_hook,
            attempt_cost_sink=attempt_cost_sink,
        )
    per_icp_results: list[dict[str, Any]] = []
    candidate_runtime_skip_reason = ""
    consecutive_candidate_timeouts = 0
    indexed_items = list(enumerate(benchmark_items))
    for chunk_start in range(0, len(indexed_items), concurrency):
        chunk = indexed_items[chunk_start : chunk_start + concurrency]
        completed_before_chunk = len(per_icp_results)
        entries: list[tuple[str, int, Mapping[str, Any], dict[str, Any] | None]] = []
        for item_index, item in chunk:
            resumed = resume_rows.get(_benchmark_item_ref(item)) if resume_rows else None
            if resumed is not None:
                entries.append(("resumed", item_index, item, dict(resumed)))
            else:
                entries.append(("run", item_index, item, None))
        pending = [(position, entry) for position, entry in enumerate(entries) if entry[0] == "run"]
        outcomes: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        if pending:
            # Settle-then-raise: blocking runner waits cannot be cancelled, so a
            # fail-fast gather would leave containers racing the failure path.
            # Wait for every ICP in the wave, then surface the first fatal error
            # in benchmark order.
            settled = await asyncio.gather(
                *(
                    _score_single_icp(
                        item=entry[2],
                        item_index=entry[1],
                        completed_icp_count=completed_before_chunk,
                        base_runner=base_runner,
                        candidate_runner=candidate_runner,
                        scorer=scorer,
                        run_context=run_context,
                        image_candidate=image_candidate,
                        runtime_patch=runtime_patch,
                        parent_freshness_check=parent_freshness_check,
                        candidate_runtime_skip_reason=candidate_runtime_skip_reason,
                        legacy_timeout_latch=legacy_timeout_latch,
                        provider_flake_retry=provider_flake_retry,
                        trace_sink=effective_trace_sink,
                        scoring_telemetry_hook=scoring_telemetry_hook,
                        attempt_cost_sink=attempt_cost_sink,
                    )
                    for _position, entry in pending
                ),
                return_exceptions=True,
            )
            fatal = [entry for entry in settled if isinstance(entry, BaseException)]
            if fatal:
                raise fatal[0]
            for (position, _entry), outcome in zip(pending, settled):
                outcomes[position] = outcome
        for position, (kind, item_index, item, resumed_row) in enumerate(entries):
            if kind == "resumed":
                per_icp_results.append(resumed_row or {})
                continue
            row, markers = outcomes[position]
            if markers["timed_out"]:
                consecutive_candidate_timeouts += 1
            elif not markers["skipped"]:
                consecutive_candidate_timeouts = 0
            if markers["latch_reason"] and not candidate_runtime_skip_reason:
                candidate_runtime_skip_reason = markers["latch_reason"]
            if (
                not legacy_timeout_latch
                and not candidate_runtime_skip_reason
                and consecutive_candidate_timeouts >= 2
            ):
                candidate_runtime_skip_reason = _candidate_runtime_skip_reason(
                    "candidate_model_runtime_timeout"
                )
            per_icp_results.append(row)
            if icp_checkpoint is not None:
                checkpoint_result = icp_checkpoint(row)
                if inspect.isawaitable(checkpoint_result):
                    await checkpoint_result
            await _run_parent_freshness_check(
                parent_freshness_check,
                {
                    "phase": "after_icp",
                    "last_icp_index": item_index,
                    "completed_icp_count": len(per_icp_results),
                    "icp_ref": str(item.get("icp_ref") or item.get("icp_hash") or ""),
                    "icp_hash": str(item.get("icp_hash") or ""),
                },
            )
    return per_icp_results


async def _score_items_work_conserving(
    *,
    benchmark_items: Sequence[Mapping[str, Any]],
    base_runner: ModelRunner | None,
    candidate_runner: ModelRunner,
    scorer: CompanyScorer,
    run_context: Mapping[str, Any],
    image_candidate: bool,
    runtime_patch: CandidatePatchManifest | None,
    parent_freshness_check: ParentFreshnessCheck | None,
    icp_checkpoint: IcpCheckpoint | None,
    resume_rows: Mapping[str, Mapping[str, Any]],
    trace_sink: TraceSink | None,
    concurrency: int,
    legacy_timeout_latch: bool,
    provider_flake_retry: bool,
    scoring_telemetry_hook: ScoringTelemetryHook | None,
    attempt_cost_sink: AttemptCostSink | None,
) -> list[dict[str, Any]]:
    """Work-conserving ICP scheduler: exactly ``concurrency`` jobs in flight.

    Replaces the fixed-wave barrier — the moment one ICP finishes, the next
    queued ICP starts, so scoring slots never idle behind a wave straggler.

    Behavior preserved from the wave loop:
      * results returned in benchmark-item order (canonical hash determinism);
      * resumed rows reused verbatim, never re-executed, never checkpointed,
        and excluded from the timeout latch;
      * per-completion checkpointing + ``after_icp`` freshness checks
        (serialized under a lock; order follows completion, which the wave
        loop already made approximate at concurrency>1);
      * the 2-consecutive-timeouts latch: once raised, jobs that have not yet
        STARTED run through the existing skip path (skip reason is read at
        job start);
      * settle-then-raise: on a fatal error no NEW jobs start, in-flight jobs
        settle (blocking container waits cannot be cancelled), then the
        lowest-benchmark-index fatal is raised.
    """
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    lock = asyncio.Lock()
    results: dict[int, dict[str, Any]] = {}
    fatals: dict[int, BaseException] = {}
    state = {
        "skip_reason": "",
        "consecutive_timeouts": 0,
        "completed": 0,
        "stop": False,
    }

    async def _run_one(position: int, item_index: int, item: Mapping[str, Any]) -> None:
        async with semaphore:
            if state["stop"]:
                return  # fatal already recorded; do not start new containers
            try:
                row, markers = await _score_single_icp(
                    item=item,
                    item_index=item_index,
                    completed_icp_count=state["completed"],
                    base_runner=base_runner,
                    candidate_runner=candidate_runner,
                    scorer=scorer,
                    run_context=run_context,
                    image_candidate=image_candidate,
                    runtime_patch=runtime_patch,
                    parent_freshness_check=parent_freshness_check,
                    candidate_runtime_skip_reason=state["skip_reason"],
                    legacy_timeout_latch=legacy_timeout_latch,
                    provider_flake_retry=provider_flake_retry,
                    trace_sink=trace_sink,
                    scoring_telemetry_hook=scoring_telemetry_hook,
                    attempt_cost_sink=attempt_cost_sink,
                )
            except BaseException as exc:  # noqa: BLE001 - settle-then-raise
                async with lock:
                    fatals[position] = exc
                    state["stop"] = True
                return
            async with lock:
                results[position] = row
                state["completed"] += 1
                if markers["timed_out"]:
                    state["consecutive_timeouts"] += 1
                elif not markers["skipped"]:
                    state["consecutive_timeouts"] = 0
                if markers["latch_reason"] and not state["skip_reason"]:
                    state["skip_reason"] = markers["latch_reason"]
                if (
                    not legacy_timeout_latch
                    and not state["skip_reason"]
                    and state["consecutive_timeouts"] >= 2
                ):
                    state["skip_reason"] = _candidate_runtime_skip_reason(
                        "candidate_model_runtime_timeout"
                    )
                if icp_checkpoint is not None:
                    checkpoint_result = icp_checkpoint(row)
                    if inspect.isawaitable(checkpoint_result):
                        await checkpoint_result
                await _run_parent_freshness_check(
                    parent_freshness_check,
                    {
                        "phase": "after_icp",
                        "last_icp_index": item_index,
                        "completed_icp_count": state["completed"],
                        "icp_ref": str(item.get("icp_ref") or item.get("icp_hash") or ""),
                        "icp_hash": str(item.get("icp_hash") or ""),
                    },
                )

    tasks: list[asyncio.Task] = []
    for position, item in enumerate(benchmark_items):
        resumed = resume_rows.get(_benchmark_item_ref(item)) if resume_rows else None
        if resumed is not None:
            results[position] = dict(resumed)
            continue
        tasks.append(asyncio.create_task(_run_one(position, position, item)))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    if fatals:
        raise fatals[min(fatals)]
    # Assemble in benchmark order; positions missing only if a fatal aborted
    # remaining jobs, and fatals raise above — so this is total.
    return [results[position] for position in sorted(results)]


@contextlib.asynccontextmanager
async def _noop_scoring_pool_slot():
    yield False


def _scoring_pool_slot():
    """Shared cross-process concurrency slot for one candidate-model run.

    Returns the global slot pool's async context when a pool size is configured,
    otherwise a no-op context so behavior is unchanged when pooling is off.
    """
    pool = get_global_scoring_pool()
    if pool is None:
        return _noop_scoring_pool_slot()
    return pool.slot()


async def _score_single_icp(
    *,
    item: Mapping[str, Any],
    item_index: int,
    completed_icp_count: int,
    base_runner: ModelRunner | None,
    candidate_runner: ModelRunner,
    scorer: CompanyScorer,
    run_context: Mapping[str, Any],
    image_candidate: bool,
    runtime_patch: CandidatePatchManifest | None,
    parent_freshness_check: ParentFreshnessCheck | None,
    candidate_runtime_skip_reason: str,
    legacy_timeout_latch: bool,
    provider_flake_retry: bool,
    trace_sink: TraceSink | None = None,
    scoring_telemetry_hook: ScoringTelemetryHook | None = None,
    attempt_cost_sink: AttemptCostSink | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run and score one benchmark ICP.

    Returns ``(row, markers)`` where ``markers`` carries the latch signals the
    ordered driver loop needs (``timed_out``, ``latch_reason``, ``skipped``).
    """
    item_ref = str(item.get("icp_ref") or item.get("icp_hash") or "")
    item_hash = str(item.get("icp_hash") or "")
    await _run_parent_freshness_check(
        parent_freshness_check,
        {
            "phase": "before_icp",
            "next_icp_index": item_index,
            "completed_icp_count": completed_icp_count,
            "icp_ref": str(item.get("icp_ref") or item.get("icp_hash") or ""),
            "icp_hash": str(item.get("icp_hash") or ""),
        },
    )
    icp = item.get("icp")
    if not isinstance(icp, Mapping):
        raise RealEvaluatorRequired("benchmark item is missing private ICP payload")
    failure_reasons: list[str] = []
    # In-container trace capture: install a per-task collector so runner calls
    # (including sync runners hopping through asyncio.to_thread, which copies
    # this context) can publish decoded trace entries. Tasks in a concurrent
    # wave each carry their own context, so collectors never cross ICPs.
    trace_entries: list[dict[str, Any]] | None = None
    trace_token = None
    base_trace_entry_count = 0
    if trace_sink is not None:
        trace_entries, trace_token = begin_incontainer_trace_collection()
    try:
        if base_runner is None:
            base_outputs = []
        else:
            try:
                base_outputs = ensure_private_model_outputs(
                    await _call_model_runner(base_runner, icp, run_context),
                    context_label=f"reference model for ICP {item.get('icp_ref') or item.get('icp_hash') or ''}",
                    require_non_empty=False,
                )
            except PrivateModelRuntimeError as exc:
                if not _is_provider_backed_sourcing_error(exc):
                    raise
                base_outputs = []
                failure_reasons.append("reference_model_runtime_provider_error")
        if trace_entries is not None:
            base_trace_entry_count = len(trace_entries)
        candidate_context = dict(run_context)
        if not image_candidate:
            if runtime_patch is None:
                raise RealEvaluatorRequired("candidate patch runtime payload is required for patch candidates")
            candidate_context["patch"] = runtime_patch.to_dict()
        markers = {"timed_out": False, "latch_reason": "", "skipped": False}
        provider_excluded = False
        if candidate_runtime_skip_reason:
            candidate_outputs = []
            failure_reasons.append(candidate_runtime_skip_reason)
            markers["skipped"] = True
            await _emit_scoring_telemetry(
                scoring_telemetry_hook,
                "attempt_skipped",
                {
                    "icp_ref": item_ref,
                    "icp_hash": item_hash,
                    "icp_ordinal": item_index,
                    "model_role": "candidate",
                    "retry_round": 0,
                    "failure_category": candidate_runtime_skip_reason,
                },
            )
        else:
            # Hold one shared pool slot only while the candidate model actually
            # runs, so total concurrent candidate-model containers stay pinned
            # at the pool size across every scoring process. The skip path above
            # holds no slot. Pooling is disabled (no-op) when unconfigured.
            async with _scoring_pool_slot():
                candidate_outputs, candidate_failure_reason, provider_excluded = await _run_candidate_with_retries(
                    candidate_runner=candidate_runner,
                    icp=icp,
                    candidate_context=candidate_context,
                    item_label=str(item.get("icp_ref") or item.get("icp_hash") or ""),
                    legacy_timeout_latch=legacy_timeout_latch,
                    provider_flake_retry=provider_flake_retry,
                    item_hash=item_hash,
                    item_ordinal=item_index,
                    scoring_telemetry_hook=scoring_telemetry_hook,
                    attempt_cost_sink=attempt_cost_sink,
                )
            if candidate_failure_reason:
                failure_reasons.append(candidate_failure_reason)
                if candidate_failure_reason == "candidate_model_runtime_timeout":
                    markers["timed_out"] = True
                    if legacy_timeout_latch:
                        markers["latch_reason"] = _candidate_runtime_skip_reason(candidate_failure_reason)
                else:
                    markers["latch_reason"] = _candidate_runtime_skip_reason(candidate_failure_reason)
        if not markers["skipped"]:
            await _emit_scoring_telemetry(
                scoring_telemetry_hook,
                "scoring_started",
                {
                    "icp_ref": item_ref,
                    "icp_hash": item_hash,
                    "icp_ordinal": item_index,
                    "model_role": "candidate",
                },
            )
        # Score via breakdowns when the scorer exposes them so the per-ICP
        # false-positive counts can be derived (the day-scoped scoring cache
        # stores breakdowns, so cache hits count identically). Falls back to
        # the score-list contract for scorers without breakdown support.
        base_breakdowns: list[Mapping[str, Any]] = []
        candidate_breakdowns: list[Mapping[str, Any]] = []
        if hasattr(scorer, "score_with_breakdowns"):
            if base_runner is not None:
                base_breakdowns = await _maybe_await(
                    scorer.score_with_breakdowns(base_outputs, icp, True)
                )
            candidate_breakdowns = await _maybe_await(
                scorer.score_with_breakdowns(candidate_outputs, icp, False)
            )
            base_scores = [
                float(item.get("final_score", 0.0) or 0.0) for item in base_breakdowns
            ]
            candidate_scores = [
                float(item.get("final_score", 0.0) or 0.0)
                for item in candidate_breakdowns
            ]
        else:
            base_scores = await _maybe_await(scorer(base_outputs, icp, True)) if base_runner is not None else []
            candidate_scores = await _maybe_await(scorer(candidate_outputs, icp, False))
    finally:
        if trace_token is not None:
            end_incontainer_trace_collection(trace_token)
    icp_has_intents = bool(
        (icp.get("intent_signals") if isinstance(icp, Mapping) else None)
        or (icp.get("intent_signal") if isinstance(icp, Mapping) else None)
    )
    base_fp_gate, base_fp_primary = count_penalizable_false_positives(
        base_breakdowns, icp_has_intent_signals=icp_has_intents
    )
    cand_fp_gate, cand_fp_primary = count_penalizable_false_positives(
        candidate_breakdowns, icp_has_intent_signals=icp_has_intents
    )
    if base_runner is not None and not base_outputs:
        failure_reasons.append("reference_model_zero_companies")
    elif base_runner is not None and not base_scores:
        failure_reasons.append("reference_model_zero_scoreable_companies")
    if not candidate_outputs:
        failure_reasons.append("candidate_model_zero_companies")
    elif not candidate_scores:
        failure_reasons.append("candidate_model_zero_scoreable_companies")
    row = {
        "icp_ref": str(item.get("icp_ref") or item.get("icp_hash") or ""),
        "icp_hash": str(item.get("icp_hash") or ""),
        "status": "completed",
        "hard_failure": False,
        # Requested company count for this ICP; the capped per-ICP score
        # normalizes against it (verifier recomputes with the same value).
        "icp_company_goal": _icp_company_goal(icp),
        # Per-side false-positive counts (deterministic-gate rejections and
        # unverified-primary-intent companies); the FP penalty knobs multiply
        # these at aggregation time and the verifier recomputes with the
        # penalty points recorded in the bundle policy.
        "base_fp_gate_count": base_fp_gate,
        "base_fp_unverified_primary_count": base_fp_primary,
        "candidate_fp_gate_count": cand_fp_gate,
        "candidate_fp_unverified_primary_count": cand_fp_primary,
        "base_company_scores": base_scores,
        "candidate_company_scores": candidate_scores,
        "failure_reason": ";".join(failure_reasons),
        "provider_excluded": provider_excluded,
        # Bug #35: a zero-company result with no recorded runtime error is the
        # silent-zero blind spot — flag it so gates/counters can see it.
        "sourced_zero_no_error": (
            not candidate_outputs
            and not any(reason.startswith("candidate_model_runtime_") for reason in failure_reasons)
        ),
        "reference_sourced_zero_no_error": (
            base_runner is not None
            and not base_outputs
            and not any(reason.startswith("reference_model_runtime_") for reason in failure_reasons)
        ),
    }
    # P12 dense per-claim reward: deterministic L0 verdicts per submitted
    # signal (check ids + statuses only) ride the row into the score bundle.
    l0_findings = _l0_per_claim_findings(candidate_outputs, icp)
    if l0_findings:
        row["l0_findings"] = l0_findings
    # P12: the gateway's trace-capturing scorer exposes its per-ICP judgment
    # pointer; duck-typed so the default scorer needs no change.
    pointer_for = getattr(scorer, "scorer_trace_pointer_for", None)
    if callable(pointer_for):
        try:
            pointer = pointer_for(row["icp_ref"] or row["icp_hash"])
        except Exception:  # noqa: BLE001 - pointer pickup is best-effort
            pointer = None
        if isinstance(pointer, Mapping) and pointer.get("s3_ref"):
            row["scorer_trace_ref"] = str(pointer["s3_ref"])
            row["scorer_trace_sha256"] = str(pointer.get("sha256") or "")
    # Per-ICP candidate funnel counts, duck-typed so the default scorer needs
    # no change.
    funnel_for = getattr(scorer, "scorer_funnel_for", None)
    if callable(funnel_for):
        try:
            funnel = funnel_for(row["icp_ref"] or row["icp_hash"])
        except Exception:  # noqa: BLE001 - funnel pickup is best-effort
            funnel = None
        if isinstance(funnel, Mapping):
            row["funnel"] = dict(funnel)
    # Per-evidence-type intent stats (intent pass rate panel); duck-typed.
    evidence_types_for = getattr(scorer, "scorer_evidence_types_for", None)
    if callable(evidence_types_for):
        try:
            evidence_types = evidence_types_for(row["icp_ref"] or row["icp_hash"])
        except Exception:  # noqa: BLE001 - evidence pickup is best-effort
            evidence_types = None
        if isinstance(evidence_types, Mapping) and evidence_types:
            row["evidence_types"] = dict(evidence_types)
    if trace_sink is not None and trace_entries:
        # POINTERS ONLY (fableanalysis §9.1 item 5): rows/bundles must never
        # carry decoded bodies — the protected-material scanner rejects records
        # containing content keys. The entries themselves go to the sink.
        tagged_entries = [
            {**entry, "runner_role": "base" if index < base_trace_entry_count else "candidate"}
            for index, entry in enumerate(trace_entries)
        ]
        row.update(
            await _finalize_incontainer_trace(
                trace_sink=trace_sink,
                icp_ref=row["icp_ref"] or row["icp_hash"],
                entries=tagged_entries,
            )
        )
        cost_summary = row.get("provider_cost_summary")
        if isinstance(cost_summary, Mapping) and (
            cost_summary.get("cap_blocked") or int(cost_summary.get("tracking_failed_count") or 0) > 0
        ):
            cap_blocked = bool(cost_summary.get("cap_blocked"))
            tracking_failed = int(cost_summary.get("tracking_failed_count") or 0) > 0
            has_scoreable_output = bool(row.get("base_company_scores")) or bool(row.get("candidate_company_scores"))
            if tracking_failed or (cap_blocked and not has_scoreable_output):
                row["base_company_scores"] = []
                row["candidate_company_scores"] = []
            row["hard_failure"] = False
            row["provider_cost_cap_blocked"] = cap_blocked
            row["provider_cost_tracking_failed"] = tracking_failed
            tokens = [token for token in _failure_reason_tokens(row.get("failure_reason"))]
            if cap_blocked and "provider_cost_cap_blocked" not in tokens:
                tokens.append("provider_cost_cap_blocked")
            if tracking_failed and "provider_cost_tracking_failed" not in tokens:
                tokens.append("provider_cost_tracking_failed")
            row["failure_reason"] = ";".join(tokens)
    return row, markers


async def _run_candidate_with_retries(
    *,
    candidate_runner: ModelRunner,
    icp: Mapping[str, Any],
    candidate_context: Mapping[str, Any],
    item_label: str,
    legacy_timeout_latch: bool,
    provider_flake_retry: bool,
    item_hash: str = "",
    item_ordinal: int = 0,
    scoring_telemetry_hook: ScoringTelemetryHook | None = None,
    attempt_cost_sink: AttemptCostSink | None = None,
) -> tuple[list[Mapping[str, Any]], str, bool]:
    """Run the candidate for one ICP with baseline-matched retry rounds.

    Returns ``(outputs, failure_reason, provider_excluded)``. The third value is
    retained for legacy row shape compatibility, but newly scored rows always
    return ``False`` so retry-exhausted provider failures contribute a zero ICP.
    """
    attempts = 0
    max_attempts = 1 + _candidate_provider_retry_rounds()
    while True:
        attempts += 1
        retry_round = attempts - 1
        attempt_context = await _emit_scoring_telemetry(
            scoring_telemetry_hook,
            "attempt_started",
            {
                "icp_ref": item_label,
                "icp_hash": item_hash,
                "icp_ordinal": item_ordinal,
                "model_role": "candidate",
                "retry_round": retry_round,
            },
        )
        attempt_entries: list[dict[str, Any]] = []
        attempt_trace_token = None
        caught_error: PrivateModelRuntimeError | None = None
        failure_reason = ""
        will_retry = False
        outputs: Sequence[Mapping[str, Any]] = ()
        if attempt_cost_sink is not None and incontainer_trace_capture_enabled():
            attempt_entries, attempt_trace_token = begin_incontainer_trace_collection()
        try:
            outputs = ensure_private_model_outputs(
                await _call_model_runner(candidate_runner, icp, candidate_context),
                context_label=f"candidate model for ICP {item_label}",
                require_non_empty=False,
            )
        except PrivateModelRuntimeError as exc:
            caught_error = exc
            failure_reason = _scoreable_candidate_runtime_failure_reason(exc)
            if not failure_reason:
                raise
            will_retry = attempts < max_attempts and _candidate_failure_should_retry(
                failure_reason,
                str(exc),
                legacy_timeout_latch=legacy_timeout_latch,
                provider_flake_retry=provider_flake_retry,
            )
        finally:
            if attempt_trace_token is not None:
                end_incontainer_trace_collection(attempt_trace_token)
                await _emit_attempt_costs(
                    attempt_cost_sink,
                    item_label,
                    attempt_entries,
                    attempt_context,
                )
                # Restore the exact legacy outer trace contents/order. IDs are
                # carried only out-of-band to the cost sink, never serialized.
                publish_incontainer_trace_entries(attempt_entries)
        if caught_error is not None:
            if will_retry:
                await _emit_scoring_telemetry(
                    scoring_telemetry_hook,
                    "attempt_failed",
                    {
                        "icp_ref": item_label,
                        "icp_hash": item_hash,
                        "icp_ordinal": item_ordinal,
                        "model_role": "candidate",
                        "retry_round": retry_round,
                        "retryable": True,
                        "failure_category": failure_reason,
                        "error": str(caught_error),
                    },
                )
                backoff_seconds = _candidate_429_retry_backoff_seconds(str(caught_error))
                if backoff_seconds > 0:
                    logger.warning(
                        "research_lab_candidate_rate_limit_retry_backoff item_ref=%s attempt=%s/%s backoff_seconds=%.1f",
                        item_label,
                        attempts,
                        max_attempts,
                        backoff_seconds,
                    )
                    await asyncio.sleep(backoff_seconds)
                continue
            # The caller turns retry-exhausted runtime failures into the
            # canonical scoreable-zero row and checkpoints that result. Leave
            # this final attempt non-terminal until that callback runs.
            return [], failure_reason, False
        await _emit_scoring_telemetry(
            scoring_telemetry_hook,
            "sourcing_completed",
            {
                "icp_ref": item_label,
                "icp_hash": item_hash,
                "icp_ordinal": item_ordinal,
                "model_role": "candidate",
                "retry_round": retry_round,
                "sourced_company_count": len(outputs),
            },
        )
        return list(outputs), "", False


def _candidate_failure_should_retry(
    failure_reason: str,
    error_text: str,
    *,
    legacy_timeout_latch: bool,
    provider_flake_retry: bool,
) -> bool:
    if failure_reason == "candidate_model_runtime_timeout":
        return not legacy_timeout_latch
    if failure_reason == "candidate_model_runtime_provider_error":
        return provider_flake_retry and _candidate_error_is_retryable(error_text)
    return False


def _candidate_error_is_retryable(error_text: str) -> bool:
    """Local mirror of the gateway baseline retry classifier semantics.

    Reimplemented here (not imported) so research_lab.eval keeps no gateway
    dependency. Transient provider/infra failures — 408/429/5xx, timeouts,
    connection resets, OOM kills — are retryable; auth and request-validation
    errors are not. Scrapingdog's 400 "Something went wrong or profile not
    found" is empirically transient-or-data-shaped, so it remains retryable.
    Production evidence showed 410 Gone retries do not produce usable content,
    so 410 is terminal.
    """
    lowered = error_text.lower()
    status = _provider_error_status_code(lowered)
    if _provider_cost_cap_error_text(error_text):
        return False
    if status in (408, 429) or status >= 500:
        return True
    if status == 400 and "something went wrong" in lowered:
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
        # Infra pressure (OOM-kill, daemon wedge) clears when fewer containers
        # run at once on the retry.
        return True
    return False


def _candidate_429_retry_backoff_seconds(error_text: str) -> float:
    """Return a short backoff only for explicit provider HTTP 429 errors."""

    if _provider_error_status_code(error_text.lower()) == 429:
        return _PROVIDER_429_RETRY_BACKOFF_SECONDS
    return 0.0


def _provider_error_status_code(lowered_error_text: str) -> int:
    # Keep this in sync with gateway logging diagnostics so candidate and
    # baseline retry verdicts see the same provider status codes.
    for code in (400, 401, 402, 403, 404, 408, 409, 410, 429, 500, 502, 503, 504):
        if (
            f"http error {code}" in lowered_error_text
            or f"status={code}" in lowered_error_text
            or f'"status":{code}' in lowered_error_text
        ):
            return code
    return 0


def _resume_rows_by_ref(
    resume_results: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in resume_results or ():
        if not isinstance(row, Mapping):
            continue
        ref = str(row.get("icp_ref") or row.get("icp_hash") or "")
        if ref:
            rows[ref] = dict(row)
    return rows


def _benchmark_item_ref(item: Mapping[str, Any]) -> str:
    return str(item.get("icp_ref") or item.get("icp_hash") or "")


# trajectoryimprovements.md P12: the deterministic L0 checks that can be graded
# without a notary snapshot. Content-grounded checks (snippet overlap,
# description grounding, date-in-content, antibot wall) need scraped page
# content that does not exist at this boundary — they are reported as not-run
# via ``snapshot_available=false``, never as silent passes.
_L0_CONTENT_INDEPENDENT_CHECKS = (
    "domain_source_match",
    "freshness_window",
    "generic_intent_description",
    "prompt_injection",
    "url_structural_validity",
)
_L0_MAX_COMPANIES = 8
_L0_MAX_CLAIMS_PER_COMPANY = 6


def _l0_per_claim_findings(
    companies: Sequence[Mapping[str, Any]],
    icp: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """P12 dense per-claim reward: run ``run_l0_checks`` per submitted signal.

    Returns one bounded row per (company, claim) carrying check ids and
    pass/fail statuses only — no free text, no evidence content — so the rows
    can ride the score bundle into ``evidence_bundles.snapshots`` under the
    corpus pointer rules. Never raises: L0 grading must not affect scoring.
    """
    try:
        from leadpoet_verifier.l0 import run_l0_checks
    except Exception:  # pragma: no cover - packaging-dependent
        return []
    rows: list[dict[str, Any]] = []
    buyer_cap_days = (icp or {}).get("intent_max_age_days")
    for company_index, company in enumerate(list(companies or ())[:_L0_MAX_COMPANIES]):
        if not isinstance(company, Mapping):
            continue
        signals = company.get("intent_signals")
        if not isinstance(signals, Sequence) or isinstance(signals, (str, bytes)):
            continue
        company_website = str(
            company.get("company_website") or company.get("website") or ""
        )
        for claim_index, signal in enumerate(list(signals)[:_L0_MAX_CLAIMS_PER_COMPANY]):
            if not isinstance(signal, Mapping):
                continue
            try:
                signal_doc = dict(signal)
                signal_doc.setdefault("company_website", company_website)
                if buyer_cap_days is not None:
                    signal_doc.setdefault("buyer_cap_days", buyer_cap_days)
                result = run_l0_checks(signal_doc, {})
                failed = {
                    finding.check_id
                    for finding in result.findings
                    if finding.severity == "fail"
                }
            except Exception:  # noqa: BLE001 - grading must never fail scoring
                logger.warning(
                    "research_lab_l0_per_claim_grading_failed company_index=%s claim_index=%s",
                    company_index,
                    claim_index,
                    exc_info=True,
                )
                continue
            rows.append(
                {
                    "company_index": company_index,
                    "claim_index": claim_index,
                    "snapshot_available": False,
                    "checks": [
                        {
                            "check_id": check_id,
                            "status": "fail" if check_id in failed else "pass",
                        }
                        for check_id in _L0_CONTENT_INDEPENDENT_CHECKS
                    ],
                    "failed_check_count": sum(
                        1
                        for check_id in _L0_CONTENT_INDEPENDENT_CHECKS
                        if check_id in failed
                    ),
                }
            )
    return rows


def _resolve_trace_sink(trace_sink: TraceSink | None, run_context: Mapping[str, Any]) -> TraceSink | None:
    """Resolve the effective in-container trace sink for one evaluation.

    Returns ``None`` when capture is disabled — the operator kill switch
    overrides injected sinks too, so ``RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE=
    false`` turns the whole pipeline off in one place.
    """
    if not incontainer_trace_capture_enabled():
        return None
    if trace_sink is not None:
        return trace_sink
    return _build_default_trace_sink(run_context)


def _build_default_trace_sink(run_context: Mapping[str, Any]) -> TraceSink:
    prefix = str(os.getenv(INCONTAINER_TRACE_S3_PREFIX_ENV) or "").strip().rstrip("/")
    run_ref = _incontainer_trace_run_ref(run_context)
    if not prefix:
        state = {"logged": False, "dropped_entries": 0}

        def _count_and_drop_sink(icp_ref: str, entries: list[dict[str, Any]]) -> str:
            state["dropped_entries"] += len(entries)
            if not state["logged"]:
                state["logged"] = True
                logger.info(
                    "research_lab_incontainer_trace_dropped run_ref=%s reason=no_s3_prefix "
                    "(set %s to persist in-container traces; further drops this eval are silent)",
                    run_ref,
                    INCONTAINER_TRACE_S3_PREFIX_ENV,
                )
            return ""

        return _count_and_drop_sink

    kms_key_id = str(os.getenv(INCONTAINER_TRACE_KMS_KEY_ENV) or "").strip()
    if not kms_key_id:
        # P5/P13: prefix-on/key-off must never write de-sanitized prompt text
        # unencrypted — refuse the upload path and drop loudly instead.
        state = {"logged": False, "dropped_entries": 0}

        def _missing_kms_drop_sink(icp_ref: str, entries: list[dict[str, Any]]) -> str:
            state["dropped_entries"] += len(entries)
            if not state["logged"]:
                state["logged"] = True
                logger.error(
                    "research_lab_incontainer_trace_dropped run_ref=%s reason=missing_kms_key "
                    "(set %s so in-container traces are written SSE-KMS encrypted; "
                    "unencrypted uploads are refused)",
                    run_ref,
                    INCONTAINER_TRACE_KMS_KEY_ENV,
                )
            return ""

        return _missing_kms_drop_sink

    async def _s3_sink(icp_ref: str, entries: list[dict[str, Any]]) -> str:
        return await asyncio.to_thread(
            _upload_incontainer_trace,
            prefix,
            run_ref,
            str(icp_ref),
            list(entries),
            kms_key_id,
        )

    return _s3_sink


def _upload_incontainer_trace(
    prefix: str,
    run_ref: str,
    icp_ref: str,
    entries: list[dict[str, Any]],
    kms_key_id: str,
) -> str:
    try:
        import boto3
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
            _safe_trace_key_component(run_ref),
            f"{_safe_trace_key_component(icp_ref)}.json",
        )
        if part
    )
    if not kms_key_id:
        # Backstop for the sink-level refusal: never PUT without aws:kms.
        raise PrivateModelRuntimeError(
            "in-container trace upload requires a KMS key id (unencrypted uploads are refused)"
        )
    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_incontainer_trace",
        "run_ref": run_ref,
        "icp_ref": icp_ref,
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


def _safe_trace_key_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-.")
    return cleaned or "unscoped"


def _incontainer_trace_run_ref(run_context: Mapping[str, Any]) -> str:
    for key in ("run_id", "candidate_build_ref", "ticket_id"):
        value = str((run_context or {}).get(key) or "").strip()
        if value:
            return value
    return "unscoped"


async def _finalize_incontainer_trace(
    *,
    trace_sink: TraceSink,
    icp_ref: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Hand entries to the sink and build the row's pointer-only fields.

    Sink failures are logged and swallowed (capture must never fail a run).
    P5: a dropped capture must not look populated — when no ref was written the
    row zeroes ``incontainer_trace_call_count`` and carries
    ``incontainer_trace_dropped=true`` plus the dropped count; the sha256 is
    kept as the attestation that the entries existed.
    """
    ref = ""
    try:
        result = trace_sink(icp_ref, entries)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            ref = result
    except Exception:
        logger.warning(
            "research_lab_incontainer_trace_sink_failed icp_ref=%s entry_count=%s",
            icp_ref,
            len(entries),
            exc_info=True,
        )
        ref = ""
    fields = {
        "incontainer_trace_ref": ref,
        "incontainer_trace_sha256": sha256_json(entries),
        "incontainer_trace_call_count": len(entries) if ref else 0,
    }
    # P13: truncation is filterable from the index, not just inside the blob.
    truncated_count = sum(1 for entry in entries if entry.get("truncated"))
    if truncated_count:
        fields["incontainer_trace_truncated_count"] = truncated_count
    # Provider evidence cache visibility: hits replayed recorded baseline
    # responses; misses fell through to live provider calls. Any miss means
    # the run used fresh evidence, so downstream comparison policy can hold
    # it to the fresh-evidence bar instead of the same-evidence bar.
    cache_hit_count = sum(1 for entry in entries if str(entry.get("phase") or "") == "cache_hit")
    cache_miss_count = sum(1 for entry in entries if str(entry.get("phase") or "") == "cache_miss")
    uninstrumented_count = sum(
        1 for entry in entries if str(entry.get("phase") or "") == "uninstrumented_http"
    )
    if cache_hit_count or cache_miss_count or uninstrumented_count:
        fields["provider_evidence_cache_hits"] = cache_hit_count
        fields["provider_evidence_cache_misses"] = cache_miss_count
    if uninstrumented_count:
        # Provider traffic outside the instrumented path is fresh evidence
        # replay never saw; it disqualifies a same-evidence classification.
        fields["provider_evidence_uninstrumented_calls"] = uninstrumented_count
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


def build_holdout_gate_result(
    *,
    public_results: Sequence[Mapping[str, Any]],
    private_results: Sequence[Mapping[str, Any]],
    public_icp_count: int,
    private_icp_count: int,
    gate: Mapping[str, Any],
    conditional_results: Sequence[Mapping[str, Any]] = (),
    conditional_icp_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assemble the (results, gate_result) pair from scored public/private ICP
    results.

    Shared by the per-candidate gate path (``_score_with_private_holdout_gate``)
    and the global (candidate, icp) queue's assembly, so both produce a
    byte-identical bundle. Pure: it derives the public gate the same way the
    per-candidate path does and, when the gate passes, folds in the private
    results and the total/delta fields. ``private_results`` is empty when the
    gate did not pass.
    """
    if bool(gate.get("conditional_validation_required")):
        return _build_conditional_holdout_gate_result(
            public_results=public_results,
            private_results=private_results,
            conditional_results=conditional_results,
            public_icp_count=public_icp_count,
            private_icp_count=private_icp_count,
            conditional_icp_count=conditional_icp_count,
            gate=gate,
        )

    baseline_public_score = float(gate.get("baseline_public_score") or 0.0)
    baseline_aggregate_score = _optional_float(gate.get("baseline_aggregate_score"))
    baseline_private_score = _optional_float(gate.get("baseline_private_score"))
    candidate_public_score = _benchmark_style_score(public_results, "candidate_company_scores")
    passed_public_gate = candidate_public_score + 1e-9 >= baseline_public_score
    gate_result = {
        "schema_version": "1.0",
        "gate_type": "public_score_before_private_holdout",
        "decision": "private_holdout_approved" if passed_public_gate else "rejected_before_private_holdout",
        "baseline_benchmark_bundle_id": str(gate.get("baseline_benchmark_bundle_id") or ""),
        "baseline_benchmark_hash": str(gate.get("baseline_benchmark_hash") or ""),
        "baseline_aggregate_score": round(baseline_aggregate_score, 6) if baseline_aggregate_score is not None else None,
        "baseline_public_score": round(baseline_public_score, 6),
        "baseline_private_score": round(baseline_private_score, 6) if baseline_private_score is not None else None,
        "candidate_public_score": round(candidate_public_score, 6),
        "paired_base_public_score": None,
        "reference_evaluation_mode": "stored_daily_baseline",
        "public_icp_count": public_icp_count,
        "private_holdout_icp_count": private_icp_count,
        "private_holdout_evaluated": bool(passed_public_gate),
        "provider_excluded_icp_ids": _provider_excluded_icp_ids(public_results),
    }
    if not passed_public_gate:
        return list(public_results), gate_result
    all_results = [*public_results, *private_results]
    candidate_total_score = _benchmark_style_score(all_results, "candidate_company_scores")
    daily_delta = (
        candidate_total_score - baseline_aggregate_score
        if baseline_aggregate_score is not None
        else None
    )
    gate_result = {
        **gate_result,
        "candidate_total_score": round(candidate_total_score, 6),
        "paired_base_total_score": None,
        "candidate_delta_vs_daily_baseline": round(daily_delta, 6) if daily_delta is not None else None,
        "provider_excluded_icp_ids": _provider_excluded_icp_ids(all_results),
    }
    return all_results, gate_result


def _build_conditional_holdout_gate_result(
    *,
    public_results: Sequence[Mapping[str, Any]],
    private_results: Sequence[Mapping[str, Any]],
    conditional_results: Sequence[Mapping[str, Any]],
    public_icp_count: int,
    private_icp_count: int,
    conditional_icp_count: int,
    gate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_public_score = float(gate.get("baseline_public_score") or 0.0)
    baseline_private_score = float(gate.get("baseline_private_score") or 0.0)
    baseline_conditional_score = float(gate.get("baseline_conditional_score") or 0.0)
    baseline_preliminary_score = float(gate.get("baseline_preliminary_score") or 0.0)
    baseline_aggregate_score = float(gate.get("baseline_aggregate_score") or 0.0)
    threshold_points = float(gate.get("threshold_points") or 0.0)
    candidate_public_score = _benchmark_style_score(
        public_results,
        "candidate_company_scores",
    )
    passed_public_gate = candidate_public_score + 1e-9 >= baseline_public_score
    base_result = {
        "schema_version": "1.1",
        "gate_type": "public_private_then_conditional_validation",
        "decision": (
            "conditional_validation_required"
            if passed_public_gate
            else "rejected_before_private_holdout"
        ),
        "baseline_benchmark_bundle_id": str(gate.get("baseline_benchmark_bundle_id") or ""),
        "baseline_benchmark_hash": str(gate.get("baseline_benchmark_hash") or ""),
        "baseline_aggregate_score": round(baseline_aggregate_score, 6),
        "baseline_preliminary_score": round(baseline_preliminary_score, 6),
        "baseline_public_score": round(baseline_public_score, 6),
        "baseline_private_score": round(baseline_private_score, 6),
        "baseline_conditional_score": round(baseline_conditional_score, 6),
        "candidate_public_score": round(candidate_public_score, 6),
        "paired_base_public_score": None,
        "reference_evaluation_mode": "stored_daily_baseline",
        "threshold_points": round(threshold_points, 6),
        "public_icp_count": int(public_icp_count),
        "private_holdout_icp_count": int(private_icp_count),
        "conditional_holdout_icp_count": int(conditional_icp_count),
        "private_holdout_evaluated": bool(passed_public_gate),
        "conditional_holdout_evaluated": False,
        "conditional_validation_required": True,
        "category_assignment_hash": str(gate.get("category_assignment_hash") or ""),
        "conditional_validation_policy_hash": str(
            gate.get("conditional_validation_policy_hash") or ""
        ),
        "provider_excluded_icp_ids": _provider_excluded_icp_ids(public_results),
    }
    if not passed_public_gate:
        return list(public_results), base_result

    preliminary_results = [*public_results, *private_results]
    candidate_private_score = _benchmark_style_score(
        private_results,
        "candidate_company_scores",
    )
    candidate_preliminary_score = _benchmark_style_score(
        preliminary_results,
        "candidate_company_scores",
    )
    preliminary_delta = candidate_preliminary_score - baseline_preliminary_score
    preliminary_passed = preliminary_delta + 1e-9 >= threshold_points
    preliminary_decision = (
        "conditional_validation_required"
        if preliminary_passed
        else "rejected_before_conditional_validation"
    )
    preliminary_doc = {
        **base_result,
        "decision": preliminary_decision,
        "candidate_private_score": round(candidate_private_score, 6),
        "candidate_preliminary_score": round(candidate_preliminary_score, 6),
        "candidate_preliminary_delta": round(preliminary_delta, 6),
        "preliminary_decision": preliminary_decision,
        "provider_excluded_icp_ids": _provider_excluded_icp_ids(preliminary_results),
    }
    preliminary_promotion_gate = gate.get("preliminary_promotion_gate")
    if isinstance(preliminary_promotion_gate, Mapping):
        preliminary_doc["preliminary_promotion_gate"] = dict(
            preliminary_promotion_gate
        )
    if not preliminary_passed or not conditional_results:
        return preliminary_results, preliminary_doc

    all_results = [*preliminary_results, *conditional_results]
    candidate_conditional_score = _benchmark_style_score(
        conditional_results,
        "candidate_company_scores",
    )
    candidate_total_score = _benchmark_style_score(
        all_results,
        "candidate_company_scores",
    )
    final_delta = candidate_total_score - baseline_aggregate_score
    final_passed = final_delta + 1e-9 >= threshold_points
    final_decision = (
        "conditional_validation_approved"
        if final_passed
        else "rejected_after_conditional_validation"
    )
    return all_results, {
        **preliminary_doc,
        "decision": final_decision,
        "candidate_conditional_score": round(candidate_conditional_score, 6),
        "candidate_total_score": round(candidate_total_score, 6),
        "paired_base_total_score": None,
        "candidate_delta_vs_daily_baseline": round(final_delta, 6),
        "conditional_holdout_evaluated": True,
        "final_decision": final_decision,
        "provider_excluded_icp_ids": _provider_excluded_icp_ids(all_results),
    }


async def _score_with_private_holdout_gate(
    *,
    benchmark_items: Sequence[Mapping[str, Any]],
    base_runner: ModelRunner | None,
    candidate_runner: ModelRunner,
    scorer: CompanyScorer,
    run_context: Mapping[str, Any],
    image_candidate: bool,
    runtime_patch: CandidatePatchManifest | None,
    gate: Mapping[str, Any],
    parent_freshness_check: ParentFreshnessCheck | None = None,
    icp_checkpoint: IcpCheckpoint | None = None,
    resume_results: Sequence[Mapping[str, Any]] | None = None,
    trace_sink: TraceSink | None = None,
    scoring_telemetry_hook: ScoringTelemetryHook | None = None,
    attempt_cost_sink: AttemptCostSink | None = None,
    holdout_transition_hook: HoldoutTransitionHook | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public_refs = {
        str(item)
        for item in gate.get("public_icp_refs", ())
        if str(item).strip()
    }
    if not public_refs:
        raise RealEvaluatorRequired("private holdout gate requires public ICP refs")
    public_items = [
        item for item in benchmark_items
        if str(item.get("icp_ref") or item.get("icp_hash") or "") in public_refs
    ]
    conditional_required = bool(gate.get("conditional_validation_required"))
    conditional_refs = {
        str(item)
        for item in gate.get("conditional_icp_refs", ())
        if str(item).strip()
    }
    private_refs = {
        str(item)
        for item in gate.get("private_icp_refs", ())
        if str(item).strip()
    }
    if conditional_required:
        if not private_refs or not conditional_refs:
            raise RealEvaluatorRequired(
                "conditional holdout gate requires private and conditional ICP refs"
            )
        private_items = [
            item for item in benchmark_items
            if str(item.get("icp_ref") or item.get("icp_hash") or "") in private_refs
        ]
        conditional_items = [
            item for item in benchmark_items
            if str(item.get("icp_ref") or item.get("icp_hash") or "") in conditional_refs
        ]
        assigned_refs = public_refs | private_refs | conditional_refs
        item_refs = {
            str(item.get("icp_ref") or item.get("icp_hash") or "")
            for item in benchmark_items
        }
        if assigned_refs != item_refs:
            raise RealEvaluatorRequired(
                "conditional holdout gate does not exactly cover the benchmark bank"
            )
        if public_refs & private_refs or public_refs & conditional_refs or private_refs & conditional_refs:
            raise RealEvaluatorRequired("conditional holdout gate ICP categories overlap")
    else:
        private_items = [
            item for item in benchmark_items
            if str(item.get("icp_ref") or item.get("icp_hash") or "") not in public_refs
        ]
        conditional_items = []
    if not public_items:
        raise RealEvaluatorRequired("private holdout gate matched zero public ICPs")
    if not private_items:
        raise RealEvaluatorRequired("private holdout gate leaves no private ICPs")
    if conditional_required and not conditional_items:
        raise RealEvaluatorRequired("conditional holdout gate leaves no conditional ICPs")

    public_results = await score_private_model_pair_items(
        benchmark_items=public_items,
        base_runner=base_runner,
        candidate_runner=candidate_runner,
        company_scorer=scorer,
        run_context=run_context,
        image_candidate=image_candidate,
        runtime_patch=runtime_patch,
        parent_freshness_check=parent_freshness_check,
        icp_checkpoint=icp_checkpoint,
        resume_results=resume_results,
        trace_sink=trace_sink,
        scoring_telemetry_hook=scoring_telemetry_hook,
        attempt_cost_sink=attempt_cost_sink,
    )
    # Public gate decides whether the private holdout runs at all; the shared
    # assembler below reproduces the same decision when building the result.
    baseline_public_score = float(gate.get("baseline_public_score") or 0.0)
    candidate_public_score = _benchmark_style_score(public_results, "candidate_company_scores")
    passed_public_gate = candidate_public_score + 1e-9 >= baseline_public_score
    if not passed_public_gate:
        for item in [*private_items, *conditional_items]:
            await _emit_scoring_telemetry(
                scoring_telemetry_hook,
                "gate_skipped",
                {
                    "icp_ref": str(item.get("icp_ref") or item.get("icp_hash") or ""),
                    "icp_hash": str(item.get("icp_hash") or ""),
                    "model_role": "candidate",
                    "failure_category": "public_gate_rejected",
                },
            )
        return build_holdout_gate_result(
            public_results=public_results,
            private_results=(),
            public_icp_count=len(public_items),
            private_icp_count=len(private_items),
            gate=gate,
            conditional_icp_count=len(conditional_items),
        )

    private_results = await score_private_model_pair_items(
        benchmark_items=private_items,
        base_runner=base_runner,
        candidate_runner=candidate_runner,
        company_scorer=scorer,
        run_context=run_context,
        image_candidate=image_candidate,
        runtime_patch=runtime_patch,
        parent_freshness_check=parent_freshness_check,
        icp_checkpoint=icp_checkpoint,
        resume_results=resume_results,
        trace_sink=trace_sink,
        scoring_telemetry_hook=scoring_telemetry_hook,
        attempt_cost_sink=attempt_cost_sink,
    )
    preliminary_results, preliminary_gate = build_holdout_gate_result(
        public_results=public_results,
        private_results=private_results,
        public_icp_count=len(public_items),
        private_icp_count=len(private_items),
        gate=gate,
        conditional_icp_count=len(conditional_items),
    )
    if not conditional_required:
        return preliminary_results, preliminary_gate
    if str(preliminary_gate.get("decision") or "") != "conditional_validation_required":
        for item in conditional_items:
            await _emit_scoring_telemetry(
                scoring_telemetry_hook,
                "gate_skipped",
                {
                    "icp_ref": str(item.get("icp_ref") or item.get("icp_hash") or ""),
                    "icp_hash": str(item.get("icp_hash") or ""),
                    "model_role": "candidate",
                    "failure_category": "preliminary_gate_rejected",
                },
            )
        return preliminary_results, preliminary_gate

    transition_result = await _emit_holdout_transition(
        holdout_transition_hook,
        "conditional_validation_started",
        {
            **preliminary_gate,
            # Private in-process input for the gateway authority callback.  The
            # worker whitelists persisted fields and never serializes this key.
            "_preliminary_results": [dict(item) for item in preliminary_results],
        },
    )
    if transition_result is None:
        raise ConditionalValidationRetryableError(
            "conditional_preliminary_authority_unavailable"
        )
    proof = transition_result.get("preliminary_promotion_gate")
    if not isinstance(proof, Mapping):
        raise ConditionalValidationRetryableError(
            "conditional_preliminary_authority_proof_missing"
        )
    gate = {**dict(gate), "preliminary_promotion_gate": dict(proof)}

    conditional_results = await score_private_model_pair_items(
        benchmark_items=conditional_items,
        base_runner=base_runner,
        candidate_runner=candidate_runner,
        company_scorer=scorer,
        run_context=run_context,
        image_candidate=image_candidate,
        runtime_patch=runtime_patch,
        parent_freshness_check=parent_freshness_check,
        icp_checkpoint=icp_checkpoint,
        resume_results=resume_results,
        trace_sink=trace_sink,
        scoring_telemetry_hook=scoring_telemetry_hook,
        attempt_cost_sink=attempt_cost_sink,
    )
    if len(conditional_results) != len(conditional_items):
        raise ConditionalValidationRetryableError(
            "conditional_validation_incomplete:result_count_mismatch"
        )
    expected_conditional_refs = {
        str(item.get("icp_ref") or item.get("icp_hash") or "")
        for item in conditional_items
    }
    actual_conditional_refs = {
        str(item.get("icp_ref") or item.get("icp_hash") or "")
        for item in conditional_results
    }
    if actual_conditional_refs != expected_conditional_refs:
        raise ConditionalValidationRetryableError(
            "conditional_validation_incomplete:result_identity_mismatch"
        )
    conditional_failures = _critical_measurement_failures(conditional_results)
    if conditional_failures:
        raise ConditionalValidationRetryableError(
            "conditional_validation_incomplete:" + ",".join(conditional_failures)
        )
    return build_holdout_gate_result(
        public_results=public_results,
        private_results=private_results,
        conditional_results=conditional_results,
        public_icp_count=len(public_items),
        private_icp_count=len(private_items),
        conditional_icp_count=len(conditional_items),
        gate=gate,
    )


def _critical_measurement_failures(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures: set[str] = set()
    for row in rows:
        reasons = _failure_reason_tokens(row.get("failure_reason"))
        if row.get("provider_excluded"):
            failures.add("provider_excluded")
        if row.get("provider_cost_cap_blocked"):
            failures.add("provider_cost_cap_blocked")
        if row.get("provider_cost_tracking_failed"):
            failures.add("provider_cost_tracking_failed")
        for reason in reasons:
            if (
                reason.startswith("reference_model_runtime_")
                or "provider_error" in reason
                or "timeout" in reason
            ):
                failures.add(reason)
    return sorted(failures)


async def _run_parent_freshness_check(
    callback: ParentFreshnessCheck | None,
    progress: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    result = callback(progress)
    if inspect.isawaitable(result):
        await result


def _global_scoring_queue_enabled() -> bool:
    """Opt-in cross-candidate global job queue (default off = per-candidate path).

    The per-candidate scheduler keeps ``concurrency`` ICPs in flight for a single
    candidate, so slots idle once a candidate's remaining ICPs drop below the
    concurrency (the tail gap) and no other candidate fills them. The global
    queue holds every candidate's ICP jobs in one fixed-size pool that pulls the
    next job — from any candidate — the moment a slot frees, so the pool stays
    saturated at ``pool_size`` until the whole set has fewer jobs than the pool.
    """
    raw = str(os.getenv("RESEARCH_LAB_GLOBAL_SCORING_QUEUE") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# Job priority: private ICPs (approved after the public gate) jump ahead of
# other candidates' public ICPs still waiting.
_GQ_PRIORITY_PRIVATE = 0
_GQ_PRIORITY_PUBLIC = 1
_GQ_PRIORITY_STOP = 2


async def score_candidates_global_queue(
    *,
    candidate_specs: Sequence[Mapping[str, Any]],
    pool_size: int,
    legacy_timeout_latch: bool = False,
    provider_flake_retry: bool = True,
    trace_sink: TraceSink | None = None,
) -> list[dict[str, Any]]:
    """Score many candidates through one fixed-size job pool.

    Each ``candidate_spec`` carries:
      candidate_id, candidate_runner, base_runner, scorer, run_context,
      image_candidate, runtime_patch, public_items, private_items,
      baseline_public_score.

    Public ICPs are enqueued first for every candidate. When a candidate's public
    set finishes, its gate is decided: if the candidate's public score meets the
    baseline, its private ICPs are pushed to the FRONT of the queue; otherwise the
    candidate stops and its private ICPs never run. Exactly ``pool_size`` jobs run
    at once regardless of which candidate they belong to.

    Returns one entry per candidate: candidate_id, per_icp_results (public then
    private, in item order), gate_result, and pool metadata.
    """
    n = len(candidate_specs)
    pool = max(1, int(pool_size))
    scorers = [spec.get("scorer") or QualificationStyleCompanyScorer() for spec in candidate_specs]

    public_rows: list[dict[int, dict[str, Any]]] = [dict() for _ in range(n)]
    private_rows: list[dict[int, dict[str, Any]]] = [dict() for _ in range(n)]
    public_remaining = [len(spec.get("public_items") or ()) for spec in candidate_specs]
    private_remaining = [0 for _ in range(n)]
    skip_reason = ["" for _ in range(n)]
    consecutive_timeouts = [0 for _ in range(n)]
    gate_passed = [False for _ in range(n)]
    gate_decided = [False for _ in range(n)]

    lock = asyncio.Lock()
    queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
    seq = 0
    outstanding = 0

    def _enqueue(priority: int, ci: int, phase: str, item: Mapping[str, Any], item_index: int) -> None:
        nonlocal seq, outstanding
        queue.put_nowait((priority, seq, ci, phase, item, item_index))
        seq += 1
        outstanding += 1

    for ci, spec in enumerate(candidate_specs):
        for item_index, item in enumerate(spec.get("public_items") or ()):
            _enqueue(_GQ_PRIORITY_PUBLIC, ci, "public", item, item_index)
        # A candidate with no public ICPs cannot be gated; treat as decided/failed
        # so it never contributes private jobs (defensive — real runs always split).
        if public_remaining[ci] == 0:
            gate_decided[ci] = True

    async def _decide_gate_locked(ci: int) -> None:
        spec = candidate_specs[ci]
        rows = [public_rows[ci][k] for k in sorted(public_rows[ci])]
        candidate_public_score = _benchmark_style_score(rows, "candidate_company_scores")
        baseline_public_score = _optional_float(spec.get("baseline_public_score")) or 0.0
        passed = candidate_public_score + 1e-9 >= baseline_public_score
        gate_passed[ci] = passed
        gate_decided[ci] = True
        if passed:
            private_items = spec.get("private_items") or ()
            private_remaining[ci] = len(private_items)
            for item_index, item in enumerate(private_items):
                _enqueue(_GQ_PRIORITY_PRIVATE, ci, "private", item, item_index)

    async def _worker() -> None:
        nonlocal outstanding
        while True:
            priority, _s, ci, phase, item, item_index = await queue.get()
            if priority == _GQ_PRIORITY_STOP:
                queue.task_done()
                return
            spec = candidate_specs[ci]
            async with lock:
                current_skip = skip_reason[ci]
                completed = len(public_rows[ci]) + len(private_rows[ci])
            row, markers = await _score_single_icp(
                item=item,
                item_index=item_index,
                completed_icp_count=completed,
                base_runner=spec.get("base_runner"),
                candidate_runner=spec["candidate_runner"],
                scorer=scorers[ci],
                run_context=spec.get("run_context") or {},
                image_candidate=bool(spec.get("image_candidate")),
                runtime_patch=spec.get("runtime_patch"),
                parent_freshness_check=None,
                candidate_runtime_skip_reason=current_skip,
                legacy_timeout_latch=legacy_timeout_latch,
                provider_flake_retry=provider_flake_retry,
                trace_sink=trace_sink,
            )
            async with lock:
                if phase == "public":
                    public_rows[ci][item_index] = row
                    public_remaining[ci] -= 1
                else:
                    private_rows[ci][item_index] = row
                    private_remaining[ci] -= 1
                if markers["timed_out"]:
                    consecutive_timeouts[ci] += 1
                elif not markers["skipped"]:
                    consecutive_timeouts[ci] = 0
                if markers["latch_reason"] and not skip_reason[ci]:
                    skip_reason[ci] = markers["latch_reason"]
                if (
                    not legacy_timeout_latch
                    and not skip_reason[ci]
                    and consecutive_timeouts[ci] >= 2
                ):
                    skip_reason[ci] = _candidate_runtime_skip_reason("candidate_model_runtime_timeout")
                if phase == "public" and public_remaining[ci] == 0 and not gate_decided[ci]:
                    await _decide_gate_locked(ci)
                outstanding -= 1
                if outstanding == 0:
                    for _ in range(pool):
                        queue.put_nowait((_GQ_PRIORITY_STOP, seq, -1, "", {}, -1))
            queue.task_done()

    if outstanding > 0:
        await asyncio.gather(*(_worker() for _ in range(pool)))

    results: list[dict[str, Any]] = []
    for ci, spec in enumerate(candidate_specs):
        ordered = [public_rows[ci][k] for k in sorted(public_rows[ci])]
        ordered.extend(private_rows[ci][k] for k in sorted(private_rows[ci]))
        results.append(
            {
                "candidate_id": str(spec.get("candidate_id") or ""),
                "per_icp_results": ordered,
                "gate_result": {
                    "decision": "private_holdout_approved" if gate_passed[ci] else "rejected_before_private_holdout",
                    "private_holdout_evaluated": bool(gate_passed[ci]),
                },
            }
        )
    return results


def _benchmark_style_score(
    per_icp_results: Sequence[Mapping[str, Any]],
    score_field: str,
) -> float:
    # Each ICP contributes the mean of its company scores; an ICP that produced no
    # scoreable company counts as 0. That is intentional: if every company the model
    # returned for an ICP failed (bad/false URLs included), the model did not source
    # that ICP and should score 0 for it. Per-company resilience lives upstream in the
    # model (one failed company is skipped, not fatal), so this 0 only happens when the
    # whole ICP yielded nothing.
    #
    # One deliberate deviation:
    # - With RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE on (bug #8), the per-ICP score
    #   is the verifier's capped sum(top-N company scores)/N — N being the
    #   ICP's requested company count — instead of the unweighted mean,
    #   closing the truncate-to-best-company exploit and paying for quantity.
    capped_top5 = _capped_top5_score_enabled()
    per_icp_scores: list[float] = []
    for row in per_icp_results:
        scores = row.get(score_field)
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes, bytearray)):
            per_icp_scores.append(0.0)
            continue
        values = [float(item or 0.0) for item in scores]
        per_icp_scores.append(
            _benchmark_icp_score(
                values,
                capped_top5=capped_top5,
                requested_count=row.get("icp_company_goal"),
                fp_penalty_total=_row_fp_penalty_total(row, score_field),
            )
        )
    return float(sum(per_icp_scores) / len(per_icp_scores)) if per_icp_scores else 0.0


def _row_fp_penalty_total(row: Mapping[str, Any], score_field: str) -> float:
    """Penalty points for one per-ICP row under the current env knobs.

    Rows stamp per-side FP counts (``fp_gate_count``/``candidate_fp_gate_count``
    etc. depending on which score_field is being aggregated); missing keys
    mean zero so legacy rows are untouched.
    """
    side = "base" if score_field.startswith("base") else "candidate"
    gate = _coerce_count(row.get(f"{side}_fp_gate_count"))
    primary = _coerce_count(row.get(f"{side}_fp_unverified_primary_count"))
    return gate * _fp_penalty_points() + primary * _fp_unverified_primary_penalty_points()


def _coerce_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def benchmark_icp_score_from_company_scores(
    scores: Sequence[float],
    requested_count: int | None = None,
    fp_penalty_total: float = 0.0,
) -> float:
    """Per-ICP score on the live-gate scale for one ICP's company scores.

    ``requested_count`` is the ICP's requested company count (max_companies);
    omitted/None keeps the fixed 5-lead budget so legacy callers and ICP sets
    are unchanged. ``fp_penalty_total`` is the pre-multiplied penalty-point
    sum for the ICP's false positives (0 keeps historical scores exactly).
    Shared entry point so the daily-baseline path and the candidate path flip
    together when RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE changes; never mix
    scores computed under different settings of these knobs.
    """
    values = [float(item or 0.0) for item in scores]
    return _benchmark_icp_score(
        values,
        capped_top5=_capped_top5_score_enabled(),
        requested_count=requested_count,
        fp_penalty_total=fp_penalty_total,
    )


def _requested_count_or_default(requested_count: Any) -> int:
    try:
        count = int(requested_count)
    except (TypeError, ValueError):
        return _TOP5_LEADS_PER_ICP
    return max(1, min(count, _MAX_ICP_COMPANY_GOAL))


# A single catastrophic ICP must hurt, but not dominate the whole benchmark
# mean by itself: per-ICP scores never go below this floor.
_FP_PENALTY_ICP_FLOOR = -100.0


def _benchmark_icp_score(
    values: list[float],
    *,
    capped_top5: bool,
    requested_count: Any = None,
    fp_penalty_total: float = 0.0,
) -> float:
    if capped_top5:
        # Match the verifier's advisory arithmetic exactly
        # (leadpoet_verifier.aggregation.per_icp_normalized_score): each company
        # score is clamped to [0, MAX_COMPANY_TOTAL_SCORE], summed over the top
        # N companies, and divided by the N-lead budget the ICP requested.
        # False-positive penalty points subtract from the sum BEFORE the
        # normalization, so junk companies drag the ICP score — negative is
        # allowed (floored) and pulls the whole benchmark mean down.
        count = _requested_count_or_default(requested_count)
        top_values = sorted(values, reverse=True)[:count]
        normalized = float(per_icp_normalized_score(top_values, max_leads=count))
        penalty = max(0.0, float(fp_penalty_total)) / count
        return max(_FP_PENALTY_ICP_FLOOR, normalized - penalty)
    return float(sum(values) / len(values)) if values else 0.0


def _provider_excluded_icp_ids(per_icp_results: Sequence[Mapping[str, Any]]) -> list[str]:
    """Legacy identifiers of ICPs excluded for unresolved provider errors.

    New scoring no longer sets ``provider_excluded``; this remains so older
    score bundles and resumed progress rows can still be inspected.
    """
    ids = {
        str(row.get("icp_ref") or row.get("icp_hash") or "")
        for row in per_icp_results
        if row.get("provider_excluded")
    }
    return sorted(ref for ref in ids if ref)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _artifact_version_doc(artifact: PrivateModelArtifactManifest) -> dict[str, Any]:
    return {
        "model_artifact_hash": artifact.model_artifact_hash,
        "manifest_hash": artifact.manifest_hash,
        "manifest_uri": artifact.manifest_uri,
        "git_commit_sha": artifact.git_commit_sha,
        "image_digest": artifact.image_digest,
        "config_hash": artifact.config_hash,
        "component_registry_version": artifact.component_registry_version,
        "scoring_adapter_version": artifact.scoring_adapter_version,
        "build_id": artifact.build_id,
    }


def _public_artifact_version_doc(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Hash private runtime pointers while retaining join-safe model identity."""

    return {
        "model_artifact_hash": str(artifact.get("model_artifact_hash") or ""),
        "manifest_hash": str(artifact.get("manifest_hash") or ""),
        "manifest_ref_hash": sha256_json({"manifest_ref": artifact.get("manifest_uri")}),
        "git_commit_sha": str(artifact.get("git_commit_sha") or ""),
        "image_ref_hash": sha256_json({"image_ref": artifact.get("image_digest")}),
        "config_hash": str(artifact.get("config_hash") or ""),
        "component_registry_version": str(artifact.get("component_registry_version") or ""),
        "scoring_adapter_version": str(artifact.get("scoring_adapter_version") or ""),
        "build_id": str(artifact.get("build_id") or ""),
    }


def public_serving_model_version_doc(serving_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validator-facing serving stamp with no raw ECR/S3 refs."""

    private_pointer_fields = {"manifest_uri", "image_digest"}
    public_doc = {
        str(key): value
        for key, value in serving_doc.items()
        if key not in {"parent_model", "candidate_model", "public_stamp_hash"}
        and key not in private_pointer_fields
    }
    parent = serving_doc.get("parent_model")
    if isinstance(parent, Mapping):
        public_doc["parent_model"] = _public_artifact_version_doc(parent)
    candidate = serving_doc.get("candidate_model")
    if isinstance(candidate, Mapping):
        public_doc["candidate_model"] = _public_artifact_version_doc(candidate)
    public_doc["public_stamp_hash"] = sha256_json(public_doc)
    return public_doc


def _serving_model_version_doc(
    *,
    artifact: PrivateModelArtifactManifest,
    benchmark_set: SealedBenchmarkSet,
    run_context: Mapping[str, Any],
    candidate_artifact: PrivateModelArtifactManifest | None,
    candidate_patch_hash: str,
) -> dict[str, Any]:
    """Pointer-only version stamp for Research Lab results.

    This intentionally contains immutable refs/hashes only. It lets later
    analysis join score outcomes back to the exact artifact/image/benchmark
    without exposing private ICP text or provider payloads.
    """

    doc: dict[str, Any] = {
        "schema_version": "research_lab_serving_model_version.v1",
        "result_role": "candidate_scoring" if candidate_artifact is not None else "baseline_scoring",
        "run_id": str(run_context.get("run_id") or ""),
        "ticket_id": str(run_context.get("ticket_id") or ""),
        "candidate_id": str(run_context.get("candidate_id") or ""),
        "private_model_version_id": str(run_context.get("private_model_version_id") or ""),
        "evaluation_epoch": int(run_context.get("evaluation_epoch") or 0),
        "benchmark_date": str(run_context.get("benchmark_date") or ""),
        "benchmark_attempt": run_context.get("benchmark_attempt"),
        "run_scope": str(run_context.get("run_scope") or ""),
        "benchmark_id": benchmark_set.benchmark_id,
        "benchmark_split_ref": benchmark_set.split_ref,
        "icp_set_hash": benchmark_set.icp_set_hash,
        "scoring_code_version": benchmark_set.scoring_version,
        "evaluator_version": str(run_context.get("evaluator_version") or ""),
        "parent_model": _artifact_version_doc(artifact),
        "candidate_patch_hash": candidate_patch_hash,
        "candidate_source_diff_hash": str(run_context.get("candidate_source_diff_hash") or ""),
        "candidate_build_ref": str(run_context.get("candidate_build_ref") or ""),
    }
    if candidate_artifact is not None:
        doc["candidate_model"] = _artifact_version_doc(candidate_artifact)
    doc["version_stamp_hash"] = sha256_json({key: value for key, value in doc.items() if key != "version_stamp_hash"})
    return doc


def _enrich_per_icp_evaluation_context(
    per_icp_results: Sequence[Mapping[str, Any]],
    *,
    benchmark_set: SealedBenchmarkSet,
    run_context: Mapping[str, Any],
    serving_model_version_hash: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(per_icp_results):
        item = dict(row)
        existing = item.get("evaluation_context") if isinstance(item.get("evaluation_context"), Mapping) else {}
        icp_ref = str(item.get("icp_ref") or item.get("icp_hash") or "")
        icp_hash = str(item.get("icp_hash") or "")
        result_hash_payload = {key: value for key, value in item.items() if key != "evaluation_context"}
        context_doc = {
            **dict(existing),
            "schema_version": "research_lab_evaluation_context.v1",
            "result_index": int(index),
            "icp_ref": icp_ref,
            "icp_hash": icp_hash,
            "benchmark_id": benchmark_set.benchmark_id,
            "benchmark_split_ref": benchmark_set.split_ref,
            "icp_set_hash": benchmark_set.icp_set_hash,
            "input_window_hash": str(run_context.get("rolling_window_hash") or benchmark_set.icp_set_hash),
            "run_id": str(run_context.get("run_id") or ""),
            "ticket_id": str(run_context.get("ticket_id") or ""),
            "candidate_id": str(run_context.get("candidate_id") or ""),
            "evaluation_epoch": int(run_context.get("evaluation_epoch") or 0),
            "benchmark_date": str(run_context.get("benchmark_date") or ""),
            "benchmark_attempt": run_context.get("benchmark_attempt"),
            "run_scope": str(run_context.get("run_scope") or ""),
            "provider_cache_day": str(run_context.get("provider_cache_day") or run_context.get("cache_day") or ""),
            "serving_model_version_hash": serving_model_version_hash,
            "result_row_hash": sha256_json(result_hash_payload),
        }
        item["evaluation_context"] = context_doc
        enriched.append(item)
    return enriched


def build_score_bundle_from_scored_icps(
    *,
    artifact_manifest: PrivateModelArtifactManifest | Mapping[str, Any],
    benchmark: SealedBenchmarkSet | Mapping[str, Any],
    patch_manifest: CandidatePatchManifest | Mapping[str, Any],
    candidate_artifact_manifest: PrivateModelArtifactManifest | Mapping[str, Any] | None = None,
    per_icp_results: Sequence[Mapping[str, Any]],
    run_context: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
    extra_bundle_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = artifact_manifest if isinstance(artifact_manifest, PrivateModelArtifactManifest) else PrivateModelArtifactManifest.from_mapping(artifact_manifest)
    benchmark_set = benchmark if isinstance(benchmark, SealedBenchmarkSet) else SealedBenchmarkSet.from_mapping(benchmark)
    image_candidate = candidate_artifact_manifest is not None
    candidate_artifact = (
        candidate_artifact_manifest
        if isinstance(candidate_artifact_manifest, PrivateModelArtifactManifest)
        else (
            PrivateModelArtifactManifest.from_mapping(candidate_artifact_manifest)
            if candidate_artifact_manifest is not None
            else None
        )
    )
    patch = None if image_candidate else (
        patch_manifest if isinstance(patch_manifest, CandidatePatchManifest) else CandidatePatchManifest.from_mapping(patch_manifest)
    )
    errors = []
    errors.extend(validate_private_model_artifact_manifest(artifact))
    errors.extend(validate_sealed_benchmark_set(benchmark_set))
    if image_candidate:
        if candidate_artifact is None:
            errors.append("candidate_artifact_manifest_required_for_image_candidate")
        else:
            errors.extend(validate_private_model_artifact_manifest(candidate_artifact))
        patch_parent = str(dict(patch_manifest).get("parent_artifact_hash") or "")
        if patch_parent and patch_parent != artifact.model_artifact_hash:
            errors.append("parent_artifact_hash_mismatch")
    else:
        errors.extend(validate_candidate_patch_manifest(patch, expected_parent_artifact_hash=artifact.model_artifact_hash))
    if errors:
        raise ValueError("; ".join(errors))
    if not per_icp_results:
        raise RealEvaluatorRequired("real scored ICP results are required")
    candidate_patch_hash = (
        sha256_json(dict(patch_manifest))
        if image_candidate
        else patch.manifest_hash()
    )
    private_serving_model_version = _serving_model_version_doc(
        artifact=artifact,
        benchmark_set=benchmark_set,
        run_context=run_context,
        candidate_artifact=candidate_artifact,
        candidate_patch_hash=candidate_patch_hash,
    )
    serving_model_version = public_serving_model_version_doc(private_serving_model_version)
    enriched_per_icp_results = _enrich_per_icp_evaluation_context(
        per_icp_results,
        benchmark_set=benchmark_set,
        run_context=run_context,
        serving_model_version_hash=str(private_serving_model_version["version_stamp_hash"]),
    )
    extra_fields = dict(extra_bundle_fields or {})
    holdout_gate = (
        extra_fields.get("private_holdout_gate")
        if isinstance(extra_fields.get("private_holdout_gate"), Mapping)
        else {}
    )

    bundle = build_research_evaluation_score_bundle(
        run_id=str(run_context["run_id"]),
        ticket_id=str(run_context["ticket_id"]),
        miner_hotkey=str(run_context["miner_hotkey"]),
        island=str(run_context["island"]),
        evaluation_epoch=int(run_context["evaluation_epoch"]),
        parent_artifact_hash=artifact.model_artifact_hash,
        candidate_artifact_hash=(
            candidate_artifact.model_artifact_hash
            if candidate_artifact is not None
            else patch.candidate_artifact_hash
        ),
        private_model_manifest_hash=artifact.manifest_hash,
        candidate_patch_hash=(
            candidate_patch_hash
        ),
        icp_set_hash=benchmark_set.icp_set_hash,
        scoring_version=benchmark_set.scoring_version,
        evaluator_version=str(run_context["evaluator_version"]),
        per_icp_results=enriched_per_icp_results,
        evidence_bundle_refs=tuple(str(item) for item in run_context.get("evidence_bundle_refs", ())),
        execution_trace_ref=str(run_context["execution_trace_ref"]),
        cost_ledger_ref=str(run_context["cost_ledger_ref"]),
        benchmark_split_ref=benchmark_set.split_ref,
        candidate_model_manifest_hash=(
            candidate_artifact.manifest_hash
            if candidate_artifact is not None
            else None
        ),
        candidate_source_diff_hash=run_context.get("candidate_source_diff_hash") or None,
        candidate_build_ref=run_context.get("candidate_build_ref") or None,
        serving_model_version=serving_model_version,
        policy=policy or {},
        signature_ref=str(run_context.get("signature_ref") or ""),
        schema_version=(
            "1.1" if bool(holdout_gate.get("conditional_validation_required")) else "1.0"
        ),
    )
    scoring_health = build_scoring_health_doc(
        enriched_per_icp_results,
        private_holdout_gate=extra_fields.get("private_holdout_gate"),
    )
    enriched = {
        **bundle,
        **extra_fields,
        "scoring_health": scoring_health,
        # Legacy surface for old/resumed provider-exclusion rows. New scoring
        # should leave this empty and include retry-exhausted ICPs as zero.
        "provider_excluded_icp_ids": _provider_excluded_icp_ids(per_icp_results),
        "score_bundle_hash": "",
        "anchored_hash": "",
    }
    enriched_hash = score_bundle_hash(enriched)
    return {**enriched, "score_bundle_hash": enriched_hash, "anchored_hash": enriched_hash}


def build_scoring_health_doc(
    per_icp_results: Sequence[Mapping[str, Any]],
    *,
    private_holdout_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic scoring-health evidence from per-ICP outcomes."""

    total = len(per_icp_results)
    counts = {
        "reference_runtime_failure_count": 0,
        "candidate_runtime_failure_count": 0,
        "reference_zero_company_count": 0,
        "candidate_zero_company_count": 0,
        "provider_error_count": 0,
        "timeout_count": 0,
        "invalid_output_count": 0,
        "skipped_candidate_count": 0,
        "provider_excluded_icp_count": 0,
        "sourced_zero_no_error_count": 0,
        "provider_cost_cap_blocked_icp_count": 0,
        "provider_cost_tracking_failed_icp_count": 0,
    }
    failure_classes: dict[str, int] = {}
    for row in per_icp_results:
        reasons = _failure_reason_tokens(row.get("failure_reason"))
        active_failures = tuple(
            reason
            for reason in reasons
            if not reason.startswith("candidate_model_runtime_skipped_after_")
        )
        for reason in reasons:
            failure_classes[reason] = failure_classes.get(reason, 0) + 1
        if any(reason.startswith("reference_model_runtime_") for reason in reasons):
            counts["reference_runtime_failure_count"] += 1
        if any(
            reason.startswith("candidate_model_runtime_")
            and not reason.startswith("candidate_model_runtime_skipped_after_")
            for reason in reasons
        ):
            counts["candidate_runtime_failure_count"] += 1
        if "reference_model_zero_companies" in reasons:
            counts["reference_zero_company_count"] += 1
        if "candidate_model_zero_companies" in reasons:
            counts["candidate_zero_company_count"] += 1
        if any("provider_error" in reason for reason in reasons):
            counts["provider_error_count"] += 1
        if any("timeout" in reason for reason in active_failures):
            counts["timeout_count"] += 1
        if any(
            marker in reason
            for reason in active_failures
            for marker in ("invalid_json", "invalid_output", "adapter_failed")
        ):
            counts["invalid_output_count"] += 1
        if any(reason.startswith("candidate_model_runtime_skipped_after_") for reason in reasons):
            counts["skipped_candidate_count"] += 1
        if row.get("provider_excluded"):
            counts["provider_excluded_icp_count"] += 1
        if row.get("sourced_zero_no_error") or row.get("reference_sourced_zero_no_error"):
            counts["sourced_zero_no_error_count"] += 1
        if row.get("provider_cost_cap_blocked"):
            counts["provider_cost_cap_blocked_icp_count"] += 1
        if row.get("provider_cost_tracking_failed"):
            counts["provider_cost_tracking_failed_icp_count"] += 1

    rates = {
        "reference_runtime_success_rate": _health_success_rate(total, counts["reference_runtime_failure_count"]),
        "candidate_runtime_success_rate": _health_success_rate(total, counts["candidate_runtime_failure_count"]),
        "reference_zero_company_rate": _health_rate(total, counts["reference_zero_company_count"]),
        "candidate_zero_company_rate": _health_rate(total, counts["candidate_zero_company_count"]),
        "provider_error_rate": _health_rate(total, counts["provider_error_count"]),
        "timeout_rate": _health_rate(total, counts["timeout_count"]),
        "invalid_output_rate": _health_rate(total, counts["invalid_output_count"]),
        "skipped_candidate_rate": _health_rate(total, counts["skipped_candidate_count"]),
        "provider_excluded_icp_rate": _health_rate(total, counts["provider_excluded_icp_count"]),
        "sourced_zero_no_error_rate": _health_rate(total, counts["sourced_zero_no_error_count"]),
        "provider_cost_cap_blocked_icp_rate": _health_rate(total, counts["provider_cost_cap_blocked_icp_count"]),
        "provider_cost_tracking_failed_icp_rate": _health_rate(total, counts["provider_cost_tracking_failed_icp_count"]),
    }
    holdout_doc = _scoring_health_holdout_doc(private_holdout_gate)
    # Only failures that invalidate the measurement itself mark scoring
    # unhealthy: provider outages/credit exhaustion, timeouts, ICPs excluded
    # for unresolved provider errors, cost-accounting loss, and reference-side
    # runtime crashes. Zero-company results, request-shaped provider
    # rejections of model-generated URLs, and candidate-quality crashes are
    # legitimate scored outcomes and stay informational (still counted and
    # rated above, but they do not flip health_status).
    critical_count_keys = (
        "reference_runtime_failure_count",
        "provider_error_count",
        "timeout_count",
        "provider_excluded_icp_count",
        "provider_cost_cap_blocked_icp_count",
        "provider_cost_tracking_failed_icp_count",
    )
    degraded = any(counts[key] > 0 for key in critical_count_keys)
    return {
        "schema_version": "1.0",
        "health_status": "degraded" if degraded else "healthy",
        "icp_count": total,
        **counts,
        **rates,
        "failure_class_counts": dict(sorted(failure_classes.items())),
        "public_holdout_decision": holdout_doc.get("decision", "not_applicable"),
        "baseline_bundle_id": holdout_doc.get("baseline_benchmark_bundle_id", ""),
        "baseline_bundle_hash": holdout_doc.get("baseline_benchmark_hash", ""),
        "private_holdout_gate": holdout_doc,
    }


def _failure_reason_tokens(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(
        token.strip()
        for token in str(value).split(";")
        if token.strip()
    )


def _health_rate(total: int, count: int) -> float:
    return round(float(count) / float(total), 6) if total else 0.0


def _health_success_rate(total: int, failure_count: int) -> float:
    return round(1.0 - _health_rate(total, failure_count), 6) if total else 0.0


def _scoring_health_holdout_doc(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "decision": "not_applicable",
            "private_holdout_evaluated": False,
            "public_icp_count": 0,
            "private_holdout_icp_count": 0,
            "provider_excluded_icp_ids": [],
        }
    return {
        "decision": str(value.get("decision") or ""),
        "baseline_benchmark_bundle_id": str(value.get("baseline_benchmark_bundle_id") or ""),
        "baseline_benchmark_hash": str(value.get("baseline_benchmark_hash") or ""),
        "baseline_aggregate_score": value.get("baseline_aggregate_score"),
        "baseline_public_score": value.get("baseline_public_score"),
        "baseline_private_score": value.get("baseline_private_score"),
        "candidate_public_score": value.get("candidate_public_score"),
        "paired_base_public_score": value.get("paired_base_public_score"),
        "candidate_total_score": value.get("candidate_total_score"),
        "paired_base_total_score": value.get("paired_base_total_score"),
        "candidate_delta_vs_daily_baseline": value.get("candidate_delta_vs_daily_baseline"),
        "public_icp_count": int(value.get("public_icp_count") or 0),
        "private_holdout_icp_count": int(value.get("private_holdout_icp_count") or 0),
        "private_holdout_evaluated": bool(value.get("private_holdout_evaluated")),
        "provider_excluded_icp_ids": [
            str(item) for item in (value.get("provider_excluded_icp_ids") or ())
        ],
    }


class QualificationStyleCompanyScorer:
    """Adapter to the current company-mode high-intent scoring path.

    The imports are dynamic so the Research Lab package has no static legacy
    qualification imports. This is a temporary bridge until the scorer is moved
    under a Leadpoet-native package.
    """

    def __init__(
        self,
        *,
        attested_epoch_id: int | None = None,
        attested_purpose: str = "",
        attested_provider_profile: str = "default",
    ) -> None:
        self._attested_epoch_id = (
            int(attested_epoch_id) if attested_epoch_id is not None else None
        )
        self._attested_purpose = str(attested_purpose or "")
        self._attested_provider_profile = str(
            attested_provider_profile or "default"
        )
        self._attested_receipts: list[dict[str, Any]] = []
        self._attested_outcome_count = 0
        self._last_attested_receipt_hash = ""

    def _record_attested_outcome(self, outcome: Mapping[str, Any] | None) -> None:
        if not isinstance(outcome, Mapping):
            return
        receipt = outcome.get("receipt")
        if not isinstance(receipt, Mapping) or not receipt.get("receipt_hash"):
            return
        receipt_hash = str(receipt["receipt_hash"])
        self._attested_outcome_count += 1
        self._last_attested_receipt_hash = receipt_hash
        if any(item.get("receipt_hash") == receipt_hash for item in self._attested_receipts):
            return
        self._attested_receipts.append(dict(receipt))

    def attested_receipts(self) -> list[dict[str, Any]]:
        """Return sidecar receipts without changing the scorer result contract."""

        return [dict(item) for item in self._attested_receipts]

    def attested_outcome_count(self) -> int:
        """Count successful measured scoring calls, including deterministic retries."""

        return int(self._attested_outcome_count)

    def last_attested_receipt_hash(self) -> str:
        """Return the receipt produced by the latest successful measured call."""

        return str(self._last_attested_receipt_hash)

    async def __call__(
        self,
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[float]:
        breakdowns = await self.score_with_breakdowns(companies, icp, is_reference_model)
        return [float(item.get("final_score", 0.0) or 0.0) for item in breakdowns]

    async def score_with_breakdowns(
        self,
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[dict[str, Any]]:
        # Gateway legacy compatibility keeps the established host scorer while
        # current main is deployed before the multi-enclave V2 runtime is
        # ready. Import lazily so this shared package remains gateway-optional.
        try:
            from gateway.research_lab.tee_protocol import legacy_v1_enabled
        except ImportError:
            legacy_protocol = False
        else:
            legacy_protocol = legacy_v1_enabled()
        if (
            legacy_protocol
            or self._attested_epoch_id is None
            or self._attested_epoch_id <= 0
            or not self._attested_purpose
        ):
            return await self._score_with_breakdowns_impl(
                companies,
                icp,
                is_reference_model,
            )
        from gateway.research_lab.attested_scoring import (
            execute_required_qualification_company_scores,
        )

        attestation: dict[str, Any] = {}
        result = await execute_required_qualification_company_scores(
            epoch_id=int(self._attested_epoch_id),
            purpose=self._attested_purpose,
            companies=[dict(item) for item in companies],
            icp=dict(icp),
            is_reference_model=bool(is_reference_model),
            provider_credential_profile=self._attested_provider_profile,
            attestation_out=attestation,
        )
        self._record_attested_outcome(attestation)
        return result

    async def _score_with_breakdowns_impl(
        self,
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[dict[str, Any]]:
        models = import_module("gateway.qualification.models")
        scorer_module = import_module("qualification.scoring.lead_scorer")
        _ensure_qualification_provider_env()
        CompanyOutput = getattr(models, "CompanyOutput")
        ICPPrompt = getattr(models, "ICPPrompt")
        score_company = getattr(scorer_module, "score_company_autoresearch_intent_v2")

        # Durable, global, day-scoped scoring cache: identical scoring inputs
        # always yield identical scores, and a result computed once today is
        # reused by every later run (candidate or baseline rerun) so the LLM
        # judge cannot drift the comparison. Keyed on the full input, so it is
        # correct under the per-call de-duplication below.
        scoring_cache = None
        cache_key = ""
        try:
            from .scored_evidence_cache import get_scored_evidence_cache, scoring_cache_key

            scoring_cache = get_scored_evidence_cache()
            if scoring_cache is not None:
                cache_key = scoring_cache_key(icp, companies, is_reference_model)
                cached = scoring_cache.get(cache_key)
                if cached is not None:
                    return [dict(item) for item in cached]
        except Exception:
            scoring_cache = None

        allowed_buckets = employee_count_buckets_for_icp(icp)
        seen_companies: set[str] = set()
        breakdowns: list[dict[str, Any]] = []
        max_scored = _max_scored_companies_per_icp()
        for company in companies:
            if max_scored and len(breakdowns) >= max_scored:
                # Cost cap (bug #8): stop LLM-scoring once the per-ICP budget
                # is reached. Shared by the baseline and candidate paths since
                # both score through this adapter. With
                # RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE on, keep this cap >= 5 so
                # the top-5 normalization still sees a full lead budget.
                break
            scoring_icp = dict(icp)
            company_bucket = normalize_employee_count_bucket(
                (company or {}).get("employee_count"),
                default="",
            )
            if not company_bucket or company_bucket not in allowed_buckets:
                continue
            scoring_icp["employee_count"] = company_bucket
            normalized_company, normalized_icp = prepare_autoresearch_scoring_payload(
                company,
                scoring_icp,
            )
            icp_obj = ICPPrompt(**normalized_icp)
            company_obj = CompanyOutput(**normalized_company)
            result = await score_company(
                company=company_obj,
                icp=icp_obj,
                run_cost_usd=0.0,
                run_time_seconds=0.0,
                seen_companies=seen_companies,
                is_reference_model=is_reference_model,
            )
            if hasattr(result, "model_dump"):
                item = result.model_dump(mode="json")
            else:
                item = dict(result)
            breakdowns.append(item)
        if scoring_cache is not None and cache_key:
            try:
                scoring_cache.put(cache_key, breakdowns)
            except Exception:
                pass
        return breakdowns


def _ensure_qualification_provider_env() -> None:
    openrouter_key = os.getenv("QUALIFICATION_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        os.environ.setdefault("QUALIFICATION_OPENROUTER_API_KEY", openrouter_key)
        for module_name in (
            "gateway.qualification.utils.helpers",
            "qualification.scoring.verification_helpers",
        ):
            module = import_module(module_name)
            if not getattr(module, "OPENROUTER_API_KEY", ""):
                setattr(module, "OPENROUTER_API_KEY", openrouter_key)


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


async def _call_model_runner(
    runner: ModelRunner,
    icp: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    if inspect.iscoroutinefunction(runner) or inspect.iscoroutinefunction(getattr(runner, "__call__", None)):
        result = runner(icp, context)
    else:
        result = await asyncio.to_thread(runner, icp, context)
    if inspect.isawaitable(result):
        return await result
    return result


def _is_provider_backed_sourcing_error(exc: PrivateModelRuntimeError) -> bool:
    return "provider-backed sourcing failed before returning companies" in str(exc).lower()


def _scoreable_candidate_runtime_failure_reason(exc: PrivateModelRuntimeError) -> str:
    """Classify candidate-only adapter failures that should score as zero output.

    The reference model remains strict for non-provider runtime failures. A
    generated candidate can hang, return malformed output, or break its adapter;
    those are model-quality failures and should produce a score bundle instead
    of aborting the whole scoring job.
    """

    if _is_provider_backed_sourcing_error(exc):
        return "candidate_model_runtime_provider_error"
    lowered = str(exc).lower()
    if "adapter timed out" in lowered:
        return "candidate_model_runtime_timeout"
    if "adapter returned invalid json" in lowered:
        return "candidate_model_runtime_invalid_json"
    if "adapter failed with code" in lowered:
        return "candidate_model_runtime_adapter_failed"
    if "adapter must return a json array" in lowered:
        return "candidate_model_runtime_invalid_output"
    if "adapter returned a non-object company row" in lowered:
        return "candidate_model_runtime_invalid_output"
    if "adapter returned raw secret material" in lowered:
        return "candidate_model_runtime_invalid_output"
    return ""


def _candidate_runtime_skip_reason(failure_reason: str) -> str:
    if failure_reason == "candidate_model_runtime_provider_error":
        return ""
    suffix = failure_reason.removeprefix("candidate_model_runtime_")
    return f"candidate_model_runtime_skipped_after_{suffix}"


def prepare_autoresearch_scoring_payload(
    company: Mapping[str, Any],
    icp: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize private ICP/output into strict CompanyOutput/ICPPrompt shapes."""
    normalized_icp = canonicalize_private_model_icp(icp)
    normalized_icp.pop("employee_count_buckets", None)
    normalized_icp.pop("employee_counts", None)
    if isinstance(normalized_icp.get("employee_count"), Sequence) and not isinstance(
        normalized_icp.get("employee_count"), (str, bytes, bytearray)
    ):
        normalized_icp["employee_count"] = "|".join(
            str(item).strip()
            for item in normalized_icp["employee_count"]
            if str(item).strip()
        )
    normalized_icp = _append_bonus_intents_to_icp_signals(normalized_icp)
    normalized_company = _normalize_company_output(company, normalized_icp)
    return normalized_company, normalized_icp


def _append_bonus_intents_to_icp_signals(icp: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(icp)
    signals: list[str] = []
    evidence_types: list[str | None] = []
    # The required intent_signals are plain strings post-canonicalization, so their
    # evidence_type isn't on the entry — fall back to the ICP's flat intent_category
    # so the verifier still routes required SOCIAL_POSTING / TECHSTACK /
    # PODCAST_APPEARANCE intents to their per-type module instead of the default.
    required_evidence_type = (str(out.get("intent_category") or "").strip().upper() or None)
    for signal in out.get("intent_signals") or []:
        text = _text_from_signal_like(signal)
        if text and text not in signals:
            signals.append(text)
            evidence_types.append(_evidence_type_from_signal_like(signal) or required_evidence_type)
    for bonus in out.get("bonus_intents") or []:
        if not isinstance(bonus, Mapping):
            continue
        text = _text_from_signal_like(bonus)
        if text and text not in signals:
            signals.append(text)
            evidence_types.append(_evidence_type_from_signal_like(bonus))
    out["intent_signals"] = signals
    out["intent_signal_evidence_types"] = evidence_types
    out.pop("bonus_intents", None)
    return out


def _normalize_company_output(
    company: Mapping[str, Any],
    icp: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize private sourcing-model output into CompanyOutput shape.

    The private model currently returns a product-facing company record with
    ``hq_country`` / ``hq_state`` and a compact ``intent`` object. The scorer's
    public contract is ``CompanyOutput`` with ``country`` / ``state`` and
    ``intent_signals``. This adapter is intentionally narrow and deterministic.
    """
    row = dict(company)
    icp_signal_texts = [
        _text_from_signal_like(item)
        for item in ((icp or {}).get("intent_signals") or [])
    ]
    intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else {}
    signals = []
    if isinstance(row.get("intent_signals"), Sequence) and not isinstance(
        row.get("intent_signals"), (str, bytes, bytearray)
    ):
        for index, signal in enumerate(row.get("intent_signals") or []):
            fallback_idx = index if index < len(icp_signal_texts) else 0
            normalized = _intent_dict_from_signal_like(signal, row, icp_signal_texts, fallback_idx)
            if normalized:
                signals.append(normalized)
    else:
        signals.append(_intent_dict_from_private_record(intent, row, 0))
    if not signals:
        signals.append(_intent_dict_from_private_record(intent, row, 0))

    for additional in row.get("additional_intents") or []:
        if not isinstance(additional, Mapping):
            continue
        matched_idx = _matched_intent_index(additional, icp_signal_texts)
        if matched_idx is None:
            continue
        signals.append(_intent_dict_from_private_record(additional, row, matched_idx))

    return {
        "company_name": row.get("company_name", ""),
        "company_website": row.get("company_website", ""),
        "company_linkedin": row.get("company_linkedin", ""),
        "industry": row.get("industry", ""),
        "sub_industry": row.get("sub_industry") or row.get("subindustry") or "",
        "employee_count": row.get("employee_count", ""),
        "company_stage": row.get("company_stage", ""),
        "country": row.get("country") or row.get("hq_country") or "",
        "state": row.get("state") or row.get("hq_state") or "",
        "description": row.get("description", ""),
        "intent_signals": signals,
        "required_attribute": _required_attribute_claim(row),
    }


def _required_attribute_claim(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Bounded pass-through of the model's required_attribute validation."""
    claim = row.get("required_attribute")
    if not isinstance(claim, Mapping):
        return None
    return {
        "text": str(claim.get("text") or "")[:2000],
        "passed": bool(claim.get("passed")),
        "evidence_url": str(claim.get("evidence_url") or "")[:2000],
        "evidence_quote": str(claim.get("evidence_quote") or "")[:2000],
        "explanation": str(claim.get("explanation") or "")[:2000],
    }


def _intent_dict_from_signal_like(
    signal: Any,
    row: Mapping[str, Any],
    icp_signal_texts: Sequence[str],
    fallback_idx: int,
) -> dict[str, Any] | None:
    if isinstance(signal, Mapping):
        matched_idx = _matched_intent_index(signal, icp_signal_texts)
        if matched_idx is None:
            raw_idx = signal.get("matched_icp_signal")
            matched_idx = raw_idx if isinstance(raw_idx, int) and raw_idx >= 0 else fallback_idx
        return _intent_dict_from_private_record(signal, row, int(matched_idx))

    text = str(signal or "").strip()
    if not text:
        return None
    return _intent_dict_from_private_record(
        {
            "signal": text,
            "description": text,
            "snippet": text,
            "source": row.get("intent_source") or "news",
            "url": row.get("intent_url") or row.get("company_website") or "",
        },
        row,
        fallback_idx,
    )


def _intent_dict_from_private_record(
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    matched_idx: int,
) -> dict[str, Any]:
    signal_text = (
        record.get("signal")
        or record.get("intent_signal")
        or record.get("description")
        or row.get("intent_signal")
        or "Private sourcing model returned intent evidence."
    )
    return {
        "source": _normalize_intent_source(record.get("source") or row.get("intent_source") or "news"),
        "description": str(signal_text)[:350],
        "url": str(record.get("url") or row.get("intent_url") or row.get("company_website") or ""),
        "date": record.get("date") or row.get("intent_date"),
        "snippet": str(
            record.get("snippet")
            or record.get("why_valid")
            or record.get("description")
            or signal_text
            or row.get("description")
            or "Private sourcing model returned intent evidence."
        )[:600],
        "matched_icp_signal": int(matched_idx),
    }


def _matched_intent_index(
    item: Mapping[str, Any],
    icp_signal_texts: Sequence[str],
) -> int | None:
    explicit = item.get("matched_icp_signal")
    if isinstance(explicit, int) and 0 <= explicit < len(icp_signal_texts):
        return explicit
    target = _normalize_match_text(_text_from_signal_like(item))
    if not target:
        return None
    for idx, text in enumerate(icp_signal_texts):
        if _normalize_match_text(text) == target:
            return idx
    return None


def _text_from_signal_like(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(
            item.get("intent_signal")
            or item.get("signal")
            or item.get("text")
            or ""
        ).strip()
    return str(item or "").strip()


def _evidence_type_from_signal_like(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return None
    raw = item.get("intent_category") or item.get("category") or item.get("evidence_type")
    text = str(raw or "").strip().upper()
    return text or None


def _normalize_match_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_intent_source(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "news": "news",
        "filing": "news",
        "job_listing": "job_board",
        "job_board": "job_board",
        "company_site": "company_website",
        "company_website": "company_website",
        "social": "social_media",
        "social_media": "social_media",
        "linkedin": "linkedin",
        "github": "github",
        "review_site": "review_site",
        "wikipedia": "wikipedia",
    }
    return aliases.get(raw, "other")
