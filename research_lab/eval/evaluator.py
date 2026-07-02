"""Private-model Research Lab evaluator orchestration boundary."""

from __future__ import annotations

import asyncio
from importlib import import_module
import inspect
import os
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union

from leadpoet_verifier.aggregation import per_icp_normalized_score
from leadpoet_verifier.research_evaluation import (
    build_research_evaluation_score_bundle,
    score_bundle_hash,
)
from research_lab.canonical import sha256_json
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
    canonicalize_private_model_icp,
    employee_count_buckets_for_icp,
    ensure_private_model_outputs,
)

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

# Company scores are normalized against a fixed lead budget per ICP when the
# capped top-5 scoring flag is on; must match the verifier's advisory recompute
# (leadpoet_verifier.research_evaluation.DEFAULT_LEADS_PER_ICP_NORMALIZER).
_TOP5_LEADS_PER_ICP = 5
_EVAL_ENV_TRUTHY = {"1", "true", "yes", "on"}


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
    """Bug #15 fairness fix: retry retryable candidate provider errors once and
    exclude still-failing ICPs from the candidate aggregate instead of booking
    a hard 0 for infra noise. Default ON; flip off to restore legacy zeroing.
    """
    return _env_flag("RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY", True)


def _capped_top5_score_enabled() -> bool:
    """Bug #8 score-scale switch: per-ICP score becomes the verifier's capped
    sum(top-5 company scores)/5 instead of the unweighted company mean.

    Default OFF — enabling changes the score scale, so the daily baseline and
    all candidates must flip in the same deploy followed by a fresh baseline
    run (never compare scores across the boundary).
    """
    return _env_flag("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", False)


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


class RealEvaluatorRequired(RuntimeError):
    """Raised when a production Research Lab evaluation lacks real inputs."""


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
) -> dict[str, Any]:
    """Run a real paired base-vs-candidate evaluation.

    Production callers must pass private model runner callables backed by the
    immutable artifact. This function deliberately has no fallback fixture model.

    ``icp_checkpoint`` (optional) is invoked with each completed per-ICP result
    so callers can persist progress; ``resume_results`` (optional) supplies
    previously completed per-ICP results (matched by icp_ref/icp_hash) that are
    reused instead of re-running their ICPs. Both default to off so existing
    callers are unchanged.
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
) -> list[dict[str, Any]]:
    """Score a subset of private benchmark items without building a bundle.

    Fairness semantics (bugs #14/#15):
    - A timed-out candidate ICP is retried once; the remaining-ICP skip latch
      only engages after 2+ consecutive post-retry timeouts
      (``RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY=true`` restores first-timeout
      latching with no retry).
    - Retryable candidate provider errors are retried once; ICPs that still
      provider-error are marked ``provider_excluded`` and dropped from
      ``_benchmark_style_score`` aggregates instead of booking a hard 0
      (``RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY=false`` restores legacy
      zeroing).
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
    """

    scorer = company_scorer or QualificationStyleCompanyScorer()
    legacy_timeout_latch = _timeout_latch_legacy_enabled()
    provider_flake_retry = _provider_flake_retry_enabled()
    concurrency = _candidate_scoring_concurrency()
    resume_rows = _resume_rows_by_ref(resume_results)
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run and score one benchmark ICP.

    Returns ``(row, markers)`` where ``markers`` carries the latch signals the
    ordered driver loop needs (``timed_out``, ``latch_reason``, ``skipped``).
    """
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
    else:
        candidate_outputs, candidate_failure_reason, provider_excluded = await _run_candidate_with_retries(
            candidate_runner=candidate_runner,
            icp=icp,
            candidate_context=candidate_context,
            item_label=str(item.get("icp_ref") or item.get("icp_hash") or ""),
            legacy_timeout_latch=legacy_timeout_latch,
            provider_flake_retry=provider_flake_retry,
        )
        if candidate_failure_reason:
            failure_reasons.append(candidate_failure_reason)
            if candidate_failure_reason == "candidate_model_runtime_timeout":
                markers["timed_out"] = True
                if legacy_timeout_latch:
                    markers["latch_reason"] = _candidate_runtime_skip_reason(candidate_failure_reason)
            else:
                markers["latch_reason"] = _candidate_runtime_skip_reason(candidate_failure_reason)
    base_scores = await _maybe_await(scorer(base_outputs, icp, True)) if base_runner is not None else []
    candidate_scores = await _maybe_await(scorer(candidate_outputs, icp, False))
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
    return row, markers


async def _run_candidate_with_retries(
    *,
    candidate_runner: ModelRunner,
    icp: Mapping[str, Any],
    candidate_context: Mapping[str, Any],
    item_label: str,
    legacy_timeout_latch: bool,
    provider_flake_retry: bool,
) -> tuple[list[Mapping[str, Any]], str, bool]:
    """Run the candidate for one ICP with at most one fairness retry.

    Returns ``(outputs, failure_reason, provider_excluded)``. Timeouts retry
    once unless the legacy latch flag is on (bug #14); provider errors the
    local classifier deems retryable retry once, and an ICP that still
    provider-errors afterwards is excluded from aggregates instead of booking
    a hard 0 (bug #15). Non-scoreable runtime failures re-raise as before.
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            outputs = ensure_private_model_outputs(
                await _call_model_runner(candidate_runner, icp, candidate_context),
                context_label=f"candidate model for ICP {item_label}",
                require_non_empty=False,
            )
            return list(outputs), "", False
        except PrivateModelRuntimeError as exc:
            failure_reason = _scoreable_candidate_runtime_failure_reason(exc)
            if not failure_reason:
                raise
            if attempts == 1 and _candidate_failure_should_retry(
                failure_reason,
                str(exc),
                legacy_timeout_latch=legacy_timeout_latch,
                provider_flake_retry=provider_flake_retry,
            ):
                continue
            provider_excluded = (
                provider_flake_retry
                and failure_reason == "candidate_model_runtime_provider_error"
            )
            return [], failure_reason, provider_excluded


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
    found" and 410 responses are empirically transient-or-data-shaped, so they
    are retryable too (bug #37 semantics).
    """
    lowered = error_text.lower()
    status = _provider_error_status_code(lowered)
    if status in (408, 410, 429) or status >= 500:
        return True
    if status == 400 and "something went wrong" in lowered:
        return True
    if status in (400, 401, 403, 404, 409):
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


def _provider_error_status_code(lowered_error_text: str) -> int:
    # 410 is matched here even though the gateway diagnostics status list
    # omits it (bug #37).
    for code in (400, 401, 403, 404, 408, 409, 410, 429, 500, 502, 503, 504):
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
    private_items = [
        item for item in benchmark_items
        if str(item.get("icp_ref") or item.get("icp_hash") or "") not in public_refs
    ]
    if not public_items:
        raise RealEvaluatorRequired("private holdout gate matched zero public ICPs")
    if not private_items:
        raise RealEvaluatorRequired("private holdout gate leaves no private ICPs")

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
        "public_icp_count": len(public_items),
        "private_holdout_icp_count": len(private_items),
        "private_holdout_evaluated": bool(passed_public_gate),
        # Fixed promotion contract (bug #15): ICPs excluded for unresolved
        # provider errors; the promotion-side merge metric drops the matching
        # baseline ICPs so the exclusion stays symmetric.
        "provider_excluded_icp_ids": _provider_excluded_icp_ids(public_results),
    }
    if not passed_public_gate:
        return public_results, gate_result

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
    )
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


async def _run_parent_freshness_check(
    callback: ParentFreshnessCheck | None,
    progress: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    result = callback(progress)
    if inspect.isawaitable(result):
        await result


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
    # Two deliberate deviations:
    # - Rows marked provider_excluded (unresolved provider flakes after retry,
    #   bug #15) are dropped from the aggregate entirely instead of booking 0;
    #   their ids ride the provider_excluded_icp_ids contract so the promotion
    #   side drops the matching baseline ICPs.
    # - With RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE on (bug #8), the per-ICP score
    #   is the verifier's capped sum(top-5 company scores)/5 instead of the
    #   unweighted mean, closing the truncate-to-best-company exploit.
    capped_top5 = _capped_top5_score_enabled()
    per_icp_scores: list[float] = []
    for row in per_icp_results:
        if row.get("provider_excluded"):
            continue
        scores = row.get(score_field)
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes, bytearray)):
            per_icp_scores.append(0.0)
            continue
        values = [float(item or 0.0) for item in scores]
        per_icp_scores.append(_benchmark_icp_score(values, capped_top5=capped_top5))
    return float(sum(per_icp_scores) / len(per_icp_scores)) if per_icp_scores else 0.0


def benchmark_icp_score_from_company_scores(scores: Sequence[float]) -> float:
    """Per-ICP score on the live-gate scale for one ICP's company scores.

    Shared entry point so the daily-baseline path and the candidate path flip
    together when RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE changes; never mix scores
    computed under different settings of that flag.
    """
    values = [float(item or 0.0) for item in scores]
    return _benchmark_icp_score(values, capped_top5=_capped_top5_score_enabled())


def _benchmark_icp_score(values: list[float], *, capped_top5: bool) -> float:
    if capped_top5:
        # Match the verifier's advisory arithmetic exactly
        # (leadpoet_verifier.aggregation.per_icp_normalized_score): each company
        # score is clamped to [0, MAX_COMPANY_TOTAL_SCORE], summed over the top
        # five companies, and divided by the fixed 5-lead budget.
        top_values = sorted(values, reverse=True)[:_TOP5_LEADS_PER_ICP]
        return float(per_icp_normalized_score(top_values, max_leads=_TOP5_LEADS_PER_ICP))
    return float(sum(values) / len(values)) if values else 0.0


def _provider_excluded_icp_ids(per_icp_results: Sequence[Mapping[str, Any]]) -> list[str]:
    """Identifiers of ICPs excluded for unresolved provider errors (bug #15).

    KEY NAME IS A FIXED CONTRACT: the promotion-side merge metric reads
    ``provider_excluded_icp_ids`` and drops the matching baseline ICPs so
    exclusion stays symmetric. Ids are the per-ICP ``icp_ref`` (falling back to
    ``icp_hash``), deduplicated and sorted.
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
            sha256_json(dict(patch_manifest))
            if image_candidate
            else patch.manifest_hash()
        ),
        icp_set_hash=benchmark_set.icp_set_hash,
        scoring_version=benchmark_set.scoring_version,
        evaluator_version=str(run_context["evaluator_version"]),
        per_icp_results=per_icp_results,
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
        policy=policy or {},
        signature_ref=str(run_context.get("signature_ref") or ""),
    )
    extra_fields = dict(extra_bundle_fields or {})
    scoring_health = build_scoring_health_doc(
        per_icp_results,
        private_holdout_gate=extra_fields.get("private_holdout_gate"),
    )
    enriched = {
        **bundle,
        **extra_fields,
        "scoring_health": scoring_health,
        # Fixed promotion contract (bug #15): also surfaced at bundle top level
        # so non-holdout-gate callers can consume it.
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
    }
    holdout_doc = _scoring_health_holdout_doc(private_holdout_gate)
    degraded = any(value > 0 for value in counts.values())
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
        models = import_module("gateway.qualification.models")
        scorer_module = import_module("qualification.scoring.lead_scorer")
        _ensure_qualification_provider_env()
        CompanyOutput = getattr(models, "CompanyOutput")
        ICPPrompt = getattr(models, "ICPPrompt")
        score_company = getattr(scorer_module, "score_company_autoresearch_intent_v2")

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
        signals = list(row.get("intent_signals") or [])
    else:
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
    }


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
