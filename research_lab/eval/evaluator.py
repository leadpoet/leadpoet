"""Private-model Research Lab evaluator orchestration boundary."""

from __future__ import annotations

import asyncio
from importlib import import_module
import os
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union

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
) -> dict[str, Any]:
    """Run a real paired base-vs-candidate evaluation.

    Production callers must pass private model runner callables backed by the
    immutable artifact. This function deliberately has no fallback fixture model.
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
    if base_runner is None or candidate_runner is None:
        raise RealEvaluatorRequired("private base_runner and candidate_runner are required")
    if not benchmark_items:
        raise RealEvaluatorRequired("sealed benchmark items are required")

    scorer = company_scorer or QualificationStyleCompanyScorer()
    runtime_patch = None if image_candidate else runtime_compatible_candidate_patch_manifest(patch)
    if private_holdout_gate:
        per_icp_results, gate_result = await _score_with_private_holdout_gate(
            benchmark_items=benchmark_items,
            base_runner=base_runner,
            candidate_runner=candidate_runner,
            scorer=scorer,
            run_context=run_context,
            image_candidate=image_candidate,
            runtime_patch=runtime_patch,
            gate=private_holdout_gate,
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
    base_runner: ModelRunner,
    candidate_runner: ModelRunner,
    company_scorer: CompanyScorer | None = None,
    run_context: Mapping[str, Any],
    image_candidate: bool,
    runtime_patch: CandidatePatchManifest | None = None,
) -> list[dict[str, Any]]:
    """Score a subset of private benchmark items without building a bundle."""

    scorer = company_scorer or QualificationStyleCompanyScorer()
    per_icp_results: list[dict[str, Any]] = []
    for item in benchmark_items:
        icp = item.get("icp")
        if not isinstance(icp, Mapping):
            raise RealEvaluatorRequired("benchmark item is missing private ICP payload")
        base_outputs = ensure_private_model_outputs(
            await _call_model_runner(base_runner, icp, run_context),
            context_label=f"reference model for ICP {item.get('icp_ref') or item.get('icp_hash') or ''}",
            require_non_empty=False,
        )
        candidate_context = dict(run_context)
        if not image_candidate:
            if runtime_patch is None:
                raise RealEvaluatorRequired("candidate patch runtime payload is required for patch candidates")
            candidate_context["patch"] = runtime_patch.to_dict()
        candidate_outputs = ensure_private_model_outputs(
            await _call_model_runner(candidate_runner, icp, candidate_context),
            context_label=f"candidate model for ICP {item.get('icp_ref') or item.get('icp_hash') or ''}",
            require_non_empty=False,
        )
        base_scores = await _maybe_await(scorer(base_outputs, icp, True))
        candidate_scores = await _maybe_await(scorer(candidate_outputs, icp, False))
        failure_reasons: list[str] = []
        if not base_outputs:
            failure_reasons.append("reference_model_zero_companies")
        elif not base_scores:
            failure_reasons.append("reference_model_zero_scoreable_companies")
        if not candidate_outputs:
            failure_reasons.append("candidate_model_zero_companies")
        elif not candidate_scores:
            failure_reasons.append("candidate_model_zero_scoreable_companies")
        per_icp_results.append(
            {
                "icp_ref": str(item.get("icp_ref") or item.get("icp_hash") or ""),
                "icp_hash": str(item.get("icp_hash") or ""),
                "status": "completed",
                "hard_failure": False,
                "base_company_scores": base_scores,
                "candidate_company_scores": candidate_scores,
                "failure_reason": ";".join(failure_reasons),
            }
        )
    return per_icp_results


async def _score_with_private_holdout_gate(
    *,
    benchmark_items: Sequence[Mapping[str, Any]],
    base_runner: ModelRunner,
    candidate_runner: ModelRunner,
    scorer: CompanyScorer,
    run_context: Mapping[str, Any],
    image_candidate: bool,
    runtime_patch: CandidatePatchManifest | None,
    gate: Mapping[str, Any],
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
    )
    baseline_public_score = float(gate.get("baseline_public_score") or 0.0)
    candidate_public_score = _benchmark_style_score(public_results, "candidate_company_scores")
    base_public_score = _benchmark_style_score(public_results, "base_company_scores")
    passed_public_gate = candidate_public_score + 1e-9 >= baseline_public_score
    gate_result = {
        "schema_version": "1.0",
        "gate_type": "public_score_before_private_holdout",
        "decision": "private_holdout_approved" if passed_public_gate else "rejected_before_private_holdout",
        "baseline_benchmark_bundle_id": str(gate.get("baseline_benchmark_bundle_id") or ""),
        "baseline_public_score": round(baseline_public_score, 6),
        "candidate_public_score": round(candidate_public_score, 6),
        "paired_base_public_score": round(base_public_score, 6),
        "public_icp_count": len(public_items),
        "private_holdout_icp_count": len(private_items),
        "private_holdout_evaluated": bool(passed_public_gate),
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
    )
    return [*public_results, *private_results], gate_result


def _benchmark_style_score(
    per_icp_results: Sequence[Mapping[str, Any]],
    score_field: str,
) -> float:
    per_icp_scores: list[float] = []
    for row in per_icp_results:
        scores = row.get(score_field)
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes, bytearray)):
            per_icp_scores.append(0.0)
            continue
        values = [float(item or 0.0) for item in scores]
        per_icp_scores.append(float(sum(values) / len(values)) if values else 0.0)
    return float(sum(per_icp_scores) / len(per_icp_scores)) if per_icp_scores else 0.0


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
    if not extra_bundle_fields:
        return bundle
    enriched = {
        **bundle,
        **dict(extra_bundle_fields),
        "score_bundle_hash": "",
        "anchored_hash": "",
    }
    enriched_hash = score_bundle_hash(enriched)
    return {**enriched, "score_bundle_hash": enriched_hash, "anchored_hash": enriched_hash}


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
        score_company = getattr(scorer_module, "score_company")

        allowed_buckets = employee_count_buckets_for_icp(icp)
        seen_companies: set[str] = set()
        breakdowns: list[dict[str, Any]] = []
        for company in companies:
            normalized_company = _normalize_company_output(company)
            company_bucket = normalize_employee_count_bucket(
                normalized_company.get("employee_count"),
                default="",
            )
            if not company_bucket or company_bucket not in allowed_buckets:
                continue
            scoring_icp = dict(icp)
            scoring_icp["employee_count"] = company_bucket
            scoring_icp.pop("employee_count_buckets", None)
            scoring_icp.pop("employee_counts", None)
            icp_obj = ICPPrompt(**canonicalize_private_model_icp(scoring_icp))
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
    return await _maybe_await(runner(icp, context))


def _normalize_company_output(company: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize private sourcing-model output into CompanyOutput shape.

    The private model currently returns a product-facing company record with
    ``hq_country`` / ``hq_state`` and a compact ``intent`` object. The scorer's
    public contract is ``CompanyOutput`` with ``country`` / ``state`` and
    ``intent_signals``. This adapter is intentionally narrow and deterministic.
    """
    row = dict(company)
    intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else {}
    signal = {
        "source": _normalize_intent_source(intent.get("source") or row.get("intent_source") or "news"),
        "description": str(
            intent.get("signal")
            or intent.get("description")
            or row.get("intent_signal")
            or "Private sourcing model returned intent evidence."
        )[:350],
        "url": str(intent.get("url") or row.get("intent_url") or row.get("company_website") or ""),
        "date": intent.get("date") or row.get("intent_date"),
        "snippet": str(
            intent.get("snippet")
            or intent.get("signal")
            or intent.get("description")
            or row.get("description")
            or "Private sourcing model returned intent evidence."
        )[:600],
        "matched_icp_signal": int(intent.get("matched_icp_signal", 0) or 0),
    }
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
        "intent_signals": row.get("intent_signals") or [signal],
    }


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
