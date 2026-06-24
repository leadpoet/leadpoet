"""Private-model Research Lab evaluator orchestration boundary."""

from __future__ import annotations

import asyncio
from importlib import import_module
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union

from leadpoet_verifier.research_evaluation import build_research_evaluation_score_bundle

from .artifacts import PrivateModelArtifactManifest, validate_private_model_artifact_manifest
from .benchmark import SealedBenchmarkSet, validate_sealed_benchmark_set
from .patches import CandidatePatchManifest, validate_candidate_patch_manifest


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
    benchmark_items: Sequence[Mapping[str, Any]],
    base_runner: ModelRunner | None,
    candidate_runner: ModelRunner | None,
    company_scorer: CompanyScorer | None = None,
    run_context: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a real paired base-vs-candidate evaluation.

    Production callers must pass private model runner callables backed by the
    immutable artifact. This function deliberately has no fallback fixture model.
    """
    artifact = artifact_manifest if isinstance(artifact_manifest, PrivateModelArtifactManifest) else PrivateModelArtifactManifest.from_mapping(artifact_manifest)
    benchmark_set = benchmark if isinstance(benchmark, SealedBenchmarkSet) else SealedBenchmarkSet.from_mapping(benchmark)
    patch = patch_manifest if isinstance(patch_manifest, CandidatePatchManifest) else CandidatePatchManifest.from_mapping(patch_manifest)

    errors = []
    errors.extend(validate_private_model_artifact_manifest(artifact))
    errors.extend(validate_sealed_benchmark_set(benchmark_set))
    errors.extend(validate_candidate_patch_manifest(patch, expected_parent_artifact_hash=artifact.model_artifact_hash))
    if errors:
        raise ValueError("; ".join(errors))
    if base_runner is None or candidate_runner is None:
        raise RealEvaluatorRequired("private base_runner and candidate_runner are required")
    if not benchmark_items:
        raise RealEvaluatorRequired("sealed benchmark items are required")

    scorer = company_scorer or QualificationStyleCompanyScorer()
    per_icp_results: list[dict[str, Any]] = []
    for item in benchmark_items:
        icp = item.get("icp")
        if not isinstance(icp, Mapping):
            raise RealEvaluatorRequired("benchmark item is missing private ICP payload")
        base_outputs = await _maybe_await(base_runner(icp, run_context))
        candidate_outputs = await _maybe_await(candidate_runner(icp, {**dict(run_context), "patch": patch.to_dict()}))
        base_scores = await _maybe_await(scorer(base_outputs, icp, True))
        candidate_scores = await _maybe_await(scorer(candidate_outputs, icp, False))
        per_icp_results.append(
            {
                "icp_ref": str(item.get("icp_ref") or item.get("icp_hash") or ""),
                "icp_hash": str(item.get("icp_hash") or ""),
                "status": "completed",
                "hard_failure": False,
                "base_company_scores": base_scores,
                "candidate_company_scores": candidate_scores,
            }
        )

    return build_score_bundle_from_scored_icps(
        artifact_manifest=artifact,
        benchmark=benchmark_set,
        patch_manifest=patch,
        per_icp_results=per_icp_results,
        run_context=run_context,
        policy=policy or {},
    )


def build_score_bundle_from_scored_icps(
    *,
    artifact_manifest: PrivateModelArtifactManifest | Mapping[str, Any],
    benchmark: SealedBenchmarkSet | Mapping[str, Any],
    patch_manifest: CandidatePatchManifest | Mapping[str, Any],
    per_icp_results: Sequence[Mapping[str, Any]],
    run_context: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = artifact_manifest if isinstance(artifact_manifest, PrivateModelArtifactManifest) else PrivateModelArtifactManifest.from_mapping(artifact_manifest)
    benchmark_set = benchmark if isinstance(benchmark, SealedBenchmarkSet) else SealedBenchmarkSet.from_mapping(benchmark)
    patch = patch_manifest if isinstance(patch_manifest, CandidatePatchManifest) else CandidatePatchManifest.from_mapping(patch_manifest)
    errors = []
    errors.extend(validate_private_model_artifact_manifest(artifact))
    errors.extend(validate_sealed_benchmark_set(benchmark_set))
    errors.extend(validate_candidate_patch_manifest(patch, expected_parent_artifact_hash=artifact.model_artifact_hash))
    if errors:
        raise ValueError("; ".join(errors))
    if not per_icp_results:
        raise RealEvaluatorRequired("real scored ICP results are required")

    return build_research_evaluation_score_bundle(
        run_id=str(run_context["run_id"]),
        ticket_id=str(run_context["ticket_id"]),
        miner_hotkey=str(run_context["miner_hotkey"]),
        island=str(run_context["island"]),
        evaluation_epoch=int(run_context["evaluation_epoch"]),
        parent_artifact_hash=artifact.model_artifact_hash,
        candidate_artifact_hash=patch.candidate_artifact_hash,
        private_model_manifest_hash=artifact.manifest_hash,
        candidate_patch_hash=patch.manifest_hash(),
        icp_set_hash=benchmark_set.icp_set_hash,
        scoring_version=benchmark_set.scoring_version,
        evaluator_version=str(run_context["evaluator_version"]),
        per_icp_results=per_icp_results,
        evidence_bundle_refs=tuple(str(item) for item in run_context.get("evidence_bundle_refs", ())),
        execution_trace_ref=str(run_context["execution_trace_ref"]),
        cost_ledger_ref=str(run_context["cost_ledger_ref"]),
        benchmark_split_ref=benchmark_set.split_ref,
        policy=policy or {},
        signature_ref=str(run_context.get("signature_ref") or ""),
    )


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
        CompanyOutput = getattr(models, "CompanyOutput")
        ICPPrompt = getattr(models, "ICPPrompt")
        score_company = getattr(scorer_module, "score_company")

        icp_obj = ICPPrompt(**dict(icp))
        seen_companies: set[str] = set()
        breakdowns: list[dict[str, Any]] = []
        for company in companies:
            company_obj = CompanyOutput(**_normalize_company_output(company))
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


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


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
