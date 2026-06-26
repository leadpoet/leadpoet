#!/usr/bin/env python3
"""Verify Research Lab real-evaluation contracts without private model access."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_verifier.research_evaluation import verify_research_evaluation_score_bundle
from research_lab.canonical import sha256_json
from research_lab.eval import (
    CandidatePatchManifest,
    PrivateModelArtifactManifest,
    RealEvaluatorRequired,
    SealedBenchmarkSet,
    build_score_bundle_from_scored_icps,
    evaluate_private_model_pair,
    validate_candidate_patch_manifest,
    validate_private_model_artifact_manifest,
    validate_sealed_benchmark_set,
)


def main() -> int:
    errors: list[str] = []
    try:
        artifact = _valid_artifact_manifest()
        patch = _valid_patch_manifest(artifact)
        benchmark = _valid_benchmark()
        run_context = _valid_run_context()

        if validate_private_model_artifact_manifest(artifact):
            errors.append("valid artifact manifest did not validate")
        if validate_candidate_patch_manifest(
            patch,
            allowed_component_ids=("source_router",),
            expected_parent_artifact_hash=artifact.model_artifact_hash,
        ):
            errors.append("valid candidate patch did not validate")
        if validate_sealed_benchmark_set(benchmark):
            errors.append("valid sealed benchmark did not validate")

        bad_artifact = {**artifact.to_dict(), "image_digest": "123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/private:latest"}
        if "image_digest_must_be_immutable_digest" not in validate_private_model_artifact_manifest(bad_artifact):
            errors.append("mutable image tag was not rejected")

        bad_patch = {**patch.to_dict(), "patch_type": "CODE_EDIT"}
        if "patch_type_deferred:CODE_EDIT" not in validate_candidate_patch_manifest(bad_patch):
            errors.append("CODE_EDIT patch was not rejected")

        score_bundle = build_score_bundle_from_scored_icps(
            artifact_manifest=artifact,
            benchmark=benchmark,
            patch_manifest=patch,
            per_icp_results=_scored_icp_results(),
            run_context=run_context,
            policy=_policy(),
        )
        verification = verify_research_evaluation_score_bundle(score_bundle, policy=_policy())
        if not verification["passed"]:
            errors.append("valid score bundle failed verification: " + "; ".join(verification["errors"]))
        if verification["on_chain_submission_allowed"]:
            errors.append("score-bundle verification must not allow direct on-chain submission")

        tampered = {
            **score_bundle,
            "aggregates": {**score_bundle["aggregates"], "candidate_score": 999.0},
        }
        tampered_verification = verify_research_evaluation_score_bundle(tampered, policy=_policy())
        if tampered_verification["passed"]:
            errors.append("tampered score bundle passed verification")

        try:
            asyncio.run(
                evaluate_private_model_pair(
                    artifact_manifest=artifact,
                    benchmark=benchmark,
                    patch_manifest=patch,
                    benchmark_items=[],
                    base_runner=None,
                    candidate_runner=None,
                    run_context=run_context,
                    policy=_policy(),
                )
            )
            errors.append("missing private runners did not fail closed")
        except RealEvaluatorRequired:
            pass

        zero_base_policy = {**_policy(), "min_successful_icps": 2, "min_delta": 0.0, "min_delta_lcb": 0.0}
        zero_base_bundle = asyncio.run(
            evaluate_private_model_pair(
                artifact_manifest=artifact,
                benchmark=benchmark,
                patch_manifest=patch,
                benchmark_items=[
                    {"icp_ref": "icp:a", "icp_hash": "sha256:" + "a" * 64, "icp": {"industry": "Software"}},
                    {"icp_ref": "icp:b", "icp_hash": "sha256:" + "b" * 64, "icp": {"industry": "Healthcare"}},
                ],
                base_runner=_empty_base_runner,
                candidate_runner=_single_candidate_runner,
                company_scorer=_fixture_company_scorer,
                run_context=run_context,
                policy=zero_base_policy,
            )
        )
        zero_base_verification = verify_research_evaluation_score_bundle(zero_base_bundle, policy=zero_base_policy)
        if not zero_base_verification["passed"]:
            errors.append("empty base-output score bundle failed verification: " + "; ".join(zero_base_verification["errors"]))
        if zero_base_bundle["aggregates"]["base_score"] != 0.0:
            errors.append("empty base-output score bundle did not score reference model as zero")
        if zero_base_bundle["aggregates"]["candidate_score"] <= 0.0:
            errors.append("empty base-output score bundle did not score candidate outputs")
        if "reference_model_zero_companies" not in zero_base_bundle["aggregates"]["per_icp_results"][0]["failure_reason"]:
            errors.append("empty base-output score bundle did not record sanitized failure reason")

        captured_runtime_patch: dict[str, object] = {}

        async def _capturing_candidate_runner(_icp: dict[str, object], context: dict[str, object]) -> list[dict[str, object]]:
            captured_runtime_patch.update(dict(context.get("patch") or {}))
            return [{"company_name": "Example Co"}]

        asyncio.run(
            evaluate_private_model_pair(
                artifact_manifest=artifact,
                benchmark=benchmark,
                patch_manifest=_legacy_param_patch_manifest(artifact),
                benchmark_items=[
                    {"icp_ref": "icp:c", "icp_hash": "sha256:" + "c" * 64, "icp": {"industry": "Software"}},
                ],
                base_runner=_empty_base_runner,
                candidate_runner=_capturing_candidate_runner,
                company_scorer=_fixture_company_scorer,
                run_context=run_context,
                policy=zero_base_policy,
            )
        )
        if captured_runtime_patch.get("patch_doc") != {"params": {"max_leads": 3}}:
            errors.append("candidate runner did not receive runtime-normalized patch_doc")
    except Exception as exc:
        errors.append(f"unexpected verifier exception: {exc}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab real-evaluation contracts verified: immutable private artifact, "
        "sealed benchmark, typed patch, score bundle verifier, tamper rejection, fail-closed runner gate."
    )
    return 0


def _valid_artifact_manifest() -> PrivateModelArtifactManifest:
    payload = {
        "model_artifact_hash": "sha256:" + "1" * 64,
        "git_commit_sha": "abcdef1234567890",
        "image_digest": "123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/private-evaluator@sha256:" + "2" * 64,
        "config_hash": "sha256:" + "3" * 64,
        "component_registry_version": "component-registry:v1",
        "scoring_adapter_version": "qualification-company-scorer:v1",
        "manifest_uri": "s3://leadpoet-private-model-artifacts/manifests/private-evaluator-v1.json",
        "signature_ref": "kms-signature:research-lab-eval:test",
        "build_id": "private-evaluator-build-001",
    }
    return PrivateModelArtifactManifest.from_mapping({**payload, "manifest_hash": sha256_json(payload)})


def _valid_patch_manifest(artifact: PrivateModelArtifactManifest) -> CandidatePatchManifest:
    return CandidatePatchManifest.from_mapping(
        {
            "patch_type": "PROMPT_EDIT",
            "target_component_id": "source_router",
            "parent_artifact_hash": artifact.model_artifact_hash,
            "patch_payload_hash": "sha256:" + "4" * 64,
            "redacted_summary": "Tighten source routing toward evidence-rich hiring and budget signals.",
            "validation_result": "passed",
            "candidate_artifact_hash": "sha256:" + "5" * 64,
            "patch_doc": {"redacted": True},
        }
    )


def _legacy_param_patch_manifest(artifact: PrivateModelArtifactManifest) -> CandidatePatchManifest:
    return CandidatePatchManifest.from_mapping(
        {
            "patch_type": "PARAM_EDIT",
            "target_component_id": "output_budget",
            "parent_artifact_hash": artifact.model_artifact_hash,
            "patch_payload_hash": "sha256:" + "9" * 64,
            "redacted_summary": "Reduce max leads to concentrate precision.",
            "validation_result": "passed",
            "candidate_artifact_hash": "sha256:" + "a" * 64,
            "patch_doc": {"param_name": "max_leads", "new_value": 3},
        }
    )


def _valid_benchmark() -> SealedBenchmarkSet:
    return SealedBenchmarkSet.from_mapping(
        {
            "benchmark_id": "sealed-benchmark:qualification:intent:v1",
            "icp_set_hash": "sha256:" + "6" * 64,
            "split_ref": "sealed_benchmark:qualification:intent:v1",
            "item_refs": ("icp:a", "icp:b", "icp:c"),
            "scoring_version": "qualification-company-scorer:v1",
            "hidden_plaintext_available": True,
        }
    )


def _valid_run_context() -> dict[str, object]:
    return {
        "run_id": "11111111-1111-4111-8111-111111111111",
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "miner_hotkey": "5FevalMiner111111111111111111111111111111111",
        "island": "generalist",
        "evaluation_epoch": 301,
        "evaluator_version": "research-lab-private-evaluator:v1",
        "evidence_bundle_refs": ["evidence_bundle:sha256:" + "7" * 64],
        "execution_trace_ref": "execution_trace:11111111-1111-4111-8111-111111111111",
        "cost_ledger_ref": "cost_ledger:sha256:" + "8" * 64,
        "signature_ref": "kms-signature:research-lab-eval:test",
    }


def _scored_icp_results() -> list[dict[str, object]]:
    return [
        {"icp_ref": "icp:a", "icp_hash": "sha256:" + "a" * 64, "base_company_scores": [80, 60], "candidate_company_scores": [90, 70]},
        {"icp_ref": "icp:b", "icp_hash": "sha256:" + "b" * 64, "base_company_scores": [40], "candidate_company_scores": [70, 20]},
        {"icp_ref": "icp:c", "icp_hash": "sha256:" + "c" * 64, "base_company_scores": [50, 0], "candidate_company_scores": [65, 5]},
    ]


def _policy() -> dict[str, object]:
    return {
        "min_delta": 2.0,
        "min_delta_lcb": 2.0,
        "min_successful_icps": 3,
        "max_hard_failures": 0,
        "min_candidate_score": 15.0,
        "observed_cost_usd": 1.25,
    }


async def _empty_base_runner(_icp: dict[str, object], _context: dict[str, object]) -> list[dict[str, object]]:
    return []


async def _single_candidate_runner(_icp: dict[str, object], _context: dict[str, object]) -> list[dict[str, object]]:
    return [{"company_name": "Example Co"}]


async def _fixture_company_scorer(
    companies: list[dict[str, object]],
    _icp: dict[str, object],
    _is_reference_model: bool,
) -> list[float]:
    return [75.0 for _company in companies]


if __name__ == "__main__":
    raise SystemExit(main())
