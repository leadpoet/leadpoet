"""Candidate waterfall sidecars use the shared PR 93 routing contracts."""

from __future__ import annotations

from dataclasses import replace
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from research_lab.canonical import sha256_json
from research_lab.candidate_routing_experiments import (
    CandidateWaterfallReceipt,
    adapt_exact_model_candidate_receipt,
    candidate_waterfall_receipt_from_model,
    evaluate_candidate_waterfall_metrics,
    validate_candidate_routing_model_runtime,
)
from research_lab.routing_experiments import (
    ExperimentCreditBudget,
    ProviderBindingIdentity,
    ProviderReceipt,
    ReceiptExecutionMode,
    RoutingDecisionReceiptV2,
    RoutingEvaluationGates,
    RoutingEvaluationMetrics,
    RoutingExperimentError,
    RoutingExperimentV2Evaluation,
    RoutingExperimentV2Input,
    RoutingExperimentV2Spec,
    RoutingExperimentV2Variant,
    RoutingExperimentV2VariantEvaluation,
    SourcingModelArtifactIdentity,
)


def H(char: str) -> str:
    return "sha256:" + char * 64


def _artifact(
    *,
    branch: str = "leadpoet-lab",
    artifact_char: str = "1",
) -> SourcingModelArtifactIdentity:
    model_artifact_hash = H(artifact_char)
    commit_sha = artifact_char * 40
    manifest_payload = {
        "model_artifact_hash": model_artifact_hash,
        "git_commit_sha": commit_sha,
        "image_digest": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sourcing-model@sha256:" + "d" * 64,
        "config_hash": H("d"),
        "component_registry_version": "registry-v1",
        "scoring_adapter_version": "scoring-v1",
        "manifest_uri": "s3://research-lab/model/manifest.json",
        "signature_ref": "s3://research-lab/model/manifest.sig",
        "build_id": "",
        "routing_contract_hash": H("c"),
        "routing_catalog_hash": H("3"),
        "routing_policy_hash": H("4"),
        "feature_schema_hash": H("5"),
        "verifier_contract_hash": H("6"),
        "candidate_waterfall_contract_sha256": "c" * 64,
    }
    manifest_hash = sha256_json(manifest_payload)
    return SourcingModelArtifactIdentity(
        repository="leadpoet/Sourcing_model",
        branch=branch,
        commit_sha=commit_sha,
        artifact_uri=f"s3://research-lab/model/candidate-{artifact_char}.tar.gz",
        model_artifact_hash=model_artifact_hash,
        manifest_hash=manifest_hash,
        routing_contract_hash=H("c"),
        routing_catalog_hash=H("3"),
        routing_policy_hash=H("4"),
        feature_schema_hash=H("5"),
        verifier_contract_hash=H("6"),
    )


def _artifact_manifest(artifact: SourcingModelArtifactIdentity) -> dict[str, object]:
    return {
        "model_artifact_hash": artifact.model_artifact_hash,
        "git_commit_sha": artifact.commit_sha,
        "image_digest": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sourcing-model@sha256:" + "d" * 64,
        "config_hash": H("d"),
        "component_registry_version": "registry-v1",
        "scoring_adapter_version": "scoring-v1",
        "manifest_uri": "s3://research-lab/model/manifest.json",
        "manifest_hash": artifact.manifest_hash,
        "signature_ref": "s3://research-lab/model/manifest.sig",
        "build_id": "",
        "routing_contract_hash": artifact.routing_contract_hash,
        "routing_catalog_hash": artifact.routing_catalog_hash,
        "routing_policy_hash": artifact.routing_policy_hash,
        "feature_schema_hash": artifact.feature_schema_hash,
        "verifier_contract_hash": artifact.verifier_contract_hash,
        "candidate_waterfall_contract_sha256": "c" * 64,
    }


def _binding() -> ProviderBindingIdentity:
    return ProviderBindingIdentity(
        binding_id="binding.registry",
        provider_id="registry",
        tool_id="candidate.registry_feed",
        source_lineage_id="lineage.registry",
        adapter_version="adapter-v2",
        manifest_hash=H("7"),
        capability_hash=H("8"),
        execution_contract_hash=H("9"),
        cost_model_hash=H("a"),
    )


def _spec() -> RoutingExperimentV2Spec:
    champion_artifact = _artifact(branch="main", artifact_char="1")
    challenger_artifact = _artifact(artifact_char="2")
    binding = _binding()
    feature_payload = {
        "schema_version": "routing-feature-set:v1",
        "features": ["company.country.us"],
    }
    experiment_input = RoutingExperimentV2Input(
        stage="candidate_acquisition",
        feature_set_hash=sha256_json(feature_payload),
        feature_set_payload=feature_payload,
        calibration_unit_refs=("icp.cal",),
        holdout_unit_refs=("icp.hold",),
        gold_label_set_hash=sha256_json(
            {"labels": [("icp.cal", True), ("icp.hold", True)]}
        ),
    )
    baseline = RoutingExperimentV2Variant(
        variant_id="baseline",
        stage="candidate_acquisition",
        artifact=champion_artifact,
        routing_payload={"profile_id": "baseline", "tools": [binding.tool_id]},
        binding_ids=(binding.binding_id,),
        artifact_authority_manifest=_artifact_manifest(champion_artifact),
    )
    candidate = RoutingExperimentV2Variant(
        variant_id="candidate",
        stage="candidate_acquisition",
        artifact=challenger_artifact,
        routing_payload={"profile_id": "candidate", "tools": [binding.tool_id]},
        binding_ids=(binding.binding_id,),
        artifact_authority_manifest=_artifact_manifest(challenger_artifact),
    )
    return RoutingExperimentV2Spec(
        experiment_id="candidate-waterfall-v2",
        input=experiment_input,
        variants=(baseline, candidate),
        baseline_variant_id="baseline",
        provider_bindings=(binding,),
        credit_budget=ExperimentCreditBudget(
            total_credit_microunits=10_000,
            provider_credit_ceilings={binding.binding_id: 10_000},
        ),
        gates=RoutingEvaluationGates(
            min_calibration_precision=0.8,
            min_holdout_precision=0.8,
            min_holdout_recall=0.1,
            max_holdout_no_signal_credit_microunits=10_000,
            min_marginal_verified_positives_per_credit=0.0,
            intent_release_policy_hash="",
        ),
    )


def _model_receipt(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "plan_sha256": "a" * 64,
        "stop_policy_sha256": "b" * 64,
        "step_order": 0,
        "attempt": 0,
        "tool_id": "candidate.registry_feed",
        "disposition": "succeeded",
        "outcome_code": "verified_candidates",
        "provider_call_count": 1,
        "estimated_cost_usd": 0.0125,
        "latency_seconds": 0.25,
        "raw_candidate_count": 8,
        "normalized_candidate_count": 6,
        "unique_candidate_count": 4,
        "verified_qualified_count": 2,
        "verification_receipt_sha256": "e" * 64,
        "sha256": lambda: "d" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _adapter(
    *,
    contract_hash: str = "c" * 64,
    receipt: object | None = None,
    target: int = 2,
    artifact_errors: tuple[str, ...] = (),
):
    attempt = receipt or _model_receipt()
    identity = {
        "contract_version": "candidate-waterfall-execution:v1",
        "stop_policy_schema_version": "candidate-stop-policy:v1",
        "progress_schema_version": "candidate-waterfall-progress:v1",
        "attempt_receipt_schema_version": "candidate-step-attempt-receipt:v1",
        "stop_metric": "verified_qualified_count",
        "decisions": ["continue", "stop"],
        "attempt_dispositions": ["deferred", "failed", "missed", "skipped", "succeeded"],
        "provider_results_can_satisfy_target": False,
        "attempt_sequence": "contiguous_zero_based",
        "step_sequence": "compiled_route_prefix_only",
        "retry_precondition": "deferred",
        "contract_sha256": contract_hash,
    }
    stop = SimpleNamespace(
        target_verified_qualified_count=target,
        sha256=lambda: "b" * 64,
    )
    runtime = SimpleNamespace(
        candidate_waterfall_execution_contract_identity=lambda: identity,
        runtime_routing_metadata=lambda: {
            "catalog_sha256": "3" * 64,
            "policy_sha256": "4" * 64,
            "candidate_waterfall_execution": identity,
        },
        evaluate_candidate_waterfall_payloads=lambda *args: {
            "progress": {},
            "decision": {},
        },
        CandidateStopPolicy=SimpleNamespace(from_payload=lambda payload: stop),
        CandidateStepAttemptReceipt=SimpleNamespace(from_payload=lambda payload: attempt),
    )
    return SimpleNamespace(
        runtime=runtime,
        validate_artifact_identity=lambda artifact: artifact_errors,
        parse_plan=lambda payload: SimpleNamespace(),
        plan_hash=lambda plan: "a" * 64,
        route_hash=lambda plan: "b" * 64,
    )


def _adapt_receipt(
    *,
    spec: RoutingExperimentV2Spec,
    decision: RoutingDecisionReceiptV2,
    provider: ProviderReceipt | None,
    adapter: object | None = None,
    published_count: int = 0,
) -> CandidateWaterfallReceipt:
    return candidate_waterfall_receipt_from_model(
        spec=spec,
        variant_id=decision.variant_id,
        decision_receipt=decision,
        provider_receipt=provider,
        plan_payload={"schema_version": "candidate-routing-plan:v1"},
        stop_policy_payload={"schema_version": "candidate-stop-policy:v1"},
        receipt_payloads=({},),
        model_adapter=adapter or _adapter(),
        published_count=published_count,
    )


def _provider_receipt(
    unit_ref: str = "icp.cal",
    *,
    request_char: str = "f",
    credit_microunits: int = 25,
    latency_ms: int = 250,
    call_count: int = 1,
) -> ProviderReceipt:
    binding = _binding()
    identity = {
        "binding_id": binding.binding_id,
        "tool_id": binding.tool_id,
        "binding_version": binding.adapter_version,
        "source_lineage_id": binding.source_lineage_id,
        "unit_ref": unit_ref,
        "request_fingerprint": H(request_char),
        "outcome": "verified",
        "evidence_hash": H("0"),
        "credit_microunits": credit_microunits,
        "latency_ms": latency_ms,
        "execution_mode": ReceiptExecutionMode.FIXTURE.value,
    }
    try:
        receipt = ProviderReceipt(
            receipt_ref="provider_receipt:"
            + sha256_json({**identity, "call_count": call_count}).split(":", 1)[1][:16],
            call_count=call_count,
            **identity,
        )
    except TypeError:
        receipt = ProviderReceipt(
            receipt_ref="provider_receipt:"
            + sha256_json(identity).split(":", 1)[1][:16],
            **identity,
        )
        object.__setattr__(receipt, "call_count", call_count)
    return receipt


def _artifact_key(spec: RoutingExperimentV2Spec, variant_id: str) -> str:
    artifact = next(item.artifact for item in spec.variants if item.variant_id == variant_id)
    return sha256_json(
        {
            "model_artifact_hash": artifact.model_artifact_hash,
            "manifest_hash": artifact.manifest_hash,
            "commit_sha": artifact.commit_sha,
        }
    )


def _decision(
    spec: RoutingExperimentV2Spec,
    provider: ProviderReceipt,
    *,
    variant_id: str = "baseline",
    skipped: bool = False,
    plan_hash: str = H("a"),
    route_hash: str = H("b"),
) -> RoutingDecisionReceiptV2:
    draft = RoutingDecisionReceiptV2(
        receipt_id="routing_decision:pending",
        experiment_id=spec.experiment_id,
        variant_id=variant_id,
        artifact_key=_artifact_key(spec, variant_id),
        stage="candidate_acquisition",
        unit_ref=provider.unit_ref,
        plan_hash=plan_hash,
        route_hash=route_hash,
        considered_tool_ids=(provider.tool_id,),
        attempted_tool_ids=() if skipped else (provider.tool_id,),
        skipped_tool_reasons=(
            ((provider.tool_id, "runtime_unavailable"),) if skipped else ()
        ),
        outcome_reasons=(
            ()
            if skipped
            else ((provider.tool_id, provider.outcome),)
        ),
        provider_receipt_refs=(
            () if skipped else (provider.receipt_ref,)
        ),
        total_credit_microunits=(
            0 if skipped else provider.credit_microunits
        ),
        latency_ms=0 if skipped else provider.latency_ms,
        execution_mode=provider.execution_mode,
    )
    payload = draft.to_dict()
    return replace(
        draft,
        receipt_id="routing_decision:"
        + sha256_json(payload).split(":", 1)[1][:16],
    )


def _empty_metrics(split: str) -> RoutingEvaluationMetrics:
    return RoutingEvaluationMetrics(
        split=split,
        unit_count=1,
        predicted_positive_count=0,
        true_positive_count=0,
        false_positive_count=0,
        false_negative_count=1,
        verified_positive_count=0,
        rejected_count=0,
        source_miss_count=1,
        adapter_failure_count=0,
        total_credit_microunits=0,
        no_signal_credit_microunits=0,
        unique_rescue_count=0,
        unique_rescue_credit_microunits=0,
        marginal_verified_positives_per_credit=0.0,
        precision=0.0,
        recall=0.0,
        mean_latency_ms=0.0,
        source_lineage_overlap_count=0,
        source_lineage_overlap_rate=0.0,
    )


def _evaluation(
    spec: RoutingExperimentV2Spec,
    receipt: CandidateWaterfallReceipt,
) -> RoutingExperimentV2Evaluation:
    variants = []
    for variant in spec.variants:
        has_receipt = variant.variant_id == receipt.variant_id
        variants.append(
            RoutingExperimentV2VariantEvaluation(
                variant_id=variant.variant_id,
                artifact_key=_artifact_key(spec, variant.variant_id),
                stage=variant.stage,
                calibration=_empty_metrics("calibration"),
                holdout=_empty_metrics("holdout"),
                passed_precision_gate=False,
                passed_recall_gate=False,
                passed_cost_gate=True,
                passed_efficiency_gate=True,
                passed=False,
                decision_receipt_refs=(receipt.decision_receipt_id,) if has_receipt else (),
                provider_receipt_refs=(
                    (receipt.provider_receipt_ref,)
                    if has_receipt and receipt.provider_receipt_ref
                    else ()
                ),
            )
        )
    return RoutingExperimentV2Evaluation(
        receipt_id="routing_evaluation_v2:" + "f" * 16,
        experiment_id=spec.experiment_id,
        experiment_hash=spec.experiment_hash(),
        variants=tuple(variants),
        baseline_variant_id=spec.baseline_variant_id,
        selected_variant_id="",
        decision_receipt_refs=(receipt.decision_receipt_id,),
        provider_receipt_refs=(
            (receipt.provider_receipt_ref,) if receipt.provider_receipt_ref else ()
        ),
        provider_cache_hits=0,
        provider_cache_misses=1,
    )


def test_model_receipt_adapter_uses_shared_experiment_provider_and_decision_contracts():
    spec = _spec()
    provider = _provider_receipt()
    decision = _decision(spec, provider)
    receipt = _adapt_receipt(
        spec=spec,
        decision=decision,
        provider=provider,
        published_count=1,
    )

    assert receipt.experiment_hash == spec.experiment_hash()
    assert receipt.decision_receipt_id == decision.receipt_id
    assert receipt.provider_receipt_ref == provider.receipt_ref
    assert receipt.model_contract_sha256 == "c" * 64
    assert receipt.verified_qualified_count == 2
    assert receipt.published_count == 1
    assert receipt.billed_credit_microunits == provider.credit_microunits
    assert receipt.target_verified_qualified_count == 2
    assert receipt.prior_attempt_receipt_sha256 == ""
    assert receipt.to_dict()["receipt_hash"].startswith("sha256:")


def test_exact_model_contract_preflight_fails_closed():
    spec = _spec()
    identity = _adapter().runtime.candidate_waterfall_execution_contract_identity()
    unsafe_adapter = _adapter()
    unsafe_adapter.runtime.runtime_routing_metadata = lambda: {
        "catalog_sha256": "3" * 64,
        "policy_sha256": "4" * 64,
        "candidate_waterfall_execution": {**identity, "contract_sha256": "9" * 64},
    }
    with pytest.raises(RoutingExperimentError, match="identity_differs_from_metadata"):
        validate_candidate_routing_model_runtime(
            spec=spec,
            variant_id="baseline",
            model_adapter=unsafe_adapter,
        )
    with pytest.raises(RoutingExperimentError, match="artifact_identity_differs"):
        validate_candidate_routing_model_runtime(
            spec=spec,
            variant_id="baseline",
            model_adapter=_adapter(
                artifact_errors=("model_artifact_hash_mismatch",)
            ),
        )
    with pytest.raises(RoutingExperimentError, match="runtime_contract_is_incomplete"):
        validate_candidate_routing_model_runtime(
            spec=spec,
            variant_id="baseline",
            model_adapter=SimpleNamespace(runtime=SimpleNamespace()),
        )


def test_candidate_preflight_requires_exact_branch_contract_and_live_signature():
    spec = _spec()
    baseline = next(item for item in spec.variants if item.variant_id == "baseline")
    candidate = next(item for item in spec.variants if item.variant_id == "candidate")
    main_baseline = replace(
        baseline,
        artifact=_artifact(branch="main", artifact_char="1"),
    )
    main_spec = replace(
        spec,
        variants=(
            replace(
                main_baseline,
                artifact_authority_manifest=_artifact_manifest(main_baseline.artifact),
            ),
            candidate,
        ),
    )
    validate_candidate_routing_model_runtime(
        spec=main_spec,
        variant_id="baseline",
        model_adapter=_adapter(),
    )

    candidate_with_main_baseline = replace(
        spec,
        variants=(main_baseline, candidate),
    )
    validate_candidate_routing_model_runtime(
        spec=candidate_with_main_baseline,
        variant_id="candidate",
        model_adapter=_adapter(),
    )

    # The required main/leadpoet-lab branch split is itself part of the
    # artifact identity. A duplicate branch identity cannot pass admission.

    forged_manifest = dict(candidate.artifact_authority_manifest or {})
    forged_manifest["candidate_waterfall_contract_sha256"] = "9" * 64
    forged_candidate = replace(
        candidate,
        artifact_authority_manifest=forged_manifest,
    )
    forged_spec = replace(
        spec,
        variants=tuple(
            forged_candidate if item.variant_id == "candidate" else item
            for item in spec.variants
        ),
    )
    with pytest.raises(RoutingExperimentError, match="signed_artifact_manifest_is_invalid"):
        validate_candidate_routing_model_runtime(
            spec=forged_spec,
            variant_id="candidate",
            model_adapter=_adapter(),
        )

    measured_spec = replace(
        spec,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    with pytest.raises(
        RoutingExperimentError,
        match="signature_authority_is_required",
    ):
        validate_candidate_routing_model_runtime(
            spec=measured_spec,
            variant_id="candidate",
            model_adapter=_adapter(),
        )

    class Authority:
        def verify(self, *, artifact, manifest):
            assert manifest["model_artifact_hash"] == artifact.model_artifact_hash
            return {
                "verified": True,
                "model_artifact_hash": artifact.model_artifact_hash,
                "manifest_hash": artifact.manifest_hash,
                "commit_sha": artifact.commit_sha,
            }

    signed_identity = validate_candidate_routing_model_runtime(
        spec=measured_spec,
        variant_id="candidate",
        model_adapter=_adapter(),
        artifact_authority=Authority(),
    )
    assert signed_identity["contract_sha256"] == "c" * 64

    class MismatchedAuthority(Authority):
        def verify(self, *, artifact, manifest):
            result = super().verify(artifact=artifact, manifest=manifest)
            result["commit_sha"] = "0" * 40
            return result

    with pytest.raises(
        RoutingExperimentError,
        match="signature_commit_sha_differs",
    ):
        validate_candidate_routing_model_runtime(
            spec=measured_spec,
            variant_id="candidate",
            model_adapter=_adapter(),
            artifact_authority=MismatchedAuthority(),
        )


def test_model_receipt_adapter_rejects_unlinked_provider_receipt():
    spec = _spec()
    provider = _provider_receipt()
    decision = _decision(spec, provider)
    other = _provider_receipt("icp.hold")
    with pytest.raises(RoutingExperimentError, match="not_in_decision"):
        _adapt_receipt(
            spec=spec,
            decision=decision,
            provider=other,
        )


@pytest.mark.parametrize(
    ("model_receipt", "message"),
    (
        (
            _model_receipt(provider_call_count=2),
            "provider_call_count_differs_from_provider_receipt",
        ),
        (
            _model_receipt(latency_seconds=0.5),
            "latency_differs_from_provider_receipt",
        ),
    ),
)
def test_model_receipt_adapter_rejects_operational_metric_drift(
    model_receipt: object,
    message: str,
):
    spec = _spec()
    provider = _provider_receipt()
    with pytest.raises(RoutingExperimentError, match=message):
        _adapt_receipt(
            spec=spec,
            decision=_decision(spec, provider),
            provider=provider,
            adapter=_adapter(receipt=model_receipt),
        )


def test_model_receipt_adapter_reconciles_two_authoritative_provider_calls():
    spec = _spec()
    provider = _provider_receipt(
        credit_microunits=55,
        latency_ms=375,
        call_count=2,
    )
    model_receipt = _model_receipt(
        provider_call_count=2,
        estimated_cost_usd=0.0275,
        latency_seconds=0.375,
    )
    receipt = _adapt_receipt(
        spec=spec,
        decision=_decision(spec, provider),
        provider=provider,
        adapter=_adapter(receipt=model_receipt),
    )
    assert receipt.provider_receipt_ref == provider.receipt_ref
    assert receipt.provider_call_count == 2
    assert receipt.billed_credit_microunits == provider.credit_microunits
    assert receipt.latency_ms == provider.latency_ms

    evaluation = _evaluation(spec, receipt)
    metrics = evaluate_candidate_waterfall_metrics(
        spec=spec,
        evaluation=evaluation,
        receipts=(receipt,),
        target_verified_qualified_count=2,
        authoritative_provider_receipts=(provider,),
    )
    assert metrics[0].provider_call_count == 2
    assert metrics[0].total_billed_credit_microunits == provider.credit_microunits
    assert metrics[0].total_latency_ms == provider.latency_ms


def test_model_receipt_adapter_rejects_partial_two_call_authority():
    spec = _spec()
    provider = _provider_receipt(call_count=2)
    with pytest.raises(
        RoutingExperimentError,
        match="provider_receipt_authority_is_missing",
    ):
        receipt = _adapt_receipt(
            spec=spec,
            decision=_decision(spec, provider),
            provider=provider,
            adapter=_adapter(receipt=_model_receipt(provider_call_count=2)),
        )
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=_evaluation(spec, receipt),
            receipts=(receipt,),
            target_verified_qualified_count=2,
            authoritative_provider_receipts=(),
        )


def test_model_receipt_adapter_rejects_two_call_count_or_billing_mismatch():
    spec = _spec()
    provider = _provider_receipt(call_count=2)
    with pytest.raises(RoutingExperimentError, match="differs_from_provider_receipt"):
        _adapt_receipt(
            spec=spec,
            decision=_decision(spec, provider),
            provider=provider,
            adapter=_adapter(receipt=_model_receipt(provider_call_count=1)),
        )
    receipt = _adapt_receipt(
        spec=spec,
        decision=_decision(spec, provider),
        provider=provider,
        adapter=_adapter(receipt=_model_receipt(provider_call_count=2)),
    )
    with pytest.raises(RoutingExperimentError, match="authority_differs"):
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=_evaluation(spec, receipt),
            receipts=(replace(receipt, billed_credit_microunits=56),),
            target_verified_qualified_count=2,
            authoritative_provider_receipts=(provider,),
        )


def test_model_skipped_receipt_uses_decision_reason_without_provider_receipt():
    spec = _spec()
    provider = _provider_receipt()
    skipped_model_receipt = SimpleNamespace(
        plan_sha256="a" * 64,
        stop_policy_sha256="b" * 64,
        step_order=0,
        attempt=0,
        tool_id=provider.tool_id,
        disposition="skipped",
        outcome_code="runtime_unavailable",
        provider_call_count=0,
        estimated_cost_usd=0.0,
        latency_seconds=0.0,
        raw_candidate_count=0,
        normalized_candidate_count=0,
        unique_candidate_count=0,
        verified_qualified_count=0,
        verification_receipt_sha256="",
        sha256=lambda: "8" * 64,
    )
    receipt = _adapt_receipt(
        spec=spec,
        decision=_decision(spec, provider, skipped=True),
        provider=None,
        adapter=_adapter(receipt=skipped_model_receipt),
    )

    assert receipt.disposition == "skipped"
    assert receipt.provider_outcome == "skipped"
    assert receipt.provider_receipt_ref == ""
    assert receipt.provider_call_count == 0
    metrics = evaluate_candidate_waterfall_metrics(
        spec=spec,
        evaluation=_evaluation(spec, receipt),
        receipts=(receipt,),
        target_verified_qualified_count=2,
        authoritative_provider_receipts=(),
    )
    assert metrics[0].waterfall_attempt_count == 1
    assert metrics[0].provider_receipt_refs == ()


@pytest.mark.skipif(
    not os.environ.get("SOURCING_MODEL_CHECKOUT"),
    reason="requires an exact Sourcing_model checkout",
)
def test_exact_sourcing_model_candidate_receipt_contract_is_compatible(request):
    checkout = Path(os.environ["SOURCING_MODEL_CHECKOUT"]).resolve()
    fixture_spec = importlib.util.spec_from_file_location(
        "exact_pr274_model_runner_fixture",
        checkout / "tests" / "test_model_runner.py",
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture = importlib.util.module_from_spec(fixture_spec)
    original_path = list(sys.path)
    isolated_prefixes = ("gateway", "qualification", "sourcing_model")
    original_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name in isolated_prefixes
        or name.startswith(tuple(prefix + "." for prefix in isolated_prefixes))
    }
    for name in original_modules:
        sys.modules.pop(name, None)
    def restore_import_state() -> None:
        sys.path[:] = original_path
        for name in tuple(sys.modules):
            if name in isolated_prefixes or name.startswith(
                tuple(prefix + "." for prefix in isolated_prefixes)
            ):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)

    request.addfinalizer(restore_import_state)
    sys.path.insert(0, str(checkout))
    fixture_spec.loader.exec_module(fixture)
    manifest = fixture._capability_manifest()
    identity = fixture._release_identity(manifest)
    start = fixture.build_model_start_request(
        input={
            "kind": "normalized_icp",
            "normalized_icp": fixture._normalized_icp(contact=False),
        },
        execution_mode="full_company",
        target_count=1,
        evaluated_on=fixture.EVALUATED_ON,
        host_capability_manifest=manifest,
        release_identity=identity,
    )
    candidate = fixture._candidate()

    def run_exact(*, candidate_latency_ms: float = 10):
        current = fixture.continue_model_runner(
            start,
            expected_release_identity=identity,
        )
        provider_receipt = None
        while current["status"] == "action_required":
            action = current["action"]
            completion = fixture._completion_for(action, candidate)
            if action["action_type"] == "execute_candidate_tool":
                provider_identity = {
                    "binding_id": "binding.exact-model-runner",
                    "tool_id": action["tool_id"],
                    "binding_version": "model-runner-v1",
                    "source_lineage_id": "lineage.exact-model-runner",
                    "unit_ref": "icp.exact-model-runner",
                    "request_fingerprint": H("f"),
                    "outcome": "verified",
                    "evidence_hash": H("e"),
                    "credit_microunits": 0,
                    "latency_ms": 10,
                    "execution_mode": ReceiptExecutionMode.FIXTURE.value,
                }
                provider_receipt = ProviderReceipt(
                    receipt_ref="provider_receipt:"
                    + sha256_json(provider_identity).split(":", 1)[1][:16],
                    **provider_identity,
                )
                completion = fixture.build_model_action_completion(
                    action,
                    outcome=completion["outcome"],
                    reason_code=completion["reason_code"],
                    provider_response=completion["provider_response"],
                    calls=completion["calls"],
                    cost_credits=completion["cost_credits"],
                    latency_ms=candidate_latency_ms,
                    provider_receipt_ref=provider_receipt.receipt_ref,
                )
            current = fixture.continue_model_runner(
                start,
                expected_release_identity=identity,
                continuation=current["continuation"],
                completion=completion,
            )
        assert provider_receipt is not None
        return current, provider_receipt

    terminal, provider_receipt = run_exact()

    adapted = adapt_exact_model_candidate_receipt(
        terminal,
        expected_release_identity_sha256=identity["release_identity_sha256"],
        expected_binding_contracts_sha256=manifest["binding_contracts_sha256"],
        expected_candidate_waterfall_contract_sha256=identity[
            "candidate_waterfall_contract_sha256"
        ],
        authoritative_provider_receipts=(provider_receipt,),
    )
    assert adapted["candidate_attempt_metrics"]
    assert adapted["candidate_attempt_metrics"][0][
        "raw_candidate_count"
    ] == 1
    assert adapted["candidate_attempt_metrics"][0]["provider_call_count"] == 1
    assert adapted["candidate_attempt_metrics"][0][
        "provider_receipt_ref"
    ] == provider_receipt.receipt_ref
    assert adapted["candidate_stop_reason"] == "target_reached"
    assert adapted["candidate_route"]["candidate_plan_sha256"]

    forged = copy.deepcopy(terminal)
    forged["result"]["receipt"]["tool_attempts"][0]["result_count"] = 50
    with pytest.raises(RoutingExperimentError, match="result_identity_differs"):
        adapt_exact_model_candidate_receipt(
            forged,
            expected_release_identity_sha256=identity[
                "release_identity_sha256"
            ],
            expected_binding_contracts_sha256=manifest[
                "binding_contracts_sha256"
            ],
            expected_candidate_waterfall_contract_sha256=identity[
                "candidate_waterfall_contract_sha256"
            ],
            authoritative_provider_receipts=(provider_receipt,),
        )

    mismatched_accounting_terminal, mismatched_accounting_provider = run_exact(
        candidate_latency_ms=11
    )
    with pytest.raises(
        RoutingExperimentError,
        match="differs_from_provider_receipt",
    ):
        adapt_exact_model_candidate_receipt(
            mismatched_accounting_terminal,
            expected_release_identity_sha256=identity[
                "release_identity_sha256"
            ],
            expected_binding_contracts_sha256=manifest[
                "binding_contracts_sha256"
            ],
            expected_candidate_waterfall_contract_sha256=identity[
                "candidate_waterfall_contract_sha256"
            ],
            authoritative_provider_receipts=(mismatched_accounting_provider,),
        )

    with pytest.raises(RoutingExperimentError, match="coverage_differs"):
        adapt_exact_model_candidate_receipt(
            terminal,
            expected_release_identity_sha256=identity[
                "release_identity_sha256"
            ],
            expected_binding_contracts_sha256=manifest[
                "binding_contracts_sha256"
            ],
            expected_candidate_waterfall_contract_sha256=identity[
                "candidate_waterfall_contract_sha256"
            ],
            authoritative_provider_receipts=(
                provider_receipt,
                _provider_receipt("icp.unreferenced"),
            ),
        )


def test_candidate_metrics_are_sidecars_on_shared_evaluation():
    spec = _spec()
    provider = _provider_receipt()
    decision = _decision(spec, provider)
    receipt = _adapt_receipt(
        spec=spec,
        decision=decision,
        provider=provider,
        published_count=1,
    )
    metrics = evaluate_candidate_waterfall_metrics(
        spec=spec,
        evaluation=_evaluation(spec, receipt),
        receipts=(receipt,),
        target_verified_qualified_count=2,
        authoritative_provider_receipts=(provider,),
    )

    assert [(item.variant_id, item.split) for item in metrics] == [
        ("baseline", "calibration"),
        ("baseline", "holdout"),
        ("candidate", "calibration"),
        ("candidate", "holdout"),
    ]
    baseline_calibration = metrics[0]
    assert baseline_calibration.fulfilled_unit_count == 1
    assert baseline_calibration.fulfillment_rate == 1.0
    assert baseline_calibration.verification_rate == 0.25
    assert baseline_calibration.publication_rate == 0.5
    assert baseline_calibration.total_billed_credit_microunits == 25
    assert baseline_calibration.verified_qualified_per_credit == 80_000.0
    assert baseline_calibration.metric_hash.startswith("sha256:")
    assert metrics[1].waterfall_attempt_count == 0


def test_candidate_metrics_reject_duplicate_attempts():
    spec = _spec()
    provider = _provider_receipt()
    receipt = _adapt_receipt(
        spec=spec,
        decision=_decision(spec, provider),
        provider=provider,
    )
    with pytest.raises(RoutingExperimentError, match="attempt_is_duplicated"):
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=_evaluation(spec, receipt),
            receipts=(receipt, receipt),
            target_verified_qualified_count=2,
            authoritative_provider_receipts=(provider,),
        )


def test_candidate_metrics_reject_duplicate_provider_receipt_sidecars():
    spec = _spec()
    provider = _provider_receipt()
    receipt = _adapt_receipt(
        spec=spec,
        decision=_decision(spec, provider),
        provider=provider,
    )
    duplicate_provider = replace(
        receipt,
        step_order=1,
        attempt_sequence=1,
        prior_attempt_receipt_sha256=receipt.attempt_receipt_sha256,
        attempt_receipt_sha256="7" * 64,
        attempt_chain_sha256="6" * 64,
    )
    with pytest.raises(RoutingExperimentError, match="sidecar_is_duplicated"):
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=_evaluation(spec, receipt),
            receipts=(receipt, duplicate_provider),
            target_verified_qualified_count=2,
            authoritative_provider_receipts=(provider,),
        )


def test_candidate_metrics_reject_stop_target_and_attempt_chain_drift():
    spec = _spec()
    provider = _provider_receipt()
    receipt = _adapt_receipt(
        spec=spec,
        decision=_decision(spec, provider),
        provider=provider,
    )
    with pytest.raises(RoutingExperimentError, match="target_differs"):
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=_evaluation(spec, receipt),
            receipts=(receipt,),
            target_verified_qualified_count=3,
            authoritative_provider_receipts=(provider,),
        )
    with pytest.raises(RoutingExperimentError, match="chain_hash_differs"):
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=_evaluation(spec, receipt),
            receipts=(replace(receipt, attempt_chain_sha256="7" * 64),),
            target_verified_qualified_count=2,
            authoritative_provider_receipts=(provider,),
        )


def test_candidate_metrics_reject_partial_provider_sidecar_coverage():
    spec = _spec()
    provider = _provider_receipt()
    receipt = _adapt_receipt(
        spec=spec,
        decision=_decision(spec, provider),
        provider=provider,
    )
    evaluation = _evaluation(spec, receipt)

    with pytest.raises(RoutingExperimentError, match="provider_sidecar_coverage"):
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=evaluation,
            receipts=(),
            target_verified_qualified_count=2,
            authoritative_provider_receipts=(provider,),
        )


def test_candidate_metrics_reject_partial_decision_sidecar_coverage():
    spec = _spec()
    provider = _provider_receipt()
    skipped_model_receipt = SimpleNamespace(
        plan_sha256="a" * 64,
        stop_policy_sha256="b" * 64,
        step_order=0,
        attempt=0,
        tool_id=provider.tool_id,
        disposition="skipped",
        outcome_code="runtime_unavailable",
        provider_call_count=0,
        estimated_cost_usd=0.0,
        latency_seconds=0.0,
        raw_candidate_count=0,
        normalized_candidate_count=0,
        unique_candidate_count=0,
        verified_qualified_count=0,
        verification_receipt_sha256="",
        sha256=lambda: "8" * 64,
    )
    skipped_receipt = _adapt_receipt(
        spec=spec,
        decision=_decision(spec, provider, skipped=True),
        provider=None,
        adapter=_adapter(receipt=skipped_model_receipt),
    )
    evaluation = _evaluation(spec, skipped_receipt)

    with pytest.raises(RoutingExperimentError, match="decision_sidecar_coverage"):
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=evaluation,
            receipts=(),
            target_verified_qualified_count=2,
        )


def test_candidate_metrics_require_authoritative_provider_facts():
    spec = _spec()
    provider = _provider_receipt()
    receipt = _adapt_receipt(
        spec=spec,
        decision=_decision(spec, provider),
        provider=provider,
    )
    evaluation = _evaluation(spec, receipt)
    with pytest.raises(RoutingExperimentError, match="authority_is_missing"):
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=evaluation,
            receipts=(receipt,),
            target_verified_qualified_count=2,
        )
    with pytest.raises(RoutingExperimentError, match="authority_differs"):
        forged_receipt = replace(
            receipt,
            billed_credit_microunits=26,
        )
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=evaluation,
            receipts=(forged_receipt,),
            target_verified_qualified_count=2,
            authoritative_provider_receipts=(provider,),
        )


def test_postgres_persistence_is_append_only_and_has_no_parallel_lifecycle():
    sql = (
        Path(__file__).parents[1]
        / "scripts"
        / "162-research-lab-candidate-routing-experiments.sql"
    ).read_text()
    assert "research_lab_candidate_waterfall_receipts" in sql
    assert "research_lab_candidate_waterfall_metrics" in sql
    for duplicate in (
        "research_lab_candidate_routing_experiments",
        "research_lab_candidate_routing_arms",
        "research_lab_candidate_routing_runs",
        "research_lab_candidate_routing_decisions",
    ):
        assert duplicate not in sql
    assert sql.count("FORCE ROW LEVEL SECURITY") == 2
    assert "BEFORE UPDATE OR DELETE" in sql
    assert "ON DELETE CASCADE" not in sql
    assert "FOR SELECT TO service_role USING (true)" in sql
    assert "CREATE OR REPLACE FUNCTION public.research_lab_candidate_append_waterfall_receipt_v1" in sql
    assert "CREATE OR REPLACE FUNCTION public.research_lab_candidate_append_waterfall_metric_v1" in sql
    assert "REVOKE ALL ON TABLE" in sql
    assert "provider_receipt_ref = ''" in sql
    assert "disposition = 'skipped'" in sql
    assert "idx_research_lab_candidate_waterfall_provider_receipt" in sql
    assert sql.count("REFERENCES public.research_lab_routing_experiments_v2") == 2
    assert "FOREIGN KEY (decision_receipt_id, experiment_hash)" in sql
    assert "FOREIGN KEY (evaluation_receipt_id, experiment_hash)" in sql
    assert "REFERENCES public.research_lab_routing_decision_receipts_v2" in sql
    assert "REFERENCES public.research_lab_routing_evaluation_receipts_v2" in sql
    assert "rl_route_decision_receipt_experiment_uq" in sql
    assert "rl_route_evaluation_receipt_experiment_uq" in sql
    assert "research_lab_routing_jsonb_hash_v2" in sql
    assert "receipt_id = 'candidate_waterfall:'" in sql
    assert "metric_id = 'candidate_metric:'" in sql
    assert "prior_attempt_receipt_sha256" in sql
    assert "attempt_chain_sha256" in sql
    assert sql.count("= jsonb_build_object(") == 2
    assert "provider_outcome IN (" in sql
    assert sql.count("jsonb_typeof(metric_doc->") == 3
    assert "auth.role()" not in sql
    assert "DROP TABLE" not in sql
    assert "DROP TRIGGER" not in sql
    assert "DROP POLICY" not in sql
