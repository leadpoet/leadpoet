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


def _artifact() -> SourcingModelArtifactIdentity:
    return SourcingModelArtifactIdentity(
        repository="leadpoet/Sourcing_model",
        branch="leadpoet-lab",
        commit_sha="1" * 40,
        artifact_uri="s3://research-lab/model/candidate.tar.gz",
        model_artifact_hash=H("1"),
        manifest_hash=H("2"),
        routing_contract_hash=H("c"),
        routing_catalog_hash=H("3"),
        routing_policy_hash=H("4"),
        feature_schema_hash=H("5"),
        verifier_contract_hash=H("6"),
    )


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
    artifact = _artifact()
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
        artifact=artifact,
        routing_payload={"profile_id": "baseline", "tools": [binding.tool_id]},
        binding_ids=(binding.binding_id,),
    )
    candidate = RoutingExperimentV2Variant(
        variant_id="candidate",
        stage="candidate_acquisition",
        artifact=artifact,
        routing_payload={"profile_id": "candidate", "tools": [binding.tool_id]},
        binding_ids=(binding.binding_id,),
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


def _adapter(
    *,
    contract_hash: str = "c" * 64,
    receipt: object | None = None,
    target: int = 2,
    artifact_errors: tuple[str, ...] = (),
):
    attempt = receipt or SimpleNamespace(
        plan_sha256="a" * 64,
        stop_policy_sha256="b" * 64,
        step_order=0,
        attempt=0,
        tool_id="candidate.registry_feed",
        disposition="succeeded",
        outcome_code="verified_candidates",
        provider_call_count=1,
        estimated_cost_usd=0.0125,
        latency_seconds=0.25,
        raw_candidate_count=8,
        normalized_candidate_count=6,
        unique_candidate_count=4,
        verified_qualified_count=2,
        verification_receipt_sha256="e" * 64,
        sha256=lambda: "d" * 64,
    )
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


def _provider_receipt(unit_ref: str = "icp.cal") -> ProviderReceipt:
    binding = _binding()
    identity = {
        "binding_id": binding.binding_id,
        "tool_id": binding.tool_id,
        "binding_version": binding.adapter_version,
        "source_lineage_id": binding.source_lineage_id,
        "unit_ref": unit_ref,
        "request_fingerprint": H("f"),
        "outcome": "verified",
        "evidence_hash": H("0"),
        "credit_microunits": 25,
        "latency_ms": 250,
        "execution_mode": ReceiptExecutionMode.FIXTURE.value,
    }
    return ProviderReceipt(
        receipt_ref="provider_receipt:"
        + sha256_json(identity).split(":", 1)[1][:16],
        **identity,
    )


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
        outcome_reasons=() if skipped else ((provider.tool_id, provider.outcome),),
        provider_receipt_refs=() if skipped else (provider.receipt_ref,),
        total_credit_microunits=0 if skipped else provider.credit_microunits,
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
        provider_receipt_refs=(receipt.provider_receipt_ref,),
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
    assert receipt.cost_microusd == 12_500
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
    )
    assert metrics[0].waterfall_attempt_count == 1
    assert metrics[0].provider_receipt_refs == ()


@pytest.mark.skipif(
    not os.environ.get("SOURCING_MODEL_CHECKOUT"),
    reason="requires an exact Sourcing_model checkout",
)
def test_exact_sourcing_model_candidate_receipt_contract_is_compatible():
    checkout = Path(os.environ["SOURCING_MODEL_CHECKOUT"]).resolve()
    fixture_spec = importlib.util.spec_from_file_location(
        "exact_pr274_model_runner_fixture",
        checkout / "tests" / "test_model_runner.py",
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture = importlib.util.module_from_spec(fixture_spec)
    original_path = list(sys.path)
    sys.path.insert(0, str(checkout))
    try:
        fixture_spec.loader.exec_module(fixture)
    finally:
        sys.path[:] = original_path
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
    terminal, _transitions = fixture._run(
        start,
        identity,
        fixture._candidate(),
    )

    adapted = adapt_exact_model_candidate_receipt(
        terminal,
        expected_release_identity_sha256=identity["release_identity_sha256"],
        expected_binding_contracts_sha256=manifest["binding_contracts_sha256"],
    )
    assert adapted["candidate_attempt_metrics"]
    assert adapted["candidate_attempt_metrics"][0][
        "verified_qualified_count"
    ] == 1
    assert adapted["candidate_route"]["stop_reason"] == "target_reached"

    forged = copy.deepcopy(terminal)
    forged["model_receipt"]["candidate_attempt_metrics"][0][
        "verified_qualified_count"
    ] = 50
    with pytest.raises(RoutingExperimentError, match="receipt_identity_differs"):
        adapt_exact_model_candidate_receipt(
            forged,
            expected_release_identity_sha256=identity[
                "release_identity_sha256"
            ],
            expected_binding_contracts_sha256=manifest[
                "binding_contracts_sha256"
            ],
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
    assert baseline_calibration.verified_qualified_per_usd == 160.0
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
        )
    with pytest.raises(RoutingExperimentError, match="chain_hash_differs"):
        evaluate_candidate_waterfall_metrics(
            spec=spec,
            evaluation=_evaluation(spec, receipt),
            receipts=(replace(receipt, attempt_chain_sha256="7" * 64),),
            target_verified_qualified_count=2,
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
    assert "FOR INSERT TO service_role WITH CHECK (true)" in sql
    assert "provider_receipt_ref = ''" in sql
    assert "disposition = 'skipped'" in sql
    assert "idx_research_lab_candidate_waterfall_provider_receipt" in sql
    assert sql.count("REFERENCES public.research_lab_routing_experiments_v2") == 2
    assert "REFERENCES public.research_lab_routing_decision_receipts_v2" in sql
    assert "REFERENCES public.research_lab_routing_evaluation_receipts_v2" in sql
    assert "prior_attempt_receipt_sha256" in sql
    assert "attempt_chain_sha256" in sql
    assert sql.count("= jsonb_build_object(") == 2
    assert "provider_outcome IN (" in sql
    assert sql.count("jsonb_typeof(metric_doc->") == 3
    assert "auth.role()" not in sql
    assert "DROP TABLE" not in sql
    assert "DROP TRIGGER" not in sql
    assert "DROP POLICY" not in sql
