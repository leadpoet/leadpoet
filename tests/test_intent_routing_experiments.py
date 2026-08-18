from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ExperimentCreditBudget,
    FrozenRoutingInput,
    JsonlProviderReceiptRepository,
    LabRoutingProfile,
    LLMRoutingProfileProposal,
    ProviderBindingIdentity,
    ProviderOutcome,
    ProviderReceipt,
    ProviderReceiptStore,
    ReceiptExecutionMode,
    RoutingAdmissionPlanAdapter,
    RoutingEvaluationGates,
    RoutingExperimentError,
    RoutingExperimentSpec,
    SourcingModelArtifactIdentity,
    admit_llm_routing_profile_proposal,
    evaluate_routing_experiment,
    lab_hash_to_model,
    model_hash_to_lab,
    promote_routing_profile_to_lab,
    provider_receipt_key,
    validate_provider_receipt,
    validate_routing_experiment_spec,
    validate_sourcing_model_artifact_identity,
    verify_lab_routing_artifact_lineage,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
HASH_D = "sha256:" + "d" * 64
HASH_E = "sha256:" + "e" * 64
HASH_F = "sha256:" + "f" * 64
HASH_0 = "sha256:" + "0" * 64


@dataclass(frozen=True)
class FakeModelPlan:
    payload: dict[str, object]


class FakeRoutingAdmissionPlanAdapter:
    """Test double for the pinned model's exact RoutingProfile schema."""

    def parse(self, payload):
        required = {"profile_id", "profile_version", "feature_set_hash", "steps"}
        if set(payload) != required:
            raise ValueError("RoutingProfile payload shape drift")
        if not isinstance(payload["steps"], list):
            raise ValueError("steps must be an array")
        return FakeModelPlan(dict(payload))

    def validate(self, plan, *, signal_type, feature_set_hash, binding_tool_ids):
        del signal_type
        errors = []
        if plan.payload["feature_set_hash"] != feature_set_hash:
            errors.append("model_feature_set_hash_mismatch")
        tools = [step.get("tool_id") for step in plan.payload["steps"]]
        if len(tools) != len(set(tools)):
            errors.append("model_route_duplicate_tool")
        if any(tool not in binding_tool_ids for tool in tools):
            errors.append("model_route_tool_not_bound")
        return errors

    def as_payload(self, plan):
        return dict(plan.payload)

    def profile_id(self, plan):
        return plan.payload["profile_id"]

    def profile_hash(self, plan):
        return sha256_json(plan.payload).split(":", 1)[1]

    def ordered_tool_ids(self, plan):
        return tuple(step["tool_id"] for step in plan.payload["steps"])


ADAPTER: RoutingAdmissionPlanAdapter = FakeRoutingAdmissionPlanAdapter()


def _artifact() -> SourcingModelArtifactIdentity:
    return SourcingModelArtifactIdentity(
        repository="leadpoet/Sourcing_model",
        branch="leadpoet-lab",
        commit_sha="1" * 40,
        artifact_uri="s3://research-lab/sourcing-model/branches/leadpoet-lab/current.json",
        model_artifact_hash=HASH_A,
        manifest_hash=HASH_B,
        routing_contract_hash=HASH_C,
        routing_catalog_hash=HASH_D,
        routing_policy_hash=HASH_E,
        feature_schema_hash=HASH_F,
        verifier_contract_hash=HASH_0,
    )


def _frozen() -> tuple[FrozenRoutingInput, dict[str, bool]]:
    labels = {"unit:cal-1": True, "unit:cal-2": False, "unit:hold-1": True, "unit:hold-2": True}
    features = ("company.country.us", "company.industry.manufacturing", "icp.company_size.201_1000")
    frozen = FrozenRoutingInput(
        segment_ref="icp_segment:manufacturing-us-midmarket",
        signal_type="hiring",
        features=features,
        feature_set_hash=HASH_F,
        calibration_unit_refs=("unit:cal-1", "unit:cal-2"),
        holdout_unit_refs=("unit:hold-1", "unit:hold-2"),
        gold_label_set_hash=sha256_json({"labels": sorted(labels.items())}),
    )
    return frozen, labels


def _binding(binding_id: str, tool_id: str, provider_id: str, lineage: str) -> ProviderBindingIdentity:
    return ProviderBindingIdentity(
        binding_id=binding_id,
        provider_id=provider_id,
        tool_id=tool_id,
        source_lineage_id=lineage,
        adapter_version="adapter:v1",
        manifest_hash=HASH_A,
        capability_hash=HASH_B,
        execution_contract_hash=HASH_C,
        cost_model_hash=HASH_D,
    )


def _profile(profile_id: str, tools: tuple[str, ...]) -> LabRoutingProfile:
    return LabRoutingProfile(
        profile_payload={
            "profile_id": profile_id,
            "profile_version": "v1",
            "feature_set_hash": HASH_F,
            "steps": [{"tool_id": tool} for tool in tools],
        }
    )


def _spec() -> tuple[RoutingExperimentSpec, dict[str, bool]]:
    frozen, labels = _frozen()
    sd = _binding("binding:sd", "intent.jobs.scrapingdog", "scrapingdog", "lineage:jobs.scrapingdog")
    bb = _binding("binding:bb", "intent.jobs.bloomberry", "bloomberry", "lineage:jobs.bloomberry")
    baseline = _profile("profile:baseline", (sd.tool_id,))
    candidate = _profile("profile:bloomberry-first", (bb.tool_id, sd.tool_id))
    spec = RoutingExperimentSpec(
        experiment_id="experiment:hiring-manufacturing-us",
        signal_type="hiring",
        artifact=_artifact(),
        frozen_input=frozen,
        provider_bindings=(sd, bb),
        variants=(baseline, candidate),
        baseline_profile_id="profile:baseline",
        gates=RoutingEvaluationGates(
            min_calibration_precision=1.0,
            min_holdout_precision=1.0,
            min_holdout_recall=0.5,
            max_holdout_no_signal_credit_microunits=100,
            min_marginal_verified_positives_per_credit=0.01,
            intent_release_policy_hash=HASH_E,
        ),
    )
    return spec, labels


def _receipt(binding: ProviderBindingIdentity, unit_ref: str, outcome: ProviderOutcome, *, execution_mode: str = ReceiptExecutionMode.FIXTURE.value) -> ProviderReceipt:
    fingerprint = sha256_json({"experiment_request": "intent-route-v1", "tool_id": binding.tool_id, "unit_ref": unit_ref})
    payload = {
        "binding_id": binding.binding_id,
        "tool_id": binding.tool_id,
        "binding_version": binding.adapter_version,
        "source_lineage_id": binding.source_lineage_id,
        "unit_ref": unit_ref,
        "request_fingerprint": fingerprint,
        "outcome": outcome.value,
        "evidence_hash": HASH_E,
        "credit_microunits": 10,
        "latency_ms": 25,
        "execution_mode": execution_mode,
    }
    return ProviderReceipt(receipt_ref="provider_receipt:" + sha256_json(payload).split(":", 1)[1][:16], **payload)


def test_hash_boundary_is_explicit_and_reversible() -> None:
    raw = "a" * 64
    assert lab_hash_to_model(model_hash_to_lab(raw)) == raw
    with pytest.raises(RoutingExperimentError):
        model_hash_to_lab(HASH_A)


def test_artifact_and_model_payload_are_exact_branch_bound() -> None:
    spec, _ = _spec()
    assert validate_sourcing_model_artifact_identity(_artifact()) == []
    assert validate_routing_experiment_spec(spec, adapter=ADAPTER) == []
    assert "artifact_branch_must_be_leadpoet_lab" in validate_sourcing_model_artifact_identity(replace(_artifact(), branch="main"))
    payload = {**spec.variants[0].profile_payload, "steps": [{"tool_id": "unknown"}]}
    tampered = replace(spec, variants=(replace(spec.variants[0], profile_payload=payload), *spec.variants[1:]))
    assert "model_route_tool_not_bound" in validate_routing_experiment_spec(tampered, adapter=ADAPTER)


def test_feature_ids_keep_model_namespace_and_hash_unchanged() -> None:
    spec, _ = _spec()
    bad_features = replace(spec.frozen_input, features=("country.us",))
    assert "feature_ids_must_use_model_namespace" in validate_routing_experiment_spec(replace(spec, frozen_input=bad_features), adapter=ADAPTER)
    assert spec.frozen_input.feature_set_hash == HASH_F


def test_live_measured_lab_requires_explicit_budget_and_rollup() -> None:
    spec, labels = _spec()
    live = replace(spec, allow_live_credit_spend=True, credit_budget=ExperimentCreditBudget(1000, {"binding:sd": 500, "binding:bb": 500}))
    assert validate_routing_experiment_spec(live, adapter=ADAPTER) == []

    def runner(binding, unit_ref):
        return _receipt(binding, unit_ref, ProviderOutcome.SOURCE_MISS, execution_mode=ReceiptExecutionMode.MEASURED_LAB.value)

    with pytest.raises(RoutingExperimentError, match="authoritative_billing_rollup"):
        evaluate_routing_experiment(live, gold_labels=labels, runner=runner, adapter=ADAPTER)

    store = ProviderReceiptStore()
    evaluation = evaluate_routing_experiment(
        live,
        gold_labels=labels,
        runner=runner,
        adapter=ADAPTER,
        receipt_store=store,
        authoritative_billing_rollup=lambda repository: {
            "total_credit_microunits": sum(
                receipt.credit_microunits
                for key in repository.repository.keys()
                if (receipt := repository.repository.get(key)) is not None
            )
        },
    )
    assert evaluation.live_credit_spend is True


def test_llm_proposal_is_payload_only_and_cannot_self_promote() -> None:
    spec, _ = _spec()
    proposal = LLMRoutingProfileProposal(proposal_id="proposal:one", experiment_id=spec.experiment_id, proposer_model_ref="model:router-proposer", feature_set_hash=spec.frozen_input.feature_set_hash, proposed_profile=spec.variants[1], rationale_hash=HASH_F, promoted=True)
    admitted = admit_llm_routing_profile_proposal(proposal, spec, adapter=ADAPTER)
    assert admitted.state == "rejected" and admitted.promoted is False
    valid = replace(proposal, promoted=False)
    assert admit_llm_routing_profile_proposal(valid, spec, adapter=ADAPTER).state == "validated"


def test_provider_receipts_are_typed_content_addressed_and_durable(tmp_path) -> None:
    spec, _ = _spec()
    binding = spec.provider_bindings[0]
    receipt = _receipt(binding, "unit:cal-1", ProviderOutcome.SOURCE_MISS)
    assert validate_provider_receipt(receipt) == []
    key = provider_receipt_key(tool_id=binding.tool_id, binding_version=binding.adapter_version, request_fingerprint=receipt.request_fingerprint, unit_ref="unit:cal-1", source_lineage_id="ignored")
    path = tmp_path / "receipts.jsonl"
    store = ProviderReceiptStore(JsonlProviderReceiptRepository(path))
    store.put(key, receipt)
    assert ProviderReceiptStore(JsonlProviderReceiptRepository(path)).get(key) == receipt
    alternate_payload = {**receipt.to_dict(), "outcome": ProviderOutcome.VERIFIED.value}
    alternate_payload["receipt_ref"] = "provider_receipt:" + sha256_json({key: value for key, value in alternate_payload.items() if key != "receipt_ref"}).split(":", 1)[1][:16]
    with pytest.raises(RoutingExperimentError, match="key collision"):
        store.put(key, ProviderReceipt.from_mapping(alternate_payload))


def test_shared_calibration_holdout_runs_reuse_receipts() -> None:
    spec, labels = _spec()
    calls = []
    outcomes = {
        ("intent.jobs.scrapingdog", "unit:cal-1"): ProviderOutcome.VERIFIED,
        ("intent.jobs.scrapingdog", "unit:cal-2"): ProviderOutcome.SOURCE_MISS,
        ("intent.jobs.scrapingdog", "unit:hold-1"): ProviderOutcome.SOURCE_MISS,
        ("intent.jobs.scrapingdog", "unit:hold-2"): ProviderOutcome.SOURCE_MISS,
        ("intent.jobs.bloomberry", "unit:cal-1"): ProviderOutcome.SOURCE_MISS,
        ("intent.jobs.bloomberry", "unit:cal-2"): ProviderOutcome.SOURCE_MISS,
        ("intent.jobs.bloomberry", "unit:hold-1"): ProviderOutcome.VERIFIED,
        ("intent.jobs.bloomberry", "unit:hold-2"): ProviderOutcome.SOURCE_MISS,
    }

    def runner(binding, unit_ref):
        calls.append((binding.tool_id, unit_ref))
        return _receipt(binding, unit_ref, outcomes[(binding.tool_id, unit_ref)])

    store = ProviderReceiptStore()
    evaluation = evaluate_routing_experiment(spec, gold_labels=labels, runner=runner, adapter=ADAPTER, receipt_store=store)
    assert evaluation.selected_profile_id == "profile:bloomberry-first"
    selected = next(item for item in evaluation.variants if item.variant_id == evaluation.selected_profile_id)
    assert selected.passed and selected.holdout.unique_rescue_count == 1
    assert selected.holdout.no_signal_credit_microunits == 20
    assert evaluation.provider_cache_hits > 0
    assert evaluation.provider_cache_misses == len(calls)


def test_promotion_preserves_exact_model_hash_and_artifact_lineage() -> None:
    spec, labels = _spec()

    def runner(binding, unit_ref):
        positive = binding.tool_id == "intent.jobs.bloomberry" and unit_ref == "unit:hold-1"
        if binding.tool_id == "intent.jobs.scrapingdog" and unit_ref == "unit:cal-1":
            positive = True
        return _receipt(binding, unit_ref, ProviderOutcome.VERIFIED if positive else ProviderOutcome.SOURCE_MISS)

    evaluation = evaluate_routing_experiment(spec, gold_labels=labels, runner=runner, adapter=ADAPTER)
    promotion = promote_routing_profile_to_lab(spec, evaluation, adapter=ADAPTER)
    assert promotion.target_branch == "leadpoet-lab" and promotion.production_activation is False
    assert promotion.profile_hash.startswith("sha256:")
    assert verify_lab_routing_artifact_lineage(spec=spec, promotion=promotion) == []


def test_adapter_failure_is_not_a_negative_signal() -> None:
    spec, labels = _spec()

    def runner(binding, unit_ref):
        return _receipt(binding, unit_ref, ProviderOutcome.ADAPTER_FAILURE)

    evaluation = evaluate_routing_experiment(spec, gold_labels=labels, runner=runner, adapter=ADAPTER)
    baseline = next(item for item in evaluation.variants if item.variant_id == spec.baseline_profile_id)
    assert baseline.holdout.adapter_failure_count == len(spec.frozen_input.holdout_unit_refs)
    assert baseline.holdout.predicted_positive_count == 0
    assert baseline.holdout.false_negative_count == 2
