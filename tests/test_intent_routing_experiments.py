from __future__ import annotations

from dataclasses import dataclass, replace
import os

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
    PinnedSourcingModelRoutingAdapter,
    ReceiptExecutionMode,
    RoutingAdmissionPlanAdapter,
    RoutingPlanStepBudget,
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
    validate_routing_evaluation_receipt,
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
FEATURE_SET_PAYLOAD = {
    "schema_version": "routing-feature-set:v1",
    "features": [
        "company.country.us",
        "company.industry.manufacturing",
        "icp.company_size.201_500",
    ],
}
FEATURE_SET_HASH = sha256_json(FEATURE_SET_PAYLOAD)


@dataclass(frozen=True)
class FakeModelPlan:
    payload: dict[str, object]
    steps: tuple[str, ...] = ()
    conditional: tuple[str, ...] = ()
    dormant: tuple[str, ...] = ()


class FakeRoutingAdmissionPlanAdapter:
    """Test double for the pinned model's exact RoutingProfile schema."""

    def parse_profile(self, payload):
        required = {
            "schema_version", "profile_id", "version", "intent_category",
            "required_features", "forbidden_features", "steps", "priority",
            "is_default", "call_cap", "seconds_cap", "credit_cap",
            "challenger_call_cap", "challenger_seconds_cap",
            "challenger_credit_cap", "max_challengers",
        }
        if set(payload) != required:
            raise ValueError("RoutingProfile payload shape drift")
        return FakeModelPlan(dict(payload))

    def parse_feature_set(self, payload):
        if set(payload) != {"schema_version", "features"}:
            raise ValueError("RoutingFeatureSet payload shape drift")
        return FakeModelPlan(dict(payload))

    def validate_feature_set(self, feature_set, *, expected_hash, expected_features):
        errors = []
        if sha256_json(feature_set.payload) != "sha256:" + expected_hash:
            errors.append("model_feature_set_hash_mismatch")
        if tuple(feature_set.payload["features"]) != tuple(expected_features):
            errors.append("model_feature_set_payload_features_mismatch")
        return errors

    def validate_artifact_identity(self, artifact):
        del artifact
        return []

    def validate_profile(self, profile, *, signal_type, feature_set, binding_tool_ids, binding_source_lineages):
        del feature_set, binding_source_lineages
        errors = []
        if profile.payload["intent_category"] != signal_type.upper():
            errors.append("model_profile_intent_category_mismatch")
        tools = [step.get("tool_id") for step in profile.payload["steps"]]
        if len(tools) != len(set(tools)):
            errors.append("model_route_duplicate_tool")
        if any(tool not in binding_tool_ids for tool in tools):
            errors.append("model_route_tool_not_bound")
        return errors

    def profile_as_payload(self, profile):
        return dict(profile.payload)

    def profile_id(self, profile):
        return profile.payload["profile_id"]

    def profile_hash(self, profile):
        return sha256_json(profile.payload).split(":", 1)[1]

    def compile_initial(self, profile, *, signal_type, feature_set, available_tools, remaining_seconds, remaining_calls, credit_cap):
        del signal_type, feature_set, remaining_seconds, remaining_calls, credit_cap
        steps = profile.payload["steps"]
        active = tuple(step["tool_id"] for step in steps if step["phase"] == "primary" and step["tool_id"] in available_tools)
        conditional = tuple(step["tool_id"] for step in steps if step["phase"] == "confirmation" and step["tool_id"] in available_tools)
        dormant = tuple(step["tool_id"] for step in steps if step["phase"] == "challenger" and step["tool_id"] in available_tools)
        return FakeModelPlan({"feature_set_sha256": FEATURE_SET_HASH.split(":", 1)[1], "steps": list(active), "conditional": list(conditional), "dormant": list(dormant)}, active, conditional, dormant)

    def compile_challenger(self, parent_plan, *, profile, feature_set, available_tools, attempted_tool_ids, attempted_source_lineages, remaining_seconds, remaining_calls, credit_cap):
        del feature_set, available_tools, attempted_source_lineages, remaining_seconds, remaining_calls, credit_cap
        remaining = tuple(tool for tool in parent_plan.dormant if tool not in attempted_tool_ids)
        if not remaining:
            raise ValueError("no distinct dormant challenger remains")
        return FakeModelPlan({"feature_set_sha256": FEATURE_SET_HASH.split(":", 1)[1], "steps": [remaining[0]], "conditional": [], "dormant": []}, (remaining[0],), (), ())

    def has_conditional_confirmation(self, plan):
        return bool(plan.conditional)

    def compile_confirmation(self, parent_plan, *, profile, feature_set, available_tools, remaining_seconds, remaining_calls, credit_cap):
        del profile, feature_set, available_tools, remaining_seconds, remaining_calls, credit_cap
        if not parent_plan.conditional:
            raise ValueError("no conditional confirmation remains")
        selected = parent_plan.conditional[0]
        return FakeModelPlan({"feature_set_sha256": FEATURE_SET_HASH.split(":", 1)[1], "steps": [selected], "conditional": [], "dormant": list(parent_plan.dormant)}, (selected,), (), parent_plan.dormant)

    def execute_plan(self, plan, invoke):
        results = [invoke(tool) for tool in plan.steps]
        return results, any(item.outcome == "verified" for item in results)

    def plan_as_payload(self, plan):
        return dict(plan.payload)

    def parse_plan(self, payload):
        return FakeModelPlan(dict(payload), tuple(payload.get("steps") or ()), tuple(payload.get("conditional") or ()), tuple(payload.get("dormant") or ()))

    def plan_hash(self, plan):
        return sha256_json({"payload": plan.payload, "steps": plan.steps, "dormant": plan.dormant}).split(":", 1)[1]

    def plan_step_budgets(self, plan):
        return tuple(
            RoutingPlanStepBudget(
                tool_id=tool_id,
                execution_mode="invoke",
                max_calls=1,
                timeout_seconds=1.0,
                credit_microunits=10,
            )
            for tool_id in plan.steps
        )

    def intent_release_policy_hash(self):
        return HASH_E


class ConfirmationUnavailableAdapter(FakeRoutingAdmissionPlanAdapter):
    def compile_confirmation(self, *args, **kwargs):
        del args, kwargs
        raise ValueError("confirmation unavailable")


class OverBudgetAdapter(FakeRoutingAdmissionPlanAdapter):
    def plan_step_budgets(self, plan):
        return tuple(
            RoutingPlanStepBudget(
                tool_id=tool_id,
                execution_mode="invoke",
                max_calls=1,
                timeout_seconds=1.0,
                credit_microunits=6_000_000,
            )
            for tool_id in plan.steps
        )


class ConfirmationOverBudgetAdapter(OverBudgetAdapter):
    def __init__(self):
        self.confirmation_compiles = 0

    def compile_confirmation(self, *args, **kwargs):
        self.confirmation_compiles += 1
        return super().compile_confirmation(*args, **kwargs)


class NoReleasePolicyAdapter(FakeRoutingAdmissionPlanAdapter):
    intent_release_policy_hash = None


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
    features = ("company.country.us", "company.industry.manufacturing", "icp.company_size.201_500")
    frozen = FrozenRoutingInput(
        segment_ref="icp_segment:manufacturing-us-midmarket",
        signal_type="hiring",
        features=features,
        feature_set_hash=FEATURE_SET_HASH,
        calibration_unit_refs=("unit:cal-1", "unit:cal-2"),
        holdout_unit_refs=("unit:hold-1", "unit:hold-2"),
        gold_label_set_hash=sha256_json({"labels": sorted(labels.items())}),
        feature_set_payload=FEATURE_SET_PAYLOAD,
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


def _profile(
    profile_id: str,
    tools: tuple[str, ...],
    phases: tuple[str, ...] | None = None,
) -> LabRoutingProfile:
    phases = phases or tuple(
        "primary" if index == 0 else "challenger"
        for index in range(len(tools))
    )
    assert len(phases) == len(tools)
    return LabRoutingProfile(
        profile_payload={
            "profile_id": profile_id,
            "schema_version": "routing-profile:v1",
            "version": "v1",
            "intent_category": "HIRING",
            "required_features": [],
            "forbidden_features": [],
            "steps": [
                {
                    "tool_id": tool,
                    "phase": phases[index],
                    "order": index,
                    "source_lineage_id": (
                        "lineage:jobs.scrapingdog"
                        if "scrapingdog" in tool
                        else (
                            "lineage:jobs.bloomberry"
                            if "bloomberry" in tool
                            else "lineage:jobs.confirmation"
                        )
                    ),
                    "required": False,
                }
                for index, tool in enumerate(tools)
            ],
            "priority": 0,
            "is_default": False,
            "call_cap": 4,
            "seconds_cap": 180.0,
            "credit_cap": 1.0,
            "challenger_call_cap": 1,
            "challenger_seconds_cap": 60.0,
            "challenger_credit_cap": 1.0,
            "max_challengers": 1 if len(tools) > 1 else 0,
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


def _confirmation_spec() -> tuple[RoutingExperimentSpec, dict[str, bool]]:
    spec, labels = _spec()
    confirmation_binding = _binding(
        "binding:confirmation",
        "intent.jobs.confirmation",
        "company-search",
        "lineage:jobs.confirmation",
    )
    candidate = _profile(
        "profile:confirmation",
        (
            "intent.jobs.bloomberry",
            "intent.jobs.confirmation",
            "intent.jobs.scrapingdog",
        ),
        phases=("primary", "confirmation", "challenger"),
    )
    return (
        replace(
            spec,
            provider_bindings=(*spec.provider_bindings, confirmation_binding),
            variants=(candidate,),
            baseline_profile_id="profile:confirmation",
        ),
        labels,
    )


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


def _receipt_with_credit(
    binding: ProviderBindingIdentity,
    unit_ref: str,
    outcome: ProviderOutcome,
    credit_microunits: int,
    *,
    execution_mode: str = ReceiptExecutionMode.FIXTURE.value,
) -> ProviderReceipt:
    receipt = _receipt(binding, unit_ref, outcome, execution_mode=execution_mode)
    payload = {**receipt.to_dict(), "credit_microunits": credit_microunits}
    payload.pop("receipt_ref")
    return ProviderReceipt(
        receipt_ref="provider_receipt:" + sha256_json(payload).split(":", 1)[1][:16],
        **payload,
    )


def test_hash_boundary_is_explicit_and_reversible() -> None:
    raw = "a" * 64
    assert lab_hash_to_model(model_hash_to_lab(raw)) == raw
    with pytest.raises(RoutingExperimentError):
        model_hash_to_lab(HASH_A)


def test_pinned_model_loader_fails_closed_when_artifact_is_unavailable(tmp_path) -> None:
    with pytest.raises(RoutingExperimentError, match="runtime_not_found"):
        PinnedSourcingModelRoutingAdapter.from_model_root(tmp_path)


def test_actual_pinned_model_profile_feature_and_admission_contract() -> None:
    # Exact model integration is opt-in because CI does not vendor the model
    # checkout. Run it locally with:
    # LEADPOET_PINNED_SOURCING_MODEL_ROOT=/path/to/Sourcing_model \
    #   pytest -q tests/test_intent_routing_experiments.py \
    #   -k actual_pinned_model
    model_root = os.getenv("LEADPOET_PINNED_SOURCING_MODEL_ROOT", "").strip()
    if not model_root:
        pytest.skip("set LEADPOET_PINNED_SOURCING_MODEL_ROOT for exact model integration")
    adapter = PinnedSourcingModelRoutingAdapter.from_model_root(model_root)
    observed_identity = adapter.observed_artifact_identity()
    bound_artifact = replace(
        _artifact(),
        commit_sha=observed_identity["commit_sha"],
        model_artifact_hash=observed_identity["model_artifact_hash"],
        routing_catalog_hash=observed_identity["routing_catalog_hash"],
        routing_policy_hash=observed_identity["routing_policy_hash"],
    )
    assert adapter.validate_artifact_identity(bound_artifact) == []
    assert "model_artifact_commit_sha_mismatch" in adapter.validate_artifact_identity(
        replace(bound_artifact, commit_sha="2" * 40)
    )
    assert "model_artifact_model_artifact_hash_mismatch" in adapter.validate_artifact_identity(
        replace(bound_artifact, model_artifact_hash=HASH_A)
    )
    feature_payload = {
        "schema_version": "routing-feature-set:v1",
        "features": ["icp.company_size.201_500"],
    }
    feature_set = adapter.parse_feature_set(feature_payload)
    profile_payload = {
        "schema_version": "routing-profile:v1",
        "profile_id": "hiring-lab",
        "version": "v1",
        "intent_category": "HIRING",
        "required_features": [],
        "forbidden_features": [],
        "steps": [
            {
                "tool_id": "intent.source_add.bloomberry_jobs",
                "phase": "primary",
                "order": 0,
                "source_lineage_id": "bloomberry-jobs",
                "required": False,
            },
            {
                "tool_id": "intent.company_search",
                "phase": "confirmation",
                "order": 0,
                "source_lineage_id": "company-search",
                "required": False,
            },
            {
                "tool_id": "intent.jobs_feed",
                "phase": "challenger",
                "order": 0,
                "source_lineage_id": "scrapingdog-jobs",
                "required": False,
            },
        ],
        "priority": 0,
        "is_default": False,
        "call_cap": 4,
        "seconds_cap": 180.0,
        "credit_cap": 10.0,
        "challenger_call_cap": 1,
        "challenger_seconds_cap": 60.0,
        "challenger_credit_cap": 1.0,
        "max_challengers": 1,
    }
    profile = adapter.parse_profile(profile_payload)
    assert adapter.validate_feature_set(
        feature_set,
        expected_hash=feature_set.sha256(),
        expected_features=feature_set.features,
    ) == []
    assert adapter.validate_profile(
        profile,
        signal_type="hiring",
        feature_set=feature_set,
        binding_tool_ids=frozenset(
            {
                "intent.source_add.bloomberry_jobs",
                "intent.company_search",
                "intent.jobs_feed",
            }
        ),
        binding_source_lineages={
            "intent.source_add.bloomberry_jobs": "bloomberry-jobs",
            "intent.company_search": "company-search",
            "intent.jobs_feed": "scrapingdog-jobs",
        },
    ) == []
    plan = adapter.compile_initial(
        profile,
        signal_type="hiring",
        feature_set=feature_set,
        available_tools={
            "intent.source_add.bloomberry_jobs": True,
            "intent.company_search": True,
            "intent.jobs_feed": True,
        },
        remaining_seconds=60.0,
        remaining_calls=4,
        credit_cap=10.0,
    )
    restored = adapter.parse_plan(adapter.plan_as_payload(plan))
    assert adapter.plan_hash(restored) == adapter.plan_hash(plan)
    assert plan.profile_version == "v1"
    assert plan.feature_set_sha256 == feature_set.sha256()
    assert plan.conditional_tool_ids == ("intent.company_search",)
    assert adapter.intent_release_policy_hash().startswith("sha256:")
    assert [item.tool_id for item in adapter.plan_step_budgets(plan)] == [
        "intent.source_add.bloomberry_jobs"
    ]
    confirmation = adapter.compile_confirmation(
        plan,
        profile=profile,
        feature_set=feature_set,
        available_tools={
            "intent.source_add.bloomberry_jobs": True,
            "intent.company_search": True,
            "intent.jobs_feed": True,
        },
        remaining_seconds=60.0,
        remaining_calls=2,
        credit_cap=1.0,
    )
    assert confirmation.active_tool_ids == ("intent.company_search",)
    assert confirmation.parent_plan_sha256 == plan.sha256()
    challenger = adapter.compile_challenger(
        plan,
        profile=profile,
        feature_set=feature_set,
        available_tools={
            "intent.source_add.bloomberry_jobs": True,
            "intent.company_search": True,
            "intent.jobs_feed": True,
        },
        attempted_tool_ids=plan.active_tool_ids,
        attempted_source_lineages=("bloomberry-jobs",),
        remaining_seconds=60.0,
        remaining_calls=2,
        credit_cap=1.0,
    )
    assert challenger.challenge_index == 1
    assert challenger.active_tool_ids == ("intent.jobs_feed",)


def test_primary_hit_runs_model_confirmation_and_retains_typed_receipts() -> None:
    spec, labels = _confirmation_spec()
    calls = []
    store = ProviderReceiptStore()

    def runner(binding, unit_ref):
        calls.append((binding.tool_id, unit_ref))
        outcome = (
            ProviderOutcome.VERIFIED
            if binding.tool_id == "intent.jobs.bloomberry"
            else ProviderOutcome.SOURCE_MISS
        )
        return _receipt(binding, unit_ref, outcome)

    evaluation = evaluate_routing_experiment(
        spec,
        gold_labels=labels,
        runner=runner,
        adapter=ADAPTER,
        receipt_store=store,
    )
    assert any(tool_id == "intent.jobs.confirmation" for tool_id, _unit in calls)
    confirmation_receipts = [
        receipt
        for key in store.repository.keys()
        if (receipt := store.repository.get(key)) is not None
        and receipt.tool_id == "intent.jobs.confirmation"
    ]
    assert confirmation_receipts
    assert all(item.outcome == ProviderOutcome.SOURCE_MISS.value for item in confirmation_receipts)
    assert evaluation.provider_receipt_refs


def test_primary_miss_skips_confirmation_and_uses_bounded_model_challenger() -> None:
    spec, labels = _confirmation_spec()
    calls = []

    def runner(binding, unit_ref):
        calls.append((binding.tool_id, unit_ref))
        outcome = (
            ProviderOutcome.VERIFIED
            if binding.tool_id == "intent.jobs.scrapingdog"
            else ProviderOutcome.SOURCE_MISS
        )
        return _receipt(binding, unit_ref, outcome)

    evaluate_routing_experiment(
        spec,
        gold_labels=labels,
        runner=runner,
        adapter=ADAPTER,
    )
    assert not any(tool_id == "intent.jobs.confirmation" for tool_id, _unit in calls)
    assert any(tool_id == "intent.jobs.scrapingdog" for tool_id, _unit in calls)


def test_confirmation_compile_failure_is_explicit_and_does_not_fall_through() -> None:
    spec, labels = _confirmation_spec()
    calls = []

    def runner(binding, unit_ref):
        calls.append((binding.tool_id, unit_ref))
        return _receipt(binding, unit_ref, ProviderOutcome.VERIFIED)

    with pytest.raises(
        RoutingExperimentError,
        match="model confirmation compilation failed:confirmation unavailable",
    ):
        evaluate_routing_experiment(
            spec,
            gold_labels=labels,
            runner=runner,
            adapter=ConfirmationUnavailableAdapter(),
        )
    assert calls == [("intent.jobs.bloomberry", "unit:cal-1")]


def test_confirmation_provider_failure_is_retained_as_typed_evidence() -> None:
    spec, labels = _confirmation_spec()
    store = ProviderReceiptStore()

    def runner(binding, unit_ref):
        outcome = (
            ProviderOutcome.VERIFIED
            if binding.tool_id == "intent.jobs.bloomberry"
            else ProviderOutcome.ADAPTER_FAILURE
        )
        return _receipt(binding, unit_ref, outcome)

    evaluation = evaluate_routing_experiment(
        spec,
        gold_labels=labels,
        runner=runner,
        adapter=ADAPTER,
        receipt_store=store,
    )
    confirmation = [
        receipt
        for key in store.repository.keys()
        if (receipt := store.repository.get(key)) is not None
        and receipt.tool_id == "intent.jobs.confirmation"
    ]
    assert confirmation
    assert all(item.outcome == ProviderOutcome.ADAPTER_FAILURE.value for item in confirmation)
    assert all(
        item.holdout.adapter_failure_count > 0
        for item in evaluation.variants
    )


def test_pre_call_ledger_rejects_challenger_before_provider_call() -> None:
    spec, labels = _spec()
    candidate = spec.variants[1]
    bounded = replace(spec, variants=(candidate,), baseline_profile_id=candidate.profile_payload["profile_id"])
    calls = []

    def runner(binding, unit_ref):
        calls.append((binding.tool_id, unit_ref))
        return _receipt_with_credit(binding, unit_ref, ProviderOutcome.SOURCE_MISS, 6_000_000)

    with pytest.raises(RoutingExperimentError, match="exceeds_remaining_unit_credit"):
        evaluate_routing_experiment(
            bounded,
            gold_labels=labels,
            runner=runner,
            adapter=OverBudgetAdapter(),
        )
    assert calls == [("intent.jobs.bloomberry", "unit:cal-1")]


def test_pre_call_ledger_applies_remaining_budget_to_confirmation() -> None:
    spec, labels = _confirmation_spec()
    calls = []
    adapter = ConfirmationOverBudgetAdapter()

    def runner(binding, unit_ref):
        calls.append((binding.tool_id, unit_ref))
        return _receipt_with_credit(binding, unit_ref, ProviderOutcome.VERIFIED, 6_000_000)

    with pytest.raises(RoutingExperimentError, match="exceeds_remaining_unit_credit"):
        evaluate_routing_experiment(
            spec,
            gold_labels=labels,
            runner=runner,
            adapter=adapter,
        )
    assert adapter.confirmation_compiles == 1
    assert calls == [("intent.jobs.bloomberry", "unit:cal-1")]


def test_pre_call_ledger_rejects_provider_ceiling_before_runner(tmp_path) -> None:
    spec, labels = _spec()
    live = replace(
        spec,
        allow_live_credit_spend=True,
        credit_budget=ExperimentCreditBudget(100, {"binding:sd": 100, "binding:bb": 5}),
    )
    calls = []

    def runner(binding, unit_ref):
        calls.append((binding.tool_id, unit_ref))
        return _receipt(
            binding,
            unit_ref,
            ProviderOutcome.SOURCE_MISS,
            execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
        )

    store = ProviderReceiptStore(JsonlProviderReceiptRepository(tmp_path / "ceiling.jsonl"))
    with pytest.raises(RoutingExperimentError, match="provider_credit_cap_would_be_exceeded"):
        evaluate_routing_experiment(
            live,
            gold_labels=labels,
            runner=runner,
            adapter=ADAPTER,
            receipt_store=store,
        )
    assert calls and all(tool_id == "intent.jobs.scrapingdog" for tool_id, _unit_ref in calls)


def test_artifact_and_model_payload_are_exact_branch_bound() -> None:
    spec, _ = _spec()
    assert validate_sourcing_model_artifact_identity(_artifact()) == []
    assert validate_routing_experiment_spec(spec, adapter=ADAPTER) == []
    assert "artifact_branch_must_be_leadpoet_lab" in validate_sourcing_model_artifact_identity(replace(_artifact(), branch="main"))
    payload = {**spec.variants[0].profile_payload, "steps": [{"tool_id": "unknown"}]}
    tampered = replace(spec, variants=(replace(spec.variants[0], profile_payload=payload), *spec.variants[1:]))
    assert "model_route_tool_not_bound" in validate_routing_experiment_spec(tampered, adapter=ADAPTER)


def test_mapping_spec_cannot_silently_default_promotion_gates() -> None:
    spec, _ = _spec()
    payload = spec.to_dict()
    payload["gates"] = dict(payload["gates"])
    payload["gates"].pop("intent_release_policy_hash")
    assert "evaluation_gates_must_be_explicit" in validate_routing_experiment_spec(
        payload,
        adapter=ADAPTER,
    )


def test_feature_ids_keep_model_namespace_and_hash_unchanged() -> None:
    spec, _ = _spec()
    bad_features = replace(spec.frozen_input, features=("country.us",))
    assert "feature_ids_must_use_model_namespace" in validate_routing_experiment_spec(replace(spec, frozen_input=bad_features), adapter=ADAPTER)
    assert spec.frozen_input.feature_set_hash == FEATURE_SET_HASH


def test_live_measured_lab_requires_explicit_durable_store_budget_and_rollup(tmp_path) -> None:
    spec, labels = _spec()
    live = replace(spec, allow_live_credit_spend=True, credit_budget=ExperimentCreditBudget(1000, {"binding:sd": 500, "binding:bb": 500}))
    assert validate_routing_experiment_spec(live, adapter=ADAPTER) == []

    def runner(binding, unit_ref):
        return _receipt(binding, unit_ref, ProviderOutcome.SOURCE_MISS, execution_mode=ReceiptExecutionMode.MEASURED_LAB.value)

    with pytest.raises(RoutingExperimentError, match="explicit_durable_receipt_repository"):
        evaluate_routing_experiment(live, gold_labels=labels, runner=runner, adapter=ADAPTER)

    with pytest.raises(RoutingExperimentError, match="requires_durable_receipt_repository"):
        evaluate_routing_experiment(
            live,
            gold_labels=labels,
            runner=runner,
            adapter=ADAPTER,
            receipt_store=ProviderReceiptStore(),
        )

    store = ProviderReceiptStore(JsonlProviderReceiptRepository(tmp_path / "measured-receipts.jsonl"))
    evaluation = evaluate_routing_experiment(
        live,
        gold_labels=labels,
        runner=runner,
        adapter=ADAPTER,
        receipt_store=store,
        authoritative_billing_rollup=lambda repository: {
            "rollup_id": "billing-rollup:test",
            "rollup_hash": HASH_A,
            "total_credit_microunits": sum(
                receipt.credit_microunits
                for key in repository.repository.keys()
                if (receipt := repository.repository.get(key)) is not None
            )
        },
    )
    assert evaluation.live_credit_spend is True
    with pytest.raises(RoutingExperimentError, match="immutable Lab evidence"):
        promote_routing_profile_to_lab(live, evaluation, adapter=ADAPTER)


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


def test_evaluation_receipt_identity_is_recomputed_before_promotion() -> None:
    spec, labels = _spec()

    def runner(binding, unit_ref):
        positive = binding.tool_id == "intent.jobs.bloomberry" and unit_ref == "unit:hold-1"
        if binding.tool_id == "intent.jobs.scrapingdog" and unit_ref == "unit:cal-1":
            positive = True
        return _receipt(binding, unit_ref, ProviderOutcome.VERIFIED if positive else ProviderOutcome.SOURCE_MISS)

    evaluation = evaluate_routing_experiment(spec, gold_labels=labels, runner=runner, adapter=ADAPTER)
    assert validate_routing_evaluation_receipt(evaluation, spec, adapter=ADAPTER) == []
    tampered = replace(evaluation, feature_set_hash=HASH_A)
    assert "evaluation_receipt_id_mismatch" in validate_routing_evaluation_receipt(
        tampered,
        spec,
        adapter=ADAPTER,
    )
    with pytest.raises(RoutingExperimentError, match="invalid evaluation receipt"):
        promote_routing_profile_to_lab(spec, tampered, adapter=ADAPTER)


def test_promotion_fails_closed_without_actual_release_policy_identity() -> None:
    spec, labels = _spec()

    def runner(binding, unit_ref):
        positive = binding.tool_id == "intent.jobs.bloomberry" and unit_ref == "unit:hold-1"
        if binding.tool_id == "intent.jobs.scrapingdog" and unit_ref == "unit:cal-1":
            positive = True
        return _receipt(binding, unit_ref, ProviderOutcome.VERIFIED if positive else ProviderOutcome.SOURCE_MISS)

    adapter = NoReleasePolicyAdapter()
    evaluation = evaluate_routing_experiment(spec, gold_labels=labels, runner=runner, adapter=adapter)
    with pytest.raises(RoutingExperimentError, match="intent_release_policy_identity_unavailable"):
        promote_routing_profile_to_lab(spec, evaluation, adapter=adapter)


def test_adapter_failure_is_not_a_negative_signal() -> None:
    spec, labels = _spec()

    def runner(binding, unit_ref):
        return _receipt(binding, unit_ref, ProviderOutcome.ADAPTER_FAILURE)

    evaluation = evaluate_routing_experiment(spec, gold_labels=labels, runner=runner, adapter=ADAPTER)
    baseline = next(item for item in evaluation.variants if item.variant_id == spec.baseline_profile_id)
    assert baseline.holdout.adapter_failure_count == len(spec.frozen_input.holdout_unit_refs)
    assert baseline.holdout.predicted_positive_count == 0
    assert baseline.holdout.false_negative_count == 2
