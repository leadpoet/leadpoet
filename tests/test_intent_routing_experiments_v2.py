from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ExperimentCreditBudget,
    InMemoryProviderReceiptRepository,
    ProviderBindingIdentity,
    ProviderOutcome,
    ProviderReceipt,
    ProviderReceiptStore,
    PinnedSourcingModelRoutingAdapter,
    ReceiptExecutionMode,
    RoutingDecisionReceiptStore,
    RoutingEvaluationGates,
    RoutingExperimentError,
    RoutingExperimentV2Input,
    RoutingExperimentV2Spec,
    RoutingExperimentV2Variant,
    IsolatedRoutingAdapter,
    RoutingExperimentV2AdapterFactory,
    RoutingPlanStepBudget,
    SourceAddProvenance,
    SourcingModelArtifactIdentity,
    evaluate_routing_experiment_v2,
    promote_routing_experiment_v2_to_lab,
    validate_routing_experiment_v2_spec,
)


H = lambda char: "sha256:" + char * 64
FEATURES = {"schema_version": "routing-feature-set:v1", "features": ["company.country.us", "icp.company_size.201_500"]}
FEATURE_HASH = sha256_json(FEATURES)


def _authority(*, model_hash: str, commit: str, route_hashes: dict[str, str]) -> dict[str, str]:
    payload = {
        "model_artifact_hash": model_hash,
        "git_commit_sha": commit,
        "image_digest": "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "1" * 64,
        "config_hash": H("1"),
        "component_registry_version": "components-v2",
        "scoring_adapter_version": "adapter-v2",
        "manifest_uri": "s3://research-lab/model/manifest.json",
        "signature_ref": "kms-signature:model-v2",
        "build_id": "build-v2",
        **route_hashes,
    }
    payload["manifest_hash"] = sha256_json(payload)
    return payload


def _artifact(char: str = "a", commit_char: str = "1") -> tuple[SourcingModelArtifactIdentity, dict[str, str]]:
    commit = commit_char * 40
    route_hashes = {
        "routing_contract_hash": H("c"),
        "routing_catalog_hash": H("d"),
        "routing_policy_hash": H("e"),
        "feature_schema_hash": H("f"),
        "verifier_contract_hash": H("0"),
    }
    model_hash = H(char)
    authority = _authority(model_hash=model_hash, commit=commit, route_hashes=route_hashes)
    artifact = SourcingModelArtifactIdentity(
        repository="leadpoet/Sourcing_model",
        branch="leadpoet-lab",
        commit_sha=commit,
        artifact_uri=f"s3://research-lab/model/{char}.json",
        model_artifact_hash=model_hash,
        manifest_hash=authority["manifest_hash"],
        **route_hashes,
    )
    return artifact, authority


@dataclass(frozen=True)
class Plan:
    payload: dict
    tools: tuple[str, ...]


class FakeV2Adapter:
    """Fake exact model contract. It supports both model stages."""

    def __init__(self, manifests: dict[str, str], *, available: bool = True):
        self.manifests = manifests
        self.available = available

    def parse_feature_set(self, payload):
        return payload

    def validate_feature_set(self, feature_set, *, expected_hash, expected_features):
        errors = []
        if sha256_json(feature_set) != f"sha256:{expected_hash}":
            errors.append("feature_hash")
        if tuple(feature_set["features"]) != tuple(expected_features):
            errors.append("feature_payload")
        return errors

    def validate_artifact_identity(self, artifact):
        return [] if self.available else ["artifact_unavailable"]

    def parse_variant_payload(self, payload, *, stage):
        if not self.available:
            raise RuntimeError("model unavailable")
        if payload.get("stage") != stage:
            raise ValueError("stage mismatch")
        return payload

    def validate_variant_payload(self, payload, *, stage, feature_set, binding_tool_ids, binding_source_lineages):
        del feature_set, binding_source_lineages
        errors = []
        for tool in payload.get("tools", ()):
            if tool not in binding_tool_ids:
                errors.append(f"model_tool_not_bound:{tool}")
        return errors

    def variant_tool_descriptors(self, payload, *, stage):
        return tuple(
            {
                "tool_id": tool,
                "stage": stage,
                "source_add": tool.startswith(("candidate.source_add.", "intent.source_add.")),
                "manifest_hash": payload.get("manifest_hashes", {}).get(tool, ""),
                "request_hash": payload.get("request_hashes", {}).get(tool, ""),
            }
            for tool in payload.get("tools", ())
        )

    def lookup_tool_descriptor(self, tool_id, *, stage):
        manifest = self.manifests.get(tool_id)
        if manifest is None:
            return None
        return {"tool_id": tool_id, "stage": stage, "source_add": tool_id.startswith(("candidate.source_add.", "intent.source_add.")), "manifest_hash": manifest}

    def validate_provider_binding(self, binding, *, stage):
        del stage
        expected = self.manifests.get(binding.tool_id)
        return [] if expected == binding.manifest_hash else [f"manifest:{binding.tool_id}"]

    def compile_variant(self, payload, *, stage, feature_set, available_tools, remaining_seconds, remaining_calls, credit_cap):
        del stage, feature_set, remaining_seconds, remaining_calls, credit_cap
        tools = tuple(tool for tool in payload.get("tools", ()) if available_tools.get(tool, False))
        return Plan({"feature_set_sha256": FEATURE_HASH.split(":", 1)[1], "steps": list(tools), "skipped": payload.get("skipped", {})}, tools)

    def execute_plan(self, plan, invoke):
        results = []
        for tool in plan.tools:
            result = invoke(tool)
            results.append(result)
            if result.outcome == ProviderOutcome.VERIFIED.value:
                break
        return results, any(item.outcome == ProviderOutcome.VERIFIED.value for item in results)

    def plan_as_payload(self, plan):
        return dict(plan.payload)

    def parse_plan(self, payload):
        return Plan(dict(payload), tuple(payload.get("steps", ())))

    def plan_hash(self, plan):
        return sha256_json(plan.payload)

    def route_hash(self, plan):
        return sha256_json({"steps": list(plan.tools)})

    def plan_step_budgets(self, plan):
        return tuple(RoutingPlanStepBudget(tool_id=tool, execution_mode="invoke", max_calls=2, timeout_seconds=1, credit_microunits=10) for tool in plan.tools)

    def plan_decision_projection(self, plan):
        return {"attempted_tool_ids": plan.tools, "skipped_tool_reasons": plan.payload.get("skipped", {}), "outcome_reasons": {}}


def _binding(binding_id: str, tool_id: str, manifest: str) -> ProviderBindingIdentity:
    return ProviderBindingIdentity(
        binding_id=binding_id,
        provider_id="new" if ".source_add." in tool_id else "provider",
        tool_id=tool_id,
        source_lineage_id=f"lineage.{tool_id}",
        adapter_version="adapter-v2",
        manifest_hash=manifest,
        capability_hash=H("1"),
        execution_contract_hash=H("2"),
        cost_model_hash=H("3"),
    )


def _spec(stage: str = "intent_evidence", *, with_source_add: bool = False, two_artifacts: bool = False, change_kind: str = "route_only"):
    tool = "intent.baseline" if stage == "intent_evidence" else "candidate.baseline"
    source_tool = f"{stage.split('_')[0]}.source_add.new"  # model-owned ID in the fake contract
    manifest = H("4")
    source_request = None
    source_manifest = ""
    if with_source_add:
        from gateway.research_lab.provider_capabilities import _normalize_source_add_v8_registration, _source_add_binding_manifest
        provider_id = "new"
        registration = _normalize_source_add_v8_registration({
            "provider_id": provider_id,
            "stage": stage,
            "priority": 80 if stage == "candidate_acquisition" else 35,
            "capabilities": ["candidate.provider_discovery"] if stage == "candidate_acquisition" else ["intent.provider_evidence"],
            "execution_mode": "invoke", "idempotency": "idempotent", "cost_class": "metered", "unit_cost": 0.1,
            "max_calls": 1, "max_results": 100 if stage == "candidate_acquisition" else 1,
            "timeout_seconds": 60.0 if stage == "candidate_acquisition" else 30.0,
            "intent_categories": [], "evidence_types": ["provider_database"] if stage == "candidate_acquisition" else ["external"],
            "category_contracts": [], "binding_requirements": [],
            "best_for": ["icp.structured_eligible"] if stage == "candidate_acquisition" else ["intent.general"],
            "avoid_when": [], "best_for_description": "Approved source", "avoid_when_description": "Avoid unavailable source",
        })
        source_request = {
            "schema_version": "leadpoet.routerverse_source_incorporation.v3", "provider_id": provider_id,
            "provider_alias": provider_id, "stage": stage, "tool_id": f"{'candidate' if stage == 'candidate_acquisition' else 'intent'}.source_add.{provider_id}",
            "registration_symbol": "sourcing_model/routing/runtime.py::SOURCE_ADD_ROUTING_REGISTRATIONS",
            "registration_type": "SourceAddRoutingRegistration", "provisioning_provenance_sha256": "a" * 64,
            "legacy_v7_manifest_sha256": "a" * 64, "binding_manifest": _source_add_binding_manifest(registration),
            **{field: (list(registration[field]) if isinstance(registration[field], tuple) else registration[field]) for field in registration},
            "runtime_binding_id": provider_id,
        }
        source_manifest = "sha256:" + str(registration["manifest_sha256"])
    bindings = [_binding("baseline", tool, manifest)]
    tools = [tool]
    if with_source_add:
        bindings.append(_binding("source", source_tool, source_manifest))
        tools.append(source_tool)
    artifact, authority = _artifact("a", "1")
    candidate_artifact, candidate_authority = _artifact("b", "2")
    payload = {"stage": stage, "tools": tools, "manifest_hashes": {tool: manifest, source_tool: source_manifest}}
    provenance = ()
    if with_source_add:
        provenance = (SourceAddProvenance("source-add-request", sha256_json(source_request), source_tool, stage, source_manifest, (candidate_artifact if two_artifacts else artifact).commit_sha, True, source_request),)
    baseline = RoutingExperimentV2Variant("baseline", stage, artifact, {"stage": stage, "tools": [tool], "manifest_hashes": {tool: manifest}}, ("baseline",), "route_only", (), authority)
    candidate = RoutingExperimentV2Variant(
        "candidate",
        stage,
        candidate_artifact if two_artifacts else artifact,
        payload,
        tuple(item.binding_id for item in bindings),
        "tool_and_route" if with_source_add else change_kind,
        provenance,
        candidate_authority if two_artifacts else authority,
        (source_tool,) if with_source_add else (),
    )
    labels = {"cal-1": True, "hold-1": False}
    inp = RoutingExperimentV2Input(stage, FEATURE_HASH, FEATURES, ("cal-1",), ("hold-1",), sha256_json({"labels": sorted(labels.items())}), "HIRING" if stage == "intent_evidence" else "")
    spec = RoutingExperimentV2Spec(
        experiment_id=f"v2-{stage}", input=inp, variants=(baseline, candidate), baseline_variant_id="baseline",
        provider_bindings=tuple(bindings), credit_budget=ExperimentCreditBudget(1000, {item.binding_id: 1000 for item in bindings}),
        gates=RoutingEvaluationGates(0.0, 0.0, 0.0, 1000, 0.0, H("e")),
    )
    adapters = {"baseline": FakeV2Adapter({tool: manifest, source_tool: source_manifest}), "candidate": FakeV2Adapter({tool: manifest, source_tool: source_manifest})}
    return spec, adapters, labels, tool, source_tool


def _runner(binding, unit_ref, request_fingerprint):
    verified = unit_ref == "cal-1" and binding.tool_id.endswith("baseline")
    identity = {
        "binding_id": binding.binding_id, "tool_id": binding.tool_id, "binding_version": binding.adapter_version,
        "source_lineage_id": binding.source_lineage_id, "unit_ref": unit_ref,
        "request_fingerprint": request_fingerprint, "outcome": "verified" if verified else "source_miss",
        "evidence_hash": sha256_json({"unit": unit_ref, "tool": binding.tool_id}), "credit_microunits": 10,
        "latency_ms": 2, "execution_mode": "fixture",
    }
    identity["receipt_ref"] = "provider_receipt:" + sha256_json({key: value for key, value in identity.items() if key != "receipt_ref"}).split(":", 1)[1][:16]
    return identity


def test_v2_intent_route_only_and_combined_source_add_run_with_decision_receipts():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence", with_source_add=True)
    assert validate_routing_experiment_v2_spec(spec, adapters=adapters) == []
    decisions = RoutingDecisionReceiptStore()
    evaluation = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=_runner, adapters=adapters, decision_store=decisions, require_isolation=False)
    assert evaluation.selected_variant_id == "baseline"
    assert evaluation.decision_receipt_refs
    assert all(item.stage == "intent_evidence" for item in evaluation.variants)
    assert all(set(receipt.attempted_tool_ids) <= {"intent.baseline", "intent.source_add.new"} for receipt in decisions.values())
    assert any(dict(receipt.skipped_tool_reasons).get("intent.source_add.new") == "model_route_stopped" for receipt in decisions.values())
    tool_only_payload = dict(spec.variants[1].routing_payload)
    tool_only_payload["tools"] = ["intent.baseline"]
    tool_only_payload["manifest_hashes"] = {"intent.baseline": H("4")}
    tool_only = replace(spec.variants[1], variant_id="tool-only", routing_payload=tool_only_payload, change_kind="tool_only")
    tool_only_spec = replace(spec, experiment_id="v2-tool-only", variants=(spec.variants[0], tool_only))
    assert validate_routing_experiment_v2_spec(tool_only_spec, adapters={"baseline": adapters["baseline"], "tool-only": adapters["candidate"]}) == []
    assert promote_routing_experiment_v2_to_lab(evaluation).startswith("sha256:")


@pytest.mark.parametrize("stage", ["candidate_acquisition", "intent_evidence"])
def test_v2_supports_stage_specific_default_and_icp_payloads(stage):
    spec, adapters, labels, tool, _source_tool = _spec(stage)
    payload = dict(spec.variants[1].routing_payload)
    payload["profile_scope"] = "default"
    broad = RoutingExperimentV2Variant("broad", stage, spec.variants[0].artifact, payload, ("baseline",), "route_only", (), spec.variants[0].artifact_authority_manifest)
    spec = RoutingExperimentV2Spec(spec.experiment_id + "-broad", spec.input, (spec.variants[0], broad), "baseline", spec.provider_bindings, spec.credit_budget, spec.gates)
    assert validate_routing_experiment_v2_spec(spec, adapters={"baseline": adapters["baseline"], "broad": adapters["candidate"]}) == []
    payload["profile_scope"] = "icp.company_size.201_500"
    assert payload["tools"] == [tool]


def test_v2_rejects_unregistered_tool_wrong_manifest_and_missing_binding():
    spec, adapters, labels, tool, _source_tool = _spec("intent_evidence")
    bad_payload = dict(spec.variants[1].routing_payload)
    bad_payload["tools"] = ["intent.unregistered"]
    bad = RoutingExperimentV2Variant("bad", "intent_evidence", spec.variants[0].artifact, bad_payload, ("baseline",), "route_only", (), spec.variants[0].artifact_authority_manifest)
    bad_spec = RoutingExperimentV2Spec("bad", spec.input, (spec.variants[0], bad), "baseline", spec.provider_bindings, spec.credit_budget, spec.gates)
    assert any("model_tool_not_bound" in error for error in validate_routing_experiment_v2_spec(bad_spec, adapters={"baseline": adapters["baseline"], "bad": adapters["candidate"]}))
    wrong = _binding("baseline", tool, H("9"))
    wrong_spec = RoutingExperimentV2Spec("wrong", spec.input, spec.variants, "baseline", (wrong,), spec.credit_budget, spec.gates)
    assert any("manifest" in error for error in validate_routing_experiment_v2_spec(wrong_spec, adapters=adapters))
    unknown = _binding("unknown", "intent.unregistered", H("8"))
    unknown_payload = dict(spec.variants[1].routing_payload)
    unknown_payload["tools"] = ["intent.unregistered"]
    unknown_variant = RoutingExperimentV2Variant("unknown", "intent_evidence", spec.variants[0].artifact, unknown_payload, ("unknown",), "route_only", (), spec.variants[0].artifact_authority_manifest)
    unknown_spec = RoutingExperimentV2Spec("unknown", spec.input, (spec.variants[0], unknown_variant), "baseline", (spec.provider_bindings[0], unknown), spec.credit_budget, spec.gates)
    assert any("manifest:intent.unregistered" in error for error in validate_routing_experiment_v2_spec(unknown_spec, adapters={"baseline": adapters["baseline"], "unknown": adapters["candidate"]}))


def test_v2_fail_closed_on_timeout_malformed_retry_and_budget():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    calls = []

    def failing_runner(binding, unit, request):
        calls.append((binding.tool_id, unit, request))
        raise TimeoutError("provider timeout")

    evaluation = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=failing_runner, adapters=adapters, require_isolation=False)
    assert all(item.holdout.adapter_failure_count >= 1 for item in evaluation.variants)
    assert len({request for _tool_id, _unit, request in calls}) == len(calls)
    def malformed_runner(binding, unit, request):
        del binding, unit, request
        return {}
    malformed = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=malformed_runner, adapters=adapters, require_isolation=False)
    assert all(item.holdout.adapter_failure_count >= 1 for item in malformed.variants)
    class RetryAdapter(FakeV2Adapter):
        def execute_plan(self, plan, invoke):
            results = [invoke(plan.tools[0])]
            if results[0].outcome == ProviderOutcome.ADAPTER_FAILURE.value:
                results.append(invoke(plan.tools[0]))
            return results, any(item.outcome == ProviderOutcome.VERIFIED.value for item in results)
    retry_calls = []
    def retry_runner(binding, unit, request):
        retry_calls.append(request)
        if len(retry_calls) == 1:
            raise TimeoutError("retryable")
        return _runner(binding, unit, request)
    retry_adapters = {key: RetryAdapter(value.manifests) for key, value in adapters.items()}
    retry = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=retry_runner, adapters=retry_adapters, require_isolation=False)
    assert retry.selected_variant_id in {"baseline", "candidate"}
    assert len(retry_calls) > len(set(retry_calls)) or len(retry_calls) >= 2
    unavailable_spec_base, unavailable_adapters, unavailable_labels, _unused_tool, _unused_source = _spec("intent_evidence", with_source_add=True)
    unavailable_spec = replace(unavailable_spec_base, availability={"candidate": {"intent.baseline": False, "intent.source_add.new": True}})
    unavailable_calls = []
    def unavailable_runner(binding, unit, request):
        unavailable_calls.append(binding.tool_id)
        return _runner(binding, unit, request)
    unavailable_decisions = RoutingDecisionReceiptStore()
    evaluate_routing_experiment_v2(unavailable_spec, gold_labels=unavailable_labels, runner=unavailable_runner, adapters=unavailable_adapters, decision_store=unavailable_decisions, require_isolation=False)
    candidate_decisions = [item for item in unavailable_decisions.values() if item.variant_id == "candidate"]
    assert candidate_decisions and all("intent.baseline" not in item.attempted_tool_ids for item in candidate_decisions)
    tiny = RoutingExperimentV2Spec(spec.experiment_id + "-budget", spec.input, spec.variants, spec.baseline_variant_id, spec.provider_bindings, ExperimentCreditBudget(1, {"baseline": 1}), spec.gates)
    budget_calls = []
    def budget_runner(binding, unit, request):
        budget_calls.append((binding.tool_id, unit, request))
        return _runner(binding, unit, request)
    with pytest.raises(RoutingExperimentError, match="budget"):
        evaluate_routing_experiment_v2(tiny, gold_labels=labels, runner=budget_runner, adapters=adapters, require_isolation=False)
    assert budget_calls == []


def test_v2_cache_isolation_by_variant_and_artifact_and_two_artifacts():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence", with_source_add=True, two_artifacts=True)
    repository = InMemoryProviderReceiptRepository()
    store = ProviderReceiptStore(repository)
    handles = {
        "baseline": IsolatedRoutingAdapter(adapters["baseline"], "proc-a", sha256_json({"model_artifact_hash": spec.variants[0].artifact.model_artifact_hash, "manifest_hash": spec.variants[0].artifact.manifest_hash, "commit_sha": spec.variants[0].artifact.commit_sha})),
        "candidate": IsolatedRoutingAdapter(adapters["candidate"], "proc-b", sha256_json({"model_artifact_hash": spec.variants[1].artifact.model_artifact_hash, "manifest_hash": spec.variants[1].artifact.manifest_hash, "commit_sha": spec.variants[1].artifact.commit_sha})),
    }
    factory = RoutingExperimentV2AdapterFactory(handles)
    assert set(factory.for_variants(spec.variants)) == {"baseline", "candidate"}
    with pytest.raises(RoutingExperimentError, match="distinct_artifacts"):
        RoutingExperimentV2AdapterFactory({"baseline": handles["baseline"], "candidate": replace(handles["candidate"], process_id="proc-a")})
    evaluation = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=_runner, adapters=handles, receipt_store=store)
    assert evaluation.provider_cache_hits == 0
    keys = tuple(repository.keys())
    assert len(keys) == len(set(keys))
    assert len(keys) >= 4
    assert len({item.artifact_key for item in evaluation.variants}) == 2


def test_v2_live_billing_is_explicit_and_never_promoted(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = RoutingExperimentV2Spec(
        experiment_id=spec.experiment_id + "-live", input=spec.input, variants=spec.variants,
        baseline_variant_id=spec.baseline_variant_id, provider_bindings=spec.provider_bindings,
        credit_budget=spec.credit_budget, gates=spec.gates, availability=spec.availability,
        allow_live_credit_spend=True, receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    durable_path = tmp_path / "leadpoet-v2-receipts-test.jsonl"
    from research_lab.routing_experiments import JsonlProviderReceiptRepository
    store = ProviderReceiptStore(JsonlProviderReceiptRepository(durable_path))
    rollup = lambda _store: {"rollup_id": "billing-v2", "rollup_hash": H("7"), "total_credit_microunits": 40}
    evaluation = evaluate_routing_experiment_v2(measured, gold_labels=labels, runner=_runner, adapters=adapters, receipt_store=store, authoritative_billing_rollup=rollup, require_isolation=False)
    assert evaluation.live_credit_spend is True
    with pytest.raises(RoutingExperimentError, match="cannot_auto_promote"):
        promote_routing_experiment_v2_to_lab(evaluation)


def test_exact_candidate_model_profile_and_plan_contract():
    adapter = PinnedSourcingModelRoutingAdapter.from_model_root("/private/tmp/leadpoet-model-candidate-routing-20260818")
    metadata = adapter.candidate_profiles.candidate_routing_profile_metadata()
    feature_set = adapter.parse_feature_set(FEATURES)
    default_payload = metadata["profiles"][0]
    profile = adapter.parse_variant_payload(default_payload, stage="candidate_acquisition")
    assert adapter.validate_variant_payload(
        profile,
        stage="candidate_acquisition",
        feature_set=feature_set,
        binding_tool_ids=frozenset(item.tool_id for item in adapter.catalog.tools),
        binding_source_lineages={},
    ) == []
    plan = adapter.compile_variant(
        profile,
        stage="candidate_acquisition",
        feature_set=feature_set,
        available_tools={item.tool_id: True for item in adapter.catalog.tools},
        remaining_seconds=900,
        remaining_calls=16,
        credit_cap=32,
    )
    parsed = adapter.parse_plan(adapter.plan_as_payload(plan))
    assert adapter.plan_hash(parsed) == adapter.plan_hash(plan)
    assert adapter.route_hash(plan) != adapter.plan_hash(plan)
    icp_payload = dict(default_payload)
    icp_payload["profile_id"] = "candidate-icp-us"
    icp_payload["is_default"] = False
    icp_payload["required_features"] = ["company.country.us"]
    registry = adapter.candidate_profiles.CandidateRoutingProfileRegistry.from_payload({
        "schema_version": "candidate-routing-registry:v1",
        "registry_version": metadata["registry_version"],
        "profiles": [default_payload, icp_payload],
    })
    assert registry.select(features=feature_set).profile_id == "candidate-icp-us"
