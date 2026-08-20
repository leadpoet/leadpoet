from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    COMPANY_FIRST_CONTINUATION_SCHEMA_VERSION,
    CompanyFirstLabReplay,
    ExperimentCreditBudget,
    InMemoryProviderReceiptRepository,
    IsolatedCompanyFirstContinuationAdapter,
    JsonlProviderReceiptRepository,
    JsonlRoutingDecisionReceiptRepository,
    ProviderBindingIdentity,
    ProviderOutcome,
    ProviderReceipt,
    ProviderReceiptStore,
    PinnedSourcingModelRoutingAdapter,
    ReceiptExecutionMode,
    RoutingDecisionReceiptV2,
    RoutingDecisionReceiptStore,
    RoutingCallAuthorization,
    RoutingEvaluationGates,
    RoutingExperimentError,
    RoutingExperimentV2Input,
    RoutingExperimentV2Spec,
    RoutingExperimentV2Variant,
    IsolatedRoutingAdapter,
    RoutingExperimentV2AdapterFactory,
    RoutingPlanStepBudget,
    RoutingWorstCaseVariantBudget,
    SourceAddProvenance,
    SourcingModelArtifactIdentity,
    evaluate_routing_experiment_v2,
    promote_routing_experiment_v2_to_lab,
    replay_company_first_orchestration,
    validate_routing_experiment_v2_spec,
    validate_routing_decision_receipt,
)


H = lambda char: "sha256:" + char * 64
FEATURES = {"schema_version": "routing-feature-set:v1", "features": ["company.country.us", "icp.company_size.201_500"]}
FEATURE_HASH = sha256_json(FEATURES)
EXACT_MODEL_ROOT = "/private/tmp/sourcing-model-main-294ce330"
EXACT_MODEL_SHA = "294ce330efaad72988b7e865450bba36be00eafd"


def _exact_model_root() -> str:
    """Return the requested model checkout, or skip when it is unavailable."""

    model_root = os.getenv(
        "LEADPOET_PINNED_SOURCING_MODEL_ROOT",
        EXACT_MODEL_ROOT,
    ).strip()
    if not model_root or not os.path.isdir(model_root):
        pytest.skip("exact Sourcing_model checkout is unavailable")
    try:
        actual_sha = subprocess.check_output(
            ["git", "-C", model_root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"could not verify exact Sourcing_model SHA: {exc}")
    if actual_sha != EXACT_MODEL_SHA:
        pytest.fail(
            f"exact Sourcing_model checkout must be {EXACT_MODEL_SHA}; "
            f"found {actual_sha}"
        )
    return model_root


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

    @property
    def conditional_tool_ids(self) -> tuple[str, ...]:
        return tuple(self.payload.get("conditional", ()))

    @property
    def dormant_tool_ids(self) -> tuple[str, ...]:
        return tuple(self.payload.get("dormant", ()))


class FakeV2Adapter:
    """Fake exact model contract. It supports both model stages."""

    def __init__(
        self,
        manifests: dict[str, str],
        *,
        available: bool = True,
        registered_source_add_tool_ids: tuple[str, ...] = (),
    ):
        self.manifests = manifests
        self.available = available
        self.registered_source_add_tool_ids_value = tuple(
            registered_source_add_tool_ids
        )

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

    def validate_variant_payload(self, payload, *, stage, feature_set, binding_tool_ids, binding_source_lineages, expected_signal_type=""):
        del feature_set, binding_source_lineages
        errors = []
        if stage == "intent_evidence" and payload.get("signal_type") != expected_signal_type:
            errors.append("model_profile_intent_category_mismatch")
        if stage == "candidate_acquisition" and expected_signal_type:
            errors.append("candidate_signal_type_must_be_empty")
        for tool in payload.get("tools", ()):
            if tool not in binding_tool_ids:
                errors.append(f"model_tool_not_bound:{tool}")
        return errors

    def routing_change_class(self, payload, *, stage):
        del stage
        return payload.get("route_change_class", "custom")

    def routing_identity(self, payload, *, stage, exclude_tool_ids=()):
        normalized = {
            key: value
            for key, value in payload.items()
            if key not in {"route_change_class", "manifest_hashes"}
        }
        excluded = set(exclude_tool_ids)
        normalized["stage"] = stage
        normalized["tools"] = [tool for tool in payload.get("tools", ()) if tool not in excluded]
        return sha256_json(normalized)

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

    def registered_source_add_tool_ids(self, *, stage):
        del stage
        return self.registered_source_add_tool_ids_value

    def lookup_tool_descriptor(self, tool_id, *, stage):
        manifest = self.manifests.get(tool_id)
        if manifest is None:
            return None
        return {"tool_id": tool_id, "stage": stage, "source_add": tool_id.startswith(("candidate.source_add.", "intent.source_add.")), "manifest_hash": manifest}

    def validate_provider_binding(self, binding, *, stage):
        del stage
        expected = self.manifests.get(binding.tool_id)
        return [] if expected == binding.manifest_hash else [f"manifest:{binding.tool_id}"]

    def compile_variant(self, payload, *, stage, feature_set, available_tools, remaining_seconds, remaining_calls, credit_cap, expected_signal_type=""):
        del stage, feature_set, remaining_seconds, remaining_calls, credit_cap
        if payload.get("stage") == "intent_evidence" and payload.get("signal_type") != expected_signal_type:
            raise RoutingExperimentError("model_profile_intent_category_mismatch")
        tools = tuple(tool for tool in payload.get("tools", ()) if available_tools.get(tool, False))
        return Plan({"feature_set_sha256": FEATURE_HASH.split(":", 1)[1], "steps": list(tools), "skipped": payload.get("skipped", {})}, tools)

    # The v2 contract owns all intent waterfall transitions.  The default
    # fixture deliberately has neither a confirmation nor a dormant
    # challenger, but it must still expose the model contract so that the Lab
    # never falls back to a site-owned route.
    def has_conditional_confirmation(self, plan):
        return bool(plan.payload.get("conditional", ()))

    def compile_confirmation(self, parent_plan, *, profile, feature_set, available_tools, remaining_seconds, remaining_calls, credit_cap):
        del profile, feature_set, available_tools, remaining_seconds, remaining_calls, credit_cap
        conditional = tuple(parent_plan.payload.get("conditional", ()))
        if not conditional:
            raise ValueError("no conditional confirmation remains")
        tool = conditional[0]
        return Plan(
            {
                "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                "steps": [tool],
                "conditional": [],
                "dormant": list(parent_plan.payload.get("dormant", ())),
            },
            (tool,),
        )

    def compile_challenger(self, parent_plan, *, profile, feature_set, available_tools, attempted_tool_ids, attempted_source_lineages, remaining_seconds, remaining_calls, credit_cap):
        del profile, feature_set, available_tools, attempted_source_lineages, remaining_seconds, remaining_calls, credit_cap
        dormant = tuple(tool for tool in parent_plan.payload.get("dormant", ()) if tool not in attempted_tool_ids)
        if not dormant:
            raise ValueError("no distinct dormant challenger remains")
        tool = dormant[0]
        return Plan(
            {
                "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                "steps": [tool],
                "conditional": [],
                "dormant": [],
            },
            (tool,),
        )

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

    def considered_tool_ids(self, plan, *, stage):
        del stage
        return tuple(
            dict.fromkeys(
                (
                    *plan.tools,
                    *plan.payload.get("considered", ()),
                    *plan.payload.get("conditional", ()),
                    *plan.payload.get("dormant", ()),
                )
            )
        )

    def plan_decision_projection(self, plan):
        return {"attempted_tool_ids": plan.tools, "skipped_tool_reasons": plan.payload.get("skipped", {}), "outcome_reasons": {}}

    def worst_case_variant_budget(
        self,
        payload,
        *,
        stage,
        feature_set,
        available_tools,
        expected_signal_type="",
    ):
        del stage, feature_set, expected_signal_type
        phases = (
            tuple(payload.get("tools", ())),
            tuple(payload.get("conditional", ())),
            tuple(payload.get("dormant", ()))[:1],
        )
        tool_credit = {}
        total = 0
        for phase in phases:
            for tool in phase:
                if available_tools.get(tool, True) is not True:
                    continue
                tool_credit[tool] = tool_credit.get(tool, 0) + 10
                total += 10
        return RoutingWorstCaseVariantBudget(total, tool_credit)


class FakeArtifactAuthority:
    """Deterministic stand-in for the runtime KMS verification authority."""

    def __init__(self, *, verified=True):
        self.verified = verified
        self.calls = []

    def verify(self, *, artifact, manifest):
        self.calls.append((artifact, manifest))
        return {
            "verified": self.verified,
            "model_artifact_hash": artifact.model_artifact_hash,
            "manifest_hash": artifact.manifest_hash,
            "commit_sha": artifact.commit_sha,
        }


class FakePromotionAuthority:
    authoritative = True

    def __init__(self):
        self.reconcile_calls = []
        self.promote_calls = []

    def reconcile(self, *, spec, evaluation):
        self.reconcile_calls.append((spec, evaluation))
        return {
            "reconciled": True,
            "experiment_hash": spec.experiment_hash(),
            "evaluation_receipt_id": evaluation.receipt_id,
            "evaluation_hash": sha256_json(evaluation.to_dict()),
            "selected_variant_id": evaluation.selected_variant_id,
        }

    def promote(self, *, spec, evaluation, reconciliation):
        self.promote_calls.append((spec, evaluation, reconciliation))
        return sha256_json(
            {
                "reference": "test",
                "evaluation": evaluation.receipt_id,
                "experiment": spec.experiment_hash(),
            }
        )


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
    payload = {
        "stage": stage,
        "tools": tools,
        "manifest_hashes": {tool: manifest, source_tool: source_manifest},
        "signal_type": "HIRING" if stage == "intent_evidence" else "",
        "route_change_class": "custom",
    }
    if not with_source_add:
        payload["route_variant"] = "candidate-route"
    if with_source_add:
        payload["skipped"] = {tool: "combined_route_change"}
    candidate_artifact_for_variant = candidate_artifact if (with_source_add or two_artifacts) else artifact
    candidate_authority_for_variant = candidate_authority if (with_source_add or two_artifacts) else authority
    provenance = ()
    if with_source_add:
        provenance = (SourceAddProvenance("source-add-request", sha256_json(source_request), source_tool, stage, source_manifest, candidate_artifact_for_variant.commit_sha, True, source_request),)
    baseline = RoutingExperimentV2Variant("baseline", stage, artifact, {"stage": stage, "tools": [tool], "manifest_hashes": {tool: manifest}, "signal_type": "HIRING" if stage == "intent_evidence" else "", "route_change_class": "custom"}, ("baseline",), "route_only", (), authority)
    candidate = RoutingExperimentV2Variant(
        "candidate",
        stage,
        candidate_artifact_for_variant,
        payload,
        tuple(item.binding_id for item in bindings),
        "tool_and_route" if with_source_add else change_kind,
        provenance,
        candidate_authority_for_variant,
        (source_tool,) if with_source_add else (),
    )
    labels = {"cal-1": True, "hold-1": False}
    inp = RoutingExperimentV2Input(
        stage,
        FEATURE_HASH,
        FEATURES,
        ("cal-1",),
        ("hold-1",),
        sha256_json({"labels": sorted(labels.items())}),
        "HIRING" if stage == "intent_evidence" else "",
        unit_input_set_hash=H("d"),
    )
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


def _measured_receipt(binding, unit_ref, request_fingerprint):
    value = dict(_runner(binding, unit_ref, request_fingerprint))
    value["execution_mode"] = ReceiptExecutionMode.MEASURED_LAB.value
    value["receipt_ref"] = "provider_receipt:" + sha256_json(
        {key: item for key, item in value.items() if key != "receipt_ref"}
    ).split(":", 1)[1][:16]
    return value


def test_v2_intent_route_only_and_combined_source_add_run_with_decision_receipts():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence", with_source_add=True)
    assert validate_routing_experiment_v2_spec(spec, adapters=adapters) == []
    decisions = RoutingDecisionReceiptStore()
    evaluation = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=_runner, adapters=adapters, decision_store=decisions, require_isolation=False)
    assert evaluation.selected_variant_id == "baseline"
    assert evaluation.decision_receipt_refs
    assert all(item.stage == "intent_evidence" for item in evaluation.variants)
    assert all("intent.baseline" in receipt.considered_tool_ids for receipt in decisions.values())
    assert all(set(receipt.attempted_tool_ids) <= {"intent.baseline", "intent.source_add.new"} for receipt in decisions.values())
    assert any(dict(receipt.skipped_tool_reasons).get("intent.source_add.new") == "model_route_stopped" for receipt in decisions.values())
    tool_only_payload = dict(spec.variants[1].routing_payload)
    tool_only_payload["tools"] = ["intent.baseline", "intent.source_add.new"]
    tool_only_payload["manifest_hashes"] = {
        "intent.baseline": H("4"),
        "intent.source_add.new": tool_only_payload["manifest_hashes"]["intent.source_add.new"],
    }
    tool_only_payload["route_change_class"] = "default"
    tool_only_payload.pop("skipped", None)
    tool_only = replace(spec.variants[1], variant_id="tool-only", routing_payload=tool_only_payload, change_kind="tool_only")
    tool_only_spec = replace(spec, experiment_id="v2-tool-only", variants=(spec.variants[0], tool_only))
    assert validate_routing_experiment_v2_spec(tool_only_spec, adapters={"baseline": adapters["baseline"], "tool-only": adapters["candidate"]}) == []
    promotion = FakePromotionAuthority()
    assert promote_routing_experiment_v2_to_lab(
        evaluation,
        spec=spec,
        authority=promotion,
    ).startswith("sha256:")
    assert promotion.promote_calls


def test_v2_variant_rejects_duplicate_tool_bindings_before_provider_calls():
    spec, adapters, labels, tool, _source_tool = _spec("intent_evidence")
    duplicate_binding = _binding("baseline-duplicate", tool, H("4"))
    candidate = replace(
        spec.variants[1],
        binding_ids=("baseline", "baseline-duplicate"),
    )
    invalid = replace(
        spec,
        experiment_id="duplicate-tool-binding",
        variants=(spec.variants[0], candidate),
        provider_bindings=(*spec.provider_bindings, duplicate_binding),
        credit_budget=replace(
            spec.credit_budget,
            provider_credit_ceilings={
                **spec.credit_budget.provider_credit_ceilings,
                "baseline-duplicate": 1000,
            },
        ),
    )
    errors = validate_routing_experiment_v2_spec(invalid, adapters=adapters)
    assert (
        "v2_variant_binding_tool_ids_must_be_unique:candidate:intent.baseline"
        in errors
    )
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _runner(*args[:3])

    with pytest.raises(RoutingExperimentError, match="variant_binding_tool_ids_must_be_unique"):
        evaluate_routing_experiment_v2(
            invalid,
            gold_labels=labels,
            runner=recording_runner,
            adapters=adapters,
            require_isolation=False,
        )
    assert calls == []


def test_v2_combined_route_allows_baseline_registered_source_add_plus_new_source_add():
    spec, _adapters, labels, tool, source_tool = _spec(
        "intent_evidence", with_source_add=True
    )
    registered_tool = "intent.source_add.baseline"
    registered_binding = _binding("registered", registered_tool, H("6"))
    baseline_payload = dict(spec.variants[0].routing_payload)
    baseline_payload["tools"] = [tool, registered_tool]
    baseline_payload["manifest_hashes"] = {tool: H("4"), registered_tool: H("6")}
    candidate_payload = dict(spec.variants[1].routing_payload)
    candidate_payload["tools"] = [tool, registered_tool, source_tool]
    candidate_payload["manifest_hashes"] = {
        tool: H("4"),
        registered_tool: H("6"),
        source_tool: candidate_payload["manifest_hashes"][source_tool],
    }
    variants = (
        replace(
            spec.variants[0],
            routing_payload=baseline_payload,
            binding_ids=("baseline", "registered"),
        ),
        replace(
            spec.variants[1],
            routing_payload=candidate_payload,
            binding_ids=("baseline", "registered", "source"),
        ),
    )
    updated = replace(
        spec,
        experiment_id="baseline-and-new-source-add",
        variants=variants,
        provider_bindings=(*spec.provider_bindings, registered_binding),
        credit_budget=replace(
            spec.credit_budget,
            provider_credit_ceilings={
                **spec.credit_budget.provider_credit_ceilings,
                "registered": 1000,
            },
        ),
    )
    manifests = {tool: H("4"), registered_tool: H("6"), source_tool: variants[1].routing_payload["manifest_hashes"][source_tool]}
    adapters = {
        "baseline": FakeV2Adapter(
            manifests, registered_source_add_tool_ids=(registered_tool,)
        ),
        "candidate": FakeV2Adapter(
            manifests, registered_source_add_tool_ids=(registered_tool,)
        ),
    }
    assert validate_routing_experiment_v2_spec(updated, adapters=adapters) == []
    evaluation = evaluate_routing_experiment_v2(
        updated,
        gold_labels=labels,
        runner=_runner,
        adapters=adapters,
        require_isolation=False,
    )
    assert evaluation.provider_receipt_refs


def test_v2_requires_complete_provider_credit_ceilings():
    spec, adapters, _labels, _tool, _source_tool = _spec("intent_evidence")
    incomplete = replace(
        spec,
        credit_budget=ExperimentCreditBudget(spec.credit_budget.total_credit_microunits, {}),
    )
    errors = validate_routing_experiment_v2_spec(incomplete, adapters=adapters)
    assert "v2_provider_credit_ceiling_missing:baseline" in errors


def test_v2_required_tool_validation_uses_parsed_model_profile():
    @dataclass(frozen=True)
    class ParsedProfile:
        payload: dict
        tools: tuple[str, ...]

    class ParsedProfileAdapter(FakeV2Adapter):
        def parse_variant_payload(self, payload, *, stage):
            raw = super().parse_variant_payload(payload, stage=stage)
            return ParsedProfile(raw, tuple(raw.get("tools", ())))

        @staticmethod
        def _raw(payload):
            return payload.payload if isinstance(payload, ParsedProfile) else payload

        def validate_variant_payload(self, payload, **kwargs):
            return super().validate_variant_payload(self._raw(payload), **kwargs)

        def routing_identity(self, payload, **kwargs):
            return super().routing_identity(self._raw(payload), **kwargs)

        def compile_variant(self, payload, **kwargs):
            raw = dict(self._raw(payload))
            # The model profile declares a required tool, but this adversarial
            # compiler drops it from the compiled route.
            raw["tools"] = []
            return super().compile_variant(raw, **kwargs)

        def variant_tool_descriptors(self, payload, *, stage):
            if not isinstance(payload, ParsedProfile):
                # This is the old bug: a raw mapping has no typed ``tools``
                # field and would make the required-tool set appear empty.
                return ()
            return tuple(
                {
                    "tool_id": tool_id,
                    "stage": stage,
                    "source_add": False,
                    "required": True,
                    "manifest_hash": payload.payload.get("manifest_hashes", {}).get(tool_id, ""),
                }
                for tool_id in payload.tools
            )

    spec, adapters, labels, tool, _source_tool = _spec("intent_evidence")
    parsed_adapters = {
        "baseline": ParsedProfileAdapter(adapters["baseline"].manifests),
        "candidate": ParsedProfileAdapter(adapters["candidate"].manifests),
    }
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _runner(*args[:3])

    with pytest.raises(
        RoutingExperimentError,
        match="required_tool_missing_from_compiled_plan:intent.baseline",
    ):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=recording_runner,
            adapters=parsed_adapters,
            require_isolation=False,
        )
    assert calls == []


def test_v2_admits_experiment_worst_case_spend_before_first_provider_call():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    # Each of two variants runs over two units at 10 microunits per initial
    # plan: 40 microunits is required before the first call can be admitted.
    underbudget = replace(
        spec,
        experiment_id="underbudget-multi-variant",
        credit_budget=replace(spec.credit_budget, total_credit_microunits=39),
    )
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _runner(*args[:3])

    with pytest.raises(
        RoutingExperimentError,
        match="v2_experiment_worst_case_credit_budget_exceeded",
    ):
        evaluate_routing_experiment_v2(
            underbudget,
            gold_labels=labels,
            runner=recording_runner,
            adapters=adapters,
            require_isolation=False,
        )
    assert calls == []


def test_v2_admits_possible_confirmation_before_first_provider_call():
    spec, adapters, labels, tool, _source_tool = _spec("intent_evidence")
    continuation_tool = "intent.confirmation"
    continuation_binding = _binding("confirmation", continuation_tool, H("5"))
    variants = tuple(
        replace(
            variant,
            binding_ids=(*variant.binding_ids, "confirmation"),
            routing_payload={
                **variant.routing_payload,
                "conditional": [continuation_tool],
            },
        )
        for variant in spec.variants
    )
    updated = replace(
        spec,
        experiment_id="underbudget-confirmation-wave",
        variants=variants,
        provider_bindings=(*spec.provider_bindings, continuation_binding),
        credit_budget=replace(
            spec.credit_budget,
            # Initial waves cost 40.  The possible confirmation wave adds 40.
            total_credit_microunits=79,
            provider_credit_ceilings={
                **spec.credit_budget.provider_credit_ceilings,
                "confirmation": 1000,
            },
        ),
    )

    class ConfirmationBudgetAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del kwargs
            return Plan(
                {
                    "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                    "steps": [tool],
                    "conditional": [continuation_tool],
                },
                (tool,),
            )

        def has_conditional_confirmation(self, plan):
            return bool(plan.payload.get("conditional"))

        def compile_confirmation(self, parent_plan, **kwargs):
            del kwargs
            return Plan(
                {
                    "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                    "steps": [continuation_tool],
                },
                (continuation_tool,),
            )

    confirmation_adapters = {
        key: ConfirmationBudgetAdapter(
            {**value.manifests, continuation_tool: H("5")}
        )
        for key, value in adapters.items()
    }
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _runner(*args[:3])

    with pytest.raises(
        RoutingExperimentError,
        match="v2_experiment_worst_case_credit_budget_exceeded",
    ):
        evaluate_routing_experiment_v2(
            updated,
            gold_labels=labels,
            runner=recording_runner,
            adapters=confirmation_adapters,
            require_isolation=False,
        )
    assert calls == []


def test_v2_declared_confirmation_source_is_bound_before_calls_and_invoked_after_hit():
    spec, adapters, labels, tool, source_tool = _spec(
        "intent_evidence", with_source_add=True
    )
    candidate = replace(
        spec.variants[1],
        routing_payload={
            **spec.variants[1].routing_payload,
            "conditional": [source_tool],
        },
    )
    conditional_spec = replace(
        spec,
        experiment_id="declared-confirmation-source",
        variants=(spec.variants[0], candidate),
    )

    class ConditionalSourceAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del kwargs
            conditional = tuple(payload.get("conditional", ()))
            primary = tuple(
                item for item in payload.get("tools", ()) if item not in conditional
            )
            return Plan(
                {
                    "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                    "steps": list(primary),
                    "conditional": list(conditional),
                },
                primary,
            )

    conditional_adapters = {
        "baseline": adapters["baseline"],
        "candidate": ConditionalSourceAdapter(adapters["candidate"].manifests),
    }
    calls = []

    def recording_runner(binding, unit, request):
        calls.append((binding.tool_id, unit))
        return _runner(binding, unit, request)

    evaluation = evaluate_routing_experiment_v2(
        conditional_spec,
        gold_labels=labels,
        runner=recording_runner,
        adapters=conditional_adapters,
        require_isolation=False,
    )
    assert evaluation.provider_receipt_refs
    assert (source_tool, "cal-1") in calls
    assert (source_tool, "hold-1") not in calls
    assert (tool, "cal-1") in calls


def test_v2_invalid_confirmation_compiler_fails_before_any_provider_call():
    spec, adapters, labels, tool, _source_tool = _spec("intent_evidence")
    confirmation_tool = "intent.confirmation"
    confirmation_binding = _binding("confirmation", confirmation_tool, H("5"))
    variants = tuple(
        replace(
            variant,
            binding_ids=(*variant.binding_ids, "confirmation"),
            routing_payload={
                **variant.routing_payload,
                "conditional": [confirmation_tool],
            },
        )
        for variant in spec.variants
    )
    invalid = replace(
        spec,
        experiment_id="invalid-confirmation-global-preflight",
        variants=variants,
        provider_bindings=(*spec.provider_bindings, confirmation_binding),
        credit_budget=replace(
            spec.credit_budget,
            provider_credit_ceilings={
                **spec.credit_budget.provider_credit_ceilings,
                "confirmation": 1_000,
            },
        ),
    )

    class BrokenConfirmationAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del kwargs
            return Plan(
                {
                    "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                    "steps": [tool],
                    "conditional": [confirmation_tool],
                },
                (tool,),
            )

        def compile_confirmation(self, parent_plan, **kwargs):
            del parent_plan, kwargs
            raise RuntimeError("confirmation compiler is invalid")

    invalid_adapters = {
        key: BrokenConfirmationAdapter(
            {**value.manifests, confirmation_tool: H("5")}
        )
        for key, value in adapters.items()
    }
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _runner(*args[:3])

    with pytest.raises(
        RoutingExperimentError,
        match="v2_model_confirmation_global_preflight_failed",
    ):
        evaluate_routing_experiment_v2(
            invalid,
            gold_labels=labels,
            runner=recording_runner,
            adapters=invalid_adapters,
            require_isolation=False,
        )
    assert calls == []


def test_v2_declared_challenger_source_is_preflighted_and_runs_after_miss():
    spec, adapters, labels, tool, source_tool = _spec(
        "intent_evidence", with_source_add=True
    )
    candidate = replace(
        spec.variants[1],
        routing_payload={
            **spec.variants[1].routing_payload,
            "dormant": [source_tool],
        },
    )
    challenger_spec = replace(
        spec,
        experiment_id="declared-challenger-source",
        variants=(spec.variants[0], candidate),
    )

    class ChallengerSourceAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del kwargs
            dormant = tuple(payload.get("dormant", ()))
            primary = tuple(
                item for item in payload.get("tools", ()) if item not in dormant
            )
            return Plan(
                {
                    "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                    "steps": list(primary),
                    "dormant": list(dormant),
                },
                primary,
            )

    challenger_adapters = {
        "baseline": adapters["baseline"],
        "candidate": ChallengerSourceAdapter(adapters["candidate"].manifests),
    }
    calls = []

    def recording_runner(binding, unit, request):
        calls.append((binding.tool_id, unit))
        return _runner(binding, unit, request)

    evaluation = evaluate_routing_experiment_v2(
        challenger_spec,
        gold_labels=labels,
        runner=recording_runner,
        adapters=challenger_adapters,
        require_isolation=False,
    )
    assert evaluation.provider_receipt_refs
    assert (source_tool, "hold-1") in calls
    assert (source_tool, "cal-1") not in calls
    assert (tool, "hold-1") in calls


def test_v2_declared_new_intent_tool_positive():
    stage = "intent_evidence"
    spec, adapters, labels, _tool, source_tool = _spec(
        stage, with_source_add=True
    )
    assert validate_routing_experiment_v2_spec(spec, adapters=adapters) == []
    decisions = RoutingDecisionReceiptStore()
    evaluation = evaluate_routing_experiment_v2(
        spec,
        gold_labels=labels,
        runner=_runner,
        adapters=adapters,
        decision_store=decisions,
        require_isolation=False,
    )
    assert evaluation.decision_receipt_refs
    assert any(source_tool in item.considered_tool_ids for item in decisions.values())


@pytest.mark.parametrize(
    ("field_name", "error_code"),
    [
        ("calibration_unit_refs", "v2_calibration_unit_refs_must_not_be_empty"),
        ("holdout_unit_refs", "v2_holdout_unit_refs_must_not_be_empty"),
    ],
)
def test_v2_empty_split_fails_before_provider_calls(field_name, error_code):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    empty_input = replace(spec.input, **{field_name: ()})
    invalid = replace(spec, experiment_id=f"empty-{field_name}", input=empty_input)
    assert error_code in validate_routing_experiment_v2_spec(invalid, adapters=adapters)
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _runner(*args[:3])

    with pytest.raises(RoutingExperimentError, match=error_code):
        evaluate_routing_experiment_v2(
            invalid,
            gold_labels=labels,
            runner=recording_runner,
            adapters=adapters,
            require_isolation=False,
        )
    assert calls == []


def test_v2_live_requires_structurally_durable_provider_store(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )

    class ForgedNonDurableRepository:
        durable = False

        def __init__(self):
            self.rows = {}

        def get(self, key):
            return self.rows.get(key)

        def append(self, key, receipt):
            self.rows[key] = receipt
            return receipt

        def keys(self):
            return tuple(self.rows)

    calls = []

    def measured_runner(binding, unit, request, authorization):
        calls.append((binding, unit, request, authorization))
        value = dict(_runner(binding, unit, request))
        value["execution_mode"] = ReceiptExecutionMode.MEASURED_LAB.value
        value["receipt_ref"] = "provider_receipt:" + sha256_json(
            {key: item for key, item in value.items() if key != "receipt_ref"}
        ).split(":", 1)[1][:16]
        return value

    with pytest.raises(
        RoutingExperimentError,
        match="v2_live_spend_requires_durable_receipt_repository",
    ):
        evaluate_routing_experiment_v2(
            measured,
            gold_labels=labels,
            runner=measured_runner,
            adapters=adapters,
            receipt_store=ProviderReceiptStore(ForgedNonDurableRepository()),
            decision_store=RoutingDecisionReceiptStore(
                JsonlRoutingDecisionReceiptRepository(tmp_path / "decisions.jsonl")
            ),
            authoritative_billing_rollup=lambda _store: {
                "rollup_id": "unused",
                "rollup_hash": H("9"),
                "total_credit_microunits": 0,
            },
            require_isolation=False,
        )
    assert calls == []


def test_v2_live_provider_readiness_probes_empty_repository(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )

    class EmptyBrokenRepository:
        durable = True

        def keys(self):
            return ()

        def get(self, key):
            del key
            raise RuntimeError("provider get unavailable")

        def append(self, key, receipt):
            raise AssertionError("readiness must not append")

    calls = []

    def measured_runner(binding, unit, request, authorization):
        calls.append((binding, unit, request, authorization))
        return _runner(binding, unit, request)

    with pytest.raises(
        RoutingExperimentError,
        match="v2_live_spend_requires_durable_receipt_repository",
    ):
        evaluate_routing_experiment_v2(
            measured,
            gold_labels=labels,
            runner=measured_runner,
            adapters=adapters,
            receipt_store=ProviderReceiptStore(EmptyBrokenRepository()),
            decision_store=RoutingDecisionReceiptStore(
                JsonlRoutingDecisionReceiptRepository(tmp_path / "decisions.jsonl")
            ),
            authoritative_billing_rollup=lambda _store: {
                "rollup_id": "unused",
                "rollup_hash": H("9"),
                "total_credit_microunits": 0,
            },
            require_isolation=False,
        )
    assert calls == []


def test_v2_invalid_cached_provider_receipt_fails_before_runner():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    repository = InMemoryProviderReceiptRepository()
    store = ProviderReceiptStore(repository)
    evaluate_routing_experiment_v2(
        spec,
        gold_labels=labels,
        runner=_runner,
        adapters=adapters,
        receipt_store=store,
        require_isolation=False,
    )
    key, receipt = next(iter(repository._rows.items()))
    repository._rows[key] = replace(receipt, evidence_hash=H("f"))
    calls = []

    def should_not_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid cached receipt must fail before runner")

    with pytest.raises(RoutingExperimentError, match="cached provider receipt invalid"):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=should_not_call,
            adapters=adapters,
            receipt_store=store,
            require_isolation=False,
        )
    assert calls == []


@pytest.mark.parametrize("stage", ["candidate_acquisition", "intent_evidence"])
def test_v2_declared_new_tool_must_be_in_model_profile_before_provider_calls(stage):
    spec, adapters, labels, _tool, source_tool = _spec(stage, with_source_add=True)
    payload = dict(spec.variants[1].routing_payload)
    baseline_tool = f"{stage.split('_')[0]}.baseline"
    payload["tools"] = [baseline_tool]
    payload["manifest_hashes"] = {baseline_tool: H("4")}
    invalid_variant = replace(
        spec.variants[1],
        variant_id="absent-new-tool",
        routing_payload=payload,
    )
    invalid = replace(
        spec,
        experiment_id="absent-new-tool",
        variants=(spec.variants[0], invalid_variant),
    )
    assert any(
        f"v2_new_tool_absent_from_model_profile:{source_tool}" in error
        for error in validate_routing_experiment_v2_spec(invalid, adapters={
            "baseline": adapters["baseline"],
            "absent-new-tool": adapters["candidate"],
        })
    )
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _runner(*args[:3])

    with pytest.raises(RoutingExperimentError, match="v2_new_tool_absent_from_model_profile"):
        evaluate_routing_experiment_v2(
            invalid,
            gold_labels=labels,
            runner=recording_runner,
            adapters={
                "baseline": adapters["baseline"],
                "absent-new-tool": adapters["candidate"],
            },
            require_isolation=False,
        )
    assert calls == []


def test_v2_preflights_later_variant_plan_before_provider_calls():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")

    class LaterCompileFailureAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            if payload.get("route_variant") == "candidate-route":
                raise RuntimeError("candidate plan is not executable")
            return super().compile_variant(payload, **kwargs)

    calls = []

    def recording_runner(binding, unit, request):
        calls.append((binding.tool_id, unit, request))
        return _runner(binding, unit, request)

    with pytest.raises(
        RoutingExperimentError,
        match="v2_variant_plan_preflight_failed:candidate:RuntimeError",
    ):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=recording_runner,
            adapters={
                "baseline": adapters["baseline"],
                "candidate": LaterCompileFailureAdapter(
                    adapters["candidate"].manifests
                ),
            },
            require_isolation=False,
        )
    assert calls == []


def test_v2_provider_receipt_is_the_only_positive_signal_authority():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")

    class LyingPredictionAdapter(FakeV2Adapter):
        def execute_plan(self, plan, invoke):
            results = [invoke(tool) for tool in plan.tools]
            return results, True

    def source_miss_runner(binding, unit, request):
        value = dict(_runner(binding, unit, request))
        value["outcome"] = ProviderOutcome.SOURCE_MISS.value
        value["receipt_ref"] = "provider_receipt:" + sha256_json(
            {key: item for key, item in value.items() if key != "receipt_ref"}
        ).split(":", 1)[1][:16]
        return value

    calls = []

    def recording_runner(binding, unit, request):
        calls.append((binding.tool_id, unit, request))
        return source_miss_runner(binding, unit, request)

    lying_adapters = {
        key: LyingPredictionAdapter(value.manifests)
        for key, value in adapters.items()
    }
    with pytest.raises(
        RoutingExperimentError,
        match="v2_model_prediction_disagrees_with_provider_receipts",
    ):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=recording_runner,
            adapters=lying_adapters,
            require_isolation=False,
        )
    assert calls


def test_pinned_adapter_invokes_atomic_workflow_once_and_accounts_full_call_units():
    tool_id = "intent.source_add.predictleads_connections"

    class Runtime:
        EXECUTION_INVOKE = "invoke"

        @staticmethod
        def runtime_catalog(_availability):
            return object()

        @staticmethod
        def runtime_policy():
            return object()

    step = SimpleNamespace(
        tool_id=tool_id,
        execution_mode="invoke",
        max_calls=3,
        stop_on_success=True,
    )
    plan = SimpleNamespace(route=SimpleNamespace(steps=(step,)))
    adapter = PinnedSourcingModelRoutingAdapter(
        runtime=Runtime(),
        profiles=object(),
        features=object(),
        atomic_workflow_tool_ids=(tool_id,),
    )
    calls = []
    results, predicted = adapter.execute_plan(
        plan,
        lambda selected: (
            calls.append(selected)
            or {"outcome": ProviderOutcome.SOURCE_MISS.value}
        ),
    )
    assert calls == [tool_id]
    assert len(results) == 1
    assert predicted is False
    assert adapter.execution_call_units(plan, tool_id) == 3

    direct = PinnedSourcingModelRoutingAdapter(
        runtime=Runtime(),
        profiles=object(),
        features=object(),
    )
    direct_calls = []
    direct.execute_plan(
        plan,
        lambda selected: (
            direct_calls.append(selected)
            or {"outcome": ProviderOutcome.SOURCE_MISS.value}
        ),
    )
    assert direct_calls == [tool_id, tool_id, tool_id]
    assert direct.execution_call_units(plan, tool_id) == 1


def test_v2_rejects_invalid_atomic_workflow_call_units_before_runner():
    spec, adapters, labels, tool, _source_tool = _spec("intent_evidence")

    class InvalidAtomicUnitsAdapter(FakeV2Adapter):
        def execution_call_units(self, plan, tool_id):
            del plan, tool_id
            return 3

    calls = []
    with pytest.raises(
        RoutingExperimentError,
        match="v2_model_plan_execution_call_units_are_invalid",
    ):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=lambda *args: calls.append(args),
            adapters={
                "baseline": InvalidAtomicUnitsAdapter({tool: H("4")}),
                "candidate": adapters["candidate"],
            },
            require_isolation=False,
        )
    assert calls == []


@pytest.mark.parametrize("case", ["omitted", "extra"])
def test_v2_source_add_provenance_must_match_declared_new_tools(case):
    spec, adapters, labels, _tool, _source_tool = _spec(
        "intent_evidence", with_source_add=True
    )
    variant = spec.variants[1]
    if case == "omitted":
        invalid_variant = replace(
            variant,
            variant_id="omitted-new-tool",
            new_tool_ids=(),
        )
    else:
        extra = SourceAddProvenance(
            request_ref="extra-source-add-request",
            request_hash=H("a"),
            tool_id="intent.source_add.extra",
            stage="intent_evidence",
            manifest_hash=H("b"),
            artifact_commit_sha=variant.artifact.commit_sha,
        )
        invalid_variant = replace(
            variant,
            variant_id="extra-source-add-provenance",
            source_add_provenance=variant.source_add_provenance + (extra,),
        )
    invalid = replace(
        spec,
        experiment_id=f"provenance-{case}",
        variants=(spec.variants[0], invalid_variant),
    )
    adapters_by_variant = {
        "baseline": adapters["baseline"],
        invalid_variant.variant_id: adapters["candidate"],
    }
    errors = validate_routing_experiment_v2_spec(invalid, adapters=adapters_by_variant)
    assert any(
        "v2_source_add_provenance_tool_ids_must_match_new_tool_ids" in error
        for error in errors
    )
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _runner(*args[:3])

    with pytest.raises(
        RoutingExperimentError,
        match="v2_source_add_provenance_tool_ids_must_match_new_tool_ids",
    ):
        evaluate_routing_experiment_v2(
            invalid,
            gold_labels=labels,
            runner=recording_runner,
            adapters=adapters_by_variant,
            require_isolation=False,
        )
    assert calls == []


@pytest.mark.parametrize("stage", ["candidate_acquisition", "intent_evidence"])
def test_v2_baseline_registered_source_add_can_be_rerouted_without_new_declaration(stage):
    spec, adapters, _labels, _tool, source_tool = _spec(
        stage, with_source_add=True
    )
    baseline = spec.variants[0]
    rerouted = replace(
        spec.variants[1],
        variant_id="registered-reroute",
        artifact=baseline.artifact,
        artifact_authority_manifest=baseline.artifact_authority_manifest,
        change_kind="route_only",
        source_add_provenance=(),
        new_tool_ids=(),
    )
    reroute_adapters = {
        "baseline": FakeV2Adapter(
            adapters["baseline"].manifests,
            registered_source_add_tool_ids=(source_tool,),
        ),
        "registered-reroute": FakeV2Adapter(
            adapters["candidate"].manifests,
            registered_source_add_tool_ids=(source_tool,),
        ),
    }
    reroute_spec = replace(
        spec,
        experiment_id=f"{spec.experiment_id}-registered-reroute",
        variants=(baseline, rerouted),
    )
    assert validate_routing_experiment_v2_spec(
        reroute_spec,
        adapters=reroute_adapters,
    ) == []


@pytest.mark.parametrize("stage", ["candidate_acquisition", "intent_evidence"])
def test_v2_new_source_add_registration_requires_new_declaration_before_calls(stage):
    spec, adapters, labels, _tool, source_tool = _spec(
        stage, with_source_add=True
    )
    undeclared = replace(
        spec.variants[1],
        variant_id="undeclared-source-registration",
        new_tool_ids=(),
    )
    undeclared_spec = replace(
        spec,
        experiment_id=f"{spec.experiment_id}-undeclared-source-registration",
        variants=(spec.variants[0], undeclared),
    )
    undeclared_adapters = {
        "baseline": FakeV2Adapter(
            adapters["baseline"].manifests,
            registered_source_add_tool_ids=(),
        ),
        "undeclared-source-registration": FakeV2Adapter(
            adapters["candidate"].manifests,
            registered_source_add_tool_ids=(source_tool,),
        ),
    }
    errors = validate_routing_experiment_v2_spec(
        undeclared_spec,
        adapters=undeclared_adapters,
    )
    assert any(
        "v2_source_add_registration_not_in_baseline_requires_new_tool"
        in error
        for error in errors
    )
    calls = []

    def recording_runner(binding, unit, request):
        calls.append((binding.tool_id, unit, request))
        return _runner(binding, unit, request)

    with pytest.raises(RoutingExperimentError, match="invalid v2 routing experiment"):
        evaluate_routing_experiment_v2(
            undeclared_spec,
            gold_labels=labels,
            runner=recording_runner,
            adapters=undeclared_adapters,
            require_isolation=False,
        )
    assert calls == []


@pytest.mark.parametrize(
    "considered, expected",
    [
        (("intent.baseline", "intent.unbound"), "considered_tool_is_unbound"),
        (("intent.baseline", "intent.baseline"), "considered_tool_ids_must_be_unique"),
    ],
    ids=["unbound", "duplicate"],
)
def test_v2_invalid_considered_tools_fail_before_provider_calls(considered, expected):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    calls = []

    class InvalidConsideredAdapter(FakeV2Adapter):
        def considered_tool_ids(self, plan, *, stage):
            del plan, stage
            return considered

    def recording_runner(binding, unit, request):
        calls.append(binding.tool_id)
        return _runner(binding, unit, request)

    invalid_adapters = {
        key: InvalidConsideredAdapter(value.manifests) for key, value in adapters.items()
    }
    with pytest.raises(RoutingExperimentError, match=expected):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=recording_runner,
            adapters=invalid_adapters,
            require_isolation=False,
        )
    assert calls == []


def test_v2_empty_considered_no_call_plan_is_allowed():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    calls = []

    class EmptyPlanAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del payload, kwargs
            return Plan({"feature_set_sha256": FEATURE_HASH.split(":", 1)[1], "steps": []}, ())

        def plan_step_budgets(self, plan):
            del plan
            return ()

        def considered_tool_ids(self, plan, *, stage):
            del plan, stage
            return ()

    def recording_runner(binding, unit, request):
        calls.append((binding.tool_id, unit, request))
        return _runner(binding, unit, request)

    decisions = RoutingDecisionReceiptStore()
    evaluate_routing_experiment_v2(
        spec,
        gold_labels=labels,
        runner=recording_runner,
        adapters={key: EmptyPlanAdapter(value.manifests) for key, value in adapters.items()},
        decision_store=decisions,
        require_isolation=False,
    )
    assert calls == []
    assert decisions.values()
    assert all(item.considered_tool_ids == () for item in decisions.values())


def test_v2_canonicalizes_skipped_receipt_order_to_considered_tools():
    spec, adapters, labels, tool, _source_tool = _spec("intent_evidence")
    second_tool = "intent.secondary"
    second_binding = _binding("secondary", second_tool, H("5"))
    candidate = replace(
        spec.variants[1],
        routing_payload={
            "stage": "intent_evidence",
            "tools": [tool, second_tool],
            "manifest_hashes": {tool: H("4"), second_tool: H("5")},
            "signal_type": "HIRING",
            "route_change_class": "custom",
            "route_variant": "candidate-reverse-skip",
        },
        binding_ids=("baseline", "secondary"),
    )
    spec = replace(
        spec,
        variants=(spec.variants[0], candidate),
        provider_bindings=(*spec.provider_bindings, second_binding),
        credit_budget=replace(
            spec.credit_budget,
            provider_credit_ceilings={
                **spec.credit_budget.provider_credit_ceilings,
                "secondary": 1000,
            },
        ),
    )
    adapters = {
        "baseline": FakeV2Adapter({tool: H("4"), second_tool: H("5")} ),
        "candidate": FakeV2Adapter({tool: H("4"), second_tool: H("5")} ),
    }

    class ReverseSkipAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del payload, kwargs
            return Plan(
                {
                    "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                    "steps": [],
                    "considered": [tool, second_tool],
                },
                (),
            )

        def plan_step_budgets(self, plan):
            del plan
            return ()

        def considered_tool_ids(self, plan, *, stage):
            del plan, stage
            return (tool, second_tool)

        def plan_decision_projection(self, plan):
            del plan
            return {
                "skipped_tool_reasons": {
                    second_tool: "second",
                    tool: "first",
                },
                "attempted_tool_ids": (),
                "outcome_reasons": {},
            }

    decisions = RoutingDecisionReceiptStore()
    evaluate_routing_experiment_v2(
        spec,
        gold_labels=labels,
        runner=_runner,
        adapters={
            "baseline": adapters["baseline"],
            "candidate": ReverseSkipAdapter(adapters["candidate"].manifests),
        },
        decision_store=decisions,
        require_isolation=False,
    )
    candidate_receipts = [item for item in decisions.values() if item.variant_id == "candidate"]
    assert candidate_receipts
    assert all(
        receipt.skipped_tool_reasons == ((tool, "first"), (second_tool, "second"))
        for receipt in candidate_receipts
    )


@pytest.mark.parametrize("stage", ["candidate_acquisition", "intent_evidence"])
def test_v2_decision_receipts_round_trip_considered_tools_for_both_stages(tmp_path, stage):
    spec, adapters, labels, _tool, source_tool = _spec(stage, with_source_add=True)
    path = tmp_path / f"decisions-{stage}.jsonl"
    decisions = RoutingDecisionReceiptStore(JsonlRoutingDecisionReceiptRepository(path))
    assert decisions.is_durable is True
    evaluate_routing_experiment_v2(
        spec,
        gold_labels=labels,
        runner=_runner,
        adapters=adapters,
        decision_store=decisions,
        require_isolation=False,
    )
    original = decisions.values()
    assert original
    assert all(receipt.considered_tool_ids for receipt in original)
    assert any(source_tool in receipt.considered_tool_ids for receipt in original)

    reloaded = RoutingDecisionReceiptStore(JsonlRoutingDecisionReceiptRepository(path))
    assert [item.to_dict() for item in reloaded.values()] == [item.to_dict() for item in original]
    for receipt in original:
        assert RoutingDecisionReceiptV2.from_mapping(receipt.to_dict()) == receipt

    malformed = original[0].to_dict()
    malformed.pop("considered_tool_ids")
    assert any("missing_fields" in error for error in validate_routing_decision_receipt(malformed))
    malformed = original[0].to_dict()
    malformed["considered_tool_ids"] = "not-an-array"
    assert any("must_be_an_array" in error for error in validate_routing_decision_receipt(malformed))
    malformed = original[0].to_dict()
    malformed["ignored_secret_extra"] = "Bearer secret"
    with pytest.raises(RoutingExperimentError, match="unknown_fields"):
        RoutingDecisionReceiptV2.from_mapping(malformed)
    assert any("unknown_fields" in error for error in validate_routing_decision_receipt(malformed))
    tampered_record = {
        "key": "receipt-key",
        "receipt": original[0].to_dict(),
        "ignored_secret_extra": "Bearer secret",
    }
    path.write_text(path.read_text() + json.dumps(tampered_record) + "\n")
    with pytest.raises(RoutingExperimentError, match="repository_unknown_fields"):
        JsonlRoutingDecisionReceiptRepository(path)

    relationship_invalid = original[0].to_dict()
    relationship_invalid["considered_tool_ids"] = []
    relationship_invalid["attempted_tool_ids"] = ["intent.unconsidered"]
    relationship_invalid["skipped_tool_reasons"] = [["intent.unconsidered", "model_route_stopped"]]
    relationship_invalid["outcome_reasons"] = [["intent.unconsidered", ProviderOutcome.SOURCE_MISS.value]]
    relationship_invalid["provider_receipt_refs"] = []
    relationship_invalid["receipt_id"] = "routing_decision:" + sha256_json(
        {**relationship_invalid, "receipt_id": "routing_decision:pending"}
    ).split(":", 1)[1][:16]
    relationship_errors = validate_routing_decision_receipt(relationship_invalid)
    assert any("attempted_tools_must_be_considered" in error for error in relationship_errors)
    assert any("skipped_tools_must_match" in error for error in relationship_errors)

    valid_receipt = next(item for item in original if item.attempted_tool_ids)
    tool_id = valid_receipt.attempted_tool_ids[0]
    provider_ref = valid_receipt.provider_receipt_refs[0]
    duplicate_refs = valid_receipt.to_dict()
    duplicate_refs["attempted_tool_ids"] = [tool_id, tool_id]
    duplicate_refs["skipped_tool_reasons"] = []
    duplicate_refs["outcome_reasons"] = [[tool_id, ProviderOutcome.VERIFIED.value]] * 2
    duplicate_refs["provider_receipt_refs"] = [provider_ref, provider_ref]
    duplicate_refs["receipt_id"] = "routing_decision:" + sha256_json(
        {**duplicate_refs, "receipt_id": "routing_decision:pending"}
    ).split(":", 1)[1][:16]
    duplicate_errors = validate_routing_decision_receipt(duplicate_refs)
    assert any("provider_receipt_refs_must_be_unique" in error for error in duplicate_errors)

    duplicate_skips = valid_receipt.to_dict()
    duplicate_skips["considered_tool_ids"] = [tool_id]
    duplicate_skips["attempted_tool_ids"] = []
    duplicate_skips["skipped_tool_reasons"] = [[tool_id, "first"], [tool_id, "second"]]
    duplicate_skips["outcome_reasons"] = []
    duplicate_skips["provider_receipt_refs"] = []
    duplicate_skips["receipt_id"] = "routing_decision:" + sha256_json(
        {**duplicate_skips, "receipt_id": "routing_decision:pending"}
    ).split(":", 1)[1][:16]
    assert any(
        "skipped_tool_ids_must_be_unique" in error
        for error in validate_routing_decision_receipt(duplicate_skips)
    )

    invalid_outcome = valid_receipt.to_dict()
    invalid_outcome["outcome_reasons"] = [[tool_id, "invented_outcome"]]
    invalid_outcome["receipt_id"] = "routing_decision:" + sha256_json(
        {**invalid_outcome, "receipt_id": "routing_decision:pending"}
    ).split(":", 1)[1][:16]
    assert any("outcomes_are_invalid" in error for error in validate_routing_decision_receipt(invalid_outcome))

    invalid_provider_ref = valid_receipt.to_dict()
    invalid_provider_ref["provider_receipt_refs"] = ["provider_receipt:" + "A" * 16]
    invalid_provider_ref["receipt_id"] = "routing_decision:" + sha256_json(
        {**invalid_provider_ref, "receipt_id": "routing_decision:pending"}
    ).split(":", 1)[1][:16]
    assert any(
        "provider_receipt_ref_format_is_invalid" in error
        for error in validate_routing_decision_receipt(invalid_provider_ref)
    )


def test_v2_live_requires_durable_decision_store_before_provider_calls(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )

    class ForgedDurableStore:
        is_durable = True

    for index, decision_store in enumerate((None, RoutingDecisionReceiptStore(), ForgedDurableStore())):
        if decision_store is not None:
            if isinstance(decision_store, RoutingDecisionReceiptStore):
                assert decision_store.is_durable is False
        calls = []

        def measured_runner(binding, unit, request, authorization):
            calls.append((binding, unit, request, authorization))
            value = dict(_runner(binding, unit, request))
            value["execution_mode"] = ReceiptExecutionMode.MEASURED_LAB.value
            value["receipt_ref"] = "provider_receipt:" + sha256_json(
                {key: item for key, item in value.items() if key != "receipt_ref"}
            ).split(":", 1)[1][:16]
            return value

        with pytest.raises(RoutingExperimentError, match="durable_decision_receipt_store"):
            evaluate_routing_experiment_v2(
                measured,
                gold_labels=labels,
                runner=measured_runner,
                adapters=adapters,
                receipt_store=ProviderReceiptStore(
                    JsonlProviderReceiptRepository(tmp_path / f"provider-{index}.jsonl")
                ),
                decision_store=decision_store,
                authoritative_billing_rollup=lambda _store: {
                    "rollup_id": "rejected",
                    "rollup_hash": H("9"),
                    "total_credit_microunits": 0,
                },
                require_isolation=False,
            )
        assert calls == []


@pytest.mark.parametrize("broken_method", ["keys", "get"])
def test_v2_live_readiness_checks_repository_without_provider_calls(tmp_path, broken_method):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )

    class BrokenRepository:
        durable = True

        def keys(self):
            if broken_method == "keys":
                raise RuntimeError("keys unavailable")
            return ()

        def get(self, key):
            del key
            if broken_method == "get":
                raise RuntimeError("get unavailable")
            return None

        def append(self, key, receipt):
            raise AssertionError("readiness must not append")

    calls = []

    def measured_runner(binding, unit, request, authorization):
        calls.append((binding, unit, request, authorization))
        value = dict(_runner(binding, unit, request))
        value["execution_mode"] = ReceiptExecutionMode.MEASURED_LAB.value
        value["receipt_ref"] = "provider_receipt:" + sha256_json(
            {key: item for key, item in value.items() if key != "receipt_ref"}
        ).split(":", 1)[1][:16]
        return value

    with pytest.raises(RoutingExperimentError, match="durable_decision_receipt_store"):
        evaluate_routing_experiment_v2(
            measured,
            gold_labels=labels,
            runner=measured_runner,
            adapters=adapters,
            receipt_store=ProviderReceiptStore(
                JsonlProviderReceiptRepository(tmp_path / f"provider-{broken_method}.jsonl")
            ),
            decision_store=RoutingDecisionReceiptStore(BrokenRepository()),
            authoritative_billing_rollup=lambda _store: {
                "rollup_id": "readiness",
                "rollup_hash": H("9"),
                "total_credit_microunits": 0,
            },
            require_isolation=False,
        )
    assert calls == []


def test_v2_decision_store_requires_structural_repository_and_keeps_custom_seam():
    class InvalidRepository:
        durable = True

    with pytest.raises(RoutingExperimentError, match="repository_invalid"):
        RoutingDecisionReceiptStore(InvalidRepository())

    class CustomRepository:
        durable = True

        def __init__(self):
            self.rows = {}

        def get(self, key):
            return self.rows.get(key)

        def append(self, key, receipt):
            self.rows[key] = receipt
            return receipt

        def keys(self):
            return tuple(self.rows)

    custom = CustomRepository()
    store = RoutingDecisionReceiptStore(custom)
    assert store.readiness_errors() == []
    assert store.is_durable is True


def test_v2_signal_type_is_frozen_and_rejected_before_provider_calls():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    mismatched_payload = dict(spec.variants[1].routing_payload)
    mismatched_payload["signal_type"] = "FUNDING"
    mismatched = replace(spec.variants[1], routing_payload=mismatched_payload)
    mismatched_spec = replace(spec, experiment_id="v2-signal-mismatch", variants=(spec.variants[0], mismatched))
    errors = validate_routing_experiment_v2_spec(
        mismatched_spec,
        adapters={"baseline": adapters["baseline"], "candidate": adapters["candidate"]},
    )
    assert any("intent_category_mismatch" in error for error in errors)
    calls = []
    with pytest.raises(RoutingExperimentError, match="invalid v2 routing experiment"):
        evaluate_routing_experiment_v2(
            mismatched_spec,
            gold_labels=labels,
            runner=lambda *args: calls.append(args),
            adapters=adapters,
            require_isolation=False,
        )
    assert calls == []


@pytest.mark.parametrize("stage", ["candidate_acquisition", "intent_evidence"])
def test_v2_change_kind_requires_baseline_artifact_and_model_owned_route_class(stage):
    spec, adapters, labels, _tool, _source_tool = _spec(stage, with_source_add=True)
    assert validate_routing_experiment_v2_spec(spec, adapters=adapters) == []

    tool_only_payload = dict(spec.variants[1].routing_payload)
    tool_only_payload["route_change_class"] = "default"
    tool_only_payload.pop("skipped", None)
    tool_only = replace(spec.variants[1], variant_id="tool-only", routing_payload=tool_only_payload, change_kind="tool_only")
    tool_only_spec = replace(spec, experiment_id=f"{spec.experiment_id}-tool-only", variants=(spec.variants[0], tool_only))
    assert validate_routing_experiment_v2_spec(tool_only_spec, adapters={"baseline": adapters["baseline"], "tool-only": adapters["candidate"]}) == []

    mislabeled_tool_only = replace(
        tool_only,
        variant_id="bad-tool-only",
        routing_payload={**tool_only_payload, "skipped": {"intent.baseline": "changed_route"}},
    )
    errors = validate_routing_experiment_v2_spec(
        replace(tool_only_spec, variants=(spec.variants[0], mislabeled_tool_only)),
        adapters={"baseline": adapters["baseline"], "bad-tool-only": adapters["candidate"]},
    )
    assert any("tool_only_routing_identity_mismatch" in error for error in errors)

    bad_combined_payload = dict(spec.variants[1].routing_payload)
    bad_combined_payload.pop("skipped", None)
    bad_combined_payload["tools"] = [_tool]
    bad_combined_payload["manifest_hashes"] = {_tool: H("4")}
    mislabeled_combined = replace(
        spec.variants[1],
        variant_id="bad-combined",
        routing_payload=bad_combined_payload,
    )
    errors = validate_routing_experiment_v2_spec(
        replace(spec, experiment_id=f"{spec.experiment_id}-bad-combined", variants=(spec.variants[0], mislabeled_combined)),
        adapters={"baseline": adapters["baseline"], "bad-combined": adapters["candidate"]},
    )
    assert any("tool_and_route_routing_identity_must_differ" in error for error in errors)

    same_artifact_tool = replace(tool_only, variant_id="same-artifact", artifact=spec.variants[0].artifact)
    errors = validate_routing_experiment_v2_spec(
        replace(spec, experiment_id=f"{spec.experiment_id}-same-artifact", variants=(spec.variants[0], same_artifact_tool)),
        adapters={"baseline": adapters["baseline"], "same-artifact": adapters["candidate"]},
    )
    assert any("tool_variant_artifact_must_differ" in error for error in errors)

    route_only_wrong_artifact = replace(spec.variants[1], variant_id="bad-route-only", change_kind="route_only", source_add_provenance=(), new_tool_ids=())
    errors = validate_routing_experiment_v2_spec(
        replace(spec, experiment_id=f"{spec.experiment_id}-bad-route-only", variants=(spec.variants[0], route_only_wrong_artifact)),
        adapters={"baseline": adapters["baseline"], "bad-route-only": adapters["candidate"]},
    )
    assert any("route_only_artifact_must_match_baseline" in error for error in errors)


@pytest.mark.parametrize("stage", ["candidate_acquisition", "intent_evidence"])
def test_v2_route_only_requires_an_exact_model_route_change(stage):
    spec, adapters, _labels, _tool, _source_tool = _spec(stage)
    changed_payload = dict(spec.variants[1].routing_payload)
    changed_payload["route_profile_revision"] = "route-change-v2"
    changed = replace(
        spec.variants[1],
        variant_id="changed-route",
        routing_payload=changed_payload,
        artifact=spec.variants[0].artifact,
        source_add_provenance=(),
        new_tool_ids=(),
        change_kind="route_only",
    )
    changed_spec = replace(spec, experiment_id=f"{spec.experiment_id}-changed-route", variants=(spec.variants[0], changed))
    assert validate_routing_experiment_v2_spec(
        changed_spec,
        adapters={"baseline": adapters["baseline"], "changed-route": adapters["candidate"]},
    ) == []

    unchanged = replace(
        changed,
        variant_id="unchanged-route",
        routing_payload=dict(spec.variants[0].routing_payload),
    )
    unchanged_spec = replace(spec, experiment_id=f"{spec.experiment_id}-unchanged-route", variants=(spec.variants[0], unchanged))
    errors = validate_routing_experiment_v2_spec(
        unchanged_spec,
        adapters={"baseline": adapters["baseline"], "unchanged-route": adapters["candidate"]},
    )
    assert any("route_only_routing_identity_must_differ" in error for error in errors)


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


def test_v2_rejects_malformed_booleans_without_coercion():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    raw = spec.to_dict()
    raw["allow_live_credit_spend"] = "false"
    parsed = RoutingExperimentV2Spec.from_mapping(raw)
    assert parsed.allow_live_credit_spend == "false"
    assert any("allow_live_credit_spend_must_be_boolean" in error for error in validate_routing_experiment_v2_spec(parsed, adapters=adapters))
    raw = spec.to_dict()
    raw["availability"] = {"candidate": {"intent.baseline": "false"}}
    parsed = RoutingExperimentV2Spec.from_mapping(raw)
    assert parsed.availability["candidate"]["intent.baseline"] == "false"
    assert any("availability_must_be_boolean" in error for error in validate_routing_experiment_v2_spec(parsed, adapters=adapters))
    with pytest.raises(RoutingExperimentError, match="gold_labels_must_use_boolean"):
        evaluate_routing_experiment_v2(spec, gold_labels={"cal-1": "true", "hold-1": False}, runner=_runner, adapters=adapters, require_isolation=False)


def test_v2_receipt_mode_is_exact_and_live_requires_authorization_aware_runner(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    store = ProviderReceiptStore()

    def measured_runner(binding, unit, request):
        measured_value = dict(_runner(binding, unit, request))
        measured_value["execution_mode"] = ReceiptExecutionMode.MEASURED_LAB.value
        measured_value["receipt_ref"] = "provider_receipt:" + sha256_json({key: item for key, item in measured_value.items() if key != "receipt_ref"}).split(":", 1)[1][:16]
        return measured_value

    with pytest.raises(RoutingExperimentError, match="runner_receipt_execution_mode_mismatch"):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=measured_runner,
            adapters=adapters,
            receipt_store=store,
            require_isolation=False,
        )
    assert len(store) == 0

    measured = replace(
        spec,
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    from research_lab.routing_experiments import JsonlProviderReceiptRepository
    durable = ProviderReceiptStore(JsonlProviderReceiptRepository(tmp_path / "legacy-live.jsonl"))
    calls = []

    def legacy_runner(binding, unit, request):
        calls.append((binding, unit, request))
        return _runner(binding, unit, request)

    with pytest.raises(RoutingExperimentError, match="requires_authorization_aware_runner"):
        evaluate_routing_experiment_v2(
            measured,
            gold_labels=labels,
            runner=legacy_runner,
            adapters=adapters,
            receipt_store=durable,
            authoritative_billing_rollup=lambda _store: {"rollup_id": "legacy", "rollup_hash": H("9"), "total_credit_microunits": 0},
            require_isolation=False,
        )
    assert calls == []


@pytest.mark.parametrize(
    "runner_factory",
    [
        lambda calls: lambda *args: calls.append(args),
        lambda calls: lambda binding, unit, request, **kwargs: calls.append(
            (binding, unit, request, kwargs)
        ),
        lambda calls: lambda binding, unit, request: calls.append(
            (binding, unit, request)
        ),
        lambda calls: lambda *, authorization: calls.append(authorization),
        lambda calls: lambda *args, authorization: calls.append((args, authorization)),
        lambda calls: lambda binding, unit, request, extra, authorization: calls.append(
            (binding, unit, request, extra, authorization)
        ),
        lambda calls: lambda binding, unit, request, extra, *, authorization: calls.append(
            (binding, unit, request, extra, authorization)
        ),
        lambda calls: lambda binding, unit, request, authorization, extra: calls.append(
            (binding, unit, request, authorization, extra)
        ),
    ],
    ids=[
        "varargs_only",
        "kwargs_only",
        "three_args",
        "authorization_only",
        "varargs_with_authorization",
        "required_positional_before_authorization",
        "required_positional_before_keyword_authorization",
        "required_positional_after_authorization",
    ],
)
def test_v2_live_runner_requires_named_authorization_parameter(tmp_path, runner_factory):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    from research_lab.routing_experiments import JsonlProviderReceiptRepository

    calls = []
    runner = runner_factory(calls)
    with pytest.raises(RoutingExperimentError, match="requires_authorization_aware_runner"):
        evaluate_routing_experiment_v2(
            measured,
            gold_labels=labels,
            runner=runner,
            adapters=adapters,
            receipt_store=ProviderReceiptStore(
                JsonlProviderReceiptRepository(tmp_path / "rejected.jsonl")
            ),
            authoritative_billing_rollup=lambda _store: {
                "rollup_id": "rejected",
                "rollup_hash": H("9"),
                "total_credit_microunits": 0,
            },
            require_isolation=False,
        )
    assert calls == []


def test_v2_live_runner_with_named_authorization_parameter_passes(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    from research_lab.routing_experiments import JsonlProviderReceiptRepository

    calls = []

    def explicit_runner(binding, unit, request, authorization, optional=None):
        calls.append((authorization, optional))
        value = dict(_runner(binding, unit, request))
        value["execution_mode"] = ReceiptExecutionMode.MEASURED_LAB.value
        value["receipt_ref"] = "provider_receipt:" + sha256_json(
            {key: item for key, item in value.items() if key != "receipt_ref"}
        ).split(":", 1)[1][:16]
        return value

    evaluation = evaluate_routing_experiment_v2(
        measured,
        gold_labels=labels,
        runner=explicit_runner,
        adapters=adapters,
        receipt_store=ProviderReceiptStore(
            JsonlProviderReceiptRepository(tmp_path / "accepted.jsonl")
        ),
        decision_store=RoutingDecisionReceiptStore(
            JsonlRoutingDecisionReceiptRepository(tmp_path / "accepted-decisions.jsonl")
        ),
        authoritative_billing_rollup=lambda _store: {
            "rollup_id": "accepted",
            "rollup_hash": H("8"),
            "total_credit_microunits": 40,
        },
        artifact_authority=FakeArtifactAuthority(),
        require_isolation=False,
    )
    assert evaluation.live_credit_spend is True
    assert len(calls) == 4


def test_v2_worker_factory_reuses_one_handle_per_artifact_and_isolates_distinct_artifacts():
    same_spec, _adapters, _labels, _tool, _source_tool = _spec("intent_evidence")
    calls = []

    def worker_factory(artifact):
        artifact_key = sha256_json({
            "model_artifact_hash": artifact.model_artifact_hash,
            "manifest_hash": artifact.manifest_hash,
            "commit_sha": artifact.commit_sha,
        })
        calls.append(artifact_key)
        return IsolatedRoutingAdapter(object(), f"proc-{len(calls)}", artifact_key)

    factory = RoutingExperimentV2AdapterFactory.from_worker_factory(same_spec.variants, worker_factory)
    assert len(calls) == 1
    assert factory.handles["baseline"] is factory.handles["candidate"]

    split_spec, _adapters, _labels, _tool, _source_tool = _spec("intent_evidence", with_source_add=True, two_artifacts=True)
    calls.clear()
    split_factory = RoutingExperimentV2AdapterFactory.from_worker_factory(split_spec.variants, worker_factory)
    assert len(calls) == 2
    assert split_factory.handles["baseline"].process_id != split_factory.handles["candidate"].process_id


def test_v2_fail_closed_on_timeout_malformed_retry_and_budget():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    calls = []

    def failing_runner(binding, unit, request):
        calls.append((binding.tool_id, unit, request))
        raise TimeoutError("provider timeout")

    evaluation = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=failing_runner, adapters=adapters, require_isolation=False)
    assert all(item.holdout.adapter_failure_count >= 1 for item in evaluation.variants)
    assert all(item.holdout.false_negative_count == 0 for item in evaluation.variants)
    assert all(item.holdout.no_signal_credit_microunits == 0 for item in evaluation.variants)
    assert all(item.calibration.false_negative_count == 0 for item in evaluation.variants)
    assert all(item.calibration.no_signal_credit_microunits == 0 for item in evaluation.variants)
    assert len({request for _tool_id, _unit, request in calls}) == len(calls)
    def malformed_runner(binding, unit, request):
        del binding, unit, request
        return {}
    malformed = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=malformed_runner, adapters=adapters, require_isolation=False)
    assert all(item.holdout.adapter_failure_count >= 1 for item in malformed.variants)
    assert all(item.holdout.false_negative_count == 0 for item in malformed.variants)
    assert all(item.holdout.no_signal_credit_microunits == 0 for item in malformed.variants)
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
    retry_decisions = RoutingDecisionReceiptStore()
    retry = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=retry_runner, adapters=retry_adapters, decision_store=retry_decisions, require_isolation=False)
    assert retry.selected_variant_id in {"baseline", "candidate"}
    assert len(retry_calls) > len(set(retry_calls)) or len(retry_calls) >= 2
    retry_receipts = [item for item in retry_decisions.values() if item.variant_id == "baseline" and item.unit_ref == "cal-1"]
    assert retry_receipts and retry_receipts[0].attempted_tool_ids == ("intent.baseline", "intent.baseline")
    assert retry_receipts[0].outcome_reasons == (("intent.baseline", "adapter_failure"), ("intent.baseline", "verified"))
    unavailable_spec_base, unavailable_adapters, unavailable_labels, _unused_tool, _unused_source = _spec("intent_evidence", with_source_add=True)
    unavailable_spec = replace(unavailable_spec_base, availability={"candidate": {"intent.baseline": False, "intent.source_add.new": True}})
    unavailable_calls = []
    def unavailable_runner(binding, unit, request):
        unavailable_calls.append(binding.tool_id)
        return _runner(binding, unit, request)
    unavailable_decisions = RoutingDecisionReceiptStore()
    class AvailabilityAwareAdapter(FakeV2Adapter):
        def considered_tool_ids(self, plan, *, stage):
            del stage
            return tuple(dict.fromkeys((*plan.tools, *plan.payload.get("skipped", {}).keys())))
    unavailable_adapters["candidate"] = AvailabilityAwareAdapter(
        unavailable_adapters["candidate"].manifests
    )
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


def test_v2_decision_outcomes_ignore_adversarial_model_projection():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")

    class LyingProjectionAdapter(FakeV2Adapter):
        def plan_decision_projection(self, plan):
            del plan
            return {
                "skipped_tool_reasons": {
                    "intent.baseline": "fabricated_skip",
                },
                "outcome_reasons": {
                    "intent.baseline": "invented_success",
                },
            }

    decisions = RoutingDecisionReceiptStore()
    evaluate_routing_experiment_v2(
        spec,
        gold_labels=labels,
        runner=_runner,
        adapters={
            "baseline": LyingProjectionAdapter(adapters["baseline"].manifests),
            "candidate": LyingProjectionAdapter(adapters["candidate"].manifests),
        },
        decision_store=decisions,
        require_isolation=False,
    )
    outcomes = [item.outcome_reasons for item in decisions.values()]
    assert outcomes
    assert all("invented_success" not in dict(item) for item in outcomes)
    assert all("intent.unattempted" not in dict(item) for item in outcomes)
    assert all(set(dict(item)) <= {"intent.baseline"} for item in outcomes)
    skips = [item.skipped_tool_reasons for item in decisions.values()]
    assert all("intent.unattempted" not in dict(item) for item in skips)
    assert all("fabricated_skip" not in dict(item).values() for item in skips)


def test_v2_rejects_unconsidered_outcome_projection_before_provider_calls():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")

    class InvalidOutcomeProjectionAdapter(FakeV2Adapter):
        def plan_decision_projection(self, plan):
            del plan
            return {"outcome_reasons": {"intent.unconsidered": "invented_verified"}}

    calls = []
    def recording_runner(binding, unit, request):
        calls.append(binding.tool_id)
        return _runner(binding, unit, request)

    with pytest.raises(RoutingExperimentError, match="projection_tool_is_not_considered"):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=recording_runner,
            adapters={
                "baseline": InvalidOutcomeProjectionAdapter(adapters["baseline"].manifests),
                "candidate": adapters["candidate"],
            },
            require_isolation=False,
        )
    assert calls == []


def test_v2_rejects_unconsidered_projection_tools_before_provider_calls():
    spec, adapters, labels, tool, _source_tool = _spec("intent_evidence")

    class ExclusionAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del payload, kwargs
            return Plan(
                {
                    "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                    "steps": [],
                    "considered": [tool],
                    "skipped": {
                        tool: "model_exclusion",
                        "intent.unconsidered": "fabricated_unbound",
                    },
                },
                (),
            )

        def plan_step_budgets(self, plan):
            del plan
            return ()

    calls = []
    def recording_runner(binding, unit, request):
        calls.append(binding.tool_id)
        return _runner(binding, unit, request)
    with pytest.raises(RoutingExperimentError, match="projection_tool_is_not_considered"):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=recording_runner,
            adapters={
                "baseline": ExclusionAdapter(adapters["baseline"].manifests),
                "candidate": adapters["candidate"],
            },
            decision_store=RoutingDecisionReceiptStore(),
            require_isolation=False,
        )
    assert calls == []


def test_v2_enforces_compiled_plan_availability_calls_time_and_credit_caps():
    spec, adapters, labels, tool, _source_tool = _spec("intent_evidence")

    class UnplannedAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del payload, kwargs
            return Plan({"feature_set_sha256": FEATURE_HASH.split(":", 1)[1], "steps": []}, ())

        def plan_step_budgets(self, plan):
            del plan
            return ()

        def execute_plan(self, plan, invoke):
            del plan
            invoke(tool)
            return (), False

    unplanned_calls = []
    with pytest.raises(RoutingExperimentError, match="invoked_tool_is_not_considered"):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=lambda *args: unplanned_calls.append(args),
            adapters={"baseline": UnplannedAdapter({tool: H("4")}), "candidate": adapters["candidate"]},
            require_isolation=False,
        )
    assert unplanned_calls == []

    class ZeroTimeAdapter(FakeV2Adapter):
        def plan_step_budgets(self, plan):
            return tuple(
                RoutingPlanStepBudget(item, "invoke", 1, 0.0, 10)
                for item in plan.tools
            )

    zero_time_calls = []
    with pytest.raises(RoutingExperimentError, match="step_time_cap_exceeded"):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=lambda *args: zero_time_calls.append(args),
            adapters={"baseline": ZeroTimeAdapter({tool: H("4")}), "candidate": adapters["candidate"]},
            require_isolation=False,
        )
    assert zero_time_calls == []

    second_tool = "intent.second"
    second_binding = _binding("second", second_tool, H("5"))
    mixed_spec = replace(
        spec,
        variants=(replace(spec.variants[0], binding_ids=("baseline", "second")), spec.variants[1]),
        provider_bindings=(*spec.provider_bindings, second_binding),
        credit_budget=replace(
            spec.credit_budget,
            provider_credit_ceilings={**spec.credit_budget.provider_credit_ceilings, "second": 1000},
        ),
    )

    class MixedTimeAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del payload, kwargs
            return Plan(
                {
                    "feature_set_sha256": FEATURE_HASH.split(":", 1)[1],
                    "steps": [tool, second_tool],
                },
                (tool, second_tool),
            )

        def plan_step_budgets(self, plan):
            del plan
            return (
                RoutingPlanStepBudget(tool, "invoke", 1, 0.0, 10),
                RoutingPlanStepBudget(second_tool, "invoke", 1, 1.0, 10),
            )

    mixed_calls = []
    with pytest.raises(RoutingExperimentError, match="step_time_cap_exceeded"):
        evaluate_routing_experiment_v2(
            mixed_spec,
            gold_labels=labels,
            runner=lambda *args: mixed_calls.append(args),
            adapters={
                "baseline": MixedTimeAdapter({tool: H("4"), second_tool: H("5")}),
                "candidate": adapters["candidate"],
            },
            require_isolation=False,
        )
    assert mixed_calls == []

    class UnavailableAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            del kwargs
            return Plan({"feature_set_sha256": FEATURE_HASH.split(":", 1)[1], "steps": [tool]}, (tool,))

        def execute_plan(self, plan, invoke):
            del plan
            invoke(tool)
            return (), False

    unavailable = replace(spec, availability={"baseline": {tool: False}})
    unavailable_calls = []
    with pytest.raises(RoutingExperimentError, match="invoked_unavailable_tool"):
        evaluate_routing_experiment_v2(
            unavailable,
            gold_labels=labels,
            runner=lambda *args: unavailable_calls.append(args),
            adapters={"baseline": UnavailableAdapter({tool: H("4")}), "candidate": adapters["candidate"]},
            require_isolation=False,
        )
    assert unavailable_calls == []

    class OverCallAdapter(FakeV2Adapter):
        def execute_plan(self, plan, invoke):
            results = [invoke(tool), invoke(tool)]
            invoke(tool)
            return results, False

        def plan_step_budgets(self, plan):
            return tuple(RoutingPlanStepBudget(item, "invoke", 2, 1, 100) for item in plan.tools)

    overcall_calls = []
    with pytest.raises(RoutingExperimentError, match="max_calls_exceeded"):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=lambda *args: (overcall_calls.append(args) or _runner(*args)),
            adapters={"baseline": OverCallAdapter({tool: H("4")}), "candidate": adapters["candidate"]},
            require_isolation=False,
        )
    assert len(overcall_calls) == 2

    def oversized_runner(binding, unit, request):
        value = dict(_runner(binding, unit, request))
        value["credit_microunits"] = 11
        value["receipt_ref"] = "provider_receipt:" + sha256_json({key: item for key, item in value.items() if key != "receipt_ref"}).split(":", 1)[1][:16]
        return value

    store = ProviderReceiptStore()
    with pytest.raises(RoutingExperimentError, match="exceeds_reserved_credit"):
        evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=oversized_runner, adapters=adapters, receipt_store=store, require_isolation=False)
    assert len(store) == 0


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
    decisions = RoutingDecisionReceiptStore(
        JsonlRoutingDecisionReceiptRepository(tmp_path / "leadpoet-v2-decisions-test.jsonl")
    )
    def measured_runner(binding, unit, request, authorization):
        assert isinstance(authorization, RoutingCallAuthorization)
        assert authorization.remaining_credit_microunits <= 10
        assert authorization.timeout_ceiling_ms <= 1000
        value = dict(_runner(binding, unit, request))
        value["execution_mode"] = ReceiptExecutionMode.MEASURED_LAB.value
        value["receipt_ref"] = "provider_receipt:" + sha256_json({key: item for key, item in value.items() if key != "receipt_ref"}).split(":", 1)[1][:16]
        return value
    rollup = lambda _store: {"rollup_id": "billing-v2", "rollup_hash": H("7"), "total_credit_microunits": 40}
    evaluation = evaluate_routing_experiment_v2(measured, gold_labels=labels, runner=measured_runner, adapters=adapters, receipt_store=store, decision_store=decisions, authoritative_billing_rollup=rollup, artifact_authority=FakeArtifactAuthority(), require_isolation=False)
    assert evaluation.live_credit_spend is True
    with pytest.raises(RoutingExperimentError, match="promotion_spec_must_be_typed"):
        promote_routing_experiment_v2_to_lab(evaluation)
    promotion = FakePromotionAuthority()
    assert promote_routing_experiment_v2_to_lab(
        evaluation,
        spec=measured,
        authority=promotion,
    ).startswith("sha256:")


def test_v2_live_resume_reconciles_all_cached_receipt_costs(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        experiment_id=spec.experiment_id + "-resume",
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    from research_lab.routing_experiments import JsonlProviderReceiptRepository
    store = ProviderReceiptStore(JsonlProviderReceiptRepository(tmp_path / "resume.jsonl"))
    decision_path = tmp_path / "resume-decisions.jsonl"
    decisions = RoutingDecisionReceiptStore(JsonlRoutingDecisionReceiptRepository(decision_path))
    run_count = [0]

    def measured_runner(binding, unit, request, authorization):
        assert authorization.execution_mode == ReceiptExecutionMode.MEASURED_LAB.value
        run_count[0] += 1
        value = dict(_runner(binding, unit, request))
        value["execution_mode"] = ReceiptExecutionMode.MEASURED_LAB.value
        value["receipt_ref"] = "provider_receipt:" + sha256_json({key: item for key, item in value.items() if key != "receipt_ref"}).split(":", 1)[1][:16]
        return value

    rollups = iter((40, 40))
    first = evaluate_routing_experiment_v2(
        measured,
        gold_labels=labels,
        runner=measured_runner,
        adapters=adapters,
        receipt_store=store,
        decision_store=decisions,
        authoritative_billing_rollup=lambda _store: {"rollup_id": "r1", "rollup_hash": H("7"), "total_credit_microunits": next(rollups)},
        artifact_authority=FakeArtifactAuthority(),
        require_isolation=False,
    )
    assert first.billing_rollup_total_credit_microunits == 40
    assert run_count[0] == 4

    def no_call_runner(binding, unit, request, authorization):
        del binding, unit, request, authorization
        raise AssertionError("resumed exact measured evaluation made a provider call")

    resumed = evaluate_routing_experiment_v2(
        measured,
        gold_labels=labels,
        runner=no_call_runner,
        adapters=adapters,
        receipt_store=store,
        decision_store=RoutingDecisionReceiptStore(JsonlRoutingDecisionReceiptRepository(decision_path)),
        authoritative_billing_rollup=lambda _store: {"rollup_id": "r2", "rollup_hash": H("8"), "total_credit_microunits": next(rollups)},
        artifact_authority=FakeArtifactAuthority(),
        require_isolation=False,
    )
    assert resumed.billing_rollup_total_credit_microunits == 40
    assert resumed.provider_cache_hits == 4
    assert all(item.holdout.total_credit_microunits == 10 for item in resumed.variants)


def test_v2_live_mixed_cache_and_fresh_receipts_reconcile_to_exact_total(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        experiment_id=spec.experiment_id + "-mixed-cache",
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    from research_lab.routing_experiments import JsonlProviderReceiptRepository

    full_store = ProviderReceiptStore(
        JsonlProviderReceiptRepository(tmp_path / "full.jsonl")
    )
    evaluate_routing_experiment_v2(
        measured,
        gold_labels=labels,
        runner=lambda binding, unit, request, authorization: _measured_receipt(
            binding, unit, request
        ),
        adapters=adapters,
        receipt_store=full_store,
        decision_store=RoutingDecisionReceiptStore(
            JsonlRoutingDecisionReceiptRepository(tmp_path / "full-decisions.jsonl")
        ),
        authoritative_billing_rollup=lambda _store: {
            "rollup_id": "full",
            "rollup_hash": H("7"),
            "total_credit_microunits": 40,
        },
        artifact_authority=FakeArtifactAuthority(),
        require_isolation=False,
    )

    partial_store = ProviderReceiptStore(
        JsonlProviderReceiptRepository(tmp_path / "partial.jsonl")
    )
    full_keys = tuple(full_store.repository.keys())
    assert len(full_keys) == 4
    for key in full_keys[:2]:
        receipt = full_store.repository.get(key)
        assert receipt is not None
        partial_store.put(key, receipt)
    fresh_calls = []

    def fresh_runner(binding, unit, request, authorization):
        fresh_calls.append((binding.tool_id, unit))
        return _measured_receipt(binding, unit, request)

    mixed = evaluate_routing_experiment_v2(
        measured,
        gold_labels=labels,
        runner=fresh_runner,
        adapters=adapters,
        receipt_store=partial_store,
        decision_store=RoutingDecisionReceiptStore(
            JsonlRoutingDecisionReceiptRepository(tmp_path / "partial-decisions.jsonl")
        ),
        authoritative_billing_rollup=lambda _store: {
            "rollup_id": "mixed",
            "rollup_hash": H("8"),
            "total_credit_microunits": 40,
        },
        artifact_authority=FakeArtifactAuthority(),
        require_isolation=False,
    )
    assert mixed.billing_rollup_total_credit_microunits == 40
    assert mixed.provider_cache_hits == 2
    assert mixed.provider_cache_misses == 2
    assert len(fresh_calls) == 2


def test_v2_live_replay_does_not_double_count_cached_receipts(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        experiment_id=spec.experiment_id + "-replay",
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    from research_lab.routing_experiments import JsonlProviderReceiptRepository

    store = ProviderReceiptStore(
        JsonlProviderReceiptRepository(tmp_path / "replay.jsonl")
    )
    decisions_path = tmp_path / "replay-decisions.jsonl"
    first = evaluate_routing_experiment_v2(
        measured,
        gold_labels=labels,
        runner=lambda binding, unit, request, authorization: _measured_receipt(
            binding, unit, request
        ),
        adapters=adapters,
        receipt_store=store,
        decision_store=RoutingDecisionReceiptStore(
            JsonlRoutingDecisionReceiptRepository(decisions_path)
        ),
        authoritative_billing_rollup=lambda _store: {
            "rollup_id": "replay-1",
            "rollup_hash": H("7"),
            "total_credit_microunits": 40,
        },
        artifact_authority=FakeArtifactAuthority(),
        require_isolation=False,
    )
    replay = evaluate_routing_experiment_v2(
        measured,
        gold_labels=labels,
        runner=lambda binding, unit, request, authorization: pytest.fail(
            "replay made a provider call"
        ),
        adapters=adapters,
        receipt_store=store,
        decision_store=RoutingDecisionReceiptStore(
            JsonlRoutingDecisionReceiptRepository(decisions_path)
        ),
        authoritative_billing_rollup=lambda _store: {
            "rollup_id": "replay-2",
            "rollup_hash": H("8"),
            "total_credit_microunits": 40,
        },
        artifact_authority=FakeArtifactAuthority(),
        require_isolation=False,
    )
    assert first.billing_rollup_total_credit_microunits == 40
    assert replay.billing_rollup_total_credit_microunits == 40
    assert replay.provider_cache_hits == 4
    assert len(store) == 4


def test_v2_live_billing_accepts_exact_experiment_budget_boundary(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        experiment_id=spec.experiment_id + "-exact-budget",
        credit_budget=replace(
            spec.credit_budget,
            total_credit_microunits=40,
            provider_credit_ceilings={"baseline": 40},
        ),
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    from research_lab.routing_experiments import JsonlProviderReceiptRepository

    evaluation = evaluate_routing_experiment_v2(
        measured,
        gold_labels=labels,
        runner=lambda binding, unit, request, authorization: _measured_receipt(
            binding, unit, request
        ),
        adapters=adapters,
        receipt_store=ProviderReceiptStore(
            JsonlProviderReceiptRepository(tmp_path / "exact-budget.jsonl")
        ),
        decision_store=RoutingDecisionReceiptStore(
            JsonlRoutingDecisionReceiptRepository(tmp_path / "exact-budget-decisions.jsonl")
        ),
        authoritative_billing_rollup=lambda _store: {
            "rollup_id": "exact-budget",
            "rollup_hash": H("9"),
            "total_credit_microunits": 40,
        },
        artifact_authority=FakeArtifactAuthority(),
        require_isolation=False,
    )
    assert evaluation.billing_rollup_total_credit_microunits == 40


def test_v2_route_payload_changes_cannot_reuse_same_request_cache():
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    store = ProviderReceiptStore()
    first_calls = []

    def recording_runner(binding, unit, request):
        first_calls.append(request)
        return _runner(binding, unit, request)

    first = evaluate_routing_experiment_v2(spec, gold_labels=labels, runner=recording_runner, adapters=adapters, receipt_store=store, require_isolation=False)
    first_keys = set(store.repository.keys())
    changed_payload = dict(spec.variants[1].routing_payload)
    changed_payload["skipped"] = {"intent.baseline": "changed_route_payload"}
    changed = replace(spec.variants[1], routing_payload=changed_payload)
    changed_spec = replace(spec, variants=(spec.variants[0], changed))
    second_calls = []
    second = evaluate_routing_experiment_v2(
        changed_spec,
        gold_labels=labels,
        runner=lambda binding, unit, request: (second_calls.append(request) or _runner(binding, unit, request)),
        adapters=adapters,
        receipt_store=store,
        require_isolation=False,
    )
    assert first.experiment_hash != second.experiment_hash
    assert second_calls
    assert set(store.repository.keys()) - first_keys
    assert not set(second_calls).intersection(first_calls)


def test_v2_live_mode_rejects_fixture_or_replay_cache_entries(tmp_path):
    spec, adapters, labels, _tool, _source_tool = _spec("intent_evidence")
    measured = replace(
        spec,
        experiment_id=spec.experiment_id + "-fixture-cache",
        allow_live_credit_spend=True,
        receipt_execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
    )
    from research_lab.routing_experiments import JsonlProviderReceiptRepository
    measured_store = ProviderReceiptStore(JsonlProviderReceiptRepository(tmp_path / "measured.jsonl"))
    decisions = RoutingDecisionReceiptStore(
        JsonlRoutingDecisionReceiptRepository(tmp_path / "measured-decisions.jsonl")
    )

    def measured_runner(binding, unit, request, authorization):
        assert authorization.request_fingerprint == request
        value = dict(_runner(binding, unit, request))
        value["execution_mode"] = ReceiptExecutionMode.MEASURED_LAB.value
        value["receipt_ref"] = "provider_receipt:" + sha256_json({key: item for key, item in value.items() if key != "receipt_ref"}).split(":", 1)[1][:16]
        return value

    evaluate_routing_experiment_v2(
        measured,
        gold_labels=labels,
        runner=measured_runner,
        adapters=adapters,
        receipt_store=measured_store,
        decision_store=decisions,
        authoritative_billing_rollup=lambda _store: {"rollup_id": "m", "rollup_hash": H("7"), "total_credit_microunits": 40},
        artifact_authority=FakeArtifactAuthority(),
        require_isolation=False,
    )
    fixture_store = ProviderReceiptStore(JsonlProviderReceiptRepository(tmp_path / "fixture.jsonl"))
    for key in measured_store.repository.keys():
        receipt = measured_store.repository.get(key)
        assert receipt is not None
        fixture = replace(receipt, execution_mode=ReceiptExecutionMode.FIXTURE.value)
        identity = fixture.to_dict()
        identity.pop("receipt_ref")
        fixture = replace(fixture, receipt_ref="provider_receipt:" + sha256_json(identity).split(":", 1)[1][:16])
        fixture_store.put(key, fixture)
    with pytest.raises(RoutingExperimentError, match="cached_receipt_execution_mode_mismatch"):
        def no_call_runner(binding, unit, request, authorization):
            del binding, unit, request, authorization
            pytest.fail("fixture cache should fail before runner")

        evaluate_routing_experiment_v2(
            measured,
            gold_labels=labels,
            runner=no_call_runner,
            adapters=adapters,
            receipt_store=fixture_store,
            decision_store=RoutingDecisionReceiptStore(
                JsonlRoutingDecisionReceiptRepository(tmp_path / "fixture-decisions.jsonl")
            ),
            authoritative_billing_rollup=lambda _store: {"rollup_id": "m2", "rollup_hash": H("8"), "total_credit_microunits": 0},
            artifact_authority=FakeArtifactAuthority(),
            require_isolation=False,
        )


def test_exact_candidate_model_profile_and_plan_contract():
    model_root = _exact_model_root()
    adapter = PinnedSourcingModelRoutingAdapter.from_model_root(model_root)
    assert adapter.observed_artifact_identity()["commit_sha"] == EXACT_MODEL_SHA
    # SHA 294 has a candidate profile contract but no candidate SourceAdd
    # registration. A candidate experiment must not invent one in the Lab.
    assert adapter.registered_source_add_tool_ids(
        stage="candidate_acquisition"
    ) == ()
    metadata = adapter.candidate_profiles.candidate_routing_profile_metadata()
    feature_set = adapter.parse_feature_set(FEATURES)
    default_payload = metadata["profiles"][0]
    profile = adapter.parse_variant_payload(default_payload, stage="candidate_acquisition")
    assert adapter.routing_change_class(profile, stage="candidate_acquisition") == "default"
    assert adapter.validate_variant_payload(
        profile,
        stage="candidate_acquisition",
        feature_set=feature_set,
        binding_tool_ids=frozenset(item.tool_id for item in adapter.catalog.tools),
        binding_source_lineages={
            item.tool_id: f"lineage.{item.tool_id}"
            for item in adapter.catalog.tools
        },
    ) == []
    missing_lineage_errors = adapter.validate_variant_payload(
        profile,
        stage="candidate_acquisition",
        feature_set=feature_set,
        binding_tool_ids=frozenset(item.tool_id for item in adapter.catalog.tools),
        binding_source_lineages={},
    )
    assert any("candidate_profile_source_lineage_missing" in error for error in missing_lineage_errors)
    wrong_stage_payload = dict(default_payload)
    wrong_stage_payload["steps"] = [
        {**wrong_stage_payload["steps"][0], "tool_id": "intent.jobs_feed"},
        *wrong_stage_payload["steps"][1:],
    ]
    wrong_stage_profile = adapter.parse_variant_payload(
        wrong_stage_payload,
        stage="candidate_acquisition",
    )
    wrong_stage_errors = adapter.validate_variant_payload(
        wrong_stage_profile,
        stage="candidate_acquisition",
        feature_set=feature_set,
        binding_tool_ids=frozenset(item.tool_id for item in adapter.catalog.tools),
        binding_source_lineages={
            item.tool_id: f"lineage.{item.tool_id}"
            for item in adapter.catalog.tools
        },
    )
    assert any("candidate_profile_tool_wrong_stage:intent.jobs_feed" in error for error in wrong_stage_errors)
    assert any("candidate_profile_tool_not_in_policy:intent.jobs_feed" in error for error in wrong_stage_errors)
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
    icp_profile = adapter.parse_variant_payload(icp_payload, stage="candidate_acquisition")
    assert adapter.routing_change_class(icp_profile, stage="candidate_acquisition") == "custom"
    registry = adapter.candidate_profiles.CandidateRoutingProfileRegistry.from_payload({
        "schema_version": "candidate-routing-registry:v1",
        "registry_version": metadata["registry_version"],
        "profiles": [default_payload, icp_payload],
    })
    assert registry.select(features=feature_set).profile_id == "candidate-icp-us"

    intent_metadata = adapter.profiles.routing_profile_metadata()
    intent_default_payload = next(item for item in intent_metadata["profiles"] if item["intent_category"] == "HIRING")
    intent_default = adapter.parse_variant_payload(intent_default_payload, stage="intent_evidence")
    assert adapter.routing_change_class(intent_default, stage="intent_evidence") == "default"
    intent_custom_payload = dict(intent_default_payload)
    intent_custom_payload["profile_id"] = "hiring-icp-us"
    intent_custom_payload["is_default"] = False
    intent_custom_payload["required_features"] = ["company.country.us"]
    intent_custom = adapter.parse_variant_payload(intent_custom_payload, stage="intent_evidence")
    assert adapter.routing_change_class(intent_custom, stage="intent_evidence") == "custom"

    inserted_payload = dict(default_payload)
    inserted_payload["profile_id"] = default_payload["profile_id"]
    inserted_payload["steps"] = [
        {"tool_id": "candidate.source_add.synthetic", "order": 0, "required": False},
        *[
            {**step, "order": int(step["order"]) + 1}
            for step in default_payload["steps"]
        ],
    ]
    inserted = adapter.parse_variant_payload(inserted_payload, stage="candidate_acquisition")
    inserted_errors = adapter.validate_variant_payload(
        inserted,
        stage="candidate_acquisition",
        feature_set=feature_set,
        binding_tool_ids={
            item.tool_id for item in adapter.catalog.tools
        } | {"candidate.source_add.synthetic"},
        binding_source_lineages={
            item.tool_id: f"lineage.{item.tool_id}"
            for item in adapter.catalog.tools
        } | {"candidate.source_add.synthetic": "lineage.synthetic"},
    )
    assert (
        "candidate_profile_tool_missing_from_catalog:"
        "candidate.source_add.synthetic"
    ) in inserted_errors
    assert adapter.routing_identity(inserted, stage="candidate_acquisition", exclude_tool_ids=("candidate.source_add.synthetic",)) == adapter.routing_identity(profile, stage="candidate_acquisition")


def test_exact_model_intent_source_adds_can_be_profiled_bound_and_considered():
    model_root = _exact_model_root()
    adapter = PinnedSourcingModelRoutingAdapter.from_model_root(model_root)
    assert adapter.observed_artifact_identity()["commit_sha"] == EXACT_MODEL_SHA
    feature_set = adapter.parse_feature_set(FEATURES)
    profile_metadata = adapter.profiles.routing_profile_metadata()["profiles"]
    registrations = tuple(
        item
        for item in adapter.runtime.SOURCE_ADD_ROUTING_REGISTRATIONS
        if item.stage == adapter.runtime.STAGE_INTENT_EVIDENCE
    )
    required_tool_ids = {
        "intent.source_add.bloomberry",
        "intent.source_add.bloomberry_jobs",
        "intent.source_add.builtwith",
        "intent.source_add.podscan",
        "intent.source_add.predictleads_connections",
        "intent.source_add.predictleads_financing",
        "intent.source_add.predictleads_jobs",
        "intent.source_add.predictleads_news",
        "intent.source_add.predictleads_technology",
        "intent.source_add.sumble",
    }
    registered_tool_ids = set(
        adapter.registered_source_add_tool_ids(stage="intent_evidence")
    )
    assert registered_tool_ids == required_tool_ids
    assert {item.tool_id for item in registrations} == registered_tool_ids

    for registration in registrations:
        signal_type = registration.intent_categories[0]
        base_payload = next(
            item
            for item in profile_metadata
            if item["intent_category"] == signal_type
        )
        phase = (
            "confirmation"
            if "routing.confirmation_only" in registration.capabilities
            else "primary"
        )
        payload = dict(base_payload)
        payload["profile_id"] = f"exact-source-add-{registration.provider_id}"
        payload["is_default"] = False
        steps = [dict(item) for item in payload["steps"]]
        for step in steps:
            if step["phase"] == phase:
                step["order"] = int(step["order"]) + 1
        steps.append(
            {
                "tool_id": registration.tool_id,
                "phase": phase,
                "order": 0,
                "source_lineage_id": registration.route_name,
                "required": False,
            }
        )
        payload["steps"] = steps
        payload["call_cap"] = 100
        payload["credit_cap"] = 1_000.0
        payload["challenger_call_cap"] = 10
        payload["challenger_credit_cap"] = 1_000.0
        profile = adapter.parse_variant_payload(payload, stage="intent_evidence")
        assert registration.tool_id in {
            step.tool_id for step in profile.steps
        }
        lineages = {
            step.tool_id: step.source_lineage_id for step in profile.steps
        }
        assert adapter.validate_variant_payload(
            profile,
            stage="intent_evidence",
            feature_set=feature_set,
            binding_tool_ids=frozenset(lineages),
            binding_source_lineages=lineages,
            expected_signal_type=signal_type,
        ) == []
        descriptor = next(
            item
            for item in adapter.variant_tool_descriptors(
                profile, stage="intent_evidence"
            )
            if item["tool_id"] == registration.tool_id
        )
        assert descriptor["source_add"] is True
        catalog_descriptor = adapter.lookup_tool_descriptor(
            registration.tool_id, stage="intent_evidence"
        )
        assert catalog_descriptor is not None
        assert descriptor["manifest_hash"] == catalog_descriptor["manifest_hash"]
        binding = ProviderBindingIdentity(
            binding_id=f"binding-{registration.provider_id}",
            provider_id=registration.provider_id,
            tool_id=registration.tool_id,
            source_lineage_id=registration.route_name,
            adapter_version="exact-model-test",
            manifest_hash=descriptor["manifest_hash"],
            capability_hash=H("1"),
            execution_contract_hash=H("2"),
            cost_model_hash=H("3"),
        )
        assert adapter.validate_provider_binding(
            binding, stage="intent_evidence"
        ) == ()
        plan = adapter.compile_variant(
            profile,
            stage="intent_evidence",
            feature_set=feature_set,
            available_tools={item.tool_id: True for item in adapter.catalog.tools},
            remaining_seconds=900,
            remaining_calls=100,
            credit_cap=1_000,
            expected_signal_type=signal_type,
        )
        assert registration.tool_id in adapter.considered_tool_ids(
            plan, stage="intent_evidence"
        )


def _assert_v2_rejected_before_provider_calls(
    spec,
    adapters,
    labels,
    *,
    error_match: str,
):
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _runner(*args[:3])

    with pytest.raises(RoutingExperimentError, match=error_match):
        evaluate_routing_experiment_v2(
            spec,
            gold_labels=labels,
            runner=recording_runner,
            adapters=adapters,
            require_isolation=False,
        )
    assert calls == []


def test_v2_source_add_false_model_descriptor_is_rejected_before_provider_calls():
    spec, adapters, labels, _tool, source_tool = _spec(
        "intent_evidence", with_source_add=True
    )

    class FalseSourceAddDescriptorAdapter(FakeV2Adapter):
        def variant_tool_descriptors(self, payload, *, stage):
            return tuple(
                {
                    **descriptor,
                    "source_add": False,
                }
                if descriptor["tool_id"] == source_tool
                else descriptor
                for descriptor in super().variant_tool_descriptors(
                    payload, stage=stage
                )
            )

    invalid_adapters = {
        "baseline": adapters["baseline"],
        "candidate": FalseSourceAddDescriptorAdapter(
            adapters["candidate"].manifests
        ),
    }
    errors = validate_routing_experiment_v2_spec(
        spec,
        adapters=invalid_adapters,
    )
    assert f"v2_new_tool_is_not_source_add:{source_tool}" in errors
    _assert_v2_rejected_before_provider_calls(
        spec,
        invalid_adapters,
        labels,
        error_match="v2_new_tool_is_not_source_add",
    )


def test_v2_source_add_catalog_binding_mismatch_is_rejected_before_provider_calls():
    spec, adapters, labels, _tool, source_tool = _spec(
        "intent_evidence", with_source_add=True
    )
    source_binding = next(
        item
        for item in spec.provider_bindings
        if item.tool_id == source_tool
    )
    invalid = replace(
        spec,
        experiment_id="source-add-binding-manifest-mismatch",
        provider_bindings=tuple(
            replace(item, manifest_hash=H("9"))
            if item.binding_id == source_binding.binding_id
            else item
            for item in spec.provider_bindings
        ),
    )
    errors = validate_routing_experiment_v2_spec(
        invalid,
        adapters=adapters,
    )
    assert any(
        source_tool in error
        and (
            "provider_binding_manifest_mismatch" in error
            or "source_add_provider_binding_manifest_hash_mismatch" in error
        )
        for error in errors
    )
    _assert_v2_rejected_before_provider_calls(
        invalid,
        adapters,
        labels,
        error_match="provider_binding_manifest_mismatch|source_add_provider_binding_manifest_hash_mismatch",
    )


def test_v2_source_add_omitted_binding_id_is_rejected_before_provider_calls():
    spec, adapters, labels, _tool, source_tool = _spec(
        "intent_evidence", with_source_add=True
    )
    invalid_variant = replace(
        spec.variants[1],
        variant_id="source-add-binding-omitted",
        binding_ids=("baseline",),
    )
    invalid = replace(
        spec,
        experiment_id="source-add-binding-omitted",
        variants=(spec.variants[0], invalid_variant),
    )
    invalid_adapters = {
        "baseline": adapters["baseline"],
        invalid_variant.variant_id: adapters["candidate"],
    }
    errors = validate_routing_experiment_v2_spec(
        invalid,
        adapters=invalid_adapters,
    )
    assert any(
        f"v2_source_add_binding_missing:{source_tool}" in error
        for error in errors
    )
    _assert_v2_rejected_before_provider_calls(
        invalid,
        invalid_adapters,
        labels,
        error_match="v2_source_add_binding_missing",
    )


@pytest.mark.parametrize("mode", ["omitted", "considered_only"])
def test_v2_declared_source_add_must_be_invokable_in_compiled_plan_before_calls(
    mode,
):
    spec, adapters, labels, baseline_tool, source_tool = _spec(
        "intent_evidence", with_source_add=True
    )

    class NonInvokableSourceAddAdapter(FakeV2Adapter):
        def compile_variant(self, payload, **kwargs):
            raw = dict(payload)
            raw["tools"] = [baseline_tool]
            if mode == "considered_only":
                raw["considered"] = [source_tool]
            return super().compile_variant(raw, **kwargs)

    invalid_adapters = {
        "baseline": adapters["baseline"],
        "candidate": NonInvokableSourceAddAdapter(
            adapters["candidate"].manifests
        ),
    }
    _assert_v2_rejected_before_provider_calls(
        spec,
        invalid_adapters,
        labels,
        error_match="v2_declared_new_tool_missing_from_compiled_plan",
    )
class FakeCompanyFirstContinuationAdapter:
    action_types = (
        "execute_candidate_tool",
        "verify_company",
        "execute_intent_tool",
        "verify_intent",
        "execute_contact_tool",
        "verify_contact",
    )
    stages = {
        "execute_candidate_tool": "candidate_acquisition",
        "verify_company": "company_qualification",
        "execute_intent_tool": "intent_evidence",
        "verify_intent": "intent_verification",
        "execute_contact_tool": "contact_acquisition",
        "verify_contact": "contact_verification",
    }

    def __init__(self, artifact: SourcingModelArtifactIdentity) -> None:
        self.observed_artifact_key = sha256_json({
            "model_artifact_hash": artifact.model_artifact_hash,
            "manifest_hash": artifact.manifest_hash,
            "commit_sha": artifact.commit_sha,
        })

    def advance(self, request, *, continuation, completion):
        assert request["normalized_icp"] == {"segments_any_of": ["software"]}
        completed = list(
            continuation["completed_actions"] if continuation else ()
        )
        if completion is not None:
            completed.append(dict(completion))
        sequence = len(completed)
        if sequence == len(self.action_types):
            return {
                "schema_version": COMPANY_FIRST_CONTINUATION_SCHEMA_VERSION,
                "status": "completed",
                "action": None,
                "continuation": {
                    "schema_version": COMPANY_FIRST_CONTINUATION_SCHEMA_VERSION,
                    "input_sha256": "1" * 64,
                    "completed_actions": completed,
                    "pending_action": None,
                    "continuation_sha256": "2" * 64,
                },
                "result": {
                    "leads": [],
                    "receipt": {"stop_reason": "fixture_complete"},
                },
            }
        action_type = self.action_types[sequence]
        action = {
            "schema_version": "company-first-action-request:v1",
            "action_id": f"{sequence + 3:064x}",
            "sequence": sequence,
            "action_type": action_type,
            "stage": self.stages[action_type],
            "request_sha256": f"{sequence + 13:064x}",
            "arguments": {
                "tool_id": "candidate.fixture"
                if action_type == "execute_candidate_tool"
                else "intent.fixture"
                if action_type == "execute_intent_tool"
                else "contact.fixture"
                if action_type == "execute_contact_tool"
                else None,
            },
        }
        state = {
            "schema_version": COMPANY_FIRST_CONTINUATION_SCHEMA_VERSION,
            "input_sha256": "1" * 64,
            "completed_actions": completed,
            "pending_action": action,
            "continuation_sha256": f"{sequence + 23:064x}",
        }
        return {
            "schema_version": COMPANY_FIRST_CONTINUATION_SCHEMA_VERSION,
            "status": "action_required",
            "action": action,
            "continuation": state,
        }

    def complete_action(self, action, result):
        return {
            "schema_version": "company-first-action-completion:v1",
            "action_id": action["action_id"],
            "action_type": action["action_type"],
            "request_sha256": action["request_sha256"],
            "result_sha256": sha256_json(result).split(":", 1)[1],
            "result": result,
        }


def test_company_first_lab_replay_preserves_all_model_action_types() -> None:
    artifact, _authority_manifest = _artifact()
    adapter = FakeCompanyFirstContinuationAdapter(artifact)
    handle = IsolatedCompanyFirstContinuationAdapter(
        adapter=adapter,
        process_id="company-first-worker-1",
        observed_artifact_key=adapter.observed_artifact_key,
    )
    invoked: list[str] = []

    replay = replay_company_first_orchestration(
        {
            "normalized_icp": {"segments_any_of": ["software"]},
            "target": 1,
        },
        artifact=artifact,
        adapter=handle,
        runner=lambda action: invoked.append(action["action_type"]) or {
            "outcome": "fixture",
        },
    )

    assert isinstance(replay, CompanyFirstLabReplay)
    assert invoked == list(FakeCompanyFirstContinuationAdapter.action_types)
    assert [item.action_type for item in replay.actions] == invoked
    assert replay.result["receipt"]["stop_reason"] == "fixture_complete"
    assert replay.final_continuation_hash == H("2")
    assert replay.replay_hash().startswith("sha256:")


def test_company_first_lab_replay_fails_closed_on_identity_and_live_mode() -> None:
    artifact, _authority_manifest = _artifact()
    adapter = FakeCompanyFirstContinuationAdapter(artifact)
    handle = IsolatedCompanyFirstContinuationAdapter(
        adapter=adapter,
        process_id="company-first-worker-1",
        observed_artifact_key=adapter.observed_artifact_key,
    )
    request = {
        "normalized_icp": {"segments_any_of": ["software"]},
        "target": 1,
    }

    with pytest.raises(
        RoutingExperimentError,
        match="company_first_adapter_artifact_mismatch",
    ):
        replay_company_first_orchestration(
            request,
            artifact=replace(artifact, commit_sha="2" * 40),
            adapter=handle,
            runner=lambda _action: {},
        )

    with pytest.raises(
        RoutingExperimentError,
        match="company_first_live_execution_is_not_supported",
    ):
        replay_company_first_orchestration(
            request,
            artifact=artifact,
            adapter=handle,
            runner=lambda _action: {},
            execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
        )

    with pytest.raises(
        RoutingExperimentError,
        match="company_first_isolated_model_adapter_is_required",
    ):
        replay_company_first_orchestration(
            request,
            artifact=artifact,
            adapter=adapter,
            runner=lambda _action: {},
        )
