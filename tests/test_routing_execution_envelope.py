from __future__ import annotations

from dataclasses import replace

import pytest

from gateway.research_lab.routing_execution_envelope import (
    RoutingExecutionEnvelopeError,
    RoutingExperimentExecutionEnvelopeV2,
    build_routing_execution_envelope_v2,
    validate_routing_execution_envelope_v2,
)
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
    VerifiedRoutingGoldLabels,
)
from gateway.research_lab.routing_experiment_store import (
    RoutingExperimentStoreError,
    SupabaseRoutingExperimentStore,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
    build_routing_model_binding_observation_result_v2,
    routing_model_binding_identity_hash,
)
from research_lab.canonical import sha256_json
from gateway.research_lab.routing_provider_bindings import (
    RoutingBindingManifest,
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from tests.routing_experiment_authority_fixture import authority_fixture, _signed_receipt


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _authorities():
    fixture = authority_fixture()
    spec = replace(
        fixture["spec"],
        allow_live_credit_spend=True,
        receipt_execution_mode="measured_lab",
    )
    binding = spec.provider_bindings[0]
    ceiling = spec.credit_budget.provider_credit_ceilings[binding.binding_id]
    manifest = RoutingBindingManifest(
        binding=binding,
        compiler_family="deepline_reviewed_action_v1",
        transport_id="deepline",
        action_id="bloomberry_search_job_postings",
        input_projection={"domain": "company_domain"},
        input_constants={},
        model_binding_requirements_hash=_hash("1"),
        output_contract_hash=_hash("2"),
        evidence_contract_hash=_hash("3"),
        retry_policy_hash=_hash("4"),
        max_results=1,
        timeout_ms=1_000,
        credit_ceiling_microunits=ceiling,
    )
    catalog = VerifiedRoutingBindingCatalog(
        manifest_uri="s3://routing/releases/catalog-1.json",
        manifest_hash=_hash("5"),
        signature_ref="s3://routing/signatures/catalog-1.sig",
        signing_key_id="kms-routing",
        catalog_version="catalog-1",
        bindings={manifest.identity_key(): manifest},
    )
    units = VerifiedRoutingUnitDataset(
        manifest_uri="s3://routing/releases/units-1.json",
        manifest_hash=_hash("6"),
        signature_ref="s3://routing/signatures/units-1.sig",
        signing_key_id="kms-units",
        unit_set_hash=spec.input.unit_input_set_hash,
        provenance_hash=_hash("7"),
        units={
            unit_ref: {"company_domain": f"{unit_ref}.example"}
            for unit_ref in (
                *spec.input.calibration_unit_refs,
                *spec.input.holdout_unit_refs,
            )
        },
    )
    labels = VerifiedRoutingGoldLabels(**fixture["labels"])
    lineage = VerifiedRoutingArtifactLineage(**fixture["lineage"])
    observation_result = build_routing_model_binding_observation_result_v2(
        artifact_lineage_hash=lineage.identity_hash(),
        requirement_hash_by_binding_identity={
            routing_model_binding_identity_hash(binding): (
                manifest.model_binding_requirements_hash
            )
        },
    )
    observation = VerifiedRoutingModelBindingRequirements.from_attested(
        observation_result,
        _signed_receipt(
            purpose="research_lab.routing_model_binding_observation.v2",
            input_root=observation_result["request_root"],
            output_root=sha256_json(observation_result),
            index=92,
        ),
    )
    return spec, catalog, units, labels, lineage, observation


def _envelope():
    spec, catalog, units, labels, lineage, observation = _authorities()
    envelope = build_routing_execution_envelope_v2(
        spec=spec,
        artifact_lineage=lineage,
        binding_catalog=catalog,
        unit_dataset=units,
        gold_labels=labels,
        model_binding_observation=observation,
    )
    return spec, catalog, units, labels, lineage, envelope


def test_execution_envelope_binds_exact_spec_catalog_and_signed_authorities():
    spec, catalog, _units, _labels, _lineage, envelope = _envelope()
    validate_routing_execution_envelope_v2(
        spec=spec,
        envelope=envelope,
        binding_catalog=catalog,
    )
    assert RoutingExperimentExecutionEnvelopeV2.from_mapping(
        envelope.to_dict()
    ).envelope_hash() == envelope.envelope_hash()


def test_execution_envelope_rejects_spec_hash_missing_extra_and_action_substitution():
    spec, catalog, _units, _labels, _lineage, envelope = _envelope()
    with pytest.raises(RoutingExecutionEnvelopeError, match="experiment hash"):
        validate_routing_execution_envelope_v2(
            spec=replace(spec, experiment_id=spec.experiment_id + "-other"),
            envelope=envelope,
        )

    extra = replace(envelope.bindings[0], binding_id="zz-extra-binding")
    with pytest.raises(RoutingExecutionEnvelopeError, match="binding set"):
        validate_routing_execution_envelope_v2(
            spec=spec,
            envelope=replace(
                envelope,
                bindings=tuple(sorted((*envelope.bindings, extra), key=lambda item: item.binding_id)),
            ),
        )

    substituted = replace(
        envelope,
        bindings=(replace(envelope.bindings[0], action_id="podscan_episodes_search"),),
    )
    with pytest.raises(RoutingExecutionEnvelopeError, match="catalog binding"):
        validate_routing_execution_envelope_v2(
            spec=spec,
            envelope=substituted,
            binding_catalog=catalog,
        )


def test_execution_envelope_rejects_unit_label_and_artifact_mismatch():
    spec, catalog, units, labels, lineage, observation = _authorities()
    common = {
        "spec": spec,
        "binding_catalog": catalog,
        "model_binding_observation": observation,
    }
    with pytest.raises(RoutingExecutionEnvelopeError, match="unit dataset"):
        build_routing_execution_envelope_v2(
            **common,
            artifact_lineage=lineage,
            unit_dataset=replace(units, unit_set_hash=_hash("9")),
            gold_labels=labels,
        )
    with pytest.raises(RoutingExecutionEnvelopeError, match="gold labels"):
        build_routing_execution_envelope_v2(
            **common,
            artifact_lineage=lineage,
            unit_dataset=units,
            gold_labels=replace(labels, label_set_hash=_hash("a")),
        )
    with pytest.raises(RoutingExecutionEnvelopeError, match="artifact"):
        build_routing_execution_envelope_v2(
            **common,
            artifact_lineage=replace(lineage, commit_sha="b" * 40),
            unit_dataset=units,
            gold_labels=labels,
        )


def test_live_store_submit_fails_before_rpc_without_execution_envelope():
    spec, _catalog, _units, _labels, _lineage, _observation = _authorities()

    class _NoRpcClient:
        def rpc(self, *_args, **_kwargs):
            raise AssertionError("live submit without an envelope reached SQL")

    with pytest.raises(RoutingExperimentStoreError, match="requires an execution envelope"):
        SupabaseRoutingExperimentStore(client=_NoRpcClient()).submit(spec)
