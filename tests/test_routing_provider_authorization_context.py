from __future__ import annotations

from dataclasses import replace

import pytest

from gateway.research_lab.routing_admission import RoutingAdmissionBundleV2
from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
    RoutingProviderAuthorizationError,
    build_routing_provider_authorization_request_v2,
    validate_routing_provider_authorization_context_v2,
    validate_routing_provider_authorization_request_v2,
)
from gateway.research_lab.routing_execution_envelope import (
    RoutingExecutionBindingV2,
    RoutingExperimentExecutionEnvelopeV2,
)
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
    build_routing_model_binding_observation_result_v2,
    routing_model_binding_identity_hash,
)
from gateway.research_lab.routing_provider_bindings import (
    DEEPLINE_ACTION_POLICIES,
    PreparedRoutingProviderCall,
    RoutingBindingManifest,
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import ProviderBindingIdentity
from tests.routing_experiment_authority_fixture import (
    _hash,
    _routing_protected_receipt,
    _signed_receipt,
)


def _context():
    binding = ProviderBindingIdentity(
        binding_id="binding-1",
        provider_id="bloomberry",
        tool_id="intent.source_add.bloomberry_jobs",
        source_lineage_id="model-source-1",
        adapter_version="adapter-v1",
        manifest_hash=_hash("1"),
        capability_hash=_hash("2"),
        execution_contract_hash=_hash("3"),
        cost_model_hash=_hash("4"),
    )
    action_id = "bloomberry_search_job_postings"
    policy = DEEPLINE_ACTION_POLICIES[action_id]
    manifest = RoutingBindingManifest(
        binding=binding,
        compiler_family="deepline_reviewed_action_v1",
        transport_id="deepline",
        action_id=action_id,
        input_projection={"domain": "domain"},
        input_constants={},
        model_binding_requirements_hash=_hash("5"),
        output_contract_hash=_hash("6"),
        evidence_contract_hash=_hash("7"),
        retry_policy_hash=_hash("8"),
        max_results=1,
        timeout_ms=5_000,
        credit_ceiling_microunits=90_000,
    )
    catalog = VerifiedRoutingBindingCatalog(
        manifest_uri="s3://catalog/releases/catalog-1.json",
        manifest_hash=_hash("9"),
        signature_ref="s3://catalog/signatures/catalog-1.sig",
        signing_key_id="kms-catalog",
        catalog_version="catalog-1",
        bindings={manifest.identity_key(): manifest},
    )
    unit_dataset = VerifiedRoutingUnitDataset(
        manifest_uri="s3://units/releases/units-1.json",
        manifest_hash=_hash("a"),
        signature_ref="s3://units/signatures/units-1.sig",
        signing_key_id="kms-units",
        unit_set_hash=_hash("b"),
        provenance_hash=_hash("c"),
        units={"unit-1": {"domain": "example.com"}},
    )
    lineage = VerifiedRoutingArtifactLineage(
        repository="leadpoet/Sourcing_model",
        branch="leadpoet-lab",
        # The private Sourcing_model artifact and the protected scoring
        # release are separate release lines.  Their commit/config identities
        # must be verified independently and need not be equal.
        commit_sha="2" * 40,
        pointer_uri="s3://model/branches/leadpoet-lab/current.json",
        pointer_document_hash=_hash("d"),
        immutable_manifest_uri="s3://model/releases/model-1.json",
        routing_lineage_manifest_uri="s3://model/releases/routing-1.json",
        routing_lineage_manifest_hash=_hash("e"),
        manifest_hash=_hash("f"),
        signature_ref="s3://model/signatures/model-1.sig",
        signature_key_id="kms-model",
        signature_algorithm="ECDSA_SHA_256",
        model_artifact_hash=_hash("0"),
        image_digest="model@" + _hash("1"),
        config_hash=_hash("9"),
        build_id="model-build-1",
        component_registry_version="components-v1",
        scoring_adapter_version="adapter-v1",
        routing_contract_hash=_hash("3"),
        routing_catalog_hash=_hash("4"),
        routing_policy_hash=_hash("5"),
        feature_schema_hash=_hash("6"),
        verifier_contract_hash=_hash("7"),
    )
    observation_result = build_routing_model_binding_observation_result_v2(
        artifact_lineage_hash=lineage.identity_hash(),
        requirement_hash_by_binding_identity={
            routing_model_binding_identity_hash(binding): _hash("5")
        },
    )
    observation = VerifiedRoutingModelBindingRequirements.from_attested(
        observation_result,
        _signed_receipt(
            purpose="research_lab.routing_model_binding_observation.v2",
            input_root=observation_result["request_root"],
            output_root=sha256_json(observation_result),
            index=701,
        ),
    )
    envelope = RoutingExperimentExecutionEnvelopeV2(
        experiment_hash=_hash("8"),
        artifact_lineage_hash=lineage.identity_hash(),
        pointer_document_hash=lineage.pointer_document_hash,
        binding_catalog_manifest_hash=catalog.manifest_hash,
        binding_catalog_version=catalog.catalog_version,
        unit_dataset_manifest_hash=unit_dataset.manifest_hash,
        unit_set_hash=unit_dataset.unit_set_hash,
        gold_label_manifest_hash=_hash("9"),
        model_binding_observation_receipt_hash=observation.observation_receipt_hash,
        model_binding_observation=observation.to_attested_dict(),
        bindings=(
            RoutingExecutionBindingV2(
                binding_id=binding.binding_id,
                provider_id=binding.provider_id,
                tool_id=binding.tool_id,
                binding_manifest_hash=binding.manifest_hash,
                action_id=action_id,
                compiler_family=manifest.compiler_family,
                transport_id=manifest.transport_id,
                model_binding_requirements_hash=manifest.model_binding_requirements_hash,
                output_contract_hash=manifest.output_contract_hash,
                evidence_contract_hash=manifest.evidence_contract_hash,
                retry_policy_hash=manifest.retry_policy_hash,
                credit_ceiling_microunits=manifest.credit_ceiling_microunits,
                timeout_ms=manifest.timeout_ms,
            ),
        ),
    )
    protected_receipt, _protected_key, _protected_pubkey = (
        _routing_protected_receipt()
    )
    release = {
        "protected_receipt_hash": protected_receipt["receipt_hash"],
        "protected_commit_sha": protected_receipt["commit_sha"],
        "protected_pcr0": protected_receipt["pcr0"],
        "protected_build_manifest_hash": protected_receipt["build_manifest_hash"],
        "protected_dependency_lock_hash": protected_receipt[
            "dependency_lock_hash"
        ],
        "protected_config_hash": protected_receipt["config_hash"],
        "protected_boot_identity_hash": protected_receipt["boot_identity_hash"],
        "protected_enclave_pubkey": protected_receipt["enclave_pubkey"],
    }
    admission = RoutingAdmissionBundleV2(
        job_id="routing-job",
        experiment_id="experiment-1",
        experiment_hash=envelope.experiment_hash,
        role="gateway_scoring",
        purpose="research_lab.routing_provider_evidence.v2",
        envelope_hash=envelope.envelope_hash(),
        artifact_lineage_hash=lineage.identity_hash(),
        pointer_document_hash=lineage.pointer_document_hash,
        immutable_manifest_hash=lineage.manifest_hash,
        model_artifact_hash=lineage.model_artifact_hash,
        gold_label_manifest_hash=envelope.gold_label_manifest_hash,
        gold_label_set_hash=_hash("a"),
        unit_dataset_manifest_hash=unit_dataset.manifest_hash,
        unit_set_hash=unit_dataset.unit_set_hash,
        binding_catalog_manifest_hash=catalog.manifest_hash,
        binding_catalog_version=catalog.catalog_version,
        model_binding_observation_hash=sha256_json(dict(observation.result)),
        model_binding_observation_receipt_hash=observation.observation_receipt_hash,
        binding_ids=(binding.binding_id,),
        protected_release_hash=sha256_json(
            {"schema_version": "leadpoet.routing_protected_release.v2", **release}
        ),
        protected_commit_sha=protected_receipt["commit_sha"],
        protected_pcr0=protected_receipt["pcr0"],
        protected_build_manifest_hash=protected_receipt["build_manifest_hash"],
        protected_dependency_lock_hash=protected_receipt["dependency_lock_hash"],
        protected_config_hash=protected_receipt["config_hash"],
        protected_boot_identity_hash=protected_receipt["boot_identity_hash"],
        protected_enclave_pubkey=protected_receipt["enclave_pubkey"],
        protected_receipt_hash=protected_receipt["receipt_hash"],
    )
    unit_input, unit_input_hash = unit_dataset.resolve("unit-1")
    prepared_body = {
        "provider": manifest.binding.provider_id,
        "operation": policy.operation,
        "payload": {"domain": unit_input["domain"]},
    }
    prepared = PreparedRoutingProviderCall(
        binding=binding,
        binding_manifest_hash=binding.manifest_hash,
        binding_catalog_manifest_hash=catalog.manifest_hash,
        binding_catalog_version=catalog.catalog_version,
        unit_ref="unit-1",
        unit_input_hash=unit_input_hash,
        unit_dataset_manifest_hash=unit_dataset.manifest_hash,
        unit_set_hash=unit_dataset.unit_set_hash,
        model_binding_requirements_hash=manifest.model_binding_requirements_hash,
        action_id=action_id,
        transport_id=manifest.transport_id,
        provider=manifest.binding.provider_id,
        operation=policy.operation,
        payload=prepared_body["payload"],
        validation_context={},
        validation_context_hash=sha256_json(
            {
                "schema_version": "leadpoet.routing_validation_context.v1",
                "action_id": action_id,
                "context": {},
            }
        ),
        request_body_hash=sha256_json(prepared_body),
        timeout_ms=manifest.timeout_ms,
        credit_ceiling_microunits=manifest.credit_ceiling_microunits,
        max_results=manifest.max_results,
        retry_policy_hash=manifest.retry_policy_hash,
        evidence_contract_hash=manifest.evidence_contract_hash,
        output_contract_hash=manifest.output_contract_hash,
    )
    grant = RoutingProviderCallAuthorizationV2(
        admission_job_id=admission.job_id,
        experiment_hash=envelope.experiment_hash,
        experiment_id=admission.experiment_id,
        purpose="research_lab.routing_provider_evidence.v2",
        envelope_hash=envelope.envelope_hash(),
        admission_bundle_hash=admission.identity_hash(),
        protected_release_hash=admission.protected_release_hash,
        protected_boot_identity_hash=admission.protected_boot_identity_hash,
        variant_id="variant-1",
        stage="intent_evidence",
        artifact_lineage_hash=lineage.identity_hash(),
        pointer_document_hash=lineage.pointer_document_hash,
        model_artifact_hash=lineage.model_artifact_hash,
        manifest_hash=lineage.manifest_hash,
        image_digest=lineage.image_digest,
        commit_sha=lineage.commit_sha,
        build_id=lineage.build_id,
        routing_contract_hash=lineage.routing_contract_hash,
        routing_catalog_hash=lineage.routing_catalog_hash,
        routing_policy_hash=lineage.routing_policy_hash,
        feature_schema_hash=lineage.feature_schema_hash,
        verifier_contract_hash=lineage.verifier_contract_hash,
        binding=binding,
        transport_id=prepared.transport_id,
        binding_catalog_manifest_hash=catalog.manifest_hash,
        binding_catalog_version=catalog.catalog_version,
        action_id=action_id,
        unit_ref=prepared.unit_ref,
        unit_input_hash=prepared.unit_input_hash,
        unit_dataset_manifest_hash=unit_dataset.manifest_hash,
        unit_set_hash=unit_dataset.unit_set_hash,
        model_binding_observation_receipt_hash=observation.observation_receipt_hash,
        attempt=0,
        core_request_fingerprint=_hash("0"),
        request_body_hash=prepared.request_body_hash,
        retry_policy_hash=prepared.retry_policy_hash,
        credit_cap_microunits=prepared.credit_ceiling_microunits,
        timeout_ms=prepared.timeout_ms,
        claim_key=_hash("1"),
        claim_generation=1,
        claim_fence_hash=_hash("2"),
    )
    return {
        "grant": grant,
        "lineage": lineage,
        "catalog": catalog,
        "unit_dataset": unit_dataset,
        "observation": observation,
        "envelope": envelope,
        "admission": admission,
        "prepared": prepared,
        "protected_receipt": protected_receipt,
    }


def _validate(context):
    validate_routing_provider_authorization_context_v2(
        context["grant"],
        artifact_lineage=context["lineage"],
        binding_catalog=context["catalog"],
        unit_dataset=context["unit_dataset"],
        model_binding_observation=context["observation"],
        execution_envelope=context["envelope"],
        admission_bundle=context["admission"],
        prepared_call=context["prepared"],
    )


def test_routing_provider_authorization_context_accepts_exact_authorities():
    _validate(_context())


def test_routing_provider_authorization_request_accepts_exact_signed_context():
    context = _context()
    request = build_routing_provider_authorization_request_v2(
        authorization=context["grant"],
        artifact_lineage=context["lineage"],
        model_binding_observation=context["observation"],
        execution_envelope=context["envelope"],
        admission_bundle=context["admission"],
        prepared_call=context["prepared"],
        protected_release_receipt=context["protected_receipt"],
    )
    grant, observation, envelope, admission, protected_receipt = (
        validate_routing_provider_authorization_request_v2(
            request,
            artifact_lineage=context["lineage"],
            binding_catalog=context["catalog"],
            unit_dataset=context["unit_dataset"],
        )
    )
    assert grant == context["grant"]
    assert observation.to_attested_dict() == context["observation"].to_attested_dict()
    assert envelope == context["envelope"]
    assert admission == context["admission"]
    assert protected_receipt == context["protected_receipt"]


@pytest.mark.parametrize("field", ("lineage", "catalog", "unit_dataset", "observation", "envelope", "admission", "prepared"))
def test_routing_provider_authorization_context_rejects_authority_substitution(field):
    context = _context()
    if field == "lineage":
        context[field] = replace(context[field], build_id="model-build-substituted")
    elif field == "catalog":
        context[field] = replace(context[field], manifest_hash=_hash("3"))
    elif field == "unit_dataset":
        context[field] = replace(context[field], units={"unit-1": {"domain": "substituted.example"}})
    elif field == "observation":
        result = build_routing_model_binding_observation_result_v2(
            artifact_lineage_hash=context["lineage"].identity_hash(),
            requirement_hash_by_binding_identity={
                routing_model_binding_identity_hash(context["grant"].binding): _hash("4")
            },
        )
        context[field] = VerifiedRoutingModelBindingRequirements.from_attested(
            result,
            _signed_receipt(
                purpose="research_lab.routing_model_binding_observation.v2",
                input_root=result["request_root"],
                output_root=sha256_json(result),
                index=702,
            ),
        )
    elif field == "envelope":
        context[field] = replace(context[field], experiment_hash=_hash("1"))
    elif field == "admission":
        context[field] = replace(context[field], protected_release_hash=_hash("3"))
    else:
        context[field] = replace(context[field], action_id="substituted_action")
    with pytest.raises(RoutingProviderAuthorizationError):
        _validate(context)
