"""Canonical redacted authority rows for routing promotion tests."""

from __future__ import annotations

from dataclasses import replace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from leadpoet_canonical.attested_v2 import (
    build_execution_receipt_body,
    create_signed_execution_receipt,
    merkle_root,
)

from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
    execute_routing_provider_call_authorization_v2,
)
from gateway.research_lab.routing_admission import build_routing_admission_bundle_v2
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
    VerifiedRoutingGoldLabels,
)
from gateway.research_lab.routing_provider_bindings import (
    RoutingBindingManifest,
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from gateway.research_lab.routing_provider_terminal import (
    build_routing_provider_terminal_body_v2,
    sign_routing_provider_terminal_v2,
)
from gateway.tee.provider_evidence_v2 import create_signed_provider_evidence_record
from gateway.research_lab.routing_execution_envelope import (
    RoutingExecutionBindingV2,
    RoutingExperimentExecutionEnvelopeV2,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
    build_routing_model_binding_observation_result_v2,
    routing_model_binding_identity_hash,
)

from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderReceiptStore,
    ReceiptExecutionMode,
    RoutingDecisionReceiptStore,
    SourcingModelArtifactIdentity,
    evaluate_routing_experiment_v2,
    provider_receipt_key,
)
from tests.test_intent_routing_experiments_v2 import _authority, _runner, _spec


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _signed_receipt(
    *,
    purpose: str,
    input_root: str,
    output_root: str,
    index: int,
    job_id: str | None = None,
):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose=purpose,
            job_id=job_id or f"routing-authority-{index}",
            epoch_id=24_300,
            sequence=index,
            commit_sha="9" * 40,
            pcr0="8" * 96,
            build_manifest_hash=_hash("7"),
            dependency_lock_hash=_hash("8"),
            config_hash=_hash("9"),
            boot_identity_hash=_hash("a"),
            input_root=input_root,
            output_root=output_root,
            transport_root_hash=merkle_root((), domain="leadpoet-transport-v2"),
            host_operation_root_hash=merkle_root(
                (), domain="leadpoet-host-operation-v2"
            ),
            artifact_root=merkle_root((), domain="leadpoet-artifact-v2"),
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-08-19T12:00:00Z",
        ),
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )


def _routing_protected_receipt():
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose="research_lab.routing_provider_evidence.v2",
            job_id="routing-job",
            epoch_id=24_300,
            sequence=10,
            commit_sha="1" * 40,
            pcr0="8" * 96,
            build_manifest_hash=_hash("7"),
            dependency_lock_hash=_hash("8"),
            config_hash=_hash("4"),
            boot_identity_hash=_hash("a"),
            input_root=_hash("1"),
            output_root=_hash("2"),
            transport_root_hash=_hash("3"),
            host_operation_root_hash=_hash("5"),
            artifact_root=_hash("6"),
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-08-19T12:00:00Z",
        ),
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )
    return receipt, key, pubkey


def authority_fixture():
    base, adapters, labels, _tool, _source = _spec("intent_evidence")
    pointer_uri = "s3://private-model/branches/leadpoet-lab/current.json"
    route_hashes = {
        "routing_contract_hash": _hash("c"),
        "routing_catalog_hash": _hash("d"),
        "routing_policy_hash": _hash("e"),
        "feature_schema_hash": _hash("f"),
        "verifier_contract_hash": _hash("0"),
    }
    authority_manifest = _authority(
        model_hash=_hash("a"),
        commit="1" * 40,
        route_hashes=route_hashes,
    )
    artifact = SourcingModelArtifactIdentity(
        repository="leadpoet/Sourcing_model",
        branch="leadpoet-lab",
        commit_sha="1" * 40,
        artifact_uri=pointer_uri,
        model_artifact_hash=_hash("a"),
        manifest_hash=authority_manifest["manifest_hash"],
        **route_hashes,
    )
    spec = replace(
        base,
        variants=tuple(
            replace(
                item,
                artifact=replace(
                    artifact,
                    branch=(
                        "main"
                        if item.variant_id == base.baseline_variant_id
                        else "leadpoet-lab"
                    ),
                ),
                artifact_authority_manifest=authority_manifest,
            )
            for item in base.variants
        ),
    )
    receipt_store = ProviderReceiptStore()
    decision_store = RoutingDecisionReceiptStore()
    evaluation = evaluate_routing_experiment_v2(
        spec,
        gold_labels=labels,
        runner=_runner,
        adapters=adapters,
        receipt_store=receipt_store,
        decision_store=decision_store,
        require_isolation=False,
    )
    decisions = [
        {
            "receipt_id": item.receipt_id,
            "experiment_hash": spec.experiment_hash(),
            "decision_doc": item.to_dict(),
        }
        for item in decision_store.values()
    ]
    decision_by_provider = {}
    for item in decision_store.values():
        for provider_ref in item.provider_receipt_refs:
            decision_by_provider.setdefault(provider_ref, item)
    attempts = []
    budgets = []
    receipts = tuple(
        receipt_store.repository.get(key)
        for key in sorted(receipt_store.repository.keys())
    )
    for index, receipt in enumerate(receipts):
        assert receipt is not None
        decision = decision_by_provider[receipt.receipt_ref]
        attempt_key = provider_receipt_key(
            tool_id=receipt.tool_id,
            binding_version=receipt.binding_version,
            request_fingerprint=receipt.request_fingerprint,
        )
        reservation_id = f"reservation-{index + 1}"
        action_id = "bloomberry_search_job_postings"
        attempts.append(
            {
                "attempt_key": attempt_key,
                "experiment_hash": spec.experiment_hash(),
                "provider_receipt_ref": receipt.receipt_ref,
                "binding_id": receipt.binding_id,
                "tool_id": receipt.tool_id,
                "variant_id": decision.variant_id,
                "unit_ref": receipt.unit_ref,
                "reservation_id": reservation_id,
                "action_id": action_id,
                "request_fingerprint": receipt.request_fingerprint,
                "outcome": receipt.outcome,
                "credit_microunits": receipt.credit_microunits,
                "latency_ms": receipt.latency_ms,
                "execution_mode": receipt.execution_mode,
                "billing_state": "known",
                "authoritative_billed_credit_microunits": receipt.credit_microunits,
                "attempt_doc": receipt.to_dict(),
            }
        )
        common = {
            "reservation_id": reservation_id,
            "binding_id": receipt.binding_id,
            "unit_ref": receipt.unit_ref,
            "variant_id": decision.variant_id,
            "request_fingerprint": receipt.request_fingerprint,
            "action_id": action_id,
        }
        budgets.extend(
            (
                {
                    "event_key": sha256_json({"reserve": reservation_id}),
                    "experiment_hash": spec.experiment_hash(),
                    "reservation_id": reservation_id,
                    "binding_id": receipt.binding_id,
                    "attempt_key": None,
                    "event_type": "reserve",
                    "credit_microunits": receipt.credit_microunits,
                    "event_doc": dict(common),
                },
                {
                    "event_key": sha256_json({"settle": reservation_id}),
                    "experiment_hash": spec.experiment_hash(),
                    "reservation_id": reservation_id,
                    "binding_id": receipt.binding_id,
                    "attempt_key": attempt_key,
                    "event_type": "settle",
                    "credit_microunits": receipt.credit_microunits,
                    "event_doc": {**common, "attempt_key": attempt_key},
                },
            )
        )
    lineage = {
        "repository": artifact.repository,
        "branch": artifact.branch,
        "commit_sha": artifact.commit_sha,
        "pointer_uri": pointer_uri,
        "pointer_document_hash": _hash("1"),
        "immutable_manifest_uri": "s3://private-model/releases/manifest-1.json",
        "routing_lineage_manifest_uri": "s3://private-model/releases/routing-lineage-1.json",
        "routing_lineage_manifest_hash": _hash("2"),
        "manifest_hash": artifact.manifest_hash,
        "signature_ref": "s3://private-model/signatures/manifest-1.sig",
        "signature_key_id": "kms-model-key",
        "signature_algorithm": "ECDSA_SHA_256",
        "model_artifact_hash": artifact.model_artifact_hash,
        "image_digest": "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@" + _hash("3"),
        "config_hash": _hash("4"),
        "build_id": "build-1",
        "component_registry_version": "components-v2",
        "scoring_adapter_version": "adapter-v2",
        "routing_contract_hash": artifact.routing_contract_hash,
        "routing_catalog_hash": artifact.routing_catalog_hash,
        "routing_policy_hash": artifact.routing_policy_hash,
        "feature_schema_hash": artifact.feature_schema_hash,
        "verifier_contract_hash": artifact.verifier_contract_hash,
    }
    label_authority = {
        "manifest_uri": "s3://routing-labels/releases/labels-1.json",
        "manifest_hash": _hash("5"),
        "signature_ref": "s3://routing-labels/signatures/labels-1.sig",
        "signing_key_id": "kms-label-key",
        "label_set_hash": spec.input.gold_label_set_hash,
        "labels": dict(sorted(labels.items())),
        "provenance_hash": _hash("6"),
    }
    model_binding = spec.provider_bindings[0]
    catalog_manifest_hash = _hash("b")
    unit_dataset_manifest_hash = _hash("c")
    runtime_binding = RoutingExecutionBindingV2(
        binding_id=model_binding.binding_id,
        provider_id=model_binding.provider_id,
        tool_id=model_binding.tool_id,
        binding_manifest_hash=model_binding.manifest_hash,
        action_id="bloomberry_search_job_postings",
        compiler_family="deepline_reviewed_action_v1",
        transport_id="deepline",
        model_binding_requirements_hash=_hash("d"),
        output_contract_hash=_hash("e"),
        evidence_contract_hash=_hash("f"),
        retry_policy_hash=_hash("0"),
        credit_ceiling_microunits=(
            spec.credit_budget.provider_credit_ceilings[model_binding.binding_id]
        ),
        timeout_ms=1_000,
    )
    artifact_lineage_hash = sha256_json(
        {"schema_version": "leadpoet.routing_artifact_lineage.v2", **lineage}
    )
    model_observation_result = build_routing_model_binding_observation_result_v2(
        artifact_lineage_hash=artifact_lineage_hash,
        requirement_hash_by_binding_identity={
            routing_model_binding_identity_hash(model_binding): _hash("d")
        },
    )
    model_observation_receipt = _signed_receipt(
        purpose="research_lab.routing_model_binding_observation.v2",
        input_root=model_observation_result["request_root"],
        output_root=sha256_json(model_observation_result),
        index=91,
    )
    model_observation = VerifiedRoutingModelBindingRequirements.from_attested(
        model_observation_result,
        model_observation_receipt,
    )
    envelope = RoutingExperimentExecutionEnvelopeV2(
        experiment_hash=spec.experiment_hash(),
        artifact_lineage_hash=artifact_lineage_hash,
        pointer_document_hash=lineage["pointer_document_hash"],
        binding_catalog_manifest_hash=catalog_manifest_hash,
        binding_catalog_version="catalog-1",
        unit_dataset_manifest_hash=unit_dataset_manifest_hash,
        unit_set_hash=spec.input.unit_input_set_hash,
        gold_label_manifest_hash=label_authority["manifest_hash"],
        model_binding_observation_receipt_hash=(
            model_observation.observation_receipt_hash
        ),
        model_binding_observation=model_observation.to_attested_dict(),
        bindings=(runtime_binding,),
    )
    lineage_authority = VerifiedRoutingArtifactLineage(**lineage)
    gold_authority = VerifiedRoutingGoldLabels(
        manifest_uri=label_authority["manifest_uri"],
        manifest_hash=label_authority["manifest_hash"],
        signature_ref=label_authority["signature_ref"],
        signing_key_id=label_authority["signing_key_id"],
        label_set_hash=label_authority["label_set_hash"],
        labels=label_authority["labels"],
        provenance_hash=label_authority["provenance_hash"],
    )
    catalog_manifest = RoutingBindingManifest(
        binding=model_binding,
        compiler_family="deepline_reviewed_action_v1",
        transport_id="deepline",
        action_id="bloomberry_search_job_postings",
        input_projection={"domain": "domain"},
        input_constants={},
        model_binding_requirements_hash=_hash("d"),
        output_contract_hash=_hash("e"),
        evidence_contract_hash=_hash("f"),
        retry_policy_hash=_hash("0"),
        max_results=1,
        timeout_ms=1_000,
        credit_ceiling_microunits=spec.credit_budget.provider_credit_ceilings[
            model_binding.binding_id
        ],
    )
    binding_catalog = VerifiedRoutingBindingCatalog(
        manifest_uri="s3://routing-catalog/releases/catalog-1.json",
        manifest_hash=catalog_manifest_hash,
        signature_ref="s3://routing-catalog/signatures/catalog-1.sig",
        signing_key_id="kms-routing-catalog",
        catalog_version="catalog-1",
        bindings={catalog_manifest.identity_key(): catalog_manifest},
    )
    unit_dataset = VerifiedRoutingUnitDataset(
        manifest_uri="s3://routing-units/releases/units-1.json",
        manifest_hash=unit_dataset_manifest_hash,
        signature_ref="s3://routing-units/signatures/units-1.sig",
        signing_key_id="kms-routing-units",
        unit_set_hash=spec.input.unit_input_set_hash,
        provenance_hash=_hash("9"),
        units={
            unit_ref: {"domain": unit_ref}
            for unit_ref in (
                *spec.input.calibration_unit_refs,
                *spec.input.holdout_unit_refs,
            )
        },
    )
    protected_receipt, protected_key, protected_pubkey = _routing_protected_receipt()
    admission = build_routing_admission_bundle_v2(
        job_id="routing-job",
        spec=spec,
        envelope=envelope,
        artifact_lineage=lineage_authority,
        gold_labels=gold_authority,
        binding_catalog=binding_catalog,
        unit_dataset=unit_dataset,
        model_binding_observation=model_observation,
        protected_release_receipt=protected_receipt,
    )
    budgets_by_reservation: dict[str, list[dict]] = {}
    receipt_ref_map: dict[str, str] = {}
    for budget in budgets:
        budgets_by_reservation.setdefault(str(budget["reservation_id"]), []).append(budget)
    for index, attempt in enumerate(attempts):
        old_receipt = receipts[index]
        credit_cap = max(1, int(old_receipt.credit_microunits))
        request_body_hash = sha256_json(
            {"action_id": attempt["action_id"], "unit_ref": attempt["unit_ref"]}
        )
        grant = RoutingProviderCallAuthorizationV2(
            admission_job_id=admission.job_id,
            experiment_hash=spec.experiment_hash(),
            experiment_id=spec.experiment_id,
            purpose="research_lab.routing_provider_evidence.v2",
            envelope_hash=envelope.envelope_hash(),
            admission_bundle_hash=admission.identity_hash(),
            protected_release_hash=admission.protected_release_hash,
            protected_boot_identity_hash=admission.protected_boot_identity_hash,
            variant_id=str(attempt["variant_id"]),
            stage="intent_evidence",
            artifact_lineage_hash=envelope.artifact_lineage_hash,
            pointer_document_hash=lineage["pointer_document_hash"],
            model_artifact_hash=lineage["model_artifact_hash"],
            manifest_hash=lineage["manifest_hash"],
            image_digest=lineage["image_digest"],
            commit_sha=lineage["commit_sha"],
            build_id=lineage["build_id"],
            routing_contract_hash=lineage["routing_contract_hash"],
            routing_catalog_hash=lineage["routing_catalog_hash"],
            routing_policy_hash=lineage["routing_policy_hash"],
            feature_schema_hash=lineage["feature_schema_hash"],
            verifier_contract_hash=lineage["verifier_contract_hash"],
            binding=model_binding,
            transport_id="deepline",
            binding_catalog_manifest_hash=catalog_manifest_hash,
            binding_catalog_version=envelope.binding_catalog_version,
            action_id=str(attempt["action_id"]),
            unit_ref=str(attempt["unit_ref"]),
            unit_input_hash=sha256_json({"unit_ref": attempt["unit_ref"]}),
            unit_dataset_manifest_hash=unit_dataset_manifest_hash,
            unit_set_hash=envelope.unit_set_hash,
            model_binding_observation_receipt_hash=(
                model_observation.observation_receipt_hash
            ),
            attempt=index,
            core_request_fingerprint=str(attempt["request_fingerprint"]),
            request_body_hash=request_body_hash,
            retry_policy_hash=_hash("0"),
            credit_cap_microunits=credit_cap,
            timeout_ms=max(1, int(attempt["latency_ms"])),
            claim_key=_hash("2"),
            claim_generation=1,
            claim_fence_hash=_hash("3"),
        )
        grant_result = execute_routing_provider_call_authorization_v2(
            grant.to_dict(), authorization_job_id="routing-authorization-job"
        )
        grant_receipt = _signed_receipt(
            purpose="research_lab.routing_provider_evidence.v2",
            input_root=grant.authorization_hash(),
            output_root=grant_result["output_root"],
            index=index,
            job_id="routing-authorization-job",
        )
        provider_record = create_signed_provider_evidence_record(
            body={
                "coordinator_boot_identity_hash": _hash("a"),
                "request_hash": request_body_hash,
                "request_fingerprint": old_receipt.request_fingerprint.split(":", 1)[1],
                "evidence": "recorded",
                "status": 200,
                "body_hash": _hash("7"),
                "encrypted_request_artifact_id": _hash("1"),
                "encrypted_response_artifact_id": _hash("2"),
                "transport_attempt_hash": _hash("8"),
                "source_record_hash": "",
                "issued_at": "2026-08-19T12:00:00Z",
            },
            coordinator_pubkey=protected_pubkey,
            sign_digest=protected_key.sign,
        )
        projection_identity = {
            "binding_id": old_receipt.binding_id,
            "tool_id": old_receipt.tool_id,
            "binding_version": old_receipt.binding_version,
            "source_lineage_id": old_receipt.source_lineage_id,
            "unit_ref": old_receipt.unit_ref,
            "request_fingerprint": old_receipt.request_fingerprint,
            "outcome": old_receipt.outcome,
            "evidence_hash": provider_record["record_hash"],
            "credit_microunits": old_receipt.credit_microunits,
            "latency_ms": old_receipt.latency_ms,
            "execution_mode": ReceiptExecutionMode.MEASURED_LAB.value,
        }
        new_receipt_ref = "provider_receipt:" + sha256_json(
            projection_identity
        ).split(":", 1)[1][:16]
        new_receipt = replace(
            old_receipt,
            receipt_ref=new_receipt_ref,
            evidence_hash=provider_record["record_hash"],
            execution_mode=ReceiptExecutionMode.MEASURED_LAB.value,
        )
        receipt_ref_map[old_receipt.receipt_ref] = new_receipt_ref
        terminal_body = build_routing_provider_terminal_body_v2(
            job_id=admission.job_id,
            experiment_hash=spec.experiment_hash(),
            admission_bundle_hash=admission.identity_hash(),
            authorization_hash=grant.authorization_hash(),
            authorization_proof_hash=grant_receipt["receipt_hash"],
            binding=model_binding,
            variant_id=str(attempt["variant_id"]),
            unit_ref=old_receipt.unit_ref,
            request_fingerprint=old_receipt.request_fingerprint,
            terminal_status="authenticated_response",
            provider_record=provider_record,
            coordinator_boot_identity={
                "boot_identity_hash": _hash("a"),
                "signing_pubkey": protected_pubkey,
            },
            billing_projection={
                "receipt_ref": new_receipt_ref,
                "outcome": old_receipt.outcome,
                "evidence_hash": provider_record["record_hash"],
                "credit_microunits": old_receipt.credit_microunits,
                "latency_ms": old_receipt.latency_ms,
                "billing_state": "known",
            },
        )
        terminal_proof = sign_routing_provider_terminal_v2(
            body=terminal_body,
            protected_receipt=protected_receipt,
            enclave_pubkey=protected_pubkey,
            sign_digest=protected_key.sign,
        )
        attempt.update(
            {
                "provider_receipt_ref": new_receipt_ref,
                "binding_id": new_receipt.binding_id,
                "tool_id": new_receipt.tool_id,
                "unit_ref": new_receipt.unit_ref,
                "request_fingerprint": new_receipt.request_fingerprint,
                "outcome": new_receipt.outcome,
                "credit_microunits": new_receipt.credit_microunits,
                "latency_ms": new_receipt.latency_ms,
                "execution_mode": new_receipt.execution_mode,
                "attempt_doc": {
                    "schema_version": "leadpoet.research_lab.routing_provider_attempt.v2",
                    "legacy_fixture": True,
                    "binding_id": new_receipt.binding_id,
                    "tool_id": new_receipt.tool_id,
                    "action_id": attempt["action_id"],
                    "binding_catalog_manifest_hash": catalog_manifest_hash,
                    "call_grant_hash": grant.authorization_hash(),
                    "call_grant_proof_hash": grant_receipt["receipt_hash"],
                    "request_body_hash": request_body_hash,
                    "variant_id": attempt["variant_id"],
                    "unit_ref": new_receipt.unit_ref,
                    "reservation_id": attempt["reservation_id"],
                    "request_fingerprint": new_receipt.request_fingerprint,
                    "execution_mode": new_receipt.execution_mode,
                    "provider_receipt": new_receipt.to_dict(),
                    "call_grant": grant.to_dict(),
                    "call_grant_result": grant_result,
                    "call_grant_receipt": grant_receipt,
                    "terminal_proof": terminal_proof,
                    "protected_release_receipt": protected_receipt,
                    "admission_bundle": admission.to_dict(),
                },
            }
        )
        attempt["binding_catalog_manifest_hash"] = catalog_manifest_hash
        attempt["authorization_hash"] = grant.authorization_hash()
        attempt["authorization_proof_hash"] = grant_receipt["receipt_hash"]
        attempt["request_body_hash"] = request_body_hash
        for budget in budgets_by_reservation[str(attempt["reservation_id"])]:
            budget["event_doc"].update(
                {
                    "tool_id": attempt["tool_id"],
                    "binding_catalog_manifest_hash": catalog_manifest_hash,
                    "call_grant_hash": grant.authorization_hash(),
                    "request_body_hash": request_body_hash,
                }
            )
            if budget["event_type"] == "reserve":
                budget["credit_microunits"] = credit_cap
    # The evaluator initially creates fixture receipts.  The trust-chain
    # projection above replaces each with its signed terminal receipt, so all
    # durable authority documents must point at the projected identities.
    updated_decisions = []
    for row in decisions:
        decision_doc = dict(row["decision_doc"])
        decision_doc["provider_receipt_refs"] = [
            receipt_ref_map.get(str(ref), str(ref))
            for ref in decision_doc["provider_receipt_refs"]
        ]
        decision_doc["receipt_id"] = "routing_decision:pending"
        decision_id = "routing_decision:" + sha256_json(decision_doc).split(":", 1)[1][:16]
        decision_doc["receipt_id"] = decision_id
        updated_decisions.append(
            {
                "receipt_id": decision_id,
                "experiment_hash": row["experiment_hash"],
                "decision_doc": decision_doc,
            }
        )
    decisions = updated_decisions
    variant_evaluations = []
    for variant_evaluation in evaluation.variants:
        variant_rows = [
            row
            for row in decisions
            if row["decision_doc"]["variant_id"] == variant_evaluation.variant_id
        ]
        variant_evaluations.append(
            replace(
                variant_evaluation,
                decision_receipt_refs=tuple(sorted(row["receipt_id"] for row in variant_rows)),
                provider_receipt_refs=tuple(
                    sorted(
                        {
                            str(ref)
                            for row in variant_rows
                            for ref in row["decision_doc"]["provider_receipt_refs"]
                        }
                    )
                ),
            )
        )
    evaluation = replace(
        evaluation,
        receipt_id="routing_evaluation_v2:pending",
        variants=tuple(variant_evaluations),
        decision_receipt_refs=tuple(sorted(row["receipt_id"] for row in decisions)),
        provider_receipt_refs=tuple(
            sorted(
                receipt_ref_map.get(str(ref), str(ref))
                for ref in evaluation.provider_receipt_refs
            )
        ),
    )
    evaluation = replace(
        evaluation,
        receipt_id="routing_evaluation_v2:"
        + sha256_json(evaluation.to_dict()).split(":", 1)[1][:16],
    )
    return {
        "spec": spec,
        "evaluation": evaluation,
        "decisions": tuple(sorted(decisions, key=lambda item: item["receipt_id"])),
        "attempts": tuple(sorted(attempts, key=lambda item: item["attempt_key"])),
        "budgets": tuple(sorted(budgets, key=lambda item: item["event_key"])),
        "lineage": lineage,
        "labels": label_authority,
        "execution_envelope": envelope.to_dict(),
        "model_binding_observation": model_observation,
    }
