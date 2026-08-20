"""Adversarial tests for the signed routing admission and terminal chain."""

from __future__ import annotations

from copy import deepcopy
import base64
import json
from pathlib import Path

import pytest

from gateway.research_lab.routing_admission import (
    ROUTING_ADMISSION_PURPOSE_V2,
    RoutingAdmissionBundleV2,
    RoutingAdmissionError,
)
from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
    execute_routing_provider_call_authorization_v2,
    routing_provider_dispatch_job_id_v2,
    routing_provider_logical_operation_id_v2,
)
from gateway.research_lab.routing_experiment_runtime import (
    AttestedScoringV2RoutingProviderCallAuthority,
    RoutingExperimentRuntimeError,
)
from gateway.research_lab.routing_provider_terminal import (
    RoutingProviderTerminalError,
    build_routing_provider_terminal_body_v2,
    sign_routing_provider_terminal_v2,
    validate_routing_provider_terminal_v2,
)
from gateway.tee.provider_evidence_v2 import create_signed_provider_evidence_record
from gateway.tee.provider_broker_v2 import (
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2Error,
    trusted_routing_peer_boot_identity,
    validate_routing_authorization_proof_v2,
)
from leadpoet_canonical.attested_v2 import (
    build_boot_identity_body,
    build_execution_receipt_body,
    create_boot_identity,
    create_signed_execution_receipt,
)
from research_lab.routing_experiments import ProviderBindingIdentity
from research_lab.canonical import sha256_json


def _h(letter: str) -> str:
    return "sha256:" + letter * 64


def _key():
    cryptography = pytest.importorskip("cryptography")
    return cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey.generate()


def _binding() -> ProviderBindingIdentity:
    return ProviderBindingIdentity(
        binding_id="binding-deepline-bloomberry-jobs",
        provider_id="deepline",
        tool_id="intent.source_add.bloomberry_jobs",
        source_lineage_id="source-lineage-bloomberry",
        adapter_version="routing-adapter-v2",
        manifest_hash=_h("1"),
        capability_hash=_h("2"),
        execution_contract_hash=_h("3"),
        cost_model_hash=_h("4"),
    )


def _boot(pubkey: str) -> dict[str, str]:
    return {"boot_identity_hash": _h("5"), "signing_pubkey": pubkey}


def _signed_provider_record(key, boot):
    body = {
        "coordinator_boot_identity_hash": boot["boot_identity_hash"],
        "request_hash": _h("6"),
        "request_fingerprint": "6" * 64,
        "evidence": "recorded",
        "status": 200,
        "body_hash": _h("7"),
        "encrypted_request_artifact_id": _h("8"),
        "encrypted_response_artifact_id": _h("9"),
        "transport_attempt_hash": _h("a"),
        "source_record_hash": "",
        "issued_at": "2026-08-19T12:00:00Z",
    }
    return create_signed_provider_evidence_record(
        body=body,
        coordinator_pubkey=boot["signing_pubkey"],
        sign_digest=lambda digest: key.sign(digest),
    )


def _protected_receipt(key, pubkey: str):
    body = build_execution_receipt_body(
        role="gateway_scoring",
        purpose=ROUTING_ADMISSION_PURPOSE_V2,
        job_id="routing-job-1",
        epoch_id=4,
        sequence=10,
        commit_sha="a" * 40,
        pcr0="b" * 96,
        build_manifest_hash=_h("c"),
        dependency_lock_hash=_h("d"),
        config_hash=_h("e"),
        boot_identity_hash=_h("f"),
        input_root=_h("1"),
        output_root=_h("2"),
        transport_root_hash=_h("3"),
        host_operation_root_hash=_h("4"),
        artifact_root=_h("5"),
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at="2026-08-19T12:00:00Z",
    )
    return create_signed_execution_receipt(
        body=body, enclave_pubkey=pubkey, sign_digest=lambda digest: key.sign(digest)
    )


def _terminal_fixture():
    key = _key()
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    boot = _boot(pubkey)
    provider_record = _signed_provider_record(key, boot)
    protected = _protected_receipt(key, pubkey)
    binding = _binding()
    request_fingerprint = _h("6")
    projection_identity = {
        "binding_id": binding.binding_id,
        "tool_id": binding.tool_id,
        "binding_version": binding.adapter_version,
        "source_lineage_id": binding.source_lineage_id,
        "unit_ref": "unit-1",
        "request_fingerprint": request_fingerprint,
        "outcome": "verified",
        "evidence_hash": provider_record["record_hash"],
        "credit_microunits": 90_000,
        "latency_ms": 240,
        "execution_mode": "measured_lab",
    }
    receipt_ref = "provider_receipt:" + sha256_json(projection_identity).split(":", 1)[1][:16]
    body = build_routing_provider_terminal_body_v2(
        job_id="routing-job-1",
        experiment_hash=_h("0"),
        admission_bundle_hash=_h("1"),
        authorization_hash=_h("2"),
        authorization_proof_hash=_h("3"),
        binding=binding,
        variant_id="baseline",
        unit_ref="unit-1",
        request_fingerprint=request_fingerprint,
        terminal_status="authenticated_response",
        provider_record=provider_record,
        coordinator_boot_identity=boot,
        billing_projection={
            "receipt_ref": receipt_ref,
            "outcome": "verified",
            "evidence_hash": provider_record["record_hash"],
            "credit_microunits": 90_000,
            "latency_ms": 240,
            "billing_state": "known",
        },
    )
    terminal = sign_routing_provider_terminal_v2(
        body=body,
        protected_receipt=protected,
        enclave_pubkey=pubkey,
        sign_digest=lambda digest: key.sign(digest),
    )
    return terminal, protected, binding


def test_admission_bundle_rejects_release_and_job_substitution():
    fields = {
        "schema_version": "leadpoet.research_lab.routing_admission.v2",
        "job_id": "routing-job-1",
        "experiment_id": "experiment-1",
        "experiment_hash": _h("0"),
        "role": "gateway_scoring",
        "purpose": ROUTING_ADMISSION_PURPOSE_V2,
        "envelope_hash": _h("1"),
        "artifact_lineage_hash": _h("2"),
        "pointer_document_hash": _h("3"),
        "immutable_manifest_hash": _h("4"),
        "model_artifact_hash": _h("5"),
        "gold_label_manifest_hash": _h("6"),
        "gold_label_set_hash": _h("7"),
        "unit_dataset_manifest_hash": _h("8"),
        "unit_set_hash": _h("9"),
        "binding_catalog_manifest_hash": _h("a"),
        "binding_catalog_version": "catalog-v1",
        "model_binding_observation_hash": _h("b"),
        "model_binding_observation_receipt_hash": _h("c"),
        "binding_ids": ["binding-1"],
        "protected_release_hash": _h("d"),
        "protected_commit_sha": "a" * 40,
        "protected_pcr0": "f" * 96,
        "protected_build_manifest_hash": _h("1"),
        "protected_dependency_lock_hash": _h("2"),
        "protected_config_hash": _h("3"),
        "protected_boot_identity_hash": _h("4"),
        "protected_enclave_pubkey": "5" * 64,
        "protected_receipt_hash": _h("6"),
    }
    bundle = RoutingAdmissionBundleV2.from_mapping(fields)
    assert bundle.identity_hash().startswith("sha256:")
    forged = bundle.to_dict()
    forged["job_id"] = "routing-job-2"
    altered = RoutingAdmissionBundleV2.from_mapping(forged)
    assert altered.identity_hash() != bundle.identity_hash()
    forged["job_id"] = ""
    with pytest.raises(RoutingAdmissionError):
        RoutingAdmissionBundleV2.from_mapping(forged)


def test_terminal_derives_only_from_signed_provider_and_billing_proof():
    terminal, protected, binding = _terminal_fixture()
    projected = validate_routing_provider_terminal_v2(
        terminal=terminal,
        binding=binding,
        protected_receipt=protected,
        expected_job_id="routing-job-1",
        expected_experiment_hash=_h("0"),
        expected_admission_bundle_hash=_h("1"),
        expected_authorization_hash=_h("2"),
        expected_authorization_proof_hash=_h("3"),
    )
    assert projected["outcome"] == "verified"
    assert projected["credit_microunits"] == 90_000

    forged = deepcopy(terminal)
    forged["body"]["billing_projection"]["credit_microunits"] = 1
    with pytest.raises(RoutingProviderTerminalError):
        validate_routing_provider_terminal_v2(
            terminal=forged,
            binding=binding,
            protected_receipt=protected,
            expected_job_id="routing-job-1",
            expected_experiment_hash=_h("0"),
            expected_admission_bundle_hash=_h("1"),
            expected_authorization_hash=_h("2"),
            expected_authorization_proof_hash=_h("3"),
        )


def test_terminal_rejects_replayed_job_and_release_receipt():
    terminal, protected, binding = _terminal_fixture()
    with pytest.raises(RoutingProviderTerminalError):
        validate_routing_provider_terminal_v2(
            terminal=terminal,
            binding=binding,
            protected_receipt=protected,
            expected_job_id="routing-job-other",
            expected_experiment_hash=_h("0"),
            expected_admission_bundle_hash=_h("1"),
            expected_authorization_hash=_h("2"),
            expected_authorization_proof_hash=_h("3"),
        )
    forged = deepcopy(terminal)
    forged["receipt"]["parent_receipt_hashes"] = [_h("9")]
    with pytest.raises(RoutingProviderTerminalError):
        validate_routing_provider_terminal_v2(
            terminal=forged,
            binding=binding,
            protected_receipt=protected,
            expected_job_id="routing-job-1",
            expected_experiment_hash=_h("0"),
            expected_admission_bundle_hash=_h("1"),
            expected_authorization_hash=_h("2"),
            expected_authorization_proof_hash=_h("3"),
        )


def test_provider_grant_result_commits_admission_and_catalog_identity():
    binding = _binding()
    grant = RoutingProviderCallAuthorizationV2(
        admission_job_id="routing-job-1",
        experiment_hash=_h("0"),
        experiment_id="experiment-1",
        purpose="research_lab.routing_provider_evidence.v2",
        envelope_hash=_h("1"),
        admission_bundle_hash=_h("2"),
        protected_release_hash=_h("3"),
        protected_boot_identity_hash=_h("4"),
        variant_id="baseline",
        stage="intent_evidence",
        artifact_lineage_hash=_h("5"),
        pointer_document_hash=_h("6"),
        model_artifact_hash=_h("7"),
        manifest_hash=_h("8"),
        image_digest="registry.example/router@sha256:" + "9" * 64,
        commit_sha="a" * 40,
        build_id="build-1",
        routing_contract_hash=_h("b"),
        routing_catalog_hash=_h("c"),
        routing_policy_hash=_h("d"),
        feature_schema_hash=_h("e"),
        verifier_contract_hash=_h("f"),
        binding=binding,
        transport_id="deepline",
        binding_catalog_manifest_hash=_h("0"),
        binding_catalog_version="catalog-v1",
        action_id="bloomberry_search_job_postings",
        unit_ref="unit-1",
        unit_input_hash=_h("1"),
        unit_dataset_manifest_hash=_h("2"),
        unit_set_hash=_h("3"),
        model_binding_observation_receipt_hash=_h("4"),
        attempt=0,
        core_request_fingerprint=_h("5"),
        request_body_hash=_h("6"),
        retry_policy_hash=_h("0"),
        credit_cap_microunits=90_000,
        timeout_ms=5_000,
        claim_key=_h("7"),
        claim_generation=1,
        claim_fence_hash=_h("8"),
    )
    result = execute_routing_provider_call_authorization_v2(
        grant.to_dict(), authorization_job_id="routing-authorization-job"
    )
    assert result["admission_bundle_hash"] == grant.admission_bundle_hash
    assert result["envelope_hash"] == grant.envelope_hash
    assert result["binding_catalog_manifest_hash"] == grant.binding_catalog_manifest_hash
    assert result["model_binding_observation_receipt_hash"] == (
        grant.model_binding_observation_receipt_hash
    )


def test_provider_grant_broker_proof_carries_exact_authorization_document_and_result():
    from gateway.research_lab.routing_provider_bindings import (
        ReviewedDeeplineActionCompiler,
    )
    from tests.test_routing_provider_authorization_context import _context

    context = _context()
    grant = context["grant"]
    key = _key()
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()

    def executor(request):
        assert request["parent_receipt_hashes"] == [
            context["protected_receipt"]["receipt_hash"],
            context["observation"].observation_receipt_hash,
        ]
        expected = execute_routing_provider_call_authorization_v2(
            request["payload"]["authorization"],
            authorization_job_id="routing-authorization-job",
        )
        body = build_execution_receipt_body(
            role="gateway_scoring",
            purpose="research_lab.routing_provider_evidence.v2",
            job_id="routing-authorization-job",
            epoch_id=4,
            sequence=11,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash=_h("c"),
            dependency_lock_hash=_h("d"),
            config_hash=_h("e"),
            boot_identity_hash=_h("f"),
            input_root=sha256_json(request["payload"]),
            output_root=expected["output_root"],
            transport_root_hash=_h("1"),
            host_operation_root_hash=_h("2"),
            artifact_root=_h("3"),
            parent_receipt_hashes=request["payload"]["parent_receipt_hashes"],
            status="succeeded",
            failure_code=None,
            issued_at="2026-08-19T12:00:00Z",
        )
        receipt = create_signed_execution_receipt(
            body=body,
            enclave_pubkey=pubkey,
            sign_digest=lambda digest: key.sign(digest),
        )
        return {
            "status": "succeeded",
            "operation": request["operation"],
            "purpose": request["purpose"],
            "result": expected,
            "execution_receipt": receipt,
        }

    authority = AttestedScoringV2RoutingProviderCallAuthority(executor=executor)
    proof = authority.authorize(
        grant,
        artifact_lineage=context["lineage"],
        model_binding_observation=context["observation"],
        execution_envelope=context["envelope"],
        admission_bundle=context["admission"],
        prepared_call=context["prepared"],
        protected_release_receipt=context["protected_receipt"],
        parent_receipt_graphs=(
            {"receipts": [dict(context["protected_receipt"])]},
            {"receipts": [dict(context["observation"].signed_receipt)]},
        ),
    )
    assert proof["authorization_receipt"]["parent_receipt_hashes"] == [
        context["protected_receipt"]["receipt_hash"],
        context["observation"].observation_receipt_hash,
    ]
    assert set(proof) == {
        "authorization_hash",
        "authorization_request_hash",
        "authorization_proof_hash",
        "request_body_hash",
        "action_id",
        "credit_cap_microunits",
        "timeout_ms",
        "authorization",
        "authorization_result",
        "authorization_receipt",
    }
    import base64
    import json

    compiler = ReviewedDeeplineActionCompiler(
        binding_catalog=context["catalog"],
        unit_dataset=context["unit_dataset"],
    )
    broker_request = dict(
        compiler.broker_request(
            prepared=context["prepared"],
            experiment_hash=grant.experiment_hash,
            dispatch_job_id=routing_provider_dispatch_job_id_v2(proof),
            variant_id=grant.variant_id,
            attempt_number=grant.attempt,
            core_request_fingerprint=grant.core_request_fingerprint,
            authorization_hash=grant.authorization_hash(),
            authorization_proof_hash=proof["authorization_proof_hash"],
        )
    )
    broker_request["routing_authorization"] = proof
    authority.validate_broker_request(
        proof,
        broker_request,
    )
    forged = deepcopy(proof)
    forged["authorization_result"] = dict(forged["authorization_result"])
    forged["authorization_result"]["output_root"] = _h("9")
    with pytest.raises(Exception, match="proof"):
        authority.validate_broker_request(
            forged,
            {**broker_request, "routing_authorization": forged},
        )


@pytest.mark.parametrize("case", ["missing", "substituted", "reversed"])
def test_provider_authorization_rejects_wrong_parent_ancestry_before_tee_call(case):
    from tests.test_routing_provider_authorization_context import _context

    context = _context()
    calls = []
    authority = AttestedScoringV2RoutingProviderCallAuthority(
        executor=lambda request: calls.append(request) or {}
    )
    expected_graphs = (
        {"receipts": [dict(context["protected_receipt"])]},
        {"receipts": [dict(context["observation"].signed_receipt)]},
    )
    if case == "missing":
        parent_graphs = ()
    elif case == "substituted":
        substituted = dict(context["protected_receipt"])
        substituted["receipt_hash"] = _h("9")
        parent_graphs = (
            {"receipts": [substituted]},
            expected_graphs[1],
        )
    else:
        parent_graphs = tuple(reversed(expected_graphs))

    with pytest.raises(RoutingExperimentRuntimeError, match="ancestry"):
        authority.authorize(
            context["grant"],
            artifact_lineage=context["lineage"],
            model_binding_observation=context["observation"],
            execution_envelope=context["envelope"],
            admission_bundle=context["admission"],
            prepared_call=context["prepared"],
            protected_release_receipt=context["protected_receipt"],
            parent_receipt_graphs=parent_graphs,
        )
    assert calls == []


def test_standard_broker_rejects_missing_authority_before_provider_transport():
    from tests.test_provider_broker_v2 import FakeTransport, _broker
    import base64
    import json

    transport = FakeTransport()
    broker = _broker(transport)
    body = {"provider": "deepline", "operation": "execute", "payload": {}}
    proof = {
        "authorization_hash": _h("1"),
        "authorization_proof_hash": _h("2"),
        "request_body_hash": sha256_json(body),
        "action_id": "bloomberry_search_job_postings",
        "credit_cap_microunits": 90_000,
        "timeout_ms": 5_000,
    }
    request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": _h("0"),
        "job_id": "routing-job-1",
        "purpose": "research_lab.routing_provider_evidence.v2",
        "provider_id": "deepline",
        "attempt_number": 0,
        "method": "POST",
        "url": "https://code.deepline.com/api/v2/integrations/bloomberry_search_job_postings/execute",
        "headers": {"Content-Type": "application/json"},
        "body_b64": base64.b64encode(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).decode(),
        "timeout_ms": 5_000,
        "retry_policy_hash": _h("0"),
        "routing_authorization": proof,
    }
    with pytest.raises(ProviderBrokerV2Error):
        broker.execute(request)
    assert transport.calls == []


def test_standard_broker_accepts_only_complete_explicit_routing_proof():
    from tests.test_provider_broker_v2 import FakeTransport, _broker
    import base64
    import json

    transport = FakeTransport()
    body = {"provider": "deepline", "operation": "execute", "payload": {}}
    grant = RoutingProviderCallAuthorizationV2(
        admission_job_id="routing-job-1",
        experiment_hash=_h("0"),
        experiment_id="experiment-1",
        purpose="research_lab.routing_provider_evidence.v2",
        envelope_hash=_h("1"),
        admission_bundle_hash=_h("2"),
        protected_release_hash=_h("3"),
        protected_boot_identity_hash=_h("4"),
        variant_id="baseline",
        stage="intent_evidence",
        artifact_lineage_hash=_h("5"),
        pointer_document_hash=_h("6"),
        model_artifact_hash=_h("7"),
        manifest_hash=_h("8"),
        image_digest="registry.example/router@sha256:" + "9" * 64,
        commit_sha="a" * 40,
        build_id="build-1",
        routing_contract_hash=_h("b"),
        routing_catalog_hash=_h("c"),
        routing_policy_hash=_h("d"),
        feature_schema_hash=_h("e"),
        verifier_contract_hash=_h("f"),
        binding=_binding(),
        transport_id="deepline",
        binding_catalog_manifest_hash=_h("0"),
        binding_catalog_version="catalog-v1",
        action_id="bloomberry_search_job_postings",
        unit_ref="unit-1",
        unit_input_hash=_h("1"),
        unit_dataset_manifest_hash=_h("2"),
        unit_set_hash=_h("3"),
        model_binding_observation_receipt_hash=_h("4"),
        attempt=0,
        core_request_fingerprint=_h("5"),
        request_body_hash=sha256_json(body),
        retry_policy_hash=_h("a"),
        credit_cap_microunits=90_000,
        timeout_ms=5_000,
        claim_key=_h("7"),
        claim_generation=1,
        claim_fence_hash=_h("8"),
    )
    expected = execute_routing_provider_call_authorization_v2(
        grant.to_dict(), authorization_job_id="routing-authorization-job"
    )
    key = _key()
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose=grant.purpose,
            job_id="routing-authorization-job",
            epoch_id=4,
            sequence=11,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash=_h("c"),
            dependency_lock_hash=_h("d"),
            config_hash=_h("e"),
            boot_identity_hash=_h("f"),
            input_root=grant.authorization_hash(),
            output_root=expected["output_root"],
            transport_root_hash=_h("1"),
            host_operation_root_hash=_h("2"),
            artifact_root=_h("3"),
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-08-19T12:00:00Z",
        ),
        enclave_pubkey=pubkey,
        sign_digest=lambda digest: key.sign(digest),
    )
    proof = {
        "authorization_hash": grant.authorization_hash(),
        "authorization_request_hash": grant.authorization_hash(),
        "authorization_proof_hash": receipt["receipt_hash"],
        "request_body_hash": grant.request_body_hash,
        "action_id": grant.action_id,
        "credit_cap_microunits": grant.credit_cap_microunits,
        "timeout_ms": grant.timeout_ms,
        "authorization": grant.to_dict(),
        "authorization_result": expected,
        "authorization_receipt": receipt,
    }
    request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": routing_provider_logical_operation_id_v2(
            experiment_hash=grant.experiment_hash,
            variant_id=grant.variant_id,
            unit_ref=grant.unit_ref,
            tool_id=grant.binding.tool_id,
            attempt=grant.attempt,
            core_request_fingerprint=grant.core_request_fingerprint,
            request_body_hash=grant.request_body_hash,
        ),
        "job_id": routing_provider_dispatch_job_id_v2(proof),
        "purpose": "research_lab.routing_provider_evidence.v2",
        "provider_id": "deepline",
        "attempt_number": 0,
        "method": "POST",
        "url": "https://code.deepline.com/api/v2/integrations/bloomberry_search_job_postings/execute",
        "headers": {
            "Content-Type": "application/json",
            "x-deepline-execute-response-intent": "raw",
        },
        "body_b64": base64.b64encode(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).decode(),
        "timeout_ms": 5_000,
        "retry_policy_hash": grant.retry_policy_hash,
        "routing_authorization": proof,
    }
    broker = _broker(transport)
    result = broker.execute(request)
    assert result["terminal_status"] == "authenticated_response"
    assert len(transport.calls) == 1
    assert request["job_id"] == routing_provider_dispatch_job_id_v2(proof)
    assert request["purpose"] == grant.purpose
    assert request["provider_id"] == grant.transport_id
    assert request["retry_policy_hash"] == grant.retry_policy_hash
    assert request["attempt_number"] == grant.attempt

    # The verifier must remain usable after the issuing authority is gone;
    # it relies only on the complete signed proof carried by the request.
    validate_routing_authorization_proof_v2(proof, request)

    bad_receipt = deepcopy(receipt)
    bad_receipt["output_root"] = _h("e")
    adversarial_cases = (
        (lambda candidate: candidate.pop("authorization_receipt"), proof, request),
        (
            lambda candidate: candidate.update(
                {"authorization_proof_hash": _h("f")}
            ),
            proof,
            request,
        ),
        (
            lambda candidate: candidate.update(
                {"authorization_receipt": bad_receipt}
            ),
            proof,
            request,
        ),
        (
            lambda candidate: candidate.update(
                {"credit_cap_microunits": grant.credit_cap_microunits + 1}
            ),
            proof,
            request,
        ),
        (
            lambda candidate: candidate["authorization"].update(
                {"admission_job_id": "substituted-admission-job"}
            ),
            proof,
            request,
        ),
        (
            lambda candidate: candidate["authorization_result"].update(
                {"authorization_job_id": "substituted-authorization-job"}
            ),
            proof,
            request,
        ),
        (
            lambda candidate: candidate["authorization_receipt"].update(
                {"job_id": "substituted-authorization-job"}
            ),
            proof,
            request,
        ),
    )
    for mutate_proof, source_proof, source_request in adversarial_cases:
        forged_proof = deepcopy(source_proof)
        mutate_proof(forged_proof)
        forged_request = dict(source_request)
        forged_request["routing_authorization"] = forged_proof
        rejected_transport = FakeTransport()
        with pytest.raises(ProviderBrokerV2Error):
            _broker(rejected_transport).execute(forged_request)
        assert rejected_transport.calls == []

    substituted_request = dict(request)
    substituted_request["job_id"] = "substituted-authorization-job"
    substituted_request["routing_authorization"] = proof
    rejected_transport = FakeTransport()
    with pytest.raises(ProviderBrokerV2Error):
        _broker(rejected_transport).execute(substituted_request)
    assert rejected_transport.calls == []


def _trusted_routing_proof_fixture():
    key = _key()
    serialization = pytest.importorskip(
        "cryptography.hazmat.primitives.serialization"
    )
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha="f" * 40,
            pcr0="1" * 96,
            build_manifest_hash=_h("2"),
            dependency_lock_hash=_h("3"),
            config_hash=_h("4"),
            boot_nonce="5" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="6" * 64,
            transport_certificate_hash=_h("7"),
            attestation_user_data_hash=_h("8"),
            issued_at="2026-08-19T12:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"attested").decode(),
    )
    binding = _binding()
    body = {"provider": "deepline", "operation": "execute", "payload": {}}
    grant = RoutingProviderCallAuthorizationV2(
        admission_job_id="routing-job-trusted",
        experiment_hash=_h("0"),
        experiment_id="experiment-trusted",
        purpose=ROUTING_ADMISSION_PURPOSE_V2,
        envelope_hash=_h("1"),
        admission_bundle_hash=_h("2"),
        protected_release_hash=_h("3"),
        protected_boot_identity_hash=boot["boot_identity_hash"],
        variant_id="baseline",
        stage="intent_evidence",
        artifact_lineage_hash=_h("4"),
        pointer_document_hash=_h("5"),
        model_artifact_hash=_h("6"),
        manifest_hash=_h("7"),
        image_digest="registry.example/router@sha256:" + "9" * 64,
        commit_sha="a" * 40,
        build_id="build-trusted",
        routing_contract_hash=_h("a"),
        routing_catalog_hash=_h("b"),
        routing_policy_hash=_h("c"),
        feature_schema_hash=_h("d"),
        verifier_contract_hash=_h("e"),
        binding=binding,
        transport_id="deepline",
        binding_catalog_manifest_hash=_h("f"),
        binding_catalog_version="catalog-trusted",
        action_id="bloomberry_search_job_postings",
        unit_ref="unit-trusted",
        unit_input_hash=_h("1"),
        unit_dataset_manifest_hash=_h("2"),
        unit_set_hash=_h("3"),
        model_binding_observation_receipt_hash=_h("4"),
        attempt=0,
        core_request_fingerprint=_h("5"),
        request_body_hash=sha256_json(body),
        retry_policy_hash=_h("6"),
        credit_cap_microunits=90_000,
        timeout_ms=5_000,
        claim_key=_h("7"),
        claim_generation=1,
        claim_fence_hash=_h("8"),
    )
    expected = execute_routing_provider_call_authorization_v2(
        grant.to_dict(), authorization_job_id="routing-authorization-job"
    )
    receipt_body = build_execution_receipt_body(
        role="gateway_scoring",
        purpose=grant.purpose,
        job_id="routing-authorization-job",
        epoch_id=1,
        sequence=1,
        commit_sha=boot["commit_sha"],
        pcr0=boot["pcr0"],
        build_manifest_hash=boot["build_manifest_hash"],
        dependency_lock_hash=boot["dependency_lock_hash"],
        config_hash=boot["config_hash"],
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=grant.authorization_hash(),
        output_root=expected["output_root"],
        transport_root_hash=_h("9"),
        host_operation_root_hash=_h("a"),
        artifact_root=_h("b"),
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at="2026-08-19T12:00:00Z",
    )
    receipt = create_signed_execution_receipt(
        body=receipt_body,
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )
    proof = {
        "authorization_hash": grant.authorization_hash(),
        "authorization_request_hash": grant.authorization_hash(),
        "authorization_proof_hash": receipt["receipt_hash"],
        "request_body_hash": grant.request_body_hash,
        "action_id": grant.action_id,
        "credit_cap_microunits": grant.credit_cap_microunits,
        "timeout_ms": grant.timeout_ms,
        "authorization": grant.to_dict(),
        "authorization_result": expected,
        "authorization_receipt": receipt,
    }
    request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": routing_provider_logical_operation_id_v2(
            experiment_hash=grant.experiment_hash,
            variant_id=grant.variant_id,
            unit_ref=grant.unit_ref,
            tool_id=grant.binding.tool_id,
            attempt=grant.attempt,
            core_request_fingerprint=grant.core_request_fingerprint,
            request_body_hash=grant.request_body_hash,
        ),
        "job_id": routing_provider_dispatch_job_id_v2(proof),
        "purpose": grant.purpose,
        "provider_id": grant.transport_id,
        "attempt_number": grant.attempt,
        "method": "POST",
        "url": (
            "https://code.deepline.com/api/v2/integrations/"
            "bloomberry_search_job_postings/execute"
        ),
        "headers": {
            "Content-Type": "application/json",
            "x-deepline-execute-response-intent": "raw",
        },
        "body_b64": base64.b64encode(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).decode(),
        "timeout_ms": grant.timeout_ms,
        "retry_policy_hash": grant.retry_policy_hash,
        "routing_authorization": proof,
    }
    return key, boot, grant, proof, request


def test_routing_proof_requires_attested_scoring_signer_and_release_identity():
    from tests.test_provider_broker_v2 import FakeTransport, _broker

    key, boot, grant, proof, request = _trusted_routing_proof_fixture()

    # The context path is the path used by nested ProviderBrokerV2 calls. The
    # explicit argument is used by the RPC boundary before provider semantics.
    with trusted_routing_peer_boot_identity(boot):
        validate_routing_authorization_proof_v2(proof, request)
    validate_routing_authorization_proof_v2(
        proof,
        request,
        trusted_peer_boot_identity=boot,
    )

    attacker_key = _key()
    serialization = pytest.importorskip(
        "cryptography.hazmat.primitives.serialization"
    )
    attacker_pubkey = attacker_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    forged_receipt_body = {
        key_name: proof["authorization_receipt"][key_name]
        for key_name in (
            "schema_version", "role", "purpose", "job_id", "epoch_id",
            "sequence", "commit_sha", "pcr0", "build_manifest_hash",
            "dependency_lock_hash", "config_hash", "boot_identity_hash",
            "input_root", "output_root", "transport_root", "host_operation_root",
            "artifact_root", "parent_receipt_hashes", "status", "failure_code",
            "issued_at",
        )
    }
    forged_receipt = create_signed_execution_receipt(
        body=forged_receipt_body,
        enclave_pubkey=attacker_pubkey,
        sign_digest=attacker_key.sign,
    )
    forged_proof = {
        **proof,
        "authorization_proof_hash": forged_receipt["receipt_hash"],
        "authorization_receipt": forged_receipt,
    }
    forged_request = {
        **request,
        "routing_authorization": forged_proof,
    }
    with pytest.raises(ProviderBrokerV2Error, match="signer identity"):
        validate_routing_authorization_proof_v2(
            forged_proof,
            forged_request,
            trusted_peer_boot_identity=boot,
        )

    substituted_body = dict(forged_receipt_body)
    substituted_body["commit_sha"] = "e" * 40
    substituted_receipt = create_signed_execution_receipt(
        body=substituted_body,
        enclave_pubkey=boot["signing_pubkey"],
        sign_digest=key.sign,
    )
    substituted_proof = {
        **proof,
        "authorization_proof_hash": substituted_receipt["receipt_hash"],
        "authorization_receipt": substituted_receipt,
    }
    with pytest.raises(ProviderBrokerV2Error, match="signer identity"):
        validate_routing_authorization_proof_v2(
            substituted_proof,
            {**request, "routing_authorization": substituted_proof},
            trusted_peer_boot_identity=boot,
        )

    changed_body = dict(request)
    changed_body["body_b64"] = base64.b64encode(
        json.dumps(
            {"provider": "deepline", "operation": "execute", "payload": {"x": 1}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).decode()
    changed_timeout = dict(request)
    changed_timeout["timeout_ms"] = grant.timeout_ms + 1
    changed_method = dict(request)
    changed_method["method"] = "GET"
    changed_job = dict(request)
    changed_job["job_id"] = "another-routing-job"
    changed_provider = dict(request)
    changed_provider["provider_id"] = "bloomberry"
    changed_retry = dict(request)
    changed_retry["retry_policy_hash"] = _h("f")
    changed_logical = dict(request)
    changed_logical["logical_operation_id"] = _h("e")
    changed_path = dict(request)
    changed_path["url"] = (
        "https://code.deepline.com/api/v2/integrations/"
        "another_action/execute"
    )
    changed_headers = dict(request)
    changed_headers["headers"] = {
        "Content-Type": "application/json",
        "x-deepline-execute-response-intent": "json",
    }
    changed_attempt = dict(request)
    changed_attempt["attempt_number"] = grant.attempt + 1
    changed_purpose = dict(request)
    changed_purpose["purpose"] = "research_lab.other_purpose.v2"
    for mutation_name, forged_request in (
        ("body", changed_body),
        ("timeout", changed_timeout),
        ("method", changed_method),
        ("job", changed_job),
        ("provider", changed_provider),
        ("retry", changed_retry),
        ("logical", changed_logical),
        ("path", changed_path),
        ("headers", changed_headers),
        ("attempt", changed_attempt),
        ("purpose", changed_purpose),
    ):
        rejected_transport = FakeTransport()
        with pytest.raises(ProviderBrokerV2Error):
            _broker(rejected_transport).execute(forged_request)
        assert rejected_transport.calls == [], mutation_name


def test_inter_enclave_rpc_binds_routing_proof_to_attested_scoring_peer(
    monkeypatch,
):
    # tee_service keeps its Nitro service dependency ``merkle`` as a top-level
    # import, so mirror the enclave runtime path for this RPC-boundary test.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "gateway" / "tee"))
    import gateway.tee.rpc_authority as rpc_authority
    import gateway.tee.tee_service as tee_service

    key, boot, _grant, proof, request = _trusted_routing_proof_fixture()
    calls = []

    class Authority:
        def execute(self, params):
            calls.append(params)
            return {"status": "accepted"}

    monkeypatch.setattr(
        rpc_authority,
        "active_enclave_role",
        lambda: "gateway_coordinator",
    )
    monkeypatch.setattr(
        tee_service,
        "get_v2_provider_semantics_authority",
        lambda: Authority(),
    )
    peer = {
        "physical_role": "gateway_scoring",
        "boot_identity": boot,
    }

    assert tee_service.handle_inter_enclave_rpc(
        "provider_execute", request, peer
    ) == {"status": "accepted"}
    assert len(calls) == 1

    serialization = pytest.importorskip(
        "cryptography.hazmat.primitives.serialization"
    )
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    receipt_fields = (
        "schema_version", "role", "purpose", "job_id", "epoch_id",
        "sequence", "commit_sha", "pcr0", "build_manifest_hash",
        "dependency_lock_hash", "config_hash", "boot_identity_hash",
        "input_root", "output_root", "transport_root", "host_operation_root",
        "artifact_root", "parent_receipt_hashes", "status", "failure_code",
        "issued_at",
    )
    receipt_body = {
        field: proof["authorization_receipt"][field]
        for field in receipt_fields
    }

    attacker_key = _key()
    attacker_pubkey = attacker_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    substitutions = (
        (
            "arbitrary key",
            create_signed_execution_receipt(
                body=receipt_body,
                enclave_pubkey=attacker_pubkey,
                sign_digest=attacker_key.sign,
            ),
        ),
        (
            "boot identity",
            create_signed_execution_receipt(
                body={**receipt_body, "boot_identity_hash": _h("9")},
                enclave_pubkey=pubkey,
                sign_digest=key.sign,
            ),
        ),
        (
            "release",
            create_signed_execution_receipt(
                body={**receipt_body, "commit_sha": "e" * 40},
                enclave_pubkey=pubkey,
                sign_digest=key.sign,
            ),
        ),
    )
    for label, receipt in substitutions:
        forged_proof = {
            **proof,
            "authorization_proof_hash": receipt["receipt_hash"],
            "authorization_receipt": receipt,
        }
        forged_request = {
            **request,
            "routing_authorization": forged_proof,
        }
        with pytest.raises(ProviderBrokerV2Error):
            # The nested context must reject before the fake semantics
            # authority can observe the request.
            tee_service.handle_inter_enclave_rpc(
                "provider_execute", forged_request, peer
            )
        assert len(calls) == 1, label


def test_inter_enclave_rpc_allows_only_fixed_protected_budget_reservation_sidecar(
    monkeypatch,
):
    """The scoring peer can call only the reviewed V3 reservation RPC."""

    import base64

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "gateway" / "tee")
    )
    import gateway.tee.rpc_authority as rpc_authority
    import gateway.tee.tee_service as tee_service

    _key_value, boot, _grant, _proof, _request = _trusted_routing_proof_fixture()
    calls = []

    class Authority:
        def execute(self, params):
            calls.append(dict(params))
            return {"status": "reserved"}

    monkeypatch.setattr(
        rpc_authority,
        "active_enclave_role",
        lambda: "gateway_coordinator",
    )
    monkeypatch.setattr(
        tee_service,
        "_v2_supabase_origin",
        lambda _configuration: "https://supabase.example.com",
    )
    monkeypatch.setattr(
        tee_service,
        "get_v2_runtime_identity",
        lambda: type(
            "Runtime",
            (),
            {"runtime_configuration": lambda self: {"configuration": {}}},
        )(),
    )
    monkeypatch.setattr(
        tee_service,
        "get_v2_provider_semantics_authority",
        lambda: Authority(),
    )
    job_id = "routing-dispatch:" + "a" * 32
    request = {
        "schema_version": "leadpoet.provider_broker.v2",
        "logical_operation_id": (
            f"{job_id}:routing-budget-reservation:" + "b" * 32
        ),
        "job_id": job_id,
        "purpose": "research_lab.routing_provider_evidence.v2",
        "provider_id": "supabase",
        "attempt_number": 0,
        "method": "POST",
        "url": (
            "https://supabase.example.com/rest/v1/rpc/"
            "research_lab_routing_reserve_budget_v3"
        ),
        "headers": {
            "accept": "application/json",
            "content-type": "application/json",
        },
        "body_b64": base64.b64encode(b"{}").decode("ascii"),
        "timeout_ms": 5_000,
        "retry_policy_hash": _h("a"),
    }
    peer = {"physical_role": "gateway_scoring", "boot_identity": boot}
    assert tee_service.handle_inter_enclave_rpc(
        "provider_execute", request, peer
    ) == {"status": "reserved"}
    assert calls == [request]

    for changed in (
        {**request, "url": request["url"].replace("reserve_budget_v3", "promote_v3")},
        {**request, "method": "GET"},
        {**request, "timeout_ms": 4_999},
        {**request, "logical_operation_id": "routing-budget-reservation:forged"},
        {**request, "headers": {"accept": "application/json"}},
    ):
        with pytest.raises((ProviderBrokerV2Error, ValueError)):
            tee_service.handle_inter_enclave_rpc(
                "provider_execute", changed, peer
            )
        assert calls == [request]

    with pytest.raises(ValueError, match="scoring peer"):
        tee_service.handle_inter_enclave_rpc(
            "provider_execute",
            request,
            {"physical_role": "gateway_autoresearch", "boot_identity": boot},
        )
    assert calls == [request]
