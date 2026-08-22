"""Focused tests for the protected scorer terminal bridge."""

from __future__ import annotations

import base64
from copy import deepcopy

import pytest

from gateway.research_lab.routing_experiment_runtime import (
    AttestedScoringV2RoutingProviderTerminalAuthority,
    RoutingExperimentRuntimeError,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    execute_protected_routing_provider_terminal_v2,
)
from gateway.tee.execution_job_manager_v2 import PARENT_RECEIPT_GRAPHS_FIELD
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_execution_receipt_body,
    create_signed_execution_receipt,
)
from research_lab.canonical import sha256_json
from tests.test_routing_provider_terminal_protected import _call_fixture


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _authority_fixture():
    compiler, prepared, request, proof, broker_result, record, boot, body, key, _ = (
        _call_fixture({"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}})
    )
    broker_result = dict(broker_result)
    broker_result["routing_provider_record"] = dict(record)
    terminal_result = execute_protected_routing_provider_terminal_v2(
        authorization_proof=proof,
        prepared_call=prepared,
        broker_request=request,
        broker_result=broker_result,
        provider_record=record,
        trusted_coordinator_boot_identity=boot,
        raw_response_body=body,
        binding_catalog=compiler.binding_catalog,
        unit_dataset=compiler.unit_dataset,
    )
    parent_graphs = [
        {
            "root_receipt_hash": proof["authorization_receipt"]["receipt_hash"],
            "receipts": [dict(proof["authorization_receipt"])],
        }
    ]
    return {
        "compiler": compiler,
        "prepared": prepared,
        "request": request,
        "proof": proof,
        "broker_result": broker_result,
        "record": record,
        "body": body,
        "key": key,
        "terminal_result": terminal_result,
        "parent_graphs": parent_graphs,
        "pubkey": boot["signing_pubkey"],
    }


def _release_and_response(fixture, request_payload, result):
    key = fixture["key"]
    pubkey = fixture["pubkey"]
    proof = fixture["proof"]
    auth_receipt_hash = proof["authorization_receipt"]["receipt_hash"]
    proof_hash = proof["authorization_proof_hash"]
    terminal_job_id = (
        "routing-terminal:"
        + sha256_json(
            {
                "schema_version": "leadpoet.routing_provider_terminal_job.v2",
                "authorization_proof_hash": proof_hash,
                "authorization_receipt_hash": auth_receipt_hash,
            }
        ).split(":", 1)[1][:32]
    )
    release = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose="research_lab.routing_provider_evidence.v2",
            job_id="release-job",
            epoch_id=1,
            sequence=1,
            commit_sha="7" * 40,
            pcr0="8" * 96,
            build_manifest_hash=_hash("9"),
            dependency_lock_hash=_hash("a"),
            config_hash=_hash("b"),
            boot_identity_hash=_hash("c"),
            input_root=_hash("d"),
            output_root=_hash("e"),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-08-19T12:00:00Z",
        ),
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )
    terminal_receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose="research_lab.routing_provider_evidence.v2",
            job_id=terminal_job_id,
            epoch_id=1,
            sequence=2,
            commit_sha=release["commit_sha"],
            pcr0=release["pcr0"],
            build_manifest_hash=release["build_manifest_hash"],
            dependency_lock_hash=release["dependency_lock_hash"],
            config_hash=release["config_hash"],
            boot_identity_hash=release["boot_identity_hash"],
            input_root=sha256_json(request_payload),
            output_root=sha256_json(result),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=[auth_receipt_hash],
            status="succeeded",
            failure_code=None,
            issued_at="2026-08-19T12:00:01Z",
        ),
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )
    return release, terminal_receipt


def _authority(fixture):
    calls = []
    release_holder = {}

    def executor(request):
        calls.append(request)
        release, receipt = _release_and_response(
            fixture, request["payload"], fixture["terminal_result"]
        )
        release_holder["release"] = release
        return {
            "status": "succeeded",
            "operation": request["operation"],
            "purpose": request["purpose"],
            "result": fixture["terminal_result"],
            "execution_receipt": receipt,
        }

    # The release is created lazily by the executor, so use a first call's
    # release only in a second authority instance. The bridge itself still
    # validates release before invoking the executor; this helper therefore
    # creates a deterministic release with the same signing identity first.
    payload_probe = {
        "schema_version": "probe",
        PARENT_RECEIPT_GRAPHS_FIELD: fixture["parent_graphs"],
    }
    release, _ = _release_and_response(
        fixture, payload_probe, fixture["terminal_result"]
    )
    return (
        AttestedScoringV2RoutingProviderTerminalAuthority(
            executor=executor,
            protected_release_receipt=release,
        ),
        calls,
    )


def test_bridge_submits_protected_operation_and_validates_standard_receipt():
    fixture = _authority_fixture()
    authority, calls = _authority(fixture)
    result = authority.execute(
        authorization_proof=fixture["proof"],
        prepared_call=fixture["prepared"],
        broker_request=fixture["request"],
        broker_result=fixture["broker_result"],
        parent_receipt_graphs=fixture["parent_graphs"],
    )
    assert result["result"] == fixture["terminal_result"]
    assert result["execution_receipt"]["job_id"].startswith("routing-terminal:")
    assert calls[0]["operation"] == "routing_provider_terminal_v2"
    assert "routing_terminal" not in calls[0]["payload"]
    assert calls[0]["job_id"] != fixture["request"]["job_id"]


def test_bridge_rejects_missing_coordinator_record_before_executor_call():
    fixture = _authority_fixture()
    authority, calls = _authority(fixture)
    broker_result = dict(fixture["broker_result"])
    broker_result.pop("routing_provider_record")
    with pytest.raises(RoutingExperimentRuntimeError, match="signed record"):
        authority.execute(
            authorization_proof=fixture["proof"],
            prepared_call=fixture["prepared"],
            broker_request=fixture["request"],
            broker_result=broker_result,
            parent_receipt_graphs=fixture["parent_graphs"],
        )
    assert calls == []


def test_bridge_rejects_provider_record_substitution():
    fixture = _authority_fixture()
    authority, calls = _authority(fixture)
    broker_result = dict(fixture["broker_result"])
    substituted = deepcopy(fixture["record"])
    substituted["record_hash"] = _hash("f")
    broker_result["provider_record"] = substituted
    with pytest.raises(RoutingExperimentRuntimeError, match="substitution"):
        authority.execute(
            authorization_proof=fixture["proof"],
            prepared_call=fixture["prepared"],
            broker_request=fixture["request"],
            broker_result=broker_result,
            parent_receipt_graphs=fixture["parent_graphs"],
        )
    assert calls == []


@pytest.mark.parametrize("field", ["input_root", "output_root", "parent_receipt_hashes"])
def test_bridge_rejects_standard_receipt_substitution(field):
    fixture = _authority_fixture()
    calls = []
    release, _ = _release_and_response(
        fixture,
        {"probe": True},
        fixture["terminal_result"],
    )
    authority = AttestedScoringV2RoutingProviderTerminalAuthority(
        executor=lambda request: calls.append(request) or {},
        protected_release_receipt=release,
    )
    # The fake executor uses a validly signed receipt with one substituted
    # commitment, so the bridge must reject the receipt rather than trusting
    # its signature alone.
    def executor(request):
        _release, receipt = _release_and_response(
            fixture, request["payload"], fixture["terminal_result"]
        )
        forged = dict(receipt)
        forged[field] = (
            _hash("f") if field != "parent_receipt_hashes" else [_hash("f")]
        )
        return {
            "status": "succeeded",
            "operation": request["operation"],
            "purpose": request["purpose"],
            "result": fixture["terminal_result"],
            "execution_receipt": forged,
        }

    authority._executor = executor
    with pytest.raises(RoutingExperimentRuntimeError, match="standard receipt"):
        authority.execute(
            authorization_proof=fixture["proof"],
            prepared_call=fixture["prepared"],
            broker_request=fixture["request"],
            broker_result=fixture["broker_result"],
            parent_receipt_graphs=fixture["parent_graphs"],
        )
    assert calls == []
