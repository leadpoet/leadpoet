from __future__ import annotations

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.execution_job_manager_v2 import (
    JOB_SCHEMA_VERSION,
    PARENT_RECEIPT_GRAPHS_FIELD,
    ExecutionJobManagerV2,
    ExecutionJobV2Error,
    ExecutionResultV2,
    TransitionSpecV2,
)
from gateway.tee.host_operation_channel_v2 import HostOperationChannelV2
from leadpoet_canonical.attested_v2 import (
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    build_transport_attempt,
    create_boot_identity,
    create_signed_execution_receipt,
    host_operation_root,
    merkle_root,
    sha256_bytes,
    validate_signed_execution_receipt,
    validate_signed_transition_command,
)


HASH = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
NOW = "2026-07-10T20:00:00Z"


def _manager(executor):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha="c" * 40,
            pcr0="d" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH,
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nitro").decode("ascii"),
    )
    manager = ExecutionJobManagerV2(
        boot_identity_supplier=lambda: boot,
        sign_digest=key.sign,
        operations={
            "score": {
                "research_lab.candidate_score.v2",
                "research_lab.baseline_score.v2",
            }
        },
        executor=executor,
        worker_count=1,
    )
    return manager, boot


def _payload():
    return json.dumps(
        {"input": 3},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _manifest(payload, **overrides):
    value = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": "score-job-1",
        "operation": "score",
        "purpose": "research_lab.candidate_score.v2",
        "epoch_id": 24000,
        "sequence": 1,
        "payload_sha256": sha256_bytes(payload),
        "payload_size_bytes": len(payload),
        "parent_receipt_hashes": [],
        "input_artifact_hashes": [HASH_B],
        "provider_credential_profile": "default",
        "provider_credential_ref_hashes": {},
    }
    value.update(overrides)
    return value


def _run(manager, payload, manifest=None):
    manifest = manifest or _manifest(payload)
    manager.submit(manifest)
    manager.put_chunk(
        job_id=manifest["job_id"],
        offset=0,
        data_b64=base64.b64encode(payload).decode("ascii"),
        chunk_sha256=sha256_bytes(payload),
    )
    manager.seal(manifest["job_id"])
    deadline = time.time() + 2
    while time.time() < deadline:
        status = manager.status(manifest["job_id"])
        if status["state"] in {"succeeded", "failed"}:
            return status
        time.sleep(0.01)
    raise AssertionError("V2 job did not terminate")


def test_success_receipt_binds_transport_artifacts_and_signed_transition():
    attempt = build_transport_attempt(
        request_id="1" * 32,
        logical_operation_id="provider-op-1",
        job_id="score-job-1",
        purpose="research_lab.candidate_score.v2",
        provider_id="openrouter",
        attempt_number=0,
        method="POST",
        destination_host="openrouter.ai",
        destination_port=443,
        path_hash=HASH,
        nonsecret_headers_hash=HASH,
        body_hash=HASH_B,
        credential_ref_hash=HASH,
        retry_policy_hash=HASH_B,
        timeout_ms=30000,
        started_at=NOW,
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=HASH,
        request_artifact_hash=HASH,
        response_artifact_hash=HASH_B,
        tls_peer_chain_hash=HASH,
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at=NOW,
    )

    def _executor(operation, payload, context):
        assert operation == "score"
        assert context.parent_receipt_hashes == ()
        context.record_transport(attempt)
        context.record_artifact(HASH)
        return ExecutionResultV2(
            output={"score": payload["input"] * 2},
            transitions=(
                TransitionSpecV2(
                    operation="insert",
                    target="research_lab_score_bundles",
                    idempotency_key="score-job-1",
                    expected_state_hash=HASH,
                    payload_hash=HASH_B,
                ),
            ),
        )

    manager, boot = _manager(_executor)
    status = _run(manager, _payload())
    assert status["state"] == "succeeded"
    receipt = manager.receipt("score-job-1")
    validate_signed_execution_receipt(receipt)
    assert receipt["boot_identity_hash"] == boot["boot_identity_hash"]
    assert receipt["transport_root"] == merkle_root(
        [attempt["attempt_hash"]], domain="leadpoet-transport-v2"
    )
    assert receipt["artifact_root"] == merkle_root(
        [HASH, HASH_B], domain="leadpoet-artifact-v2"
    )
    transition = manager.transitions("score-job-1")[0]
    validate_signed_transition_command(transition)
    assert transition["receipt_hash"] == receipt["receipt_hash"]
    result = manager.result_chunk(job_id="score-job-1")
    assert json.loads(base64.b64decode(result["data_b64"])) == {"score": 6}


def test_receipt_output_projection_binds_authoritative_result_only():
    full_output = {
        "allocation": {"allocation_hash": HASH},
        "source_state": {"epoch": 24000},
    }
    receipt_output = {"allocation": full_output["allocation"]}
    manager, _boot = _manager(
        lambda _operation, _payload, _context: ExecutionResultV2(
            output=full_output,
            receipt_output=receipt_output,
        )
    )
    payload = _payload()
    status = _run(
        manager,
        payload,
        _manifest(payload, parent_receipt_hashes=[]),
    )

    assert status["state"] == "succeeded"
    receipt = manager.receipt("score-job-1")
    assert receipt["output_root"] == sha256_bytes(
        json.dumps(
            receipt_output,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert status["result_sha256"] == sha256_bytes(
        json.dumps(
            full_output,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    result = manager.result_chunk(job_id="score-job-1")
    assert json.loads(base64.b64decode(result["data_b64"])) == full_output


def test_executor_failure_has_signed_failure_receipt_and_canonical_terminal_result():
    def _executor(_operation, _payload, _context):
        raise RuntimeError("private detail must not leave enclave")

    manager, _ = _manager(_executor)
    status = _run(manager, _payload())
    assert status["state"] == "failed"
    assert "private detail" not in str(status)
    receipt = manager.receipt("score-job-1")
    validate_signed_execution_receipt(receipt)
    assert receipt["status"] == "failed"
    result = manager.result_chunk(job_id="score-job-1")
    terminal = json.loads(base64.b64decode(result["data_b64"]))
    assert terminal == {
        "status": "failed",
        "failure_code": receipt["failure_code"],
    }
    assert receipt["output_root"] == sha256_bytes(
        json.dumps(terminal, sort_keys=True, separators=(",", ":")).encode()
    )


def test_stage_receipts_form_a_measured_chain_before_root_receipt():
    def _executor(_operation, payload, context):
        context.record_stage(
            purpose="research_lab.baseline_score.v2",
            input_root=sha256_bytes(json.dumps(payload, sort_keys=True).encode()),
            output_root=HASH_B,
            artifact_hashes=(HASH,),
        )
        return {"score": payload["input"]}

    manager, _ = _manager(_executor)
    status = _run(manager, _payload())
    assert status["state"] == "succeeded"
    receipts = manager.receipts("score-job-1")
    assert len(receipts) == 2
    stage, root = receipts
    assert stage["purpose"] == "research_lab.baseline_score.v2"
    assert root["parent_receipt_hashes"] == [stage["receipt_hash"]]
    assert stage["parent_receipt_hashes"] == []


def test_nested_receipt_graph_is_bound_to_root_and_retained_for_graph_merge():
    nested_key = Ed25519PrivateKey.generate()
    nested_pubkey = nested_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    nested_boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha="1" * 40,
            pcr0="2" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH,
            boot_nonce="3" * 32,
            signing_pubkey=nested_pubkey,
            transport_pubkey="4" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nested-nitro").decode("ascii"),
    )
    nested_receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose="research_lab.candidate_test.v2",
            job_id="nested-dev-score",
            epoch_id=24000,
            sequence=0,
            commit_sha="1" * 40,
            pcr0="2" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH,
            boot_identity_hash=nested_boot["boot_identity_hash"],
            input_root=HASH,
            output_root=HASH_B,
            transport_root_hash=merkle_root((), domain="leadpoet-transport-v2"),
            host_operation_root_hash=merkle_root(
                (), domain="leadpoet-host-operation-v2"
            ),
            artifact_root=merkle_root((), domain="leadpoet-artifact-v2"),
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at=NOW,
        ),
        enclave_pubkey=nested_pubkey,
        sign_digest=nested_key.sign,
    )
    nested_graph = build_receipt_graph(
        root_receipt_hash=nested_receipt["receipt_hash"],
        boot_identities=(nested_boot,),
        receipts=(nested_receipt,),
        transport_attempts=(),
        host_operations=(),
    )

    def _executor(_operation, payload, context):
        assert context.parent_receipt_hashes == (nested_receipt["receipt_hash"],)
        assert context.external_receipt_graphs == [nested_graph]
        context.record_stage(
            purpose="research_lab.baseline_score.v2",
            input_root=sha256_bytes(json.dumps(payload, sort_keys=True).encode()),
            output_root=HASH_B,
        )
        return {"score": payload["input"]}

    manager, _ = _manager(_executor)
    payload = json.dumps(
        {
            "input": 3,
            PARENT_RECEIPT_GRAPHS_FIELD: [nested_graph],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = _manifest(
        payload,
        parent_receipt_hashes=[nested_receipt["receipt_hash"]],
    )
    assert _run(manager, payload, manifest)["state"] == "succeeded"
    receipts = manager.receipts("score-job-1")
    assert receipts[-1]["parent_receipt_hashes"] == sorted(
        [receipts[-2]["receipt_hash"], nested_receipt["receipt_hash"]]
    )
    assert manager.external_receipt_graphs("score-job-1") == (nested_graph,)
    assert manager.status("score-job-1")["external_receipt_graph_count"] == 1


def test_job_id_and_payload_are_immutable_and_canonical():
    manager, _ = _manager(lambda _op, value, _ctx: value)
    payload = _payload()
    manifest = _manifest(payload)
    manager.submit(manifest)
    assert manager.submit(manifest)["manifest_hash"]
    with pytest.raises(ExecutionJobV2Error, match="another manifest"):
        manager.submit({**manifest, "epoch_id": 24001})
    changed_payload = b'{"input":4}'
    manager.put_chunk(
        job_id="score-job-1",
        offset=0,
        data_b64=base64.b64encode(changed_payload).decode("ascii"),
        chunk_sha256=sha256_bytes(changed_payload),
    )
    with pytest.raises(ExecutionJobV2Error, match="payload hash"):
        manager.seal("score-job-1")


def test_v1_purpose_and_unknown_operation_fail_closed():
    manager, _ = _manager(lambda _op, value, _ctx: value)
    payload = _payload()
    with pytest.raises(ExecutionJobV2Error, match="not authorized"):
        manager.submit(
            _manifest(payload, purpose="research_lab.candidate_score.v1")
        )
    with pytest.raises(ExecutionJobV2Error, match="not authorized"):
        manager.submit(_manifest(payload, operation="blind_sign"))


def test_autoresearch_job_binds_signed_host_operation_ledger():
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_autoresearch",
            physical_role="gateway_autoresearch",
            commit_sha="c" * 40,
            pcr0="d" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH,
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nitro").decode("ascii"),
    )

    def channel_factory(job_id, purpose):
        return HostOperationChannelV2(
            job_id=job_id,
            purpose=purpose,
            boot_identity=boot,
            sign_digest=key.sign,
            allowed_operations={"build_candidate_image"},
        )

    def executor(_operation, payload, context):
        response = context.execute_host_operation(
            operation="build_candidate_image",
            payload={"input": payload["input"]},
            expected_state_hash=HASH,
            timeout_seconds=5,
            response_validator=lambda value: {
                "candidate_manifest_hash": value["candidate_manifest_hash"]
            },
        )
        return {"candidate_manifest_hash": response["candidate_manifest_hash"]}

    manager = ExecutionJobManagerV2(
        boot_identity_supplier=lambda: boot,
        sign_digest=key.sign,
        operations={
            "loop": {"research_lab.candidate_build.v2"},
        },
        executor=executor,
        worker_count=1,
        host_operation_channel_factory=channel_factory,
    )
    payload = _payload()
    manifest = _manifest(
        payload,
        job_id="autoresearch-job-1",
        operation="loop",
        purpose="research_lab.candidate_build.v2",
    )
    manager.submit(manifest)
    manager.put_chunk(
        job_id=manifest["job_id"],
        offset=0,
        data_b64=base64.b64encode(payload).decode("ascii"),
        chunk_sha256=sha256_bytes(payload),
    )
    manager.seal(manifest["job_id"])
    command = None
    deadline = time.time() + 2
    while time.time() < deadline and command is None:
        command = manager.next_host_operation(
            job_id=manifest["job_id"], wait_ms=50
        )
    assert command is not None
    manager.complete_host_operation(
        job_id=manifest["job_id"],
        request_hash=command["request"]["request_hash"],
        terminal_status="succeeded",
        response={"candidate_manifest_hash": HASH_B},
    )
    deadline = time.time() + 2
    while time.time() < deadline:
        status = manager.status(manifest["job_id"])
        if status["state"] == "succeeded":
            break
        time.sleep(0.01)
    assert status["state"] == "succeeded"
    records = manager.host_operations(manifest["job_id"])
    assert len(records) == 1
    assert manager.receipt(manifest["job_id"])["host_operation_root"] == (
        host_operation_root(records)
    )
