from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.attested_autoresearch_v2 import (
    AttestedAutoresearchV2Error,
    execute_autoresearch_v2,
)
from gateway.tee.autoresearch_executor_v2 import AUTORESEARCH_OPERATIONS_V2
from gateway.tee.execution_job_manager_v2 import ExecutionJobManagerV2
from gateway.tee.host_operation_channel_v2 import HostOperationChannelV2
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.attested_v2 import (
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    create_boot_identity,
    create_signed_execution_receipt,
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    sha256_json,
    validate_receipt_graph,
)


def _hash(character):
    return "sha256:" + character * 64


def _release():
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "abcdef0123456789"[index]
        values = {
            "commit_sha": "1" * 40,
            "pcr0": character * 96,
            "normalized_image_hash": _hash(character),
            "eif_hash": _hash(character),
            "source_manifest_hash": _hash("2"),
            "build_identity_hash": _hash(character),
            "execution_manifest_hash": _hash(character),
            "dependency_lock_hash": _hash("3"),
            "dockerfile_hash": _hash("4"),
            "topology_hash": topology_hash(),
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                rows.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": domain + "-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **values,
                    }
                )
    return build_release_manifest(
        rows, acceptance_signer_pubkey_hash="sha256:" + "f" * 64
    )


def _nested_scoring_graph():
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring_a",
            commit_sha="7" * 40,
            pcr0="8" * 96,
            build_manifest_hash=_hash("7"),
            dependency_lock_hash=_hash("8"),
            config_hash=_hash("9"),
            boot_nonce="a" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="b" * 64,
            transport_certificate_hash=_hash("c"),
            attestation_user_data_hash=_hash("d"),
            issued_at="2026-07-10T00:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"scoring-nitro").decode(),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose="research_lab.candidate_test.v2",
            job_id="nested-dev-score",
            epoch_id=12,
            sequence=0,
            commit_sha="7" * 40,
            pcr0="8" * 96,
            build_manifest_hash=_hash("7"),
            dependency_lock_hash=_hash("8"),
            config_hash=_hash("9"),
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=_hash("1"),
            output_root=_hash("2"),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T00:00:00Z",
        ),
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )
    return build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
        host_operations=(),
    )


class _Client:
    def __init__(self, release, *, external_graph=None):
        role = "gateway_autoresearch"
        summary = release["roles"][role]
        self.key = Ed25519PrivateKey.generate()
        pubkey = self.key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        self.boot = create_boot_identity(
            body=build_boot_identity_body(
                role="gateway_autoresearch",
                physical_role=role,
                commit_sha=summary["commit_sha"],
                pcr0=summary["pcr0"],
                build_manifest_hash=summary["execution_manifest_hash"],
                dependency_lock_hash=summary["dependency_lock_hash"],
                config_hash=_hash("9"),
                boot_nonce="a" * 32,
                signing_pubkey=pubkey,
                transport_pubkey="b" * 64,
                transport_certificate_hash=_hash("c"),
                attestation_user_data_hash=_hash("d"),
                issued_at="2026-07-10T00:00:00Z",
            ),
            attestation_document_b64=base64.b64encode(b"nitro").decode(),
        )

        def executor(_operation, payload, context):
            if external_graph is not None:
                context.record_external_receipt_graph(external_graph)
            response = context.execute_host_operation(
                operation="echo_state",
                payload={"value": payload["value"]},
                expected_state_hash=_hash("e"),
                timeout_seconds=5,
                response_validator=lambda value: dict(value),
            )
            context.record_stage(
                purpose="research_lab.source_inspection.v2",
                input_root=_hash("e"),
                output_root=sha256_json(response),
            )
            return {"echo": response["value"]}

        self.manager = ExecutionJobManagerV2(
            boot_identity_supplier=lambda: self.boot,
            sign_digest=self.key.sign,
            operations=AUTORESEARCH_OPERATIONS_V2,
            executor=executor,
            worker_count=10,
            host_operation_channel_factory=lambda job_id, purpose: HostOperationChannelV2(
                job_id=job_id,
                purpose=purpose,
                boot_identity=self.boot,
                sign_digest=self.key.sign,
                allowed_operations={"echo_state"},
            ),
        )

    async def autoresearch_v2_health(self):
        return self.manager.health()

    async def v2_get_boot_identity(self):
        return self.boot

    async def autoresearch_v2_submit_job(self, manifest):
        return self.manager.submit(manifest)

    async def autoresearch_v2_put_chunk(self, *, job_id, offset, data):
        from leadpoet_canonical.attested_v2 import sha256_bytes

        return self.manager.put_chunk(
            job_id=job_id,
            offset=offset,
            data_b64=base64.b64encode(data).decode(),
            chunk_sha256=sha256_bytes(data),
        )

    async def autoresearch_v2_seal_job(self, job_id):
        return self.manager.seal(job_id)

    async def autoresearch_v2_next_host_operation(self, job_id, *, wait_ms=0):
        return self.manager.next_host_operation(job_id=job_id, wait_ms=wait_ms)

    async def autoresearch_v2_complete_host_operation(self, **kwargs):
        return self.manager.complete_host_operation(**kwargs)

    async def autoresearch_v2_get_status(self, job_id):
        return self.manager.status(job_id)

    async def autoresearch_v2_cancel_job(self, job_id):
        return self.manager.cancel(job_id)

    async def autoresearch_v2_get_result(self, job_id, *, offset=0):
        return self.manager.result_chunk(job_id=job_id, offset=offset)

    async def autoresearch_v2_get_receipt(self, job_id):
        return self.manager.receipt(job_id)

    async def autoresearch_v2_get_receipts(self, job_id):
        return list(self.manager.receipts(job_id))

    async def autoresearch_v2_get_transport_attempts(self, job_id):
        return list(self.manager.transport_attempts(job_id))

    async def autoresearch_v2_get_artifact_hashes(self, job_id):
        return list(self.manager.artifact_hashes(job_id))

    async def autoresearch_v2_get_host_operations(self, job_id):
        return list(self.manager.host_operations(job_id))

    async def autoresearch_v2_get_external_receipt_graphs(self, job_id):
        return list(self.manager.external_receipt_graphs(job_id))

    async def autoresearch_v2_get_transitions(self, job_id):
        return list(self.manager.transitions(job_id))


@pytest.mark.asyncio
async def test_autoresearch_bridge_dispatches_signed_host_op_and_persists_full_chain():
    release = _release()
    client = _Client(release)
    persisted = []

    async def persist(graph):
        validate_receipt_graph(graph)
        persisted.append(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_autoresearch_v2(
        operation="run_code_edit_loop",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=12,
        sequence=0,
        payload={"value": 7},
        host_operation_handlers={
            "echo_state": lambda payload, _request: {"value": payload["value"]}
        },
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )

    assert result["result"] == {"echo": 7}
    assert len(result["receipt_graph"]["receipts"]) == 2
    assert len(result["receipt_graph"]["host_operations"]) == 1
    assert persisted[0]["root_receipt_hash"] == result["receipt"]["receipt_hash"]


@pytest.mark.asyncio
async def test_autoresearch_bridge_merges_nested_scoring_graph_into_root_ancestry():
    release = _release()
    nested = _nested_scoring_graph()
    client = _Client(release, external_graph=nested)

    async def persist(graph):
        return graph

    result = await execute_autoresearch_v2(
        operation="run_code_edit_loop",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=12,
        sequence=0,
        payload={"value": 7},
        host_operation_handlers={
            "echo_state": lambda payload, _request: {"value": payload["value"]}
        },
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )

    graph = result["receipt_graph"]
    validate_receipt_graph(
        graph,
        required_purposes=(
            "research_lab.candidate_decision.v2",
            "research_lab.candidate_test.v2",
        ),
    )
    nested_root = nested["root_receipt_hash"]
    root = next(
        item
        for item in graph["receipts"]
        if item["receipt_hash"] == graph["root_receipt_hash"]
    )
    assert nested_root in root["parent_receipt_hashes"]


@pytest.mark.asyncio
async def test_autoresearch_bridge_fails_closed_when_host_handler_is_missing():
    release = _release()
    client = _Client(release)
    persisted = []

    async def persist(graph, **kwargs):
        failed_hashes = tuple(kwargs.get("allowed_failed_receipt_hashes") or ())
        validate_receipt_graph(
            graph,
            allowed_failed_receipt_hashes=failed_hashes,
        )
        persisted.append((graph, failed_hashes))
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    with pytest.raises(AttestedAutoresearchV2Error, match="failed closed") as caught:
        await execute_autoresearch_v2(
            operation="run_code_edit_loop",
            purpose="research_lab.candidate_decision.v2",
            epoch_id=12,
            sequence=0,
            payload={"value": 7},
            host_operation_handlers={},
            release_manifest=release,
            client=client,
            persist_graph=persist,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )
    authority = caught.value.authority
    assert authority["status"] == "failed"
    assert authority["result"]["status"] == "failed"
    assert authority["receipt"]["status"] == "failed"
    assert persisted[0][1] == (authority["receipt"]["receipt_hash"],)
