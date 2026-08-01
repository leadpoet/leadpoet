from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.attested_autoresearch_v2 import (
    AttestedAutoresearchV2Error,
    derive_autoresearch_job_id_v2,
    execute_autoresearch_v2,
)
from gateway.tee.autoresearch_executor_v2 import AUTORESEARCH_OPERATIONS_V2
from gateway.tee.execution_job_manager_v2 import (
    ExecutionJobManagerV2,
    PARENT_ANCESTRY_PROOFS_FIELD,
    PARENT_RECEIPT_GRAPHS_FIELD,
)
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
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    build_compact_ancestry_proof_from_delta_v2,
    issue_ancestry_certificate_v2,
    validate_compact_ancestry_proof_v2,
)


_ANCESTRY_LINEAGE_ID = "sha256:" + "6" * 64
_ANCESTRY_ISSUER_ROLES = (
    "gateway_autoresearch",
    "gateway_coordinator",
    "gateway_scoring",
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


def _nested_scoring_authority():
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
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
    graph = build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
        host_operations=(),
    )
    local_delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": receipt["receipt_hash"],
        "boot_identities": [boot],
        "receipts": [receipt],
        "transport_attempts": [],
        "host_operations": [],
    }
    certificate = issue_ancestry_certificate_v2(
        local_delta=local_delta,
        lineage_id=_ANCESTRY_LINEAGE_ID,
        certificate_sequence=0,
        issuer_boot_identity=boot,
        issued_at="2026-07-10T00:00:00Z",
        sign_digest=key.sign,
        boot_attestation_verifier=lambda identity: identity,
        allowed_issuer_roles=_ANCESTRY_ISSUER_ROLES,
        required_purposes=("research_lab.candidate_test.v2",),
    )
    proof = build_compact_ancestry_proof_from_delta_v2(
        local_delta,
        certificate,
        expected_lineage_id=_ANCESTRY_LINEAGE_ID,
        boot_attestation_verifier=lambda identity: identity,
        allowed_issuer_roles=_ANCESTRY_ISSUER_ROLES,
    )
    return graph, proof


def _nested_scoring_graph():
    return _nested_scoring_authority()[0]


class _Client:
    def __init__(self, release, *, external_graph=None, worker_count=10):
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
            worker_count=worker_count,
            host_operation_channel_factory=lambda job_id, purpose: HostOperationChannelV2(
                job_id=job_id,
                purpose=purpose,
                boot_identity=self.boot,
                sign_digest=self.key.sign,
                allowed_operations={"echo_state"},
            ),
            ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
            ancestry_boot_attestation_verifier=lambda identity: identity,
            ancestry_allowed_issuer_roles=_ANCESTRY_ISSUER_ROLES,
        )
        self.uploaded_payloads = {}

    async def autoresearch_v2_health(self):
        return self.manager.health()

    async def v2_get_boot_identity(self):
        return self.boot

    async def autoresearch_v2_submit_job(self, manifest):
        return self.manager.submit(manifest)

    async def autoresearch_v2_put_chunk(self, *, job_id, offset, data):
        from leadpoet_canonical.attested_v2 import sha256_bytes

        payload = self.uploaded_payloads.setdefault(job_id, bytearray())
        assert len(payload) == offset
        payload.extend(data)
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

    async def autoresearch_v2_get_ancestry_compact_proof(self, job_id):
        return dict(self.manager.ancestry_compact_proof(job_id))

    async def autoresearch_v2_get_transitions(self, job_id):
        return list(self.manager.transitions(job_id))


class _TamperedHostEvidenceClient(_Client):
    async def autoresearch_v2_get_host_operations(self, job_id):
        rows = await super().autoresearch_v2_get_host_operations(job_id)
        rows[0]["terminal"]["response_hash"] = _hash("0")
        return rows


class _SubmitProbeClient(_Client):
    async def autoresearch_v2_submit_job(self, manifest):
        self.submitted_manifest = dict(manifest)
        raise RuntimeError("submitted")


async def _persist_checkpoint(proof, **kwargs):
    checkpointed_graph = kwargs.pop("checkpointed_graph")
    normalized = validate_compact_ancestry_proof_v2(proof, **kwargs)
    certificate = normalized["certificate"]
    claim = certificate["claim"]
    assert checkpointed_graph["root_receipt_hash"] == claim["output_root_receipt_hash"]
    assert any(
        item["receipt_hash"] == claim["output_root_receipt_hash"]
        for item in checkpointed_graph["receipts"]
    )
    return {
        "root_receipt_hash": claim["output_root_receipt_hash"],
        "certificate_hash": certificate["certificate_hash"],
        "proof_hash": normalized["proof_hash"],
    }


@pytest.mark.asyncio
async def test_autoresearch_bridge_dispatches_signed_host_op_and_persists_full_chain():
    release = _release()
    client = _Client(release, worker_count=13)
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
        persist_ancestry_checkpoint=_persist_checkpoint,
        ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
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
        persist_ancestry_checkpoint=_persist_checkpoint,
        ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
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
async def test_autoresearch_bridge_transports_and_persists_bounded_checkpoint_graph():
    release = _release()
    nested, nested_proof = _nested_scoring_authority()
    client = _Client(release)
    persisted = []

    async def persist(graph):
        persisted.append(dict(graph))
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def no_loader(*_args, **_kwargs):
        raise AssertionError("explicit compact proof must not reload storage")

    result = await execute_autoresearch_v2(
        operation="run_code_edit_loop",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=12,
        sequence=0,
        payload={"value": 7},
        host_operation_handlers={
            "echo_state": lambda payload, _request: {"value": payload["value"]}
        },
        parent_graphs=(nested,),
        parent_ancestry_proofs=(nested_proof,),
        release_manifest=release,
        client=client,
        persist_graph=persist,
        load_ancestry_proofs=no_loader,
        persist_ancestry_checkpoint=_persist_checkpoint,
        ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )

    uploaded = json.loads(bytes(next(iter(client.uploaded_payloads.values()))))
    assert PARENT_ANCESTRY_PROOFS_FIELD in uploaded
    assert PARENT_RECEIPT_GRAPHS_FIELD not in uploaded
    assert all(
        item["receipt_hash"] != nested["root_receipt_hash"]
        for item in persisted[0]["receipts"]
    )
    authorities = result["ancestry_compact_proof"]["certificate"]["claim"][
        "parent_authorities"
    ]
    assert len(authorities) == 1
    assert authorities[0]["authority_kind"] == "certificate"
    assert authorities[0]["parent_receipt_hash"] == nested["root_receipt_hash"]


@pytest.mark.asyncio
async def test_autoresearch_job_identity_is_stable_across_parent_transport_upgrade():
    release = _release()
    nested, nested_proof = _nested_scoring_authority()

    async def persist(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def no_persisted_proofs(*_args, **_kwargs):
        return {}

    common = {
        "operation": "run_code_edit_loop",
        "purpose": "research_lab.candidate_decision.v2",
        "epoch_id": 12,
        "sequence": 0,
        "payload": {"value": 7},
        "host_operation_handlers": {
            "echo_state": lambda payload, _request: {"value": payload["value"]}
        },
        "parent_graphs": (nested,),
        "release_manifest": release,
        "persist_graph": persist,
        "persist_ancestry_checkpoint": _persist_checkpoint,
        "ancestry_lineage_id": _ANCESTRY_LINEAGE_ID,
        "boot_verifier": lambda identity: identity,
        "poll_seconds": 0.001,
    }
    bootstrap_client = _Client(release)
    bootstrap = await execute_autoresearch_v2(
        **common,
        client=bootstrap_client,
        load_ancestry_proofs=no_persisted_proofs,
    )
    compact_client = _Client(release)
    compact = await execute_autoresearch_v2(
        **common,
        client=compact_client,
        parent_ancestry_proofs=(nested_proof,),
    )

    bootstrap_payload = json.loads(
        bytes(next(iter(bootstrap_client.uploaded_payloads.values())))
    )
    compact_payload = json.loads(
        bytes(next(iter(compact_client.uploaded_payloads.values())))
    )
    assert PARENT_RECEIPT_GRAPHS_FIELD in bootstrap_payload
    assert PARENT_ANCESTRY_PROOFS_FIELD not in bootstrap_payload
    assert PARENT_ANCESTRY_PROOFS_FIELD in compact_payload
    assert PARENT_RECEIPT_GRAPHS_FIELD not in compact_payload
    assert bootstrap["receipt"]["job_id"] == compact["receipt"]["job_id"]
    assert bootstrap["receipt"]["input_root"] != compact["receipt"]["input_root"]
    assert bootstrap["receipt"]["job_id"] == derive_autoresearch_job_id_v2(
        operation="run_code_edit_loop",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=12,
        sequence=0,
        payload_sha256=sha256_json({"value": 7}),
        parent_receipt_hashes=(nested["root_receipt_hash"],),
        input_artifact_hashes=(),
        release_hash=release["release_hash"],
    )


@pytest.mark.asyncio
async def test_autoresearch_bridge_prefers_exact_persisted_parent_checkpoint():
    release = _release()
    nested, nested_proof = _nested_scoring_authority()
    client = _Client(release)
    loaded_roots = []

    async def load(roots, **_kwargs):
        loaded_roots.extend(roots)
        return {nested["root_receipt_hash"]: nested_proof}

    async def persist(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    await execute_autoresearch_v2(
        operation="run_code_edit_loop",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=12,
        sequence=0,
        payload={"value": 7},
        host_operation_handlers={
            "echo_state": lambda payload, _request: {"value": payload["value"]}
        },
        parent_graphs=(nested,),
        release_manifest=release,
        client=client,
        persist_graph=persist,
        load_ancestry_proofs=load,
        persist_ancestry_checkpoint=_persist_checkpoint,
        ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    uploaded = json.loads(bytes(next(iter(client.uploaded_payloads.values()))))
    assert loaded_roots == [nested["root_receipt_hash"]]
    assert PARENT_ANCESTRY_PROOFS_FIELD in uploaded
    assert PARENT_RECEIPT_GRAPHS_FIELD not in uploaded


@pytest.mark.asyncio
async def test_autoresearch_bridge_persists_graph_before_checkpoint_and_fails_closed():
    release = _release()
    client = _Client(release)
    order = []

    async def persist(graph):
        order.append(("graph", graph["root_receipt_hash"]))
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def fail_checkpoint(proof, **_kwargs):
        assert _kwargs["checkpointed_graph"]["root_receipt_hash"] == (
            _proof_root_for_test(proof)
        )
        order.append(("checkpoint", _proof_root_for_test(proof)))
        raise RuntimeError("sidecar unavailable")

    with pytest.raises(RuntimeError, match="sidecar unavailable"):
        await execute_autoresearch_v2(
            operation="run_code_edit_loop",
            purpose="research_lab.candidate_decision.v2",
            epoch_id=12,
            sequence=0,
            payload={"value": 7},
            host_operation_handlers={
                "echo_state": lambda payload, _request: {
                    "value": payload["value"]
                }
            },
            release_manifest=release,
            client=client,
            persist_graph=persist,
            persist_ancestry_checkpoint=fail_checkpoint,
            ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )
    assert [stage for stage, _root in order] == ["graph", "checkpoint"]


def _proof_root_for_test(proof):
    return proof["certificate"]["claim"]["output_root_receipt_hash"]


@pytest.mark.asyncio
async def test_autoresearch_bridge_rejects_reserved_compact_authority_payload():
    release = _release()
    client = _Client(release)
    with pytest.raises(AttestedAutoresearchV2Error, match="reserved"):
        await execute_autoresearch_v2(
            operation="run_code_edit_loop",
            purpose="research_lab.candidate_decision.v2",
            epoch_id=12,
            sequence=0,
            payload={
                "value": 7,
                PARENT_ANCESTRY_PROOFS_FIELD: [],
            },
            host_operation_handlers={},
            release_manifest=release,
            client=client,
            persist_graph=lambda graph: graph,
            persist_ancestry_checkpoint=_persist_checkpoint,
            ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )


@pytest.mark.asyncio
async def test_autoresearch_bridge_rejects_host_evidence_omitted_from_transport_proof():
    release = _release()
    client = _TamperedHostEvidenceClient(release)
    persisted = []

    async def persist(graph):
        persisted.append(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    with pytest.raises(Exception, match="host|delta|commitment"):
        await execute_autoresearch_v2(
            operation="run_code_edit_loop",
            purpose="research_lab.candidate_decision.v2",
            epoch_id=12,
            sequence=0,
            payload={"value": 7},
            host_operation_handlers={
                "echo_state": lambda payload, _request: {
                    "value": payload["value"]
                }
            },
            release_manifest=release,
            client=client,
            persist_graph=persist,
            persist_ancestry_checkpoint=_persist_checkpoint,
            ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )
    assert persisted == []


@pytest.mark.asyncio
async def test_autoresearch_bridge_binds_stale_parent_profile_to_repair_operation():
    release = _release()
    accepted = _SubmitProbeClient(release)
    with pytest.raises(RuntimeError, match="submitted"):
        await execute_autoresearch_v2(
            operation="repair_stale_parent",
            purpose="research_lab.stale_parent_repair.v2",
            epoch_id=12,
            sequence=0,
            payload={"value": 7},
            host_operation_handlers={},
            provider_credential_profile="stale_parent_repair",
            release_manifest=release,
            client=accepted,
            ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
            boot_verifier=lambda identity: identity,
        )
    assert accepted.submitted_manifest["provider_credential_profile"] == (
        "stale_parent_repair"
    )

    rejected = _SubmitProbeClient(release)
    with pytest.raises(AttestedAutoresearchV2Error, match="differs from operation"):
        await execute_autoresearch_v2(
            operation="repair_stale_parent",
            purpose="research_lab.stale_parent_repair.v2",
            epoch_id=12,
            sequence=0,
            payload={"value": 7},
            host_operation_handlers={},
            provider_credential_profile="default",
            release_manifest=release,
            client=rejected,
            ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
            boot_verifier=lambda identity: identity,
        )
    assert not hasattr(rejected, "submitted_manifest")


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
            persist_ancestry_checkpoint=_persist_checkpoint,
            ancestry_lineage_id=_ANCESTRY_LINEAGE_ID,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )
    authority = caught.value.authority
    assert authority["status"] == "failed"
    assert authority["result"]["status"] == "failed"
    assert authority["receipt"]["status"] == "failed"
    assert persisted[0][1] == (authority["receipt"]["receipt_hash"],)
