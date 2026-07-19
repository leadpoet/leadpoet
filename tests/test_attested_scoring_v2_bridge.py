import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.attested_scoring_v2 import (
    AttestedScoringV2Error,
    execute_scoring_v2,
)
from gateway.tee.execution_job_manager_v2 import (
    ExecutionJobManagerV2,
    ExecutionResultV2,
)
from gateway.tee.coordinator_executor_v2 import (
    COORDINATOR_OPERATIONS_V2,
    CoordinatorExecutorV2,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.scoring_executor_v2 import SCORING_OPERATIONS_V2
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.attested_v2 import (
    build_boot_identity_body,
    build_transport_attempt,
    create_boot_identity,
    sha256_bytes,
    sha256_json,
    transport_root,
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


class _Client:
    def __init__(self, release, *, executor=None, configured_worker_count=1):
        role = "gateway_scoring"
        summary = release["roles"][role]
        self.key = Ed25519PrivateKey.generate()
        pubkey = self.key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        body = build_boot_identity_body(
            role="gateway_scoring",
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
        )
        self.boot = create_boot_identity(
            body=body,
            attestation_document_b64=base64.b64encode(b"nitro").decode(),
        )
        self.manager = ExecutionJobManagerV2(
            boot_identity_supplier=lambda: self.boot,
            sign_digest=self.key.sign,
            operations=SCORING_OPERATIONS_V2,
            executor=executor
            or (
                lambda operation, payload, context: {
                    "operation": operation,
                    "echo": payload,
                }
            ),
            worker_count=1,
            configured_worker_count=configured_worker_count,
        )

    async def scoring_v2_health(self):
        return self.manager.health()

    async def v2_get_boot_identity(self):
        return self.boot

    async def scoring_v2_submit_job(self, manifest):
        return self.manager.submit(manifest)

    async def scoring_v2_put_chunk(self, *, job_id, offset, data):
        import hashlib

        return self.manager.put_chunk(
            job_id=job_id,
            offset=offset,
            data_b64=base64.b64encode(data).decode(),
            chunk_sha256="sha256:" + hashlib.sha256(data).hexdigest(),
        )

    async def scoring_v2_seal_job(self, job_id):
        return self.manager.seal(job_id)

    async def scoring_v2_get_status(self, job_id):
        return self.manager.status(job_id)

    async def scoring_v2_cancel_job(self, job_id):
        return self.manager.cancel(job_id)

    async def scoring_v2_get_result(self, job_id, *, offset=0):
        return self.manager.result_chunk(job_id=job_id, offset=offset)

    async def scoring_v2_get_receipt(self, job_id):
        return self.manager.receipt(job_id)

    async def scoring_v2_get_receipts(self, job_id):
        return list(self.manager.receipts(job_id))

    async def scoring_v2_get_transport_attempts(self, job_id):
        return list(self.manager.transport_attempts(job_id))

    async def scoring_v2_get_artifact_hashes(self, job_id):
        return list(self.manager.artifact_hashes(job_id))

    async def scoring_v2_get_transitions(self, job_id):
        return list(self.manager.transitions(job_id))


class _CoordinatorClient(_Client):
    def __init__(self, release):
        super().__init__(release)
        role = "gateway_coordinator"
        summary = release["roles"][role]
        pubkey = self.key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        body = build_boot_identity_body(
            role=role,
            physical_role=role,
            commit_sha=summary["commit_sha"],
            pcr0=summary["pcr0"],
            build_manifest_hash=summary["execution_manifest_hash"],
            dependency_lock_hash=summary["dependency_lock_hash"],
            config_hash=_hash("9"),
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=_hash("a"),
            attestation_user_data_hash=_hash("b"),
            issued_at="2026-07-10T00:00:00Z",
        )
        self.boot = create_boot_identity(
            body=body,
            attestation_document_b64=base64.b64encode(b"nitro").decode(),
        )
        self.manager = ExecutionJobManagerV2(
            boot_identity_supplier=lambda: self.boot,
            sign_digest=self.key.sign,
            operations=COORDINATOR_OPERATIONS_V2,
            executor=lambda operation, payload, context: {
                "operation": operation,
                "echo": payload,
            },
            worker_count=1,
            configured_worker_count=0,
        )

    coordinator_v2_health = _Client.scoring_v2_health
    coordinator_v2_submit_job = _Client.scoring_v2_submit_job
    coordinator_v2_put_chunk = _Client.scoring_v2_put_chunk
    coordinator_v2_seal_job = _Client.scoring_v2_seal_job
    coordinator_v2_get_status = _Client.scoring_v2_get_status
    coordinator_v2_cancel_job = _Client.scoring_v2_cancel_job
    coordinator_v2_get_result = _Client.scoring_v2_get_result
    coordinator_v2_get_receipt = _Client.scoring_v2_get_receipt
    coordinator_v2_get_receipts = _Client.scoring_v2_get_receipts
    coordinator_v2_get_transport_attempts = _Client.scoring_v2_get_transport_attempts
    coordinator_v2_get_artifact_hashes = _Client.scoring_v2_get_artifact_hashes
    coordinator_v2_get_transitions = _Client.scoring_v2_get_transitions


@pytest.mark.asyncio
async def test_v2_bridge_returns_only_durable_release_verified_result():
    release = _release()
    client = _Client(release, configured_worker_count=13)
    persisted = []

    def load_profile(
        profile,
        *,
        execution_role,
        worker_index,
        require_egress_proxy,
    ):
        assert profile == "default"
        assert execution_role == "gateway_scoring"
        assert worker_index == 12
        assert require_egress_proxy is False
        return {
            "profile": profile,
            "credential_ref_hashes": {},
            "envelopes": [],
        }

    async def persist(graph):
        validate_receipt_graph(graph)
        persisted.append(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=12,
        sequence=0,
        payload={"scores": [1.0, 2.0]},
        worker_index=12,
        provider_profile_loader=load_profile,
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    assert result["result"] == {
        "operation": "benchmark_icp_score",
        "echo": {"scores": [1.0, 2.0]},
    }
    assert result["status"] == "succeeded"
    assert result["physical_role"] == "gateway_scoring"
    assert persisted[0]["root_receipt_hash"] == result["receipt"]["receipt_hash"]


@pytest.mark.asyncio
async def test_v2_bridge_verifies_projected_receipt_output():
    release = _release()
    full_output = {
        "allocation": {"allocation_hash": _hash("7")},
        "source_state": {"epoch": 12},
    }
    receipt_output = {"allocation": full_output["allocation"]}
    client = _Client(
        release,
        executor=lambda _operation, _payload, _context: ExecutionResultV2(
            output=full_output,
            receipt_output=receipt_output,
        ),
    )

    async def persist(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=12,
        sequence=0,
        payload={"scores": [1.0]},
        worker_index=0,
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
        receipt_output_projector=lambda _operation, output: {
            "allocation": output["allocation"]
        },
    )

    assert result["result"] == full_output
    assert result["receipt"]["output_root"] == sha256_json(receipt_output)


@pytest.mark.asyncio
async def test_v2_bridge_accepts_measured_coordinator_internal_worker_capacity():
    release = _release()
    client = _CoordinatorClient(release)

    async def persist(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="promotion_improvement",
        purpose="research_lab.ranking.v2",
        epoch_id=12,
        sequence=0,
        payload={"score_bundle": {}},
        worker_index=0,
        provider_profile_loader=lambda *args, **kwargs: {
            "profile": "default",
            "credential_ref_hashes": {},
            "envelopes": [],
        },
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
        operation_registry=COORDINATOR_OPERATIONS_V2,
        physical_role_override="gateway_coordinator",
        expected_service_role="gateway_coordinator",
        rpc_namespace="coordinator_v2",
    )

    assert result["status"] == "succeeded"
    assert result["physical_role"] == "gateway_coordinator"
    assert client.manager.health()["configured_worker_count"] == 0


@pytest.mark.asyncio
async def test_v2_bridge_preserves_complete_local_stage_receipt_chain():
    release = _release()

    def executor(operation, payload, context):
        context.record_stage(
            purpose="research_lab.provider_evidence_tape.v2",
            input_root=sha256_json({"trace": "baseline"}),
            output_root=sha256_json({"cache": "baseline"}),
            artifact_hashes=(_hash("a"),),
        )
        return {"operation": operation, "echo": payload}

    client = _Client(release, executor=executor)
    persisted = []

    async def persist(graph):
        validate_receipt_graph(graph)
        persisted.append(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="run_model_sandbox_v2",
        purpose="research_lab.private_model_run.v2",
        epoch_id=12,
        sequence=0,
        payload={"model_kind": "private"},
        worker_index=0,
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    receipts = {
        item["receipt_hash"]: item for item in result["receipt_graph"]["receipts"]
    }
    root = receipts[result["receipt"]["receipt_hash"]]
    assert len(receipts) == 2
    assert len(root["parent_receipt_hashes"]) == 1
    stage = receipts[root["parent_receipt_hashes"][0]]
    assert stage["purpose"] == "research_lab.provider_evidence_tape.v2"
    assert stage["parent_receipt_hashes"] == []
    assert persisted[0] == result["receipt_graph"]


@pytest.mark.asyncio
async def test_v2_bridge_leases_and_releases_attested_benchmark_profile():
    release = _release()
    credential_hash = _hash("7")
    events = []

    def executor(operation, payload, context):
        events.append("execute")
        assert context.provider_credential_profile == "benchmark_model"
        assert context.provider_credential_ref_hashes == {"exa": credential_hash}
        assert payload.pop("_v2_provider_credential_profile") == "benchmark_model"
        assert payload.pop("_v2_provider_credential_ref_hashes") == {
            "exa": credential_hash
        }
        return {"operation": operation, "echo": payload}

    client = _Client(release, executor=executor)

    class _CredentialClient:
        async def v2_release_job_credentials(self, job_id):
            events.append("release")
            return {
                "status": "released",
                "job_id": job_id,
                "released_slot_count": 1,
            }

    def load_profile(
        profile,
        *,
        execution_role,
        worker_index,
        require_egress_proxy,
    ):
        assert profile == "benchmark_model"
        assert execution_role == "gateway_scoring"
        assert worker_index == 0
        assert require_egress_proxy is False
        return {
            "profile": profile,
            "credential_ref_hashes": {"exa": credential_hash},
            "envelopes": [{"encrypted": True}],
        }

    async def provision_profile(document, *, job_id, client):
        del client
        events.append("provision")
        return {
            "profile": document["profile"],
            "job_id": job_id,
            "credential_ref_hashes": dict(document["credential_ref_hashes"]),
            "leased_credential_count": 1,
            "results": [{"status": "ready"}],
        }

    async def persist_graph(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="run_model_sandbox_v2",
        purpose="research_lab.private_model_run.v2",
        epoch_id=12,
        sequence=0,
        payload={"model_kind": "private"},
        worker_index=0,
        provider_credential_profile="benchmark_model",
        provider_profile_loader=load_profile,
        provider_profile_provisioner=provision_profile,
        credential_coordinator_client=_CredentialClient(),
        release_manifest=release,
        client=client,
        persist_graph=persist_graph,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    assert result["result"] == {
        "operation": "run_model_sandbox_v2",
        "echo": {"model_kind": "private"},
    }
    assert events == ["provision", "execute", "release"]


@pytest.mark.asyncio
async def test_v2_bridge_binds_dynamic_provider_id_to_derived_job_slot():
    release = _release()
    credential_hash = _hash("7")
    key_ref_hash = _hash("8")
    events = []

    def executor(operation, payload, context):
        events.append("execute")
        assert context.provider_credential_ref_hashes == {
            "source_one": credential_hash
        }
        assert payload.pop("_v2_provider_credential_ref_hashes") == {
            "source_one": credential_hash
        }
        return {"operation": operation, "echo": payload}

    client = _Client(release, executor=executor)

    class _CredentialClient:
        async def v2_release_job_credentials(self, job_id):
            events.append("release")
            return {
                "status": "released",
                "job_id": job_id,
                "released_slot_count": 1,
            }

    def load_profile(*_args, **_kwargs):
        return {
            "profile": "default",
            "credential_ref_hashes": {},
            "envelopes": [],
        }

    def build_envelopes(job_id):
        ciphertext = b"encrypted-source-add"
        context = {"adapter_ref": "source_add:adapter:test"}
        return [
            {
                "schema_version": "leadpoet.job_provider_credential_envelope.v2",
                "job_id": job_id,
                "credential_slot": "source_add_" + "d" * 32,
                "credential_ref_hash": credential_hash,
                "credential_value_hash": credential_hash,
                "key_ref_hash": key_ref_hash,
                "ciphertext_blob_b64": base64.b64encode(ciphertext).decode(),
                "ciphertext_blob_hash": sha256_bytes(ciphertext),
                "kms_key_id_hash": _hash("9"),
                "encryption_context": context,
                "encryption_context_hash": sha256_json(context),
            }
        ]

    async def provision_job(envelope, *, client):
        del client
        events.append("provision_dynamic")
        return {
            "status": "ready",
            "job_id": envelope["job_id"],
            "credential_slot": envelope["credential_slot"],
            "credential_ref_hash": envelope["credential_value_hash"],
        }

    async def persist_graph(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="run_model_sandbox_v2",
        purpose="research_lab.private_model_run.v2",
        epoch_id=12,
        sequence=0,
        payload={"model_kind": "private"},
        worker_index=0,
        provider_credential_ref_hashes={"source_one": credential_hash},
        provider_profile_loader=load_profile,
        additional_job_credential_envelope_builder=build_envelopes,
        job_credential_provisioner=provision_job,
        credential_coordinator_client=_CredentialClient(),
        release_manifest=release,
        client=client,
        persist_graph=persist_graph,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )

    assert result["status"] == "succeeded"
    assert events == ["provision_dynamic", "execute", "release"]


@pytest.mark.asyncio
async def test_v2_bridge_fails_when_persistence_does_not_read_back_root():
    release = _release()
    client = _Client(release)

    async def persist(_graph):
        return {"root_receipt_hash": _hash("f")}

    with pytest.raises(AttestedScoringV2Error, match="durable readback"):
        await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0]},
            worker_index=0,
            release_manifest=release,
            client=client,
            persist_graph=persist,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )


@pytest.mark.asyncio
async def test_v2_bridge_durably_persists_signed_failure_before_raising():
    release = _release()

    def fail_executor(_operation, _payload, _context):
        raise ValueError("measured scoring failure")

    client = _Client(release, executor=fail_executor)
    persisted = []

    async def persist(graph, *, allowed_failed_receipt_hashes=()):
        allowed = set(allowed_failed_receipt_hashes)
        validate_receipt_graph(
            graph,
            required_purposes={"research_lab.benchmark.v2"},
            allowed_failed_receipt_hashes=allowed,
        )
        assert allowed == {graph["root_receipt_hash"]}
        persisted.append(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    with pytest.raises(AttestedScoringV2Error, match="failed closed") as captured:
        await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0]},
            worker_index=0,
            release_manifest=release,
            client=client,
            persist_graph=persist,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )

    authority = captured.value.authority
    assert authority is not None
    assert authority["status"] == "failed"
    assert authority["execution_receipt"]["status"] == "failed"
    assert authority["result"] == {
        "status": "failed",
        "failure_code": "execution_valueerror",
    }
    assert persisted[0] == authority["receipt_graph"]


@pytest.mark.asyncio
async def test_v2_bridge_rejects_unauthorized_purpose_before_rpc():
    release = _release()
    with pytest.raises(AttestedScoringV2Error, match="purpose"):
        await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.allocation.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0]},
            worker_index=0,
            release_manifest=release,
            client=object(),
        )


def _authenticated_attempt(context):
    return build_transport_attempt(
        request_id="a" * 32,
        logical_operation_id="provider-operation",
        job_id=context.job_id,
        purpose=context.purpose,
        provider_id="openrouter",
        attempt_number=0,
        method="POST",
        destination_host="openrouter.ai",
        destination_port=443,
        path_hash=_hash("1"),
        nonsecret_headers_hash=_hash("2"),
        body_hash=_hash("3"),
        credential_ref_hash=_hash("4"),
        retry_policy_hash=_hash("5"),
        timeout_ms=30000,
        started_at="2026-07-10T00:00:00Z",
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=_hash("6"),
        request_artifact_hash=_hash("8"),
        response_artifact_hash=_hash("6"),
        tls_peer_chain_hash=_hash("7"),
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at="2026-07-10T00:00:01Z",
    )


def _storage_attempts_for_job(artifact_id, job_id):
    attempts = []
    for ordinal, method in enumerate(("GET", "HEAD")):
        attempts.append(
            build_transport_attempt(
                request_id=("c" if ordinal == 0 else "d") * 32,
                logical_operation_id="%s:%s" % (artifact_id, method.lower()),
                job_id=job_id,
                purpose="leadpoet.artifact_persistence.v2",
                provider_id="aws_s3_object_lock",
                attempt_number=ordinal,
                method=method,
                destination_host="immutable.example.s3.us-east-1.amazonaws.com",
                destination_port=443,
                path_hash=_hash("1"),
                nonsecret_headers_hash=_hash("2"),
                body_hash=_hash("3"),
                credential_ref_hash=_hash("4"),
                retry_policy_hash=_hash("5"),
                timeout_ms=30000,
                started_at="2026-07-10T00:00:00Z",
                terminal_status="authenticated_response",
                http_status=200,
                response_hash=_hash("6"),
                request_artifact_hash=_hash("8"),
                response_artifact_hash=_hash("6"),
                tls_peer_chain_hash=_hash("7"),
                tls_protocol="TLSv1.3",
                failure_code=None,
                completed_at="2026-07-10T00:00:01Z",
            )
        )
    return attempts


def _storage_attempts(artifact_id):
    return _storage_attempts_for_job(artifact_id, artifact_id)


class _ArtifactCoordinator:
    def __init__(self, release, plaintext_hashes):
        if isinstance(plaintext_hashes, str):
            plaintext_hashes = (plaintext_hashes,)
        self.plaintext_hashes = tuple(plaintext_hashes)
        role = "gateway_coordinator"
        summary = release["roles"][role]
        self.key = Ed25519PrivateKey.generate()
        pubkey = self.key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        body = build_boot_identity_body(
            role="gateway_coordinator",
            physical_role=role,
            commit_sha=summary["commit_sha"],
            pcr0=summary["pcr0"],
            build_manifest_hash=summary["execution_manifest_hash"],
            dependency_lock_hash=summary["dependency_lock_hash"],
            config_hash=_hash("9"),
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=_hash("a"),
            attestation_user_data_hash=_hash("b"),
            issued_at="2026-07-10T00:00:00Z",
        )
        self.boot = create_boot_identity(
            body=body,
            attestation_document_b64=base64.b64encode(b"nitro").decode(),
        )
        self.artifacts = [
            {
                "artifact_id": _hash(character),
                "plaintext_hash": plaintext_hash,
            }
            for character, plaintext_hash in zip(
                "89abcdef",
                self.plaintext_hashes,
            )
        ]

        def evidence_for(artifact, context):
            attempts = _storage_attempts_for_job(
                artifact["artifact_id"],
                context.job_id,
            )
            return {
                **artifact,
                "ciphertext_hash": _hash("c"),
                "artifact_ref": "s3://immutable/%s.json"
                % artifact["artifact_id"].split(":", 1)[1][:8],
                "storage_document_hash": _hash("d"),
                "encryption_context_hash": _hash("e"),
                "object_lock_mode": "COMPLIANCE",
                "retain_until": "2027-07-10T12:00:00Z",
                "transport_root": transport_root(attempts),
                "transport_attempts": attempts,
                "persisted": True,
            }
        self.manager = ExecutionJobManagerV2(
            boot_identity_supplier=lambda: self.boot,
            sign_digest=self.key.sign,
            operations=COORDINATOR_OPERATIONS_V2,
            executor=CoordinatorExecutorV2(
                artifact_evidence_supplier=lambda ids, context: [
                    evidence_for(artifact, context)
                    for artifact in self.artifacts
                    if artifact["artifact_id"] in ids
                ]
            ),
            worker_count=1,
            configured_worker_count=0,
        )

    async def v2_list_encrypted_artifacts(self, *, job_id, purpose):
        return {
            "artifacts": [
                {**artifact, "job_id": job_id, "purpose": purpose}
                for artifact in self.artifacts
            ]
        }

    async def v2_get_boot_identity(self):
        return self.boot

    async def coordinator_v2_health(self):
        return self.manager.health()

    async def coordinator_v2_submit_job(self, manifest):
        return self.manager.submit(manifest)

    async def coordinator_v2_put_chunk(self, *, job_id, offset, data):
        import hashlib

        return self.manager.put_chunk(
            job_id=job_id,
            offset=offset,
            data_b64=base64.b64encode(data).decode(),
            chunk_sha256="sha256:" + hashlib.sha256(data).hexdigest(),
        )

    async def coordinator_v2_seal_job(self, job_id):
        return self.manager.seal(job_id)

    async def coordinator_v2_get_status(self, job_id):
        return self.manager.status(job_id)

    async def coordinator_v2_cancel_job(self, job_id):
        return self.manager.cancel(job_id)

    async def coordinator_v2_get_result(self, job_id, *, offset=0):
        return self.manager.result_chunk(job_id=job_id, offset=offset)

    async def coordinator_v2_get_receipt(self, job_id):
        return self.manager.receipt(job_id)

    async def coordinator_v2_get_receipts(self, job_id):
        return list(self.manager.receipts(job_id))

    async def coordinator_v2_get_transport_attempts(self, job_id):
        return list(self.manager.transport_attempts(job_id))

    async def coordinator_v2_get_artifact_hashes(self, job_id):
        return list(self.manager.artifact_hashes(job_id))

    async def coordinator_v2_get_transitions(self, job_id):
        return list(self.manager.transitions(job_id))


@pytest.mark.asyncio
async def test_v2_bridge_persists_every_authenticated_provider_artifact_first():
    release = _release()

    def executor(operation, payload, context):
        for digest in (_hash("8"), _hash("9"), _hash("6")):
            context.record_artifact(digest)
        return ExecutionResultV2(
            output={"operation": operation, "echo": payload},
            transport_attempts=(_authenticated_attempt(context),),
        )

    client = _Client(release, executor=executor)
    persisted_artifacts = []
    persisted_graphs = []
    persisted_sidecars = []

    async def persist_artifact(artifact_id, **kwargs):
        persisted_artifacts.append((artifact_id, kwargs))
        return {
            "status": "persisted",
            "artifact_id": artifact_id,
            "storage_document_hash": _hash("d"),
            "artifact_kind": "provider_response",
            "artifact_hash": _hash("c"),
            "encryption_context_hash": _hash("e"),
            "object_lock_mode": "COMPLIANCE",
            "retain_until": "2027-07-10T12:00:00Z",
            "transport_root": transport_root(
                _storage_attempts_for_job(
                    artifact_id,
                    kwargs["attestation_job_id"],
                )
            ),
        }

    async def persist_graph(graph):
        persisted_graphs.append(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def persist_sidecars(**kwargs):
        persisted_sidecars.append(kwargs)
        return {"artifact_link_count": 1, "transition_count": 0}

    result = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=12,
        sequence=0,
        payload={"scores": [1.0]},
        worker_index=0,
        release_manifest=release,
        client=client,
        artifact_coordinator_client=_ArtifactCoordinator(
            release,
            (_hash("8"), _hash("6")),
        ),
        persist_artifact=persist_artifact,
        artifact_bucket="immutable-bucket",
        persist_graph=persist_graph,
        persist_sidecars=persist_sidecars,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    assert {item[0] for item in persisted_artifacts} == {_hash("8"), _hash("9")}
    assert persisted_artifacts[0][1]["bucket"] == "immutable-bucket"
    assert result["artifact_persistence"][0]["status"] == "persisted"
    assert result["receipt"]["purpose"] == "leadpoet.artifact_persistence.v2"
    assert result["execution_receipt"]["purpose"] == "research_lab.benchmark.v2"
    assert len(persisted_graphs) == 1
    assert persisted_sidecars[0]["artifact_receipt_hash"] == result["receipt"][
        "receipt_hash"
    ]


@pytest.mark.asyncio
async def test_v2_bridge_rejects_missing_encrypted_provider_artifact():
    release = _release()

    def executor(operation, payload, context):
        for digest in (_hash("8"), _hash("9"), _hash("f")):
            context.record_artifact(digest)
        return ExecutionResultV2(
            output={"operation": operation, "echo": payload},
            transport_attempts=(_authenticated_attempt(context),),
        )

    with pytest.raises(AttestedScoringV2Error, match="execution commitments"):
        await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0]},
            worker_index=0,
            release_manifest=release,
            client=_Client(release, executor=executor),
            artifact_coordinator_client=_ArtifactCoordinator(
                release,
                (_hash("8"), _hash("f")),
            ),
            artifact_bucket="immutable-bucket",
            persist_graph=lambda graph: graph,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )
