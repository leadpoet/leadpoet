from __future__ import annotations

import base64
import json
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.routing_model_binding_issuer import (
    RoutingModelBindingObservationIssuerError,
    RoutingModelBindingObservationIssuerV2,
)
from gateway.research_lab.routing_model_binding_observation import (
    ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
    build_routing_model_binding_observation_result_v2,
    routing_model_binding_requirements_hash,
)
from gateway.tee.execution_job_manager_v2 import JOB_SCHEMA_VERSION
from gateway.tee.scoring_executor_v2 import OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2
from leadpoet_canonical.attested_v2 import (
    build_execution_receipt_body,
    canonical_json,
    create_signed_execution_receipt,
    sha256_bytes,
)
from research_lab.canonical import sha256_json

from tests.test_routing_model_binding_producer import (
    _artifact_and_lineage,
)
from tests.test_routing_model_binding_observation import _binding_and_row


def _hash(char: str) -> str:
    return "sha256:" + char * 64


class _ObservationExecutor:
    def __init__(self, *, mutation: str | None = None):
        self.mutation = mutation
        self.manifest = None
        self.payload = b""
        self.result = None
        self.receipts = ()
        self._key = Ed25519PrivateKey.generate()
        self._pubkey = self._key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()

    def _summary(self, *, state: str, uploaded: int):
        assert self.manifest is not None
        return {
            "job_id": self.manifest["job_id"],
            "operation": self.manifest["operation"],
            "purpose": self.manifest["purpose"],
            "manifest_hash": sha256_bytes(canonical_json(self.manifest).encode()),
            "expected_bytes": self.manifest["payload_size_bytes"],
            "uploaded_bytes": uploaded,
            "state": state,
        }

    def submit_job(self, manifest):
        self.manifest = dict(manifest)
        return self._summary(state="uploading", uploaded=0)

    def put_chunk(self, *, job_id, offset, data_b64, chunk_sha256):
        assert job_id == self.manifest["job_id"]
        chunk = base64.b64decode(data_b64, validate=True)
        assert sha256_bytes(chunk) == chunk_sha256
        assert offset == len(self.payload)
        self.payload += chunk
        return self._summary(state="uploading", uploaded=len(self.payload))

    def seal_job(self, job_id):
        assert job_id == self.manifest["job_id"]
        payload = json.loads(self.payload.decode())
        binding = payload["provider_bindings"][0]
        binding_identity = sha256_json(
            {"schema_version": "leadpoet.routing_provider_binding.v1", "binding": binding}
        )
        observation = build_routing_model_binding_observation_result_v2(
            artifact_lineage_hash=sha256_json(
                {
                    "schema_version": "leadpoet.routing_artifact_lineage.v2",
                    **payload["artifact_lineage"],
                }
            ),
            requirement_hash_by_binding_identity={
                binding_identity: _hash("a"),
            },
        )
        self.result = {
            "schema_version": "leadpoet.routing_model_binding_result.v2",
            "operation": OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2,
            "artifact_lineage_hash": observation["artifact_lineage_hash"],
            "observation": observation,
        }
        stage_job_id = "stage:%s:0" % self.manifest["payload_sha256"].split(":", 1)[1][:24]
        release = {
            "commit_sha": "c" * 40,
            "pcr0": "d" * 96,
            "build_manifest_hash": _hash("b"),
            "dependency_lock_hash": _hash("e"),
            "config_hash": _hash("f"),
            "boot_identity_hash": _hash("1"),
        }
        stage = self._receipt(
            job_id=stage_job_id,
            sequence=0,
            input_root=observation["request_root"],
            output_root=sha256_json(observation),
            parents=[],
            **release,
        )
        final_release = dict(release)
        final_job_id = self.manifest["job_id"]
        final_input = sha256_bytes(self.payload)
        final_output = sha256_json(self.result)
        final_parents = [stage["receipt_hash"]]
        if self.mutation == "release":
            final_release["commit_sha"] = "9" * 40
        elif self.mutation == "job":
            final_job_id = "wrong-job"
        elif self.mutation == "input":
            final_input = _hash("8")
        elif self.mutation == "output":
            final_output = _hash("7")
        elif self.mutation == "parent":
            final_parents = []
        final = self._receipt(
            job_id=final_job_id,
            sequence=self.manifest["sequence"],
            input_root=final_input,
            output_root=final_output,
            parents=final_parents,
            **final_release,
        )
        if self.mutation == "signer":
            other_key = Ed25519PrivateKey.generate()
            final["enclave_pubkey"] = other_key.public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            ).hex()
        self.receipts = (stage, final)
        return self._summary(state="succeeded", uploaded=len(self.payload))

    def _receipt(self, *, job_id, sequence, input_root, output_root, parents, **release):
        body = build_execution_receipt_body(
            role="gateway_scoring",
            purpose=ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
            job_id=job_id,
            epoch_id=self.manifest["epoch_id"],
            sequence=sequence,
            input_root=input_root,
            output_root=output_root,
            transport_root_hash=_hash("2"),
            host_operation_root_hash=_hash("3"),
            artifact_root=_hash("4"),
            parent_receipt_hashes=parents,
            status="succeeded",
            failure_code=None,
            issued_at="2026-08-19T00:00:00Z",
            **release,
        )
        return create_signed_execution_receipt(
            body=body,
            enclave_pubkey=self._pubkey,
            sign_digest=self._key.sign,
        )

    def get_status(self, job_id):
        return self._summary(state="succeeded", uploaded=len(self.payload))

    def get_result_chunk(self, *, job_id, offset, max_bytes):
        body = canonical_json(self.result).encode()
        chunk = body[offset : offset + max_bytes]
        return {
            "job_id": job_id,
            "offset": offset,
            "data_b64": base64.b64encode(chunk).decode(),
            "chunk_sha256": sha256_bytes(chunk),
            "result_sha256": sha256_bytes(body),
            "total_size_bytes": len(body),
            "eof": offset + len(chunk) >= len(body),
        }

    def get_receipts(self, job_id):
        return self.receipts


def _inputs(tmp_path):
    artifact, lineage, source_bundle = _artifact_and_lineage(tmp_path)
    binding, _row = _binding_and_row()
    return artifact, lineage, source_bundle, binding


def test_issuer_uploads_exact_no_credential_observation_and_verifies_chain(tmp_path):
    artifact, lineage, source_bundle, binding = _inputs(tmp_path)
    executor = _ObservationExecutor()
    observation = RoutingModelBindingObservationIssuerV2(
        executor=executor,
        poll_interval_seconds=0,
    ).issue(
        job_id="routing-observation-issuer",
        epoch_id=24301,
        sequence=4,
        model_kind="private",
        artifact_lineage=lineage,
        artifact_document=artifact.to_dict(),
        source_bundle=source_bundle,
        provider_bindings=(binding,),
    )
    assert observation.observation_receipt_hash == executor.receipts[0]["receipt_hash"]
    assert executor.manifest["provider_credential_profile"] == "default"
    assert executor.manifest["provider_credential_ref_hashes"] == {}
    request = json.loads(executor.payload)
    assert set(request) == {
        "schema_version",
        "model_kind",
        "artifact_lineage",
        "artifact",
        "source_bundle",
        "provider_bindings",
    }
    assert not any("credential" in key.lower() for key in request)


@pytest.mark.parametrize("mutation", ["signer", "release", "job", "input", "output", "parent"])
def test_issuer_rejects_receipt_identity_substitution(tmp_path, mutation):
    artifact, lineage, source_bundle, binding = _inputs(tmp_path)
    executor = _ObservationExecutor(mutation=mutation)
    with pytest.raises(RoutingModelBindingObservationIssuerError):
        RoutingModelBindingObservationIssuerV2(
            executor=executor,
            poll_interval_seconds=0,
        ).issue(
            job_id="routing-observation-issuer",
            epoch_id=24301,
            sequence=4,
            model_kind="private",
            artifact_lineage=lineage,
            artifact_document=artifact.to_dict(),
            source_bundle=source_bundle,
            provider_bindings=(binding,),
        )
