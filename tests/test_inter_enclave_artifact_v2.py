import base64

import pytest

from gateway.tee.inter_enclave_artifact_v2 import (
    InterEnclaveArtifactIngestV2,
    InterEnclaveArtifactV2Error,
    seal_artifact_over_attested_tls_v2,
)
from leadpoet_canonical.attested_v2 import sha256_bytes


HASH = "sha256:" + "1" * 64


class _Vault:
    def __init__(self):
        self.calls = []

    def seal(self, plaintext, **kwargs):
        self.calls.append((bytes(plaintext), dict(kwargs)))
        return {
            "artifact_id": "sha256:" + "2" * 64,
            "plaintext_hash": sha256_bytes(plaintext),
            "ciphertext_hash": "sha256:" + "3" * 64,
            "artifact_kind": kwargs["artifact_kind"],
            "job_id": kwargs["job_id"],
            "purpose": kwargs["purpose"],
            "object_lock_mode": "COMPLIANCE",
            "retain_until": "2027-07-10T00:00:00Z",
            "encryption_context_hash": "sha256:" + "4" * 64,
            "persisted": False,
        }


def _peer(role="gateway_scoring", physical_role="gateway_scoring"):
    return {
        "physical_role": physical_role,
        "service_role": role,
        "boot_identity": {
            "role": role,
            "boot_identity_hash": HASH,
        },
    }


class _Client:
    def __init__(self, ingest, peer):
        self.ingest = ingest
        self.peer = peer
        self.methods = []

    def call(self, *, method, params, **_kwargs):
        self.methods.append(method)
        action = {
            "artifact_seal_begin": self.ingest.begin,
            "artifact_seal_chunk": self.ingest.put_chunk,
            "artifact_seal_finish": self.ingest.finish,
            "artifact_seal_cancel": self.ingest.cancel,
        }[method]
        return action(params, peer=self.peer)


def test_chunked_artifact_is_sealed_only_after_exact_attested_upload():
    vault = _Vault()
    ingest = InterEnclaveArtifactIngestV2(vault=vault)
    client = _Client(ingest, _peer())
    plaintext = b"measured" * 200000
    result = seal_artifact_over_attested_tls_v2(
        client=client,
        plaintext=plaintext,
        job_id="model-job-1",
        purpose="research_lab.private_model_run.v2",
        artifact_kind="model_trace",
    )
    assert result["plaintext_hash"] == sha256_bytes(plaintext)
    assert vault.calls == [
        (
            plaintext,
            {
                "job_id": "model-job-1",
                "purpose": "research_lab.private_model_run.v2",
                "artifact_kind": "model_trace",
            },
        )
    ]
    assert client.methods[0] == "artifact_seal_begin"
    assert client.methods[-1] == "artifact_seal_finish"
    assert client.methods.count("artifact_seal_chunk") > 1


def test_artifact_ingest_rejects_modified_chunk_and_different_peer():
    ingest = InterEnclaveArtifactIngestV2(vault=_Vault())
    started = ingest.begin(
        {
            "schema_version": "leadpoet.inter_enclave_artifact_upload.v2",
            "job_id": "model-job-1",
            "purpose": "research_lab.private_model_run.v2",
            "artifact_kind": "model_output",
            "plaintext_hash": sha256_bytes(b"expected"),
            "size_bytes": len(b"expected"),
        },
        peer=_peer(),
    )
    with pytest.raises(InterEnclaveArtifactV2Error, match="chunk hash"):
        ingest.put_chunk(
            {
                "upload_id": started["upload_id"],
                "offset": 0,
                "data_b64": base64.b64encode(b"altered").decode(),
                "chunk_sha256": sha256_bytes(b"expected"),
            },
            peer=_peer(),
        )
    with pytest.raises(InterEnclaveArtifactV2Error, match="peer differs"):
        ingest.put_chunk(
            {
                "upload_id": started["upload_id"],
                "offset": 0,
                "data_b64": base64.b64encode(b"expected").decode(),
                "chunk_sha256": sha256_bytes(b"expected"),
            },
            peer=_peer(physical_role="gateway_autoresearch"),
        )


def test_artifact_ingest_rejects_unauthorized_role_purpose_and_incomplete_upload():
    ingest = InterEnclaveArtifactIngestV2(vault=_Vault())
    with pytest.raises(InterEnclaveArtifactV2Error, match="not authorized"):
        ingest.begin(
            {
                "schema_version": "leadpoet.inter_enclave_artifact_upload.v2",
                "job_id": "job",
                "purpose": "validator.weights.computed.v2",
                "artifact_kind": "model_output",
                "plaintext_hash": sha256_bytes(b"expected"),
                "size_bytes": 8,
            },
            peer=_peer(),
        )
    started = ingest.begin(
        {
            "schema_version": "leadpoet.inter_enclave_artifact_upload.v2",
            "job_id": "job",
            "purpose": "research_lab.private_model_run.v2",
            "artifact_kind": "model_output",
            "plaintext_hash": sha256_bytes(b"expected"),
            "size_bytes": 8,
        },
        peer=_peer(),
    )
    with pytest.raises(InterEnclaveArtifactV2Error, match="incomplete"):
        ingest.finish({"upload_id": started["upload_id"]}, peer=_peer())
