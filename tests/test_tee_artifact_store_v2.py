from __future__ import annotations

import json

import pytest

from gateway.utils.tee_artifact_store_v2 import (
    TEEArtifactStoreV2Error,
    persist_enclave_artifact_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json


ARTIFACT_ID = "sha256:" + "a" * 64


def _document():
    return {
        "schema_version": "leadpoet.encrypted_artifact.v2",
        "artifact_id": ARTIFACT_ID,
        "plaintext_hash": "sha256:" + "b" * 64,
        "ciphertext_hash": "sha256:" + "c" * 64,
        "nonce_b64": "bm9uY2U=",
        "aad_b64": "YWFk",
        "encryption_context_hash": "sha256:" + "e" * 64,
        "ciphertext_b64": "Y2lwaGVydGV4dA==",
        "object_lock_mode": "COMPLIANCE",
        "retain_until": "2027-07-10T12:00:00Z",
    }


class _Client:
    def __init__(self, *, verified=True):
        self.document = _document()
        self.verified = verified
        self.verification = None

    async def v2_export_encrypted_artifact(self, artifact_id):
        assert artifact_id == ARTIFACT_ID
        return {
            "storage_document": self.document,
            "storage_document_hash": sha256_json(self.document),
        }

    async def v2_verify_encrypted_artifact_persistence(self, **kwargs):
        self.verification = kwargs
        if not self.verified:
            return {"status": "failed", "failure_code": "object_lock_verification_failed"}
        return {"status": "persisted", "transport_root": "sha256:" + "d" * 64}


class _S3:
    def __init__(self):
        self.put = None
        self.presigns = []

    def put_object(self, **kwargs):
        self.put = kwargs
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def generate_presigned_url(self, operation, **kwargs):
        self.presigns.append((operation, kwargs))
        return "https://immutable.example.s3.us-east-1.amazonaws.com/path?" + operation


@pytest.mark.asyncio
async def test_parent_uploads_only_canonical_ciphertext_and_requests_readback() -> None:
    client = _Client()
    s3 = _S3()
    result = await persist_enclave_artifact_v2(
        ARTIFACT_ID,
        bucket="immutable-example",
        attestation_job_id="artifact-lineage-job",
        client=client,
        s3_client=s3,
    )

    body = json.loads(s3.put["Body"])
    assert body == _document()
    assert b"private response" not in s3.put["Body"]
    assert s3.put["ObjectLockMode"] == "COMPLIANCE"
    assert [item[0] for item in s3.presigns] == ["get_object", "head_object"]
    assert client.verification["artifact_ref"].startswith("s3://immutable-example/")
    assert result["status"] == "persisted"


@pytest.mark.asyncio
async def test_parent_fails_when_enclave_rejects_storage() -> None:
    with pytest.raises(TEEArtifactStoreV2Error, match="enclave rejected"):
        await persist_enclave_artifact_v2(
            ARTIFACT_ID,
            bucket="immutable-example",
            attestation_job_id="artifact-lineage-job",
            client=_Client(verified=False),
            s3_client=_S3(),
        )


@pytest.mark.asyncio
async def test_parent_rejects_export_hash_mismatch() -> None:
    client = _Client()

    async def bad_export(_artifact_id):
        return {
            "storage_document": _document(),
            "storage_document_hash": "sha256:" + "f" * 64,
        }

    client.v2_export_encrypted_artifact = bad_export
    with pytest.raises(TEEArtifactStoreV2Error, match="hash mismatch"):
        await persist_enclave_artifact_v2(
            ARTIFACT_ID,
            bucket="immutable-example",
            attestation_job_id="artifact-lineage-job",
            client=client,
            s3_client=_S3(),
        )
