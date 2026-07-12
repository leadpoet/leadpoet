from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gateway.tee.artifact_persistence_v2 import (
    ARTIFACT_POLICY_SCHEMA_VERSION,
    ArtifactPersistenceV2Error,
    ArtifactPersistenceVerifierV2,
    validate_artifact_policy,
)
from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from leadpoet_canonical.attested_v2 import canonical_json


POLICY = {
    "schema_version": ARTIFACT_POLICY_SCHEMA_VERSION,
    "bucket_host": "immutable.example.s3.us-east-1.amazonaws.com",
    "key_prefix": "/attested-v2/artifacts/",
    "minimum_retention_days": 365,
}
QUERY = (
    "X-Amz-Algorithm=AWS4-HMAC-SHA256&"
    "X-Amz-Credential=credential&X-Amz-Date=20260710T120000Z&"
    "X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature=abc123"
)


def _url(host=POLICY["bucket_host"]):
    return "https://%s/attested-v2/artifacts/item.json?%s" % (host, QUERY)


def _vault():
    return EncryptedArtifactVaultV2(
        master_key=bytes(range(32)),
        boot_identity_hash="sha256:" + "a" * 64,
        retention_days=365,
        clock=lambda: datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc),
    )


def _sealed(vault):
    descriptor = vault.seal(
        b"private response",
        job_id="job-1",
        purpose="research_lab.company_score.v2",
        artifact_kind="provider_response",
    )
    document = vault.export_ciphertext(descriptor["artifact_id"])["storage_document"]
    return descriptor, document


class _Transport:
    def __init__(self, document, *, get_status=200, head_status=200, head_mode="COMPLIANCE"):
        self.document = document
        self.get_status = get_status
        self.head_status = head_status
        self.head_mode = head_mode
        self.calls = []

    def __call__(self, *, method, url, headers, body, timeout_ms):
        self.calls.append((method, url, headers, body, timeout_ms))
        response_body = (
            canonical_json(self.document).encode("utf-8") if method == "GET" else b""
        )
        return {
            "http_status": self.get_status if method == "GET" else self.head_status,
            "headers": {
                "x-amz-object-lock-mode": self.head_mode,
                "x-amz-object-lock-retain-until-date": "2027-07-10T12:00:00Z",
            },
            "body": response_body,
            "tls_peer_chain_hash": "sha256:" + "b" * 64,
            "tls_protocol": "TLSv1.3",
        }


def test_policy_requires_exact_s3_https_boundary() -> None:
    assert validate_artifact_policy(POLICY) == POLICY
    with pytest.raises(ArtifactPersistenceV2Error, match="bucket host"):
        validate_artifact_policy({**POLICY, "bucket_host": "storage.example.com"})
    with pytest.raises(ArtifactPersistenceV2Error, match="key prefix"):
        validate_artifact_policy({**POLICY, "key_prefix": "/../artifacts/"})


def test_verifier_fetches_ciphertext_and_lock_headers_inside_tls() -> None:
    vault = _vault()
    descriptor, document = _sealed(vault)
    transport = _Transport(document)
    verifier = ArtifactPersistenceVerifierV2(
        vault=vault,
        policy=POLICY,
        transport=transport,
        clock=lambda: "2026-07-10T12:00:00Z",
    )

    result = verifier.verify(
        artifact_id=descriptor["artifact_id"],
        attestation_job_id="artifact-lineage-job",
        artifact_ref="s3://immutable.example/attested-v2/artifacts/item.json",
        get_url=_url(),
        head_url=_url(),
    )

    assert result["status"] == "persisted"
    assert result["artifact"]["persisted"] is True
    assert len(result["transport_attempts"]) == 2
    assert [call[0] for call in transport.calls] == ["GET", "HEAD"]


def test_verifier_rejects_host_redirection_and_unsigned_urls() -> None:
    vault = _vault()
    descriptor, document = _sealed(vault)
    verifier = ArtifactPersistenceVerifierV2(
        vault=vault,
        policy=POLICY,
        transport=_Transport(document),
    )
    with pytest.raises(ArtifactPersistenceV2Error, match="violates policy"):
        verifier.verify(
            artifact_id=descriptor["artifact_id"],
            attestation_job_id="artifact-lineage-job",
            artifact_ref="s3://immutable.example/item.json",
            get_url=_url("evil.example.com"),
            head_url=_url(),
        )
    with pytest.raises(ArtifactPersistenceV2Error, match="SigV4"):
        verifier.verify(
            artifact_id=descriptor["artifact_id"],
            attestation_job_id="artifact-lineage-job",
            artifact_ref="s3://immutable.example/item.json",
            get_url=_url().split("?", 1)[0],
            head_url=_url(),
        )


def test_authenticated_s3_error_is_not_a_transport_failure_or_success() -> None:
    vault = _vault()
    descriptor, document = _sealed(vault)
    verifier = ArtifactPersistenceVerifierV2(
        vault=vault,
        policy=POLICY,
        transport=_Transport(document, get_status=403),
        clock=lambda: "2026-07-10T12:00:00Z",
    )
    result = verifier.verify(
        artifact_id=descriptor["artifact_id"],
        attestation_job_id="artifact-lineage-job",
        artifact_ref="s3://immutable.example/item.json",
        get_url=_url(),
        head_url=_url(),
    )
    assert result["status"] == "failed"
    assert result["failure_code"] == "authenticated_http_403"
    assert result["transport_attempts"][0]["terminal_status"] == "authenticated_response"
    assert vault.descriptor(descriptor["artifact_id"])["persisted"] is False


def test_object_lock_mismatch_fails_after_authenticated_readback() -> None:
    vault = _vault()
    descriptor, document = _sealed(vault)
    verifier = ArtifactPersistenceVerifierV2(
        vault=vault,
        policy=POLICY,
        transport=_Transport(document, head_mode="GOVERNANCE"),
        clock=lambda: "2026-07-10T12:00:00Z",
    )
    result = verifier.verify(
        artifact_id=descriptor["artifact_id"],
        attestation_job_id="artifact-lineage-job",
        artifact_ref="s3://immutable.example/item.json",
        get_url=_url(),
        head_url=_url(),
    )
    assert result["status"] == "failed"
    assert result["failure_code"] == "object_lock_verification_failed"
    assert vault.descriptor(descriptor["artifact_id"])["persisted"] is False


def test_parent_drop_is_a_visible_transport_failure() -> None:
    vault = _vault()
    descriptor, _ = _sealed(vault)

    def dropped(**_kwargs):
        raise ConnectionResetError("relay reset")

    verifier = ArtifactPersistenceVerifierV2(
        vault=vault,
        policy=POLICY,
        transport=dropped,
        clock=lambda: "2026-07-10T12:00:00Z",
    )
    result = verifier.verify(
        artifact_id=descriptor["artifact_id"],
        attestation_job_id="artifact-lineage-job",
        artifact_ref="s3://immutable.example/item.json",
        get_url=_url(),
        head_url=_url(),
    )
    assert result["status"] == "failed"
    assert result["failure_code"] == "connection_reset"
    assert result["transport_attempts"][0]["terminal_status"] == "transport_failure"
