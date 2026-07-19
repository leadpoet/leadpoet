from __future__ import annotations

import base64
from datetime import datetime, timezone

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from gateway.tee.artifact_vault_v2 import (
    ARTIFACT_MASTER_KEY_HASH_DOMAIN,
    ArtifactVaultV2Error,
    EncryptedArtifactVaultV2,
    artifact_master_key_reference_hash,
)
from leadpoet_canonical.attested_v2 import build_transport_attempt, sha256_json


FIXED_NOW = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)
MASTER_KEY = bytes(range(32))
BOOT_HASH = "sha256:" + "a" * 64


def _vault() -> EncryptedArtifactVaultV2:
    return EncryptedArtifactVaultV2(
        master_key=MASTER_KEY,
        boot_identity_hash=BOOT_HASH,
        retention_days=30,
        clock=lambda: FIXED_NOW,
    )


def _sealed(vault: EncryptedArtifactVaultV2):
    return vault.seal(
        b"hidden provider response",
        job_id="job-1",
        purpose="research_lab.company_scoring.v2",
        artifact_kind="provider_response",
    )


def _headers(**overrides):
    values = {
        "x-amz-object-lock-mode": "COMPLIANCE",
        "x-amz-object-lock-retain-until-date": "2026-08-09T12:00:00.500Z",
    }
    values.update(overrides)
    return values


def _attempts(artifact_id, request_chars=("a", "b")):
    output = []
    for ordinal, method in enumerate(("GET", "HEAD")):
        output.append(
            build_transport_attempt(
                request_id=request_chars[ordinal] * 32,
                logical_operation_id="%s:%s" % (artifact_id, method.lower()),
                job_id=artifact_id,
                purpose="leadpoet.artifact_persistence.v2",
                provider_id="aws_s3_object_lock",
                attempt_number=ordinal,
                method=method,
                destination_host="immutable.example.s3.us-east-1.amazonaws.com",
                destination_port=443,
                path_hash="sha256:" + "1" * 64,
                nonsecret_headers_hash="sha256:" + "2" * 64,
                body_hash="sha256:" + "3" * 64,
                credential_ref_hash="sha256:" + "4" * 64,
                retry_policy_hash="sha256:" + "5" * 64,
                timeout_ms=30000,
                started_at="2026-07-10T12:00:00Z",
                terminal_status="authenticated_response",
                http_status=200,
                response_hash="sha256:" + "6" * 64,
                request_artifact_hash="sha256:" + "8" * 64,
                response_artifact_hash="sha256:" + "6" * 64,
                tls_peer_chain_hash="sha256:" + "7" * 64,
                tls_protocol="TLSv1.3",
                failure_code=None,
                completed_at="2026-07-10T12:00:01Z",
            )
        )
    return output


def test_artifact_master_key_reference_is_domain_separated() -> None:
    expected = __import__("hashlib").sha256(
        ARTIFACT_MASTER_KEY_HASH_DOMAIN + MASTER_KEY
    ).hexdigest()
    assert artifact_master_key_reference_hash(MASTER_KEY) == "sha256:" + expected


def test_seal_exposes_only_ciphertext_to_parent() -> None:
    vault = _vault()
    descriptor = _sealed(vault)
    exported = vault.export_ciphertext(descriptor["artifact_id"])
    document = exported["storage_document"]

    assert descriptor["persisted"] is False
    assert "ciphertext_b64" not in descriptor
    assert "hidden provider response" not in repr(exported)
    assert exported["storage_document_hash"] == sha256_json(document)

    plaintext = AESGCM(MASTER_KEY).decrypt(
        base64.b64decode(document["nonce_b64"]),
        base64.b64decode(document["ciphertext_b64"]),
        base64.b64decode(document["aad_b64"]),
    )
    assert plaintext == b"hidden provider response"


def test_persisted_envelope_reopens_after_coordinator_restart() -> None:
    first_boot = _vault()
    artifact_id = _sealed(first_boot)["artifact_id"]
    storage_document = first_boot.export_ciphertext(artifact_id)["storage_document"]

    restarted_boot = EncryptedArtifactVaultV2(
        master_key=MASTER_KEY,
        boot_identity_hash="sha256:" + "b" * 64,
        retention_days=30,
        clock=lambda: FIXED_NOW,
    )

    assert restarted_boot.decrypt_storage_document(storage_document) == (
        b"hidden provider response"
    )


def test_transient_envelope_can_be_released_after_durable_ciphertext_readback() -> None:
    vault = _vault()
    artifact_id = _sealed(vault)["artifact_id"]
    storage_document = vault.export_ciphertext(artifact_id)["storage_document"]

    vault.release_transient(artifact_id)

    with pytest.raises(ArtifactVaultV2Error, match="unavailable"):
        vault.descriptor(artifact_id)
    assert vault.decrypt_storage_document(storage_document) == b"hidden provider response"


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("ciphertext_b64", base64.b64encode(b"tampered").decode("ascii")),
        ("plaintext_hash", "sha256:" + "f" * 64),
        ("artifact_id", "sha256:" + "e" * 64),
        ("retain_until", "2026-08-10T12:00:00Z"),
    ),
)
def test_persisted_envelope_rejects_tampering(field, replacement) -> None:
    vault = _vault()
    artifact_id = _sealed(vault)["artifact_id"]
    storage_document = vault.export_ciphertext(artifact_id)["storage_document"]

    with pytest.raises(ArtifactVaultV2Error):
        vault.decrypt_storage_document(
            {**storage_document, field: replacement}
        )


def test_persisted_envelope_rejects_another_master_key() -> None:
    vault = _vault()
    artifact_id = _sealed(vault)["artifact_id"]
    storage_document = vault.export_ciphertext(artifact_id)["storage_document"]
    wrong_key_vault = EncryptedArtifactVaultV2(
        master_key=b"x" * 32,
        boot_identity_hash=BOOT_HASH,
        retention_days=30,
        clock=lambda: FIXED_NOW,
    )

    with pytest.raises(ArtifactVaultV2Error, match="authentication failed"):
        wrong_key_vault.decrypt_storage_document(storage_document)


def test_confirm_persistence_requires_exact_ciphertext_and_compliance_lock() -> None:
    vault = _vault()
    descriptor = _sealed(vault)
    artifact_id = descriptor["artifact_id"]
    document = vault.export_ciphertext(artifact_id)["storage_document"]

    confirmed = vault.confirm_persistence(
        artifact_id=artifact_id,
        artifact_ref="s3://immutable-bucket/artifacts/job-1.json",
        observed_storage_document=document,
        response_headers=_headers(),
        transport_attempts=_attempts(artifact_id),
    )

    assert confirmed["persisted"] is True
    assert confirmed["artifact_ref"].startswith("s3://immutable-bucket/")
    vault.require_persisted([artifact_id])

    with pytest.raises(ArtifactVaultV2Error, match="ciphertext differs"):
        vault.confirm_persistence(
            artifact_id=artifact_id,
            artifact_ref="s3://immutable-bucket/artifacts/job-1.json",
            observed_storage_document={**document, "ciphertext_b64": "tampered"},
            response_headers=_headers(),
            transport_attempts=_attempts(artifact_id),
        )

    with pytest.raises(ArtifactVaultV2Error, match="COMPLIANCE"):
        _vault().confirm_persistence(
            artifact_id=_sealed(_vault())["artifact_id"],
            artifact_ref="s3://immutable-bucket/artifacts/job-1.json",
            observed_storage_document=document,
            response_headers=_headers(**{"x-amz-object-lock-mode": "GOVERNANCE"}),
            transport_attempts=_attempts(artifact_id),
        )


def test_confirm_persistence_rejects_short_or_unzoned_retention() -> None:
    vault = _vault()
    artifact_id = _sealed(vault)["artifact_id"]
    document = vault.export_ciphertext(artifact_id)["storage_document"]

    with pytest.raises(ArtifactVaultV2Error, match="too short"):
        vault.confirm_persistence(
            artifact_id=artifact_id,
            artifact_ref="s3://immutable-bucket/artifacts/job-1.json",
            observed_storage_document=document,
            response_headers=_headers(
                **{"x-amz-object-lock-retain-until-date": "2026-08-09T11:59:59Z"}
            ),
            transport_attempts=_attempts(artifact_id),
        )

    with pytest.raises(ArtifactVaultV2Error, match="include timezone"):
        vault.confirm_persistence(
            artifact_id=artifact_id,
            artifact_ref="s3://immutable-bucket/artifacts/job-1.json",
            observed_storage_document=document,
            response_headers=_headers(
                **{"x-amz-object-lock-retain-until-date": "2026-08-09T12:00:00"}
            ),
            transport_attempts=_attempts(artifact_id),
        )


def test_persistence_record_is_immutable() -> None:
    vault = _vault()
    artifact_id = _sealed(vault)["artifact_id"]
    document = vault.export_ciphertext(artifact_id)["storage_document"]
    vault.confirm_persistence(
        artifact_id=artifact_id,
        artifact_ref="s3://immutable-bucket/artifacts/job-1.json",
        observed_storage_document=document,
        response_headers=_headers(),
        transport_attempts=_attempts(artifact_id),
    )

    with pytest.raises(ArtifactVaultV2Error, match="immutable"):
        vault.confirm_persistence(
            artifact_id=artifact_id,
            artifact_ref="s3://immutable-bucket/artifacts/rebound.json",
            observed_storage_document=document,
            response_headers=_headers(),
            transport_attempts=_attempts(artifact_id),
        )


def test_persistence_confirmation_is_idempotent_across_transport_retries() -> None:
    vault = _vault()
    artifact_id = _sealed(vault)["artifact_id"]
    document = vault.export_ciphertext(artifact_id)["storage_document"]
    first_attempts = _attempts(artifact_id)
    vault.confirm_persistence(
        artifact_id=artifact_id,
        artifact_ref="s3://immutable-bucket/artifacts/job-1.json",
        observed_storage_document=document,
        response_headers=_headers(),
        transport_attempts=first_attempts,
    )
    original_evidence = vault.persistence_evidence(artifact_id)

    confirmed = vault.confirm_persistence(
        artifact_id=artifact_id,
        artifact_ref="s3://immutable-bucket/artifacts/job-1.json",
        observed_storage_document=document,
        response_headers=_headers(),
        transport_attempts=_attempts(artifact_id, request_chars=("c", "d")),
    )

    assert confirmed["persisted"] is True
    assert vault.persistence_evidence(artifact_id) == original_evidence


def test_job_artifacts_are_job_and_purpose_scoped() -> None:
    vault = _vault()
    expected = _sealed(vault)
    vault.seal(
        b"other",
        job_id="job-2",
        purpose="research_lab.company_scoring.v2",
        artifact_kind="provider_response",
    )

    artifacts = vault.job_artifacts(
        job_id="job-1", purpose="research_lab.company_scoring.v2"
    )
    assert [item["artifact_id"] for item in artifacts] == [expected["artifact_id"]]
