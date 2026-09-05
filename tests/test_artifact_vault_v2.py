from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from gateway.tee import artifact_vault_v2
from gateway.tee.artifact_vault_v2 import (
    ARTIFACT_MASTER_KEY_HASH_DOMAIN,
    ArtifactVaultV2Error,
    EncryptedArtifactVaultV2,
    artifact_master_key_reference_hash,
)
from gateway.tee.topology import COORDINATOR_ROLE, ROLE_SPECS, topology_document
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


def test_artifact_capacity_is_bounded_by_measured_coordinator_memory() -> None:
    coordinator_bytes = int(ROLE_SPECS[COORDINATOR_ROLE]["memory_mib"]) * 1024 * 1024
    assert artifact_vault_v2.MAX_IN_MEMORY_ARTIFACT_BYTES == coordinator_bytes // 4


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
    assert vault.transient_capacity_state()["transient_artifact_bytes"] == 0
    assert vault.decrypt_storage_document(storage_document) == b"hidden provider response"


def test_measured_scoring_pool_crosses_old_byte_ceiling_and_recovers_after_persistence(
    monkeypatch,
) -> None:
    vault = _vault()
    jobs = int(topology_document()["benchmark_concurrency"])
    envelopes_per_job = 5000
    measured_encoded_bytes_per_envelope = 64 * 1024
    monkeypatch.setattr(
        vault,
        "_record_memory_bytes",
        lambda _record: measured_encoded_bytes_per_envelope,
    )
    first_artifacts = []

    for job_index in range(jobs):
        for artifact_index in range(envelopes_per_job):
            descriptor = vault.seal(
                f"provider-{job_index}-{artifact_index}".encode(),
                job_id=f"scoring-job-{job_index}",
                purpose="research_lab.source_add_judge.v2",
                artifact_kind="provider_response",
            )
            if artifact_index == 0:
                first_artifacts.append(descriptor)

    capacity = vault.transient_capacity_state()
    assert capacity["transient_artifact_count"] == jobs * envelopes_per_job
    assert capacity["transient_artifact_count"] > 16384
    assert capacity["transient_artifact_bytes"] > 1024 * 1024 * 1024
    assert capacity["transient_artifact_count"] < (
        artifact_vault_v2.MAX_IN_MEMORY_ARTIFACTS
    )
    assert capacity["transient_artifact_bytes"] < (
        artifact_vault_v2.MAX_IN_MEMORY_ARTIFACT_BYTES
    )
    assert capacity["active_artifact_job_count"] == jobs

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        with vault.transient_artifact_transaction():
            vault.seal(
                b"failed checkpoint",
                job_id="scoring-job-0",
                purpose="research_lab.source_add_judge.v2",
                artifact_kind="provider_outcome_checkpoint",
            )
            raise RuntimeError("checkpoint failed")
    assert vault.transient_capacity_state()["transient_artifact_count"] == (
        jobs * envelopes_per_job
    )

    for descriptor in first_artifacts:
        artifact_id = descriptor["artifact_id"]
        vault.confirm_persistence(
            artifact_id=artifact_id,
            artifact_ref=f"s3://immutable-bucket/artifacts/{artifact_id}.json",
            observed_storage_document=vault.export_ciphertext(artifact_id)[
                "storage_document"
            ],
            response_headers=_headers(),
            transport_attempts=_attempts(artifact_id),
        )

    recovered = vault.seal(
        b"recovered checkpoint",
        job_id="scoring-job-recovery",
        purpose="research_lab.source_add_judge.v2",
        artifact_kind="provider_outcome_checkpoint",
    )
    assert recovered["persisted"] is False
    assert vault.transient_capacity_state()["transient_artifact_count"] == (
        jobs * envelopes_per_job - jobs + 1
    )


def test_transient_transaction_discards_only_failed_thread_artifacts() -> None:
    vault = _vault()
    failing_sealed = threading.Event()
    successful_sealed = threading.Event()

    def failing_transaction():
        with pytest.raises(RuntimeError, match="persistence failed"):
            with vault.transient_artifact_transaction():
                descriptor = vault.seal(
                    b"failed checkpoint",
                    job_id="shared-job",
                    purpose="research_lab.provider_preflight.v2",
                    artifact_kind="provider_outcome_checkpoint",
                )
                failing_sealed.set()
                assert successful_sealed.wait(timeout=2.0)
                raise RuntimeError("persistence failed")
        return descriptor["artifact_id"]

    def successful_transaction():
        assert failing_sealed.wait(timeout=2.0)
        with vault.transient_artifact_transaction():
            descriptor = vault.seal(
                b"successful provider response",
                job_id="shared-job",
                purpose="research_lab.provider_preflight.v2",
                artifact_kind="provider_response",
            )
            successful_sealed.set()
        return descriptor["artifact_id"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        failed_id = executor.submit(failing_transaction)
        successful_id = executor.submit(successful_transaction)
        failed_id = failed_id.result(timeout=3.0)
        successful_id = successful_id.result(timeout=3.0)

    with pytest.raises(ArtifactVaultV2Error, match="unavailable"):
        vault.descriptor(failed_id)
    assert vault.descriptor(successful_id)["persisted"] is False
    capacity = vault.transient_capacity_state()
    assert capacity["transient_artifact_count"] == 1
    assert capacity["transient_artifact_bytes"] > 0
    assert [item["artifact_id"] for item in vault.job_artifacts(
        job_id="shared-job",
        purpose="research_lab.provider_preflight.v2",
    )] == [successful_id]


def test_transient_transaction_discards_cancelled_artifacts() -> None:
    vault = _vault()

    with pytest.raises(KeyboardInterrupt):
        with vault.transient_artifact_transaction():
            vault.seal(
                b"cancelled provider response",
                job_id="cancelled-job",
                purpose="research_lab.provider_preflight.v2",
                artifact_kind="provider_response",
            )
            raise KeyboardInterrupt

    assert vault.job_artifacts(
        job_id="cancelled-job",
        purpose="research_lab.provider_preflight.v2",
    ) == ()
    assert vault.transient_capacity_state()["transient_artifact_bytes"] == 0


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


def test_confirmed_artifact_stops_consuming_active_vault_capacity(
    monkeypatch,
) -> None:
    from gateway.tee import artifact_vault_v2

    monkeypatch.setattr(artifact_vault_v2, "MAX_IN_MEMORY_ARTIFACTS", 1)
    vault = _vault()
    first = _sealed(vault)
    first_id = first["artifact_id"]
    document = vault.export_ciphertext(first_id)["storage_document"]

    vault.confirm_persistence(
        artifact_id=first_id,
        artifact_ref="s3://immutable-bucket/artifacts/job-1.json",
        observed_storage_document=document,
        response_headers=_headers(),
        transport_attempts=_attempts(first_id),
    )

    second = vault.seal(
        b"second hidden provider response",
        job_id="job-2",
        purpose="research_lab.company_scoring.v2",
        artifact_kind="provider_response",
    )
    assert second["persisted"] is False
    assert vault.descriptor(first_id)["persisted"] is True
    assert vault.persistence_evidence(first_id)["artifact_ref"].startswith(
        "s3://immutable-bucket/"
    )
    with pytest.raises(ArtifactVaultV2Error, match="unavailable"):
        vault.export_ciphertext(first_id)


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
