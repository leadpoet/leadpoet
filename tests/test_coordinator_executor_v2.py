import pytest

from gateway.tee.artifact_vault_v2 import (
    ARTIFACT_PERSISTENCE_MAX_ATTEMPTS_PER_METHOD,
)
from gateway.tee.coordinator_executor_v2 import (
    OP_ATTEST_ARTIFACT_PERSISTENCE,
    CoordinatorExecutorV2,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    RECEIPT_GRAPH_SCHEMA_VERSION,
    build_transport_attempt,
    sha256_json,
    transport_root,
)



def _artifact_transport_attempts(
    artifact_id,
    job_id,
    sequence=(("GET", "transport_failure"), ("GET", "ok"), ("HEAD", "ok")),
):
    attempts = []
    for ordinal, (method, outcome) in enumerate(sequence):
        successful = outcome == "ok"
        attempts.append(
            build_transport_attempt(
                request_id=("%032x" % (ordinal + 1)),
                logical_operation_id="%s:%s" % (artifact_id, method.lower()),
                job_id=job_id,
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
                terminal_status=(
                    "authenticated_response"
                    if successful
                    else "transport_failure"
                ),
                http_status=200 if successful else None,
                response_hash="sha256:" + "6" * 64 if successful else None,
                request_artifact_hash="sha256:" + "8" * 64,
                response_artifact_hash=(
                    "sha256:" + "6" * 64 if successful else None
                ),
                tls_peer_chain_hash=(
                    "sha256:" + "7" * 64 if successful else None
                ),
                tls_protocol="TLSv1.3" if successful else None,
                failure_code=None if successful else "unexpected_eof",
                completed_at="2026-07-10T12:00:01Z",
            )
        )
    return attempts


def _artifact_evidence(artifact_id, plaintext_hash, attempts):
    return {
        "artifact_id": artifact_id,
        "plaintext_hash": plaintext_hash,
        "ciphertext_hash": "sha256:" + "e" * 64,
        "artifact_ref": "s3://immutable/artifact.json",
        "storage_document_hash": "sha256:" + "f" * 64,
        "encryption_context_hash": "sha256:" + "3" * 64,
        "object_lock_mode": "COMPLIANCE",
        "retain_until": "2027-07-10T12:00:00Z",
        "transport_root": transport_root(attempts),
        "transport_attempts": attempts,
        "persisted": True,
    }










@pytest.mark.asyncio
async def test_coordinator_rejects_operation_outside_measured_authority():
    with pytest.raises(ValueError, match="unsupported"):
        await CoordinatorExecutorV2()(
            "promotion_improvement",
            {},
            ExecutionContextV2(
                job_id="score:test",
                purpose="research_lab.ranking.v2",
                epoch_id=1,
            ),
        )






@pytest.mark.asyncio


@pytest.mark.asyncio


@pytest.mark.asyncio
async def test_coordinator_attests_only_complete_persisted_artifact_evidence():
    artifact_id = "sha256:" + "a" * 64
    plaintext_hash = "sha256:" + "b" * 64
    job_id = "artifact:test"
    attempts = _artifact_transport_attempts(artifact_id, job_id)
    evidence = _artifact_evidence(artifact_id, plaintext_hash, attempts)
    executor = CoordinatorExecutorV2(
        artifact_evidence_supplier=lambda _ids, _context: [evidence]
    )
    result = await executor(
        OP_ATTEST_ARTIFACT_PERSISTENCE,
        {
            "source_receipt_hash": "sha256:" + "2" * 64,
            "artifact_ids": [artifact_id],
            "artifact_plaintext_hashes": [plaintext_hash],
        },
        ExecutionContextV2(
            job_id=job_id,
            purpose="leadpoet.artifact_persistence.v2",
            epoch_id=1,
        ),
    )
    assert result.output["source_receipt_hash"] == "sha256:" + "2" * 64
    assert result.output["artifacts"][0]["artifact_id"] == artifact_id
    assert list(result.transport_attempts) == attempts


@pytest.mark.asyncio
async def test_coordinator_rejects_artifact_transport_root_mismatch():
    artifact_id = "sha256:" + "a" * 64
    plaintext_hash = "sha256:" + "b" * 64
    attempts = _artifact_transport_attempts(artifact_id, "artifact:test")
    evidence = {
        **_artifact_evidence(artifact_id, plaintext_hash, attempts),
        "transport_root": "sha256:" + "9" * 64,
    }
    executor = CoordinatorExecutorV2(
        artifact_evidence_supplier=lambda _ids, _context: [evidence]
    )
    with pytest.raises(ValueError, match="transport root differs"):
        await executor(
            OP_ATTEST_ARTIFACT_PERSISTENCE,
            {
                "source_receipt_hash": "sha256:" + "2" * 64,
                "artifact_ids": [artifact_id],
                "artifact_plaintext_hashes": [plaintext_hash],
            },
            ExecutionContextV2(
                job_id="artifact:test",
                purpose="leadpoet.artifact_persistence.v2",
                epoch_id=1,
            ),
        )


@pytest.mark.asyncio
async def test_coordinator_rejects_artifact_retry_evidence_over_budget():
    artifact_id = "sha256:" + "a" * 64
    plaintext_hash = "sha256:" + "b" * 64
    sequence = tuple(
        ("GET", "transport_failure")
        for _ in range(ARTIFACT_PERSISTENCE_MAX_ATTEMPTS_PER_METHOD)
    ) + (("GET", "ok"), ("HEAD", "ok"))
    attempts = _artifact_transport_attempts(
        artifact_id,
        "artifact:test",
        sequence=sequence,
    )
    evidence = _artifact_evidence(artifact_id, plaintext_hash, attempts)
    executor = CoordinatorExecutorV2(
        artifact_evidence_supplier=lambda _ids, _context: [evidence]
    )
    with pytest.raises(ValueError, match="transport evidence is incomplete"):
        await executor(
            OP_ATTEST_ARTIFACT_PERSISTENCE,
            {
                "source_receipt_hash": "sha256:" + "2" * 64,
                "artifact_ids": [artifact_id],
                "artifact_plaintext_hashes": [plaintext_hash],
            },
            ExecutionContextV2(
                job_id="artifact:test",
                purpose="leadpoet.artifact_persistence.v2",
                epoch_id=1,
            ),
        )


@pytest.mark.asyncio
async def test_coordinator_rejects_artifact_plaintext_mismatch():
    executor = CoordinatorExecutorV2(
        artifact_evidence_supplier=lambda _ids, _context: [
            {
                "artifact_id": "sha256:" + "a" * 64,
                "plaintext_hash": "sha256:" + "b" * 64,
                "persisted": True,
            }
        ]
    )
    with pytest.raises(ValueError, match="plaintext commitments differ"):
        await executor(
            OP_ATTEST_ARTIFACT_PERSISTENCE,
            {
                "source_receipt_hash": "sha256:" + "2" * 64,
                "artifact_ids": ["sha256:" + "a" * 64],
                "artifact_plaintext_hashes": ["sha256:" + "f" * 64],
            },
            ExecutionContextV2(
                job_id="artifact:test",
                purpose="leadpoet.artifact_persistence.v2",
                epoch_id=1,
            ),
        )
