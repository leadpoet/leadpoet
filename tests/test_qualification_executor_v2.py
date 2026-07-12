from __future__ import annotations

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.qualification_executor_v2 import (
    QUALIFICATION_BATCH_INPUT_SCHEMA_VERSION,
    QUALIFICATION_EMAIL_INPUT_SCHEMA_VERSION,
    QUALIFICATION_EPOCH_INPUT_SCHEMA_VERSION,
    QualificationExecutorV2,
    QualificationExecutorV2Error,
)
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_execution_receipt_body,
    create_signed_execution_receipt,
    sha256_json,
)


HASH = "sha256:" + "a" * 64
SALT = "12" * 32
NOW = "2026-07-10T20:00:00Z"


def _receipt(*, role, purpose, epoch, output, job_id):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    body = build_execution_receipt_body(
        role=role,
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch,
        sequence=0,
        commit_sha="b" * 40,
        pcr0="c" * 96,
        build_manifest_hash=HASH,
        dependency_lock_hash="sha256:" + "d" * 64,
        config_hash="sha256:" + "e" * 64,
        boot_identity_hash="sha256:" + "f" * 64,
        input_root=HASH,
        output_root=sha256_json(output),
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=EMPTY_ARTIFACT_ROOT,
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )


def _lead():
    return {
        "lead_id": "lead-1",
        "miner_hotkey": "miner-a",
        "lead_blob": {"email": "a@example.com"},
    }


@pytest.mark.asyncio
async def test_batch_executes_existing_runner_and_binds_both_source_receipts():
    observed = {}

    async def run_batch(leads, **kwargs):
        observed["leads"] = leads
        observed.update(kwargs)
        return [
            (
                True,
                {
                    "rep_score": {"total_score": 40},
                    "is_icp_multiplier": 20.0,
                },
            )
        ]

    leads = [_lead()]
    email_results = {"a@example.com": {"passed": True}}
    admission_doc = {
        "epoch_id": 100,
        "container_id": 2,
        "sequence_start": 7,
        "leads": leads,
        "salt_hex": SALT,
    }
    email_doc = {
        "epoch_id": 100,
        "precomputed_email_results": email_results,
    }
    admission = _receipt(
        role="gateway_coordinator",
        purpose="research_lab.admission.v2",
        epoch=100,
        output=admission_doc,
        job_id="qualification-admission:100:2",
    )
    email = _receipt(
        role="gateway_scoring",
        purpose="qualification.email_evidence.v2",
        epoch=100,
        output=email_doc,
        job_id="qualification-email:100",
    )
    context = ExecutionContextV2(
        job_id="qualification-batch:100:2",
        purpose="qualification.lead_decision.v2",
        epoch_id=100,
        parent_receipt_hashes=(admission["receipt_hash"], email["receipt_hash"]),
    )
    result = await QualificationExecutorV2(run_batch=run_batch).execute_batch(
        {
            "schema_version": QUALIFICATION_BATCH_INPUT_SCHEMA_VERSION,
            "epoch_id": 100,
            "container_id": 2,
            "sequence_start": 7,
            "leads": leads,
            "precomputed_email_results": email_results,
            "salt_hex": SALT,
            "admission_receipt": admission,
            "email_evidence_receipt": email,
        },
        context,
    )
    assert observed["leads"] == [leads[0]["lead_blob"]]
    assert observed["container_id"] == 2
    assert observed["precomputed_email_results"] is None
    assert observed["current_epoch"] == 100
    assert result.output["sourcing_decisions"][0]["effective_rep_score"] == 60
    assert len(result.artifact_hashes) == 2

    missing_parent = ExecutionContextV2(
        job_id="qualification-batch:100:2",
        purpose="qualification.lead_decision.v2",
        epoch_id=100,
        parent_receipt_hashes=(admission["receipt_hash"],),
    )
    with pytest.raises(QualificationExecutorV2Error, match="declared input"):
        await QualificationExecutorV2(run_batch=run_batch).execute_batch(
            {
                "schema_version": QUALIFICATION_BATCH_INPUT_SCHEMA_VERSION,
                "epoch_id": 100,
                "container_id": 2,
                "sequence_start": 7,
                "leads": leads,
                "precomputed_email_results": email_results,
                "salt_hex": SALT,
                "admission_receipt": admission,
                "email_evidence_receipt": email,
            },
            missing_parent,
        )


@pytest.mark.asyncio
async def test_email_evidence_executes_existing_centralized_truelist_once():
    calls = []

    async def run_email_batch(leads):
        calls.append(leads)
        return {"a@example.com": {"passed": True, "status": "email_ok"}}

    result = await QualificationExecutorV2(
        run_email_batch=run_email_batch
    ).execute_email_evidence(
        {
            "schema_version": QUALIFICATION_EMAIL_INPUT_SCHEMA_VERSION,
            "epoch_id": 100,
            "leads": [_lead()],
        },
        ExecutionContextV2(
            job_id="qualification-email:100",
            purpose="qualification.email_evidence.v2",
            epoch_id=100,
        ),
    )

    assert calls == [[_lead()]]
    assert result.output == {
        "epoch_id": 100,
        "precomputed_email_results": {
            "a@example.com": {"passed": True, "status": "email_ok"}
        },
    }
    assert len(result.artifact_hashes) == 2


def test_epoch_aggregation_requires_every_signed_batch_parent():
    from leadpoet_canonical.qualification_batch_v2 import (
        build_qualification_batch_output_v2,
    )

    batch = build_qualification_batch_output_v2(
        epoch_id=100,
        container_id=0,
        sequence_start=0,
        leads=[_lead()],
        batch_results=[
            (True, {"rep_score": {"total_score": 40}, "is_icp_multiplier": 20.0})
        ],
        salt_hex=SALT,
    )
    receipt = _receipt(
        role="gateway_scoring",
        purpose="qualification.lead_decision.v2",
        epoch=100,
        output=batch,
        job_id="qualification-batch:100:0",
    )
    context = ExecutionContextV2(
        job_id="qualification-epoch:100",
        purpose="qualification.sourcing_epoch.v2",
        epoch_id=100,
        parent_receipt_hashes=(receipt["receipt_hash"],),
    )
    result = QualificationExecutorV2().aggregate_epoch(
        {
            "schema_version": QUALIFICATION_EPOCH_INPUT_SCHEMA_VERSION,
            "epoch_id": 100,
            "batches": [{"receipt": receipt, "output": batch}],
        },
        context,
    )
    assert result.output["miner_scores"] == [{"hotkey": "miner-a", "score": 60}]
    assert result.output["approved_lead_count"] == 1
    assert result.output["decision_count"] == 1
