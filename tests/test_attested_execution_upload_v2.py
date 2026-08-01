from __future__ import annotations

import pytest

from gateway.research_lab.attested_execution_upload_v2 import (
    AttestedExecutionUploadV2Error,
    upload_attested_execution_job_v2,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes


def _manifest(payload: bytes) -> dict:
    return {
        "schema_version": "leadpoet.tee_execution_job.v2",
        "job_id": "concurrent-job",
        "operation": "benchmark_icp_score",
        "purpose": "research_lab.benchmark.v2",
        "epoch_id": 42,
        "sequence": 0,
        "payload_sha256": sha256_bytes(payload),
        "payload_size_bytes": len(payload),
        "parent_receipt_hashes": [],
        "input_artifact_hashes": [],
        "provider_credential_profile": "default",
        "provider_credential_ref_hashes": {},
    }


def _summary(manifest: dict, *, uploaded_bytes: int = 0) -> dict:
    return {
        "job_id": manifest["job_id"],
        "operation": manifest["operation"],
        "purpose": manifest["purpose"],
        "state": "uploading",
        "manifest_hash": sha256_bytes(
            canonical_json(manifest).encode("utf-8")
        ),
        "uploaded_bytes": uploaded_bytes,
        "expected_bytes": manifest["payload_size_bytes"],
    }


@pytest.mark.asyncio
async def test_upload_rejects_summary_for_another_immutable_manifest():
    payload = b'{"value":1}'
    manifest = _manifest(payload)

    async def submit(_manifest):
        return {
            **_summary(manifest),
            "manifest_hash": "sha256:" + "f" * 64,
        }

    with pytest.raises(
        AttestedExecutionUploadV2Error,
        match="differs from immutable manifest",
    ):
        await upload_attested_execution_job_v2(
            manifest=manifest,
            payload=payload,
            chunk_size=4,
            submit_job=submit,
            put_chunk=lambda **_kwargs: None,
            seal_job=lambda _job_id: None,
            get_status=lambda _job_id: None,
        )


@pytest.mark.asyncio
async def test_upload_does_not_retry_a_chunk_without_observed_progress():
    payload = b'{"value":1}'
    manifest = _manifest(payload)
    chunk_calls = 0

    async def submit(_manifest):
        return _summary(manifest)

    async def put_chunk(**_kwargs):
        nonlocal chunk_calls
        chunk_calls += 1
        raise RuntimeError("rejected")

    async def status(_job_id):
        return _summary(manifest)

    with pytest.raises(
        AttestedExecutionUploadV2Error,
        match="failed without monotonic progress",
    ):
        await upload_attested_execution_job_v2(
            manifest=manifest,
            payload=payload,
            chunk_size=4,
            submit_job=submit,
            put_chunk=put_chunk,
            seal_job=lambda _job_id: None,
            get_status=status,
        )

    assert chunk_calls == 1
