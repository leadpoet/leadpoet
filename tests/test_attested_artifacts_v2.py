from __future__ import annotations

import pytest

from gateway.research_lab import attested_artifacts_v2, attested_v2_store
from leadpoet_canonical.attested_v2 import merkle_root


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _artifacts(*, persisted: bool) -> list[dict]:
    return [
        {
            "artifact_id": _hash(character),
            "plaintext_hash": plaintext_hash,
            "ciphertext_hash": _hash("c"),
            "encryption_context_hash": _hash("e"),
            "artifact_kind": "provider_response",
            "persisted": persisted,
        }
        for character, plaintext_hash in (("a", _hash("1")), ("b", _hash("2")))
    ]


def _lineage_evidence(artifacts: list[dict]) -> list[dict]:
    return [
        {
            "artifact_id": item["artifact_id"],
            "plaintext_hash": item["plaintext_hash"],
            "ciphertext_hash": item["ciphertext_hash"],
            "artifact_ref": "s3://immutable/%s.json" % index,
            "storage_document_hash": _hash("d"),
            "encryption_context_hash": item["encryption_context_hash"],
            "object_lock_mode": "COMPLIANCE",
            "retain_until": "2027-07-10T12:00:00Z",
            "transport_root": _hash("f"),
        }
        for index, item in enumerate(artifacts)
    ]


async def _exercise(
    monkeypatch: pytest.MonkeyPatch,
    *,
    replay: bool,
    partial: bool = False,
) -> dict:
    artifacts = _artifacts(persisted=replay)
    if partial:
        artifacts[0]["persisted"] = True
    committed = [_hash("1"), _hash("2")]
    source_receipt = {
        "receipt_hash": _hash("9"),
        "artifact_root": merkle_root(
            committed,
            domain="leadpoet-artifact-v2",
        ),
    }
    source_graph = {
        "root_receipt_hash": source_receipt["receipt_hash"],
        "receipts": [],
    }
    transport_attempts = [
        {
            "request_artifact_hash": _hash("1"),
            "response_artifact_hash": _hash("2"),
            "terminal_status": "authenticated_response",
        }
    ]
    persistence_job_ids = []

    class Client:
        async def v2_list_encrypted_artifacts(self, *, job_id, purpose):
            assert job_id == "source-job"
            assert purpose == "research_lab.test.v2"
            return {"artifacts": artifacts}

    async def persist_artifact(artifact_id, **kwargs):
        if replay:
            raise AssertionError("persisted artifacts must not be uploaded again")
        persistence_job_ids.append(kwargs["attestation_job_id"])
        descriptor = next(
            item for item in artifacts if item["artifact_id"] == artifact_id
        )
        return {
            "status": "persisted",
            "artifact_id": artifact_id,
            "artifact_ref": "s3://immutable/%s.json" % artifact_id[-1],
            "artifact_kind": descriptor["artifact_kind"],
            "artifact_hash": descriptor["ciphertext_hash"],
            "encryption_context_hash": descriptor["encryption_context_hash"],
            "object_lock_mode": "COMPLIANCE",
            "retain_until": "2027-07-10T12:00:00Z",
            "storage_document_hash": _hash("d"),
            "transport_root": _hash("f"),
        }

    async def execute(**_kwargs):
        job_id = persistence_job_ids[0] if persistence_job_ids else expected_job_id[0]
        receipt = {
            "job_id": job_id,
            "receipt_hash": _hash("8"),
        }
        return {
            "status": "succeeded",
            "result": {"artifacts": _lineage_evidence(artifacts)},
            "receipt": receipt,
            "receipt_graph": {
                "root_receipt_hash": receipt["receipt_hash"],
                "receipts": [receipt],
            },
        }

    async def persist_sidecars(**kwargs):
        return {"artifact_link_count": len(kwargs["artifacts"])}

    expected_job_id = []
    original_derive = __import__(
        "gateway.research_lab.attested_scoring_v2",
        fromlist=["derive_execution_job_id_v2"],
    ).derive_execution_job_id_v2

    def capture_job_id(**kwargs):
        value = original_derive(**kwargs)
        expected_job_id.append(value)
        return value

    monkeypatch.setattr(attested_artifacts_v2, "validate_receipt_graph", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "gateway.research_lab.attested_scoring_v2.derive_execution_job_id_v2",
        capture_job_id,
    )
    monkeypatch.setattr(
        attested_artifacts_v2,
        "persist_enclave_artifact_v2",
        persist_artifact,
    )
    monkeypatch.setattr(attested_artifacts_v2, "execute_coordinator_v2", execute)
    monkeypatch.setattr(
        attested_v2_store,
        "persist_execution_sidecars_v2",
        persist_sidecars,
    )

    result = await attested_artifacts_v2.persist_execution_transport_artifacts_v2(
        job_id="source-job",
        purpose="research_lab.test.v2",
        epoch_id=12,
        sequence=3,
        source_receipt=source_receipt,
        source_graph=source_graph,
        transport_attempts=transport_attempts,
        execution_artifact_hashes=committed,
        release_manifest={"release_hash": _hash("7")},
        client=Client(),
        bucket=None if replay else "immutable-bucket",
    )
    assert result["receipt"]["job_id"] == expected_job_id[0]
    if partial:
        assert len(persistence_job_ids) == 1
    return result


@pytest.mark.asyncio
async def test_transport_artifacts_bind_plaintext_commitments_to_lineage_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _exercise(monkeypatch, replay=False)
    assert len(result["artifacts"]) == 2


@pytest.mark.asyncio
async def test_transport_artifacts_reuse_attested_persistence_on_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _exercise(monkeypatch, replay=True)
    assert all(item["status"] == "persisted" for item in result["artifacts"])


@pytest.mark.asyncio
async def test_transport_artifacts_resume_after_partial_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _exercise(monkeypatch, replay=False, partial=True)
    assert len(result["artifacts"]) == 2
    assert all(item["status"] == "persisted" for item in result["artifacts"])
