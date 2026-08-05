from __future__ import annotations

import pytest

from gateway.research_lab import (
    attested_artifacts_v2,
    attested_scoring_v2,
    attested_v2_store,
)
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


def test_only_source_receipt_committed_envelopes_are_selected():
    artifacts = _artifacts(persisted=False)
    artifacts.append(
        {
            "artifact_id": _hash("f"),
            "plaintext_hash": _hash("3"),
            "ciphertext_hash": _hash("4"),
            "encryption_context_hash": _hash("5"),
            "artifact_kind": "provider_request",
            "persisted": False,
        }
    )

    selected = attested_artifacts_v2._select_committed_encrypted_artifacts(
        artifacts,
        committed_hashes=(_hash("1"), _hash("2")),
    )

    assert [item["artifact_id"] for item in selected] == [
        _hash("a"),
        _hash("b"),
    ]


def test_descriptor_commitments_exclude_same_plaintext_retry_envelope():
    committed_artifact = {
        "artifact_id": _hash("a"),
        "plaintext_hash": _hash("1"),
        "ciphertext_hash": _hash("c"),
        "encryption_context_hash": _hash("e"),
        "artifact_kind": "provider_response",
        "persisted": False,
    }
    retry_orphan = {
        **committed_artifact,
        "artifact_id": _hash("b"),
        "ciphertext_hash": _hash("d"),
    }
    committed_hashes = tuple(
        committed_artifact[field]
        for field in (
            "artifact_id",
            "plaintext_hash",
            "ciphertext_hash",
            "encryption_context_hash",
        )
    )

    plaintext_selected = (
        attested_artifacts_v2._select_committed_encrypted_artifacts(
            [committed_artifact, retry_orphan],
            committed_hashes=committed_hashes,
        )
    )
    descriptor_selected = (
        attested_artifacts_v2._select_committed_encrypted_artifacts(
            [committed_artifact, retry_orphan],
            committed_hashes=committed_hashes,
            require_descriptor_commitments=True,
        )
    )

    assert [item["artifact_id"] for item in plaintext_selected] == [
        _hash("a"),
        _hash("b"),
    ]
    assert [item["artifact_id"] for item in descriptor_selected] == [
        _hash("a")
    ]


async def _exercise(
    monkeypatch: pytest.MonkeyPatch,
    *,
    replay: bool,
    partial: bool = False,
    source_failed: bool = False,
    child_status: str = "succeeded",
    policy_events: list | None = None,
    duplicate_transport: bool = False,
    missing_distinct: bool = False,
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
        "status": "failed" if source_failed else "succeeded",
    }
    source_graph = {
        "root_receipt_hash": source_receipt["receipt_hash"],
        "receipts": [source_receipt],
    }
    transport_attempts = [
        {
            "request_artifact_hash": _hash("1"),
            "response_artifact_hash": _hash("2"),
            "terminal_status": "authenticated_response",
        }
    ]
    if duplicate_transport:
        transport_attempts.append(dict(transport_attempts[0]))
    if missing_distinct:
        committed.append(_hash("3"))
        source_receipt["artifact_root"] = merkle_root(
            committed,
            domain="leadpoet-artifact-v2",
        )
        transport_attempts.append(
            {
                "request_artifact_hash": _hash("3"),
                "response_artifact_hash": _hash("2"),
                "terminal_status": "authenticated_response",
            }
        )
    persistence_job_ids = []
    durability_events = []
    source_proof = {"proof_hash": _hash("4")}
    final_proof = {"proof_hash": _hash("5")}

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

    async def execute(**kwargs):
        assert durability_events[:2] == [
            ("graph", source_receipt["receipt_hash"]),
            ("checkpoint", source_receipt["receipt_hash"]),
        ]
        durability_events.append(("child", None))
        assert kwargs["parent_ancestry_proofs"] == (source_proof,)
        assert set(kwargs["allowed_failed_parent_receipt_hashes"]) == (
            {source_receipt["receipt_hash"]} if source_failed else set()
        )
        job_id = persistence_job_ids[0] if persistence_job_ids else expected_job_id[0]
        receipt = {
            "job_id": job_id,
            "receipt_hash": _hash("8"),
            "status": child_status,
        }
        return {
            "status": "succeeded",
            "result": {"artifacts": _lineage_evidence(artifacts)},
            "receipt": receipt,
            "receipt_graph": {
                "root_receipt_hash": receipt["receipt_hash"],
                "receipts": [receipt],
            },
            "ancestry_compact_proof": final_proof,
        }

    async def persist_sidecars(**kwargs):
        return {"artifact_link_count": len(kwargs["artifacts"])}

    async def persist_checkpoint(proof, *, checkpointed_graph, **_kwargs):
        durability_events.append(
            ("checkpoint", checkpointed_graph["root_receipt_hash"])
        )
        return {
            "root_receipt_hash": checkpointed_graph["root_receipt_hash"],
            "proof_hash": proof["proof_hash"],
        }

    async def persist_graph(graph, **_kwargs):
        durability_events.append(("graph", graph["root_receipt_hash"]))
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    expected_job_id = []
    original_derive = __import__(
        "gateway.research_lab.attested_scoring_v2",
        fromlist=["derive_execution_job_id_v2"],
    ).derive_execution_job_id_v2

    def capture_job_id(**kwargs):
        value = original_derive(**kwargs)
        expected_job_id.append(value)
        return value

    def validate_graph(graph, **kwargs):
        if policy_events is not None:
            policy_events.append(
                (
                    graph["root_receipt_hash"],
                    set(kwargs.get("allowed_failed_receipt_hashes") or ()),
                )
            )

    monkeypatch.setattr(
        attested_artifacts_v2,
        "validate_receipt_graph",
        validate_graph,
    )
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
        attested_scoring_v2,
        "_gateway_ancestry_lineage_id",
        lambda: _hash("6"),
    )
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
        source_ancestry_compact_proof=source_proof,
        persist_graph=persist_graph,
        persist_ancestry_checkpoint=persist_checkpoint,
        boot_verifier=lambda identity: identity,
    )
    assert result["receipt"]["job_id"] == expected_job_id[0]
    assert durability_events.index(("child", None)) > durability_events.index(
        ("checkpoint", source_receipt["receipt_hash"])
    )
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


@pytest.mark.asyncio
async def test_transport_artifacts_deduplicate_repeated_plaintext_commitments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _exercise(
        monkeypatch,
        replay=False,
        duplicate_transport=True,
    )
    assert len(result["artifacts"]) == 2


@pytest.mark.asyncio
async def test_transport_artifacts_reject_missing_distinct_plaintext_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        attested_artifacts_v2.AttestedArtifactPersistenceV2Error,
        match="coordinator artifacts differ from execution commitments",
    ):
        await _exercise(
            monkeypatch,
            replay=False,
            missing_distinct=True,
        )


@pytest.mark.asyncio
async def test_failed_source_policy_does_not_leak_into_successful_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_events = []
    await _exercise(
        monkeypatch,
        replay=True,
        source_failed=True,
        policy_events=policy_events,
    )

    assert policy_events == [
        (_hash("9"), {_hash("9")}),
        (_hash("8"), set()),
    ]


@pytest.mark.asyncio
async def test_failed_source_rejects_tampered_failed_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        attested_artifacts_v2.AttestedArtifactPersistenceV2Error,
        match="lineage did not succeed",
    ):
        await _exercise(
            monkeypatch,
            replay=True,
            source_failed=True,
            child_status="failed",
        )
