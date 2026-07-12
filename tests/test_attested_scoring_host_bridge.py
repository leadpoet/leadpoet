from __future__ import annotations

import pytest

from gateway.research_lab import attested_scoring, attested_v2_store, v2_authority


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def test_v2_authority_cannot_be_disabled_by_legacy_environment(monkeypatch):
    for value in ("off", "shadow", "required", "invalid"):
        monkeypatch.setenv("RESEARCH_LAB_ATTESTED_SCORING_MODE", value)
        assert attested_scoring.attested_scoring_mode() == "required"
        assert attested_scoring.attested_receipt_persistence_enabled() is True
        assert attested_scoring.attested_live_provider_enabled() is True


@pytest.mark.asyncio
async def test_legacy_rpc_surface_fails_closed():
    with pytest.raises(attested_scoring.AttestedScoringError, match="legacy"):
        await attested_scoring.execute_attested_scoring_operation()
    with pytest.raises(attested_scoring.AttestedScoringError, match="not an authority"):
        await attested_scoring.compare_qualification_company_scores()


@pytest.mark.asyncio
async def test_company_facade_routes_only_to_v2(monkeypatch):
    captured = {}

    async def execute(**kwargs):
        captured.update(kwargs)
        kwargs["attestation_out"].update({"receipt": {"receipt_hash": HASH_A}})
        return [{"final_score": 4.5}]

    monkeypatch.setattr(v2_authority, "execute_company_scores_v2", execute)
    sidecar = {}
    result = await attested_scoring.execute_required_qualification_company_scores(
        epoch_id=9,
        purpose="research_lab.candidate_score.v1",
        companies=[{"company_name": "Example"}],
        icp={"industry": "Software"},
        is_reference_model=False,
        attestation_out=sidecar,
    )
    assert result == [{"final_score": 4.5}]
    assert captured["epoch_id"] == 9
    assert sidecar["receipt"]["receipt_hash"] == HASH_A


@pytest.mark.asyncio
async def test_score_bundle_facade_preserves_direct_parent_set(monkeypatch):
    captured = {}

    async def compare(**kwargs):
        captured.update(kwargs)
        return {"status": "matched"}

    monkeypatch.setattr(v2_authority, "compare_score_bundle_v2", compare)
    result = await attested_scoring.compare_score_bundle(
        epoch_id=9,
        purpose="research_lab.candidate_score.v1",
        build_payload={"input": "same"},
        expected_score_bundle={"score_bundle_hash": HASH_A},
        parent_receipts=[{"receipt_hash": HASH_A}, {"receipt_hash": HASH_B}],
        direct_parent_receipt_hashes=[HASH_B],
    )
    assert result == {"status": "matched"}
    assert captured["parent_receipt_hashes"] == [HASH_B]


@pytest.mark.asyncio
async def test_v2_artifact_link_facade_uses_business_lineage_store(monkeypatch):
    captured = {}

    async def persist(**kwargs):
        captured.update(kwargs)
        return {"business_artifact_link_count": 1}

    monkeypatch.setattr(attested_v2_store, "persist_business_artifact_links_v2", persist)
    links = [
        {
            "artifact_kind": "score_bundle",
            "artifact_ref": "score_bundle:" + "a" * 64,
            "artifact_hash": HASH_A,
        }
    ]
    status = await attested_scoring.persist_attested_outcome_artifact_links(
        {
            "status": "matched",
            "receipt": {"receipt_hash": HASH_B},
        },
        artifact_links=links,
    )
    assert status == "persisted"
    assert captured == {"receipt_hash": HASH_B, "artifacts": links}


@pytest.mark.asyncio
async def test_v2_artifact_lineage_returns_root_and_complete_graph(monkeypatch):
    root = {"receipt_hash": HASH_A}
    ancestor = {"receipt_hash": HASH_B}

    async def load(**kwargs):
        assert kwargs["artifact_hash"] == HASH_A
        return {
            "root_receipt_hash": HASH_A,
            "receipts": [ancestor, root],
        }

    monkeypatch.setattr(attested_v2_store, "load_business_artifact_graph_v2", load)
    resolved, receipts = await attested_scoring.resolve_attested_artifact_lineage(
        artifact_kind="score_bundle",
        artifact_ref="score_bundle:" + "a" * 64,
        artifact_hash=HASH_A,
    )
    assert resolved == root
    assert receipts == [ancestor, root]


def test_scoring_worker_partition_routes_exact_13_12_shards():
    assert [
        attested_scoring.scoring_enclave_shard_for_worker(index)
        for index in range(25)
    ] == [0] * 13 + [1] * 12
    with pytest.raises(attested_scoring.AttestedScoringError):
        attested_scoring.scoring_enclave_shard_for_worker(25)
