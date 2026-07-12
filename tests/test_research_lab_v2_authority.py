from __future__ import annotations

import pytest

from gateway.research_lab import v2_authority


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _outcome(result):
    receipt = {"receipt_hash": HASH_A}
    return {
        "status": "succeeded",
        "result": result,
        "receipt": receipt,
        "receipt_graph": {"root_receipt_hash": HASH_A, "receipts": [receipt]},
    }


def test_malformed_allocation_schedule_fails_closed():
    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="schedule epoch fields",
    ):
        v2_authority._allocation_epoch_active(
            {"start_epoch": "invalid", "epoch_count": 20},
            100,
        )


@pytest.mark.asyncio
async def test_provider_preflight_uses_unique_measured_jobs_and_benchmark_profile(
    monkeypatch,
):
    calls = []

    async def execute(**kwargs):
        calls.append(kwargs)
        return _outcome(
            {"healthy": True, "pause_worthy": False, "verdicts": []}
        )

    monkeypatch.setattr(v2_authority.time, "time_ns", lambda: 123456789)
    result = await v2_authority.execute_provider_preflight_v2(
        scope_key="scoring:worker-4",
        worker_index=4,
        settings={
            "enabled": True,
            "ttl_seconds": 600.0,
            "timeout_seconds": 12.0,
            "failure_streak_threshold": 3,
        },
        execute=execute,
    )
    assert result["healthy"] is True
    assert calls[0]["purpose"] == "research_lab.provider_preflight.v2"
    assert calls[0]["sequence"] == 123456789
    assert calls[0]["worker_index"] == 4
    assert calls[0]["provider_credential_profile"] == "benchmark_model"


@pytest.mark.asyncio
async def test_company_scores_are_returned_only_from_v2_enclave(monkeypatch):
    captured = {}

    async def execute(**kwargs):
        captured.update(kwargs)
        return _outcome(
            {
                "breakdowns": [{"final_score": 7.5}],
                "scores": [7.5],
            }
        )

    monkeypatch.setenv("RESEARCH_LAB_SCORING_WORKER_INDEX", "24")
    result = await v2_authority.execute_company_scores_v2(
        epoch_id=10,
        purpose="research_lab.candidate_score.v1",
        companies=[{"company_name": "Example"}],
        icp={"industry": "Software"},
        is_reference_model=False,
        execute=execute,
    )

    assert result == [{"final_score": 7.5}]
    assert captured["purpose"] == "research_lab.company_score.v2"
    assert captured["payload"]["provider_execution_mode"] == "live_enclave"
    assert captured["payload"]["scoring_context_purpose"] == "research_lab.candidate_score.v2"
    assert captured["worker_index"] == 24


@pytest.mark.asyncio
async def test_company_score_projection_mismatch_fails_closed():
    async def execute(**_kwargs):
        return _outcome(
            {
                "breakdowns": [{"final_score": 7.5}],
                "scores": [8.0],
            }
        )

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="projection",
    ):
        await v2_authority.execute_company_scores_v2(
            epoch_id=10,
            purpose="research_lab.candidate_score.v1",
            companies=[],
            icp={},
            is_reference_model=False,
            execute=execute,
        )


@pytest.mark.asyncio
async def test_score_bundle_requires_exact_enclave_bytes_and_persists_lineage():
    bundle = {"score_bundle_hash": HASH_B, "aggregates": {"score": 1.0}}
    links = []

    async def execute(**kwargs):
        assert kwargs["purpose"] == "research_lab.candidate_score.v2"
        assert kwargs["parent_graphs"] == []
        return _outcome({"score_bundle": bundle})

    async def persist_links(**kwargs):
        links.append(kwargs)
        return {"business_artifact_link_count": 1}

    outcome = await v2_authority.compare_score_bundle_v2(
        epoch_id=10,
        purpose="research_lab.candidate_score.v1",
        build_payload={"payload": "unchanged"},
        expected_score_bundle=bundle,
        parent_receipt_hashes=(),
        execute=execute,
        persist_links=persist_links,
    )

    assert outcome["status"] == "matched"
    assert links[0]["receipt_hash"] == HASH_A
    assert links[0]["artifacts"] == (
        {
            "artifact_kind": "score_bundle",
            "artifact_ref": "score_bundle:" + "b" * 64,
            "artifact_hash": HASH_B,
        },
    )


@pytest.mark.asyncio
async def test_score_bundle_mismatch_blocks_business_link():
    expected = {"score_bundle_hash": HASH_B}
    called = False

    async def execute(**_kwargs):
        return _outcome({"score_bundle": {"score_bundle_hash": HASH_A}})

    async def persist_links(**_kwargs):
        nonlocal called
        called = True

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="score bundle",
    ):
        await v2_authority.compare_score_bundle_v2(
            epoch_id=10,
            purpose="research_lab.candidate_score.v1",
            build_payload={},
            expected_score_bundle=expected,
            parent_receipt_hashes=(),
            execute=execute,
            persist_links=persist_links,
        )
    assert called is False


@pytest.mark.asyncio
async def test_promotion_decision_requires_metric_graph(monkeypatch):
    monkeypatch.setattr(v2_authority, "validate_receipt_graph", lambda *args, **kwargs: None)

    async def execute(**kwargs):
        assert kwargs["parent_graphs"][0]["root_receipt_hash"] == HASH_A
        return _outcome({"decision": {"status": "promotion_passed"}})

    async def persist_links(**_kwargs):
        return {"business_artifact_link_count": 1}

    outcome = await v2_authority.compare_promotion_decision_v2(
        epoch_id=10,
        score_bundle={"score_bundle_hash": HASH_B},
        decision_payload={
            "candidate_kind": "image_build",
            "candidate_parent": HASH_A,
            "active_parent": HASH_A,
            "threshold_points": 1.0,
            "auto_promotion_enabled": True,
        },
        expected_decision={"status": "promotion_passed"},
        metric_outcome={"receipt_graph": {"root_receipt_hash": HASH_A}},
        execute=execute,
        persist_links=persist_links,
    )
    assert outcome["status"] == "matched"


@pytest.mark.asyncio
async def test_allocation_binds_every_reward_parent(monkeypatch):
    monkeypatch.setattr(v2_authority, "validate_receipt_graph", lambda *args, **kwargs: None)
    expected = {"allocation_hash": HASH_A}
    allocation_inputs = {
        "epoch": 10,
        "policy": {},
        "active_reimbursement_obligations": [],
        "active_champion_obligations": [],
    }
    source_state = {
        "epoch": 10,
        "policy": {},
        "reimbursement_obligations": [],
        "champion_obligations": [],
    }

    async def load_parent_graphs(**kwargs):
        assert kwargs == {"epoch_id": 10, "netuid": 71, "policy": {}}
        return [
            {
                "root_receipt_hash": HASH_B,
                "receipts": [
                    {
                        "receipt_hash": HASH_B,
                        "purpose": "research_lab.reward_decision.v2",
                        "role": "gateway_coordinator",
                    }
                ],
            }
        ]

    async def execute(**kwargs):
        assert kwargs["parent_graphs"][0]["root_receipt_hash"] == HASH_B
        assert kwargs["payload"] == {"epoch": 10, "netuid": 71}
        return _outcome(
            {
                "allocation": expected,
                "allocation_inputs": allocation_inputs,
                "source_state": source_state,
                "source_state_hash": v2_authority.sha256_json(source_state),
            }
        )

    async def persist_links(**_kwargs):
        return {"business_artifact_link_count": 1}

    outcome = await v2_authority.compare_allocation_v2(
        epoch_id=10,
        netuid=71,
        payload=allocation_inputs,
        expected_allocation=expected,
        execute=execute,
        persist_links=persist_links,
        load_allocation_parent_graphs=load_parent_graphs,
    )
    assert outcome["lineage_complete"] is True
