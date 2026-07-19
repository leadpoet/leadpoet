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


@pytest.mark.asyncio
async def test_historical_settlement_authority_is_scope_bound_and_durable(
    monkeypatch,
):
    from leadpoet_canonical import legacy_settlement_v2

    document = {
        "netuid": 71,
        "epoch_id": 100,
        "settlement_hash": HASH_B,
    }
    captured = {}
    links = []
    migrations = []

    async def execute(**kwargs):
        captured.update(kwargs)
        receipt = {"receipt_hash": HASH_A, "output_root": v2_authority.sha256_json(document)}
        return {
            "status": "succeeded",
            "result": document,
            "execution_receipt": receipt,
            "receipt_graph": {
                "root_receipt_hash": HASH_A,
                "receipts": [receipt],
            },
        }

    async def persist_links(**kwargs):
        links.append(kwargs)
        return {"business_artifact_link_count": 1}

    async def persist_migration(**kwargs):
        migrations.append(kwargs)
        return {"settlement_hash": HASH_B}

    monkeypatch.setattr(v2_authority, "legacy_v1_enabled", lambda: True)
    monkeypatch.setattr(
        legacy_settlement_v2,
        "validate_legacy_settlement_document_v2",
        lambda value: dict(value),
    )
    result = await v2_authority.attest_historical_champion_settlement_v2(
        epoch_id=101,
        netuid=71,
        settlement_epoch_id=100,
        execute=execute,
        persist_links=persist_links,
        persist_migration=persist_migration,
    )
    assert result["status"] == "matched"
    assert captured["sequence"] == 100
    assert captured["payload"] == {
        "schema_version": "leadpoet.legacy_finalized_allocation_request.v2",
        "netuid": 71,
        "epoch_id": 100,
    }
    assert links[0]["artifacts"][0] == {
        "artifact_kind": "legacy_finalized_allocation",
        "artifact_ref": "71:100",
        "artifact_hash": HASH_B,
    }
    assert migrations == [{"settlement": document, "receipt_hash": HASH_A}]


@pytest.mark.asyncio
async def test_historical_nonfinalization_is_persisted_without_settlement(
    monkeypatch,
):
    from leadpoet_canonical import legacy_settlement_v2

    document = {
        "schema_version": "leadpoet.legacy_allocation_nonfinalization.v2",
        "netuid": 71,
        "epoch_id": 100,
        "finding_hash": HASH_B,
    }
    links = []
    findings = []
    settlements = []

    async def execute(**_kwargs):
        receipt = {
            "receipt_hash": HASH_A,
            "output_root": v2_authority.sha256_json(document),
        }
        return {
            "status": "succeeded",
            "result": document,
            "execution_receipt": receipt,
            "receipt_graph": {
                "root_receipt_hash": HASH_A,
                "receipts": [receipt],
            },
        }

    async def persist_links(**kwargs):
        links.append(kwargs)
        return {"business_artifact_link_count": 1}

    async def persist_finding(**kwargs):
        findings.append(kwargs)
        return {"finding_hash": HASH_B}

    async def persist_settlement(**kwargs):
        settlements.append(kwargs)

    monkeypatch.setattr(
        legacy_settlement_v2,
        "validate_legacy_nonfinalization_document_v2",
        lambda value: dict(value),
    )
    result = await v2_authority.classify_historical_champion_allocation_v2(
        epoch_id=101,
        netuid=71,
        settlement_epoch_id=100,
        execute=execute,
        persist_links=persist_links,
        persist_migration=persist_settlement,
        persist_nonfinalization=persist_finding,
    )

    assert result["status"] == "not_finalized"
    assert links[0]["artifacts"][0] == {
        "artifact_kind": "legacy_allocation_nonfinalization",
        "artifact_ref": "71:100",
        "artifact_hash": HASH_B,
    }
    assert findings == [{"finding": document, "receipt_hash": HASH_A}]
    assert settlements == []


@pytest.mark.asyncio
async def test_historical_champion_reward_migration_runs_before_v2_cutover(
    monkeypatch,
):
    reward_id = "champion_reward:sha256:" + "c" * 64
    result = {"decision_kind": "champion", "reward": {"id": reward_id}}
    captured = {}

    async def execute(**kwargs):
        captured.update(kwargs)
        projection = {"champion_reward_id": reward_id}
        receipt = {
            "receipt_hash": HASH_A,
            "output_root": v2_authority.sha256_json(projection),
        }
        return {
            "status": "succeeded",
            "result": result,
            "execution_receipt": receipt,
            "receipt_graph": {
                "root_receipt_hash": HASH_A,
                "receipts": [receipt],
            },
        }

    async def persist_links(**_kwargs):
        return {"business_artifact_link_count": 1}

    monkeypatch.setattr(v2_authority, "legacy_v1_enabled", lambda: True)
    monkeypatch.setattr(
        v2_authority,
        "reward_receipt_projection_v2",
        lambda _result: {"champion_reward_id": reward_id},
    )
    outcome = await v2_authority.attest_historical_champion_reward_v2(
        epoch_id=101,
        champion_reward_id=reward_id,
        execute=execute,
        persist_links=persist_links,
    )

    assert outcome["status"] == "matched"
    assert captured["payload"] == {
        "decision_kind": "champion_migration",
        "decision_payload": {"champion_reward_id": reward_id},
    }
    assert captured["sequence"] == 1


@pytest.mark.asyncio
async def test_allocation_parent_loader_uses_legacy_settlement_receipt(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, champion_settlement_v2, store

    reward_id = "champion_reward:sha256:" + "c" * 64
    reward_receipt = "sha256:" + "d" * 64
    settlement_receipt = "sha256:" + "e" * 64
    champion_row = {
        "champion_reward_id": reward_id,
        "current_reward_status": "active",
        "start_epoch": 99,
        "epoch_count": 20,
        "desired_alpha_percent": 1.0,
    }
    business_refs = []
    receipt_roots = []

    async def select_all(table, **kwargs):
        if table == "research_lab_champion_reward_current":
            filters = dict(
                (item[0], item[1])
                for item in kwargs.get("filters") or ()
                if len(item) == 2
            )
            return [champion_row] if filters.get("current_reward_status") == "active" else []
        return []

    async def select_one(*_args, **_kwargs):
        return None

    async def load_history(**_kwargs):
        return [
            {
                "epoch": 99,
                "netuid": 71,
                "allocation_hash": HASH_A,
                "allocation_doc": {
                    "allocation_hash": HASH_A,
                    "champion_allocations": [],
                    "queued_champion_allocations": [],
                },
                "allocation_receipt_hash": settlement_receipt,
                "legacy_settlement_receipt_hash": settlement_receipt,
                "authority_types": ["legacy_finalized_chain_migration_v2"],
                "finalized_bundle_hashes": [],
                "finalization_receipt_hashes": [],
            }
        ]

    async def load_business(artifacts):
        requested = sorted(artifacts)
        business_refs.extend(requested)
        return {
            key: {
                "root_receipt_hash": reward_receipt,
                "receipts": [{"receipt_hash": reward_receipt}],
            }
            for key in requested
        }

    async def load_receipts(receipt_hashes):
        requested = sorted(receipt_hashes)
        receipt_roots.extend(requested)
        return {
            receipt_hash: {
                "root_receipt_hash": receipt_hash,
                "receipts": [{"receipt_hash": receipt_hash}],
            }
            for receipt_hash in requested
        }

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(store, "select_one", select_one)
    monkeypatch.setattr(
        champion_settlement_v2,
        "load_finalized_allocation_history_v2",
        load_history,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_business,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graphs_v2",
        load_receipts,
    )

    graphs = await v2_authority._load_allocation_parent_graphs_v2(
        epoch_id=100,
        netuid=71,
        policy={},
    )

    assert business_refs == [("champion_reward_decision", reward_id)]
    assert receipt_roots == [settlement_receipt]
    assert {graph["root_receipt_hash"] for graph in graphs} == {
        reward_receipt,
        settlement_receipt,
    }


@pytest.mark.asyncio
async def test_allocation_parent_loader_skips_fully_paid_legacy_source_receipt(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, champion_settlement_v2, store

    reward_ref = "source_add_reward:test"
    settlement_receipt = "sha256:" + "e" * 64
    source_row = {
        "reward_ref": reward_ref,
        "current_reward_status": "active",
        "start_epoch": 99,
        "epoch_count": 1,
        "desired_alpha_percent": 1.0,
    }
    business_refs = []
    receipt_roots = []

    async def select_all(table, **kwargs):
        if table == "research_lab_source_add_reward_current":
            filters = dict(
                (item[0], item[1])
                for item in kwargs.get("filters") or ()
                if len(item) == 2
            )
            return [source_row] if filters.get("current_reward_status") == "active" else []
        return []

    async def load_history(**_kwargs):
        allocation_payload = {
            "epoch": 99,
            "source_add_allocations": [
                {
                    "source_add_reward_id": reward_ref,
                    "paid_alpha_percent": 1.0,
                }
            ],
        }
        allocation = {
            **allocation_payload,
            "allocation_hash": v2_authority.sha256_json(allocation_payload),
        }
        return [
            {
                "epoch": 99,
                "netuid": 71,
                "allocation_hash": allocation["allocation_hash"],
                "allocation_doc": allocation,
                "allocation_receipt_hash": settlement_receipt,
                "legacy_settlement_receipt_hash": settlement_receipt,
                "authority_types": ["legacy_finalized_chain_migration_v2"],
                "finalized_bundle_hashes": [],
                "finalization_receipt_hashes": [],
            }
        ]

    async def load_business(artifacts):
        business_refs.extend(sorted(artifacts))
        return {}

    async def load_receipts(receipt_hashes):
        requested = sorted(receipt_hashes)
        receipt_roots.extend(requested)
        return {
            receipt_hash: {
                "root_receipt_hash": receipt_hash,
                "receipts": [{"receipt_hash": receipt_hash}],
            }
            for receipt_hash in requested
        }

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        champion_settlement_v2,
        "load_finalized_allocation_history_v2",
        load_history,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_business,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graphs_v2",
        load_receipts,
    )

    graphs = await v2_authority._load_allocation_parent_graphs_v2(
        epoch_id=100,
        netuid=71,
        policy={},
    )

    assert business_refs == []
    assert receipt_roots == [settlement_receipt]
    assert [graph["root_receipt_hash"] for graph in graphs] == [
        settlement_receipt
    ]


@pytest.mark.asyncio
async def test_allocation_parent_loader_batches_reimbursement_awards(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store

    schedules = [
        {
            "award_id": f"award:{index}",
            "schedule_status": "scheduled",
            "start_epoch": 90,
            "epoch_count": 20,
        }
        for index in range(125)
    ]
    awards = {
        f"award:{index}": {
            "award_id": f"award:{index}",
            "current_award_status": "awarded",
        }
        for index in range(125)
    }
    batch_sizes = []
    business_refs = []

    async def select_all(table, *, filters=(), **_kwargs):
        if table == "research_reimbursement_schedules":
            return list(schedules)
        if table == "research_reimbursement_award_current":
            requested = list(filters[0][2])
            batch_sizes.append(len(requested))
            return [dict(awards[award_id]) for award_id in requested]
        if table in {
            "research_lab_champion_reward_current",
            "research_lab_source_add_reward_current",
        }:
            return []
        raise AssertionError(f"unexpected table: {table}")

    async def load_business(artifacts):
        requested = sorted(artifacts)
        business_refs.extend(requested)
        return {
            key: {
                "root_receipt_hash": HASH_A,
                "receipts": [{"receipt_hash": HASH_A}],
            }
            for key in requested
        }

    async def load_receipts(receipt_hashes):
        assert not list(receipt_hashes)
        return {}

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_business,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graphs_v2",
        load_receipts,
    )

    graphs = await v2_authority._load_allocation_parent_graphs_v2(
        epoch_id=100,
        netuid=71,
        policy={"reimbursement_epochs": 20},
    )

    assert batch_sizes == [50, 50, 25]
    assert business_refs == [
        ("reimbursement_decision", f"award:{index}")
        for index in sorted(range(125), key=lambda value: f"award:{value}")
    ]
    assert [graph["root_receipt_hash"] for graph in graphs] == [HASH_A]


@pytest.mark.asyncio
async def test_allocation_parent_loader_rejects_ambiguous_reimbursement_award(
    monkeypatch,
):
    from gateway.research_lab import store

    award = {
        "award_id": "award:1",
        "current_award_status": "awarded",
    }

    async def select_all(table, **_kwargs):
        if table == "research_reimbursement_schedules":
            return [
                {
                    "award_id": "award:1",
                    "schedule_status": "scheduled",
                    "start_epoch": 90,
                    "epoch_count": 20,
                }
            ]
        if table == "research_reimbursement_award_current":
            return [dict(award), dict(award)]
        return []

    monkeypatch.setattr(store, "select_all", select_all)

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="reimbursement award is ambiguous",
    ):
        await v2_authority._load_allocation_parent_graphs_v2(
            epoch_id=100,
            netuid=71,
            policy={"reimbursement_epochs": 20},
        )
