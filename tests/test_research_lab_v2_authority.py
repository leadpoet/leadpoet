from __future__ import annotations

import threading

import pytest

from gateway.research_lab import v2_authority
from gateway.research_lab.attested_scoring_v2 import (
    derive_execution_job_id_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _frontier(epoch=10):
    return build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=epoch,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )


def _outcome(result):
    receipt = {"receipt_hash": HASH_A}
    return {
        "status": "succeeded",
        "result": result,
        "receipt": receipt,
        "receipt_graph": {"root_receipt_hash": HASH_A, "receipts": [receipt]},
    }


@pytest.mark.asyncio
async def test_business_links_bind_execution_graph_before_artifact_wrapper():
    execution_receipt = {"receipt_hash": HASH_A}
    artifact_receipt = {"receipt_hash": HASH_B}
    observed = {}

    async def persist_links(**kwargs):
        observed.update(kwargs)
        return {"business_artifact_link_count": 1}

    result = await v2_authority._persist_business_links(
        {
            "execution_receipt": execution_receipt,
            "execution_receipt_graph": {
                "root_receipt_hash": HASH_A,
                "receipts": [execution_receipt],
            },
            "receipt": artifact_receipt,
            "receipt_graph": {
                "root_receipt_hash": HASH_B,
                "receipts": [artifact_receipt],
            },
        },
        ({"artifact_kind": "allocation"},),
        persist_links=persist_links,
    )

    assert result == {"business_artifact_link_count": 1}
    assert observed == {
        "receipt_hash": HASH_A,
        "artifacts": ({"artifact_kind": "allocation"},),
    }


@pytest.mark.asyncio
async def test_business_links_reject_execution_graph_with_different_root():
    execution_receipt = {"receipt_hash": HASH_A}

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="execution receipt is absent",
    ):
        await v2_authority._persist_business_links(
            {
                "execution_receipt": execution_receipt,
                "execution_receipt_graph": {
                    "root_receipt_hash": HASH_B,
                    "receipts": [execution_receipt],
                },
            },
            (),
            persist_links=lambda **_kwargs: None,
        )


@pytest.mark.asyncio
async def test_catalog_snapshot_validates_measured_execution_before_artifact_wrapper(
    monkeypatch,
):
    runtime_catalog = v2_authority.build_source_add_runtime_catalog_v2([])
    result = {
        "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
        "provisioned_sources": [],
        "private_registry_rows": [],
        "runtime_catalog": runtime_catalog,
        "provisioned_sources_hash": v2_authority.sha256_json([]),
        "private_registry_rows_hash": v2_authority.sha256_json([]),
        "runtime_catalog_hash": runtime_catalog["catalog_hash"],
    }
    execution_receipt = {
        "receipt_hash": HASH_A,
        "output_root": v2_authority.sha256_json(result),
    }
    execution_graph = {
        "root_receipt_hash": HASH_A,
        "receipts": [execution_receipt],
    }
    artifact_receipt = {
        "receipt_hash": HASH_B,
        "output_root": HASH_B,
    }
    artifact_graph = {
        "root_receipt_hash": HASH_B,
        "receipts": [artifact_receipt],
    }
    validated = []

    async def execute(**_kwargs):
        return {
            "status": "succeeded",
            "result": result,
            "receipt": artifact_receipt,
            "receipt_graph": artifact_graph,
            "execution_receipt": execution_receipt,
            "execution_receipt_graph": execution_graph,
        }

    def validate(graph, **kwargs):
        validated.append((graph, kwargs))

    monkeypatch.setattr(v2_authority, "validate_receipt_graph", validate)

    outcome = await v2_authority.load_source_add_catalog_snapshot_v2(
        epoch_id=42,
        execute=execute,
    )

    assert outcome["receipt"] == artifact_receipt
    assert validated == [
        (
            execution_graph,
            {
                "required_purposes": {
                    "research_lab.source_add_catalog_snapshot.v2"
                }
            },
        )
    ]


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
async def test_provider_preflight_uses_unique_measured_jobs_and_dedicated_profile(
    monkeypatch,
):
    calls = []
    job_ids = []
    measurement_ids = iter(("1" * 32, "2" * 32))

    async def execute(**kwargs):
        calls.append(kwargs)
        job_ids.append(
            derive_execution_job_id_v2(
                operation=kwargs["operation"],
                purpose=kwargs["purpose"],
                epoch_id=kwargs["epoch_id"],
                sequence=kwargs["sequence"],
                payload_sha256=v2_authority.sha256_json(kwargs["payload"]),
                parent_receipt_hashes=(),
                input_artifact_hashes=(),
                release_hash=HASH_B,
                physical_role="gateway_scoring",
            )
        )
        return _outcome(
            {"healthy": True, "pause_worthy": False, "verdicts": []}
        )

    monkeypatch.setattr(
        v2_authority.uuid,
        "uuid4",
        lambda: type("MeasurementId", (), {"hex": next(measurement_ids)})(),
    )
    for _ in range(2):
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
    assert [call["sequence"] for call in calls] == [0, 0]
    assert [call["payload"]["measurement_id"] for call in calls] == [
        "1" * 32,
        "2" * 32,
    ]
    assert len(set(job_ids)) == 2
    assert all(call["worker_index"] == 4 for call in calls)
    assert all(
        call["provider_credential_profile"] == "provider_preflight"
        for call in calls
    )


@pytest.mark.asyncio
async def test_allocation_binds_every_reward_parent(monkeypatch):
    loop_thread = threading.get_ident()
    validation_threads = []
    fallback_obligations = [
        {
            "source_id": "historical_compute_fallback:" + "c" * 64,
            "uid": 2,
            "miner_hotkey": "compute-hotkey",
        }
    ]

    def validate_receipt_graphs(*_args, **_kwargs):
        validation_threads.append(threading.get_ident())

    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graphs",
        validate_receipt_graphs,
    )
    expected = {"allocation_hash": HASH_A}
    allocation_inputs = {
        "epoch": 10,
        "policy": {},
        "active_reimbursement_obligations": [],
        "active_champion_obligations": [],
        "fallback_reimbursement_obligations": fallback_obligations,
    }
    source_state = {
        "epoch": 10,
        "netuid": 71,
        "policy": {},
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "fallback_reimbursement_obligations": fallback_obligations,
        "settlement_frontier": _frontier(),
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
    assert validation_threads
    assert set(validation_threads) == {validation_threads[0]}
    assert validation_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_allocation_fallback_projection_mismatch_fails_closed(
    monkeypatch,
):
    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    allocation = {"allocation_hash": HASH_A}
    source_state = {
        "epoch": 10,
        "netuid": 71,
        "policy": {},
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "fallback_reimbursement_obligations": [
            {
                "source_id": "historical_compute_fallback:" + "c" * 64,
                "uid": 2,
            }
        ],
        "settlement_frontier": _frontier(),
    }

    async def load_parent_graphs(**_kwargs):
        return []

    async def execute(**_kwargs):
        return _outcome(
            {
                "allocation": allocation,
                "allocation_inputs": {
                    "epoch": 10,
                    "policy": {},
                    "active_reimbursement_obligations": [],
                    "active_champion_obligations": [],
                    "fallback_reimbursement_obligations": [
                        {
                            "source_id": (
                                "historical_compute_fallback:" + "d" * 64
                            ),
                            "uid": 3,
                        }
                    ],
                },
                "source_state": source_state,
                "source_state_hash": v2_authority.sha256_json(source_state),
            }
        )

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="allocation source projection",
    ):
        await v2_authority.compare_allocation_v2(
            epoch_id=10,
            netuid=71,
            payload={},
            expected_allocation=allocation,
            execute=execute,
            load_allocation_parent_graphs=load_parent_graphs,
        )


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
async def test_historical_source_add_migration_creates_reward_artifact_link(
    monkeypatch,
):
    reward_ref = "source_add_reward:" + "c" * 16
    result = {"decision_kind": "source_add_leg1", "reward": {"id": reward_ref}}
    captured = {}

    async def execute(**kwargs):
        captured.update(kwargs)
        projection = {"reward_ref": reward_ref}
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
        lambda _result: {"reward_ref": reward_ref},
    )
    outcome = await v2_authority.attest_historical_source_add_reward_v2(
        epoch_id=101,
        reward_ref=reward_ref,
        execute=execute,
        persist_links=persist_links,
    )

    assert outcome["status"] == "matched"
    assert captured["payload"] == {
        "decision_kind": "source_add_migration",
        "decision_payload": {"reward_ref": reward_ref},
    }
    assert captured["parent_graphs"] == ()


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
        "load_settled_allocation_history_v2",
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
async def test_allocation_parent_loader_prunes_expired_legacy_paid_history(
    monkeypatch,
):
    from gateway.research_lab import (
        attested_v2_store,
        champion_settlement_v2,
        store,
    )

    async def select_all(table, **kwargs):
        if table == "research_lab_champion_reward_current":
            filters = dict(
                (item[0], item[1])
                for item in kwargs.get("filters") or ()
                if len(item) == 2
            )
            if filters.get("current_reward_status") == "paid":
                return [
                    {
                        "champion_reward_id": (
                            "champion_reward:expired:%04d" % index
                        ),
                        "current_reward_status": "paid",
                        "start_epoch": 10,
                        "epoch_count": 20,
                        "desired_alpha_percent": 7.0,
                    }
                    for index in range(2500)
                ]
        return []

    async def reject_history(**_kwargs):
        raise AssertionError("expired paid reward loaded settlement history")

    async def empty_graphs(_values):
        return {}

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        champion_settlement_v2,
        "load_settled_allocation_history_v2",
        reject_history,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_v2",
        empty_graphs,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        empty_graphs,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graphs_v2",
        empty_graphs,
    )

    graphs = await v2_authority._load_allocation_parent_graphs_v2(
        epoch_id=100,
        netuid=71,
        policy={"enable_champ_cap": False},
    )

    assert graphs == []


@pytest.mark.asyncio
async def test_allocation_parent_loader_uses_finalized_allocation_hash(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store

    allocation_receipt = "sha256:" + "d" * 64
    finalization_receipt = "sha256:" + "e" * 64
    reward_id = "champion_reward:sha256:" + "f" * 64
    exact_requests = []
    by_ref_requests = []

    async def select_all(table, **kwargs):
        if table == "research_lab_champion_reward_current":
            filters = dict(
                (item[0], item[1])
                for item in kwargs.get("filters") or ()
                if len(item) == 2
            )
            if filters.get("current_reward_status") == "active":
                return [
                    {
                        "champion_reward_id": reward_id,
                        "current_reward_status": "active",
                        "start_epoch": 99,
                        "epoch_count": 20,
                        "desired_alpha_percent": 1.0,
                    }
                ]
        return []

    async def load_exact(artifacts):
        requested = sorted(artifacts)
        exact_requests.extend(requested)
        return {
            key: {
                "root_receipt_hash": allocation_receipt,
                "receipts": [{"receipt_hash": allocation_receipt}],
            }
            for key in requested
        }

    async def load_by_ref(artifacts):
        requested = sorted(artifacts)
        by_ref_requests.extend(requested)
        return {
            key: {
                "root_receipt_hash": HASH_B,
                "receipts": [{"receipt_hash": HASH_B}],
            }
            for key in requested
        }

    async def load_receipts(receipt_hashes):
        requested = sorted(receipt_hashes)
        return {
            receipt_hash: {
                "root_receipt_hash": receipt_hash,
                "receipts": [{"receipt_hash": receipt_hash}],
            }
            for receipt_hash in requested
        }

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_v2",
        load_exact,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_by_ref,
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
        finalized_champion_history=(
            {
                "epoch": 99,
                "netuid": 71,
                "allocation_hash": HASH_A,
                "allocation_doc": {
                    "allocation_hash": HASH_A,
                    "champion_allocations": [],
                    "queued_champion_allocations": [],
                },
                "authority_types": ["native_v2_finalization"],
                "finalization_receipt_hashes": [finalization_receipt],
            },
        ),
        preloaded_business_graphs={},
    )

    assert exact_requests == [("allocation", "epoch:99", HASH_A)]
    assert by_ref_requests == [("champion_reward_decision", reward_id)]
    assert {graph["root_receipt_hash"] for graph in graphs} == {
        allocation_receipt,
        finalization_receipt,
        HASH_B,
    }


@pytest.mark.asyncio
async def test_allocation_parent_loader_selects_exact_source_add_retry_hash(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store

    reward_ref = "source_add_reward:0123456789abcdef"
    decision_receipt = "sha256:" + "7" * 64
    source_row = {
        "reward_ref": reward_ref,
        "adapter_id": "adapter:exact-source-add-retry",
        "miner_hotkey": "5ExactSourceAddRetry",
        "leg": 1,
        "reward_kind": "source_acceptance",
        "alpha_percent": 1.0,
        "reward_epochs": 20,
        "start_epoch": 99,
        "trigger_evidence_doc": {"final_acceptance_stage": "accepted"},
        "public_label": "Source acceptance reward",
        "current_reward_status": "active",
        "epoch_count": 20,
        "desired_alpha_percent": 1.0,
    }
    expected_hash = v2_authority.sha256_json(
        v2_authority.source_add_reward_row_projection_v2(
            "source_add_leg1",
            {**source_row, "initial_reward_status": "active"},
        )
    )
    exact_requests = []
    by_ref_requests = []

    async def select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_source_add_reward_current":
            status = next(
                (value for field, value in filters if field == "current_reward_status"),
                "",
            )
            return [source_row] if status == "active" else []
        return []

    async def load_exact(artifacts):
        requested = sorted(artifacts)
        exact_requests.extend(requested)
        return {
            key: {
                "root_receipt_hash": decision_receipt,
                "receipts": [{"receipt_hash": decision_receipt}],
            }
            for key in requested
        }

    async def load_by_ref(artifacts):
        by_ref_requests.extend(sorted(artifacts))
        return {}

    async def load_receipts(receipt_hashes):
        assert not list(receipt_hashes)
        return {}

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_v2",
        load_exact,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_by_ref,
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
        finalized_champion_history=(),
    )

    assert exact_requests == [
        ("source_add_reward_decision", reward_ref, expected_hash)
    ]
    assert by_ref_requests == []
    assert [graph["root_receipt_hash"] for graph in graphs] == [
        decision_receipt
    ]


@pytest.mark.asyncio
async def test_allocation_parent_loader_reuses_raw_authority_graphs(
    monkeypatch,
):
    from gateway.research_lab import (
        attested_v2_store,
        champion_settlement_v2,
        store,
    )

    native_root = "sha256:" + "1" * 64
    unattributed_root = "sha256:" + "2" * 64
    out_of_scope_root = "sha256:" + "4" * 64
    loaded_receipt_roots = []

    async def select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_champion_reward_current":
            status = next(
                (
                    value
                    for field, value in filters
                    if field == "current_reward_status"
                ),
                "",
            )
            if status == "active":
                return [
                    {
                        "champion_reward_id": "champion:test",
                        "current_reward_status": "active",
                        "start_epoch": 99,
                        "epoch_count": 1,
                        "desired_alpha_percent": 0.0,
                    }
                ]
        return []

    business_requests = []

    async def load_empty_business(artifacts):
        business_requests.append(list(artifacts))
        return {}

    async def load_receipts(receipt_hashes):
        loaded_receipt_roots.extend(sorted(receipt_hashes))
        return {}

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_v2",
        load_empty_business,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_empty_business,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graphs_v2",
        load_receipts,
    )

    raw_graph_records = {
        root: {
            "epoch_id": 99,
            "graph": {
                "root_receipt_hash": root,
                "receipts": [{"receipt_hash": root}],
            },
        }
        for root in (native_root, unattributed_root)
    }
    raw_graph_records[out_of_scope_root] = {
        "epoch_id": 50,
        "graph": {
            "root_receipt_hash": out_of_scope_root,
            "receipts": [{"receipt_hash": out_of_scope_root}],
        },
    }
    graphs = await v2_authority._load_allocation_parent_graphs_v2(
        epoch_id=100,
        netuid=71,
        policy={},
        finalized_champion_history=(
            {
                "epoch": 99,
                "netuid": 71,
                "authority_types": [
                    "chain_realized_unattributed_observation_v1"
                ],
                "allocation_doc": {
                    "champion_allocations": [],
                    "queued_champion_allocations": [],
                },
            },
        ),
        preloaded_receipt_graph_records=raw_graph_records,
        preloaded_business_graphs={},
    )

    assert loaded_receipt_roots == []
    assert {graph["root_receipt_hash"] for graph in graphs} == {
        native_root,
        unattributed_root,
    }
    assert business_requests == [
        [],
        [("champion_reward_decision", "champion:test")],
    ]

    async def load_history(**kwargs):
        kwargs["_receipt_graph_records_out"].update(raw_graph_records)
        return [
            {
                "epoch": 99,
                "netuid": 71,
                "authority_types": [
                    "chain_realized_unattributed_observation_v1"
                ],
                "allocation_doc": {
                    "champion_allocations": [],
                    "queued_champion_allocations": [],
                },
            }
        ]

    monkeypatch.setattr(
        champion_settlement_v2,
        "load_settled_allocation_history_v2",
        load_history,
    )
    direct_graphs = await v2_authority._load_allocation_parent_graphs_v2(
        epoch_id=100,
        netuid=71,
        policy={},
    )
    assert {graph["root_receipt_hash"] for graph in direct_graphs} == {
        native_root,
        unattributed_root,
    }


@pytest.mark.asyncio
async def test_default_allocation_threads_readiness_authority_graphs(
    monkeypatch,
):
    from gateway.research_lab import champion_settlement_v2

    authority_root = "sha256:" + "3" * 64
    authority_graph = {
        "root_receipt_hash": authority_root,
        "receipts": [{"receipt_hash": authority_root}],
    }
    captured = {}
    settlement_calls = []

    async def ready(**kwargs):
        kwargs["_authority_graph_records_out"][authority_root] = {
            "epoch_id": 99,
            "graph": authority_graph,
        }
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def no_settlement_repair(**kwargs):
        settlement_calls.append(kwargs)
        return {}

    class ParentLoaderReached(RuntimeError):
        pass

    async def parent_loader(**kwargs):
        captured.update(kwargs)
        raise ParentLoaderReached

    monkeypatch.setattr(
        champion_settlement_v2,
        "champion_v2_cutover_readiness",
        ready,
    )
    monkeypatch.setattr(
        v2_authority,
        "ensure_chain_realized_settlements_v1",
        no_settlement_repair,
    )

    async def no_frontier(**_kwargs):
        return None

    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store."
        "load_allocation_settlement_frontier_context_v2",
        no_frontier,
    )
    monkeypatch.setattr(
        v2_authority,
        "_load_allocation_parent_graphs_v2",
        parent_loader,
    )

    with pytest.raises(ParentLoaderReached):
        await v2_authority.build_allocation_v2(
            epoch_id=100,
            netuid=71,
            policy={},
            allocation_sequence=4,
        )

    assert settlement_calls == [
        {
            "epoch_id": 100,
            "netuid": 71,
            "execute": v2_authority.execute_coordinator_v2,
            "settlement_attempt": 4,
        }
    ]
    assert captured["preloaded_receipt_graph_records"] == {
        authority_root: {
            "epoch_id": 99,
            "graph": authority_graph,
        }
    }


@pytest.mark.asyncio
async def test_default_allocation_uses_frontier_without_legacy_readiness(
    monkeypatch,
):
    from gateway.research_lab import champion_settlement_v2

    frontier = _frontier(epoch=98)
    frontier_receipt = "sha256:" + "7" * 64
    frontier_context = {
        "frontier": frontier,
        "row": {"source_receipt_hash": frontier_receipt},
        "source": {
            "receipt_graph": {
                "root_receipt_hash": frontier_receipt,
                "receipts": [{"receipt_hash": frontier_receipt}],
            }
        },
    }
    captured = {}

    async def load_frontier(**kwargs):
        assert kwargs == {"netuid": 71, "before_epoch": 101}
        return frontier_context

    async def readiness_must_not_run(**_kwargs):
        raise AssertionError("legacy full-history readiness was invoked")

    class ParentLoaderReached(RuntimeError):
        pass

    async def parent_loader(**kwargs):
        captured.update(kwargs)
        raise ParentLoaderReached

    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store."
        "load_allocation_settlement_frontier_context_v2",
        load_frontier,
    )
    monkeypatch.setattr(
        champion_settlement_v2,
        "champion_v2_cutover_readiness",
        readiness_must_not_run,
    )
    monkeypatch.setattr(
        v2_authority,
        "_load_allocation_parent_graphs_v2",
        parent_loader,
    )

    async def execute(**_kwargs):
        raise AssertionError("execution should not begin before parent loading")

    with pytest.raises(ParentLoaderReached):
        await v2_authority.build_allocation_v2(
            epoch_id=100,
            netuid=71,
            policy={},
            execute=execute,
        )

    assert captured["settlement_frontier_context"] == frontier_context
    assert captured["finalized_champion_history"] is None
    assert captured["preloaded_receipt_graph_records"] == {}
    assert captured["preloaded_business_graphs"] == {}


@pytest.mark.asyncio
async def test_default_allocation_recovers_exact_current_frontier(monkeypatch):
    frontier = _frontier(epoch=100)
    parent_root = "sha256:" + "6" * 64
    receipt_hash = "sha256:" + "7" * 64
    source_state = {
        "epoch": 100,
        "netuid": 71,
        "policy": {},
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "settlement_frontier": frontier,
    }
    allocation_payload = {"epoch": 100, "netuid": 71}
    allocation = {
        **allocation_payload,
        "allocation_hash": v2_authority.sha256_json(allocation_payload),
    }
    authority_result = {
        "allocation": allocation,
        "allocation_inputs": {
            "epoch": 100,
            "policy": {},
            "active_reimbursement_obligations": [],
            "active_champion_obligations": [],
        },
        "source_state": source_state,
        "source_state_hash": v2_authority.sha256_json(source_state),
    }
    source_receipt = {
        "receipt_hash": receipt_hash,
        "parent_receipt_hashes": [parent_root],
        "role": "gateway_coordinator",
        "purpose": "research_lab.allocation.v2",
        "status": "succeeded",
        "epoch_id": 100,
        "commit_sha": "1" * 40,
    }
    source_graph = {
        "root_receipt_hash": receipt_hash,
        "receipts": [source_receipt],
        "boot_identities": [],
        "transport_attempts": [],
        "host_operations": [],
    }
    context = {
        "frontier": frontier,
        "row": {"source_receipt_hash": receipt_hash},
        "source": {
            "row": {
                "receipt_hash": receipt_hash,
                "operation": v2_authority.OP_RESEARCH_LAB_ALLOCATION,
                "purpose": "research_lab.allocation.v2",
                "role": "gateway_coordinator",
                "epoch_id": 100,
                "release_hash": "sha256:" + "8" * 64,
            },
            "result": authority_result,
            "receipt": source_receipt,
            "receipt_graph": source_graph,
            "artifact_hashes": [],
        },
    }
    parent_graph = {"root_receipt_hash": parent_root, "receipts": []}
    async def load_frontier(**kwargs):
        assert kwargs == {"netuid": 71, "before_epoch": 101}
        return context

    async def load_graphs(roots, **_kwargs):
        assert roots == [parent_root]
        return [parent_graph]

    async def parent_loader(**_kwargs):
        raise AssertionError("current frontier recovery rebuilt allocation inputs")

    async def execute(**_kwargs):
        raise AssertionError("current frontier recovery re-executed allocation")

    async def persist_links(**_kwargs):
        return {"business_artifact_link_count": 1}

    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store."
        "load_allocation_settlement_frontier_context_v2",
        load_frontier,
    )
    monkeypatch.setattr(v2_authority, "_graphs_for_roots", load_graphs)
    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        v2_authority,
        "_validate_allocation_parent_graphs",
        lambda graphs: [
            {
                "receipt_hash": graphs[0]["root_receipt_hash"],
                "receipt_purpose": "research_lab.reward_decision.v2",
                "receipt_role": "gateway_coordinator",
            }
        ],
    )
    monkeypatch.setattr(
        v2_authority,
        "_load_allocation_parent_graphs_v2",
        parent_loader,
    )

    recovered = await v2_authority.build_allocation_v2(
        epoch_id=100,
        netuid=71,
        policy={},
        execute=execute,
        persist_links=persist_links,
    )

    assert recovered["status"] == "matched"
    assert recovered["result"] == authority_result
    assert recovered["receipt"] == source_receipt
    assert recovered["receipt_graph"] == source_graph
    assert recovered["replay_status"] == "durable_current_frontier"


@pytest.mark.asyncio
async def test_current_frontier_recovery_rejects_frontier_source_mismatch(
    monkeypatch,
):
    frontier = _frontier(epoch=100)
    source_receipt = {
        "receipt_hash": HASH_A,
        "parent_receipt_hashes": [],
        "role": "gateway_coordinator",
        "purpose": "research_lab.allocation.v2",
        "status": "succeeded",
        "epoch_id": 100,
    }
    source_state = {
        "epoch": 100,
        "netuid": 71,
        "policy": {},
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "settlement_frontier": frontier,
    }
    result = {
        "allocation": {
            "allocation_hash": v2_authority.sha256_json({"epoch": 100})
        },
        "allocation_inputs": {
            "epoch": 100,
            "policy": {},
            "active_reimbursement_obligations": [],
            "active_champion_obligations": [],
        },
        "source_state": source_state,
        "source_state_hash": v2_authority.sha256_json(source_state),
    }

    async def load_frontier(**_kwargs):
        return {
            "frontier": frontier,
            "row": {"source_receipt_hash": HASH_B},
            "source": {
                "row": {
                    "receipt_hash": HASH_A,
                    "operation": v2_authority.OP_RESEARCH_LAB_ALLOCATION,
                    "purpose": "research_lab.allocation.v2",
                    "role": "gateway_coordinator",
                    "epoch_id": 100,
                    "release_hash": "sha256:" + "8" * 64,
                },
                "result": result,
                "receipt": source_receipt,
                "receipt_graph": {
                    "root_receipt_hash": HASH_A,
                    "receipts": [source_receipt],
                    "boot_identities": [],
                    "transport_attempts": [],
                    "host_operations": [],
                },
                "artifact_hashes": [],
            },
        }

    async def execute(**_kwargs):
        raise AssertionError("mismatched frontier source was re-executed")

    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store."
        "load_allocation_settlement_frontier_context_v2",
        load_frontier,
    )

    async def load_graphs(_roots):
        return []

    monkeypatch.setattr(v2_authority, "_graphs_for_roots", load_graphs)
    monkeypatch.setattr(
        v2_authority,
        "_validate_allocation_parent_graphs",
        lambda _graphs: [],
    )
    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="source authority differs",
    ):
        await v2_authority.build_allocation_v2(
            epoch_id=100,
            netuid=71,
            policy={},
            execute=execute,
        )


@pytest.mark.asyncio
async def test_current_frontier_recovery_rejects_invalid_source_authority(
    monkeypatch,
):
    frontier = _frontier(epoch=100)
    source_receipt = {
        "receipt_hash": HASH_A,
        "parent_receipt_hashes": [],
        "role": "gateway_coordinator",
        "purpose": "research_lab.allocation.v2",
        "status": "succeeded",
        "epoch_id": 100,
    }
    source_state = {
        "epoch": 100,
        "netuid": 71,
        "policy": {},
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "settlement_frontier": frontier,
    }
    result = {
        "allocation": {
            "allocation_hash": v2_authority.sha256_json({"epoch": 100})
        },
        "allocation_inputs": {
            "epoch": 100,
            "policy": {},
            "active_reimbursement_obligations": [],
            "active_champion_obligations": [],
        },
        "source_state": source_state,
        "source_state_hash": v2_authority.sha256_json(source_state),
    }

    async def load_frontier(**_kwargs):
        return {
            "frontier": frontier,
            "row": {"source_receipt_hash": HASH_A},
            "source": {
                "row": {
                    "receipt_hash": HASH_A,
                    "operation": v2_authority.OP_RESEARCH_LAB_ALLOCATION,
                    "purpose": "research_lab.allocation.v2",
                    "role": "gateway_coordinator",
                    "epoch_id": 100,
                    "release_hash": "invalid",
                },
                "result": result,
                "receipt": source_receipt,
                "receipt_graph": {
                    "root_receipt_hash": HASH_A,
                    "receipts": [source_receipt],
                    "boot_identities": [],
                    "transport_attempts": [],
                    "host_operations": [],
                },
                "artifact_hashes": [],
            },
        }

    async def execute(**_kwargs):
        raise AssertionError("invalid current frontier source was re-executed")

    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store."
        "load_allocation_settlement_frontier_context_v2",
        load_frontier,
    )
    monkeypatch.setattr(
        v2_authority,
        "_validate_allocation_parent_graphs",
        lambda _graphs: [],
    )

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="source authority differs",
    ):
        await v2_authority.build_allocation_v2(
            epoch_id=100,
            netuid=71,
            policy={},
            execute=execute,
        )


@pytest.mark.asyncio
async def test_allocation_parent_loader_reads_only_frontier_delta(monkeypatch):
    from gateway.research_lab import (
        attested_v2_store,
        champion_settlement_v2,
        store,
    )

    frontier = _frontier(epoch=98)
    frontier_receipt = "sha256:" + "7" * 64
    delta_receipt = "sha256:" + "8" * 64
    history_calls = []
    receipt_requests = []

    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda _graph: None,
    )

    async def select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_champion_reward_current":
            status = next(
                (
                    value
                    for field, value in filters
                    if field == "current_reward_status"
                ),
                "",
            )
            if status == "active":
                return [
                    {
                        "champion_reward_id": "champion:active",
                        "current_reward_status": "active",
                        "start_epoch": 1,
                        "epoch_count": 200,
                        "desired_alpha_percent": 1.0,
                    }
                ]
        return []

    async def load_history(**kwargs):
        history_calls.append(
            {
                key: value
                for key, value in kwargs.items()
                if key != "_receipt_graph_records_out"
            }
        )
        kwargs["_receipt_graph_records_out"][delta_receipt] = {
            "epoch_id": 99,
            "graph": {
                "root_receipt_hash": delta_receipt,
                "receipts": [{"receipt_hash": delta_receipt}],
            },
        }
        return [
            {
                "epoch": 99,
                "netuid": 71,
                "authority_types": [
                    "chain_realized_unattributed_observation_v1"
                ],
                "allocation_doc": {
                    "champion_allocations": [],
                    "queued_champion_allocations": [],
                },
            }
        ]

    async def load_business(_artifacts):
        return {}

    async def load_receipts(receipt_hashes):
        receipt_requests.extend(sorted(receipt_hashes))
        return {}

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        champion_settlement_v2,
        "load_settled_allocation_history_v2",
        load_history,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_v2",
        load_business,
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
        settlement_frontier_context={
            "frontier": frontier,
            "row": {"source_receipt_hash": frontier_receipt},
            "activation": {"source_receipt_hash": frontier_receipt},
            "source": {
                "receipt_graph": {
                    "root_receipt_hash": frontier_receipt,
                    "receipts": [{"receipt_hash": frontier_receipt}],
                }
            },
            "activation_source": {
                "receipt_graph": {
                    "root_receipt_hash": frontier_receipt,
                    "receipts": [{"receipt_hash": frontier_receipt}],
                }
            },
        },
    )

    assert history_calls == [
        {"netuid": 71, "start_epoch": 98, "end_epoch": 99}
    ]
    assert receipt_requests == []
    assert [graph["root_receipt_hash"] for graph in graphs] == [
        frontier_receipt,
        delta_receipt,
    ]


@pytest.mark.asyncio
async def test_allocation_parent_loader_skips_fully_paid_legacy_source_receipt(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, champion_settlement_v2, store

    reward_ref = "source_add_reward:test"
    settlement_receipt = "sha256:" + "e" * 64
    source_row = {
        "reward_ref": reward_ref,
        "adapter_id": "adapter:fully-paid-source",
        "miner_hotkey": "5FullyPaidSource",
        "leg": 1,
        "reward_kind": "source_acceptance",
        "alpha_percent": 1.0,
        "reward_epochs": 1,
        "current_reward_status": "active",
        "start_epoch": 99,
        "epoch_count": 1,
        "desired_alpha_percent": 1.0,
        "trigger_evidence_doc": {},
        "public_label": "Source acceptance reward",
    }
    expected_decision_hash = v2_authority.sha256_json(
        v2_authority.source_add_reward_row_projection_v2(
            "source_add_leg1",
            {**source_row, "initial_reward_status": "active"},
        )
    )
    business_refs = []
    receipt_roots = []
    source_statuses = []

    async def select_all(table, **kwargs):
        if table == "research_lab_source_add_reward_current":
            filters = dict(
                (item[0], item[1])
                for item in kwargs.get("filters") or ()
                if len(item) == 2
            )
            source_statuses.append(filters.get("current_reward_status"))
            return (
                [source_row]
                if filters.get("current_reward_status") == "active"
                else []
            )
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
        "load_settled_allocation_history_v2",
        load_history,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_v2",
        load_business,
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
        policy={"enable_champ_cap": False},
    )

    assert source_statuses == ["active", "queued", "partially_paid"]
    assert business_refs == [
        ("source_add_reward_decision", reward_ref, expected_decision_hash)
    ]
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


@pytest.mark.asyncio
async def test_allocation_parent_loader_binds_prior_compute_fallback_authority(
    monkeypatch,
):
    from gateway.research_lab import (
        allocations,
        attested_v2_store,
        champion_settlement_v2,
        store,
    )

    settlement_receipt = "sha256:" + "c" * 64
    allocation_payload = {
        "epoch": 99,
        "reimbursement_allocations": [
            {
                "uid": 7,
                "miner_hotkey": "compute-hotkey",
                "source_id": "reimbursement_schedule:source",
                "spend_microusd": 1_000_000,
                "eligible_compute_microusd": 2_000_000,
                "reason": "full_reimbursement",
            }
        ],
    }
    allocation_hash = v2_authority.sha256_json(allocation_payload)
    allocation = {
        **allocation_payload,
        "allocation_hash": allocation_hash,
    }
    source_row = {
        "epoch": 99,
        "netuid": 71,
        "allocation_hash": allocation_hash,
        "allocation_doc": allocation,
    }
    observed_fallback_filters = []

    async def select_all(table, **kwargs):
        if table == allocations.LATEST_LEGACY_COMPUTE_AUTHORITY_TABLE:
            observed_fallback_filters.extend(kwargs["filters"])
            return [
                {
                    **source_row,
                    "epoch_id": source_row["epoch"],
                }
            ]
        return []

    async def load_history(**kwargs):
        assert kwargs == {
            "netuid": 71,
            "start_epoch": 99,
            "end_epoch": 99,
        }
        return [
            {
                **source_row,
                "authority_types": [
                    "legacy_finalized_chain_migration_v2"
                ],
                "legacy_settlement_receipt_hash": settlement_receipt,
            }
        ]

    async def load_business(_artifacts):
        return {}

    async def load_receipts(receipt_hashes):
        assert set(receipt_hashes) == {settlement_receipt}
        return {
            settlement_receipt: {
                "root_receipt_hash": settlement_receipt,
                "receipts": [{"receipt_hash": settlement_receipt}],
            }
        }

    monkeypatch.setattr(allocations, "select_all", select_all)
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
        policy={
            "enable_conservative": False,
            "enable_champ_cap": False,
            "reimbursement_epochs": 20,
        },
    )

    assert ("epoch_id", "lt", 100) in observed_fallback_filters
    assert [graph["root_receipt_hash"] for graph in graphs] == [
        settlement_receipt
    ]


@pytest.mark.asyncio
async def test_allocation_parent_loader_rejects_unsettled_compute_fallback(
    monkeypatch,
):
    from gateway.research_lab import (
        allocations,
        champion_settlement_v2,
        store,
    )

    allocation_payload = {
        "epoch": 99,
        "reimbursement_allocations": [
            {
                "uid": 7,
                "miner_hotkey": "compute-hotkey",
                "source_id": "reimbursement_schedule:source",
                "spend_microusd": 1_000_000,
            }
        ],
    }
    allocation_hash = v2_authority.sha256_json(allocation_payload)

    async def select_all(table, **_kwargs):
        if table == allocations.LATEST_LEGACY_COMPUTE_AUTHORITY_TABLE:
            return [
                {
                    "epoch_id": 99,
                    "netuid": 71,
                    "allocation_hash": allocation_hash,
                    "allocation_doc": {
                        **allocation_payload,
                        "allocation_hash": allocation_hash,
                    },
                }
            ]
        return []

    async def no_history(**_kwargs):
        return []

    monkeypatch.setattr(allocations, "select_all", select_all)
    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        champion_settlement_v2,
        "load_finalized_allocation_history_v2",
        no_history,
    )

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="lacks finalized allocation authority",
    ):
        await v2_authority._load_allocation_parent_graphs_v2(
            epoch_id=100,
            netuid=71,
            policy={
                "enable_conservative": False,
                "enable_champ_cap": False,
                "reimbursement_epochs": 20,
            },
        )
