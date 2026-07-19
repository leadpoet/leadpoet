from __future__ import annotations

import copy

import pytest

from gateway.research_lab import champion_settlement_v2 as settlement
from leadpoet_canonical.attested_v2 import sha256_json


def _allocation(*, paid: float = 5.0) -> dict:
    payload = {
        "schema_version": "leadpoet.research_lab_allocation.v2",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": "champion_reward:test",
                "paid_alpha_percent": paid,
                "base_desired_alpha_percent": 5.0,
            }
        ],
        "queued_champion_allocations": [],
    }
    return {**payload, "allocation_hash": sha256_json(payload)}


def test_legacy_classification_detects_source_add_payments():
    payload = {
        "schema_version": "leadpoet.research_lab_allocation.v2",
        "epoch": 100,
        "champion_allocations": [],
        "queued_champion_allocations": [],
        "source_add_allocations": [
            {
                "source_add_reward_id": "source_add_reward:test",
                "paid_alpha_percent": 1.0,
            }
        ],
    }
    allocation = {**payload, "allocation_hash": sha256_json(payload)}

    epoch, allocation_hash, pays_active = (
        settlement._legacy_allocation_active_champion_payment_v2(
            {
                "epoch": 100,
                "netuid": 71,
                "allocation_hash": allocation["allocation_hash"],
                "allocation_doc": allocation,
            },
            netuid=71,
            active_reward_ids=set(),
            active_source_reward_ids={"source_add_reward:test"},
        )
    )

    assert epoch == 100
    assert allocation_hash == allocation["allocation_hash"]
    assert pays_active is True


def _authority_row(
    marker: str,
    *,
    allocation: dict,
) -> tuple[dict, dict, dict]:
    bundle_hash = "sha256:" + marker * 64
    root_hash = "sha256:" + chr(ord(marker) + 1) * 64
    finalization_root = "sha256:" + chr(ord(marker) + 2) * 64
    publication_root = "sha256:" + chr(ord(marker) + 3) * 64
    allocation_receipt = "sha256:" + "a" * 64
    verified_bundle = {
        "bundle_hash": bundle_hash,
        "netuid": 71,
        "epoch_id": 100,
        "block": 36099,
        "validator_hotkey": f"validator-{marker}",
        "root_receipt_hash": root_hash,
        "weights_hash": marker * 64,
        "snapshot_hash": "sha256:" + "b" * 64,
        "weight_receipt_hash": "sha256:" + "c" * 64,
    }
    bundle_doc = {
        "schema_version": "leadpoet.published_weight_bundle.v2",
        "fixture_marker": marker,
        "weight_snapshot": {
            "calculation_snapshot": {
                "research_lab_allocation_doc": allocation,
            },
            "input_receipt_hashes": {
                "research_lab_allocation": allocation_receipt,
            },
        },
    }
    expected_bundle_row = {
        "bundle_hash": bundle_hash,
        "schema_version": bundle_doc["schema_version"],
        "netuid": 71,
        "epoch_id": 100,
        "block": 36099,
        "validator_hotkey": f"validator-{marker}",
        "root_receipt_hash": root_hash,
        "weights_hash": marker * 64,
        "snapshot_hash": "sha256:" + "b" * 64,
        "bundle_doc": bundle_doc,
    }
    durable_hash = sha256_json(expected_bundle_row)
    transparency_hash = "sha256:" + "d" * 64
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": bundle_hash,
        "root_receipt_hash": root_hash,
        "durable_readback_hash": durable_hash,
        "transparency_event_hash": transparency_hash,
    }
    submission_event = sha256_json(
        {
            "bundle_hash": bundle_hash,
            "publication_receipt_hash": publication_root,
            "transparency_event_hash": transparency_hash,
            "durable_readback_hash": durable_hash,
        }
    )
    verified_finalization = {
        "validator_hotkey": f"validator-{marker}",
        "netuid": 71,
        "epoch_id": 100,
        "weights_hash": marker * 64,
        "weight_receipt_hash": "sha256:" + "c" * 64,
        "extrinsic_authorization_hash": "sha256:" + "e" * 64,
        "extrinsic_hash": "0x" + marker * 64,
        "finalized_block": 36105,
        "finalized_block_hash": marker * 64,
        "state_transition_hash": "sha256:" + "f" * 64,
    }
    finalization_doc = {"fixture_marker": marker}
    finalization_event = sha256_json(
        {
            "weight_submission_event_hash": submission_event,
            "bundle_hash": bundle_hash,
            "finalization_receipt_hash": finalization_root,
            "extrinsic_authorization_hash": verified_finalization[
                "extrinsic_authorization_hash"
            ],
            "extrinsic_hash": verified_finalization["extrinsic_hash"],
            "finalized_block": verified_finalization["finalized_block"],
            "finalized_block_hash": verified_finalization[
                "finalized_block_hash"
            ],
            "state_transition_hash": verified_finalization[
                "state_transition_hash"
            ],
        }
    )
    row = {
        **expected_bundle_row,
        "weight_submission_event_hash": submission_event,
        "publication_receipt_hash": publication_root,
        "transparency_event_hash": transparency_hash,
        "durable_readback_hash": durable_hash,
        "publication_doc": publication_doc,
        "weight_finalization_event_hash": finalization_event,
        "finalization_receipt_hash": finalization_root,
        "extrinsic_authorization_hash": verified_finalization[
            "extrinsic_authorization_hash"
        ],
        "extrinsic_hash": verified_finalization["extrinsic_hash"],
        "finalized_block": verified_finalization["finalized_block"],
        "finalized_block_hash": verified_finalization["finalized_block_hash"],
        "state_transition_hash": verified_finalization[
            "state_transition_hash"
        ],
        "finalization_doc": finalization_doc,
    }
    graph = {
        "root_receipt_hash": finalization_root,
        "receipts": [{"receipt_hash": finalization_root}],
    }
    return row, graph, {
        "bundle": verified_bundle,
        "finalization": verified_finalization,
    }


def _install_validators(monkeypatch, validations: dict[str, dict]) -> None:
    monkeypatch.setattr(
        settlement,
        "validate_published_weight_bundle_v2",
        lambda document: validations[str(document["fixture_marker"])]["bundle"],
    )
    monkeypatch.setattr(
        settlement,
        "validate_weight_finalization_submission_v2",
        lambda submission: validations[
            str(submission["finalization"]["fixture_marker"])
        ]["finalization"],
    )


def test_finalized_allocation_authority_collapses_validator_duplicates(monkeypatch):
    allocation = _allocation()
    first, first_graph, first_verified = _authority_row(
        "1", allocation=allocation
    )
    second, second_graph, second_verified = _authority_row(
        "2", allocation=allocation
    )
    _install_validators(
        monkeypatch,
        {"1": first_verified, "2": second_verified},
    )

    result = settlement.validate_finalized_allocation_authorities_v2(
        [first, second],
        finalization_graphs={
            first_graph["root_receipt_hash"]: first_graph,
            second_graph["root_receipt_hash"]: second_graph,
        },
    )

    assert len(result) == 1
    assert result[0]["allocation_doc"] == allocation
    assert result[0]["finalized_authority_count"] == 2
    assert result[0]["allocation_receipt_hash"] == "sha256:" + "a" * 64


def test_finalized_allocation_authority_fails_on_missing_or_tampered_evidence(
    monkeypatch,
):
    row, graph, verified = _authority_row("1", allocation=_allocation())
    _install_validators(monkeypatch, {"1": verified})

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="receipt graph is missing",
    ):
        settlement.validate_finalized_allocation_authorities_v2(
            [row], finalization_graphs={}
        )

    tampered = copy.deepcopy(row)
    tampered["bundle_doc"]["weight_snapshot"]["calculation_snapshot"][
        "research_lab_allocation_doc"
    ]["champion_allocations"][0]["paid_alpha_percent"] = 99.0
    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="publication differs from its bundle",
    ):
        settlement.validate_finalized_allocation_authorities_v2(
            [tampered],
            finalization_graphs={graph["root_receipt_hash"]: graph},
        )


@pytest.mark.asyncio
async def test_cutover_requires_receipts_for_every_positive_balance(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    settled_id = "champion_reward:sha256:" + "1" * 64
    positive_id = "champion_reward:sha256:" + "2" * 64

    def reward_row(reward_id: str, uid: int) -> dict:
        return {
            "champion_reward_id": reward_id,
            "score_bundle_id": f"score-{uid}",
            "candidate_id": f"candidate-{uid}",
            "run_id": f"run-{uid}",
            "miner_hotkey": f"miner-{uid}",
            "miner_uid": uid,
            "island": "generalist",
            "evaluation_epoch": 99,
            "start_epoch": 100,
            "epoch_count": 2,
            "improvement_points": 2.0,
            "threshold_points": 1.0,
            "desired_alpha_percent": 5.0,
            "input_hash": "sha256:" + "3" * 64,
            "anchored_hash": "sha256:" + str(uid + 3) * 64,
            "current_reward_status": "active",
        }

    rows = [reward_row(settled_id, 1), reward_row(positive_id, 2)]

    async def select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_emission_allocation_current":
            return []
        requested_status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return rows if requested_status == "active" else []

    finalized_allocation_body = {
        "schema_version": "1.0",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": settled_id,
                "paid_alpha_percent": 10.0,
            },
            {
                "source_id": positive_id,
                "paid_alpha_percent": 5.0,
            },
        ],
        "queued_champion_allocations": [],
    }
    finalized_allocation = {
        **finalized_allocation_body,
        "allocation_hash": sha256_json(finalized_allocation_body),
    }
    finalized = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": finalized_allocation["allocation_hash"],
            "allocation_doc": finalized_allocation,
        }
    ]

    async def load_finalized(**_kwargs):
        return finalized

    positive_projection = champion_reward_row_projection_v2(rows[1])
    root_hash = "sha256:" + "9" * 64

    async def load_graph(**_kwargs):
        return {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": root_hash,
                    "role": "gateway_coordinator",
                    "purpose": "research_lab.reward_decision.v2",
                    "output_root": sha256_json(positive_projection),
                }
            ],
        }

    async def load_graphs(artifacts):
        return {
            key: await load_graph(
                artifact_kind=key[0],
                artifact_ref=key[1],
            )
            for key in artifacts
        }

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_finalized_allocation_history_v2",
        load_finalized,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_graphs,
    )

    ready = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert ready["ready"] is True
    assert ready["receipt_coverage"] == 1.0
    assert ready["covered_positive_balance_count"] == 1
    assert [
        item["champion_reward_id"] for item in ready["zero_balance_active_rows"]
    ] == [settled_id]

    async def missing_graph(**_kwargs):
        raise RuntimeError("not migrated")

    async def missing_graphs(_artifacts):
        raise RuntimeError("not migrated")

    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        missing_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        missing_graphs,
    )
    blocked = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert blocked["ready"] is False
    assert blocked["receipt_coverage"] == 0.0
    assert blocked["missing"] == [
        {
            "champion_reward_id": positive_id,
            "remaining_alpha_percent": 5.0,
            "reason": "missing_or_invalid_v2_reward_receipt",
        }
    ]


@pytest.mark.asyncio
async def test_cutover_does_not_trust_paid_status_when_chain_balance_remains(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store

    reward_id = "champion_reward:sha256:" + "4" * 64
    reward = {
        "champion_reward_id": reward_id,
        "start_epoch": 100,
        "epoch_count": 2,
        "desired_alpha_percent": 5.0,
        "current_reward_status": "paid",
    }

    async def select_all(table, *, filters=(), **_kwargs):
        if table in {
            "research_lab_emission_allocation_current",
            "research_lab_emission_allocation_snapshots",
            "research_lab_arweave_epoch_audit_anchor_current",
            "published_weight_bundles",
        }:
            return []
        status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return [reward] if status == "paid" else []

    async def no_finalized_payments(**_kwargs):
        return []

    async def no_receipt(**_kwargs):
        raise RuntimeError("not migrated")

    async def no_receipts(_artifacts):
        raise RuntimeError("not migrated")

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_finalized_allocation_history_v2",
        no_finalized_payments,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        no_receipt,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        no_receipts,
    )

    readiness = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )

    assert readiness["ready"] is False
    assert readiness["required_positive_balance_count"] == 1
    assert readiness["missing"][0]["champion_reward_id"] == reward_id
    assert readiness["missing"][0]["remaining_alpha_percent"] == 10.0


@pytest.mark.asyncio
async def test_cutover_blocks_until_every_historical_payment_epoch_is_attested(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    reward_id = "champion_reward:sha256:" + "6" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score-1",
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "miner_hotkey": "miner-1",
        "miner_uid": 1,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 2,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 5.0,
        "input_hash": "sha256:" + "7" * 64,
        "anchored_hash": "sha256:" + "8" * 64,
        "current_reward_status": "active",
    }
    allocation_body = {
        "schema_version": "1.0",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": reward_id,
                "paid_alpha_percent": 5.0,
                "base_desired_alpha_percent": 5.0,
            }
        ],
        "queued_champion_allocations": [],
    }
    allocation = {
        **allocation_body,
        "allocation_hash": sha256_json(allocation_body),
    }

    async def select_all(table, *, filters=(), **_kwargs):
        if table in {
            "research_lab_emission_allocation_current",
            "research_lab_emission_allocation_snapshots",
        }:
            return [
                {
                    "epoch": 100,
                    "netuid": 71,
                    "allocation_hash": allocation["allocation_hash"],
                    "allocation_doc": allocation,
                }
            ]
        if table == "research_lab_arweave_epoch_audit_anchor_current":
            return [
                {
                    "epoch": 100,
                    "allocation_hash": allocation["allocation_hash"],
                    "weights_hash": "sha256:" + "a" * 64,
                    "current_arweave_tx_id": "A" * 43,
                    "current_transparency_event_hash": "b" * 64,
                }
            ]
        if table == "published_weight_bundles":
            return [
                {
                    "epoch_id": 100,
                    "weights_hash": "a" * 64,
                }
            ]
        status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return [reward] if status == "active" else []

    state = {"finalized": [], "nonfinalized": []}

    async def load_finalized(**_kwargs):
        return list(state["finalized"])

    async def load_nonfinalized(**_kwargs):
        return list(state["nonfinalized"])

    root_hash = "sha256:" + "9" * 64

    async def load_graph(**_kwargs):
        return {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": root_hash,
                    "role": "gateway_coordinator",
                    "purpose": "research_lab.reward_decision.v2",
                    "output_root": sha256_json(
                        champion_reward_row_projection_v2(reward)
                    ),
                }
            ],
        }

    async def load_graphs(artifacts):
        return {
            key: await load_graph(
                artifact_kind=key[0],
                artifact_ref=key[1],
            )
            for key in artifacts
        }

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_finalized_allocation_history_v2",
        load_finalized,
    )
    monkeypatch.setattr(
        settlement,
        "load_legacy_allocation_nonfinalizations_v2",
        load_nonfinalized,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_graphs,
    )

    blocked = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert blocked["ready"] is False
    assert blocked["receipt_coverage"] == 1.0
    assert blocked["historical_settlement_coverage"] == 0.0
    assert blocked["missing_historical_settlements"] == [
        {
            "epoch": 100,
            "allocation_hash": allocation["allocation_hash"],
                "reason": "missing_finalized_chain_classification_authority",
        }
    ]

    state["nonfinalized"] = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": allocation["allocation_hash"],
            "allocation_doc": allocation,
            "finding_hash": "sha256:" + "f" * 64,
        }
    ]
    classified_unpaid = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert classified_unpaid["ready"] is True
    assert classified_unpaid[
        "covered_historical_nonfinalization_epochs"
    ] == [100]
    assert classified_unpaid["migrated_finalized_allocation_epoch_count"] == 0
    assert classified_unpaid["unproven_historical_allocations"] == [
        {
            "epoch": 100,
            "allocation_hash": allocation["allocation_hash"],
            "reason": "finalized_chain_vector_mismatch",
        }
    ]

    state["nonfinalized"] = []
    state["finalized"] = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": allocation["allocation_hash"],
            "allocation_doc": allocation,
            "authority_types": ["legacy_finalized_chain_migration_v2"],
        }
    ]
    ready = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert ready["ready"] is True
    assert ready["historical_settlement_coverage"] == 1.0

    mismatched_body = {
        key: value
        for key, value in allocation.items()
        if key != "allocation_hash"
    }
    mismatched_body["netuid"] = 72
    allocation.clear()
    allocation.update(
        {
            **mismatched_body,
            "allocation_hash": sha256_json(mismatched_body),
        }
    )
    blocked_scope = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert blocked_scope["ready"] is False
    assert blocked_scope["missing_historical_settlements"][0]["reason"] == (
        "invalid_historical_allocation"
    )


@pytest.mark.asyncio
async def test_cutover_uses_anchor_bound_snapshot_not_later_current_view(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    reward_id = "champion_reward:sha256:" + "1" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score-anchor",
        "candidate_id": "candidate-anchor",
        "run_id": "run-anchor",
        "miner_hotkey": "miner-anchor",
        "miner_uid": 1,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 2,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 5.0,
        "input_hash": "sha256:" + "2" * 64,
        "anchored_hash": "sha256:" + "3" * 64,
        "current_reward_status": "active",
    }
    anchored_body = {
        "schema_version": "1.0",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": reward_id,
                "paid_alpha_percent": 5.0,
                "base_desired_alpha_percent": 5.0,
            }
        ],
        "queued_champion_allocations": [],
        "snapshot_generation": "submitted",
    }
    anchored = {
        **anchored_body,
        "allocation_hash": sha256_json(anchored_body),
    }
    current_body = {
        **anchored_body,
        "snapshot_generation": "recomputed",
    }
    current = {
        **current_body,
        "allocation_hash": sha256_json(current_body),
    }

    async def select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_emission_allocation_current":
            documents = [current]
        elif table == "research_lab_emission_allocation_snapshots":
            documents = [anchored, current]
        else:
            documents = []
        if documents:
            return [
                {
                    "epoch": 100,
                    "netuid": 71,
                    "allocation_hash": document["allocation_hash"],
                    "allocation_doc": document,
                }
                for document in documents
            ]
        if table == "research_lab_arweave_epoch_audit_anchor_current":
            return [
                {
                    "epoch": 100,
                    "allocation_hash": anchored["allocation_hash"],
                    "weights_hash": "sha256:" + "4" * 64,
                    "current_arweave_tx_id": "A" * 43,
                    "current_transparency_event_hash": "5" * 64,
                }
            ]
        if table == "published_weight_bundles":
            return [{"epoch_id": 100, "weights_hash": "4" * 64}]
        status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return [reward] if status == "active" else []

    state = {"finalized": []}

    async def load_finalized(**_kwargs):
        return list(state["finalized"])

    root_hash = "sha256:" + "6" * 64
    receipt_graph = {
        "root_receipt_hash": root_hash,
        "receipts": [
            {
                "receipt_hash": root_hash,
                "role": "gateway_coordinator",
                "purpose": "research_lab.reward_decision.v2",
                "output_root": sha256_json(
                    champion_reward_row_projection_v2(reward)
                ),
            }
        ],
    }

    async def load_graph(**_kwargs):
        return receipt_graph

    async def load_graphs(artifacts):
        return {key: receipt_graph for key in artifacts}

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_finalized_allocation_history_v2",
        load_finalized,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_graphs,
    )

    blocked = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )

    assert blocked["missing_historical_settlements"] == [
        {
            "epoch": 100,
            "allocation_hash": anchored["allocation_hash"],
                "reason": "missing_finalized_chain_classification_authority",
        }
    ]
    assert blocked["unproven_historical_allocations"] == [
        {
            "epoch": 100,
            "allocation_hash": current["allocation_hash"],
            "reason": "current_allocation_not_checkpointed",
        }
    ]

    state["finalized"] = [
        {
            "epoch": 100,
            "netuid": 71,
            "allocation_hash": anchored["allocation_hash"],
            "allocation_doc": anchored,
            "authority_types": ["legacy_finalized_chain_migration_v2"],
        }
    ]
    ready = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )
    assert ready["ready"] is True
    assert ready["covered_historical_settlement_epochs"] == [100]


@pytest.mark.asyncio
async def test_cutover_does_not_credit_unsubmitted_historical_allocation(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, store
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    reward_id = "champion_reward:sha256:" + "3" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score-3",
        "candidate_id": "candidate-3",
        "run_id": "run-3",
        "miner_hotkey": "miner-3",
        "miner_uid": 3,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 2,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 5.0,
        "input_hash": "sha256:" + "4" * 64,
        "anchored_hash": "sha256:" + "5" * 64,
        "current_reward_status": "active",
    }
    allocation_body = {
        "schema_version": "1.0",
        "epoch": 100,
        "champion_allocations": [
            {
                "source_id": reward_id,
                "paid_alpha_percent": 5.0,
                "base_desired_alpha_percent": 5.0,
            }
        ],
        "queued_champion_allocations": [],
    }
    allocation = {
        **allocation_body,
        "allocation_hash": sha256_json(allocation_body),
    }

    async def select_all(table, *, filters=(), **_kwargs):
        if table in {
            "research_lab_emission_allocation_current",
            "research_lab_emission_allocation_snapshots",
        }:
            return [
                {
                    "epoch": 100,
                    "netuid": 71,
                    "allocation_hash": allocation["allocation_hash"],
                    "allocation_doc": allocation,
                }
            ]
        if table in {
            "research_lab_arweave_epoch_audit_anchor_current",
            "published_weight_bundles",
        }:
            return []
        status = next(
            (
                item[1]
                for item in filters
                if len(item) == 2 and item[0] == "current_reward_status"
            ),
            "",
        )
        return [reward] if status == "active" else []

    async def no_finalized_payments(**_kwargs):
        return []

    root_hash = "sha256:" + "6" * 64
    receipt_graph = {
        "root_receipt_hash": root_hash,
        "receipts": [
            {
                "receipt_hash": root_hash,
                "role": "gateway_coordinator",
                "purpose": "research_lab.reward_decision.v2",
                "output_root": sha256_json(
                    champion_reward_row_projection_v2(reward)
                ),
            }
        ],
    }

    async def load_graph(**_kwargs):
        return receipt_graph

    async def load_graphs(artifacts):
        return {key: receipt_graph for key in artifacts}

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.setattr(
        settlement,
        "load_finalized_allocation_history_v2",
        no_finalized_payments,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graphs_by_ref_v2",
        load_graphs,
    )

    readiness = await settlement.champion_v2_cutover_readiness(
        epoch=102,
        netuid=71,
    )

    assert readiness["ready"] is True
    assert readiness["required_positive_balance_count"] == 1
    assert readiness["covered_positive_balance_count"] == 1
    assert readiness["required_historical_settlement_count"] == 0
    assert readiness["unproven_historical_allocation_count"] == 1
    assert readiness["unproven_historical_allocations"] == [
        {
            "epoch": 100,
            "allocation_hash": allocation["allocation_hash"],
            "reason": "no_checkpointed_audit_anchor",
        }
    ]


@pytest.mark.asyncio
async def test_status_reconciler_only_closes_finalized_chain_balances(monkeypatch):
    from gateway.research_lab import maintenance
    from gateway.research_lab import store
    from gateway.research_lab import allocations

    reward_id = "champion_reward:sha256:" + "7" * 64
    reward = {
        "champion_reward_id": reward_id,
        "miner_uid": 7,
        "desired_alpha_percent": 5.0,
        "epoch_count": 2,
        "current_reward_status": "active",
    }

    async def select_all(_table, *, filters=(), **_kwargs):
        status = next(
            (value for field, value in filters if field == "current_reward_status"),
            "",
        )
        return [reward] if status == "active" else []

    async def fully_settled(**_kwargs):
        return {reward_id: 99.0}

    writes = []

    async def create_event(**kwargs):
        writes.append(kwargs)
        return {"seq": 2, "anchored_hash": "sha256:" + "8" * 64}

    monkeypatch.setattr(maintenance, "select_all", select_all)
    monkeypatch.setattr(
        allocations,
        "_champion_finalized_paid_alpha_to_date",
        fully_settled,
    )
    monkeypatch.setattr(store, "create_champion_reward_event", create_event)

    result = await maintenance.reconcile_champion_reward_statuses(
        epoch=102,
        netuid=71,
        actor_ref="test:reconciler",
        dry_run=False,
    )

    assert result["ok"] is True
    assert result["repaired_count"] == 1
    assert result["repaired"][0]["paid_alpha_percent_to_date"] == 10.0
    assert len(writes) == 1
    assert writes[0]["event_type"] == "paid"
    assert writes[0]["event_doc"]["settlement_authority"] == (
        "finalized_v2_weight_extrinsics"
    )

    async def partially_settled(**_kwargs):
        return {reward_id: 9.999}

    monkeypatch.setattr(
        allocations,
        "_champion_finalized_paid_alpha_to_date",
        partially_settled,
    )
    writes.clear()
    held = await maintenance.reconcile_champion_reward_statuses(
        epoch=102,
        netuid=71,
        dry_run=False,
    )
    assert held["planned_count"] == 0
    assert writes == []


@pytest.mark.asyncio
async def test_default_v2_allocation_path_blocks_incomplete_champion_coverage(
    monkeypatch,
):
    from gateway.research_lab import v2_authority

    async def not_ready(**_kwargs):
        return {
            "ready": False,
            "receipt_coverage": 0.5,
            "missing": [{"champion_reward_id": "champion:missing"}],
        }

    monkeypatch.setattr(
        settlement,
        "champion_v2_cutover_readiness",
        not_ready,
    )

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="champion V2 cutover blocked",
    ):
        await v2_authority.build_allocation_v2(
            epoch_id=102,
            netuid=71,
            policy={},
        )


@pytest.mark.asyncio
async def test_source_add_receipt_backfill_is_idempotent_and_measured(monkeypatch):
    from gateway.research_lab import attested_v2_store, maintenance, v2_authority
    from gateway.tee.reward_executor_v2 import source_add_reward_row_projection_v2

    reward_ref = "source_add_reward:201a08f0d2b503bf"
    reward = {
        "reward_ref": reward_ref,
        "adapter_id": "adapter:uspto-patents-center-api-86bb73c0149e",
        "miner_hotkey": "miner-1",
        "leg": 1,
        "reward_kind": "source_acceptance",
        "alpha_percent": 1.0,
        "reward_epochs": 20,
        "start_epoch": 23870,
        "trigger_evidence_doc": {
            "submission_id": "source_add_submission:a3d8f3e562dca636",
            "precheck_status": "provenance_precheck_passed",
            "reward_trigger": "provenance_precheck_passed",
        },
        "public_label": "Source acceptance reward",
        "current_reward_status": "active",
        "created_at": "2026-07-10T00:00:00Z",
    }

    async def select_all(_table, *, filters=(), **_kwargs):
        status = next(
            (value for field, value in filters if field == "current_reward_status"),
            "",
        )
        return [reward] if status == "active" else []

    migrated = []
    state = {"covered": False}
    root_hash = "sha256:" + "e" * 64

    async def load_graph(**_kwargs):
        if not state["covered"]:
            raise RuntimeError("not migrated")
        return {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": root_hash,
                    "purpose": "research_lab.reward_decision.v2",
                    "output_root": sha256_json(
                        source_add_reward_row_projection_v2(
                            "source_add_leg1",
                            {**reward, "initial_reward_status": "active"},
                        )
                    ),
                }
            ],
        }

    async def attest(**kwargs):
        migrated.append(kwargs)
        state["covered"] = True
        return {"execution_receipt": {"receipt_hash": root_hash}}

    monkeypatch.setattr(maintenance, "select_all", select_all)
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        v2_authority,
        "attest_historical_source_add_reward_v2",
        attest,
    )

    first = await maintenance.backfill_source_add_reward_v2_authority(
        epoch=24038,
        dry_run=False,
    )
    second = await maintenance.backfill_source_add_reward_v2_authority(
        epoch=24038,
        dry_run=False,
    )

    assert first["migrated_count"] == 1
    assert second["already_covered_count"] == 1
    assert second["migrated_count"] == 0
    assert migrated == [{"epoch_id": 24038, "reward_ref": reward_ref}]


@pytest.mark.asyncio
async def test_champion_receipt_backfill_is_idempotent_and_measured(monkeypatch):
    from gateway.research_lab import attested_v2_store, maintenance, v2_authority
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    reward_id = "champion_reward:sha256:" + "a" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": "score_bundle:" + "b" * 64,
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "miner_hotkey": "miner-1",
        "miner_uid": 1,
        "island": "generalist",
        "evaluation_epoch": 99,
        "start_epoch": 100,
        "epoch_count": 20,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 5.0,
        "input_hash": "sha256:" + "c" * 64,
        "anchored_hash": "sha256:" + "d" * 64,
        "current_reward_status": "active",
        "created_at": "2026-07-10T00:00:00Z",
    }

    async def select_all(_table, *, filters=(), **_kwargs):
        status = next(
            (value for field, value in filters if field == "current_reward_status"),
            "",
        )
        return [reward] if status == "active" else []

    migrated = []
    state = {"covered": False}
    root_hash = "sha256:" + "e" * 64

    async def load_graph(**_kwargs):
        if not state["covered"]:
            raise RuntimeError("not migrated")
        return {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": root_hash,
                    "purpose": "research_lab.reward_decision.v2",
                    "output_root": sha256_json(
                        champion_reward_row_projection_v2(reward)
                    ),
                }
            ],
        }

    async def attest(**kwargs):
        migrated.append(kwargs)
        state["covered"] = True
        return {"execution_receipt": {"receipt_hash": root_hash}}

    monkeypatch.setattr(maintenance, "select_all", select_all)
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_by_ref_v2",
        load_graph,
    )
    monkeypatch.setattr(
        v2_authority,
        "attest_historical_champion_reward_v2",
        attest,
    )

    first = await maintenance.backfill_champion_reward_v2_authority(
        epoch=102,
        dry_run=False,
    )
    second = await maintenance.backfill_champion_reward_v2_authority(
        epoch=102,
        dry_run=False,
    )

    assert first["migrated_count"] == 1
    assert second["already_covered_count"] == 1
    assert second["migrated_count"] == 0
    assert migrated == [
        {"epoch_id": 102, "champion_reward_id": reward_id}
    ]


@pytest.mark.asyncio
async def test_champion_settlement_backfill_is_dry_run_safe_and_resumable(
    monkeypatch,
):
    from gateway.research_lab import maintenance, v2_authority

    missing = {
        "epoch": 100,
        "allocation_hash": "sha256:" + "1" * 64,
        "reason": "missing_finalized_chain_classification_authority",
    }
    state = {"covered": False}
    calls = []

    async def readiness(**kwargs):
        assert kwargs == {"epoch": 102, "netuid": 71}
        return {
            "ready": state["covered"],
                "missing_historical_classifications": (
                    [] if state["covered"] else [missing]
                ),
        }

    async def classify(**kwargs):
        calls.append(kwargs)
        state["covered"] = True
        return {
            "status": "finalized",
            "result": {"settlement_hash": "sha256:" + "2" * 64},
            "execution_receipt": {"receipt_hash": "sha256:" + "3" * 64},
        }

    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        readiness,
    )
    monkeypatch.setattr(
        v2_authority,
        "classify_historical_champion_allocation_v2",
        classify,
    )
    dry = await maintenance.backfill_champion_settlement_v2_authority(
        epoch=102,
        netuid=71,
        dry_run=True,
    )
    assert dry["planned"] == [
        {"epoch": 100, "allocation_hash": "sha256:" + "1" * 64}
    ]
    assert calls == []

    written = await maintenance.backfill_champion_settlement_v2_authority(
        epoch=102,
        netuid=71,
        dry_run=False,
    )
    assert written["ok"] is True
    assert written["migrated_count"] == 1
    assert written["readiness_after"]["ready"] is True
    assert calls == [
        {"epoch_id": 102, "netuid": 71, "settlement_epoch_id": 100}
    ]

    repeated = await maintenance.backfill_champion_settlement_v2_authority(
        epoch=102,
        netuid=71,
        dry_run=False,
    )
    assert repeated["planned_count"] == 0
    assert repeated["migrated_count"] == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_champion_settlement_backfill_never_migrates_invalid_evidence(
    monkeypatch,
):
    from gateway.research_lab import maintenance

    async def readiness(**_kwargs):
        return {
            "ready": False,
            "missing_historical_settlements": [
                {
                    "epoch": 100,
                    "reason": "finalized_chain_allocation_hash_mismatch",
                }
            ],
        }

    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        readiness,
    )
    result = await maintenance.backfill_champion_settlement_v2_authority(
        epoch=102,
        netuid=71,
        dry_run=False,
    )
    assert result["ok"] is False
    assert result["planned_count"] == 0
    assert result["blocked"][0]["reason"] == (
        "finalized_chain_allocation_hash_mismatch"
    )


@pytest.mark.asyncio
async def test_legacy_v1_paid_helper_keeps_snapshot_accounting(monkeypatch):
    from gateway.research_lab import allocations

    reward_id = "champion_reward:sha256:" + "9" * 64
    calls = []

    async def select_all(table, **kwargs):
        calls.append((table, kwargs))
        return [
            {
                "epoch": 100,
                "allocation_doc": {
                    "champion_allocations": [
                        {
                            "source_id": reward_id,
                            "paid_alpha_percent": 5.0,
                            "base_desired_alpha_percent": 5.0,
                        }
                    ]
                },
            }
        ]

    monkeypatch.setattr(allocations, "select_all", select_all)
    paid = await allocations._champion_paid_alpha_to_date(
        epoch=102,
        netuid=71,
        champion_rows=[
            {
                "champion_reward_id": reward_id,
                "start_epoch": 100,
                "epoch_count": 20,
                "desired_alpha_percent": 5.0,
            }
        ],
    )

    assert paid == {reward_id: 5.0}
    assert calls[0][0] == "research_lab_emission_allocation_current"
    assert calls[0][1]["allow_partial"] is True
