from __future__ import annotations

import pytest

from gateway.research_lab import v2_authority
from gateway.research_lab import champion_settlement_v2 as settlement
from leadpoet_canonical.attested_v2 import sha256_json


HASH = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
HASH_D = "sha256:" + "d" * 64


def _activation(epoch: int = 100) -> dict[str, object]:
    return {
        "netuid": 71,
        "schema_version": (
            "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        ),
        "first_epoch_id": epoch,
        "source_bundle_hash": HASH,
        "source_bundle_epoch_id": epoch,
        "source_finalized_block": epoch * 360,
    }


@pytest.mark.asyncio
async def test_settlement_backlog_fills_contiguously_through_prior_epoch():
    calls: list[tuple[int, int]] = []

    async def load_latest(table, **_kwargs):
        if table.endswith("_activation_v1"):
            return [_activation()]
        return []

    async def settle(*, epoch_id, settlement_attempt, **_kwargs):
        calls.append((epoch_id, settlement_attempt))
        return {"epoch_id": epoch_id}

    results = await v2_authority.ensure_chain_realized_settlements_v1(
        epoch_id=200,
        netuid=71,
        settlement_attempt=4,
        load_latest=load_latest,
        settle=settle,
    )

    assert calls == [(epoch_id, 4) for epoch_id in range(100, 200)]
    assert [item["epoch_id"] for item in results] == list(range(100, 200))


@pytest.mark.asyncio
async def test_settlement_backlog_rejects_more_than_policy_limit():
    async def load_latest(table, **_kwargs):
        if table.endswith("_activation_v1"):
            return [_activation()]
        return []

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="backlog exceeds policy",
    ):
        await v2_authority.ensure_chain_realized_settlements_v1(
            epoch_id=201,
            netuid=71,
            load_latest=load_latest,
        )


@pytest.mark.asyncio
async def test_settlement_backlog_resumes_after_latest_durable_epoch():
    calls: list[int] = []

    async def load_latest(table, **_kwargs):
        if table.endswith("_activation_v1"):
            return [_activation()]
        return [{"netuid": 71, "epoch_id": 105}]

    async def settle(*, epoch_id, **_kwargs):
        calls.append(epoch_id)
        return {"epoch_id": epoch_id}

    await v2_authority.ensure_chain_realized_settlements_v1(
        epoch_id=109,
        netuid=71,
        load_latest=load_latest,
        settle=settle,
    )

    assert calls == [106, 107, 108]


@pytest.mark.asyncio
async def test_settlement_activation_must_be_unique_and_exact():
    for rows in ([], [_activation(), _activation()]):
        async def load_latest(table, **_kwargs):
            if table.endswith("_activation_v1"):
                return rows
            return []

        with pytest.raises(
            v2_authority.ResearchLabV2AuthorityError,
            match="activation is unavailable or ambiguous",
        ):
            await v2_authority.ensure_chain_realized_settlements_v1(
                epoch_id=101,
                netuid=71,
                load_latest=load_latest,
            )


def _observation(*, epoch_id: int = 101) -> dict[str, object]:
    weights = [[0, 65_535], [7, 3_449]]
    return {
        "schema_version": settlement.CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
        "netuid": 71,
        "epoch_id": epoch_id,
        "official_subnet_epoch_id": epoch_id,
        "cutover_mapping_hash": HASH,
        "close_block": 1_359,
        "close_block_hash": "1" * 64,
        "close_state_root": "2" * 64,
        "next_epoch_block": 1_360,
        "next_epoch_block_hash": "3" * 64,
        "validator_hotkey": "validator-hotkey",
        "validator_uid": 0,
        "metagraph_hotkeys": [
            "validator-hotkey",
            "unused-1",
            "unused-2",
            "unused-3",
            "unused-4",
            "unused-5",
            "unused-6",
            "miner-hotkey",
        ],
        "weights": weights,
        "weights_storage_key": "0x1234",
        "last_update_storage_key": "0x5678",
        "last_update_block": 1_345,
        "last_update_block_hash": "4" * 64,
        "last_update_official_subnet_epoch_id": 100,
        "active_source_epoch_id": 100,
        "weights_vector_hash": sha256_json(
            {
                "uids": [item[0] for item in weights],
                "weights_u16": [item[1] for item in weights],
            }
        ),
    }


@pytest.mark.asyncio
async def test_settle_chain_realized_epoch_executes_both_measured_authorities(
    monkeypatch,
):
    observation = _observation()
    observation_hash = sha256_json(observation)
    settlement_doc = {
        "schema_version": (
            settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1
        ),
        "netuid": 71,
        "epoch_id": 101,
        "credit_hashes": [],
        "observation_summary": {},
    }
    settlement_hash = sha256_json(settlement_doc)
    package = {
        "settlement_doc": settlement_doc,
        "settlement_hash": settlement_hash,
        "credits": [],
    }
    candidate = {
        "bundle_hash": HASH_B,
        "finalization_receipt_hash": HASH_C,
    }
    calls: list[dict[str, object]] = []
    persisted: list[dict[str, object]] = []

    async def execute(**kwargs):
        calls.append(kwargs)
        if kwargs["operation"] == v2_authority.OP_OBSERVE_CHAIN_REALIZED_WEIGHTS_V1:
            receipt = {
                "receipt_hash": HASH,
                "role": "gateway_coordinator",
                "purpose": v2_authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
                "status": "succeeded",
                "epoch_id": 101,
                "output_root": observation_hash,
            }
            return {
                "result": observation,
                "execution_receipt": receipt,
                "execution_receipt_graph": {
                    "root_receipt_hash": HASH,
                    "receipts": [receipt],
                },
                "receipt_graph": {
                    "root_receipt_hash": HASH_D,
                    "receipts": [receipt],
                },
            }
        receipt = {
            "receipt_hash": HASH_B,
            "role": "gateway_coordinator",
            "purpose": v2_authority.CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1,
            "status": "succeeded",
            "epoch_id": 101,
            "output_root": settlement_hash,
        }
        return {
            "result": package,
            "execution_receipt": receipt,
            "execution_receipt_graph": {
                "root_receipt_hash": HASH_B,
                "receipts": [receipt],
            },
            "receipt_graph": {
                "root_receipt_hash": HASH_D,
                "receipts": [receipt],
            },
        }

    async def select_candidates(*_args, **_kwargs):
        return [{"candidate": True}]

    async def load_graph(root):
        assert root == HASH_C
        return {"root_receipt_hash": HASH_C, "receipts": []}

    async def persist_settlement(**kwargs):
        persisted.append(kwargs)
        return {"durable": True}

    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        settlement,
        "select_chain_realized_bundle_candidate_v1",
        lambda _rows, *, observation: (
            candidate if observation == _observation() else None
        ),
    )
    monkeypatch.setattr(
        settlement,
        "validate_chain_realized_epoch_settlements_v1",
        lambda rows, **_kwargs: list(rows),
    )
    monkeypatch.setattr(
        settlement,
        "validate_chain_realized_obligation_credits_v1",
        lambda rows, **_kwargs: list(rows),
    )

    result = await v2_authority.settle_chain_realized_epoch_v1(
        epoch_id=101,
        netuid=71,
        settlement_attempt=3,
        execute=execute,
        persist_settlement=persist_settlement,
        select_candidates=select_candidates,
        load_graph=load_graph,
    )

    assert [call["operation"] for call in calls] == [
        v2_authority.OP_OBSERVE_CHAIN_REALIZED_WEIGHTS_V1,
        v2_authority.OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
    ]
    assert [call["sequence"] for call in calls] == [6, 7]
    assert calls[1]["payload"]["bundle_hash"] == HASH_B
    assert calls[1]["parent_graphs"][0]["root_receipt_hash"] == HASH
    assert calls[1]["parent_graphs"][1]["root_receipt_hash"] == HASH_C
    assert persisted == [{"package": package, "receipt_hash": HASH_B}]
    assert result["status"] == "settled"
    assert result["durable_settlement"] == {"durable": True}


@pytest.mark.asyncio
async def test_settlement_attempt_advances_from_durable_failed_receipt_history():
    calls: list[tuple[str, dict[str, object]]] = []

    async def load_attempt_history(table, **kwargs):
        calls.append((table, kwargs))
        return [{"sequence": 1}]

    resolved = await v2_authority._resolve_chain_settlement_attempt_v1(
        epoch_id=101,
        requested_attempt=0,
        load_attempt_history=load_attempt_history,
    )

    assert resolved == 1
    assert calls == [
        (
            "research_lab_attested_execution_receipts_v2",
            {
                "columns": "purpose,sequence",
                "filters": (
                    ("role", "gateway_coordinator"),
                    ("epoch_id", 101),
                    (
                        "purpose",
                        "in",
                        (
                            v2_authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
                            v2_authority.CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1,
                        ),
                    ),
                ),
                "order_by": (("sequence", True),),
                "limit": 1,
            },
        )
    ]


@pytest.mark.asyncio
async def test_settlement_attempt_history_does_not_rewind_process_retry():
    async def load_attempt_history(*_args, **_kwargs):
        return [{"sequence": 1}]

    assert await v2_authority._resolve_chain_settlement_attempt_v1(
        epoch_id=101,
        requested_attempt=4,
        load_attempt_history=load_attempt_history,
    ) == 4


@pytest.mark.asyncio
async def test_settlement_attempt_history_fails_closed_when_sequence_is_invalid():
    async def load_attempt_history(*_args, **_kwargs):
        return [{"sequence": "1"}]

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="attempt history is invalid",
    ):
        await v2_authority._resolve_chain_settlement_attempt_v1(
            epoch_id=101,
            requested_attempt=0,
            load_attempt_history=load_attempt_history,
        )


@pytest.mark.asyncio
async def test_settle_chain_realized_epoch_rejects_missing_finalization_graph(
    monkeypatch,
):
    observation = _observation()
    receipt = {
        "receipt_hash": HASH,
        "role": "gateway_coordinator",
        "purpose": v2_authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }

    async def execute(**_kwargs):
        return {
            "result": observation,
            "execution_receipt": receipt,
            "receipt_graph": {
                "root_receipt_hash": HASH,
                "receipts": [receipt],
            },
        }

    async def select_candidates(*_args, **_kwargs):
        return [{"candidate": True}]

    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        settlement,
        "select_chain_realized_bundle_candidate_v1",
        lambda _rows, **_kwargs: {
            "bundle_hash": HASH_B,
            "finalization_receipt_hash": HASH_C,
        },
    )

    async def load_graph(_root):
        return {"root_receipt_hash": HASH, "receipts": []}

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="root differs",
    ):
        await v2_authority.settle_chain_realized_epoch_v1(
            epoch_id=101,
            netuid=71,
            execute=execute,
            select_candidates=select_candidates,
            load_graph=load_graph,
        )
