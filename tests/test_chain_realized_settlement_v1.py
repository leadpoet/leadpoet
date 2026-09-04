from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from gateway.research_lab import v2_authority
from gateway.research_lab import champion_settlement_v2 as settlement
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.chain_source_v2 import (
    last_update_storage_key,
    reveal_period_epochs_storage_key,
    weights_storage_key,
)


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


def _observation_v2(
    *,
    revealed_bundle_hash: str | None = None,
    reveal_proof: dict[str, object] | None = None,
) -> dict[str, object]:
    observation = _observation()
    observation["schema_version"] = (
        settlement.CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V2
    )
    observation.pop("active_source_epoch_id")
    observation["last_update_official_subnet_epoch_id"] = 101
    observation["latest_commit_source_epoch_id"] = 101
    observation["epoch_start_block"] = 1_000
    observation["epoch_start_block_hash"] = "%064x" % 1_000
    observation["reveal_window_start_block"] = 1_000
    observation["reveal_window_start_block_hash"] = "%064x" % 1_000
    observation["scheduled_reveal_subnet_epoch_id"] = 100
    observation["scheduled_reveal_source_epoch_id"] = 100
    observation["revealed_bundle_hash"] = revealed_bundle_hash
    observation["reveal_proof"] = reveal_proof
    observation["subnet_reveal_period_epochs"] = 1
    observation["weights_storage_key"] = weights_storage_key(
        netuid=71, validator_uid=0
    )
    observation["last_update_storage_key"] = last_update_storage_key(
        netuid=71
    )
    observation["reveal_period_storage_key"] = (
        reveal_period_epochs_storage_key(netuid=71)
    )
    observation["reveal_period_storage_override"] = None
    observation["reveal_period_metadata_hash"] = (
        "sha256:79fc9235a87651a0cd5b93856d4b5696ffb8a0bd26c6f30a1f1402ac8aaad195"
    )
    observation["reveal_period_runtime_spec_version"] = 452
    profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text(encoding="utf-8")
    )
    observation["chain_signing_profile"] = profile
    observation["chain_signing_profile_hash"] = sha256_json(profile)
    return observation


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

    async def select_candidates(table, **_kwargs):
        if table == settlement.COMPACT_WEIGHT_AUTHORITY_TABLE_V2:
            return []
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
async def test_settlement_attempt_advances_from_stale_failed_receipt_history():
    calls: list[tuple[str, dict[str, object]]] = []

    async def load_attempt_history(table, **kwargs):
        calls.append((table, kwargs))
        return [
            {
                "sequence": 1,
                "receipt_status": "failed",
                "issued_at": (
                    datetime.now(timezone.utc) - timedelta(seconds=301)
                ).isoformat(),
            }
        ]

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
                "columns": "purpose,sequence,receipt_status,issued_at",
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
        return [
            {
                "sequence": 1,
                "receipt_status": "succeeded",
                "issued_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

    assert await v2_authority._resolve_chain_settlement_attempt_v1(
        epoch_id=101,
        requested_attempt=4,
        load_attempt_history=load_attempt_history,
    ) == 4


@pytest.mark.asyncio
async def test_settlement_attempt_history_fails_closed_when_sequence_is_invalid():
    async def load_attempt_history(*_args, **_kwargs):
        return [
            {
                "sequence": "1",
                "receipt_status": "failed",
                "issued_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

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
async def test_recent_failed_settlement_attempt_cools_down_before_execute():
    execute_calls = 0

    async def load_attempt_history(*_args, **_kwargs):
        return [
            {
                "sequence": 0,
                "receipt_status": "failed",
                "issued_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

    async def execute(**_kwargs):
        nonlocal execute_calls
        execute_calls += 1
        raise AssertionError("chain execution must not run during cooldown")

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="retry is cooling down",
    ):
        await v2_authority.settle_chain_realized_epoch_v1(
            epoch_id=101,
            netuid=71,
            execute=execute,
            load_attempt_history=load_attempt_history,
        )

    assert execute_calls == 0


@pytest.mark.asyncio
async def test_failed_settlement_attempt_with_invalid_timestamp_fails_closed():
    async def load_attempt_history(*_args, **_kwargs):
        return [
            {
                "sequence": 0,
                "receipt_status": "failed",
                "issued_at": "not-a-timestamp",
            }
        ]

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

    async def select_candidates(table, **_kwargs):
        if table == settlement.COMPACT_WEIGHT_AUTHORITY_TABLE_V2:
            return []
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


@pytest.mark.asyncio
async def test_post_cutover_host_never_queries_legacy_full_authority(
    monkeypatch,
):
    observation = _observation()
    observation_hash = sha256_json(observation)
    candidate = {
        "bundle_hash": HASH_B,
        "finalization_receipt_hash": HASH_C,
    }
    table_calls = []
    measured_payloads = []

    class StopAfterAuthoritySelection(RuntimeError):
        pass

    async def execute(**kwargs):
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
            }
        measured_payloads.append(dict(kwargs["payload"]))
        raise StopAfterAuthoritySelection

    async def select_candidates(table, **kwargs):
        table_calls.append((table, kwargs))
        if kwargs.get("columns") == "epoch_id":
            return [{"epoch_id": 100}]
        if table == settlement.COMPACT_WEIGHT_AUTHORITY_TABLE_V2:
            return [{"compact": True}]
        raise AssertionError("legacy full authority must not be queried")

    async def load_graph(root):
        return {"root_receipt_hash": root, "receipts": []}

    async def load_attempt_history(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        settlement,
        "select_compact_chain_realized_bundle_candidate_v2",
        lambda rows, **_kwargs: candidate if rows == [{"compact": True}] else None,
    )

    with pytest.raises(StopAfterAuthoritySelection):
        await v2_authority.settle_chain_realized_epoch_v1(
            epoch_id=101,
            netuid=71,
            execute=execute,
            load_attempt_history=load_attempt_history,
            select_candidates=select_candidates,
            load_graph=load_graph,
        )

    assert [table for table, _kwargs in table_calls] == [
        settlement.COMPACT_WEIGHT_AUTHORITY_TABLE_V2,
        settlement.COMPACT_WEIGHT_AUTHORITY_TABLE_V2,
    ]
    assert measured_payloads == [
        {
            "schema_version": "leadpoet.chain_realized_settlement_request.v1",
            "netuid": 71,
            "epoch_id": 101,
            "observation": observation,
            "observation_receipt_hash": HASH,
            "authority_mode": "finalized_bundle",
            "bundle_hash": HASH_B,
        }
    ]


@pytest.mark.asyncio
async def test_v2_host_selects_only_event_proved_exact_bundle(
    monkeypatch,
):
    observation = _observation_v2(
        revealed_bundle_hash=HASH_B,
        reveal_proof={"bundle_hash": HASH_B},
    )
    observation_hash = sha256_json(observation)
    full_row = {"compact": True}
    candidate = {
        "bundle_hash": HASH_B,
        "finalization_receipt_hash": HASH_C,
    }
    table_calls = []
    measured_payloads = []

    class StopAfterAuthoritySelection(RuntimeError):
        pass

    async def execute(**kwargs):
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
            }
        measured_payloads.append(dict(kwargs["payload"]))
        raise StopAfterAuthoritySelection

    async def select_candidates(table, **kwargs):
        table_calls.append((table, kwargs))
        if (
            table == settlement.COMPACT_WEIGHT_AUTHORITY_TABLE_V2
            and kwargs.get("limit") == 2
        ):
            return [full_row]
        raise AssertionError(kwargs)

    async def load_graph(root):
        return {"root_receipt_hash": root, "receipts": []}

    async def load_attempt_history(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        settlement,
        "validate_chain_weight_observation_v1",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        settlement,
        "select_compact_chain_realized_bundle_candidate_v2",
        lambda rows, **_kwargs: candidate if rows == [full_row] else None,
    )

    with pytest.raises(StopAfterAuthoritySelection):
        await v2_authority.settle_chain_realized_epoch_v1(
            epoch_id=101,
            netuid=71,
            execute=execute,
            load_attempt_history=load_attempt_history,
            select_candidates=select_candidates,
            load_graph=load_graph,
        )

    assert [kwargs.get("limit") for _table, kwargs in table_calls] == [2]
    assert table_calls[0][1]["filters"] == (
        ("netuid", 71),
        ("bundle_hash", HASH_B),
        ("authority_stage", "finalized"),
    )
    assert measured_payloads[0]["bundle_hash"] == HASH_B
    assert measured_payloads[0]["authority_mode"] == "finalized_bundle"


@pytest.mark.asyncio
async def test_v2_host_does_not_use_legacy_vector_without_reveal_proof(
    monkeypatch,
):
    observation = _observation_v2()
    observation_hash = sha256_json(observation)
    table_calls = []
    measured_payloads = []

    class StopAfterAuthoritySelection(RuntimeError):
        pass

    async def execute(**kwargs):
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
            }
        measured_payloads.append(dict(kwargs["payload"]))
        raise StopAfterAuthoritySelection

    async def select_candidates(table, **kwargs):
        table_calls.append((table, kwargs))
        raise AssertionError((table, kwargs))

    async def load_graph(root):
        return {"root_receipt_hash": root, "receipts": []}

    async def load_attempt_history(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(StopAfterAuthoritySelection):
        await v2_authority.settle_chain_realized_epoch_v1(
            epoch_id=101,
            netuid=71,
            execute=execute,
            load_attempt_history=load_attempt_history,
            select_candidates=select_candidates,
            load_graph=load_graph,
        )

    assert table_calls == []
    assert measured_payloads[0]["bundle_hash"] is None
    assert measured_payloads[0]["authority_mode"] == "unattributed"
