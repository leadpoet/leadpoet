from __future__ import annotations

import pytest

from gateway.tee import coordinator_chain_realized_settlement_v1 as authority
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import sha256_json


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


class _Reader:
    def __init__(self, rows_by_policy):
        self.rows_by_policy = rows_by_policy
        self.calls = []

    def read(self, **kwargs):
        self.calls.append(dict(kwargs))
        return list(self.rows_by_policy[kwargs["policy_id"]])


class _Chain:
    def __init__(self, observation):
        self.observation = observation
        self.calls = []

    def read_stateful_epoch_close_weights(self, **kwargs):
        self.calls.append(dict(kwargs))
        return dict(self.observation)


def _context(*, graphs=()):
    return ExecutionContextV2(
        job_id="chain-realized:101",
        purpose=authority.CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1,
        epoch_id=101,
        external_receipt_graphs=tuple(graphs),
    )


def _chain_state():
    return {
        "official_subnet_epoch_id": 101,
        "cutover_mapping_hash": HASH_A,
        "close_block": 1_359,
        "close_block_hash": "1" * 64,
        "close_header": {"state_root": "2" * 64},
        "next_epoch_block": 1_360,
        "next_epoch_block_hash": "3" * 64,
        "validator_hotkey": "validator-hotkey",
        "validator_uid": 0,
        "metagraph_hotkeys": ["validator-hotkey", "miner-hotkey"],
        "weights": [[0, 65_535], [1, 3_449]],
        "weights_storage_key": "0x1234",
        "last_update_storage_key": "0x5678",
        "last_update_block": 1_345,
        "last_update_block_hash": "4" * 64,
        "last_update_official_subnet_epoch_id": 100,
        "active_source_epoch_id": 100,
    }


def _observation():
    state = _chain_state()
    close_header = state.pop("close_header")
    return {
        "schema_version": authority.CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
        "netuid": 71,
        "epoch_id": 101,
        **state,
        "close_state_root": close_header["state_root"],
        "weights_vector_hash": sha256_json(
            {
                "uids": [item[0] for item in state["weights"]],
                "weights_u16": [item[1] for item in state["weights"]],
            }
        ),
    }


def test_observation_uses_latest_finalized_primary_identity(monkeypatch):
    reader = _Reader(
        {
            "latest_finalized_allocation_authority": [
                {"validator_hotkey": "old", "finalized_block": 900},
                {
                    "validator_hotkey": "validator-hotkey",
                    "finalized_block": 1_000,
                },
            ]
        }
    )
    chain = _Chain(_chain_state())
    monkeypatch.setattr(
        authority,
        "_preliminary_finalized_bundle_authority_v1",
        lambda row: dict(row),
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=chain,
    )

    observed = source.observe(
        payload={
            "schema_version": (
                authority.CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1
            ),
            "netuid": 71,
            "epoch_id": 101,
        },
        context=_context(),
    )

    assert observed == _observation()
    assert chain.calls[0]["validator_hotkey"] == "validator-hotkey"
    assert reader.calls[0]["policy_id"] == (
        "latest_finalized_allocation_authority"
    )


def test_observation_rejects_ambiguous_latest_primary_identity(monkeypatch):
    reader = _Reader(
        {
            "latest_finalized_allocation_authority": [
                {"validator_hotkey": "first", "finalized_block": 1_000},
                {"validator_hotkey": "second", "finalized_block": 1_000},
            ]
        }
    )
    monkeypatch.setattr(
        authority,
        "_preliminary_finalized_bundle_authority_v1",
        lambda row: dict(row),
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
    )

    with pytest.raises(
        authority.CoordinatorChainRealizedSettlementV1Error,
        match="identity is ambiguous",
    ):
        source.observe(
            payload={
                "schema_version": (
                    authority.CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1
                ),
                "netuid": 71,
                "epoch_id": 101,
            },
            context=_context(),
        )


def test_settlement_rereads_exact_vector_and_rejects_host_substitution(
    monkeypatch,
):
    observation = _observation()
    observation_receipt = {
        "receipt_hash": HASH_A,
        "role": "gateway_coordinator",
        "purpose": authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }
    finalization_receipt = {"receipt_hash": HASH_C}
    graphs = (
        {"root_receipt_hash": HASH_A, "receipts": [observation_receipt]},
        {"root_receipt_hash": HASH_C, "receipts": [finalization_receipt]},
    )
    reader = _Reader(
        {
            "finalized_authority_by_chain_vector": [
                {"candidate": True, "bundle_hash": HASH_B}
            ]
        }
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
    )
    monkeypatch.setattr(
        authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "select_chain_realized_bundle_candidate_v1",
        lambda _rows, **_kwargs: {
            "bundle_hash": HASH_B,
            "finalization_receipt_hash": HASH_C,
        },
    )

    with pytest.raises(
        authority.CoordinatorChainRealizedSettlementV1Error,
        match="host-selected",
    ):
        source.settle(
            payload={
                "schema_version": (
                    authority.CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1
                ),
                "netuid": 71,
                "epoch_id": 101,
                "observation": observation,
                "observation_receipt_hash": HASH_A,
                "bundle_hash": "sha256:" + "d" * 64,
            },
            context=_context(graphs=graphs),
        )

    filters = reader.calls[0]["parameters"]
    assert filters["source_epoch_id"] == 100
    assert filters["validator_hotkey"] == "validator-hotkey"
    assert filters["finalized_block"] == 1_345
    assert filters["finalized_block_hash"] == "4" * 64
    assert filters["uids"] == [0, 1]
    assert filters["weights_u16"] == [65_535, 3_449]


def test_settlement_accepts_only_independently_verified_exact_authority(
    monkeypatch,
):
    observation = _observation()
    observation_receipt = {
        "receipt_hash": HASH_A,
        "role": "gateway_coordinator",
        "purpose": authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }
    finalization_receipt = {"receipt_hash": HASH_C}
    graphs = [
        {"root_receipt_hash": HASH_A, "receipts": [observation_receipt]},
        {"root_receipt_hash": HASH_C, "receipts": [finalization_receipt]},
    ]
    candidate = {
        "bundle_hash": HASH_B,
        "finalization_receipt_hash": HASH_C,
    }
    package = {
        "settlement_doc": {"epoch_id": 101},
        "settlement_hash": "sha256:" + "d" * 64,
        "credits": [],
    }
    reader = _Reader(
        {
            "finalized_authority_by_chain_vector": [
                {"candidate": True, "bundle_hash": HASH_B}
            ]
        }
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
    )
    monkeypatch.setattr(
        authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "select_chain_realized_bundle_candidate_v1",
        lambda _rows, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        authority,
        "validate_finalized_allocation_authorities_v2",
        lambda rows, **_kwargs: list(rows),
    )
    monkeypatch.setattr(
        authority,
        "build_chain_realized_settlement_package_v1",
        lambda **kwargs: package if kwargs["authority"] == candidate else None,
    )

    result = source.settle(
        payload={
            "schema_version": (
                authority.CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1
            ),
            "netuid": 71,
            "epoch_id": 101,
            "observation": observation,
            "observation_receipt_hash": HASH_A,
            "bundle_hash": HASH_B,
        },
        context=_context(graphs=graphs),
    )

    assert result == package
