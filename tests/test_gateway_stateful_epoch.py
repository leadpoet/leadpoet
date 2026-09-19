from __future__ import annotations

import json

import pytest
from fastapi import Response

from Leadpoet.utils.subnet_epoch import (
    CUTOVER_JSON_ENV,
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
)


GENESIS = "0x" + "11" * 32
CUTOVER_HASH = "0x" + "22" * 32
HEAD_HASH = "0x" + "33" * 32
WINDOW_HASH = "0x" + "44" * 32
NEXT_HASH = "0x" + "55" * 32
FOLLOWING_HASH = "0x" + "66" * 32


def _cutover() -> SubnetEpochCutover:
    return SubnetEpochCutover(
        network_genesis_hash=GENESIS,
        netuid=71,
        cutover_block=8_637_516,
        cutover_block_hash=NEXT_HASH,
        first_subnet_epoch_index=23_928,
        first_settlement_epoch_id=23_993,
        last_legacy_epoch_id=23_992,
    )


def _snapshot(**updates) -> SubnetEpochSnapshot:
    values = {
        "network_genesis_hash": GENESIS,
        "netuid": 71,
        "head_kind": "best",
        "block_hash": HEAD_HASH,
        "current_block": 8_637_520,
        "last_epoch_block": 8_637_516,
        "pending_epoch_at": 0,
        "subnet_epoch_index": 23_928,
        "tempo": 360,
        "blocks_since_last_step": 4,
        "observed_at": "2026-07-16T22:45:00Z",
    }
    values.update(updates)
    return SubnetEpochSnapshot(**values)


def test_epoch_time_estimate_normalizes_observed_offset_to_utc():
    from gateway.utils import epoch as epoch_utils

    snapshot = _snapshot(observed_at="2026-07-16T15:45:00-07:00")
    observed = epoch_utils._observed_datetime(snapshot)
    assert observed.isoformat() == "2026-07-16T22:45:00"


class _Scale:
    def __init__(self, value: int):
        self.value = value


class _Substrate:
    def __init__(self):
        self.by_block = {
            8_637_515: {
                "hash": CUTOVER_HASH,
                "Tempo": 360,
                "LastEpochBlock": 8_637_156,
                "PendingEpochAt": 0,
                "SubnetEpochIndex": 23_927,
                "BlocksSinceLastStep": 359,
            },
            8_637_516: {
                "hash": NEXT_HASH,
                "Tempo": 360,
                "LastEpochBlock": 8_637_516,
                "PendingEpochAt": 0,
                "SubnetEpochIndex": 23_928,
                "BlocksSinceLastStep": 0,
            },
            8_637_520: {
                "hash": HEAD_HASH,
                "Tempo": 360,
                "LastEpochBlock": 8_637_516,
                "PendingEpochAt": 0,
                "SubnetEpochIndex": 23_928,
                "BlocksSinceLastStep": 4,
            },
            8_637_811: {
                "hash": "0x" + "88" * 32,
                "Tempo": 360,
                "LastEpochBlock": 8_637_516,
                "PendingEpochAt": 0,
                "SubnetEpochIndex": 23_928,
                "BlocksSinceLastStep": 295,
            },
            8_637_816: {
                "hash": "0x" + "99" * 32,
                "Tempo": 360,
                "LastEpochBlock": 8_637_516,
                "PendingEpochAt": 0,
                "SubnetEpochIndex": 23_928,
                "BlocksSinceLastStep": 300,
            },
            8_637_850: {
                "hash": WINDOW_HASH,
                "Tempo": 360,
                "LastEpochBlock": 8_637_516,
                "PendingEpochAt": 0,
                "SubnetEpochIndex": 23_928,
                "BlocksSinceLastStep": 334,
            },
            8_637_865: {
                "hash": "0x" + "77" * 32,
                "Tempo": 360,
                "LastEpochBlock": 8_637_516,
                "PendingEpochAt": 0,
                "SubnetEpochIndex": 23_928,
                "BlocksSinceLastStep": 349,
            },
            8_637_876: {
                "hash": FOLLOWING_HASH,
                "Tempo": 360,
                "LastEpochBlock": 8_637_876,
                "PendingEpochAt": 0,
                "SubnetEpochIndex": 23_929,
                "BlocksSinceLastStep": 0,
            },
        }
        self.current_block = 8_637_520
        self.calls = []

    @property
    def current_hash(self):
        return self.by_block[self.current_block]["hash"]

    def get_chain_head(self):
        return self.current_hash

    def get_chain_finalised_head(self):
        return self.current_hash

    def get_block_hash(self, block):
        if block == 0:
            return GENESIS
        if block == 8_637_516:
            return NEXT_HASH
        return self.by_block[int(block)]["hash"]

    def get_block_number(self, block_hash):
        if block_hash == NEXT_HASH:
            return 8_637_516
        for block, values in self.by_block.items():
            if values["hash"] == block_hash:
                return block
        raise KeyError(block_hash)

    def query(self, *, module, storage_function, params, block_hash):
        if module == "Timestamp":
            assert storage_function == "Now"
            assert params == []
            return _Scale(1_752_707_100_000)
        assert module == "SubtensorModule"
        assert params == [71]
        block = self.get_block_number(block_hash)
        self.calls.append((storage_function, block_hash))
        return _Scale(self.by_block[block][storage_function])


class _Subtensor:
    def __init__(self):
        self.substrate = _Substrate()

    def get_current_block(self):
        return self.substrate.current_block


@pytest.fixture
def stateful(monkeypatch):
    cutover = _cutover()
    monkeypatch.setenv(CUTOVER_JSON_ENV, json.dumps(cutover.to_dict()))
    monkeypatch.setenv("BITTENSOR_NETUID", "71")
    from gateway.utils import epoch as epoch_utils

    monkeypatch.setattr(
        epoch_utils, "_validate_cutover_authority_sync", lambda _cutover: None
    )
    monkeypatch.setattr(
        epoch_utils,
        "validate_cutover_anchor_from_archive",
        lambda _cutover: None,
    )

    async def lifecycle(**_kwargs):
        return {
            "lifecycle_state": "stateful_active",
            "mapping_hash": cutover.mapping_hash,
        }

    monkeypatch.setattr(
        epoch_utils,
        "validate_epoch_runtime_lifecycle_async",
        lifecycle,
    )
    return cutover


@pytest.mark.asyncio
async def test_gateway_status_separates_official_and_settlement_ids(
    monkeypatch, stateful
):
    from gateway.utils import epoch as epoch_utils

    subtensor = _Subtensor()
    monkeypatch.setattr(epoch_utils, "_sync_subtensor", subtensor)
    monkeypatch.setattr(epoch_utils, "_validated_cutover_anchor_key", None)

    status = await epoch_utils.get_epoch_authority_status_async()

    assert status["official_subnet_epoch_id"] == 23_928
    assert status["epoch_id"] == 23_928
    assert status["workflow_epoch_id"] == 23_993
    assert status["settlement_epoch_id"] == 23_993
    assert status["epoch_block"] == 4
    assert status["blocks_remaining"] == 356


def test_live_state_stays_on_finney_while_cutover_anchor_uses_archive(
    monkeypatch,
    stateful,
):
    from gateway.utils import epoch as epoch_utils

    finney = object()
    snapshot = _snapshot()
    observed = []
    monkeypatch.setattr(epoch_utils, "_sync_subtensor", finney)
    monkeypatch.setattr(epoch_utils, "_validated_cutover_anchor_key", None)
    monkeypatch.setattr(
        epoch_utils,
        "read_subnet_epoch_snapshot",
        lambda source, **kwargs: (
            observed.append(("live", source, kwargs)) or snapshot
        ),
    )
    monkeypatch.setattr(
        epoch_utils,
        "validate_cutover_anchor_from_archive",
        lambda cutover: observed.append(
            ("archive-anchor", cutover.mapping_hash)
        ),
    )

    assert epoch_utils._read_subnet_epoch_snapshot_sync() is snapshot
    assert observed == [
        ("live", finney, {"netuid": 71, "finalized": False}),
        ("archive-anchor", stateful.mapping_hash),
    ]


@pytest.mark.asyncio
async def test_pending_due_epoch_is_not_active(monkeypatch, stateful):
    from gateway.utils import epoch as epoch_utils

    snapshot = _snapshot(
        current_block=8_637_530,
        pending_epoch_at=8_637_530,
        blocks_since_last_step=14,
    )
    info = epoch_utils.get_current_epoch_info_from_snapshot(snapshot)
    assert info["is_active"] is False
    assert info["is_closed"] is False
    assert info["phase"] == "transition_pending"


@pytest.mark.asyncio
async def test_epoch_state_route_is_explicitly_no_store(monkeypatch):
    from gateway.api import epoch as epoch_api

    async def status():
        return {"official_subnet_epoch_id": 23_928}

    from gateway.utils import epoch as epoch_utils

    monkeypatch.setattr(epoch_utils, "get_epoch_authority_status_async", status)
    response = Response()
    result = await epoch_api.get_epoch_state(response)
    assert result == {"official_subnet_epoch_id": 23_928}
    assert response.headers["cache-control"] == "private, no-store"
