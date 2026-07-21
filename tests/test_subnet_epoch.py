from __future__ import annotations

from dataclasses import replace
import json

import pytest

from Leadpoet.utils.subnet_epoch import (
    CUTOVER_JSON_ENV,
    CUTOVER_PATH_ENV,
    EPOCH_SCHEME,
    OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
    load_subnet_epoch_cutover,
    read_subnet_epoch_snapshot,
    validate_subnet_epoch_cutover_anchor,
)


GENESIS = "0x" + "11" * 32
BLOCK_HASH = "0x" + "22" * 32
CUTOVER_BLOCK_HASH = "0x" + "66" * 32
PRE_CUTOVER_BLOCK_HASH = "0x" + "88" * 32


def _snapshot(**updates) -> SubnetEpochSnapshot:
    values = {
        "network_genesis_hash": GENESIS,
        "netuid": 71,
        "head_kind": "best",
        "block_hash": BLOCK_HASH,
        "current_block": 8_637_160,
        "last_epoch_block": 8_637_156,
        "pending_epoch_at": 0,
        "subnet_epoch_index": 23_927,
        "tempo": 360,
        "blocks_since_last_step": 4,
        "observed_at": "2026-07-16T22:45:00Z",
    }
    values.update(updates)
    return SubnetEpochSnapshot(**values)


def _cutover(**updates) -> SubnetEpochCutover:
    values = {
        "network_genesis_hash": GENESIS,
        "netuid": 71,
        "cutover_block": 8_637_516,
        "cutover_block_hash": CUTOVER_BLOCK_HASH,
        "first_subnet_epoch_index": 23_928,
        "first_settlement_epoch_id": 23_993,
        "last_legacy_epoch_id": 23_992,
    }
    values.update(updates)
    return SubnetEpochCutover(**values)


def test_historical_sn71_vector_uses_stateful_epoch() -> None:
    state = _snapshot()

    assert state.subnet_epoch_index == 23_927
    assert state.epoch_block == 4
    assert state.next_epoch_block == 8_637_516
    assert state.blocks_remaining == 356


def test_pending_epoch_moves_boundary_earlier_and_due_state_stays_on_same_index() -> None:
    pending = _snapshot(pending_epoch_at=8_637_170)
    assert pending.next_epoch_block == 8_637_170
    assert pending.blocks_remaining == 10

    deferred = replace(
        pending,
        current_block=8_637_171,
        blocks_since_last_step=15,
    )
    assert deferred.subnet_epoch_index == pending.subnet_epoch_index
    assert deferred.epoch_block == 15
    assert deferred.blocks_remaining == 0


def test_tempo_reset_uses_observed_last_epoch_block() -> None:
    before = _snapshot(
        current_block=8_995,
        last_epoch_block=8_900,
        tempo=360,
        subnet_epoch_index=50,
        blocks_since_last_step=95,
    )
    state = _snapshot(
        current_block=9_000,
        last_epoch_block=8_990,
        tempo=720,
        subnet_epoch_index=50,
        blocks_since_last_step=400,
    )
    assert state.epoch_block == 10
    assert state.next_epoch_block == 9_710
    assert state.blocks_remaining == 710
    assert state.epoch_ref == before.epoch_ref


@pytest.mark.parametrize(
    ("blocks_since_last_step", "blocks_until_safety"),
    ((50_399, 2), (50_400, 1), (50_401, 0)),
)
def test_blocks_since_last_step_safety_deadline_uses_post_block_counter(
    blocks_since_last_step: int,
    blocks_until_safety: int,
) -> None:
    state = _snapshot(
        current_block=9_000,
        last_epoch_block=8_990,
        tempo=720,
        blocks_since_last_step=blocks_since_last_step,
    )
    assert state.next_epoch_block == 9_000 + blocks_until_safety
    assert state.blocks_remaining == blocks_until_safety


def test_cutover_keeps_settlement_order_without_relabeling_official_epoch() -> None:
    cutover = _cutover()
    first = replace(
        _snapshot(),
        current_block=8_637_516,
        last_epoch_block=8_637_516,
        subnet_epoch_index=23_928,
        blocks_since_last_step=0,
    )
    following = replace(
        first,
        current_block=8_637_876,
        last_epoch_block=8_637_876,
        subnet_epoch_index=23_929,
    )

    assert first.settlement_epoch_id(cutover) == 23_993
    assert following.settlement_epoch_id(cutover) == 23_994
    assert first.to_dict(cutover=cutover)["epoch_id"] == 23_928
    assert first.to_dict(cutover=cutover)["settlement_epoch_id"] == 23_993
    assert first.epoch_ref != following.epoch_ref
    assert cutover.mapping_hash.startswith("sha256:")


def test_cutover_rejects_hash_tampering_and_wrong_lineage() -> None:
    cutover = _cutover()
    with pytest.raises(SubnetEpochError, match="mapping hash mismatch"):
        SubnetEpochCutover.from_mapping(
            {**cutover.to_dict(), "mapping_hash": "sha256:" + "00" * 32}
        )
    with pytest.raises(SubnetEpochError, match="genesis hashes differ"):
        _snapshot(
            network_genesis_hash="0x" + "33" * 32,
            current_block=8_637_516,
            last_epoch_block=8_637_516,
            subnet_epoch_index=23_928,
        ).settlement_epoch_id(cutover)
    with pytest.raises(SubnetEpochError, match="predates"):
        cutover.settlement_epoch_id(23_926)


def test_cutover_loader_requires_one_valid_manifest(tmp_path) -> None:
    manifest = _cutover().to_dict()
    raw = json.dumps(manifest)

    assert load_subnet_epoch_cutover({CUTOVER_JSON_ENV: raw}) == _cutover()

    path = tmp_path / "cutover.json"
    path.write_text(raw, encoding="utf-8")
    assert load_subnet_epoch_cutover({CUTOVER_PATH_ENV: str(path)}) == _cutover()

    with pytest.raises(SubnetEpochError, match="requires a cutover"):
        load_subnet_epoch_cutover({})
    with pytest.raises(SubnetEpochError, match="only one"):
        load_subnet_epoch_cutover(
            {CUTOVER_JSON_ENV: raw, CUTOVER_PATH_ENV: str(path)}
        )


class _Scale:
    def __init__(self, value: int):
        self.value = value


class _Substrate:
    def __init__(self):
        self.calls = []
        self.block_numbers = {
            BLOCK_HASH: 8_637_160,
            "0x" + "44" * 32: 8_637_159,
            "0x" + "55" * 32: 8_600_000,
            PRE_CUTOVER_BLOCK_HASH: 8_637_515,
            CUTOVER_BLOCK_HASH: 8_637_516,
        }
        self.values = {
            "Tempo": 360,
            "LastEpochBlock": 8_637_156,
            "PendingEpochAt": 0,
            "SubnetEpochIndex": 23_927,
            "BlocksSinceLastStep": 4,
        }

    def get_chain_head(self):
        return BLOCK_HASH

    def get_chain_finalised_head(self):
        return "0x" + "44" * 32

    def get_block_number(self, block_hash):
        self.calls.append(("number", block_hash))
        return self.block_numbers[block_hash]

    def get_block_hash(self, block):
        if block == 0:
            return GENESIS
        if block == 8_600_000:
            return "0x" + "55" * 32
        if block == 8_637_515:
            return PRE_CUTOVER_BLOCK_HASH
        if block == 8_637_516:
            return CUTOVER_BLOCK_HASH
        raise AssertionError(f"unexpected block: {block}")

    def query(self, *, module, storage_function, params, block_hash):
        self.calls.append((storage_function, block_hash))
        if module == "Timestamp":
            assert storage_function == "Now"
            assert params == []
            return _Scale(1_752_707_100_123)
        assert module == "SubtensorModule"
        assert params == [71]
        value = self.values[storage_function]
        if block_hash == PRE_CUTOVER_BLOCK_HASH:
            if storage_function == "LastEpochBlock":
                value = 8_637_156
            elif storage_function == "SubnetEpochIndex":
                value = 23_927
            elif storage_function == "BlocksSinceLastStep":
                value = 359
        elif block_hash == CUTOVER_BLOCK_HASH:
            if storage_function == "LastEpochBlock":
                value = 8_637_516
            elif storage_function == "SubnetEpochIndex":
                value = 23_928
            elif storage_function == "BlocksSinceLastStep":
                value = 0
        if block_hash == "0x" + "55" * 32:
            if storage_function == "LastEpochBlock":
                value = 8_599_996
            elif storage_function == "SubnetEpochIndex":
                value = 23_824
        return _Scale(value)


class _Subtensor:
    def __init__(self, chain_endpoint=OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT):
        self.chain_endpoint = chain_endpoint
        self.substrate = _Substrate()


def test_reader_pins_every_storage_field_to_one_hash() -> None:
    subtensor = _Subtensor()
    state = read_subnet_epoch_snapshot(
        subtensor,
        netuid=71,
        observed_at="2026-07-16T22:45:00Z",
    )

    assert state.epoch_scheme == EPOCH_SCHEME
    assert state.epoch_block == 4
    storage_calls = [call for call in subtensor.substrate.calls if call[0] != "number"]
    assert len(storage_calls) == 5
    assert {block_hash for _, block_hash in storage_calls} == {BLOCK_HASH}


def test_reader_uses_exact_chain_timestamp_when_observation_is_omitted() -> None:
    state = read_subnet_epoch_snapshot(_Subtensor(), netuid=71)
    assert state.observed_at == "2025-07-16T23:05:00Z"


def test_reader_fails_closed_on_missing_storage() -> None:
    subtensor = _Subtensor()
    del subtensor.substrate.values["SubnetEpochIndex"]

    with pytest.raises(SubnetEpochError, match="failed to read SubnetEpochIndex"):
        read_subnet_epoch_snapshot(subtensor, netuid=71)


def test_cutover_anchor_is_proven_against_chain_hashes() -> None:
    subtensor = _Subtensor()
    validate_subnet_epoch_cutover_anchor(subtensor, _cutover())

    with pytest.raises(SubnetEpochError, match="block hash differs"):
        validate_subnet_epoch_cutover_anchor(
            subtensor,
            _cutover(cutover_block_hash="0x" + "77" * 32),
        )

    with pytest.raises(SubnetEpochError, match="official Bittensor archive"):
        validate_subnet_epoch_cutover_anchor(
            _Subtensor(chain_endpoint="wss://entrypoint-finney.opentensor.ai:443"),
            _cutover(),
        )


def test_reader_supports_exact_block_number_and_hash() -> None:
    by_number = _Subtensor()
    numbered = read_subnet_epoch_snapshot(
        by_number,
        netuid=71,
        block_number=8_600_000,
        observed_at="2026-07-16T22:45:00Z",
    )
    assert numbered.head_kind == "exact"
    assert numbered.current_block == 8_600_000
    assert numbered.block_hash == "0x" + "55" * 32
    assert {
        call[1]
        for call in by_number.substrate.calls
        if call[0] not in {"number"}
    } == {"0x" + "55" * 32}

    by_hash = _Subtensor()
    hashed = read_subnet_epoch_snapshot(
        by_hash,
        netuid=71,
        block_hash="55" * 32,
        observed_at="2026-07-16T22:45:00Z",
    )
    assert hashed.head_kind == "exact"
    assert hashed.current_block == 8_600_000
    assert hashed.block_hash == "0x" + "55" * 32


def test_reader_rejects_ambiguous_or_mismatched_exact_block() -> None:
    subtensor = _Subtensor()
    with pytest.raises(SubnetEpochError, match="only one"):
        read_subnet_epoch_snapshot(
            subtensor,
            netuid=71,
            block_number=8_600_000,
            block_hash="0x" + "55" * 32,
        )
    with pytest.raises(SubnetEpochError, match="cannot be combined"):
        read_subnet_epoch_snapshot(
            subtensor,
            netuid=71,
            finalized=True,
            block_number=8_600_000,
        )

    subtensor.substrate.block_numbers["0x" + "55" * 32] = 8_600_001
    with pytest.raises(SubnetEpochError, match="block hash and block number differ"):
        read_subnet_epoch_snapshot(
            subtensor,
            netuid=71,
            block_number=8_600_000,
        )


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"tempo": 0}, "tempo must be positive"),
        ({"last_epoch_block": 8_637_161}, "cannot exceed"),
        ({"head_kind": "mixed"}, "best, finalized, or exact"),
    ],
)
def test_invalid_snapshots_fail_closed(updates, error) -> None:
    with pytest.raises(SubnetEpochError, match=error):
        _snapshot(**updates)
