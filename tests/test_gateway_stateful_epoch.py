from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import time

import pytest
from fastapi import HTTPException, Response

from Leadpoet.utils.subnet_epoch import (
    EPOCH_MODE_ENV,
    CUTOVER_JSON_ENV,
    STATEFUL_EPOCH_MODE,
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
    monkeypatch.setenv(EPOCH_MODE_ENV, STATEFUL_EPOCH_MODE)
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

    async def validated(value=None):
        return value or cutover

    async def lifecycle(**_kwargs):
        return {
            "lifecycle_state": "stateful_active",
            "mapping_hash": cutover.mapping_hash,
        }

    monkeypatch.setattr(
        epoch_utils, "validate_stateful_cutover_authority_async", validated
    )
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
    assert status["mode"] == STATEFUL_EPOCH_MODE


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
    monkeypatch.setattr(
        epoch_utils,
        "get_current_subnet_epoch_snapshot_async",
        lambda **_kwargs: _async_value(snapshot),
    )

    assert await epoch_utils.is_epoch_active_async(23_993) is False
    assert await epoch_utils.is_epoch_closed_async(23_993) is False


@pytest.mark.asyncio
async def test_admission_uses_finalized_identity_and_best_head_timing(
    monkeypatch, stateful
):
    from gateway.utils import epoch as epoch_utils

    finalized = replace(
        _snapshot(current_block=8_637_520),
        head_kind="finalized",
    )
    best = _snapshot(
        current_block=8_637_522,
        blocks_since_last_step=6,
    )
    calls = []

    async def snapshot(*, finalized=False):
        calls.append(finalized)
        return finalized_snapshot if finalized else best

    finalized_snapshot = finalized
    monkeypatch.setattr(
        epoch_utils,
        "get_current_subnet_epoch_snapshot_async",
        snapshot,
    )

    authority, timing, workflow_epoch = (
        await epoch_utils.get_current_epoch_admission_context_async()
    )
    assert authority is finalized_snapshot
    assert timing is best
    assert workflow_epoch == 23_993
    assert timing.blocks_remaining == 354
    assert calls == [True, False]


@pytest.mark.asyncio
async def test_admission_rejects_best_head_rollover_during_finality_lag(
    monkeypatch, stateful
):
    from gateway.utils import epoch as epoch_utils

    finalized = replace(
        _snapshot(
            current_block=8_637_875,
            blocks_since_last_step=359,
        ),
        head_kind="finalized",
    )
    best = _snapshot(
        current_block=8_637_876,
        last_epoch_block=8_637_876,
        subnet_epoch_index=23_929,
        blocks_since_last_step=0,
    )

    async def snapshot(*, finalized=False):
        return finalized_snapshot if finalized else best

    finalized_snapshot = finalized
    monkeypatch.setattr(
        epoch_utils,
        "get_current_subnet_epoch_snapshot_async",
        snapshot,
    )

    with pytest.raises(SubnetEpochError, match="no longer live"):
        await epoch_utils.get_current_epoch_admission_context_async()


@pytest.mark.asyncio
async def test_admission_rejects_due_best_head_before_index_advances(
    monkeypatch, stateful
):
    from gateway.utils import epoch as epoch_utils

    finalized = replace(
        _snapshot(
            current_block=8_637_875,
            blocks_since_last_step=359,
        ),
        head_kind="finalized",
    )
    due_best = _snapshot(
        current_block=8_637_876,
        pending_epoch_at=8_637_876,
        blocks_since_last_step=360,
    )

    async def snapshot(*, finalized=False):
        return finalized_snapshot if finalized else due_best

    finalized_snapshot = finalized
    monkeypatch.setattr(
        epoch_utils,
        "get_current_subnet_epoch_snapshot_async",
        snapshot,
    )

    with pytest.raises(SubnetEpochError, match="no longer live"):
        await epoch_utils.get_current_epoch_admission_context_async()


@pytest.mark.asyncio
async def test_validation_assignment_uses_durable_stateful_authority(monkeypatch):
    from gateway.api import validate as validate_api
    from gateway.tasks import epoch_lifecycle

    async def durable(event_type, epoch_id):
        assert (event_type, epoch_id) == ("EPOCH_INITIALIZATION", 23_993)
        return {
            "payload": {
                "epoch_id": 23_993,
                "epoch_key_semantics": "settlement_ordinal",
                "assignment": {"assigned_lead_ids": ["lead-1"]},
            }
        }

    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", durable)
    payload = await validate_api._load_epoch_initialization_payload(
        23_993, stateful_epoch_mode=True
    )
    assert payload["assignment"]["assigned_lead_ids"] == ["lead-1"]

    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        epoch_lifecycle, "get_durable_epoch_event", unavailable
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        await validate_api._load_epoch_initialization_payload(
            23_993, stateful_epoch_mode=True
        )


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_fulfillment_reward_epoch_fails_closed_in_stateful_mode(
    monkeypatch, stateful
):
    from gateway.fulfillment import lifecycle
    from gateway.utils import epoch as epoch_utils

    async def unavailable():
        raise RuntimeError("chain unavailable")

    monkeypatch.setattr(
        epoch_utils,
        "get_current_epoch_id_async",
        unavailable,
    )
    with pytest.raises(
        RuntimeError,
        match="stateful fulfillment reward epoch authority is unavailable",
    ):
        await lifecycle._get_current_epoch()


@pytest.mark.asyncio
async def test_weight_authority_uses_exact_submitted_block_mapping(
    monkeypatch, stateful
):
    from gateway.api import weights

    subtensor = _Subtensor()
    # Best head is in the strict final-15 submission window while the
    # enclave-bound finalized snapshot trails it by 15 blocks.  The exact
    # snapshot may use the 30-block lag buffer, but both remain in one
    # official subnet epoch.
    subtensor.substrate.current_block = 8_637_865
    monkeypatch.setattr(weights, "get_subtensor", lambda: subtensor)
    archive_anchors = []
    monkeypatch.setattr(
        weights,
        "validate_cutover_anchor_from_archive",
        lambda cutover: archive_anchors.append(cutover.mapping_hash),
    )

    previous = await weights._verify_epoch_block_authority(
        netuid=71,
        epoch_id=23_993,
        submitted_block=8_637_850,
        require_submission_window=True,
    )
    assert previous["official_subnet_epoch_id"] == 23_928
    assert previous["blocks_remaining"] == 26
    assert archive_anchors == [stateful.mapping_hash]

    with pytest.raises(HTTPException, match="does not map"):
        await weights._verify_epoch_block_authority(
            netuid=71,
            epoch_id=23_994,
            submitted_block=8_637_850,
            require_submission_window=True,
        )

    # Once best head has crossed the official boundary, a prepared bundle
    # from the prior official epoch is stale rather than recoverable.
    subtensor.substrate.current_block = 8_637_876
    with pytest.raises(HTTPException, match="not in the live official"):
        await weights._verify_epoch_block_authority(
            netuid=71,
            epoch_id=23_993,
            submitted_block=8_637_865,
            require_submission_window=True,
        )


@pytest.mark.asyncio
async def test_weight_authority_rejects_early_and_due_windows(
    monkeypatch, stateful
):
    from gateway.api import weights

    subtensor = _Subtensor()
    monkeypatch.setattr(weights, "get_subtensor", lambda: subtensor)
    monkeypatch.setattr(
        weights,
        "validate_cutover_anchor_from_archive",
        lambda cutover: None,
    )

    with pytest.raises(HTTPException, match="outside"):
        await weights._verify_epoch_block_authority(
            netuid=71,
            epoch_id=23_993,
            submitted_block=8_637_520,
            require_submission_window=True,
        )


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


@pytest.mark.asyncio
async def test_stateful_metagraph_cache_checks_epoch_before_reuse(
    monkeypatch, stateful
):
    from gateway.utils import epoch as epoch_utils
    from gateway.utils import registry

    old = type("Metagraph", (), {"hotkeys": ["old"]})()
    fresh = type("Metagraph", (), {"hotkeys": ["fresh"]})()

    class AsyncSubtensor:
        network = "finney"

        def __init__(self):
            self.calls = 0

        async def metagraph(self, *, netuid):
            assert netuid == 71
            self.calls += 1
            return fresh

    source = AsyncSubtensor()

    async def current_epoch():
        return stateful.first_settlement_epoch_id

    monkeypatch.setattr(epoch_utils, "get_current_epoch_id_async", current_epoch)
    monkeypatch.setattr(registry, "_async_subtensor", source)
    monkeypatch.setattr(registry, "_metagraph_cache", old)
    monkeypatch.setattr(
        registry,
        "_cache_epoch",
        stateful.first_settlement_epoch_id - 1,
    )
    monkeypatch.setattr(registry, "_cache_epoch_timestamp", time.time())
    monkeypatch.setattr(registry, "_fetch_in_progress", False)

    assert await registry.get_metagraph_async() is fresh
    assert source.calls == 1
    assert registry._cache_epoch == stateful.first_settlement_epoch_id


@pytest.mark.asyncio
async def test_stateful_metagraph_refresh_never_returns_prior_epoch(
    monkeypatch, stateful
):
    from gateway.utils import epoch as epoch_utils
    from gateway.utils import registry

    old = type("Metagraph", (), {"hotkeys": ["old"]})()

    class FailingAsyncSubtensor:
        network = "finney"

        async def metagraph(self, *, netuid):
            raise RuntimeError("async chain unavailable")

    class FailingSyncSubtensor:
        def metagraph(self, netuid):
            raise RuntimeError("sync chain unavailable")

    async def current_epoch():
        return stateful.first_settlement_epoch_id

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(epoch_utils, "get_current_epoch_id_async", current_epoch)
    monkeypatch.setattr(registry, "_async_subtensor", FailingAsyncSubtensor())
    monkeypatch.setattr(registry, "_metagraph_cache", old)
    monkeypatch.setattr(
        registry,
        "_cache_epoch",
        stateful.first_settlement_epoch_id - 1,
    )
    monkeypatch.setattr(registry, "_cache_epoch_timestamp", time.time())
    monkeypatch.setattr(registry, "_fetch_in_progress", False)
    monkeypatch.setattr(
        registry.bt,
        "Subtensor",
        lambda **_kwargs: FailingSyncSubtensor(),
    )
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    with pytest.raises(Exception, match="no cache available"):
        await registry.get_metagraph_async()


@pytest.mark.asyncio
async def test_research_lab_stateful_path_never_uses_stale_hint(
    monkeypatch, stateful
):
    from gateway.research_lab import chain
    from gateway.utils import epoch as epoch_utils

    snapshot = replace(_snapshot(), head_kind="finalized")

    async def context(*, finalized=False):
        assert finalized is True
        return snapshot, 23_993

    monkeypatch.setattr(epoch_utils, "get_current_epoch_context_async", context)
    chain._EPOCH_CACHE = (1, 1, "stale", 0.0)

    assert await chain.resolve_research_lab_evaluation_epoch() == (
        23_993,
        8_637_520,
        "gateway_epoch_utils:finalized",
    )


@pytest.mark.asyncio
async def test_research_lab_stateful_failure_does_not_fall_back_to_hint(
    monkeypatch, stateful
):
    from gateway.research_lab import chain
    from gateway.utils import epoch as epoch_utils

    async def broken_context(*, finalized=False):
        raise RuntimeError("chain unavailable")

    monkeypatch.setattr(epoch_utils, "get_current_epoch_context_async", broken_context)
    monkeypatch.setattr(
        chain,
        "_fetch_current_chain_epoch_direct",
        lambda: (_ for _ in ()).throw(RuntimeError("direct unavailable")),
    )
    monkeypatch.setenv("RESEARCH_LAB_GATEWAY_EPOCH_HINT", "99999")
    monkeypatch.setenv("RESEARCH_LAB_GATEWAY_EPOCH_HINT_TS", "9999999999")

    with pytest.raises(RuntimeError, match="exact-hash"):
        await chain.resolve_research_lab_evaluation_epoch()


@pytest.mark.asyncio
async def test_epoch_monitor_transitions_on_official_index_and_remaining_windows(
    stateful,
):
    from gateway.tasks.epoch_monitor import EpochMonitor

    monitor = EpochMonitor(network="finney")
    monitor.startup_block_count = 10
    calls = []

    async def initialize(epoch_id, epoch_snapshot=None):
        calls.append(("initialize", epoch_id, epoch_snapshot.subnet_epoch_index))
        monitor.initialized_epochs.add(epoch_id)
        monitor.initializing_epochs.discard(epoch_id)

    async def cleanup(epoch_id):
        calls.append(("cleanup", epoch_id))

    async def consensus(epoch_id, skip_closed_check=False, epoch_close=None):
        calls.append(("consensus", epoch_id, skip_closed_check, epoch_close is not None))

    async def validation_end(epoch_id, **kwargs):
        calls.append(
            (
                "validation_end",
                epoch_id,
                kwargs["epoch_start"],
                kwargs["epoch_end"],
            )
        )

    monitor._on_epoch_start = initialize
    monitor._run_miner_cleanup = cleanup
    monitor._check_for_reveals = consensus
    monitor._on_validation_end = validation_end

    await monitor._process_stateful_snapshot(_snapshot())
    await asyncio.sleep(0)

    consensus_window = _snapshot(
        current_block=8_637_856,
        block_hash=WINDOW_HASH,
        blocks_since_last_step=340,
        observed_at="2026-07-16T23:51:36Z",
    )
    await monitor._process_stateful_snapshot(consensus_window)
    await asyncio.sleep(0)

    near_end = _snapshot(
        current_block=8_637_873,
        block_hash="0x" + "88" * 32,
        blocks_since_last_step=357,
        observed_at="2026-07-16T23:55:00Z",
    )
    await monitor._process_stateful_snapshot(near_end)
    await asyncio.sleep(0)

    transitioned = _snapshot(
        current_block=8_637_876,
        last_epoch_block=8_637_876,
        subnet_epoch_index=23_929,
        blocks_since_last_step=0,
        block_hash=FOLLOWING_HASH,
        observed_at="2026-07-16T23:55:36Z",
    )
    await monitor._process_stateful_snapshot(transitioned)
    await asyncio.sleep(0)

    assert ("initialize", 23_993, 23_928) in calls
    assert ("cleanup", 23_993) in calls
    assert ("consensus", 23_993, True, True) in calls
    assert any(call[:2] == ("validation_end", 23_993) for call in calls)
    assert ("initialize", 23_994, 23_929) in calls
    assert monitor.last_official_epoch == 23_929
