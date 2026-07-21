from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover, SubnetEpochSnapshot


GENESIS = "0x" + "11" * 32


def _cutover() -> SubnetEpochCutover:
    return SubnetEpochCutover(
        network_genesis_hash=GENESIS,
        netuid=71,
        cutover_block=100,
        cutover_block_hash="0x" + "22" * 32,
        first_subnet_epoch_index=7,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )


def _snapshot(
    *,
    block: int = 104,
    official_epoch: int = 7,
    last_epoch_block: int = 100,
    tempo: int = 360,
    observed_at: str = "2026-07-16T12:00:00Z",
) -> SubnetEpochSnapshot:
    return SubnetEpochSnapshot(
        network_genesis_hash=GENESIS,
        netuid=71,
        head_kind="finalized",
        block_hash="0x" + f"{block:064x}"[-64:],
        current_block=block,
        last_epoch_block=last_epoch_block,
        pending_epoch_at=0,
        subnet_epoch_index=official_epoch,
        tempo=tempo,
        blocks_since_last_step=max(0, block - last_epoch_block),
        observed_at=observed_at,
    )


@pytest.mark.asyncio
async def test_stateful_monitor_uses_finalized_scheduler_snapshot(monkeypatch):
    from gateway.tasks import epoch_monitor
    from gateway.utils import epoch as epoch_utils

    cutover = _cutover()
    snapshot = _snapshot()
    observed = {}
    processed = asyncio.Event()

    class _Subtensor:
        pass

    monkeypatch.setattr(epoch_monitor.bt, "Subtensor", lambda **_kwargs: _Subtensor())
    monkeypatch.setattr(epoch_monitor, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(
        epoch_monitor,
        "validate_cutover_anchor_from_archive",
        lambda *_args, **_kwargs: None,
    )

    async def validate_authority(_cutover):
        return None

    monkeypatch.setattr(
        epoch_utils,
        "validate_stateful_cutover_authority_async",
        validate_authority,
    )

    def read(_subtensor, **kwargs):
        observed.update(kwargs)
        return snapshot

    monkeypatch.setattr(epoch_monitor, "read_subnet_epoch_snapshot", read)
    monitor = epoch_monitor.EpochMonitor()

    async def hydrate(_snapshot):
        return None

    async def process(_snapshot):
        processed.set()

    monkeypatch.setattr(monitor, "_hydrate_stateful_state", hydrate)
    monkeypatch.setattr(monitor, "_process_stateful_snapshot", process)
    task = asyncio.create_task(monitor.start())
    await asyncio.wait_for(processed.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert observed["netuid"] == 71
    assert observed["finalized"] is True


@pytest.mark.asyncio
async def test_stateful_phase_contract_separates_consensus_and_cleanup(monkeypatch):
    from gateway.tasks import epoch_monitor

    cutover = _cutover()
    monkeypatch.setattr(epoch_monitor, "load_subnet_epoch_cutover", lambda: cutover)
    monitor = epoch_monitor.EpochMonitor()
    calls = []

    async def initialize(epoch_id, epoch_snapshot=None):
        calls.append(("initialize", epoch_id))
        monitor.initialized_epochs.add(epoch_id)
        monitor.initializing_epochs.discard(epoch_id)

    async def consensus(epoch_id, **_kwargs):
        calls.append(("consensus", epoch_id))
        monitor.processing_epochs.discard(epoch_id)

    async def cleanup(epoch_id):
        calls.append(("cleanup", epoch_id))

    monkeypatch.setattr(monitor, "_on_epoch_start", initialize)
    monkeypatch.setattr(monitor, "_check_for_reveals", consensus)
    monkeypatch.setattr(monitor, "_run_miner_cleanup", cleanup)

    consensus_snapshot = _snapshot(block=440, last_epoch_block=100)
    assert consensus_snapshot.blocks_remaining == 20
    await monitor._process_stateful_snapshot(consensus_snapshot)
    await asyncio.sleep(0)

    cleanup_snapshot = _snapshot(block=457, last_epoch_block=100)
    assert cleanup_snapshot.blocks_remaining == 3
    await monitor._process_stateful_snapshot(cleanup_snapshot)
    await asyncio.sleep(0)

    assert calls.count(("consensus", 101)) == 1
    assert calls.count(("cleanup", 101)) == 1


@pytest.mark.asyncio
async def test_consensus_retry_refuses_to_cross_into_weight_window(monkeypatch):
    from gateway.tasks import epoch_monitor
    from gateway.utils import epoch as epoch_utils

    late = _snapshot(block=445, last_epoch_block=100)
    assert late.blocks_remaining == 15

    async def context(*, finalized=None):
        assert finalized is True
        return late, 101

    monkeypatch.setenv("LEADPOET_EPOCH_MODE", "stateful_v1")
    monkeypatch.setattr(epoch_utils, "get_current_epoch_context_async", context)
    monitor = epoch_monitor.EpochMonitor()
    monitor.processing_epochs.add(101)

    await monitor._check_for_reveals(101, skip_closed_check=True)

    assert 101 not in monitor.processing_epochs
    assert 101 not in monitor.closed_epochs


@pytest.mark.asyncio
async def test_restart_hydration_closes_previous_epoch_from_durable_init(
    monkeypatch,
):
    from gateway.tasks import epoch_lifecycle, epoch_monitor

    cutover = _cutover()
    current = _snapshot(
        block=460,
        official_epoch=8,
        last_epoch_block=460,
        observed_at="2026-07-16T13:11:12Z",
    )
    initialization = {
        "id": 1,
        "payload": {
            "epoch_id": 101,
            "epoch_boundaries": {
                "start_timestamp": "2026-07-16T12:00:00"
            },
        },
    }

    async def durable(event_type, epoch_id, **_kwargs):
        if event_type == "EPOCH_INITIALIZATION" and epoch_id == 101:
            return initialization
        return None

    monkeypatch.setattr(epoch_monitor, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setenv("LEADPOET_EPOCH_MODE", "stateful_v1")
    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", durable)
    monitor = epoch_monitor.EpochMonitor()
    boundary = current
    closed = []

    async def find(*_args, **_kwargs):
        return boundary

    async def close(epoch_id, observed_boundary):
        closed.append((epoch_id, observed_boundary.current_block))

    monkeypatch.setattr(monitor, "_find_stateful_transition_snapshot", find)
    monkeypatch.setattr(monitor, "_close_stateful_epoch_at_boundary", close)
    await monitor._hydrate_stateful_state(current)

    assert closed == [(101, 460)]


@pytest.mark.asyncio
async def test_restart_skips_missing_initialization_after_boundary(
    monkeypatch,
):
    # A boundary that passed while no lifecycle process observed it can
    # never be initialized. The monitor must record the epoch as skipped
    # (it settles nonfinalized) instead of wedging the polling loop; the
    # invariant that initializations are never originated late holds.
    from gateway.tasks import epoch_lifecycle, epoch_monitor

    cutover = _cutover()
    late = _snapshot(block=110)

    async def missing(*_args, **_kwargs):
        return None

    async def boundary(*_args, **_kwargs):
        return _snapshot(block=100)

    monkeypatch.setattr(epoch_monitor, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", missing)
    monitor = epoch_monitor.EpochMonitor()
    monkeypatch.setattr(monitor, "_find_stateful_transition_snapshot", boundary)

    await monitor._hydrate_stateful_state(late)

    assert 101 in monitor.skipped_epochs
    assert 101 not in monitor.initialized_epochs
    assert 101 not in monitor.initializing_epochs


@pytest.mark.asyncio
async def test_restart_skips_uninitialized_previous_epoch(monkeypatch):
    # Restarting inside epoch N+1 after epoch N was fully missed (down
    # across both boundaries) must skip N and keep the loop alive.
    from gateway.tasks import epoch_lifecycle, epoch_monitor

    cutover = _cutover()
    late = _snapshot(block=470, official_epoch=8, last_epoch_block=460)

    async def missing(*_args, **_kwargs):
        return None

    async def boundary(*_args, **_kwargs):
        return _snapshot(block=460, official_epoch=8, last_epoch_block=460)

    monkeypatch.setattr(epoch_monitor, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", missing)
    monitor = epoch_monitor.EpochMonitor()
    monkeypatch.setattr(monitor, "_find_stateful_transition_snapshot", boundary)

    await monitor._hydrate_stateful_state(late)

    assert {101, 102} <= monitor.skipped_epochs


@pytest.mark.asyncio
async def test_transition_after_skipped_epoch_initializes_next_epoch(
    monkeypatch,
):
    # The first boundary observed after a skipped epoch must initialize
    # the new epoch normally and must not schedule a close for the epoch
    # that was never initialized.
    from gateway.tasks import epoch_lifecycle, epoch_monitor

    cutover = _cutover()
    monitor = epoch_monitor.EpochMonitor()
    monitor.subtensor = object()
    monitor.last_official_epoch = 7
    monitor.last_epoch = 101
    monitor.last_epoch_snapshot = _snapshot(block=450)
    observed = _snapshot(
        block=462,
        official_epoch=8,
        last_epoch_block=460,
        observed_at="2026-07-16T13:11:36Z",
    )
    boundary = _snapshot(
        block=460,
        official_epoch=8,
        last_epoch_block=460,
        observed_at="2026-07-16T13:11:12Z",
    )
    calls = []

    async def find(*_args, **_kwargs):
        return boundary

    async def durable(*_args, **_kwargs):
        return None

    async def initialize(epoch_id, epoch_snapshot=None):
        calls.append(("initialize", epoch_id, epoch_snapshot.current_block))
        monitor.initialized_epochs.add(epoch_id)
        monitor.initializing_epochs.discard(epoch_id)

    monkeypatch.setattr(epoch_monitor, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", durable)
    monkeypatch.setattr(monitor, "_find_stateful_transition_snapshot", find)
    monkeypatch.setattr(monitor, "_on_epoch_start", initialize)

    await monitor._process_stateful_snapshot(observed)
    await asyncio.sleep(0)

    assert ("initialize", 102, 460) in calls
    assert 101 in monitor.skipped_epochs
    assert 101 not in monitor.pending_validation_ends


@pytest.mark.asyncio
async def test_multi_epoch_gap_reconciles_sequentially_when_inits_exist(
    monkeypatch,
):
    from gateway.tasks import epoch_lifecycle, epoch_monitor

    cutover = _cutover()
    monitor = epoch_monitor.EpochMonitor()
    monitor.last_official_epoch = 7
    monitor.last_epoch_snapshot = _snapshot(block=150)
    current = _snapshot(block=820, official_epoch=9, last_epoch_block=820)
    boundaries = {
        8: _snapshot(block=460, official_epoch=8, last_epoch_block=460),
        9: current,
    }
    closed = []

    async def find(target, *_args, **_kwargs):
        return boundaries[target]

    async def close(epoch_id, boundary):
        closed.append((epoch_id, boundary.subnet_epoch_index))

    async def durable(event_type, epoch_id, **_kwargs):
        if event_type == "EPOCH_INITIALIZATION" and epoch_id == 102:
            return {"id": 2, "payload": {"epoch_id": 102}}
        return None

    monkeypatch.setattr(epoch_monitor, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", durable)
    monkeypatch.setattr(monitor, "_find_stateful_transition_snapshot", find)
    monkeypatch.setattr(monitor, "_close_stateful_epoch_at_boundary", close)

    await monitor._reconcile_stateful_gap(current)

    # Epoch 101 has no durable initialization, so there is nothing to
    # close: it is skipped (nonfinalized) and only epoch 102 — whose
    # initialization is durable — is closed at its observed boundary.
    assert closed == [(102, 9)]
    assert 101 in monitor.skipped_epochs
    assert 102 in monitor.initialized_epochs


@pytest.mark.asyncio
async def test_normal_transition_uses_exact_boundary_and_durable_start(
    monkeypatch,
):
    from gateway.tasks import epoch_lifecycle, epoch_monitor

    cutover = _cutover()
    monitor = epoch_monitor.EpochMonitor()
    monitor.subtensor = object()
    monitor.last_official_epoch = 7
    monitor.last_epoch = 101
    monitor.last_epoch_snapshot = _snapshot(block=450)
    monitor.initialized_epochs.add(101)
    observed = _snapshot(
        block=462,
        official_epoch=8,
        last_epoch_block=460,
        observed_at="2026-07-16T13:11:36Z",
    )
    boundary = _snapshot(
        block=460,
        official_epoch=8,
        last_epoch_block=460,
        observed_at="2026-07-16T13:11:12Z",
    )
    calls = []

    async def find(*_args, **_kwargs):
        return boundary

    async def durable(event_type, epoch_id, **_kwargs):
        assert (event_type, epoch_id) == ("EPOCH_INITIALIZATION", 101)
        return {
            "id": 1,
            "payload": {
                "epoch_id": 101,
                "epoch_boundaries": {
                    "start_timestamp": "2026-07-16T12:00:00Z"
                },
            },
        }

    async def initialize(epoch_id, epoch_snapshot=None):
        calls.append(("initialize", epoch_id, epoch_snapshot.current_block))
        monitor.initialized_epochs.add(epoch_id)
        monitor.initializing_epochs.discard(epoch_id)

    async def validation_end(epoch_id, **kwargs):
        calls.append(
            (
                "end",
                epoch_id,
                kwargs["epoch_start"],
                kwargs["epoch_end"],
            )
        )
        return True

    monkeypatch.setattr(epoch_monitor, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setenv("LEADPOET_EPOCH_MODE", "stateful_v1")
    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", durable)
    monkeypatch.setattr(monitor, "_find_stateful_transition_snapshot", find)
    monkeypatch.setattr(monitor, "_on_epoch_start", initialize)
    monkeypatch.setattr(monitor, "_on_validation_end", validation_end)

    await monitor._process_stateful_snapshot(observed)
    await asyncio.sleep(0)

    assert ("initialize", 102, 460) in calls
    end_call = next(call for call in calls if call[0] == "end")
    assert end_call[1] == 101
    assert end_call[2] == datetime(2026, 7, 16, 12, 0, 0)
    assert end_call[3] == datetime(2026, 7, 16, 13, 11, 12)


@pytest.mark.asyncio
async def test_transition_search_uses_official_index_not_last_epoch_block(
    monkeypatch,
):
    from gateway.tasks import epoch_monitor

    cutover = _cutover()
    monitor = epoch_monitor.EpochMonitor()

    async def read(block):
        # Simulate a mid-epoch tempo reset: LastEpochBlock changes at 175 but
        # the official index does not advance until block 200.
        official = 7 if block < 200 else 8
        last_epoch = 175 if 175 <= block < 200 else (200 if block >= 200 else 100)
        return _snapshot(
            block=block,
            official_epoch=official,
            last_epoch_block=last_epoch,
        )

    monkeypatch.setattr(epoch_monitor, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(monitor, "_read_exact_stateful_snapshot", read)
    low = await read(150)
    high = await read(250)

    boundary = await monitor._find_stateful_transition_snapshot(
        8,
        high,
        low_snapshot=low,
    )

    assert boundary.current_block == 200
    assert boundary.subnet_epoch_index == 8


@pytest.mark.asyncio
async def test_validation_end_marks_memory_only_after_both_durable_events(
    monkeypatch,
):
    from gateway.tasks import epoch_lifecycle, epoch_monitor

    monitor = epoch_monitor.EpochMonitor()
    now = datetime(2026, 7, 16, 12, 0, 0)
    end_payloads = []

    async def log(_event_type, _epoch_id, payload):
        end_payloads.append(payload)
        return 1

    async def failed_inputs(*_args, **_kwargs):
        raise RuntimeError("inputs unavailable")

    authority = {"cutover_mapping_hash": "sha256:" + "a" * 64}

    async def initialization_authority(epoch_id):
        assert epoch_id == 101
        return authority

    monkeypatch.setattr(epoch_lifecycle, "log_epoch_event", log)
    monkeypatch.setattr(epoch_lifecycle, "compute_and_log_epoch_inputs", failed_inputs)
    monkeypatch.setattr(
        epoch_lifecycle,
        "load_stateful_epoch_initialization_authority",
        initialization_authority,
    )
    assert not await monitor._on_validation_end(
        101,
        epoch_start=now,
        epoch_end=now,
        epoch_close=now,
    )
    assert 101 not in monitor.validation_ended_epochs
    assert end_payloads[-1]["epoch_key_semantics"] == "settlement_ordinal"
    assert end_payloads[-1]["epoch_authority"] == authority

    inputs_kwargs = {}

    async def inputs(*_args, **kwargs):
        inputs_kwargs.update(kwargs)
        return 2

    monkeypatch.setattr(epoch_lifecycle, "compute_and_log_epoch_inputs", inputs)
    assert await monitor._on_validation_end(
        101,
        epoch_start=now,
        epoch_end=now,
        epoch_close=now,
    )
    assert 101 in monitor.validation_ended_epochs
    assert end_payloads[-1]["epoch_key_semantics"] == "settlement_ordinal"
    assert end_payloads[-1]["epoch_authority"] == authority
    assert inputs_kwargs["epoch_authority"] == authority
