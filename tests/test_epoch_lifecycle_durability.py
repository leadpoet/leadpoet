from __future__ import annotations

from datetime import datetime, timedelta
import hashlib

import pytest


class _Result:
    def __init__(self, data):
        self.data = data


class _PagedTable:
    def __init__(self, rows):
        self.rows = rows
        self.ranges = []
        self._range = (0, len(rows) - 1)

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def gte(self, *_args, **_kwargs):
        return self

    def lte(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def range(self, start, end):
        self._range = (start, end)
        self.ranges.append(self._range)
        return self

    def execute(self):
        start, end = self._range
        return _Result(self.rows[start : end + 1])


class _FilteredTable(_PagedTable):
    def __init__(self, rows):
        super().__init__(rows)
        self.filters = []

    def select(self, *_args, **_kwargs):
        self.filters = []
        return self

    def eq(self, field, value):
        self.filters.append((field, value))
        return self

    def execute(self):
        rows = list(self.rows)
        for field, value in self.filters:
            if field == "event_type":
                rows = [row for row in rows if row.get("event_type") == value]
            elif field.startswith("payload->>"):
                key = field.removeprefix("payload->>")
                rows = [
                    row
                    for row in rows
                    if str((row.get("payload") or {}).get(key)) == value
                ]
        start, end = self._range
        return _Result(rows[start : end + 1])


class _Client:
    def __init__(self, table):
        self._table = table

    def table(self, name):
        assert name in {"transparency_log", "leads_private"}
        return self._table


def _stateful_authority(epoch_id: int):
    from Leadpoet.utils.subnet_epoch import (
        SubnetEpochCutover,
        SubnetEpochSnapshot,
    )

    cutover = SubnetEpochCutover(
        network_genesis_hash="0x" + "11" * 32,
        netuid=71,
        cutover_block=100,
        cutover_block_hash="0x" + "22" * 32,
        first_subnet_epoch_index=7,
        first_settlement_epoch_id=epoch_id,
        last_legacy_epoch_id=epoch_id - 1,
    )
    snapshot = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=71,
        head_kind="finalized",
        block_hash=cutover.cutover_block_hash,
        current_block=104,
        last_epoch_block=100,
        pending_epoch_at=0,
        subnet_epoch_index=7,
        tempo=360,
        blocks_since_last_step=4,
        observed_at="2026-07-16T12:00:00Z",
    )
    return cutover, snapshot.to_dict(cutover=cutover)


@pytest.mark.asyncio
async def test_durable_event_lookup_paginates_and_rejects_duplicates(monkeypatch):
    from gateway.tasks import epoch_lifecycle

    rows = [
        {
            "id": index,
            "event_type": "EPOCH_END",
            "payload": {
                "epoch_id": 77,
                "epoch_key_semantics": "settlement_ordinal",
            },
        }
        for index in range(1001)
    ]
    table = _PagedTable(rows)
    monkeypatch.setattr(epoch_lifecycle, "supabase", _Client(table))

    with pytest.raises(RuntimeError, match="duplicate EPOCH_END"):
        await epoch_lifecycle.get_durable_epoch_event("EPOCH_END", 77)

    assert table.ranges == [(0, 999), (1000, 1999)]


@pytest.mark.asyncio
async def test_stateful_durable_lookup_ignores_legacy_numeric_collision(monkeypatch):
    from gateway.tasks import epoch_lifecycle

    rows = [
        {
            "id": 1,
            "event_type": "EPOCH_INITIALIZATION",
            "payload": {"epoch_id": 77, "assignment": {"legacy": True}},
        },
        {
            "id": 2,
            "event_type": "EPOCH_INITIALIZATION",
            "payload": {
                "epoch_id": 77,
                "epoch_key_semantics": "settlement_ordinal",
                "assignment": {"assigned_lead_ids": ["lead-1"]},
            },
        },
    ]
    table = _FilteredTable(rows)
    monkeypatch.setattr(epoch_lifecycle, "supabase", _Client(table))

    result = await epoch_lifecycle.get_durable_epoch_event(
        "EPOCH_INITIALIZATION", 77
    )

    assert result["id"] == 2
    assert result["payload"]["epoch_key_semantics"] == "settlement_ordinal"


@pytest.mark.asyncio
async def test_log_epoch_event_propagates_failed_durable_write(monkeypatch):
    from gateway.tasks import epoch_lifecycle
    from gateway.utils import logger

    async def missing(*_args, **_kwargs):
        return None

    async def failed_write(_entry):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", missing)
    monkeypatch.setattr(logger, "log_event", failed_write)
    _cutover, authority = _stateful_authority(88)
    monkeypatch.setattr(
        epoch_lifecycle,
        "load_subnet_epoch_cutover",
        lambda: _cutover,
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        await epoch_lifecycle.log_epoch_event(
            "EPOCH_END",
            88,
            {
                "epoch_id": 88,
                "epoch_key_semantics": "settlement_ordinal",
                "epoch_authority": authority,
                "phase": "epoch_ended",
            },
        )


@pytest.mark.asyncio
async def test_stateful_log_event_requires_exact_durable_readback(monkeypatch):
    from gateway.tasks import epoch_lifecycle
    from gateway.utils import logger

    monkeypatch.setenv("LEADPOET_EPOCH_MODE", "stateful_v1")
    cutover, authority = _stateful_authority(89)
    monkeypatch.setattr(
        epoch_lifecycle,
        "load_subnet_epoch_cutover",
        lambda: cutover,
    )
    payload = {
        "epoch_id": 89,
        "epoch_key_semantics": "settlement_ordinal",
        "epoch_authority": authority,
        "phase": "epoch_ended",
    }
    reads = 0

    async def durable(_event_type, _epoch_id, *, expected_payload=None):
        nonlocal reads
        reads += 1
        assert expected_payload == payload
        if reads == 1:
            return None
        return {"id": 44, "payload": payload, "tee_sequence": 12}

    written = {}

    async def write(entry):
        written.update(entry)
        return {"status": "buffered", "sequence": 12}

    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", durable)
    monkeypatch.setattr(logger, "log_event", write)

    assert await epoch_lifecycle.log_epoch_event(
        "EPOCH_END",
        89,
        payload,
    ) == 12
    assert written["nonce"] == epoch_lifecycle._durable_epoch_nonce(
        "EPOCH_END",
        89,
    )
    assert reads == 2


@pytest.mark.asyncio
async def test_stateful_epoch_inputs_include_settlement_key_semantics(monkeypatch):
    from gateway.tasks import epoch_lifecycle

    monkeypatch.setenv("LEADPOET_EPOCH_MODE", "stateful_v1")
    monkeypatch.setattr(epoch_lifecycle, "supabase", _Client(_PagedTable([])))
    _cutover, authority = _stateful_authority(90)

    async def missing(*_args, **_kwargs):
        return None

    async def capture(_event_type, _epoch_id, payload):
        capture.payload = payload
        return 1

    async def initialization_authority(_epoch_id):
        assert _epoch_id == 90
        return authority

    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", missing)
    monkeypatch.setattr(
        epoch_lifecycle,
        "load_stateful_epoch_initialization_authority",
        initialization_authority,
    )
    monkeypatch.setattr(epoch_lifecycle, "log_epoch_event", capture)
    now = datetime(2026, 7, 16, 12, 0, 0)

    await epoch_lifecycle.compute_and_log_epoch_inputs(
        90,
        epoch_start=now,
        epoch_end=now + timedelta(minutes=72),
    )

    assert capture.payload["epoch_key_semantics"] == "settlement_ordinal"
    assert capture.payload["epoch_authority"] == authority


@pytest.mark.asyncio
async def test_stateful_lifecycle_authority_requires_exact_initialization(monkeypatch):
    from gateway.tasks import epoch_lifecycle

    cutover, authority = _stateful_authority(91)
    monkeypatch.setattr(
        epoch_lifecycle,
        "load_subnet_epoch_cutover",
        lambda: cutover,
    )

    async def missing(*_args, **_kwargs):
        return None

    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", missing)
    with pytest.raises(RuntimeError, match="EPOCH_INITIALIZATION.*missing"):
        await epoch_lifecycle.load_stateful_epoch_initialization_authority(91)

    async def present(*_args, **_kwargs):
        return {
            "payload": {
                "epoch_id": 91,
                "epoch_key_semantics": "settlement_ordinal",
                "epoch_authority": authority,
            }
        }

    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", present)
    assert (
        await epoch_lifecycle.load_stateful_epoch_initialization_authority(91)
    ) == authority

    tampered = dict(authority)
    tampered["cutover_mapping_hash"] = "sha256:" + "f" * 64

    async def conflicting(*_args, **_kwargs):
        return {
            "payload": {
                "epoch_id": 91,
                "epoch_key_semantics": "settlement_ordinal",
                "epoch_authority": tampered,
            }
        }

    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", conflicting)
    with pytest.raises(RuntimeError, match="not canonical|active cutover"):
        await epoch_lifecycle.load_stateful_epoch_initialization_authority(91)


@pytest.mark.asyncio
async def test_epoch_inputs_hashes_every_paginated_event_and_propagates(monkeypatch):
    from gateway.tasks import epoch_lifecycle

    rows = [
        {
            "id": index,
            "event_type": "VALIDATION_RESULT_BATCH",
            "payload_hash": hashlib.sha256(str(index).encode()).hexdigest(),
        }
        for index in range(1001)
    ]
    table = _PagedTable(rows)
    monkeypatch.setattr(epoch_lifecycle, "supabase", _Client(table))

    async def missing(*_args, **_kwargs):
        return None

    captured = {}

    async def capture(event_type, epoch_id, payload):
        captured.update(
            event_type=event_type,
            epoch_id=epoch_id,
            payload=payload,
        )
        return 9

    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", missing)
    monkeypatch.setattr(epoch_lifecycle, "log_epoch_event", capture)
    _cutover, authority = _stateful_authority(99)

    async def initialization_authority(epoch_id):
        if epoch_id not in {99, 100}:
            raise AssertionError(epoch_id)
        return authority

    monkeypatch.setattr(
        epoch_lifecycle,
        "load_stateful_epoch_initialization_authority",
        initialization_authority,
    )
    start = datetime(2026, 7, 16, 12, 0, 0)
    end = start + timedelta(minutes=72)

    assert await epoch_lifecycle.compute_and_log_epoch_inputs(
        99,
        epoch_start=start,
        epoch_end=end,
    ) == 9
    assert table.ranges == [(0, 999), (1000, 1999)]
    assert captured["event_type"] == "EPOCH_INPUTS"
    assert captured["payload"]["event_count"] == 1001
    expected = hashlib.sha256(
        "".join(row["payload_hash"] for row in rows).encode()
    ).hexdigest()
    assert captured["payload"]["inputs_hash"] == expected

    async def rejected(*_args, **_kwargs):
        raise RuntimeError("write rejected")

    monkeypatch.setattr(epoch_lifecycle, "log_epoch_event", rejected)
    with pytest.raises(RuntimeError, match="write rejected"):
        await epoch_lifecycle.compute_and_log_epoch_inputs(
            100,
            epoch_start=start,
            epoch_end=end,
        )


@pytest.mark.asyncio
async def test_initialization_keeps_legacy_end_block_and_stateful_expected_end(
    monkeypatch,
):
    from Leadpoet.utils.subnet_epoch import (
        SubnetEpochCutover,
        SubnetEpochSnapshot,
    )
    from gateway.tasks import epoch_lifecycle
    from gateway.utils import assignment

    cutover = SubnetEpochCutover(
        network_genesis_hash="0x" + "11" * 32,
        netuid=71,
        cutover_block=100,
        cutover_block_hash="0x" + "22" * 32,
        first_subnet_epoch_index=7,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )
    snapshot = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=71,
        head_kind="finalized",
        block_hash=cutover.cutover_block_hash,
        current_block=104,
        last_epoch_block=100,
        pending_epoch_at=0,
        subnet_epoch_index=7,
        tempo=360,
        blocks_since_last_step=4,
        observed_at="2026-07-16T12:00:00Z",
    )

    async def missing(*_args, **_kwargs):
        return None

    async def validators(
        _epoch_id,
        *,
        fail_closed=False,
        metagraph_cache_epoch_id=None,
    ):
        assert fail_closed is True
        assert metagraph_cache_epoch_id == 101
        return ["validator"]

    async def capture(_event_type, _epoch_id, payload):
        capture.payload = payload
        return 1

    monkeypatch.setattr(epoch_lifecycle, "get_durable_epoch_event", missing)
    monkeypatch.setattr(epoch_lifecycle, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(epoch_lifecycle, "supabase", _Client(_PagedTable([])))
    monkeypatch.setattr(assignment, "get_validator_set", validators)
    monkeypatch.setattr(epoch_lifecycle, "log_epoch_event", capture)
    start = datetime(2026, 7, 16, 11, 59, 12)
    end = datetime(2026, 7, 16, 13, 11, 12)

    await epoch_lifecycle.compute_and_log_epoch_initialization(
        101,
        start,
        end,
        end,
        epoch_snapshot=snapshot,
    )

    boundaries = capture.payload["epoch_boundaries"]
    assert boundaries["end_block"] == 460
    assert boundaries["expected_end_block"] == 460
    assert capture.payload["epoch_authority"]["settlement_epoch_id"] == 101
