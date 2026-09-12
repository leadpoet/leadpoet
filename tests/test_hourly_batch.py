"""Tests for the Arweave hourly batch task."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import types
from datetime import datetime
from pathlib import Path


logger_stub = types.ModuleType("gateway.utils.logger")
logger_stub.log_event = None
sys.modules.setdefault("gateway.utils.logger", logger_stub)

from gateway.tasks.hourly_batch import (
    build_arweave_checkpoint_log_event,
    serialize_and_compress_events,
)


def test_checkpoint_log_event_persists_arweave_tx_id_on_insert():
    header = {
        "checkpoint_number": 42,
        "event_count": 7,
        "merkle_root": "sha256:merkle",
        "time_range": {"start": "2026-07-03T12:00:00Z", "end": "2026-07-03T15:00:00Z"},
    }

    event = build_arweave_checkpoint_log_event(
        tx_id="arweave-tx-123",
        header=header,
        compressed_size_bytes=2048,
    )

    assert event["event_type"] == "ARWEAVE_CHECKPOINT"
    assert event["arweave_tx_id"] == "arweave-tx-123"
    assert event["payload"]["arweave_tx_id"] == "arweave-tx-123"
    assert event["payload"]["viewblock_url"].endswith("/arweave-tx-123")

    payload_json = json.dumps(event["payload"], sort_keys=True, default=str)
    assert event["payload_hash"] == hashlib.sha256(payload_json.encode()).hexdigest()


def test_serialize_and_compress_events_round_trips():
    events = [
        {"sequence": 1, "event_type": "LEAD_ACCEPTED"},
        {"sequence": 2, "event_type": "ARWEAVE_CHECKPOINT"},
    ]

    events_bytes, compressed = serialize_and_compress_events(events)

    assert json.loads(events_bytes.decode("utf-8")) == events
    assert json.loads(gzip.decompress(compressed).decode("utf-8")) == events


def test_serialize_and_compress_events_stringifies_non_json_values():
    events = [{"sequence": 1, "at": datetime(2026, 9, 9, 12, 0, 0)}]

    events_bytes, compressed = serialize_and_compress_events(events)

    assert b"2026-09-09 12:00:00" in events_bytes
    assert gzip.decompress(compressed) == events_bytes


def test_checkpoint_compression_is_offloaded_from_the_event_loop():
    """A full buffer is seconds of CPU, and this task shares the request loop."""
    source = (
        Path(__file__).resolve().parent.parent
        / "gateway"
        / "tasks"
        / "hourly_batch.py"
    ).read_text(encoding="utf-8")
    body = source.split("async def hourly_batch_task")[1]

    assert "asyncio.to_thread(" in body
    assert "serialize_and_compress_events, events" in body
    assert "gzip.compress(" not in body
    assert "json.dumps(events" not in body
