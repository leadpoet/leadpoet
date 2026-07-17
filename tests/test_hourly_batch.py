"""Tests for the Arweave hourly batch task."""

from __future__ import annotations

import hashlib
import json
import sys
import types


logger_stub = types.ModuleType("gateway.utils.logger")
logger_stub.log_event = None
sys.modules.setdefault("gateway.utils.logger", logger_stub)

from gateway.tasks.hourly_batch import build_arweave_checkpoint_log_event


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
