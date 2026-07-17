from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from gateway.research_lab import arweave_audit
from gateway.research_lab.bundles import sha256_json
from gateway.utils import tee_client as tee_client_module


def _signed_fixture(*, epoch: int = 23706, netuid: int = 71):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    payload = {
        "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
        "epoch": epoch,
        "netuid": netuid,
        "audit_kind": "active",
        "actor_hotkey": "validator",
    }
    signed_event = {
        "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
        "timestamp": "2026-07-17T00:00:00Z",
        "boot_id": "boot",
        "monotonic_seq": 10,
        "prev_event_hash": "0" * 64,
        "payload": payload,
    }
    event_hash = hashlib.sha256(
        json.dumps(
            signed_event,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    signed_log_entry = {
        "signed_event": signed_event,
        "event_hash": event_hash,
        "enclave_pubkey": pubkey,
        "enclave_signature": key.sign(bytes.fromhex(event_hash)).hex(),
    }
    anchor = {
        "anchor_id": "research_lab_arweave_anchor:" + "a" * 64,
        "epoch": epoch,
        "netuid": netuid,
        "audit_kind": "active",
        "payload_hash": sha256_json(payload),
        "transparency_event_hash": event_hash,
        "current_transparency_event_hash": event_hash,
        "current_anchor_status": "checkpointed",
        "current_arweave_tx_id": "x" * 43,
        "current_checkpoint_number": 41,
        "current_checkpoint_merkle_root": "b" * 64,
        "current_status_at": "2026-07-17T00:00:00Z",
    }
    log_row = {
        "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
        "event_hash": event_hash,
        "payload_hash": sha256_json(payload),
        "enclave_pubkey": pubkey,
        "signed_log_entry": signed_log_entry,
    }
    return anchor, log_row


class _FakeTeeClient:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def get_buffer(self):
        return list(self.events)

    async def append_event(self, event):
        self.events.append(dict(event))
        return {"status": "buffered", "sequence": len(self.events)}


@pytest.mark.asyncio
async def test_selected_checkpoint_recovery_reuses_exact_signed_event(
    monkeypatch,
):
    anchor, log_row = _signed_fixture()
    fake_tee = _FakeTeeClient()
    lifecycle_events: list[dict[str, Any]] = []

    async def fake_select_many(table, **kwargs):
        assert table == "research_lab_arweave_epoch_audit_anchor_current"
        return [anchor]

    async def fake_select_one(table, **kwargs):
        assert table == "transparency_log"
        return log_row

    async def fake_create_event(**kwargs):
        lifecycle_events.append(dict(kwargs))
        return {"seq": 3, **kwargs}

    monkeypatch.setattr(arweave_audit, "select_many", fake_select_many)
    monkeypatch.setattr(arweave_audit, "select_one", fake_select_one)
    monkeypatch.setattr(
        arweave_audit,
        "create_arweave_epoch_audit_anchor_event",
        fake_create_event,
    )
    monkeypatch.setattr(tee_client_module, "tee_client", fake_tee)

    dry_run = (
        await arweave_audit.recover_research_lab_checkpointed_audit_epochs(
            epochs=[23706],
            netuid=71,
            dry_run=True,
        )
    )
    assert dry_run["ok"] is True
    assert dry_run["planned_count"] == 1
    assert fake_tee.events == []
    assert lifecycle_events == []

    result = (
        await arweave_audit.recover_research_lab_checkpointed_audit_epochs(
            epochs=[23706],
            netuid=71,
            dry_run=False,
        )
    )
    assert result["ok"] is True
    assert result["recovered_count"] == 1
    assert fake_tee.events == [
        {
            "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
            "event_hash": log_row["event_hash"],
            "payload_hash": log_row["payload_hash"],
            "signed_log_entry": log_row["signed_log_entry"],
        }
    ]
    assert lifecycle_events[0]["anchor_status"] == "buffered"
    assert lifecycle_events[0]["event_doc"] == {
        "reason": "operator_selected_historical_recheckpoint",
        "previous_anchor_status": "checkpointed",
        "previous_arweave_tx_id": "x" * 43,
        "previous_checkpoint_number": 41,
        "signed_event_reused": True,
    }


@pytest.mark.asyncio
async def test_selected_checkpoint_recovery_rejects_tampered_signature(
    monkeypatch,
):
    anchor, log_row = _signed_fixture()
    log_row["signed_log_entry"] = {
        **log_row["signed_log_entry"],
        "enclave_signature": "0" * 128,
    }
    fake_tee = _FakeTeeClient()

    async def fake_select_many(table, **kwargs):
        return [anchor]

    async def fake_select_one(table, **kwargs):
        return log_row

    async def fail_create_event(**kwargs):
        raise AssertionError("invalid history must not be rebuffered")

    monkeypatch.setattr(arweave_audit, "select_many", fake_select_many)
    monkeypatch.setattr(arweave_audit, "select_one", fake_select_one)
    monkeypatch.setattr(
        arweave_audit,
        "create_arweave_epoch_audit_anchor_event",
        fail_create_event,
    )
    monkeypatch.setattr(tee_client_module, "tee_client", fake_tee)

    result = (
        await arweave_audit.recover_research_lab_checkpointed_audit_epochs(
            epochs=[23706],
            netuid=71,
            dry_run=False,
        )
    )
    assert result["ok"] is False
    assert result["blocked"] == [
        {
            "epoch": 23706,
            "reason": "signed_transparency_event_invalid",
            "error": "signed transparency-log entry is invalid",
        }
    ]
    assert fake_tee.events == []
