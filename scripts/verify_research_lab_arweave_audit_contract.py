#!/usr/bin/env python3
"""Local contract checks for Research Lab Arweave audit payloads."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
import sys
import types
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.research_lab.bundles import sha256_json
from leadpoet_canonical.events import compute_event_hash

store_stub = types.ModuleType("gateway.research_lab.store")
store_stub.canonical_hash = lambda payload: sha256_json(payload)  # type: ignore[attr-defined]
store_stub.create_arweave_epoch_audit_anchor = None
store_stub.create_arweave_epoch_audit_anchor_event = None
store_stub.select_all = None
store_stub.select_many = None
store_stub.select_one = None
sys.modules.setdefault("gateway.research_lab.store", store_stub)

from gateway.research_lab import arweave_audit


def _signed_rebuffer_fixture() -> tuple[dict[str, Any], str, str]:
    payload = {
        "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
        "epoch": 123,
        "netuid": 401,
        "audit_kind": "shadow",
    }
    signed_event = {
        "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
        "timestamp": "2026-01-01T00:00:00Z",
        "boot_id": "boot",
        "monotonic_seq": 1,
        "prev_event_hash": None,
        "payload": payload,
    }
    private_key = Ed25519PrivateKey.generate()
    event_hash = compute_event_hash(signed_event)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    signed_log_entry = {
        "signed_event": signed_event,
        "event_hash": event_hash,
        "enclave_pubkey": public_key,
        "enclave_signature": private_key.sign(bytes.fromhex(event_hash)).hex(),
    }
    payload_hash = sha256_json(payload)
    return signed_log_entry, event_hash, payload_hash


_REBUFFER_LOG_ENTRY, _REBUFFER_EVENT_HASH, _REBUFFER_PAYLOAD_HASH = _signed_rebuffer_fixture()


async def main() -> int:
    original_select_many = arweave_audit.select_many
    original_select_all = arweave_audit.select_all
    original_select_one = arweave_audit.select_one
    original_existing_anchor = arweave_audit._existing_anchor_for_payload
    original_create_event = arweave_audit.create_arweave_epoch_audit_anchor_event
    try:
        arweave_audit.select_many = _fake_select_many  # type: ignore[assignment]
        arweave_audit.select_all = _fake_select_many  # type: ignore[assignment]
        payload = await arweave_audit.build_research_lab_epoch_audit_payload(
            epoch=123,
            netuid=401,
            audit_kind="shadow",
            weight_bundle=_weight_bundle(),
        )
        expected_hash = sha256_json({key: value for key, value in payload.items() if key != "payload_hash"})
        assert payload["payload_hash"] == expected_hash
        assert payload["weights"]["weights_hash"] == "weights:abc"
        assert payload["lab_allocation"]["allocation_hash"] == "sha256:" + "1" * 64
        assert payload["lab_allocation"]["allocations"]["reimbursements"][0]["miner_hotkey"] == "hk1"
        assert payload["observability"]["champion_reward_count"] == 1
        assert "audit_bundle" not in payload
        assert "private_baseline_benchmarks" not in payload
        assert "private_model_versions" not in payload

        secret_row = _allocation()
        secret_row["allocation_doc"]["reimbursement_allocations"][0]["proxy_url"] = "http://user:pass@example.test:8080"
        async def fake_secret_select_many(table: str, **kwargs: Any) -> list[dict[str, Any]]:
            if table == "research_lab_emission_allocation_current":
                return [secret_row]
            return await _fake_select_many(table, **kwargs)
        arweave_audit.select_many = fake_secret_select_many  # type: ignore[assignment]
        arweave_audit.select_all = _fake_select_many  # type: ignore[assignment]
        try:
            await arweave_audit.build_research_lab_epoch_audit_payload(
                epoch=123,
                netuid=401,
                audit_kind="shadow",
                weight_bundle=_weight_bundle(),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("secret material was not rejected")

        captured_events: list[dict[str, Any]] = []
        async def fake_existing_anchor(**kwargs: Any) -> dict[str, Any] | None:
            raise AssertionError(f"payload-hash fallback should not be needed: {kwargs}")
        async def fake_anchor_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
            assert table == "research_lab_arweave_epoch_audit_anchor_current"
            assert ("current_transparency_event_hash", "e" * 64) in kwargs["filters"]
            return {"anchor_id": "research_lab_arweave_anchor:" + "2" * 64}
        async def fake_create_event(**kwargs: Any) -> dict[str, Any]:
            captured_events.append(dict(kwargs))
            return dict(kwargs)
        arweave_audit.select_one = fake_anchor_select_one  # type: ignore[assignment]
        arweave_audit._existing_anchor_for_payload = fake_existing_anchor  # type: ignore[assignment]
        arweave_audit.create_arweave_epoch_audit_anchor_event = fake_create_event  # type: ignore[assignment]
        recorded = await arweave_audit.record_research_lab_checkpointed_events(
            events=[_tee_event(payload)],
            header={
                "checkpoint_number": 7,
                "merkle_root": "3" * 64,
                "sequence_range": {"first": 10, "last": 11},
                "event_count": 1,
            },
            arweave_tx_id="arweave_tx_fixture",
        )
        assert recorded == 1
        assert captured_events[0]["event_type"] == "checkpointed"
        assert captured_events[0]["arweave_tx_id"] == "arweave_tx_fixture"
        assert captured_events[0]["checkpoint_number"] == 7
        assert captured_events[0]["checkpoint_merkle_root"] == "3" * 64
        assert captured_events[0]["tee_sequence"] == 11
        assert captured_events[0]["transparency_event_hash"] == "e" * 64
        assert captured_events[0]["event_doc"]["arweave_tx_id"] == "arweave_tx_fixture"

        appended_events: list[dict[str, Any]] = []
        arweave_audit.select_many = _fake_rebuffer_select_many  # type: ignore[assignment]
        arweave_audit.select_one = _fake_rebuffer_select_one  # type: ignore[assignment]
        tee_stub = types.SimpleNamespace(
            get_buffer=lambda: _fake_get_buffer([]),
            append_event=lambda event: _capture_append_event(appended_events, event)
        )
        sys.modules["gateway.utils.tee_client"] = types.SimpleNamespace(tee_client=tee_stub)
        rebuffered = await arweave_audit.rebuffer_research_lab_buffered_audit_events(limit=10)
        assert rebuffered == 1
        assert appended_events == [
            {
                "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
                "event_hash": _REBUFFER_EVENT_HASH,
                "payload_hash": _REBUFFER_PAYLOAD_HASH,
                "signed_log_entry": _REBUFFER_LOG_ENTRY,
            }
        ]

        appended_events.clear()
        tee_stub = types.SimpleNamespace(
            get_buffer=lambda: _fake_get_buffer([{"event_hash": _REBUFFER_EVENT_HASH}]),
            append_event=lambda event: _capture_append_event(appended_events, event),
        )
        sys.modules["gateway.utils.tee_client"] = types.SimpleNamespace(tee_client=tee_stub)
        rebuffered = await arweave_audit.rebuffer_research_lab_buffered_audit_events(limit=10)
        assert rebuffered == 0
        assert appended_events == []

    finally:
        arweave_audit.select_many = original_select_many  # type: ignore[assignment]
        arweave_audit.select_all = original_select_all  # type: ignore[assignment]
        arweave_audit.select_one = original_select_one  # type: ignore[assignment]
        arweave_audit._existing_anchor_for_payload = original_existing_anchor  # type: ignore[assignment]
        arweave_audit.create_arweave_epoch_audit_anchor_event = original_create_event  # type: ignore[assignment]

    print("Research Lab Arweave audit contract verified")
    return 0


async def _fake_select_many(table: str, **kwargs: Any) -> list[dict[str, Any]]:
    if table == "research_lab_emission_allocation_current":
        return [_allocation()]
    if table == "research_lab_champion_reward_current":
        return [_champion()]
    if table == "research_reimbursement_award_current":
        return [_reimbursement()]
    return []


async def _fake_rebuffer_select_many(table: str, **kwargs: Any) -> list[dict[str, Any]]:
    if table == "research_lab_arweave_epoch_audit_anchor_current":
        return [
            {
                "anchor_id": "research_lab_arweave_anchor:" + "f" * 64,
                "epoch": 123,
                "netuid": 401,
                "audit_kind": "shadow",
                "payload_hash": _REBUFFER_PAYLOAD_HASH,
                "current_transparency_event_hash": _REBUFFER_EVENT_HASH,
                "current_anchor_status": "buffered",
            }
        ]
    return []


async def _fake_rebuffer_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
    if table == "transparency_log":
        return {
            "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
            "event_hash": _REBUFFER_EVENT_HASH,
            "payload_hash": _REBUFFER_PAYLOAD_HASH,
            "enclave_pubkey": _REBUFFER_LOG_ENTRY["enclave_pubkey"],
            "signed_log_entry": _REBUFFER_LOG_ENTRY,
        }
    return None


async def _capture_append_event(appended_events: list[dict[str, Any]], event: dict[str, Any]) -> dict[str, Any]:
    appended_events.append(dict(event))
    return {"status": "buffered", "sequence": len(appended_events)}


async def _fake_get_buffer(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return events


def _weight_bundle() -> dict[str, Any]:
    return {
        "netuid": 401,
        "epoch_id": 123,
        "block": 456,
        "weights_hash": "weights:abc",
        "validator_hotkey": "validator_hk",
        "validator_pcr0": "pcr0",
        "pcr0_commit_hash": "commit",
        "chain_snapshot_compare_hash": "chain_hash",
        "weight_submission_event_hash": "4" * 64,
    }


def _allocation() -> dict[str, Any]:
    return {
        "allocation_id": "lab_allocation:sha256:" + "1" * 64,
        "epoch": 123,
        "netuid": 401,
        "policy_id": "policy",
        "snapshot_status": "shadow",
        "lab_cap_alpha_percent": 10,
        "reimbursement_alpha_percent": 1,
        "champion_alpha_percent": 9,
        "queued_champion_alpha_percent": 0,
        "unallocated_alpha_percent": 0,
        "input_hash": "sha256:" + "5" * 64,
        "allocation_hash": "sha256:" + "1" * 64,
        "allocation_doc": {
            "reimbursement_allocations": [{"uid": 1, "miner_hotkey": "hk1", "paid_alpha_percent": "1"}],
            "champion_allocations": [{"uid": 2, "miner_hotkey": "hk2", "paid_alpha_percent": "9"}],
            "queued_champion_allocations": [],
        },
    }


def _champion() -> dict[str, Any]:
    return {"champion_reward_id": "champion_reward:sha256:" + "d" * 64, "miner_hotkey": "hk2"}


def _reimbursement() -> dict[str, Any]:
    return {"award_id": "award", "miner_hotkey": "hk1", "target_reimbursement_usd": 5}


def _tee_event(payload: dict[str, Any]) -> dict[str, Any]:
    signed_event = {
        "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
        "timestamp": "2026-01-01T00:00:00Z",
        "boot_id": "boot",
        "monotonic_seq": 1,
        "prev_event_hash": None,
        "payload": copy.deepcopy(payload),
    }
    return {
        "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
        "sequence": 11,
        "signed_log_entry": {
            "signed_event": signed_event,
            "event_hash": "e" * 64,
            "enclave_pubkey": "pub",
            "enclave_signature": "sig",
        },
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
