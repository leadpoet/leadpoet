#!/usr/bin/env python3
"""Local contract checks for Research Lab Arweave audit payloads."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.research_lab import arweave_audit
from gateway.research_lab.bundles import sha256_json


async def main() -> int:
    original_select_many = arweave_audit.select_many
    original_existing_anchor = arweave_audit._existing_anchor_for_payload
    original_create_event = arweave_audit.create_arweave_epoch_audit_anchor_event
    try:
        arweave_audit.select_many = _fake_select_many  # type: ignore[assignment]
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
        assert payload["score_bundles"][0]["aggregates"]["mean_delta"] == 1.25
        assert payload["observability"]["champion_reward_count"] == 1

        secret_row = _allocation()
        secret_row["allocation_doc"]["reimbursement_allocations"][0]["proxy_url"] = "http://user:pass@example.test:8080"
        async def fake_secret_select_many(table: str, **kwargs: Any) -> list[dict[str, Any]]:
            if table == "research_lab_emission_allocation_current":
                return [secret_row]
            return await _fake_select_many(table, **kwargs)
        arweave_audit.select_many = fake_secret_select_many  # type: ignore[assignment]
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
            return {"anchor_id": "research_lab_arweave_anchor:" + "2" * 64}
        async def fake_create_event(**kwargs: Any) -> dict[str, Any]:
            captured_events.append(dict(kwargs))
            return dict(kwargs)
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
        assert captured_events[0]["event_doc"]["arweave_tx_id"] == "arweave_tx_fixture"

    finally:
        arweave_audit.select_many = original_select_many  # type: ignore[assignment]
        arweave_audit._existing_anchor_for_payload = original_existing_anchor  # type: ignore[assignment]
        arweave_audit.create_arweave_epoch_audit_anchor_event = original_create_event  # type: ignore[assignment]

    print("Research Lab Arweave audit contract verified")
    return 0


async def _fake_select_many(table: str, **kwargs: Any) -> list[dict[str, Any]]:
    if table == "research_lab_signed_audit_bundle_current":
        return [_audit_bundle()]
    if table == "research_lab_emission_allocation_current":
        return [_allocation()]
    if table == "research_evaluation_score_bundle_current":
        return [_score_bundle()]
    if table == "research_lab_private_model_benchmark_current":
        return [_benchmark()]
    if table == "research_lab_rolling_icp_windows":
        return [_window()]
    if table == "research_lab_private_model_version_current":
        return [_model_version()]
    if table == "research_lab_candidate_promotion_events":
        return [_promotion()]
    if table == "research_lab_public_benchmark_report_current":
        return [_public_report()]
    if table == "research_lab_champion_reward_current":
        return [_champion()]
    if table == "research_reimbursement_award_current":
        return [_reimbursement()]
    return []


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


def _audit_bundle() -> dict[str, Any]:
    return {
        "audit_bundle_id": "research_lab_audit:" + "a" * 64,
        "audit_bundle_hash": "sha256:" + "a" * 64,
        "epoch": 123,
        "signature_ref": "kms:signature",
        "current_audit_status": "created",
        "current_event_hash": "sha256:" + "b" * 64,
        "anchored_hash": "sha256:" + "a" * 64,
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


def _score_bundle() -> dict[str, Any]:
    return {
        "score_bundle_id": "score_bundle:" + "6" * 64,
        "run_id": "run",
        "ticket_id": "ticket",
        "miner_hotkey": "hk2",
        "island": "generalist",
        "evaluation_epoch": 123,
        "bundle_status": "scored",
        "score_bundle_hash": "sha256:" + "6" * 64,
        "anchored_hash": "sha256:" + "6" * 64,
        "score_bundle_doc": {"aggregates": {"mean_delta": 1.25, "icp_count": 50}},
    }


def _benchmark() -> dict[str, Any]:
    return {"benchmark_bundle_id": "private_benchmark:" + "7" * 64, "aggregate_score": 50}


def _window() -> dict[str, Any]:
    return {"rolling_window_hash": "sha256:" + "8" * 64, "selected_icp_count": 50}


def _model_version() -> dict[str, Any]:
    return {"private_model_version_id": "private_model_version:" + "9" * 64, "version_hash": "sha256:" + "9" * 64}


def _promotion() -> dict[str, Any]:
    return {"promotion_event_id": "promotion", "promotion_status": "promoted", "improvement_points": 1.25}


def _public_report() -> dict[str, Any]:
    return {"report_id": "public_benchmark:" + "c" * 64, "report_doc": {"summary": "sanitized"}}


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
