import random

import numpy as np
import pytest
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    WeightComputationError,
    compute_final_weights,
    normalize_to_u16_with_uids_pure,
    weight_config_hash,
)
from leadpoet_canonical.attested_receipts import WEIGHT_PURPOSE, validate_signed_receipt


BURN_HOTKEY = "burn-hotkey"


def _snapshot(**overrides):
    value = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 100,
        "block": 36099,
        "commit_sha": "a" * 40,
        "config_hash": "",
        "parent_receipt_hashes": [],
        "research_lab_allocation_receipt_hash": "",
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": BURN_HOTKEY,
        "metagraph_hotkeys": [BURN_HOTKEY, "fulfillment-hotkey", "lab-hotkey", "source-hotkey"],
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
        "ff_enabled": True,
        "base_burn_share": 0.0,
        "champion_share": 0.0,
        "champion_uid": None,
        "effective_champion_share": 0.0,
        "research_lab_fallback_share": 0.2,
        "research_lab_allocation_doc": {
            "lab_cap_percent": 20.0,
            "unallocated_percent": 15.0,
            "reimbursement_allocations": [],
            "champion_allocations": [
                {"uid": 2, "miner_hotkey": "lab-hotkey", "paid_alpha_percent": 5.0}
            ],
            "queued_champion_allocations": [],
        },
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [{"miner_hotkey": "fulfillment-hotkey", "wins": 9}],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.705,
        "fulfillment_rows": [{"hotkey": "fulfillment-hotkey", "share": 0.705}],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125_000,
        "min_total_rep_for_distribution": 100,
    }
    value.update(overrides)
    if "config_hash" not in overrides:
        value["config_hash"] = weight_config_hash(value)
    return value


def test_full_two_host_allocation_matches_current_order_and_totals():
    result = compute_final_weights(_snapshot())
    assert result["uids"] == [0, 1, 2]
    assert result["weights"] == pytest.approx([0.195, 0.755, 0.05], abs=1e-15)
    assert sum(result["weights"]) == pytest.approx(1.0, abs=1e-15)
    assert result["sparse_uids"] == [0, 1, 2]
    assert len(result["weights_hash"]) == 64


def test_disabled_tracks_resolve_to_exact_full_burn():
    snapshot = _snapshot(
        ff_enabled=False,
        fulfillment_share=0.0,
        fulfillment_rows=[],
        leaderboard_entries=[],
        research_lab_allocation_doc={},
    )
    result = compute_final_weights(snapshot)
    assert result["uids"] == [0]
    assert result["weights"] == [1.0]
    assert result["sparse_uids"] == [0]
    assert result["sparse_weights_u16"] == [65535]


def test_deregistered_fulfillment_and_research_lab_payments_burn():
    allocation = dict(_snapshot()["research_lab_allocation_doc"])
    allocation["champion_allocations"] = [
        {"uid": 3, "miner_hotkey": "wrong-hotkey", "paid_alpha_percent": 5.0}
    ]
    result = compute_final_weights(
        _snapshot(
            fulfillment_rows=[{"hotkey": "deregistered", "share": 0.705}],
            research_lab_allocation_doc=allocation,
        )
    )
    assert result["uids"] == [0, 1]
    assert result["weights"] == pytest.approx([0.95, 0.05], abs=1e-15)


def test_sourcing_threshold_and_rep_distribution_are_derived_inside_core():
    result = compute_final_weights(
        _snapshot(
            research_lab_allocation_doc={"lab_cap_percent": 0, "unallocated_percent": 0},
            research_lab_fallback_share=0.0,
            leaderboard_bonus_share=0.0,
            leaderboard_rank_shares=[],
            leaderboard_entries=[],
            fulfillment_share=0.0,
            fulfillment_rows=[],
            rolling_lead_count=62_500,
            rolling_scores=[
                {"hotkey": "source-hotkey", "score": 300.0},
                {"hotkey": "deregistered", "score": 100.0},
            ],
        )
    )
    # Fulfillment residual is 100% under the production residual formula, so
    # sourcing remains 0%; the unused fulfillment pool burns.
    assert result["uids"] == [0, 3]
    assert result["weights"] == [1.0, 0.0]
    assert result["sparse_uids"] == [0]


def test_burn_owner_and_banned_score_checks_fail_closed():
    with pytest.raises(WeightComputationError, match="ownership mismatch"):
        compute_final_weights(_snapshot(expected_burn_target_hotkey="wrong"))
    with pytest.raises(WeightComputationError, match="banned hotkey"):
        compute_final_weights(
            _snapshot(
                banned_hotkeys=["source-hotkey"],
                rolling_scores=[{"hotkey": "source-hotkey", "score": 1.0}],
            )
        )


def test_config_hash_binds_every_weight_behavior_setting():
    snapshot = _snapshot()
    snapshot["leaderboard_bonus_share"] = 0.1
    with pytest.raises(WeightComputationError, match="config_hash does not match"):
        compute_final_weights(snapshot)


def test_pure_u16_conversion_matches_pinned_bittensor_for_random_vectors():
    rng = random.Random(7102026)
    for size in range(1, 65):
        for _ in range(40):
            uids = list(range(size))
            weights = [rng.random() ** 8 for _ in range(size)]
            expected_uids, expected_weights = convert_weights_and_uids_for_emit(
                np.array(uids, dtype=np.int64),
                np.array(weights, dtype=np.float32),
            )
            actual_uids, actual_weights = normalize_to_u16_with_uids_pure(uids, weights)
            assert actual_uids == list(expected_uids)
            assert actual_weights == list(expected_weights)


def test_validator_enclave_v2_computes_then_signs_its_own_result():
    from validator_tee.enclave import tee_service

    response = tee_service.handle_request({"command": "compute_weights_v2", "snapshot": _snapshot()})
    assert response["status"] == "ok"
    result = response["weight_result"]
    assert result == compute_final_weights(_snapshot())
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(response["receipt"]["enclave_pubkey"])).verify(
        bytes.fromhex(response["weights_signature"]),
        bytes.fromhex(result["weights_hash"]),
    )
    validate_signed_receipt(response["receipt"])
    assert response["receipt"]["purpose"] == WEIGHT_PURPOSE
    assert response["attestation_user_data"]["purpose"] == WEIGHT_PURPOSE


def test_validator_enclave_v2_rejects_tampered_snapshot_before_signing():
    from validator_tee.enclave import tee_service

    snapshot = _snapshot(expected_burn_target_hotkey="attacker")
    response = tee_service.handle_request({"command": "compute_weights_v2", "snapshot": snapshot})
    assert response["status"] == "error"
    assert "ownership mismatch" in response["error"]
