import copy

import pytest

from leadpoet_canonical.attested_receipts import WEIGHT_PURPOSE
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    compute_final_weights,
    weight_config_hash,
)
from validator_tee.enclave import tee_service
from validator_tee.host.weight_shadow import (
    AttestedWeightVerificationError,
    build_weight_bundle_v2,
    execute_attested_weight_mode,
    verify_enclave_weight_response,
)


def _snapshot():
    value = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 200,
        "block": 72099,
        "commit_sha": "a" * 40,
        "config_hash": "",
        "parent_receipt_hashes": [],
        "research_lab_allocation_receipt_hash": "",
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": "burn",
        "metagraph_hotkeys": ["burn", "ff", "lab"],
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
                {"uid": 2, "miner_hotkey": "lab", "paid_alpha_percent": 5.0},
            ],
            "queued_champion_allocations": [],
        },
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [{"miner_hotkey": "ff", "wins": 3}],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.705,
        "fulfillment_rows": [{"hotkey": "ff", "share": 0.705}],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125000,
        "min_total_rep_for_distribution": 100,
    }
    value["config_hash"] = weight_config_hash(value)
    return value


def _response(snapshot):
    response = tee_service.handle_request({"command": "compute_weights_v2", "snapshot": snapshot})
    assert response["status"] == "ok"
    return {key: value for key, value in response.items() if key != "status"}


def test_shadow_verifier_accepts_exact_enclave_and_host_result():
    snapshot = _snapshot()
    result = compute_final_weights(snapshot)
    verified = verify_enclave_weight_response(
        snapshot=snapshot,
        response=_response(snapshot),
        host_uids=result["uids"],
        host_weights=result["weights"],
        require_real_attestation=False,
    )
    assert verified == result


@pytest.mark.parametrize("target", ["host", "result", "signature", "purpose", "commit"])
def test_shadow_verifier_rejects_every_tampered_boundary(target):
    snapshot = _snapshot()
    result = compute_final_weights(snapshot)
    response = _response(snapshot)
    host_weights = list(result["weights"])
    if target == "host":
        host_weights[0] += 1e-12
    elif target == "result":
        response["weight_result"] = copy.deepcopy(response["weight_result"])
        response["weight_result"]["weights"][0] += 1e-12
    elif target == "signature":
        response["weights_signature"] = "00" * 64
    elif target == "purpose":
        response["attestation_user_data"] = dict(response["attestation_user_data"])
        response["attestation_user_data"]["purpose"] = "validator_weights"
    elif target == "commit":
        response["receipt"] = dict(response["receipt"])
        response["receipt"]["commit_sha"] = "b" * 40
    with pytest.raises(Exception):
        verify_enclave_weight_response(
            snapshot=snapshot,
            response=response,
            host_uids=result["uids"],
            host_weights=host_weights,
            require_real_attestation=False,
        )


def test_required_mode_rejects_mock_attestation():
    snapshot = _snapshot()
    result = compute_final_weights(snapshot)
    response = _response(snapshot)
    assert response["attestation_user_data"]["purpose"] == WEIGHT_PURPOSE
    with pytest.raises(AttestedWeightVerificationError, match="mock attestation"):
        verify_enclave_weight_response(
            snapshot=snapshot,
            response=response,
            host_uids=result["uids"],
            host_weights=result["weights"],
            require_real_attestation=True,
        )


def test_v2_payload_builder_self_verifies_before_network_submission():
    snapshot = _snapshot()
    response = _response(snapshot)
    bundle = build_weight_bundle_v2(
        snapshot=snapshot,
        enclave_response=response,
        validator_hotkey="validator-hotkey",
        binding_message="binding",
        validator_hotkey_signature="hotkey-signature",
    )
    assert bundle["weight_snapshot"] == snapshot
    assert bundle["weight_receipt"]["receipt_hash"] == response["receipt"]["receipt_hash"]


@pytest.mark.asyncio
async def test_off_mode_never_calls_the_enclave(monkeypatch):
    def fail_if_called(_snapshot):
        raise AssertionError("off mode called the enclave")

    assert await execute_attested_weight_mode(
        mode="off",
        snapshot={},
        host_uids=[0],
        host_weights=[1.0],
        compute_weights=fail_if_called,
    ) is None


@pytest.mark.asyncio
async def test_shadow_failure_is_observational_but_required_failure_is_closed(monkeypatch):
    def fail(_snapshot):
        raise RuntimeError("enclave unavailable")

    snapshot = _snapshot()
    result = compute_final_weights(snapshot)
    assert await execute_attested_weight_mode(
        mode="shadow",
        snapshot=snapshot,
        host_uids=result["uids"],
        host_weights=result["weights"],
        compute_weights=fail,
    ) is None
    with pytest.raises(AttestedWeightVerificationError, match="required validator enclave"):
        await execute_attested_weight_mode(
            mode="required",
            snapshot=snapshot,
            host_uids=result["uids"],
            host_weights=result["weights"],
            compute_weights=fail,
        )
