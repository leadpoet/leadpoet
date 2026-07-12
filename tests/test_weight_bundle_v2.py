import copy
import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_receipts import (
    WEIGHT_PURPOSE,
    WEIGHT_ROLE,
    build_receipt_body,
    create_signed_receipt,
)
from leadpoet_canonical.weight_bundle_v2 import (
    WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
    WeightBundleV2Error,
    validate_weight_bundle_v2,
)
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    compute_final_weights,
    sha256_json,
    weight_config_hash,
)


def _snapshot():
    value = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 300,
        "block": 108350,
        "commit_sha": "a" * 40,
        "config_hash": "",
        "parent_receipt_hashes": [],
        "research_lab_allocation_receipt_hash": "",
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": "burn",
        "metagraph_hotkeys": ["burn", "miner"],
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
        "ff_enabled": False,
        "base_burn_share": 0.0,
        "champion_share": 0.0,
        "champion_uid": None,
        "effective_champion_share": 0.0,
        "research_lab_fallback_share": 0.2,
        "research_lab_allocation_doc": {},
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.0,
        "fulfillment_rows": [],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125000,
        "min_total_rep_for_distribution": 100,
    }
    value["config_hash"] = weight_config_hash(value)
    return value


def _historical_response(snapshot):
    """Build a read-only historical V1 fixture without a live signing RPC."""

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    result = compute_final_weights(snapshot)
    body = build_receipt_body(
        role=WEIGHT_ROLE,
        purpose=WEIGHT_PURPOSE,
        job_id="historical-validator-weights:%s" % result["epoch_id"],
        epoch_id=result["epoch_id"],
        commit_sha=str(snapshot["commit_sha"]),
        build_manifest_hash="sha256:" + "1" * 64,
        config_hash=str(snapshot["config_hash"]),
        input_root=result["snapshot_hash"],
        output_root=sha256_json(result),
        evidence_roots=(
            {
                "research_lab_allocation_receipt": snapshot[
                    "research_lab_allocation_receipt_hash"
                ]
            }
            if snapshot["research_lab_allocation_receipt_hash"]
            else {}
        ),
        parent_receipt_hashes=snapshot["parent_receipt_hashes"],
        status="succeeded",
        issued_at="2026-07-10T00:00:00Z",
    )
    receipt = create_signed_receipt(
        body=body,
        enclave_pubkey=public_key,
        attestation_document_b64=base64.b64encode(
            b"historical-nitro-attestation"
        ).decode(),
        sign_digest=private_key.sign,
    )
    return {
        "weight_result": result,
        "weights_signature": private_key.sign(
            bytes.fromhex(result["weights_hash"])
        ).hex(),
        "receipt": receipt,
    }


def _bundle():
    snapshot = _snapshot()
    response = _historical_response(snapshot)
    return {
        "schema_version": WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
        "validator_hotkey": "validator-hotkey",
        "binding_message": "binding-message",
        "validator_hotkey_signature": "binding-signature",
        "weight_snapshot": snapshot,
        "weight_result": response["weight_result"],
        "weights_signature": response["weights_signature"],
        "weight_receipt": response["receipt"],
        "parent_receipts": [],
    }


def _signed_scoring_receipt(*, purpose, epoch_id=300, parents=()):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    body = build_receipt_body(
        role="gateway_scoring",
        purpose=purpose,
        job_id="%s:%s" % (purpose, len(parents)),
        epoch_id=epoch_id,
        commit_sha="b" * 40,
        build_manifest_hash="sha256:" + "c" * 64,
        config_hash="sha256:" + "d" * 64,
        input_root="sha256:" + "e" * 64,
        output_root="sha256:" + "f" * 64,
        evidence_roots={},
        parent_receipt_hashes=list(parents),
        status="succeeded",
        issued_at="2026-07-10T00:00:00Z",
    )
    return create_signed_receipt(
        body=body,
        enclave_pubkey=public_key,
        attestation_document_b64=base64.b64encode(b"nitro-attestation").decode(),
        sign_digest=private_key.sign,
    )


def _bundle_with_allocation_lineage():
    score_receipt = _signed_scoring_receipt(
        purpose="research_lab.candidate_score.v1",
    )
    allocation_receipt = _signed_scoring_receipt(
        purpose="research_lab.allocation.v1",
        parents=[score_receipt["receipt_hash"]],
    )
    snapshot = _snapshot()
    snapshot["parent_receipt_hashes"] = [allocation_receipt["receipt_hash"]]
    snapshot["research_lab_allocation_receipt_hash"] = allocation_receipt["receipt_hash"]
    snapshot["config_hash"] = weight_config_hash(snapshot)
    response = _historical_response(snapshot)
    return {
        "schema_version": WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
        "validator_hotkey": "validator-hotkey",
        "binding_message": "binding-message",
        "validator_hotkey_signature": "binding-signature",
        "weight_snapshot": snapshot,
        "weight_result": response["weight_result"],
        "weights_signature": response["weights_signature"],
        "weight_receipt": response["receipt"],
        "parent_receipts": [score_receipt, allocation_receipt],
    }


def test_v2_bundle_recomputes_and_accepts_only_enclave_derived_weights():
    verified = validate_weight_bundle_v2(_bundle(), require_allocation_ancestry=False)
    assert verified["uids"] == [0]
    assert verified["weights_u16"] == [65535]


@pytest.mark.parametrize(
    "tamper",
    ["snapshot", "result", "weights_signature", "receipt", "unknown_field"],
)
def test_v2_bundle_rejects_tampering(tamper):
    bundle = _bundle()
    if tamper == "snapshot":
        bundle["weight_snapshot"] = copy.deepcopy(bundle["weight_snapshot"])
        bundle["weight_snapshot"]["block"] += 1
    elif tamper == "result":
        bundle["weight_result"] = copy.deepcopy(bundle["weight_result"])
        bundle["weight_result"]["weights"][0] = 0.5
    elif tamper == "weights_signature":
        bundle["weights_signature"] = "00" * 64
    elif tamper == "receipt":
        bundle["weight_receipt"] = copy.deepcopy(bundle["weight_receipt"])
        bundle["weight_receipt"]["epoch_id"] += 1
    else:
        bundle["unknown"] = True
    with pytest.raises(Exception):
        validate_weight_bundle_v2(bundle, require_allocation_ancestry=False)


def test_v2_required_mode_rejects_missing_allocation_ancestry():
    with pytest.raises(WeightBundleV2Error, match="allocation receipt is required"):
        validate_weight_bundle_v2(_bundle(), require_allocation_ancestry=True)


def test_v2_required_bundle_accepts_complete_connected_allocation_lineage():
    verified = validate_weight_bundle_v2(
        _bundle_with_allocation_lineage(),
        require_allocation_ancestry=True,
    )
    assert verified["epoch_id"] == 300


def test_v2_bundle_rejects_disconnected_extra_receipt():
    bundle = _bundle_with_allocation_lineage()
    bundle["parent_receipts"].append(
        _signed_scoring_receipt(purpose="research_lab.benchmark.v1")
    )
    with pytest.raises(WeightBundleV2Error, match="disconnected"):
        validate_weight_bundle_v2(bundle, require_allocation_ancestry=True)


def test_v2_bundle_rejects_snapshot_parent_that_is_not_the_allocation_receipt():
    allocation_receipt = _signed_scoring_receipt(
        purpose="research_lab.allocation.v1",
    )
    score_receipt = _signed_scoring_receipt(
        purpose="research_lab.candidate_score.v1",
        parents=[allocation_receipt["receipt_hash"]],
    )
    snapshot = _snapshot()
    snapshot["parent_receipt_hashes"] = [score_receipt["receipt_hash"]]
    snapshot["research_lab_allocation_receipt_hash"] = allocation_receipt["receipt_hash"]
    snapshot["config_hash"] = weight_config_hash(snapshot)
    response = _historical_response(snapshot)
    bundle = {
        "schema_version": WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
        "validator_hotkey": "validator-hotkey",
        "binding_message": "binding-message",
        "validator_hotkey_signature": "binding-signature",
        "weight_snapshot": snapshot,
        "weight_result": response["weight_result"],
        "weights_signature": response["weights_signature"],
        "weight_receipt": response["receipt"],
        "parent_receipts": [allocation_receipt, score_receipt],
    }
    with pytest.raises(WeightBundleV2Error, match="not a direct weight parent"):
        validate_weight_bundle_v2(bundle, require_allocation_ancestry=True)
