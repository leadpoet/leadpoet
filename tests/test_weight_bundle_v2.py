import copy
import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_receipts import build_receipt_body, create_signed_receipt
from leadpoet_canonical.auditor_v2 import (
    IDENTITY_CACHE_SCHEMA_VERSION,
    AuditorV2Error,
    verify_attested_weight_bundle_v2,
)
from leadpoet_canonical.weight_bundle_v2 import (
    WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
    WeightBundleV2Error,
    validate_weight_bundle_v2,
)
from leadpoet_canonical.weight_computation import WEIGHT_SNAPSHOT_SCHEMA_VERSION, weight_config_hash
from validator_tee.enclave import tee_service


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


def _bundle():
    snapshot = _snapshot()
    response = tee_service.handle_request({"command": "compute_weights_v2", "snapshot": snapshot})
    assert response["status"] == "ok"
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
    response = tee_service.handle_request({"command": "compute_weights_v2", "snapshot": snapshot})
    assert response["status"] == "ok"
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
    response = tee_service.handle_request({"command": "compute_weights_v2", "snapshot": snapshot})
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


def test_auditor_v2_requires_three_build_identity_for_every_receipt():
    bundle = _bundle_with_allocation_lineage()
    identities = {
        "schema_version": IDENTITY_CACHE_SCHEMA_VERSION,
        "entries": [
            {
                "role": "validator_weights",
                "commit_sha": "a" * 40,
                "pcr0": "1" * 96,
                "verified_build_count": 3,
            },
            {
                "role": "gateway_scoring",
                "commit_sha": "b" * 40,
                "pcr0": "2" * 96,
                "verified_build_count": 3,
            },
        ],
    }

    def _nitro(**kwargs):
        role = "validator_weights" if kwargs["role"] == "validator" else "gateway_scoring"
        pcr0 = next(item["pcr0"] for item in identities["entries"] if item["role"] == role)
        return True, {
            "purpose": kwargs["expected_purpose"],
            "epoch_id": kwargs["expected_epoch_id"],
            "enclave_pubkey": kwargs["expected_pubkey"],
            "pcr0": pcr0,
        }

    verified = verify_attested_weight_bundle_v2(
        bundle,
        identity_cache=identities,
        nitro_verifier=_nitro,
    )
    assert len(verified["independent_receipt_identities"]) == 3
    assert all(
        item["verified_build_count"] == 3
        for item in verified["independent_receipt_identities"]
    )

    identities["entries"][1]["verified_build_count"] = 2
    with pytest.raises(AuditorV2Error, match="three"):
        verify_attested_weight_bundle_v2(
            bundle,
            identity_cache=identities,
            nitro_verifier=_nitro,
        )
