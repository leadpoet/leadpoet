import json
from pathlib import Path

import pytest

from validator_tee.host.verify_chain_signing_profile_v2 import (
    ChainSigningProfileV2Error,
    verify_chain_signing_profile_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.enclave.hotkey_authority_v2 import load_chain_signing_profile


def _profile():
    root = Path(__file__).resolve().parents[1]
    profile = json.loads(
        (
            root
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text()
    )
    profile.pop("supported_spec_versions", None)
    profile.pop("runtime_upgrade_policy", None)
    profile["spec_version"] = 437
    profile["genesis_hash"] = "ab" * 32
    return profile


def _compatible_profile():
    return {
        **_profile(),
        "spec_version": 438,
        "supported_spec_versions": [437, 438],
    }


def test_candidate_bundles_hash_selectable_testnet_profile(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    production_path = (
        root / "validator_tee/enclave/chain_signing_profile_v2.json"
    )
    test_profile = json.loads(
        production_path.with_name("chain_signing_profile_test_v2.json").read_text()
    )
    selected = load_chain_signing_profile(
        production_path, expected_hash=sha256_json(test_profile)
    )
    assert selected["network"] == "test"
    assert selected["chain_endpoint"] == "wss://test.finney.opentensor.ai:443"
    assert selected["genesis_hash"] == (
        "8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
    )
    assert selected["tempo"] == 99
    result = verify_chain_signing_profile_v2(
        profile=selected,
        runtime_version={"specVersion": 447, "transactionVersion": 1},
        genesis_hash="0x" + selected["genesis_hash"],
    )
    assert result["status"] == "ready"


def test_chain_signing_profile_accepts_exact_live_runtime():
    result = verify_chain_signing_profile_v2(
        profile=_profile(),
        runtime_version={"specVersion": 437, "transactionVersion": 1},
        genesis_hash="0x" + "ab" * 32,
    )

    assert result["status"] == "ready"
    assert result["spec_version"] == 437
    assert result["transaction_version"] == 1


@pytest.mark.parametrize("spec_version", [437, 438])
def test_chain_signing_profile_accepts_only_explicit_compatible_versions(
    spec_version,
):
    result = verify_chain_signing_profile_v2(
        profile=_compatible_profile(),
        runtime_version={
            "specVersion": spec_version,
            "transactionVersion": 1,
        },
        genesis_hash="0x" + "ab" * 32,
    )

    assert result["status"] == "ready"
    assert result["spec_version"] == spec_version
    assert result["selected_profile_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    ("runtime_version", "genesis_hash", "match"),
    [
        (
            {"specVersion": 438, "transactionVersion": 1},
            "0x" + "ab" * 32,
            "runtime specVersion is not explicitly supported",
        ),
        (
            {"specVersion": 437, "transactionVersion": 2},
            "0x" + "ab" * 32,
            "runtime transactionVersion differs from the measured profile",
        ),
        (
            {"specVersion": 437, "transactionVersion": 1},
            "0x" + "cd" * 32,
            "runtime genesis differs from the measured profile",
        ),
    ],
)
def test_chain_signing_profile_rejects_live_runtime_mismatch(
    runtime_version, genesis_hash, match
):
    with pytest.raises(ChainSigningProfileV2Error, match=match):
        verify_chain_signing_profile_v2(
            profile=_profile(),
            runtime_version=runtime_version,
            genesis_hash=genesis_hash,
        )


@pytest.mark.parametrize(
    "runtime_version",
    [
        {"specVersion": True, "transactionVersion": 1},
        {"specVersion": 438, "transactionVersion": "1"},
        {"specVersion": 438},
    ],
)
def test_chain_signing_profile_rejects_malformed_runtime(runtime_version):
    with pytest.raises(
        ChainSigningProfileV2Error,
        match="live runtime version response is invalid",
    ):
        verify_chain_signing_profile_v2(
            profile=_compatible_profile(),
            runtime_version=runtime_version,
            genesis_hash="0x" + "ab" * 32,
        )


def test_chain_signing_profile_accepts_invariant_checked_future_spec():
    profile = {
        **_compatible_profile(),
        "runtime_upgrade_policy": {
            "mode": "exact_payload_invariants_v1",
            "minimum_spec_version": 437,
        },
    }
    result = verify_chain_signing_profile_v2(
        profile=profile,
        runtime_version={"specVersion": 440, "transactionVersion": 1},
        genesis_hash="0x" + "ab" * 32,
        call_metadata={
            "commit_timelocked_mechanism_weights": {
                "call_index": profile["commit_call_index"],
                "fields": [
                    {"name": "netuid", "typeName": "NetUid"},
                    {"name": "mecid", "typeName": "MechId"},
                    {
                        "name": "commit",
                        "typeName": "BoundedVec<u8, ConstU32<MAX_CRV3_COMMIT_SIZE_BYTES>>",
                    },
                    {"name": "reveal_round", "typeName": "u64"},
                    {"name": "commit_reveal_version", "typeName": "u16"},
                ],
            },
            "serve_axon": {
                "call_index": profile["serve_axon_call_index"],
                "fields": [
                    {"name": "netuid", "typeName": "NetUid"},
                    {"name": "version", "typeName": "u32"},
                    {"name": "ip", "typeName": "u128"},
                    {"name": "port", "typeName": "u16"},
                    {"name": "ip_type", "typeName": "u8"},
                    {"name": "protocol", "typeName": "u8"},
                    {"name": "placeholder1", "typeName": "u8"},
                    {"name": "placeholder2", "typeName": "u8"},
                ],
            },
        },
    )
    assert result["spec_version"] == 440


def test_chain_signing_profile_rejects_future_spec_with_changed_call():
    profile = {
        **_compatible_profile(),
        "runtime_upgrade_policy": {
            "mode": "exact_payload_invariants_v1",
            "minimum_spec_version": 437,
        },
    }
    with pytest.raises(
        ChainSigningProfileV2Error, match="call contract differs"
    ):
        verify_chain_signing_profile_v2(
            profile=profile,
            runtime_version={"specVersion": 440, "transactionVersion": 1},
            genesis_hash="0x" + "ab" * 32,
            call_metadata={
                "commit_timelocked_mechanism_weights": {
                    "call_index": "0777",
                    "fields": [],
                },
                "serve_axon": {
                    "call_index": profile["serve_axon_call_index"],
                    "fields": [],
                },
            },
        )
