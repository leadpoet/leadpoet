import json
from pathlib import Path

import pytest

from validator_tee.host.verify_chain_signing_profile_v2 import (
    ChainSigningProfileV2Error,
    verify_chain_signing_profile_v2,
)


def _profile():
    root = Path(__file__).resolve().parents[1]
    profile = json.loads(
        (
            root
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text()
    )
    profile.pop("supported_spec_versions", None)
    profile["spec_version"] = 437
    profile["genesis_hash"] = "ab" * 32
    return profile


def _compatible_profile():
    return {
        **_profile(),
        "spec_version": 438,
        "supported_spec_versions": [437, 438],
    }


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
