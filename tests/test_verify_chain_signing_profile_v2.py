import pytest

from validator_tee.host.verify_chain_signing_profile_v2 import (
    ChainSigningProfileV2Error,
    verify_chain_signing_profile_v2,
)


def _profile():
    return {
        "spec_version": 437,
        "transaction_version": 1,
        "genesis_hash": "ab" * 32,
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


@pytest.mark.parametrize(
    ("runtime_version", "genesis_hash", "match"),
    [
        (
            {"specVersion": 438, "transactionVersion": 1},
            "0x" + "ab" * 32,
            "specVersion live=438 measured=437",
        ),
        (
            {"specVersion": 437, "transactionVersion": 2},
            "0x" + "ab" * 32,
            "transactionVersion live=2 measured=1",
        ),
        (
            {"specVersion": 437, "transactionVersion": 1},
            "0x" + "cd" * 32,
            "genesis hash differs",
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
