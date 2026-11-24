import json
from pathlib import Path

import pytest

from leadpoet_canonical.hotkey_authority_v2 import (
    HotkeyAuthorityV2Error,
    chain_signing_profiles,
    encode_commit_timelocked_call,
    encode_weight_signature_payload,
    select_chain_signing_profile,
)


PROFILE_PATH = Path(__file__).parents[1] / "lab_arena/chain_signing_profile_v2.json"
GENESIS = "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
BLOCK_HASH = "0" * 64


def _profile():
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("spec", [494, 495])
def test_measured_finney_runtime_uses_mechanism_call_and_transaction_version_two(spec):
    profile = select_chain_signing_profile(
        _profile(),
        runtime_version={"specVersion": spec, "transactionVersion": 2},
        genesis_hash=GENESIS,
    )
    assert profile["spec_version"] == spec
    assert profile["transaction_version"] == 2
    assert profile["extrinsic_period"] == 32
    assert profile["commit_call_index"] == "0776"

    call = encode_commit_timelocked_call(
        profile=profile,
        netuid=71,
        commitment=bytes.fromhex("11" * 32),
        reveal_round=123456,
    )
    assert call == bytes.fromhex("077647000080" + "11" * 32 + "40e20100000000000400")

    preimage, _ = encode_weight_signature_payload(
        profile=profile,
        call_bytes=call,
        era_current=9572067,
        nonce=0,
        block_hash=BLOCK_HASH,
    )
    expected = b"".join((
        call,
        bytes.fromhex("3400"),  # Period 32, current block 9572067.
        b"\x00",  # Nonce.
        b"\x00",  # Zero transaction tip.
        b"\x00",  # Disabled metadata hash.
        spec.to_bytes(4, "little"),
        (2).to_bytes(4, "little"),
        bytes.fromhex(GENESIS),
        bytes.fromhex(BLOCK_HASH),
        b"\x00",  # No metadata hash.
    ))
    assert preimage == expected


def test_finney_runtime_493_keeps_its_measured_transaction_version():
    profile = select_chain_signing_profile(
        _profile(),
        runtime_version={"specVersion": 493, "transactionVersion": 1},
        genesis_hash=GENESIS,
    )
    assert profile["transaction_version"] == 1
    assert profile["extrinsic_period"] == 32
    assert profile["commit_call_index"] == "0776"


def test_runtime_494_rejects_unmeasured_transaction_version():
    with pytest.raises(HotkeyAuthorityV2Error, match="transactionVersion differs"):
        select_chain_signing_profile(
            _profile(),
            runtime_version={"specVersion": 494, "transactionVersion": 3},
            genesis_hash=GENESIS,
        )


def test_later_runtime_does_not_inherit_the_measured_transaction_version():
    with pytest.raises(HotkeyAuthorityV2Error, match="transactionVersion differs"):
        select_chain_signing_profile(
            _profile(),
            runtime_version={"specVersion": 496, "transactionVersion": 2},
            genesis_hash=GENESIS,
        )


def test_chain_signing_profile_expands_transaction_version_overrides():
    profiles = {item["spec_version"]: item for item in chain_signing_profiles(_profile())}
    assert profiles[437]["transaction_version"] == 1
    assert profiles[438]["transaction_version"] == 1
    assert profiles[437]["extrinsic_period"] == 32
    assert profiles[438]["extrinsic_period"] == 32
    assert profiles[494]["transaction_version"] == 2
    assert profiles[494]["extrinsic_period"] == 32


def test_transaction_version_override_must_name_supported_spec_version():
    profile = _profile()
    profile["transaction_version_overrides"] = {"493": 2}
    with pytest.raises(HotkeyAuthorityV2Error, match="transaction version overrides"):
        chain_signing_profiles(profile)


def test_transaction_version_override_keys_are_validated_before_sorting():
    profile = _profile()
    profile["transaction_version_overrides"] = {"494": 2, 494: 2}
    with pytest.raises(HotkeyAuthorityV2Error, match="transaction version overrides"):
        chain_signing_profiles(profile)


def test_extrinsic_period_must_be_a_power_of_two():
    profile = _profile()
    profile["extrinsic_period"] = 30
    with pytest.raises(HotkeyAuthorityV2Error, match="power of two"):
        chain_signing_profiles(profile)
