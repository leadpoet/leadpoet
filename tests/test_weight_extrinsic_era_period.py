"""The SDK era period must match the measured chain profile.

The enclave rebuilds the weight-extrinsic signature payload from the
measured chain signing profile, whose ``extrinsic_period`` fixes the mortal
era bytes. bittensor's ``set_weights`` default period (``DEFAULT_PERIOD``)
differs from the profile, so any call site that omits ``period=`` produces a
payload the enclave refuses to sign on every attempt.
"""

import inspect
import json
from pathlib import Path

import pytest

from leadpoet_canonical.hotkey_authority_v2 import (
    encode_commit_timelocked_call,
    encode_weight_signature_payload,
)

PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "validator_tee"
    / "enclave"
    / "chain_signing_profile_v2.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def test_sdk_default_period_diverges_from_profile_payload():
    profile = _profile()
    from bittensor.core.settings import DEFAULT_PERIOD

    assert DEFAULT_PERIOD != profile["extrinsic_period"], (
        "if the SDK default ever equals the profile period, this pin and the "
        "explicit period= argument can be revisited together"
    )
    call = encode_commit_timelocked_call(
        profile=profile,
        netuid=71,
        commitment=bytes(range(96)),
        reveal_round=1234567,
    )
    _, signed_profile = encode_weight_signature_payload(
        profile=profile,
        call_bytes=call,
        era_current=8_600_000,
        nonce=5,
        block_hash="ab" * 32,
    )
    sdk_default_profile = dict(profile, extrinsic_period=int(DEFAULT_PERIOD))
    _, signed_sdk_default = encode_weight_signature_payload(
        profile=sdk_default_profile,
        call_bytes=call,
        era_current=8_600_000,
        nonce=5,
        block_hash="ab" * 32,
    )
    assert signed_profile != signed_sdk_default


def test_set_weights_call_pins_profile_period():
    from neurons.validator import Validator

    source = inspect.getsource(Validator._set_weights_until_epoch_end)
    assert "load_chain_signing_profile" in source
    assert "period=extrinsic_period" in source
    assert "expected_era_period=extrinsic_period" in source


def test_signing_context_rejects_mismatched_era_period():
    from validator_tee.host import enclave_hotkey_v2 as mod

    context = object.__new__(mod.AuthoritativeSetWeightsContextV2)
    context._expected_era_period = int(_profile()["extrinsic_period"])
    context._commit_queue = [{"commit_authorization_id": "sha256:" + "0" * 64}]

    class _Substrate:
        def get_chain_finalised_head(self):
            return "0x" + "a" * 64

        def init_runtime(self, block_hash=None):
            assert block_hash == "0x" + "a" * 64
            return None

        def get_account_nonce(self, _address):
            return 7

    class _Wallet:
        hotkey = None

    context.substrate = _Substrate()
    context.wallet = _Wallet()

    with pytest.raises(mod.EnclaveHotkeyV2Error, match="era period 128"):
        mod.AuthoritativeSetWeightsContextV2._create_signed_extrinsic(
            context,
            call=object(),
            keypair=None,
            era={"period": 128},
            nonce=7,
            tip=0,
        )
