import json
import os
from pathlib import Path

import sr25519
from scalecodec.utils.ss58 import ss58_encode

from validator_tee.host.hotkey_bootstrap_v2 import validate_hotkey_envelope
from validator_tee.host.prepare_hotkey_assets_v2 import (
    ValidatorHotkeyAssetPreparationV2Error,
    prepare_hotkey_assets_v2,
)


class KMS:
    def encrypt(self, **request):
        assert len(request["Plaintext"]) == 32
        return {
            "KeyId": "arn:aws:kms:us-east-1:123:key/validator-v2",
            "CiphertextBlob": b"sealed-hotkey",
        }


def _inputs(tmp_path):
    seed = bytes(range(32))
    public_key, _ = sr25519.pair_from_seed(seed)
    hotkey = ss58_encode(public_key.hex(), ss58_format=42)
    seed_path = tmp_path / "seed"
    seed_path.write_bytes(seed)
    seed_path.chmod(0o600)
    profile = Path("validator_tee/enclave/chain_signing_profile_v2.json")
    drand = tmp_path / "drand.sha256"
    drand.write_text("a" * 64 + "\n")
    return seed_path, hotkey, profile, drand


def test_prepares_seed_bound_config_and_ciphertext(tmp_path):
    seed, hotkey, profile, drand = _inputs(tmp_path)
    config = tmp_path / "config.json"
    envelope = tmp_path / "envelope.json"

    result = prepare_hotkey_assets_v2(
        seed_file=seed,
        validator_hotkey=hotkey,
        kms_key_id="alias/validator-v2",
        chain_profile_path=profile,
        drand_hash_path=drand,
        config_output=config,
        envelope_output=envelope,
        kms_client=KMS(),
    )

    assert result["validator_hotkey"] == hotkey
    assert json.loads(config.read_text())["hotkey_public_key"] == result["hotkey_public_key"]
    assert validate_hotkey_envelope(json.loads(envelope.read_text()))
    assert stat_mode(config) == 0o600
    assert stat_mode(envelope) == 0o600


def test_rejects_seed_for_another_hotkey(tmp_path):
    seed, _hotkey, profile, drand = _inputs(tmp_path)
    try:
        prepare_hotkey_assets_v2(
            seed_file=seed,
            validator_hotkey="5" + "A" * 47,
            kms_key_id="alias/validator-v2",
            chain_profile_path=profile,
            drand_hash_path=drand,
            config_output=tmp_path / "config.json",
            envelope_output=tmp_path / "envelope.json",
            kms_client=KMS(),
        )
    except ValidatorHotkeyAssetPreparationV2Error as exc:
        assert "another SS58" in str(exc)
    else:
        raise AssertionError("mismatched hotkey accepted")


def stat_mode(path):
    return os.stat(path).st_mode & 0o777
