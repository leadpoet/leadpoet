from __future__ import annotations

import json
import os
from pathlib import Path

import sr25519
import pytest
from scalecodec.utils.ss58 import ss58_encode

from scripts.seal_temporary_testnet_validator_hotkey import (
    TemporaryTestnetHotkeySealError,
    main,
    seal_temporary_testnet_hotkey,
)
from validator_tee.host.hotkey_bootstrap_v2 import validate_hotkey_envelope


class KMS:
    def __init__(self):
        self.seed = None

    def encrypt(self, **request):
        self.seed = request["Plaintext"]
        return {
            "KeyId": "arn:aws:kms:us-east-1:123:key/testnet-validator-v2",
            "CiphertextBlob": b"sealed-testnet-validator-hotkey",
        }


class Keypair:
    def __init__(self, hotkey, public_key):
        self.ss58_address = hotkey
        self.public_key = public_key


class Keyfile:
    def __init__(self, *, path, hotkey, public_key, encrypted=False):
        self.path = path
        self._encrypted = encrypted
        self.keypair = Keypair(hotkey, public_key)

    def is_encrypted(self):
        return self._encrypted


def test_memory_only_keyfile_seal_writes_public_config_and_ciphertext(tmp_path):
    seed = bytes(range(32))
    public_key, _ = sr25519.pair_from_seed(seed)
    hotkey = ss58_encode(public_key.hex(), ss58_format=42)
    keyfile_path = tmp_path / "default"
    keyfile_path.write_text(
        json.dumps(
            {
                "accountId": "0x" + public_key.hex(),
                "publicKey": "0x" + public_key.hex(),
                "privateKey": "0x" + "9" * 128,
                "secretPhrase": "not printed by the helper",
                "secretSeed": "0x" + seed.hex(),
                "ss58Address": hotkey,
            }
        ),
        encoding="utf-8",
    )
    keyfile_path.chmod(0o600)
    drand = tmp_path / "drand.sha256"
    drand.write_text("a" * 64 + "\n", encoding="ascii")
    config = tmp_path / "output" / "config.json"
    envelope = tmp_path / "output" / "envelope.json"
    kms = KMS()

    result = seal_temporary_testnet_hotkey(
        keyfile_path=keyfile_path,
        expected_hotkey=hotkey,
        kms_key_id="alias/testnet-validator-v2",
        chain_profile_path=Path(
            "validator_tee/enclave/chain_signing_profile_test_v2.json"
        ),
        drand_hash_path=drand,
        config_output=config,
        envelope_output=envelope,
        kms_client=kms,
        keyfile_factory=lambda **kwargs: Keyfile(
            **kwargs,
            hotkey=hotkey,
            public_key=bytes(public_key),
        ),
    )

    assert kms.seed == seed
    assert result["validator_hotkey"] == hotkey
    assert json.loads(config.read_text())["validator_hotkey"] == hotkey
    sealed = envelope.read_text()
    assert seed.hex() not in sealed
    assert "not printed by the helper" not in sealed
    assert validate_hotkey_envelope(json.loads(sealed))
    assert os.stat(config).st_mode & 0o777 == 0o600
    assert os.stat(envelope).st_mode & 0o777 == 0o600


def test_seal_refuses_encrypted_keyfile_before_kms(tmp_path):
    seed = bytes(range(32))
    public_key, _ = sr25519.pair_from_seed(seed)
    hotkey = ss58_encode(public_key.hex(), ss58_format=42)
    keyfile_path = tmp_path / "default"
    keyfile_path.write_text(
        json.dumps({"secretSeed": "0x" + seed.hex()}), encoding="utf-8"
    )
    keyfile_path.chmod(0o600)
    drand = tmp_path / "drand.sha256"
    drand.write_text("a" * 64 + "\n", encoding="ascii")
    kms = KMS()

    with pytest.raises(TemporaryTestnetHotkeySealError, match="encrypted"):
        seal_temporary_testnet_hotkey(
            keyfile_path=keyfile_path,
            expected_hotkey=hotkey,
            kms_key_id="alias/testnet-validator-v2",
            chain_profile_path=Path(
                "validator_tee/enclave/chain_signing_profile_test_v2.json"
            ),
            drand_hash_path=drand,
            config_output=tmp_path / "config.json",
            envelope_output=tmp_path / "envelope.json",
            kms_client=kms,
            keyfile_factory=lambda **kwargs: Keyfile(
                **kwargs,
                hotkey=hotkey,
                public_key=bytes(public_key),
                encrypted=True,
            ),
        )

    assert kms.seed is None
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "envelope.json").exists()


def test_cli_reports_only_generic_failure(monkeypatch, capsys, tmp_path):
    def fail(**_kwargs):
        raise RuntimeError("secret diagnostic must not escape")

    monkeypatch.setattr(
        "scripts.seal_temporary_testnet_validator_hotkey.seal_temporary_testnet_hotkey",
        fail,
    )
    result = main(
        [
            "--keyfile",
            str(tmp_path / "keyfile"),
            "--expected-hotkey",
            "public-hotkey",
            "--kms-key-id",
            "alias/test",
            "--chain-profile",
            str(tmp_path / "profile.json"),
            "--drand-hash",
            str(tmp_path / "drand.sha256"),
            "--config-output",
            str(tmp_path / "config.json"),
            "--envelope-output",
            str(tmp_path / "envelope.json"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "temporary hotkey sealing failed" in captured.err
    assert "secret diagnostic" not in captured.err
    assert captured.out == ""
