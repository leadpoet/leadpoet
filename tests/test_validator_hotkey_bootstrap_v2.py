import base64

import pytest

from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from validator_tee.enclave.hotkey_authority_v2 import (
    HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
    HOTKEY_ENCRYPTION_ALGORITHM,
    HOTKEY_RECIPIENT_SCHEMA_VERSION,
    MEASURED_DRAND_LIBRARY_PATH,
)
from validator_tee.host.hotkey_bootstrap_v2 import (
    HOTKEY_ENVELOPE_SCHEMA_VERSION,
    ValidatorHotkeyBootstrapV2Error,
    kms_key_reference_hash,
    provision_validator_hotkey_v2,
)


HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
PUBLIC_KEY = "a6bfe69c29bf9e4db65c63ac6f6d1e23c252ca871744afb6edc5623d9bc39004"


def _configuration():
    return {
        "schema_version": HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
        "validator_hotkey": HOTKEY,
        "hotkey_public_key": PUBLIC_KEY,
        "chain_signing_profile_hash": "sha256:" + "1" * 64,
        "drand_library_path": MEASURED_DRAND_LIBRARY_PATH,
        "drand_library_sha256": "2" * 64,
    }


def _envelope():
    ciphertext = b"kms-encrypted-validator-seed"
    context = {"service": "leadpoet", "purpose": "validator-hotkey-v2"}
    return {
        "schema_version": HOTKEY_ENVELOPE_SCHEMA_VERSION,
        "validator_hotkey": HOTKEY,
        "hotkey_public_key": PUBLIC_KEY,
        "ciphertext_blob_b64": base64.b64encode(ciphertext).decode(),
        "ciphertext_blob_hash": sha256_bytes(ciphertext),
        "kms_key_id_hash": kms_key_reference_hash("kms-key-1"),
        "encryption_context": context,
        "encryption_context_hash": sha256_json(context),
    }


class Client:
    def __init__(self):
        self.forwarded = None
        self.state = {
            "validator_hotkey": HOTKEY,
            "hotkey_public_key": PUBLIC_KEY,
            "provisioned": False,
        }

    def configure_hotkey_authority_v2(self, configuration, expected_config_hash):
        self.configuration = dict(configuration)
        self.expected_config_hash = expected_config_hash
        return dict(self.state)

    def get_hotkey_recipient_v2(self):
        return {
            "schema_version": HOTKEY_RECIPIENT_SCHEMA_VERSION,
            "purpose": "leadpoet.validator_hotkey_unseal.v2",
            "validator_hotkey": HOTKEY,
            "key_encryption_algorithm": HOTKEY_ENCRYPTION_ALGORITHM,
            "attestation_document_b64": base64.b64encode(b"nitro").decode(),
        }

    def provision_hotkey_v2(self, ciphertext):
        self.forwarded = base64.b64decode(ciphertext)
        self.state["provisioned"] = True
        return dict(self.state)

    def get_hotkey_state_v2(self):
        return dict(self.state)


class KMS:
    def __init__(self, response=None):
        self.request = None
        self.response = response or {
            "KeyId": "kms-key-1",
            "CiphertextForRecipient": b"recipient-ciphertext",
        }

    def decrypt(self, **request):
        self.request = request
        return self.response


def test_parent_relays_only_validator_kms_ciphertexts():
    client = Client()
    kms = KMS()
    result = provision_validator_hotkey_v2(
        hotkey_configuration=_configuration(),
        hotkey_envelope=_envelope(),
        client=client,
        kms_client=kms,
    )
    assert result["provisioned"] is True
    assert kms.request == {
        "CiphertextBlob": b"kms-encrypted-validator-seed",
        "EncryptionContext": {
            "purpose": "validator-hotkey-v2",
            "service": "leadpoet",
        },
        "Recipient": {
            "KeyEncryptionAlgorithm": HOTKEY_ENCRYPTION_ALGORITHM,
            "AttestationDocument": b"nitro",
        },
    }
    assert client.forwarded == b"recipient-ciphertext"


def test_parent_rejects_plaintext_or_wrong_kms_key():
    with pytest.raises(ValidatorHotkeyBootstrapV2Error, match="plaintext"):
        provision_validator_hotkey_v2(
            hotkey_configuration=_configuration(),
            hotkey_envelope=_envelope(),
            client=Client(),
            kms_client=KMS(
                {
                    "KeyId": "kms-key-1",
                    "Plaintext": b"forbidden",
                    "CiphertextForRecipient": b"ciphertext",
                }
            ),
        )
    with pytest.raises(ValidatorHotkeyBootstrapV2Error, match="key differs"):
        provision_validator_hotkey_v2(
            hotkey_configuration=_configuration(),
            hotkey_envelope=_envelope(),
            client=Client(),
            kms_client=KMS(
                {
                    "KeyId": "wrong-key",
                    "CiphertextForRecipient": b"ciphertext",
                }
            ),
        )


def test_hotkey_envelope_must_match_measured_configuration():
    envelope = _envelope()
    envelope["hotkey_public_key"] = "3" * 64
    with pytest.raises(ValidatorHotkeyBootstrapV2Error, match="differs"):
        provision_validator_hotkey_v2(
            hotkey_configuration=_configuration(),
            hotkey_envelope=envelope,
            client=Client(),
            kms_client=KMS(),
        )
