"""Credential admission and runtime isolation for miner submissions."""

from __future__ import annotations

import base64
import hashlib

import pytest
from bittensor_wallet import Keypair

from lab_arena import credentials as cr


SUBMISSION_ID = "sub-credential-test"
MINER_HOTKEY = Keypair.create_from_uri("//credential-miner").ss58_address
RUNTIME_KEY = "sk-or-v1-" + "r" * 32
MANAGEMENT_KEY = "sk-or-v1-" + "m" * 32
DEEPLINE_KEY = "deepline-" + "d" * 32


class FakeKms:
    def __init__(self) -> None:
        self.encrypt_calls = []
        self.decrypt_calls = []
        self.plaintexts = {}
        self.fail_decrypt = False

    def encrypt(self, **kwargs):
        self.encrypt_calls.append(kwargs)
        ciphertext = b"sealed:" + kwargs["Plaintext"]
        self.plaintexts[ciphertext] = kwargs["Plaintext"]
        return {"CiphertextBlob": ciphertext}

    def decrypt(self, **kwargs):
        self.decrypt_calls.append(kwargs)
        if self.fail_decrypt:
            raise RuntimeError("KMS unavailable")
        return {"Plaintext": self.plaintexts[kwargs["CiphertextBlob"]]}


def valid_probe(url, bearer_token, timeout_seconds):
    assert timeout_seconds == 7
    if url == cr.OPENROUTER_CURRENT_KEY_URL:
        assert bearer_token == RUNTIME_KEY
        return {
            "data": {
                "is_management_key": False,
                "disabled": False,
                "limit_remaining": 12.5,
            }
        }
    if url == cr.OPENROUTER_KEY_URL.format(
        key_hash=hashlib.sha256(RUNTIME_KEY.encode()).hexdigest()
    ):
        assert bearer_token == MANAGEMENT_KEY
        return {
            "data": {
                "hash": hashlib.sha256(RUNTIME_KEY.encode()).hexdigest(),
                "disabled": False,
            }
        }
    assert url == cr.DEEPLINE_BALANCE_URL
    assert bearer_token == DEEPLINE_KEY
    return {"balance": 100}


def submitted_credentials():
    return {
        "openrouter_api_key": RUNTIME_KEY,
        "openrouter_management_key": MANAGEMENT_KEY,
        "deepline_api_key": DEEPLINE_KEY,
    }


def test_validation_stores_only_runtime_ciphertexts_and_runtime_decrypt_is_bound():
    kms = FakeKms()
    manager = cr.CredentialManager(
        kms_key_id="alias/lab-arena-miner-keys",
        kms_client=kms,
        http_get=valid_probe,
        timeout_seconds=7,
    )

    encrypted = manager.validate_and_encrypt(
        submitted_credentials(),
        submission_id=SUBMISSION_ID,
        miner_hotkey=MINER_HOTKEY,
    )

    assert set(encrypted) == {"openrouter", "deepline"}
    assert len(kms.encrypt_calls) == 2
    assert all(call["KeyId"] == "alias/lab-arena-miner-keys" for call in kms.encrypt_calls)
    assert all(call["Plaintext"] != MANAGEMENT_KEY.encode() for call in kms.encrypt_calls)
    row = {
        "submission_id": SUBMISSION_ID,
        "miner_hotkey": MINER_HOTKEY,
        "provider": "openrouter",
        "ciphertext_b64": encrypted["openrouter"],
    }
    assert manager.runtime_key(row, "openrouter") == RUNTIME_KEY
    assert kms.decrypt_calls[-1]["KeyId"] == "alias/lab-arena-miner-keys"
    assert kms.decrypt_calls[-1]["EncryptionContext"] == cr.kms_encryption_context(
        submission_id=SUBMISSION_ID,
        miner_hotkey=MINER_HOTKEY,
        provider="openrouter",
    )
    assert MANAGEMENT_KEY not in repr(encrypted)


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda body: body.pop("deepline_api_key"), "submission_credentials_invalid"),
        (lambda body: body.__setitem__("deepline_api_key", "contains space" * 2), "deepline_api_key_invalid"),
        (lambda body: body.__setitem__("openrouter_api_key", "not-openrouter-key"), "openrouter_api_key_invalid"),
    ],
)
def test_invalid_submitted_shapes_fail_before_any_probe(mutate, code):
    calls = []
    manager = cr.CredentialManager(
        kms_key_id="alias/key",
        kms_client=FakeKms(),
        http_get=lambda *args: calls.append(args),
    )
    body = submitted_credentials()
    mutate(body)
    with pytest.raises(cr.CredentialError, match=code):
        manager.validate_and_encrypt(
            body, submission_id=SUBMISSION_ID, miner_hotkey=MINER_HOTKEY
        )
    assert calls == []


def test_management_key_must_resolve_the_exact_runtime_key_hash():
    def mismatched_probe(url, token, timeout):
        result = valid_probe(url, token, 7)
        if url != cr.OPENROUTER_CURRENT_KEY_URL and url != cr.DEEPLINE_BALANCE_URL:
            result = {"data": {"hash": "0" * 64, "disabled": False}}
        return result

    manager = cr.CredentialManager(
        kms_key_id="alias/key",
        kms_client=FakeKms(),
        http_get=mismatched_probe,
        timeout_seconds=7,
    )
    with pytest.raises(cr.CredentialError, match="openrouter_management_key_invalid"):
        manager.validate_and_encrypt(
            submitted_credentials(),
            submission_id=SUBMISSION_ID,
            miner_hotkey=MINER_HOTKEY,
        )


def test_runtime_refuses_management_and_marks_kms_outage_retryable():
    kms = FakeKms()
    manager = cr.CredentialManager(
        kms_key_id="alias/key", kms_client=kms, http_get=valid_probe, timeout_seconds=7
    )
    encrypted = manager.validate_and_encrypt(
        submitted_credentials(), submission_id=SUBMISSION_ID, miner_hotkey=MINER_HOTKEY
    )
    row = {
        "submission_id": SUBMISSION_ID,
        "miner_hotkey": MINER_HOTKEY,
        "provider": "openrouter",
        "ciphertext_b64": encrypted["openrouter"],
    }
    with pytest.raises(cr.CredentialError, match="miner_provider_not_configured"):
        manager.runtime_key(row, "openrouter_management")
    kms.fail_decrypt = True
    with pytest.raises(cr.CredentialError) as caught:
        manager.runtime_key(row, "openrouter")
    assert caught.value.code == "credential_kms_unavailable"
    assert caught.value.retryable is True


def test_redirect_handler_never_forwards_bearer_request():
    assert cr._RejectRedirect().redirect_request(
        object(), None, 302, "redirect", {}, "https://other.example/"
    ) is None


def test_malformed_ciphertext_is_a_nonretryable_miner_credential_error():
    manager = cr.CredentialManager(
        kms_key_id="alias/key", kms_client=FakeKms(), http_get=valid_probe
    )
    row = {
        "submission_id": SUBMISSION_ID,
        "miner_hotkey": MINER_HOTKEY,
        "provider": "deepline",
        "ciphertext_b64": "not-base64!",
    }
    with pytest.raises(cr.CredentialError) as caught:
        manager.runtime_key(row, "deepline")
    assert caught.value.code == "miner_credentials_unavailable"
    assert caught.value.retryable is False
