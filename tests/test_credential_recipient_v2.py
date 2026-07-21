import base64
import hashlib

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical import credential_recipient_v2 as recipient_module


MINER = "5" + "M" * 47


def _recipient(kind="runtime"):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    claim = {
        "schema_version": recipient_module.OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION,
        "purpose": recipient_module.OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
        "boot_identity_hash": "sha256:" + "b" * 64,
        "miner_hotkey_hash": "sha256:" + hashlib.sha256(MINER.encode()).hexdigest(),
        "credential_kind": kind,
        "credential_slot": (
            "openrouter" if kind == "runtime" else "openrouter_management"
        ),
        "recipient_public_key_hash": "sha256:" + hashlib.sha256(public_der).hexdigest(),
        "request_nonce": "a" * 32,
    }
    request_id = sha256_json(claim)
    recipient = {
        **claim,
        "request_id": request_id,
        "recipient_public_key_der_b64": base64.b64encode(public_der).decode(),
        "attestation_document_b64": base64.b64encode(b"attestation").decode(),
        "key_encryption_algorithm": recipient_module.KMS_KEY_ENCRYPTION_ALGORITHM,
    }
    attestation = {
        "attestation_public_key": public_der.hex(),
        "user_data": {
            "schema_version": recipient_module.OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION,
            "purpose": recipient_module.OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
            "claim_hash": request_id,
        },
    }
    return private_key, recipient, attestation


def test_verified_recipient_encrypts_credential_locally(monkeypatch):
    private_key, recipient, attestation = _recipient()
    monkeypatch.setattr(
        recipient_module,
        "verify_nitro_attestation_full",
        lambda **_kwargs: (True, attestation),
    )

    encrypted = recipient_module.verify_and_encrypt_openrouter_credential_v2(
        recipient,
        "sk-or-v1-secret",
        miner_hotkey=MINER,
        credential_kind="runtime",
    )

    plaintext = private_key.decrypt(
        base64.b64decode(encrypted["ciphertext_b64"], validate=True),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    assert plaintext == b"sk-or-v1-secret"
    assert encrypted["request_id"] == recipient["request_id"]


def test_client_policy_matches_gateway_recipient_policy():
    from gateway.tee import kms_recipient_v2 as gateway_recipient

    assert (
        recipient_module.OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION
        == gateway_recipient.OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION
    )
    assert (
        recipient_module.OPENROUTER_INGRESS_RECIPIENT_PURPOSE
        == gateway_recipient.OPENROUTER_INGRESS_RECIPIENT_PURPOSE
    )
    assert (
        recipient_module.KMS_KEY_ENCRYPTION_ALGORITHM
        == gateway_recipient.KMS_KEY_ENCRYPTION_ALGORITHM
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(miner_hotkey_hash="sha256:" + "c" * 64), "miner binding"),
        (lambda value: value.update(credential_slot="openrouter_management"), "policy"),
        (lambda value: value.update(request_id="sha256:" + "d" * 64), "claim hash"),
    ],
)
def test_recipient_rejects_rebinding(monkeypatch, mutation, message):
    _private_key, recipient, attestation = _recipient()
    mutation(recipient)
    monkeypatch.setattr(
        recipient_module,
        "verify_nitro_attestation_full",
        lambda **_kwargs: (True, attestation),
    )
    with pytest.raises(recipient_module.CredentialRecipientV2Error, match=message):
        recipient_module.verify_and_encrypt_openrouter_credential_v2(
            recipient,
            "sk-or-v1-secret",
            miner_hotkey=MINER,
            credential_kind="runtime",
        )


def test_recipient_requires_valid_nitro_attestation(monkeypatch):
    _private_key, recipient, _attestation = _recipient()
    monkeypatch.setattr(
        recipient_module,
        "verify_nitro_attestation_full",
        lambda **_kwargs: (False, {"error": "invalid"}),
    )
    with pytest.raises(
        recipient_module.CredentialRecipientV2Error,
        match="Nitro attestation",
    ):
        recipient_module.verify_and_encrypt_openrouter_credential_v2(
            recipient,
            "sk-or-v1-secret",
            miner_hotkey=MINER,
            credential_kind="runtime",
        )
