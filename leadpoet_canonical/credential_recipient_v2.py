"""Client verification and encryption for attested credential recipients."""

from __future__ import annotations

import base64
import hashlib
from typing import Any, Mapping

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from leadpoet_canonical.nitro import verify_nitro_attestation_full


OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION = (
    "leadpoet.openrouter_ingress_recipient.v2"
)
OPENROUTER_INGRESS_RECIPIENT_PURPOSE = (
    "leadpoet.openrouter_credential_ingress.v2"
)
KMS_KEY_ENCRYPTION_ALGORITHM = "RSAES_OAEP_SHA_256"

_REQUIRED_FIELDS = {
    "schema_version",
    "purpose",
    "request_id",
    "boot_identity_hash",
    "miner_hotkey_hash",
    "credential_kind",
    "credential_slot",
    "recipient_public_key_hash",
    "request_nonce",
    "recipient_public_key_der_b64",
    "attestation_document_b64",
    "key_encryption_algorithm",
}
_CLAIM_FIELDS = (
    "schema_version",
    "purpose",
    "boot_identity_hash",
    "miner_hotkey_hash",
    "credential_kind",
    "credential_slot",
    "recipient_public_key_hash",
    "request_nonce",
)


class CredentialRecipientV2Error(ValueError):
    """An attested recipient is invalid or cannot safely carry a credential."""


def verify_and_encrypt_openrouter_credential_v2(
    recipient: Mapping[str, Any],
    credential: str,
    *,
    miner_hotkey: str,
    credential_kind: str,
) -> dict[str, str]:
    """Verify a one-use Nitro recipient and encrypt one credential locally."""

    expected_slot = {
        "runtime": "openrouter",
        "management": "openrouter_management",
    }.get(credential_kind)
    if expected_slot is None:
        raise CredentialRecipientV2Error("OpenRouter credential kind is invalid")
    if not isinstance(recipient, Mapping) or set(recipient) != _REQUIRED_FIELDS:
        raise CredentialRecipientV2Error("OpenRouter recipient fields are invalid")
    if (
        recipient.get("schema_version")
        != OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION
        or recipient.get("purpose") != OPENROUTER_INGRESS_RECIPIENT_PURPOSE
        or recipient.get("key_encryption_algorithm")
        != KMS_KEY_ENCRYPTION_ALGORITHM
        or recipient.get("credential_kind") != credential_kind
        or recipient.get("credential_slot") != expected_slot
    ):
        raise CredentialRecipientV2Error("OpenRouter recipient policy is invalid")

    expected_miner_hash = "sha256:" + hashlib.sha256(
        miner_hotkey.encode("utf-8")
    ).hexdigest()
    if recipient.get("miner_hotkey_hash") != expected_miner_hash:
        raise CredentialRecipientV2Error("OpenRouter recipient miner binding differs")

    encoded_credential = credential.encode("utf-8")
    if not encoded_credential or b"\x00" in encoded_credential:
        raise CredentialRecipientV2Error("OpenRouter credential is empty or invalid")
    try:
        public_der = base64.b64decode(
            str(recipient["recipient_public_key_der_b64"]), validate=True
        )
    except Exception as exc:
        raise CredentialRecipientV2Error(
            "OpenRouter recipient public key is invalid"
        ) from exc
    public_hash = "sha256:" + hashlib.sha256(public_der).hexdigest()
    if public_hash != recipient.get("recipient_public_key_hash"):
        raise CredentialRecipientV2Error(
            "OpenRouter recipient public key hash differs"
        )

    claim = {name: recipient[name] for name in _CLAIM_FIELDS}
    request_id = sha256_json(claim)
    if request_id != recipient.get("request_id"):
        raise CredentialRecipientV2Error("OpenRouter recipient claim hash differs")

    valid, attestation = verify_nitro_attestation_full(
        attestation_b64=str(recipient["attestation_document_b64"]),
        expected_pubkey=None,
        expected_purpose=OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
        role="gateway",
    )
    if not valid:
        raise CredentialRecipientV2Error(
            "OpenRouter recipient Nitro attestation verification failed"
        )
    expected_user_data = {
        "schema_version": OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION,
        "purpose": OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
        "claim_hash": request_id,
    }
    if (
        attestation.get("attestation_public_key") != public_der.hex()
        or attestation.get("user_data") != expected_user_data
        or canonical_json(attestation.get("user_data"))
        != canonical_json(expected_user_data)
    ):
        raise CredentialRecipientV2Error(
            "OpenRouter attestation is not bound to the recipient claim"
        )

    try:
        public_key = serialization.load_der_public_key(public_der)
    except Exception as exc:
        raise CredentialRecipientV2Error(
            "OpenRouter recipient RSA key is invalid"
        ) from exc
    if not isinstance(public_key, rsa.RSAPublicKey) or public_key.key_size < 2048:
        raise CredentialRecipientV2Error(
            "OpenRouter recipient key policy is invalid"
        )
    max_plaintext = (
        (public_key.key_size // 8) - (2 * hashes.SHA256.digest_size) - 2
    )
    if len(encoded_credential) > max_plaintext:
        raise CredentialRecipientV2Error(
            "OpenRouter credential exceeds the attested RSA-OAEP limit"
        )
    ciphertext = public_key.encrypt(
        encoded_credential,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return {
        "request_id": request_id,
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }
