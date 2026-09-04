"""Encrypted SOURCE_ADD credential handling."""

from __future__ import annotations

import base64

from gateway.research_lab.store import canonical_hash


class SourceAddCredentialVaultError(RuntimeError):
    """Raised when a SOURCE_ADD credential cannot be encrypted or decrypted."""


def encrypt_source_add_credential(
    *,
    raw_credential: str,
    kms_key_id: str,
    miner_hotkey: str,
    adapter_ref: str,
) -> dict[str, str]:
    """Encrypt a SOURCE_ADD provider credential with AWS KMS (W5).

    Same envelope pattern as the OpenRouter vault but with its own purpose
    context; no provider-specific format validation — miner sources bring
    arbitrary key shapes.
    """

    if not kms_key_id:
        raise SourceAddCredentialVaultError("SOURCE_ADD credential KMS key id is required")
    credential = (raw_credential or "").strip()
    if not (8 <= len(credential) <= 512):
        raise SourceAddCredentialVaultError("SOURCE_ADD credential length is out of bounds")
    context = source_add_kms_encryption_context(
        miner_hotkey=miner_hotkey,
        adapter_ref=adapter_ref,
    )
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise SourceAddCredentialVaultError("boto3 is required for SOURCE_ADD credential encryption") from exc
    response = boto3.client("kms").encrypt(
        KeyId=kms_key_id,
        Plaintext=credential.encode("utf-8"),
        EncryptionContext=context,
    )
    ciphertext = response.get("CiphertextBlob")
    if not ciphertext:
        raise SourceAddCredentialVaultError("KMS encryption returned no ciphertext")
    return {
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        "kms_key_id": str(response.get("KeyId") or kms_key_id),
        "encryption_context_hash": canonical_hash(context),
    }


def decrypt_source_add_credential(
    *,
    ciphertext_b64: str,
    miner_hotkey: str,
    adapter_ref: str,
) -> str:
    """Decrypt a SOURCE_ADD credential for use inside the trial proxy only."""

    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise SourceAddCredentialVaultError("boto3 is required for SOURCE_ADD credential decryption") from exc
    try:
        ciphertext = base64.b64decode(ciphertext_b64.encode("ascii"), validate=True)
    except Exception as exc:
        raise SourceAddCredentialVaultError("stored SOURCE_ADD credential ciphertext is invalid base64") from exc
    response = boto3.client("kms").decrypt(
        CiphertextBlob=ciphertext,
        EncryptionContext=source_add_kms_encryption_context(
            miner_hotkey=miner_hotkey,
            adapter_ref=adapter_ref,
        ),
    )
    plaintext = response.get("Plaintext")
    if not plaintext:
        raise SourceAddCredentialVaultError("KMS decryption returned no plaintext")
    return plaintext.decode("utf-8")


def source_add_kms_encryption_context(
    *, miner_hotkey: str, adapter_ref: str
) -> dict[str, str]:
    return {
        "purpose": "leadpoet_research_lab_source_add_credential",
        "miner_hotkey": str(miner_hotkey),
        "adapter_ref": str(adapter_ref),
    }


def _source_add_kms_encryption_context(
    *, miner_hotkey: str, adapter_ref: str
) -> dict[str, str]:
    """Compatibility alias for existing callers."""

    return source_add_kms_encryption_context(
        miner_hotkey=miner_hotkey,
        adapter_ref=adapter_ref,
    )
