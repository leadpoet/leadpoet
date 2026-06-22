"""Encrypted OpenRouter key handling for hosted Research Lab runs."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any, Mapping
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from gateway.research_lab.store import canonical_hash


OPENROUTER_KEY_RE = re.compile(r"^sk-or-v1-[A-Za-z0-9_-]{24,}$")
OPENROUTER_KEY_INFO_URL = "https://openrouter.ai/api/v1/key"


class OpenRouterKeyVaultError(RuntimeError):
    """Raised when OpenRouter key registration or decryption fails."""


def validate_openrouter_key_format(raw_key: str) -> str:
    value = (raw_key or "").strip()
    if not OPENROUTER_KEY_RE.match(value):
        raise OpenRouterKeyVaultError("OpenRouter key must start with sk-or-v1- and look like a valid API key")
    return value


def preflight_openrouter_key(raw_key: str, *, timeout_seconds: int = 12) -> dict[str, Any]:
    """Verify a raw OpenRouter key and return only non-secret metadata."""
    key = validate_openrouter_key_format(raw_key)
    req = urlrequest.Request(
        OPENROUTER_KEY_INFO_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise OpenRouterKeyVaultError("OpenRouter key preflight failed: key is invalid or unauthorized") from exc
        raise OpenRouterKeyVaultError(f"OpenRouter key preflight failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise OpenRouterKeyVaultError(f"OpenRouter key preflight failed: {exc.reason}") from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenRouterKeyVaultError("OpenRouter key preflight returned invalid JSON") from exc
    data = decoded.get("data")
    if not isinstance(data, Mapping):
        raise OpenRouterKeyVaultError("OpenRouter key preflight returned no key metadata")
    if data.get("disabled") is True:
        raise OpenRouterKeyVaultError("OpenRouter key is disabled")
    return {
        "key_hash": str(data.get("hash") or _local_key_hash(key)),
        "limit": data.get("limit"),
        "limit_remaining": data.get("limit_remaining"),
        "limit_reset": data.get("limit_reset"),
        "usage": data.get("usage"),
        "is_free_tier": data.get("is_free_tier"),
        "is_management_key": data.get("is_management_key"),
        "expires_at": data.get("expires_at"),
    }


def encrypt_openrouter_key(
    *,
    raw_key: str,
    kms_key_id: str,
    miner_hotkey: str,
    key_ref: str,
) -> dict[str, str]:
    """Encrypt a raw OpenRouter key with AWS KMS and return storable metadata."""
    if not kms_key_id:
        raise OpenRouterKeyVaultError("RESEARCH_LAB_OPENROUTER_KEY_KMS_KEY_ID is required")
    key = validate_openrouter_key_format(raw_key)
    context = _kms_encryption_context(miner_hotkey=miner_hotkey, key_ref=key_ref)
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise OpenRouterKeyVaultError("boto3 is required for OpenRouter key encryption") from exc
    response = boto3.client("kms").encrypt(
        KeyId=kms_key_id,
        Plaintext=key.encode("utf-8"),
        EncryptionContext=context,
    )
    ciphertext = response.get("CiphertextBlob")
    if not ciphertext:
        raise OpenRouterKeyVaultError("KMS encryption returned no ciphertext")
    return {
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        "kms_key_id": str(response.get("KeyId") or kms_key_id),
        "encryption_context_hash": canonical_hash(context),
    }


def decrypt_openrouter_key(
    *,
    ciphertext_b64: str,
    miner_hotkey: str,
    key_ref: str,
) -> str:
    """Decrypt a KMS-encrypted OpenRouter key for runtime use only."""
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise OpenRouterKeyVaultError("boto3 is required for OpenRouter key decryption") from exc
    try:
        ciphertext = base64.b64decode(ciphertext_b64.encode("ascii"), validate=True)
    except Exception as exc:
        raise OpenRouterKeyVaultError("stored OpenRouter key ciphertext is invalid base64") from exc
    response = boto3.client("kms").decrypt(
        CiphertextBlob=ciphertext,
        EncryptionContext=_kms_encryption_context(miner_hotkey=miner_hotkey, key_ref=key_ref),
    )
    plaintext = response.get("Plaintext")
    if not plaintext:
        raise OpenRouterKeyVaultError("KMS decryption returned no plaintext")
    key = plaintext.decode("utf-8")
    return validate_openrouter_key_format(key)


def openrouter_key_ref(*, miner_hotkey: str, key_hash: str) -> str:
    stable = hashlib.sha256(f"{miner_hotkey}:{key_hash}".encode("utf-8")).hexdigest()[:32]
    return f"encrypted_ref:openrouter:{stable}"


def _local_key_hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _kms_encryption_context(*, miner_hotkey: str, key_ref: str) -> dict[str, str]:
    return {
        "purpose": "leadpoet_research_lab_openrouter_key",
        "miner_hotkey": str(miner_hotkey),
        "key_ref": str(key_ref),
    }
