"""Validate and KMS-encrypt miner-owned Lab Arena runtime credentials.

The OpenRouter management key is an admission-only proof.  It is never
returned from this module, encrypted, or stored.  Runtime callers can decrypt
only the submitted OpenRouter runtime key and Deepline workspace key.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


OPENROUTER_CURRENT_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/keys/{key_hash}"
DEEPLINE_BALANCE_URL = "https://code.deepline.com/api/v2/billing/balance"
MAX_PROBE_RESPONSE_BYTES = 1_048_576
MAX_CREDENTIAL_BYTES = 4096

_OPENROUTER_KEY_RE = re.compile(r"^sk-or-v1-[A-Za-z0-9_-]{24,}$")
_SUBMISSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_HOTKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")


class _RejectRedirect(urlrequest.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


_DIRECT_URLOPEN = urlrequest.build_opener(
    urlrequest.ProxyHandler({}), _RejectRedirect()
).open

RUNTIME_PROVIDERS = ("openrouter", "deepline")
SUBMITTED_CREDENTIAL_FIELDS = (
    "openrouter_api_key",
    "openrouter_management_key",
    "deepline_api_key",
)


class CredentialError(RuntimeError):
    """A safe credential admission/runtime error with no secret in its text."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = str(code)
        self.retryable = bool(retryable)
        super().__init__(self.code)


class _ProbeError(Exception):
    def __init__(self, status: Optional[int]) -> None:
        self.status = status
        super().__init__(status)


HttpGet = Callable[[str, str, int], Mapping[str, Any]]


def _default_http_get(url: str, bearer_token: str, timeout_seconds: int) -> Mapping[str, Any]:
    request = urlrequest.Request(
        url,
        headers={"Authorization": "Bearer " + bearer_token, "Accept": "application/json"},
        method="GET",
    )
    try:
        with _DIRECT_URLOPEN(request, timeout=int(timeout_seconds)) as response:
            raw = response.read(MAX_PROBE_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise _ProbeError(int(exc.code)) from None
    except (OSError, URLError):
        raise _ProbeError(None) from None
    if len(raw) > MAX_PROBE_RESPONSE_BYTES:
        raise _ProbeError(None)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise _ProbeError(None) from None
    if not isinstance(document, Mapping):
        raise _ProbeError(None)
    return dict(document)


def _bounded_secret(value: Any, code: str) -> str:
    if not isinstance(value, str):
        raise CredentialError(code)
    secret = value.strip()
    if value != secret:
        raise CredentialError(code)
    encoded = secret.encode("utf-8")
    if not 16 <= len(encoded) <= MAX_CREDENTIAL_BYTES:
        raise CredentialError(code)
    if any(character.isspace() or ord(character) < 0x20 for character in secret):
        raise CredentialError(code)
    return secret


def _openrouter_key(value: Any, code: str) -> str:
    key = _bounded_secret(value, code)
    if not _OPENROUTER_KEY_RE.fullmatch(key):
        raise CredentialError(code)
    return key


def _identity(submission_id: str, miner_hotkey: str) -> tuple[str, str]:
    submission = str(submission_id or "")
    hotkey = str(miner_hotkey or "")
    if not _SUBMISSION_ID_RE.fullmatch(submission) or not _HOTKEY_RE.fullmatch(hotkey):
        raise CredentialError("credential_identity_invalid")
    return submission, hotkey


def kms_encryption_context(
    *, submission_id: str, miner_hotkey: str, provider: str
) -> dict[str, str]:
    submission, hotkey = _identity(submission_id, miner_hotkey)
    if provider not in RUNTIME_PROVIDERS:
        raise CredentialError("miner_provider_not_configured")
    return {
        "purpose": "leadpoet_lab_arena_miner_runtime_credential",
        "submission_id": submission,
        "miner_hotkey": hotkey,
        "credential_kind": "openrouter_runtime" if provider == "openrouter" else "deepline",
    }


class CredentialManager:
    """Admission validation and narrowly scoped runtime decryption."""

    def __init__(
        self,
        *,
        kms_key_id: str,
        kms_client: Any = None,
        http_get: HttpGet = _default_http_get,
        timeout_seconds: int = 12,
    ) -> None:
        if not str(kms_key_id or "").strip():
            raise CredentialError("credential_kms_unavailable", retryable=True)
        if not callable(http_get) or not 1 <= int(timeout_seconds) <= 60:
            raise CredentialError("credential_configuration_invalid")
        if kms_client is None:
            try:
                import boto3  # type: ignore
            except Exception:
                raise CredentialError("credential_kms_unavailable", retryable=True) from None
            kms_client = boto3.client("kms")
        self._kms_key_id = str(kms_key_id).strip()
        self._kms = kms_client
        self._http_get = http_get
        self._timeout_seconds = int(timeout_seconds)

    def _probe(self, url: str, secret: str, invalid_code: str) -> Mapping[str, Any]:
        try:
            return self._http_get(url, secret, self._timeout_seconds)
        except _ProbeError as exc:
            if exc.status in (401, 403, 404):
                raise CredentialError(invalid_code) from None
            raise CredentialError("credential_validation_unavailable", retryable=True) from None
        except CredentialError:
            raise
        except Exception:
            raise CredentialError("credential_validation_unavailable", retryable=True) from None

    def _validate_openrouter(self, runtime_key: str, management_key: str) -> None:
        current = self._probe(
            OPENROUTER_CURRENT_KEY_URL,
            runtime_key,
            "openrouter_api_key_invalid",
        )
        current_data = current.get("data")
        if not isinstance(current_data, Mapping):
            raise CredentialError("openrouter_api_key_invalid")
        if current_data.get("disabled") is True or current_data.get("is_management_key") is not False:
            raise CredentialError("openrouter_api_key_invalid")
        remaining = current_data.get("limit_remaining")
        if isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and remaining <= 0:
            raise CredentialError("openrouter_api_key_no_credit")

        runtime_hash = hashlib.sha256(runtime_key.encode("utf-8")).hexdigest()
        managed = self._probe(
            OPENROUTER_KEY_URL.format(key_hash=runtime_hash),
            management_key,
            "openrouter_management_key_invalid",
        )
        managed_data = managed.get("data")
        if (
            not isinstance(managed_data, Mapping)
            or managed_data.get("hash") != runtime_hash
            or managed_data.get("disabled") is True
        ):
            raise CredentialError("openrouter_management_key_invalid")

    def _validate_deepline(self, deepline_key: str) -> None:
        self._probe(DEEPLINE_BALANCE_URL, deepline_key, "deepline_api_key_invalid")

    def _encrypt(
        self, plaintext: str, *, submission_id: str, miner_hotkey: str, provider: str
    ) -> str:
        try:
            response = self._kms.encrypt(
                KeyId=self._kms_key_id,
                Plaintext=plaintext.encode("utf-8"),
                EncryptionContext=kms_encryption_context(
                    submission_id=submission_id,
                    miner_hotkey=miner_hotkey,
                    provider=provider,
                ),
            )
        except CredentialError:
            raise
        except Exception:
            raise CredentialError("credential_kms_unavailable", retryable=True) from None
        ciphertext = response.get("CiphertextBlob") if isinstance(response, Mapping) else None
        if not isinstance(ciphertext, (bytes, bytearray)) or not ciphertext:
            raise CredentialError("credential_kms_unavailable", retryable=True)
        return base64.b64encode(bytes(ciphertext)).decode("ascii")

    def validate_and_encrypt(
        self,
        credentials: Mapping[str, Any],
        *,
        submission_id: str,
        miner_hotkey: str,
    ) -> dict[str, str]:
        """Validate all submitted keys and return only runtime ciphertexts."""

        _identity(submission_id, miner_hotkey)
        if not isinstance(credentials, Mapping) or set(credentials) != set(
            SUBMITTED_CREDENTIAL_FIELDS
        ):
            raise CredentialError("submission_credentials_invalid")
        runtime_key = _openrouter_key(
            credentials.get("openrouter_api_key"), "openrouter_api_key_invalid"
        )
        management_key = _openrouter_key(
            credentials.get("openrouter_management_key"),
            "openrouter_management_key_invalid",
        )
        deepline_key = _bounded_secret(
            credentials.get("deepline_api_key"), "deepline_api_key_invalid"
        )

        self._validate_openrouter(runtime_key, management_key)
        self._validate_deepline(deepline_key)
        return {
            "openrouter": self._encrypt(
                runtime_key,
                submission_id=submission_id,
                miner_hotkey=miner_hotkey,
                provider="openrouter",
            ),
            "deepline": self._encrypt(
                deepline_key,
                submission_id=submission_id,
                miner_hotkey=miner_hotkey,
                provider="deepline",
            ),
        }

    def runtime_key(self, row: Mapping[str, Any], provider: str) -> str:
        """Decrypt one provider runtime key; management has no runtime slot."""

        if provider not in RUNTIME_PROVIDERS:
            raise CredentialError("miner_provider_not_configured")
        if not isinstance(row, Mapping) or row.get("provider") != provider:
            raise CredentialError("miner_credentials_unavailable")
        submission_id, miner_hotkey = _identity(
            str(row.get("submission_id") or ""), str(row.get("miner_hotkey") or "")
        )
        ciphertext_b64 = row.get("ciphertext_b64")
        if not isinstance(ciphertext_b64, str):
            raise CredentialError("miner_credentials_unavailable")
        try:
            ciphertext = base64.b64decode(ciphertext_b64.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError, binascii.Error):
            raise CredentialError("miner_credentials_unavailable") from None
        try:
            response = self._kms.decrypt(
                KeyId=self._kms_key_id,
                CiphertextBlob=ciphertext,
                EncryptionContext=kms_encryption_context(
                    submission_id=submission_id,
                    miner_hotkey=miner_hotkey,
                    provider=provider,
                ),
            )
        except Exception:
            raise CredentialError("credential_kms_unavailable", retryable=True) from None
        plaintext = response.get("Plaintext") if isinstance(response, Mapping) else None
        if not isinstance(plaintext, (bytes, bytearray)) or not plaintext:
            raise CredentialError("miner_credentials_unavailable")
        try:
            value = bytes(plaintext).decode("utf-8")
        except UnicodeDecodeError:
            raise CredentialError("miner_credentials_unavailable") from None
        if provider == "openrouter":
            return _openrouter_key(value, "miner_credentials_unavailable")
        return _bounded_secret(value, "miner_credentials_unavailable")


__all__ = [
    "CredentialError",
    "CredentialManager",
    "DEEPLINE_BALANCE_URL",
    "OPENROUTER_CURRENT_KEY_URL",
    "OPENROUTER_KEY_URL",
    "RUNTIME_PROVIDERS",
    "SUBMITTED_CREDENTIAL_FIELDS",
    "kms_encryption_context",
]
