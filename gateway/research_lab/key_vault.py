"""Encrypted OpenRouter key handling for hosted Research Lab runs."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib import parse as urlparse
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from gateway.research_lab.store import canonical_hash


OPENROUTER_KEY_RE = re.compile(r"^sk-or-v1-[A-Za-z0-9_-]{24,}$")
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_INFO_URL = f"{OPENROUTER_API_BASE_URL}/key"
OPENROUTER_LOGGING_DISABLED_PATCH: dict[str, bool] = {
    "is_observability_io_logging_enabled": False,
    "is_data_discount_logging_enabled": False,
    "is_observability_broadcast_enabled": False,
}
OPENROUTER_LOGGING_FLAG_FIELDS = tuple(OPENROUTER_LOGGING_DISABLED_PATCH)
# Workspace input/output logging is enforced with the management API immediately
# before every hidden call. Keep request policy routeable for the configured
# production models; live OpenRouter probes showed global `zdr` and
# `require_parameters` reject those routes before prompts can run.
STRICT_OPENROUTER_PROVIDER_POLICY: dict[str, Any] = {
    "data_collection": "deny",
    "allow_fallbacks": False,
}


class OpenRouterKeyVaultError(RuntimeError):
    """Raised when OpenRouter key registration or decryption fails."""


def validate_openrouter_key_format(raw_key: str) -> str:
    value = (raw_key or "").strip()
    prefix = "sk-or-v1-"
    if value[: len(prefix)].lower() == prefix and value[: len(prefix)] != prefix:
        value = prefix + value[len(prefix) :]
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
        "key_label_hash": _optional_hash(data.get("label")),
        "creator_user_id_hash": _optional_hash(data.get("creator_user_id")),
        "limit": data.get("limit"),
        "limit_remaining": data.get("limit_remaining"),
        "limit_reset": data.get("limit_reset"),
        "usage": data.get("usage"),
        "is_free_tier": data.get("is_free_tier"),
        "is_management_key": data.get("is_management_key"),
        "expires_at": data.get("expires_at"),
    }


def verify_openrouter_workspace_privacy(
    *,
    runtime_key: str,
    management_key: str,
    timeout_seconds: int = 15,
    stage: str = "registration",
    request_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the runtime key's workspace to be controlled by the management key.

    This function stores and returns only redacted proof metadata. It PATCHes the
    workspace logging flags off, then GET-verifies the flags are off before the
    caller may send hidden Research Lab prompts.
    """
    runtime = validate_openrouter_key_format(runtime_key)
    management = validate_openrouter_key_format(management_key)
    key_doc = preflight_openrouter_key(runtime, timeout_seconds=timeout_seconds)
    runtime_label_hash = key_doc.get("key_label_hash")
    runtime_creator_hash = key_doc.get("creator_user_id_hash")
    if not runtime_label_hash or not runtime_creator_hash:
        raise OpenRouterKeyVaultError("OpenRouter runtime key metadata is missing workspace match fields")

    workspace = _find_runtime_key_workspace(
        management_key=management,
        runtime_label_hash=str(runtime_label_hash),
        runtime_creator_hash=str(runtime_creator_hash),
        timeout_seconds=timeout_seconds,
    )
    workspace_id = str(workspace["id"])
    _openrouter_api_request(
        f"/workspaces/{urlparse.quote(workspace_id, safe='')}",
        bearer_token=management,
        method="PATCH",
        payload=OPENROUTER_LOGGING_DISABLED_PATCH,
        timeout_seconds=timeout_seconds,
    )
    get_doc = _openrouter_api_request(
        f"/workspaces/{urlparse.quote(workspace_id, safe='')}",
        bearer_token=management,
        method="GET",
        timeout_seconds=timeout_seconds,
    )
    workspace_after = _workspace_data(get_doc)
    if not workspace_after or str(workspace_after.get("id") or "") != workspace_id:
        raise OpenRouterKeyVaultError("OpenRouter workspace verification returned the wrong workspace")
    flags = _extract_logging_flags(workspace_after)
    unsafe = {key: value for key, value in flags.items() if key in OPENROUTER_LOGGING_FLAG_FIELDS and value is True}
    if unsafe:
        raise OpenRouterKeyVaultError("OpenRouter workspace logging could not be verified off")

    policy = dict(request_policy or STRICT_OPENROUTER_PROVIDER_POLICY)
    proof_doc = {
        "source": "openrouter_workspace_privacy_guard",
        "stage": str(stage or "unknown"),
        "workspace_id_hash": _local_key_hash(workspace_id),
        "runtime_key_hash": str(key_doc["key_hash"]),
        "runtime_key_label_hash": runtime_label_hash,
        "runtime_key_creator_user_id_hash": runtime_creator_hash,
        "management_key_hash": _local_key_hash(management),
        "logging_flags": flags,
        "request_policy": policy,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    proof_doc["proof_hash"] = canonical_hash(proof_doc)
    return proof_doc


def strict_openrouter_provider_policy() -> dict[str, Any]:
    return dict(STRICT_OPENROUTER_PROVIDER_POLICY)


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


def openrouter_key_ref(*, miner_hotkey: str, key_hash: str, management_key_hash: str = "") -> str:
    stable = hashlib.sha256(
        f"{miner_hotkey}:{key_hash}:{management_key_hash}".encode("utf-8")
    ).hexdigest()[:32]
    return f"encrypted_ref:openrouter:{stable}"


def _local_key_hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _optional_hash(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return _local_key_hash(text) if text else None


def _kms_encryption_context(*, miner_hotkey: str, key_ref: str) -> dict[str, str]:
    return {
        "purpose": "leadpoet_research_lab_openrouter_key",
        "miner_hotkey": str(miner_hotkey),
        "key_ref": str(key_ref),
    }


def _openrouter_api_request(
    path: str,
    *,
    bearer_token: str,
    method: str = "GET",
    params: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    url = f"{OPENROUTER_API_BASE_URL}{path}"
    if params:
        url = f"{url}?{urlparse.urlencode(params)}"
    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, data=data, headers=headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=int(timeout_seconds)) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise OpenRouterKeyVaultError("OpenRouter management key is invalid or unauthorized") from exc
        raise OpenRouterKeyVaultError(f"OpenRouter workspace privacy check failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise OpenRouterKeyVaultError(f"OpenRouter workspace privacy check failed: {exc.reason}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenRouterKeyVaultError("OpenRouter workspace privacy check returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise OpenRouterKeyVaultError("OpenRouter workspace privacy check returned no JSON object")
    return dict(decoded)


def _find_runtime_key_workspace(
    *,
    management_key: str,
    runtime_label_hash: str,
    runtime_creator_hash: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    workspaces_doc = _openrouter_api_request(
        "/workspaces",
        bearer_token=management_key,
        timeout_seconds=timeout_seconds,
    )
    workspaces = workspaces_doc.get("data")
    if not isinstance(workspaces, list) or not workspaces:
        raise OpenRouterKeyVaultError("OpenRouter management key does not expose any workspaces")
    for workspace in workspaces:
        if not isinstance(workspace, Mapping):
            continue
        workspace_id = str(workspace.get("id") or "").strip()
        if not workspace_id:
            continue
        keys_doc = _openrouter_api_request(
            "/keys",
            bearer_token=management_key,
            params={"workspace_id": workspace_id},
            timeout_seconds=timeout_seconds,
        )
        keys = keys_doc.get("data")
        if not isinstance(keys, list):
            continue
        for key_doc in keys:
            if not isinstance(key_doc, Mapping):
                continue
            if (
                _optional_hash(key_doc.get("label")) == runtime_label_hash
                and _optional_hash(key_doc.get("creator_user_id")) == runtime_creator_hash
            ):
                return dict(workspace)
    raise OpenRouterKeyVaultError("OpenRouter management key does not control the runtime key workspace")


def _workspace_data(doc: Mapping[str, Any]) -> dict[str, Any]:
    data = doc.get("data")
    if isinstance(data, Mapping):
        return dict(data)
    if isinstance(doc, Mapping) and any(key in doc for key in OPENROUTER_LOGGING_FLAG_FIELDS):
        return dict(doc)
    return {}


def _extract_logging_flags(workspace_doc: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in OPENROUTER_LOGGING_FLAG_FIELDS if key not in workspace_doc]
    if missing:
        raise OpenRouterKeyVaultError("OpenRouter workspace response did not include required logging flags")
    flags = {key: bool(workspace_doc.get(key)) for key in OPENROUTER_LOGGING_FLAG_FIELDS}
    ids = workspace_doc.get("io_logging_api_key_ids")
    if isinstance(ids, list):
        flags["io_logging_api_key_ids_count"] = len(ids)
    elif ids is None:
        flags["io_logging_api_key_ids_count"] = 0
    sampling_rate = workspace_doc.get("io_logging_sampling_rate")
    if sampling_rate is not None:
        flags["io_logging_sampling_rate"] = sampling_rate
    return flags
