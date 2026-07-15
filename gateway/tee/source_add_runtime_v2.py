"""Measured runtime contracts for dynamically provisioned SOURCE_ADD providers.

The coordinator authenticates the database snapshot.  Every runner derives the
same non-secret route document from that snapshot, while credential plaintext
is unwrapped only inside the coordinator through a hash-bound KMS recipient.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Dict, Mapping, Sequence
from urllib.parse import urlsplit

from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json


SOURCE_ADD_CREDENTIAL_ENVELOPE_SCHEMA_VERSION = (
    "leadpoet.source_add_credential_envelope.v2"
)
SOURCE_ADD_SEALED_CREDENTIAL_ENVELOPE_SCHEMA_VERSION = (
    "leadpoet.source_add_credential_envelope.enclave.v2"
)
SOURCE_ADD_SEALED_JOB_ENVELOPE_SCHEMA_VERSION = (
    "leadpoet.source_add_job_credential_envelope.enclave.v2"
)
SOURCE_ADD_RUNTIME_ROUTE_SCHEMA_VERSION = "leadpoet.source_add_runtime_route.v2"
SOURCE_ADD_RUNTIME_CATALOG_SCHEMA_VERSION = (
    "leadpoet.source_add_runtime_catalog.v2"
)
SOURCE_ADD_PROBE_CONFIG_SCHEMA_VERSION = "leadpoet.source_add_probe_config.v2"
SOURCE_ADD_DYNAMIC_RETRY_SCHEMA_VERSION = (
    "leadpoet.source_add_dynamic_retry_policy.v2"
)
SOURCE_ADD_JOB_SLOT_PREFIX = "source_add_"

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,79}$")
_ENV_REF_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_CREDENTIAL_REF_RE = re.compile(r"^encrypted_ref:source_add:[0-9a-f]{32}$")
_JOB_SLOT_RE = re.compile(r"^source_add_[0-9a-f]{32}$")
_VALID_AUTH_KINDS = frozenset({"header", "query", "bearer", "none"})
_VALID_METHODS = frozenset({"GET", "POST"})
_RESERVED_PROVIDER_IDS = frozenset(
    {
        "bittensor_chain",
        "coingecko",
        "deepline",
        "dns",
        "exa",
        "openrouter",
        "openrouter_management",
        "or",
        "public_web",
        "rdap",
        "scrapingdog",
        "sd",
        "supabase",
        "truelist",
        "wayback",
    }
)
_AUTH_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
_FORBIDDEN_AUTH_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "proxy-authorization",
        "transfer-encoding",
    }
)
_SECRET_QUERY_NAMES = frozenset(
    {"access_token", "api-key", "api_key", "apikey", "key", "token"}
)


class SourceAddRuntimeV2Error(RuntimeError):
    """A dynamic provider row is not safe for V2 measured execution."""


def source_add_job_credential_slot(provider_id: str) -> str:
    normalized = str(provider_id or "").strip().lower()
    if not _PROVIDER_ID_RE.fullmatch(normalized):
        raise SourceAddRuntimeV2Error("SOURCE_ADD provider id is invalid")
    digest = sha256_json(
        {
            "schema_version": SOURCE_ADD_RUNTIME_ROUTE_SCHEMA_VERSION,
            "provider_id": normalized,
            "credential_scope": "coordinator_job",
        }
    ).split(":", 1)[1][:32]
    return SOURCE_ADD_JOB_SLOT_PREFIX + digest


def source_add_dynamic_job_slot(value: str) -> bool:
    return bool(_JOB_SLOT_RE.fullmatch(str(value or "")))


def validate_source_add_credential_envelope_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    common_fields = {
        "schema_version",
        "ciphertext_b64",
        "ciphertext_blob_hash",
        "kms_key_id",
        "kms_key_id_hash",
        "encryption_context",
        "encryption_context_hash",
        "credential_ref",
        "credential_value_hash",
        "key_ref_hash",
    }
    if not isinstance(value, Mapping):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 credential envelope fields are invalid"
        )
    schema_version = str(value.get("schema_version") or "")
    if schema_version == SOURCE_ADD_CREDENTIAL_ENVELOPE_SCHEMA_VERSION:
        if set(value) != common_fields:
            raise SourceAddRuntimeV2Error(
                "SOURCE_ADD V2 credential envelope fields are invalid"
            )
        envelope_kind = "kms"
    elif schema_version == SOURCE_ADD_SEALED_CREDENTIAL_ENVELOPE_SCHEMA_VERSION:
        if set(value) != common_fields | {"envelope_kind"} or value.get(
            "envelope_kind"
        ) != "coordinator_sealed":
            raise SourceAddRuntimeV2Error(
                "SOURCE_ADD sealed credential envelope fields are invalid"
            )
        envelope_kind = "coordinator_sealed"
    else:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 credential envelope schema is invalid"
        )
    try:
        ciphertext = base64.b64decode(
            str(value.get("ciphertext_b64") or ""), validate=True
        )
    except (binascii.Error, ValueError) as exc:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 credential ciphertext is invalid"
        ) from exc
    if not 8 <= len(ciphertext) <= 64 * 1024:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 credential ciphertext is outside limit"
        )
    if sha256_bytes(ciphertext) != value.get("ciphertext_blob_hash"):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 credential ciphertext hash differs"
        )
    credential_ref = str(value.get("credential_ref") or "")
    if not _CREDENTIAL_REF_RE.fullmatch(credential_ref):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 credential reference is invalid"
        )
    key_ref_hash = str(value.get("key_ref_hash") or "").lower()
    if key_ref_hash != sha256_bytes(credential_ref.encode("utf-8")):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 key reference commitment differs"
        )
    credential_value_hash = str(
        value.get("credential_value_hash") or ""
    ).lower()
    kms_key_id_hash = str(value.get("kms_key_id_hash") or "").lower()
    if not _HASH_RE.fullmatch(credential_value_hash) or not _HASH_RE.fullmatch(
        kms_key_id_hash
    ):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 credential commitment is invalid"
        )
    kms_key_id = str(value.get("kms_key_id") or "").strip()
    if not kms_key_id or "\x00" in kms_key_id:
        raise SourceAddRuntimeV2Error("SOURCE_ADD V2 KMS key id is invalid")
    context = value.get("encryption_context")
    if not isinstance(context, Mapping) or not context:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 encryption context is missing"
        )
    normalized_context = {
        str(name): str(item) for name, item in sorted(context.items())
    }
    if any(
        not name or not item or "\x00" in name + item
        for name, item in normalized_context.items()
    ):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 encryption context is invalid"
        )
    if sha256_json(normalized_context) != value.get("encryption_context_hash"):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD V2 encryption context hash differs"
        )
    return {
        **dict(value),
        "envelope_kind": envelope_kind,
        "ciphertext_blob": bytes(ciphertext),
        "credential_ref": credential_ref,
        "credential_value_hash": credential_value_hash,
        "key_ref_hash": key_ref_hash,
        "kms_key_id": kms_key_id,
        "kms_key_id_hash": kms_key_id_hash,
        "encryption_context": normalized_context,
    }


def validate_source_add_sealed_job_envelope_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "job_id",
        "credential_slot",
        "credential_ref_hash",
        "credential_value_hash",
        "key_ref_hash",
        "ciphertext_blob_b64",
        "ciphertext_blob_hash",
        "encryption_context",
        "encryption_context_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD sealed job credential fields are invalid"
        )
    if value.get("schema_version") != SOURCE_ADD_SEALED_JOB_ENVELOPE_SCHEMA_VERSION:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD sealed job credential schema is invalid"
        )
    job_id = str(value.get("job_id") or "")
    slot = str(value.get("credential_slot") or "")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}", job_id)
        or not source_add_dynamic_job_slot(slot)
    ):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD sealed job credential scope is invalid"
        )
    try:
        ciphertext = base64.b64decode(
            str(value.get("ciphertext_blob_b64") or ""), validate=True
        )
    except (binascii.Error, ValueError) as exc:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD sealed job credential ciphertext is invalid"
        ) from exc
    if not ciphertext or len(ciphertext) > 64 * 1024:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD sealed job credential ciphertext is outside limit"
        )
    hashes = {
        field: str(value.get(field) or "").lower()
        for field in (
            "credential_ref_hash",
            "credential_value_hash",
            "key_ref_hash",
            "ciphertext_blob_hash",
            "encryption_context_hash",
        )
    }
    if any(not _HASH_RE.fullmatch(item) for item in hashes.values()):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD sealed job credential commitment is invalid"
        )
    if sha256_bytes(ciphertext) != hashes["ciphertext_blob_hash"]:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD sealed job credential ciphertext hash differs"
        )
    context = value.get("encryption_context")
    if not isinstance(context, Mapping) or not context:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD sealed job credential context is invalid"
        )
    normalized_context = {
        str(name): str(item) for name, item in sorted(context.items())
    }
    if sha256_json(normalized_context) != hashes["encryption_context_hash"]:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD sealed job credential context hash differs"
        )
    return {
        **dict(value),
        **hashes,
        "job_id": job_id,
        "credential_slot": slot,
        "ciphertext_blob": ciphertext,
        "encryption_context": normalized_context,
        "envelope_kind": "coordinator_sealed",
    }


def _safe_path(value: Any) -> str:
    path = str(value or "")
    if (
        not path.startswith("/")
        or "?" in path
        or "#" in path
        or "%" in path
        or "\\" in path
        or re.search(r"[{}<>\[\]]|(^|/):[A-Za-z_]", path)
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or any(character.isspace() for character in path)
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        raise SourceAddRuntimeV2Error("SOURCE_ADD route path is unsafe")
    return path


def _safe_header_value(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) <= 500
        and all(ord(character) >= 32 and ord(character) != 127 for character in value)
    )


def _base_url(value: Any) -> tuple[str, str, str]:
    raw = str(value or "").strip().rstrip("/")
    parsed = urlsplit(raw)
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise SourceAddRuntimeV2Error("SOURCE_ADD provider port is invalid") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or port != 443
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD provider base URL must be HTTPS on port 443"
        )
    base_path = (parsed.path or "").rstrip("/")
    if base_path:
        _safe_path(base_path)
    destination_host = parsed.hostname.lower()
    url_host = "[%s]" % destination_host if ":" in destination_host else destination_host
    normalized = "https://" + url_host + base_path
    return normalized, destination_host, base_path


def _int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise SourceAddRuntimeV2Error("%s is invalid" % field)
    try:
        result = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise SourceAddRuntimeV2Error("%s is invalid" % field) from exc
    if result < minimum:
        raise SourceAddRuntimeV2Error("%s is invalid" % field)
    return result


def _validate_probe_body_json(value: Any) -> None:
    node_count = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if depth > 12 or node_count > 2_000:
            raise SourceAddRuntimeV2Error(
                "SOURCE_ADD probe body exceeds structural limits"
            )
        if isinstance(item, Mapping):
            if len(item) > 500:
                raise SourceAddRuntimeV2Error(
                    "SOURCE_ADD probe body has too many keys"
                )
            for key, child in item.items():
                if not isinstance(key, str) or not key or len(key) > 120:
                    raise SourceAddRuntimeV2Error(
                        "SOURCE_ADD probe body key is invalid"
                    )
                visit(child, depth + 1)
            return
        if isinstance(item, list):
            if len(item) > 500:
                raise SourceAddRuntimeV2Error(
                    "SOURCE_ADD probe body list is too large"
                )
            for child in item:
                visit(child, depth + 1)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            if isinstance(item, str) and len(item) > 4_096:
                raise SourceAddRuntimeV2Error(
                    "SOURCE_ADD probe body string is too large"
                )
            return
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD probe body contains an unsupported value"
        )

    visit(value, 0)
    try:
        encoded = canonical_json(value).encode("utf-8")
    except Exception as exc:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD probe body is not canonicalizable"
        ) from exc
    if len(encoded) > 65_536:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD probe body exceeds 64 KiB"
        )


def build_source_add_runtime_route_v2(
    row: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(row, Mapping):
        raise SourceAddRuntimeV2Error("SOURCE_ADD source row is invalid")
    if str(row.get("provision_status") or "") != (
        "provisioned_autoresearch_eligible"
    ):
        raise SourceAddRuntimeV2Error("SOURCE_ADD source row is not eligible")
    provision = row.get("provision_doc")
    if not isinstance(provision, Mapping):
        raise SourceAddRuntimeV2Error("SOURCE_ADD provision document is missing")
    provider = provision.get("provider_registry_entry")
    if not isinstance(provider, Mapping):
        raise SourceAddRuntimeV2Error("SOURCE_ADD provider registry entry is missing")
    provider_id = str(provider.get("id") or "").strip().lower()
    if (
        not _PROVIDER_ID_RE.fullmatch(provider_id)
        or provider_id != str(row.get("registry_provider_id") or "").lower()
    ):
        raise SourceAddRuntimeV2Error("SOURCE_ADD provider identity differs")
    if provider_id in _RESERVED_PROVIDER_IDS:
        raise SourceAddRuntimeV2Error("SOURCE_ADD provider id is reserved")
    normalized_base, destination_host, base_path = _base_url(
        provider.get("base_url")
    )
    auth_kind = str(provider.get("auth_kind") or "none").strip().lower()
    auth_name = str(provider.get("auth_name") or "").strip()
    if auth_kind not in _VALID_AUTH_KINDS:
        raise SourceAddRuntimeV2Error("SOURCE_ADD provider auth kind is invalid")
    if auth_kind in {"header", "query"} and not auth_name:
        raise SourceAddRuntimeV2Error("SOURCE_ADD provider auth name is missing")
    if auth_kind == "bearer" and not auth_name:
        auth_name = "Authorization"
    if auth_kind != "none" and not _AUTH_NAME_RE.fullmatch(auth_name):
        raise SourceAddRuntimeV2Error("SOURCE_ADD provider auth name is invalid")
    if (
        auth_kind in {"header", "bearer"}
        and auth_name.lower() in _FORBIDDEN_AUTH_HEADERS
    ):
        raise SourceAddRuntimeV2Error("SOURCE_ADD provider auth header is unsafe")
    raw_refs = provider.get("credential_ref") or ()
    if isinstance(raw_refs, str):
        raw_refs = (raw_refs,)
    credential_env_refs = sorted(
        {
            str(item)
            for item in raw_refs
            if _ENV_REF_RE.fullmatch(str(item or ""))
        }
    )

    raw_endpoints = provision.get("probe_endpoints")
    if not isinstance(raw_endpoints, list) or not raw_endpoints:
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe endpoint set is empty")
    allowed_routes = set()
    for endpoint in raw_endpoints:
        if not isinstance(endpoint, Mapping):
            raise SourceAddRuntimeV2Error("SOURCE_ADD probe endpoint is invalid")
        if str(endpoint.get("provider_id") or "").lower() != provider_id:
            raise SourceAddRuntimeV2Error(
                "SOURCE_ADD probe endpoint provider differs"
            )
        method = str(endpoint.get("method") or "").upper()
        if method not in _VALID_METHODS:
            raise SourceAddRuntimeV2Error("SOURCE_ADD route method is invalid")
        endpoint_path = _safe_path(endpoint.get("path"))
        allowed_routes.add((method, (base_path + endpoint_path) or "/"))
    policy = provider.get("capability_policy")
    policy_routes = policy.get("routes") if isinstance(policy, Mapping) else None
    if not isinstance(policy_routes, list):
        raise SourceAddRuntimeV2Error("SOURCE_ADD route policy is missing")
    declared_routes = {
        (
            str(item.get("method") or "").upper(),
            (base_path + _safe_path(item.get("path"))) or "/",
        )
        for item in policy_routes
        if isinstance(item, Mapping) and item.get("path")
    }
    if declared_routes != allowed_routes:
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD route policy differs from probe endpoints"
        )
    request_headers = provision.get("request_headers") or {}
    if not isinstance(request_headers, Mapping) or len(request_headers) > 16 or any(
        not _AUTH_NAME_RE.fullmatch(str(name))
        or str(name).lower() in _FORBIDDEN_AUTH_HEADERS
        or str(name).lower() in {"authorization", "x-api-key", "api-key"}
        or (
            auth_kind in {"header", "bearer"}
            and str(name).lower() == auth_name.lower()
        )
        or not _safe_header_value(value)
        for name, value in request_headers.items()
    ):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD provider request headers are invalid"
        )

    credential_slot = ""
    credential_value_hash = ""
    key_ref_hash = ""
    envelope_hash = ""
    if auth_kind != "none":
        envelope = validate_source_add_credential_envelope_v2(
            row.get("credential_envelope") or {}
        )
        expected_context = {
            "adapter_ref": "source_add:%s" % str(row.get("adapter_id") or ""),
            "miner_hotkey": str(row.get("miner_hotkey") or ""),
            "purpose": "leadpoet_research_lab_source_add_credential",
        }
        if (
            not expected_context["adapter_ref"].removeprefix("source_add:")
            or not expected_context["miner_hotkey"]
            or envelope["encryption_context"] != expected_context
        ):
            raise SourceAddRuntimeV2Error(
                "SOURCE_ADD credential context differs from source identity"
            )
        credential_slot = source_add_job_credential_slot(provider_id)
        credential_value_hash = envelope["credential_value_hash"]
        key_ref_hash = envelope["key_ref_hash"]
        envelope_hash = sha256_json(
            {
                key: value
                for key, value in envelope.items()
                if key != "ciphertext_blob"
            }
        )
    elif row.get("credential_envelope"):
        raise SourceAddRuntimeV2Error(
            "unauthenticated SOURCE_ADD route carries a credential envelope"
        )

    cost_model = provider.get("cost_model")
    if not isinstance(cost_model, Mapping):
        cost_model = {}
    body = {
        "schema_version": SOURCE_ADD_RUNTIME_ROUTE_SCHEMA_VERSION,
        "provider_id": provider_id,
        "base_url": normalized_base,
        "destination_host": destination_host,
        "auth_kind": auth_kind,
        "auth_name": auth_name,
        "credential_slot": credential_slot,
        "credential_value_hash": credential_value_hash,
        "key_ref_hash": key_ref_hash,
        "credential_env_refs": credential_env_refs,
        "allowed_routes": [
            {"method": method, "path": path}
            for method, path in sorted(allowed_routes)
        ],
        "request_headers": {
            str(name): str(value)
            for name, value in sorted(request_headers.items())
        },
        "per_day_quota": _int(
            provider.get("per_day_quota"), "SOURCE_ADD per-day quota"
        ),
        "est_cost_microusd_per_call": _int(
            cost_model.get("est_cost_microusd_per_call"),
            "SOURCE_ADD estimated cost",
        ),
        "source_row_hash": sha256_json(dict(row)),
        "credential_envelope_hash": envelope_hash,
    }
    return {**body, "route_hash": sha256_json(body)}


def build_source_add_probe_route_v2(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a provisional measured route without catalog eligibility."""

    if not isinstance(row, Mapping):
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe row is invalid")
    probe = row.get("probe_doc")
    if not isinstance(probe, Mapping) or probe.get("schema_version") != SOURCE_ADD_PROBE_CONFIG_SCHEMA_VERSION:
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe document is invalid")
    expected_fields = {
        "schema_version",
        "provider_id",
        "base_url",
        "auth_kind",
        "auth_name",
        "request_headers",
        "probes",
    }
    if set(probe) != expected_fields:
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe fields are invalid")
    provider_id = str(probe.get("provider_id") or "").strip().lower()
    if not _PROVIDER_ID_RE.fullmatch(provider_id) or provider_id in _RESERVED_PROVIDER_IDS:
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe provider id is invalid")
    normalized_base, destination_host, base_path = _base_url(probe.get("base_url"))
    auth_kind = str(probe.get("auth_kind") or "none").strip().lower()
    auth_name = str(probe.get("auth_name") or "").strip()
    if auth_kind not in _VALID_AUTH_KINDS:
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe auth kind is invalid")
    if auth_kind == "bearer" and not auth_name:
        auth_name = "Authorization"
    if auth_kind in {"header", "query"} and not auth_name:
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe auth name is missing")
    if auth_kind != "none" and not _AUTH_NAME_RE.fullmatch(auth_name):
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe auth name is invalid")
    if auth_kind in {"header", "bearer"} and auth_name.lower() in _FORBIDDEN_AUTH_HEADERS:
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe auth header is unsafe")

    request_headers = probe.get("request_headers")
    if not isinstance(request_headers, Mapping) or len(request_headers) > 16 or any(
        not _AUTH_NAME_RE.fullmatch(str(name))
        or str(name).lower() in _FORBIDDEN_AUTH_HEADERS
        or str(name).lower() in {"authorization", "x-api-key", "api-key"}
        or (
            auth_kind in {"header", "bearer"}
            and str(name).lower() == auth_name.lower()
        )
        or not _safe_header_value(value)
        for name, value in request_headers.items()
    ):
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe request headers are invalid")
    probes = probe.get("probes")
    if not isinstance(probes, list) or not 1 <= len(probes) <= 3:
        raise SourceAddRuntimeV2Error("SOURCE_ADD probe count is invalid")
    allowed_routes = set()
    for item in probes:
        if not isinstance(item, Mapping) or set(item) != {"method", "path", "query", "body_json"}:
            raise SourceAddRuntimeV2Error("SOURCE_ADD probe example is invalid")
        method = str(item.get("method") or "").upper()
        if method not in _VALID_METHODS:
            raise SourceAddRuntimeV2Error("SOURCE_ADD probe method is invalid")
        path = _safe_path(item.get("path"))
        query = item.get("query")
        if not isinstance(query, Mapping) or len(query) > 20 or any(
            not isinstance(name, str)
            or not name
            or len(name) > 120
            or name.lower() in _SECRET_QUERY_NAMES
            or (auth_kind == "query" and name.lower() == auth_name.lower())
            or not isinstance(value, (str, int, float, bool))
            or len(str(value)) > 500
            for name, value in query.items()
        ):
            raise SourceAddRuntimeV2Error("SOURCE_ADD probe query is invalid")
        body_json = item.get("body_json")
        if method == "GET" and body_json not in ({}, None):
            raise SourceAddRuntimeV2Error("SOURCE_ADD GET probe body is forbidden")
        if method == "POST" and not isinstance(body_json, (Mapping, list)):
            raise SourceAddRuntimeV2Error("SOURCE_ADD POST probe body is invalid")
        if method == "POST":
            _validate_probe_body_json(body_json)
        allowed_routes.add((method, (base_path + path) or "/"))

    credential_slot = ""
    credential_value_hash = ""
    key_ref_hash = ""
    envelope_hash = ""
    if auth_kind != "none":
        envelope = validate_source_add_credential_envelope_v2(row.get("credential_envelope") or {})
        expected_context = {
            "adapter_ref": "source_add:%s" % str(row.get("adapter_id") or ""),
            "miner_hotkey": str(row.get("miner_hotkey") or ""),
            "purpose": "leadpoet_research_lab_source_add_credential",
        }
        if envelope["encryption_context"] != expected_context:
            raise SourceAddRuntimeV2Error("SOURCE_ADD probe credential context differs")
        credential_slot = source_add_job_credential_slot(provider_id)
        credential_value_hash = envelope["credential_value_hash"]
        key_ref_hash = envelope["key_ref_hash"]
        envelope_hash = sha256_json(
            {key: value for key, value in envelope.items() if key != "ciphertext_blob"}
        )
    elif row.get("credential_envelope"):
        raise SourceAddRuntimeV2Error("SOURCE_ADD unauthenticated probe carries a credential")

    body = {
        "schema_version": SOURCE_ADD_RUNTIME_ROUTE_SCHEMA_VERSION,
        "provider_id": provider_id,
        "base_url": normalized_base,
        "destination_host": destination_host,
        "auth_kind": auth_kind,
        "auth_name": auth_name,
        "credential_slot": credential_slot,
        "credential_value_hash": credential_value_hash,
        "key_ref_hash": key_ref_hash,
        "credential_env_refs": [],
        "allowed_routes": [
            {"method": method, "path": path}
            for method, path in sorted(allowed_routes)
        ],
        "request_headers": {
            str(name): str(value)
            for name, value in sorted(request_headers.items())
        },
        "per_day_quota": 0,
        "est_cost_microusd_per_call": 0,
        "source_row_hash": sha256_json(
            {
                "submission_id": str(row.get("submission_id") or ""),
                "adapter_id": str(row.get("adapter_id") or ""),
                "config_ref": str(row.get("config_ref") or ""),
                "probe_doc": dict(probe),
            }
        ),
        "credential_envelope_hash": envelope_hash,
    }
    return validate_source_add_runtime_route_v2({**body, "route_hash": sha256_json(body)})


def validate_source_add_runtime_route_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "provider_id",
        "base_url",
        "destination_host",
        "auth_kind",
        "auth_name",
        "credential_slot",
        "credential_value_hash",
        "key_ref_hash",
        "credential_env_refs",
        "allowed_routes",
        "request_headers",
        "per_day_quota",
        "est_cost_microusd_per_call",
        "source_row_hash",
        "credential_envelope_hash",
        "route_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime route fields are invalid")
    body = {key: value[key] for key in fields if key != "route_hash"}
    if body.get("schema_version") != SOURCE_ADD_RUNTIME_ROUTE_SCHEMA_VERSION:
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime route schema is invalid")
    provider_id = str(body.get("provider_id") or "")
    if not _PROVIDER_ID_RE.fullmatch(provider_id):
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime provider id is invalid")
    normalized_base, host, _base_path = _base_url(body.get("base_url"))
    if body.get("base_url") != normalized_base or body.get("destination_host") != host:
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime origin differs")
    auth_kind = str(body.get("auth_kind") or "")
    if auth_kind not in _VALID_AUTH_KINDS:
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime auth kind is invalid")
    slot = str(body.get("credential_slot") or "")
    credential_hash = str(body.get("credential_value_hash") or "")
    key_ref_hash = str(body.get("key_ref_hash") or "")
    envelope_hash = str(body.get("credential_envelope_hash") or "")
    if auth_kind == "none":
        if slot or credential_hash or key_ref_hash or envelope_hash:
            raise SourceAddRuntimeV2Error(
                "SOURCE_ADD unauthenticated route has credential commitments"
            )
    elif (
        not source_add_dynamic_job_slot(slot)
        or not _HASH_RE.fullmatch(credential_hash)
        or not _HASH_RE.fullmatch(key_ref_hash)
        or not _HASH_RE.fullmatch(envelope_hash)
    ):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD runtime credential commitment is invalid"
        )
    refs = body.get("credential_env_refs")
    if not isinstance(refs, list) or refs != sorted(set(refs)) or any(
        not _ENV_REF_RE.fullmatch(str(item or "")) for item in refs
    ):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD runtime credential environment refs are invalid"
        )
    routes = body.get("allowed_routes")
    if not isinstance(routes, list) or not routes:
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime route set is empty")
    request_headers = body.get("request_headers")
    if not isinstance(request_headers, Mapping) or len(request_headers) > 16 or any(
        not _AUTH_NAME_RE.fullmatch(str(name))
        or str(name).lower() in _FORBIDDEN_AUTH_HEADERS
        or str(name).lower() in {"authorization", "x-api-key", "api-key"}
        or (
            auth_kind in {"header", "bearer"}
            and str(name).lower() == str(body.get("auth_name") or "").lower()
        )
        or not _safe_header_value(value)
        for name, value in request_headers.items()
    ):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD runtime request headers are invalid"
        )
    normalized_routes = []
    for item in routes:
        if not isinstance(item, Mapping) or set(item) != {"method", "path"}:
            raise SourceAddRuntimeV2Error("SOURCE_ADD runtime route is invalid")
        method = str(item.get("method") or "").upper()
        path = _safe_path(item.get("path"))
        if method not in _VALID_METHODS:
            raise SourceAddRuntimeV2Error("SOURCE_ADD runtime method is invalid")
        normalized_routes.append({"method": method, "path": path})
    if normalized_routes != sorted(
        normalized_routes, key=lambda item: (item["method"], item["path"])
    ) or len({(item["method"], item["path"]) for item in normalized_routes}) != len(
        normalized_routes
    ):
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime routes are not canonical")
    for field in ("source_row_hash",):
        if not _HASH_RE.fullmatch(str(body.get(field) or "")):
            raise SourceAddRuntimeV2Error("%s is invalid" % field)
    _int(body.get("per_day_quota"), "SOURCE_ADD runtime quota")
    _int(body.get("est_cost_microusd_per_call"), "SOURCE_ADD runtime cost")
    if sha256_json(body) != value.get("route_hash"):
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime route hash differs")
    return {
        **dict(body),
        "allowed_routes": normalized_routes,
        "request_headers": {
            str(name): str(value)
            for name, value in sorted(request_headers.items())
        },
        "route_hash": str(value["route_hash"]),
    }


def build_source_add_runtime_catalog_v2(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    routes = [build_source_add_runtime_route_v2(row) for row in rows]
    routes.sort(key=lambda item: item["provider_id"])
    provider_ids = [str(item["provider_id"]) for item in routes]
    if len(provider_ids) != len(set(provider_ids)):
        raise SourceAddRuntimeV2Error(
            "SOURCE_ADD runtime catalog contains duplicate provider ids"
        )
    body = {
        "schema_version": SOURCE_ADD_RUNTIME_CATALOG_SCHEMA_VERSION,
        "routes": routes,
    }
    return {**body, "catalog_hash": sha256_json(body)}


def validate_source_add_runtime_catalog_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "routes",
        "catalog_hash",
    }:
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime catalog fields are invalid")
    if value.get("schema_version") != SOURCE_ADD_RUNTIME_CATALOG_SCHEMA_VERSION:
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime catalog schema is invalid")
    raw_routes = value.get("routes")
    if not isinstance(raw_routes, list):
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime catalog routes are invalid")
    routes = [validate_source_add_runtime_route_v2(item) for item in raw_routes]
    if routes != sorted(routes, key=lambda item: item["provider_id"]):
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime catalog is not canonical")
    if len({item["provider_id"] for item in routes}) != len(routes):
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime provider is duplicated")
    body = {
        "schema_version": SOURCE_ADD_RUNTIME_CATALOG_SCHEMA_VERSION,
        "routes": routes,
    }
    if sha256_json(body) != value.get("catalog_hash"):
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime catalog hash differs")
    return {**body, "catalog_hash": str(value["catalog_hash"])}


def source_add_route_for_provider_v2(
    catalog: Mapping[str, Any], provider_id: str
) -> Dict[str, Any] | None:
    normalized = validate_source_add_runtime_catalog_v2(catalog)
    matches = [
        route
        for route in normalized["routes"]
        if route["provider_id"] == str(provider_id or "").lower()
    ]
    if len(matches) > 1:
        raise SourceAddRuntimeV2Error("SOURCE_ADD runtime provider is ambiguous")
    return dict(matches[0]) if matches else None


def source_add_route_for_url_v2(
    catalog: Mapping[str, Any], method: str, url: str
) -> Dict[str, Any] | None:
    parsed = urlsplit(str(url or ""))
    try:
        port = parsed.port or 443
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or not parsed.hostname or port != 443:
        return None
    normalized_method = str(method or "").upper()
    path = parsed.path or "/"
    for route in validate_source_add_runtime_catalog_v2(catalog)["routes"]:
        if route["destination_host"] != parsed.hostname.lower():
            continue
        if any(
            item["method"] == normalized_method and item["path"] == path
            for item in route["allowed_routes"]
        ):
            return dict(route)
    return None


def source_add_dynamic_retry_policy_hash(route: Mapping[str, Any]) -> str:
    normalized = validate_source_add_runtime_route_v2(route)
    return sha256_json(
        {
            "schema_version": SOURCE_ADD_DYNAMIC_RETRY_SCHEMA_VERSION,
            "provider_id": normalized["provider_id"],
            "route_hash": normalized["route_hash"],
            "authority": "existing_measured_caller_retry_logic",
        }
    )


def source_add_runtime_credential_refs_v2(
    catalog: Mapping[str, Any],
) -> Dict[str, str]:
    normalized = validate_source_add_runtime_catalog_v2(catalog)
    return {
        route["provider_id"]: route["credential_value_hash"]
        for route in normalized["routes"]
        if route["credential_value_hash"]
    }


def source_add_runtime_retry_hashes_v2(
    catalog: Mapping[str, Any],
) -> Dict[str, str]:
    normalized = validate_source_add_runtime_catalog_v2(catalog)
    return {
        route["provider_id"]: source_add_dynamic_retry_policy_hash(route)
        for route in normalized["routes"]
    }


def source_add_placeholder_environment_v2(
    catalog: Mapping[str, Any],
) -> Dict[str, str]:
    normalized = validate_source_add_runtime_catalog_v2(catalog)
    return {
        env_name: "leadpoet-coordinator-managed-v2"
        for route in normalized["routes"]
        for env_name in route["credential_env_refs"]
    }


def build_source_add_job_envelope_v2(
    row: Mapping[str, Any], *, job_id: str
) -> Dict[str, Any] | None:
    route = build_source_add_runtime_route_v2(row)
    if route["auth_kind"] == "none":
        return None
    envelope = validate_source_add_credential_envelope_v2(
        row.get("credential_envelope") or {}
    )
    if envelope["envelope_kind"] == "coordinator_sealed":
        normalized = validate_source_add_sealed_job_envelope_v2(
            {
                "schema_version": SOURCE_ADD_SEALED_JOB_ENVELOPE_SCHEMA_VERSION,
                "job_id": str(job_id),
                "credential_slot": route["credential_slot"],
                "credential_ref_hash": route["credential_value_hash"],
                "credential_value_hash": route["credential_value_hash"],
                "key_ref_hash": route["key_ref_hash"],
                "ciphertext_blob_b64": str(envelope["ciphertext_b64"]),
                "ciphertext_blob_hash": str(envelope["ciphertext_blob_hash"]),
                "encryption_context": dict(envelope["encryption_context"]),
                "encryption_context_hash": str(envelope["encryption_context_hash"]),
            }
        )
        return {
            key: item
            for key, item in normalized.items()
            if key not in {"ciphertext_blob", "envelope_kind"}
        }
    from gateway.utils.tee_kms_provision_v2 import (
        JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
        validate_job_provider_envelope,
    )

    value = {
        "schema_version": JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
        "job_id": str(job_id),
        "credential_slot": route["credential_slot"],
        "credential_ref_hash": route["credential_value_hash"],
        "credential_value_hash": route["credential_value_hash"],
        "key_ref_hash": route["key_ref_hash"],
        "ciphertext_blob_b64": str(envelope["ciphertext_b64"]),
        "ciphertext_blob_hash": str(envelope["ciphertext_blob_hash"]),
        "kms_key_id_hash": str(envelope["kms_key_id_hash"]),
        "encryption_context": dict(envelope["encryption_context"]),
        "encryption_context_hash": str(envelope["encryption_context_hash"]),
    }
    normalized = validate_job_provider_envelope(value)
    return {
        key: item
        for key, item in normalized.items()
        if key != "ciphertext_blob"
    }


def build_source_add_probe_job_envelope_v2(
    row: Mapping[str, Any], *, job_id: str
) -> Dict[str, Any] | None:
    """Create a job-scoped credential envelope for a provisional probe."""

    route = build_source_add_probe_route_v2(row)
    if route["auth_kind"] == "none":
        return None
    envelope = validate_source_add_credential_envelope_v2(
        row.get("credential_envelope") or {}
    )
    if envelope["envelope_kind"] == "coordinator_sealed":
        normalized = validate_source_add_sealed_job_envelope_v2(
            {
                "schema_version": SOURCE_ADD_SEALED_JOB_ENVELOPE_SCHEMA_VERSION,
                "job_id": str(job_id),
                "credential_slot": route["credential_slot"],
                "credential_ref_hash": route["credential_value_hash"],
                "credential_value_hash": route["credential_value_hash"],
                "key_ref_hash": route["key_ref_hash"],
                "ciphertext_blob_b64": str(envelope["ciphertext_b64"]),
                "ciphertext_blob_hash": str(envelope["ciphertext_blob_hash"]),
                "encryption_context": dict(envelope["encryption_context"]),
                "encryption_context_hash": str(envelope["encryption_context_hash"]),
            }
        )
        return {
            key: item
            for key, item in normalized.items()
            if key not in {"ciphertext_blob", "envelope_kind"}
        }
    from gateway.utils.tee_kms_provision_v2 import (
        JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
        validate_job_provider_envelope,
    )

    normalized = validate_job_provider_envelope(
        {
            "schema_version": JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
            "job_id": str(job_id),
            "credential_slot": route["credential_slot"],
            "credential_ref_hash": route["credential_value_hash"],
            "credential_value_hash": route["credential_value_hash"],
            "key_ref_hash": route["key_ref_hash"],
            "ciphertext_blob_b64": str(envelope["ciphertext_b64"]),
            "ciphertext_blob_hash": str(envelope["ciphertext_blob_hash"]),
            "kms_key_id_hash": str(envelope["kms_key_id_hash"]),
            "encryption_context": dict(envelope["encryption_context"]),
            "encryption_context_hash": str(envelope["encryption_context_hash"]),
        }
    )
    return {key: item for key, item in normalized.items() if key != "ciphertext_blob"}
