"""Additive encrypted credential commitments for V2 enclave job leases."""

from __future__ import annotations

import base64
import re
from typing import Any, Dict, Mapping

from gateway.research_lab.key_vault import (
    openrouter_credential_value_hash,
    openrouter_kms_encryption_context,
    source_add_kms_encryption_context,
)
from gateway.research_lab.store import insert_row, select_one
from gateway.utils.tee_kms_provision_v2 import kms_key_reference_hash
from gateway.utils.tee_kms_provision_v2 import (
    JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
    validate_job_provider_envelope,
)
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from gateway.tee.source_add_runtime_v2 import (
    SOURCE_ADD_CREDENTIAL_ENVELOPE_SCHEMA_VERSION,
    validate_source_add_credential_envelope_v2,
)
from gateway.tee.openrouter_credential_v2 import (
    OPENROUTER_SEALED_ENVELOPE_SCHEMA_VERSION,
    build_openrouter_sealed_job_envelope_v2,
    validate_openrouter_sealed_envelope_v2,
)


CREDENTIAL_ENVELOPE_SCHEMA_VERSION = "leadpoet.provider_credential_envelope.v2"
_KEY_REF_RE = re.compile(r"^encrypted_ref:openrouter:[0-9a-f]{32}$")
_KINDS = frozenset({"runtime", "management"})


class V2CredentialEnvelopeError(RuntimeError):
    """A stored KMS envelope cannot be safely leased to the coordinator."""


def build_source_add_credential_envelope_v2(
    *,
    credential_ref: str,
    miner_hotkey: str,
    adapter_ref: str,
    raw_credential: str,
    encrypted: Mapping[str, Any],
) -> Dict[str, Any]:
    required = {"ciphertext_b64", "kms_key_id", "encryption_context_hash"}
    if not isinstance(encrypted, Mapping) or set(encrypted) != required:
        raise V2CredentialEnvelopeError(
            "SOURCE_ADD encrypted envelope is invalid"
        )
    try:
        ciphertext = base64.b64decode(
            str(encrypted["ciphertext_b64"]), validate=True
        )
    except Exception as exc:
        raise V2CredentialEnvelopeError(
            "SOURCE_ADD encrypted envelope is not valid base64"
        ) from exc
    if not ciphertext or len(ciphertext) > 64 * 1024:
        raise V2CredentialEnvelopeError(
            "SOURCE_ADD encrypted envelope is outside the size limit"
        )
    credential = str(raw_credential or "").strip()
    if not credential or "\x00" in credential:
        raise V2CredentialEnvelopeError("SOURCE_ADD credential is invalid")
    context = source_add_kms_encryption_context(
        miner_hotkey=str(miner_hotkey),
        adapter_ref=str(adapter_ref),
    )
    context_hash = sha256_json(context)
    if context_hash != str(encrypted["encryption_context_hash"]):
        raise V2CredentialEnvelopeError(
            "SOURCE_ADD encrypted envelope context differs"
        )
    value = {
        "schema_version": SOURCE_ADD_CREDENTIAL_ENVELOPE_SCHEMA_VERSION,
        "ciphertext_b64": str(encrypted["ciphertext_b64"]),
        "ciphertext_blob_hash": sha256_bytes(ciphertext),
        "kms_key_id": str(encrypted["kms_key_id"]),
        "kms_key_id_hash": kms_key_reference_hash(
            str(encrypted["kms_key_id"])
        ),
        "encryption_context": context,
        "encryption_context_hash": context_hash,
        "credential_ref": str(credential_ref),
        "credential_value_hash": sha256_bytes(credential.encode("utf-8")),
        "key_ref_hash": sha256_bytes(str(credential_ref).encode("utf-8")),
    }
    try:
        normalized = validate_source_add_credential_envelope_v2(value)
    except Exception as exc:
        raise V2CredentialEnvelopeError(
            "SOURCE_ADD V2 credential envelope is invalid"
        ) from exc
    return {
        key: item
        for key, item in normalized.items()
        if key != "ciphertext_blob"
    }


def build_openrouter_credential_envelope_v2(
    *,
    key_ref: str,
    miner_hotkey: str,
    credential_kind: str,
    raw_credential: str,
    encrypted: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized_ref = str(key_ref or "")
    normalized_kind = str(credential_kind or "")
    if not _KEY_REF_RE.fullmatch(normalized_ref):
        raise V2CredentialEnvelopeError("OpenRouter key ref is invalid")
    if not str(miner_hotkey or ""):
        raise V2CredentialEnvelopeError("OpenRouter miner hotkey is empty")
    if normalized_kind not in _KINDS:
        raise V2CredentialEnvelopeError("OpenRouter credential kind is invalid")
    required = {"ciphertext_b64", "kms_key_id", "encryption_context_hash"}
    if not isinstance(encrypted, Mapping) or set(encrypted) != required:
        raise V2CredentialEnvelopeError("OpenRouter encrypted envelope is invalid")
    try:
        ciphertext = base64.b64decode(
            str(encrypted["ciphertext_b64"]),
            validate=True,
        )
    except Exception as exc:
        raise V2CredentialEnvelopeError(
            "OpenRouter encrypted envelope is not valid base64"
        ) from exc
    if not ciphertext or len(ciphertext) > 64 * 1024:
        raise V2CredentialEnvelopeError(
            "OpenRouter encrypted envelope is outside the size limit"
        )
    context = openrouter_kms_encryption_context(
        miner_hotkey=str(miner_hotkey),
        key_ref=normalized_ref,
    )
    context_hash = sha256_json(context)
    if context_hash != str(encrypted["encryption_context_hash"]):
        raise V2CredentialEnvelopeError(
            "OpenRouter encrypted envelope context differs"
        )
    body = {
        "schema_version": CREDENTIAL_ENVELOPE_SCHEMA_VERSION,
        "key_ref": normalized_ref,
        "key_ref_hash": sha256_bytes(normalized_ref.encode("utf-8")),
        "miner_hotkey_hash": sha256_bytes(str(miner_hotkey).encode("utf-8")),
        "credential_kind": normalized_kind,
        "credential_slot": (
            "openrouter" if normalized_kind == "runtime" else "openrouter_management"
        ),
        "credential_value_hash": openrouter_credential_value_hash(raw_credential),
        "ciphertext_blob_b64": str(encrypted["ciphertext_b64"]),
        "ciphertext_blob_hash": sha256_bytes(ciphertext),
        "kms_key_id_hash": kms_key_reference_hash(str(encrypted["kms_key_id"])),
        "encryption_context": context,
        "encryption_context_hash": context_hash,
    }
    return {**body, "envelope_hash": sha256_json(body)}


async def persist_openrouter_credential_envelope_v2(
    envelope: Mapping[str, Any],
) -> Dict[str, Any]:
    required = {
        "schema_version",
        "envelope_hash",
        "key_ref",
        "key_ref_hash",
        "miner_hotkey_hash",
        "credential_kind",
        "credential_slot",
        "credential_value_hash",
        "ciphertext_blob_b64",
        "ciphertext_blob_hash",
        "kms_key_id_hash",
        "encryption_context",
        "encryption_context_hash",
    }
    normalized = dict(envelope)
    if set(normalized) != required:
        raise V2CredentialEnvelopeError("V2 credential envelope fields are invalid")
    if normalized.get("schema_version") == OPENROUTER_SEALED_ENVELOPE_SCHEMA_VERSION:
        try:
            normalized = validate_openrouter_sealed_envelope_v2(normalized)
        except Exception as exc:
            raise V2CredentialEnvelopeError(
                "sealed V2 credential envelope is invalid"
            ) from exc
    elif normalized.get("schema_version") != CREDENTIAL_ENVELOPE_SCHEMA_VERSION:
        raise V2CredentialEnvelopeError(
            "V2 credential envelope schema is invalid"
        )
    existing = await select_one(
        "research_lab_provider_credential_envelopes_v2",
        filters=(
            ("key_ref", normalized["key_ref"]),
            ("credential_kind", normalized["credential_kind"]),
        ),
    )
    if existing:
        stable_fields = {
            "key_ref",
            "key_ref_hash",
            "miner_hotkey_hash",
            "credential_kind",
            "credential_slot",
            "credential_value_hash",
        }
        comparable = {
            key: existing.get(key)
            for key in stable_fields
        }
        expected = {
            key: normalized[key]
            for key in stable_fields
        }
        if comparable != expected:
            raise V2CredentialEnvelopeError(
                "existing V2 credential envelope differs for key ref"
            )
        return dict(existing)
    return await insert_row(
        "research_lab_provider_credential_envelopes_v2",
        normalized,
    )


async def load_openrouter_job_credential_envelope_v2(
    *,
    key_ref: str,
    credential_kind: str,
    job_id: str,
) -> Dict[str, Any]:
    if not _KEY_REF_RE.fullmatch(str(key_ref or "")):
        raise V2CredentialEnvelopeError("OpenRouter key ref is invalid")
    if credential_kind not in _KINDS:
        raise V2CredentialEnvelopeError("OpenRouter credential kind is invalid")
    row = await _load_openrouter_credential_envelope_row_v2(
        key_ref=str(key_ref),
        credential_kind=credential_kind,
    )
    expected_slot = (
        "openrouter" if credential_kind == "runtime" else "openrouter_management"
    )
    if str(row.get("credential_slot") or "") != expected_slot:
        raise V2CredentialEnvelopeError(
            "OpenRouter V2 credential slot differs from credential kind"
        )
    if row.get("schema_version") == OPENROUTER_SEALED_ENVELOPE_SCHEMA_VERSION:
        sealed_fields = {
            "schema_version",
            "envelope_hash",
            "key_ref",
            "key_ref_hash",
            "miner_hotkey_hash",
            "credential_kind",
            "credential_slot",
            "credential_value_hash",
            "ciphertext_blob_b64",
            "ciphertext_blob_hash",
            "kms_key_id_hash",
            "encryption_context",
            "encryption_context_hash",
        }
        try:
            return build_openrouter_sealed_job_envelope_v2(
                {key: row[key] for key in sealed_fields},
                job_id=str(job_id),
            )
        except Exception as exc:
            raise V2CredentialEnvelopeError(
                "OpenRouter sealed job credential envelope is invalid"
            ) from exc
    envelope = {
        "schema_version": JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
        "job_id": str(job_id),
        "credential_slot": expected_slot,
        # Job leases bind the exact credential value, not the boot-wide slot.
        "credential_ref_hash": str(row["credential_value_hash"]),
        "credential_value_hash": str(row["credential_value_hash"]),
        "key_ref_hash": str(row["key_ref_hash"]),
        "ciphertext_blob_b64": str(row["ciphertext_blob_b64"]),
        "ciphertext_blob_hash": str(row["ciphertext_blob_hash"]),
        "kms_key_id_hash": str(row["kms_key_id_hash"]),
        "encryption_context": dict(row["encryption_context"]),
        "encryption_context_hash": str(row["encryption_context_hash"]),
    }
    try:
        normalized = validate_job_provider_envelope(envelope)
        return {
            key: item
            for key, item in normalized.items()
            if key != "ciphertext_blob"
        }
    except Exception as exc:
        raise V2CredentialEnvelopeError(
            "OpenRouter V2 job credential envelope is invalid"
        ) from exc


async def load_openrouter_credential_commitments_v2(
    *,
    key_ref: str,
) -> Dict[str, Any]:
    runtime = await _load_openrouter_credential_envelope_row_v2(
        key_ref=str(key_ref),
        credential_kind="runtime",
    )
    management = await _load_openrouter_credential_envelope_row_v2(
        key_ref=str(key_ref),
        credential_kind="management",
    )
    if (
        runtime.get("credential_slot") != "openrouter"
        or management.get("credential_slot") != "openrouter_management"
        or runtime.get("key_ref_hash") != management.get("key_ref_hash")
        or runtime.get("miner_hotkey_hash") != management.get("miner_hotkey_hash")
    ):
        raise V2CredentialEnvelopeError(
            "OpenRouter V2 runtime and management commitments differ"
        )
    return {
        "key_ref_hash": str(runtime["key_ref_hash"]),
        "miner_hotkey_hash": str(runtime["miner_hotkey_hash"]),
        "runtime_credential_value_hash": str(runtime["credential_value_hash"]),
        "management_credential_value_hash": str(
            management["credential_value_hash"]
        ),
    }


async def _load_openrouter_credential_envelope_row_v2(
    *,
    key_ref: str,
    credential_kind: str,
) -> Dict[str, Any]:
    row = await select_one(
        "research_lab_provider_credential_envelopes_v2",
        filters=(("key_ref", str(key_ref)), ("credential_kind", credential_kind)),
    )
    if not row:
        raise V2CredentialEnvelopeError(
            "OpenRouter V2 credential envelope is unavailable"
        )
    return dict(row)
