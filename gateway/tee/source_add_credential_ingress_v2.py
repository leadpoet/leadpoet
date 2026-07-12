"""Coordinator-only SOURCE_ADD credential ingress and job leasing."""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, Mapping

from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.provider_broker_v2 import credential_value_hash
from gateway.tee.source_add_runtime_v2 import (
    SOURCE_ADD_SEALED_CREDENTIAL_ENVELOPE_SCHEMA_VERSION,
    validate_source_add_credential_envelope_v2,
    validate_source_add_sealed_job_envelope_v2,
)
from gateway.utils.tee_kms_provision_v2 import kms_key_reference_hash
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json


COORDINATOR_SEALED_KEY_ID = "leadpoet:coordinator-artifact-master-key:v2"
SOURCE_ADD_STORED_SECRET_SCHEMA_VERSION = "leadpoet.source_add_stored_secret.v2"


class SourceAddCredentialIngressV2Error(RuntimeError):
    """A client ciphertext or sealed SOURCE_ADD envelope is invalid."""


def source_add_encryption_context(
    *, miner_hotkey: str, adapter_ref: str
) -> Dict[str, str]:
    normalized_miner = str(miner_hotkey or "")
    normalized_adapter = str(adapter_ref or "")
    if not normalized_miner or not normalized_adapter.startswith("source_add:"):
        raise SourceAddCredentialIngressV2Error(
            "SOURCE_ADD credential context is invalid"
        )
    return {
        "adapter_ref": normalized_adapter,
        "miner_hotkey": normalized_miner,
        "purpose": "leadpoet_research_lab_source_add_credential",
    }


def seal_source_add_ingress_credential_v2(
    lease: Mapping[str, Any],
    *,
    vault: EncryptedArtifactVaultV2,
) -> Dict[str, Any]:
    required = {
        "request_id",
        "miner_hotkey",
        "adapter_ref",
        "credential_ref",
        "key_ref_hash",
        "credential_value_hash",
        "credential",
    }
    if not isinstance(lease, Mapping) or set(lease) != required:
        raise SourceAddCredentialIngressV2Error(
            "SOURCE_ADD ingress lease fields are invalid"
        )
    credential = str(lease.get("credential") or "")
    if (
        not credential
        or "\x00" in credential
        or credential_value_hash(credential) != lease.get("credential_value_hash")
    ):
        raise SourceAddCredentialIngressV2Error(
            "SOURCE_ADD ingress credential commitment differs"
        )
    context = source_add_encryption_context(
        miner_hotkey=str(lease["miner_hotkey"]),
        adapter_ref=str(lease["adapter_ref"]),
    )
    secret_document = {
        "schema_version": SOURCE_ADD_STORED_SECRET_SCHEMA_VERSION,
        "request_id": str(lease["request_id"]),
        "miner_hotkey_hash": sha256_bytes(
            str(lease["miner_hotkey"]).encode("utf-8")
        ),
        "adapter_ref": str(lease["adapter_ref"]),
        "credential_ref_hash": sha256_bytes(
            str(lease["credential_ref"]).encode("utf-8")
        ),
        "credential_value_hash": str(lease["credential_value_hash"]),
        "credential": credential,
    }
    secret_bytes = canonical_json(secret_document).encode("utf-8")
    descriptor = vault.seal(
        secret_bytes,
        job_id="source-add-ingress:%s" % str(lease["request_id"]),
        purpose="research_lab.source_add_credential.v2",
        artifact_kind="source_add_credential",
    )
    exported = vault.export_ciphertext(str(descriptor["artifact_id"]))
    storage_bytes = canonical_json(exported["storage_document"]).encode("utf-8")
    value = {
        "schema_version": SOURCE_ADD_SEALED_CREDENTIAL_ENVELOPE_SCHEMA_VERSION,
        "envelope_kind": "coordinator_sealed",
        "ciphertext_b64": base64.b64encode(storage_bytes).decode("ascii"),
        "ciphertext_blob_hash": sha256_bytes(storage_bytes),
        "kms_key_id": COORDINATOR_SEALED_KEY_ID,
        "kms_key_id_hash": kms_key_reference_hash(COORDINATOR_SEALED_KEY_ID),
        "encryption_context": context,
        "encryption_context_hash": sha256_json(context),
        "credential_ref": str(lease["credential_ref"]),
        "credential_value_hash": str(lease["credential_value_hash"]),
        "key_ref_hash": str(lease["key_ref_hash"]),
    }
    normalized = validate_source_add_credential_envelope_v2(value)
    vault.release_transient(str(descriptor["artifact_id"]))
    return {
        key: item
        for key, item in normalized.items()
        if key != "ciphertext_blob"
    }


def unseal_source_add_job_credential_v2(
    envelope: Mapping[str, Any],
    *,
    vault: EncryptedArtifactVaultV2,
) -> Dict[str, str]:
    normalized = validate_source_add_sealed_job_envelope_v2(envelope)
    try:
        storage_document = json.loads(
            normalized["ciphertext_blob"].decode("utf-8")
        )
    except Exception as exc:
        raise SourceAddCredentialIngressV2Error(
            "SOURCE_ADD sealed storage document is invalid"
        ) from exc
    if (
        not isinstance(storage_document, Mapping)
        or canonical_json(dict(storage_document)).encode("utf-8")
        != normalized["ciphertext_blob"]
    ):
        raise SourceAddCredentialIngressV2Error(
            "SOURCE_ADD sealed storage document is not canonical"
        )
    plaintext = vault.decrypt_storage_document(storage_document)
    try:
        secret_document = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise SourceAddCredentialIngressV2Error(
            "SOURCE_ADD sealed credential document is invalid"
        ) from exc
    expected_context = normalized["encryption_context"]
    if (
        not isinstance(secret_document, Mapping)
        or canonical_json(dict(secret_document)).encode("utf-8") != plaintext
        or set(secret_document)
        != {
            "schema_version",
            "request_id",
            "miner_hotkey_hash",
            "adapter_ref",
            "credential_ref_hash",
            "credential_value_hash",
            "credential",
        }
        or secret_document.get("schema_version")
        != SOURCE_ADD_STORED_SECRET_SCHEMA_VERSION
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(secret_document.get("request_id") or "")
        )
        or secret_document.get("miner_hotkey_hash")
        != sha256_bytes(str(expected_context.get("miner_hotkey") or "").encode("utf-8"))
        or secret_document.get("adapter_ref")
        != str(expected_context.get("adapter_ref") or "")
        or secret_document.get("credential_ref_hash") != normalized["key_ref_hash"]
        or secret_document.get("credential_value_hash")
        != normalized["credential_value_hash"]
    ):
        raise SourceAddCredentialIngressV2Error(
            "SOURCE_ADD sealed credential scope differs"
        )
    credential = str(secret_document.get("credential") or "")
    if credential_value_hash(credential) != normalized["credential_value_hash"]:
        raise SourceAddCredentialIngressV2Error(
            "SOURCE_ADD sealed credential hash differs"
        )
    aad = json.loads(
        base64.b64decode(str(storage_document.get("aad_b64") or ""), validate=True)
    )
    if (
        not isinstance(aad, Mapping)
        or aad.get("artifact_kind") != "source_add_credential"
        or aad.get("job_id")
        != "source-add-ingress:%s" % str(secret_document["request_id"])
        or aad.get("purpose") != "research_lab.source_add_credential.v2"
        or aad.get("plaintext_hash") != sha256_bytes(plaintext)
    ):
        raise SourceAddCredentialIngressV2Error(
            "SOURCE_ADD sealed credential artifact scope differs"
        )
    return {
        "job_id": str(normalized["job_id"]),
        "credential_slot": str(normalized["credential_slot"]),
        "credential_value_hash": str(normalized["credential_value_hash"]),
        "key_ref_hash": str(normalized["key_ref_hash"]),
        "credential": credential,
    }
