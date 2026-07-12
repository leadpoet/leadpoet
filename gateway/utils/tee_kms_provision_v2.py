"""Provision coordinator credentials through Nitro KMS recipient envelopes.

The parent can call AWS KMS and relay the returned recipient ciphertext, but it
never receives plaintext.  Envelopes are prepared out of band and contain only
KMS ciphertext plus non-secret encryption context.
"""

from __future__ import annotations

import asyncio
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.tee.kms_recipient_v2 import (
    KMS_KEY_ENCRYPTION_ALGORITHM,
    KMS_RECIPIENT_SCHEMA_VERSION,
)
from gateway.utils.tee_client import coordinator_tee_client
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json


PROVIDER_ENVELOPE_SCHEMA_VERSION = "leadpoet.provider_credential_envelope.v2"
JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION = "leadpoet.job_provider_credential_envelope.v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class TEEKMSProvisionV2Error(RuntimeError):
    """An encrypted credential envelope or KMS recipient response is unsafe."""


def kms_key_reference_hash(key_id: str) -> str:
    value = str(key_id or "").strip()
    if not value:
        raise TEEKMSProvisionV2Error("KMS key id is empty")
    return sha256_bytes(("leadpoet-kms-key-v2:" + value).encode("utf-8"))


def validate_provider_envelope(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "credential_slot",
        "credential_ref_hash",
        "ciphertext_blob_b64",
        "ciphertext_blob_hash",
        "kms_key_id_hash",
        "encryption_context",
        "encryption_context_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TEEKMSProvisionV2Error("provider envelope fields are invalid")
    if value.get("schema_version") != PROVIDER_ENVELOPE_SCHEMA_VERSION:
        raise TEEKMSProvisionV2Error("provider envelope schema is invalid")
    slot = str(value.get("credential_slot") or "")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", slot):
        raise TEEKMSProvisionV2Error("provider envelope slot is invalid")
    credential_ref_hash = str(value.get("credential_ref_hash") or "").lower()
    if not _HASH_RE.fullmatch(credential_ref_hash):
        raise TEEKMSProvisionV2Error(
            "provider envelope credential reference is invalid"
        )
    try:
        ciphertext = base64.b64decode(
            str(value.get("ciphertext_blob_b64") or ""),
            validate=True,
        )
    except Exception as exc:
        raise TEEKMSProvisionV2Error("provider envelope ciphertext is invalid") from exc
    if not ciphertext or len(ciphertext) > 64 * 1024:
        raise TEEKMSProvisionV2Error("provider envelope ciphertext is outside limit")
    if sha256_bytes(ciphertext) != value.get("ciphertext_blob_hash"):
        raise TEEKMSProvisionV2Error("provider envelope ciphertext hash mismatch")
    context = value.get("encryption_context")
    if not isinstance(context, Mapping) or not context:
        raise TEEKMSProvisionV2Error("provider envelope context is missing")
    normalized_context = {
        str(name): str(item) for name, item in sorted(context.items())
    }
    if any(not name or not item or "\x00" in name + item for name, item in normalized_context.items()):
        raise TEEKMSProvisionV2Error("provider envelope context is invalid")
    if sha256_json(normalized_context) != value.get("encryption_context_hash"):
        raise TEEKMSProvisionV2Error("provider envelope context hash mismatch")
    if not _HASH_RE.fullmatch(str(value.get("kms_key_id_hash") or "")):
        raise TEEKMSProvisionV2Error("provider envelope KMS key hash is invalid")
    return {
        **dict(value),
        "credential_slot": slot,
        "credential_ref_hash": credential_ref_hash,
        "encryption_context": normalized_context,
        "ciphertext_blob": ciphertext,
    }


async def provision_provider_envelope_v2(
    envelope: Mapping[str, Any],
    *,
    client: Any = coordinator_tee_client,
    kms_client: Any = None,
) -> Dict[str, Any]:
    normalized = validate_provider_envelope(envelope)
    recipient = await client.v2_get_kms_recipient(
        normalized["credential_slot"]
    )
    if (
        recipient.get("schema_version") != KMS_RECIPIENT_SCHEMA_VERSION
        or recipient.get("credential_slot") != normalized["credential_slot"]
        or recipient.get("credential_ref_hash")
        != normalized["credential_ref_hash"]
        or recipient.get("key_encryption_algorithm")
        != KMS_KEY_ENCRYPTION_ALGORITHM
    ):
        raise TEEKMSProvisionV2Error("coordinator KMS recipient request is invalid")
    try:
        attestation_document = base64.b64decode(
            str(recipient["attestation_document_b64"]),
            validate=True,
        )
    except Exception as exc:
        raise TEEKMSProvisionV2Error("coordinator KMS attestation is invalid") from exc
    if kms_client is None:
        import boto3

        kms_client = boto3.client("kms")
    response = await asyncio.to_thread(
        kms_client.decrypt,
        CiphertextBlob=normalized["ciphertext_blob"],
        EncryptionContext=normalized["encryption_context"],
        Recipient={
            "KeyEncryptionAlgorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
            "AttestationDocument": attestation_document,
        },
    )
    if "Plaintext" in response:
        raise TEEKMSProvisionV2Error("KMS returned plaintext to the parent")
    key_id = str(response.get("KeyId") or "")
    if kms_key_reference_hash(key_id) != normalized["kms_key_id_hash"]:
        raise TEEKMSProvisionV2Error("KMS response key differs from envelope")
    ciphertext_for_recipient = response.get("CiphertextForRecipient")
    if not isinstance(ciphertext_for_recipient, (bytes, bytearray)) or not ciphertext_for_recipient:
        raise TEEKMSProvisionV2Error("KMS recipient ciphertext is missing")
    result = await client.v2_provision_encrypted_secret(
        credential_slot=normalized["credential_slot"],
        ciphertext_for_recipient_b64=base64.b64encode(
            bytes(ciphertext_for_recipient)
        ).decode("ascii"),
    )
    if result.get("status") not in {"provisioning", "ready"}:
        raise TEEKMSProvisionV2Error("coordinator rejected encrypted credential")
    return {
        "credential_slot": normalized["credential_slot"],
        "status": result["status"],
        "configured_slots": list(result.get("credential_slots") or []),
        "missing_slots": list(result.get("missing_credential_slots") or []),
    }


def validate_job_provider_envelope(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "job_id",
        "credential_slot",
        "credential_ref_hash",
        "credential_value_hash",
        "key_ref_hash",
        "ciphertext_blob_b64",
        "ciphertext_blob_hash",
        "kms_key_id_hash",
        "encryption_context",
        "encryption_context_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TEEKMSProvisionV2Error("job provider envelope fields are invalid")
    if value.get("schema_version") != JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION:
        raise TEEKMSProvisionV2Error("job provider envelope schema is invalid")
    job_id = str(value.get("job_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}", job_id):
        raise TEEKMSProvisionV2Error("job provider envelope job id is invalid")
    base = {
        "schema_version": PROVIDER_ENVELOPE_SCHEMA_VERSION,
        "credential_slot": value["credential_slot"],
        "credential_ref_hash": value["credential_ref_hash"],
        "ciphertext_blob_b64": value["ciphertext_blob_b64"],
        "ciphertext_blob_hash": value["ciphertext_blob_hash"],
        "kms_key_id_hash": value["kms_key_id_hash"],
        "encryption_context": value["encryption_context"],
        "encryption_context_hash": value["encryption_context_hash"],
    }
    normalized = validate_provider_envelope(base)
    credential_value_hash = str(value.get("credential_value_hash") or "").lower()
    key_ref_hash = str(value.get("key_ref_hash") or "").lower()
    if (
        not _HASH_RE.fullmatch(credential_value_hash)
        or not _HASH_RE.fullmatch(key_ref_hash)
    ):
        raise TEEKMSProvisionV2Error(
            "job provider envelope credential commitment is invalid"
        )
    return {
        **normalized,
        "schema_version": JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
        "job_id": job_id,
        "credential_value_hash": credential_value_hash,
        "key_ref_hash": key_ref_hash,
    }


async def provision_job_provider_envelope_v2(
    envelope: Mapping[str, Any],
    *,
    client: Any = coordinator_tee_client,
    kms_client: Any = None,
) -> Dict[str, Any]:
    """KMS unwrap one per-miner key directly into the coordinator enclave."""

    normalized = validate_job_provider_envelope(envelope)
    recipient = await client.v2_get_job_kms_recipient(
        job_id=normalized["job_id"],
        credential_slot=normalized["credential_slot"],
        credential_value_hash=normalized["credential_value_hash"],
        key_ref_hash=normalized["key_ref_hash"],
    )
    if (
        recipient.get("schema_version") != "leadpoet.kms_job_recipient.v2"
        or recipient.get("job_id") != normalized["job_id"]
        or recipient.get("credential_slot") != normalized["credential_slot"]
        or recipient.get("credential_value_hash")
        != normalized["credential_value_hash"]
        or recipient.get("key_ref_hash") != normalized["key_ref_hash"]
        or recipient.get("key_encryption_algorithm")
        != KMS_KEY_ENCRYPTION_ALGORITHM
    ):
        raise TEEKMSProvisionV2Error("coordinator job KMS recipient is invalid")
    try:
        attestation_document = base64.b64decode(
            str(recipient["attestation_document_b64"]),
            validate=True,
        )
    except Exception as exc:
        raise TEEKMSProvisionV2Error(
            "coordinator job KMS attestation is invalid"
        ) from exc
    if kms_client is None:
        import boto3

        kms_client = boto3.client("kms")
    response = await asyncio.to_thread(
        kms_client.decrypt,
        CiphertextBlob=normalized["ciphertext_blob"],
        EncryptionContext=normalized["encryption_context"],
        Recipient={
            "KeyEncryptionAlgorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
            "AttestationDocument": attestation_document,
        },
    )
    if "Plaintext" in response:
        raise TEEKMSProvisionV2Error("KMS returned job plaintext to the parent")
    if kms_key_reference_hash(str(response.get("KeyId") or "")) != normalized[
        "kms_key_id_hash"
    ]:
        raise TEEKMSProvisionV2Error("job KMS response key differs from envelope")
    ciphertext_for_recipient = response.get("CiphertextForRecipient")
    if not isinstance(ciphertext_for_recipient, (bytes, bytearray)) or not ciphertext_for_recipient:
        raise TEEKMSProvisionV2Error("job KMS recipient ciphertext is missing")
    result = await client.v2_provision_job_encrypted_secret(
        request_id=str(recipient["request_id"]),
        ciphertext_for_recipient_b64=base64.b64encode(
            bytes(ciphertext_for_recipient)
        ).decode("ascii"),
    )
    if (
        result.get("status") != "ready"
        or result.get("job_id") != normalized["job_id"]
        or result.get("credential_slot") != normalized["credential_slot"]
        or result.get("credential_ref_hash")
        != normalized["credential_value_hash"]
    ):
        raise TEEKMSProvisionV2Error("coordinator rejected job credential lease")
    return dict(result)


def validate_job_credential_envelope_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate either an existing KMS envelope or coordinator-sealed SOURCE_ADD."""

    from gateway.tee.openrouter_credential_v2 import (
        OPENROUTER_SEALED_JOB_ENVELOPE_SCHEMA_VERSION,
        validate_openrouter_sealed_job_envelope_v2,
    )
    from gateway.tee.source_add_runtime_v2 import (
        SOURCE_ADD_SEALED_JOB_ENVELOPE_SCHEMA_VERSION,
        validate_source_add_sealed_job_envelope_v2,
    )

    if value.get("schema_version") == OPENROUTER_SEALED_JOB_ENVELOPE_SCHEMA_VERSION:
        return validate_openrouter_sealed_job_envelope_v2(value)
    if value.get("schema_version") == SOURCE_ADD_SEALED_JOB_ENVELOPE_SCHEMA_VERSION:
        return validate_source_add_sealed_job_envelope_v2(value)
    return validate_job_provider_envelope(value)


async def provision_job_credential_envelope_v2(
    envelope: Mapping[str, Any],
    *,
    client: Any = coordinator_tee_client,
    kms_client: Any = None,
) -> Dict[str, Any]:
    """Lease one validated job credential without exposing plaintext to parent."""

    from gateway.tee.openrouter_credential_v2 import (
        OPENROUTER_SEALED_JOB_ENVELOPE_SCHEMA_VERSION,
    )
    from gateway.tee.source_add_runtime_v2 import (
        SOURCE_ADD_SEALED_JOB_ENVELOPE_SCHEMA_VERSION,
    )

    normalized = validate_job_credential_envelope_v2(envelope)
    wire = {
        key: item
        for key, item in normalized.items()
        if key not in {"ciphertext_blob", "envelope_kind"}
    }
    if normalized["schema_version"] not in {
        SOURCE_ADD_SEALED_JOB_ENVELOPE_SCHEMA_VERSION,
        OPENROUTER_SEALED_JOB_ENVELOPE_SCHEMA_VERSION,
    }:
        return await provision_job_provider_envelope_v2(
            wire,
            client=client,
            kms_client=kms_client,
        )
    if normalized["schema_version"] == SOURCE_ADD_SEALED_JOB_ENVELOPE_SCHEMA_VERSION:
        result = await client.v2_provision_job_sealed_source_add_secret(
            envelope=wire,
        )
        credential_label = "SOURCE_ADD"
    else:
        result = await client.v2_provision_job_sealed_openrouter_secret(
            envelope=wire,
        )
        credential_label = "OpenRouter"
    if (
        result.get("status") != "ready"
        or result.get("job_id") != normalized["job_id"]
        or result.get("credential_slot") != normalized["credential_slot"]
        or result.get("credential_ref_hash")
        != normalized["credential_value_hash"]
    ):
        raise TEEKMSProvisionV2Error(
            "coordinator rejected sealed %s job credential" % credential_label
        )
    return dict(result)


def load_provider_envelopes(paths: Sequence[Path]) -> list[Dict[str, Any]]:
    envelopes = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TEEKMSProvisionV2Error(
                "provider envelope file is unavailable: %s" % path
            ) from exc
        envelopes.append(validate_provider_envelope(value))
    slots = [item["credential_slot"] for item in envelopes]
    if len(slots) != len(set(slots)):
        raise TEEKMSProvisionV2Error("provider envelope slots are duplicated")
    return envelopes


def provider_reference_hashes_from_envelopes(
    envelopes: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    return {
        str(item["credential_slot"]): str(item["credential_ref_hash"])
        for item in envelopes
        if str(item["credential_slot"]) != "artifact_master_key"
    }


async def _main_async(args: argparse.Namespace) -> Dict[str, Any]:
    envelopes = load_provider_envelopes(args.envelope)
    results = []
    for envelope in envelopes:
        results.append(await provision_provider_envelope_v2(envelope))
    return {
        "schema_version": "leadpoet.kms_boot_provision.v2",
        "status": "ready",
        "credential_slots": sorted(
            item["credential_slot"] for item in envelopes
        ),
        "results": results,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope", action="append", required=True, type=Path)
    args = parser.parse_args(argv)
    result = asyncio.run(_main_async(args))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
