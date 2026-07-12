"""Coordinator-only OpenRouter credential ingress, registration, and leasing."""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, Mapping

from gateway.research_lab.key_vault import (
    openrouter_key_ref,
    openrouter_kms_encryption_context,
    preflight_openrouter_key,
    verify_openrouter_workspace_privacy,
)
from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionResultV2,
)
from gateway.tee.provider_broker_v2 import (
    ProviderBrokerV2,
    credential_value_hash,
)
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.source_add_credential_ingress_v2 import (
    COORDINATOR_SEALED_KEY_ID,
)
from gateway.utils.tee_kms_provision_v2 import kms_key_reference_hash
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json


OPENROUTER_INGRESS_ENVELOPE_SCHEMA_VERSION = (
    "leadpoet.openrouter_ingress_credential.enclave.v2"
)
OPENROUTER_SEALED_ENVELOPE_SCHEMA_VERSION = (
    "leadpoet.provider_credential_envelope.enclave.v2"
)
OPENROUTER_SEALED_JOB_ENVELOPE_SCHEMA_VERSION = (
    "leadpoet.job_provider_credential_envelope.enclave.v2"
)
OPENROUTER_REGISTRATION_REQUEST_SCHEMA_VERSION = (
    "leadpoet.openrouter_credential_registration_request.v2"
)
OPENROUTER_REGISTRATION_RESULT_SCHEMA_VERSION = (
    "leadpoet.openrouter_credential_registration_result.v2"
)
OPENROUTER_PREFLIGHT_REQUEST_SCHEMA_VERSION = (
    "leadpoet.openrouter_credit_preflight_request.v2"
)
OPENROUTER_PREFLIGHT_RESULT_SCHEMA_VERSION = (
    "leadpoet.openrouter_credit_preflight_result.v2"
)
_KINDS = frozenset({"runtime", "management"})
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_KEY_REF_RE = re.compile(r"^encrypted_ref:openrouter:[0-9a-f]{32}$")
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


class OpenRouterCredentialV2Error(RuntimeError):
    """An OpenRouter credential escaped scope or failed measured validation."""


def _slot(kind: str) -> str:
    normalized = str(kind or "")
    if normalized not in _KINDS:
        raise OpenRouterCredentialV2Error("OpenRouter credential kind is invalid")
    return "openrouter" if normalized == "runtime" else "openrouter_management"


def _miner_hash(miner_hotkey: str) -> str:
    value = str(miner_hotkey or "")
    if not value:
        raise OpenRouterCredentialV2Error("OpenRouter miner hotkey is empty")
    return sha256_bytes(value.encode("utf-8"))


def _decode_storage_document(ciphertext_blob_b64: Any) -> tuple[bytes, Dict[str, Any]]:
    try:
        packed = base64.b64decode(str(ciphertext_blob_b64 or ""), validate=True)
        document = json.loads(packed.decode("utf-8"))
    except Exception as exc:
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed storage document is invalid"
        ) from exc
    if (
        not packed
        or len(packed) > 64 * 1024
        or not isinstance(document, Mapping)
        or canonical_json(dict(document)).encode("utf-8") != packed
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed storage document is not canonical"
        )
    return packed, dict(document)


def _seal_storage_document(
    plaintext: bytes,
    *,
    vault: EncryptedArtifactVaultV2,
    job_id: str,
    purpose: str,
    artifact_kind: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    descriptor = vault.seal(
        bytes(plaintext),
        job_id=str(job_id),
        purpose=str(purpose),
        artifact_kind=str(artifact_kind),
    )
    artifact_id = str(descriptor["artifact_id"])
    exported = vault.export_ciphertext(artifact_id)
    storage_document = dict(exported["storage_document"])
    packed = canonical_json(storage_document).encode("utf-8")
    vault.release_transient(artifact_id)
    return (
        {
            "ciphertext_blob_b64": base64.b64encode(packed).decode("ascii"),
            "ciphertext_blob_hash": sha256_bytes(packed),
            "artifact_id": artifact_id,
            "storage_document_hash": str(exported["storage_document_hash"]),
        },
        storage_document,
    )


def seal_openrouter_ingress_credential_v2(
    lease: Mapping[str, Any],
    *,
    vault: EncryptedArtifactVaultV2,
) -> Dict[str, Any]:
    required = {
        "request_id",
        "miner_hotkey",
        "credential_kind",
        "credential_slot",
        "credential_value_hash",
        "credential",
    }
    if not isinstance(lease, Mapping) or set(lease) != required:
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress lease fields are invalid"
        )
    kind = str(lease["credential_kind"])
    slot = _slot(kind)
    credential = str(lease["credential"])
    if (
        str(lease["credential_slot"]) != slot
        or credential_value_hash(credential) != lease["credential_value_hash"]
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress credential commitment differs"
        )
    secret_document = {
        "schema_version": "leadpoet.openrouter_ingress_secret.v2",
        "request_id": str(lease["request_id"]),
        "miner_hotkey_hash": _miner_hash(str(lease["miner_hotkey"])),
        "credential_kind": kind,
        "credential_slot": slot,
        "credential_value_hash": str(lease["credential_value_hash"]),
        "credential": credential,
    }
    sealed, _document = _seal_storage_document(
        canonical_json(secret_document).encode("utf-8"),
        vault=vault,
        job_id="openrouter-ingress:%s" % str(lease["request_id"]),
        purpose="research_lab.openrouter_credential.v2",
        artifact_kind="openrouter_credential_ingress",
    )
    context = {
        "credential_kind": kind,
        "miner_hotkey_hash": _miner_hash(str(lease["miner_hotkey"])),
        "purpose": "leadpoet_research_lab_openrouter_credential_ingress",
        "request_id": str(lease["request_id"]),
    }
    body = {
        "schema_version": OPENROUTER_INGRESS_ENVELOPE_SCHEMA_VERSION,
        "credential_kind": kind,
        "credential_slot": slot,
        "credential_value_hash": str(lease["credential_value_hash"]),
        "miner_hotkey_hash": context["miner_hotkey_hash"],
        "ciphertext_blob_b64": sealed["ciphertext_blob_b64"],
        "ciphertext_blob_hash": sealed["ciphertext_blob_hash"],
        "kms_key_id_hash": kms_key_reference_hash(COORDINATOR_SEALED_KEY_ID),
        "encryption_context": context,
        "encryption_context_hash": sha256_json(context),
        "artifact_id": sealed["artifact_id"],
        "storage_document_hash": sealed["storage_document_hash"],
    }
    return {**body, "envelope_hash": sha256_json(body)}


def validate_openrouter_ingress_envelope_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "envelope_hash",
        "credential_kind",
        "credential_slot",
        "credential_value_hash",
        "miner_hotkey_hash",
        "ciphertext_blob_b64",
        "ciphertext_blob_hash",
        "kms_key_id_hash",
        "encryption_context",
        "encryption_context_hash",
        "artifact_id",
        "storage_document_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress envelope fields are invalid"
        )
    body = {key: value[key] for key in fields if key != "envelope_hash"}
    if (
        value.get("schema_version") != OPENROUTER_INGRESS_ENVELOPE_SCHEMA_VERSION
        or value.get("envelope_hash") != sha256_json(body)
        or value.get("credential_slot") != _slot(str(value.get("credential_kind") or ""))
        or not _HASH_RE.fullmatch(str(value.get("credential_value_hash") or ""))
        or not _HASH_RE.fullmatch(str(value.get("miner_hotkey_hash") or ""))
        or not _HASH_RE.fullmatch(str(value.get("artifact_id") or ""))
        or not _HASH_RE.fullmatch(str(value.get("storage_document_hash") or ""))
        or not _HASH_RE.fullmatch(str(value.get("kms_key_id_hash") or ""))
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress envelope commitment is invalid"
        )
    packed, document = _decode_storage_document(value["ciphertext_blob_b64"])
    context = value.get("encryption_context")
    if (
        sha256_bytes(packed) != value.get("ciphertext_blob_hash")
        or not isinstance(context, Mapping)
        or set(context)
        != {
            "credential_kind",
            "miner_hotkey_hash",
            "purpose",
            "request_id",
        }
        or sha256_json(dict(context)) != value.get("encryption_context_hash")
        or context.get("credential_kind") != value.get("credential_kind")
        or context.get("miner_hotkey_hash") != value.get("miner_hotkey_hash")
        or context.get("purpose")
        != "leadpoet_research_lab_openrouter_credential_ingress"
        or not _HASH_RE.fullmatch(str(context.get("request_id") or ""))
        or document.get("artifact_id") != value.get("artifact_id")
        or sha256_json(document) != value.get("storage_document_hash")
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress envelope scope differs"
        )
    return dict(value)


def unseal_openrouter_ingress_credential_v2(
    envelope: Mapping[str, Any],
    *,
    miner_hotkey: str,
    vault: EncryptedArtifactVaultV2,
) -> Dict[str, str]:
    normalized = validate_openrouter_ingress_envelope_v2(envelope)
    if normalized["miner_hotkey_hash"] != _miner_hash(miner_hotkey):
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress miner scope differs"
        )
    _packed, document = _decode_storage_document(
        normalized["ciphertext_blob_b64"]
    )
    plaintext = vault.decrypt_storage_document(document)
    try:
        secret_document = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress credential document is invalid"
        ) from exc
    if (
        not isinstance(secret_document, Mapping)
        or canonical_json(dict(secret_document)).encode("utf-8") != plaintext
        or set(secret_document)
        != {
            "schema_version",
            "request_id",
            "miner_hotkey_hash",
            "credential_kind",
            "credential_slot",
            "credential_value_hash",
            "credential",
        }
        or secret_document.get("schema_version")
        != "leadpoet.openrouter_ingress_secret.v2"
        or secret_document.get("request_id")
        != normalized["encryption_context"].get("request_id")
        or secret_document.get("miner_hotkey_hash")
        != normalized["miner_hotkey_hash"]
        or secret_document.get("credential_kind")
        != normalized["credential_kind"]
        or secret_document.get("credential_slot")
        != normalized["credential_slot"]
        or secret_document.get("credential_value_hash")
        != normalized["credential_value_hash"]
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress secret scope differs"
        )
    credential = str(secret_document.get("credential") or "")
    if credential_value_hash(credential) != normalized["credential_value_hash"]:
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress credential hash differs"
        )
    aad = json.loads(
        base64.b64decode(str(document.get("aad_b64") or ""), validate=True)
    )
    if (
        not isinstance(aad, Mapping)
        or aad.get("artifact_kind") != "openrouter_credential_ingress"
        or aad.get("job_id")
        != "openrouter-ingress:%s"
        % str(normalized["encryption_context"]["request_id"])
        or aad.get("purpose") != "research_lab.openrouter_credential.v2"
        or aad.get("plaintext_hash") != sha256_bytes(plaintext)
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter ingress artifact scope differs"
        )
    return {
        "credential_kind": str(normalized["credential_kind"]),
        "credential_slot": str(normalized["credential_slot"]),
        "credential_value_hash": str(normalized["credential_value_hash"]),
        "credential": credential,
    }


def seal_openrouter_persistent_credential_v2(
    *,
    credential: str,
    credential_kind: str,
    key_ref: str,
    miner_hotkey: str,
    job_id: str,
    purpose: str,
    vault: EncryptedArtifactVaultV2,
) -> Dict[str, Any]:
    kind = str(credential_kind)
    slot = _slot(kind)
    key_ref_hash = sha256_bytes(str(key_ref).encode("utf-8"))
    miner_hotkey_hash = _miner_hash(miner_hotkey)
    value_hash = credential_value_hash(credential)
    secret_document = {
        "schema_version": "leadpoet.openrouter_stored_secret.v2",
        "key_ref_hash": key_ref_hash,
        "miner_hotkey_hash": miner_hotkey_hash,
        "credential_kind": kind,
        "credential_slot": slot,
        "credential_value_hash": value_hash,
        "credential": credential,
    }
    sealed, _document = _seal_storage_document(
        canonical_json(secret_document).encode("utf-8"),
        vault=vault,
        job_id=job_id,
        purpose=purpose,
        artifact_kind="openrouter_credential",
    )
    context = openrouter_kms_encryption_context(
        miner_hotkey=str(miner_hotkey),
        key_ref=str(key_ref),
    )
    body = {
        "schema_version": OPENROUTER_SEALED_ENVELOPE_SCHEMA_VERSION,
        "key_ref": str(key_ref),
        "key_ref_hash": key_ref_hash,
        "miner_hotkey_hash": miner_hotkey_hash,
        "credential_kind": kind,
        "credential_slot": slot,
        "credential_value_hash": value_hash,
        "ciphertext_blob_b64": sealed["ciphertext_blob_b64"],
        "ciphertext_blob_hash": sealed["ciphertext_blob_hash"],
        "kms_key_id_hash": kms_key_reference_hash(COORDINATOR_SEALED_KEY_ID),
        "encryption_context": context,
        "encryption_context_hash": sha256_json(context),
    }
    return {**body, "envelope_hash": sha256_json(body)}


def validate_openrouter_sealed_envelope_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
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
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed envelope fields are invalid"
        )
    body = {key: value[key] for key in fields if key != "envelope_hash"}
    context = value.get("encryption_context")
    packed, document = _decode_storage_document(value["ciphertext_blob_b64"])
    expected_context = openrouter_kms_encryption_context(
        miner_hotkey=str(context.get("miner_hotkey") or "")
        if isinstance(context, Mapping)
        else "",
        key_ref=str(value.get("key_ref") or ""),
    )
    if (
        value.get("schema_version") != OPENROUTER_SEALED_ENVELOPE_SCHEMA_VERSION
        or value.get("envelope_hash") != sha256_json(body)
        or value.get("credential_slot") != _slot(str(value.get("credential_kind") or ""))
        or not _KEY_REF_RE.fullmatch(str(value.get("key_ref") or ""))
        or value.get("key_ref_hash")
        != sha256_bytes(str(value.get("key_ref") or "").encode("utf-8"))
        or not _HASH_RE.fullmatch(str(value.get("miner_hotkey_hash") or ""))
        or not _HASH_RE.fullmatch(str(value.get("credential_value_hash") or ""))
        or not _HASH_RE.fullmatch(str(value.get("kms_key_id_hash") or ""))
        or sha256_bytes(packed) != value.get("ciphertext_blob_hash")
        or not isinstance(context, Mapping)
        or dict(context) != expected_context
        or _miner_hash(str(context.get("miner_hotkey") or ""))
        != value.get("miner_hotkey_hash")
        or sha256_json(dict(context)) != value.get("encryption_context_hash")
        or document.get("ciphertext_hash") is None
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed envelope commitment is invalid"
        )
    return dict(value)


def build_openrouter_sealed_job_envelope_v2(
    row: Mapping[str, Any],
    *,
    job_id: str,
) -> Dict[str, Any]:
    normalized = validate_openrouter_sealed_envelope_v2(row)
    body = {
        "schema_version": OPENROUTER_SEALED_JOB_ENVELOPE_SCHEMA_VERSION,
        "job_id": str(job_id),
        "credential_slot": str(normalized["credential_slot"]),
        "credential_kind": str(normalized["credential_kind"]),
        "miner_hotkey_hash": str(normalized["miner_hotkey_hash"]),
        "credential_ref_hash": str(normalized["credential_value_hash"]),
        "credential_value_hash": str(normalized["credential_value_hash"]),
        "key_ref_hash": str(normalized["key_ref_hash"]),
        "ciphertext_blob_b64": str(normalized["ciphertext_blob_b64"]),
        "ciphertext_blob_hash": str(normalized["ciphertext_blob_hash"]),
        "kms_key_id_hash": str(normalized["kms_key_id_hash"]),
        "encryption_context": dict(normalized["encryption_context"]),
        "encryption_context_hash": str(normalized["encryption_context_hash"]),
    }
    return validate_openrouter_sealed_job_envelope_v2(body)


def validate_openrouter_sealed_job_envelope_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "job_id",
        "credential_slot",
        "credential_kind",
        "miner_hotkey_hash",
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
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed job envelope fields are invalid"
        )
    slot = str(value.get("credential_slot") or "")
    if slot not in {"openrouter", "openrouter_management"}:
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed job slot is invalid"
        )
    packed, _document = _decode_storage_document(value["ciphertext_blob_b64"])
    context = value.get("encryption_context")
    if (
        value.get("schema_version")
        != OPENROUTER_SEALED_JOB_ENVELOPE_SCHEMA_VERSION
        or not _JOB_ID_RE.fullmatch(str(value.get("job_id") or ""))
        or value.get("credential_slot")
        != _slot(str(value.get("credential_kind") or ""))
        or not _HASH_RE.fullmatch(str(value.get("miner_hotkey_hash") or ""))
        or not _HASH_RE.fullmatch(str(value.get("credential_value_hash") or ""))
        or not _HASH_RE.fullmatch(str(value.get("key_ref_hash") or ""))
        or not _HASH_RE.fullmatch(str(value.get("kms_key_id_hash") or ""))
        or value.get("credential_ref_hash") != value.get("credential_value_hash")
        or sha256_bytes(packed) != value.get("ciphertext_blob_hash")
        or not isinstance(context, Mapping)
        or sha256_json(dict(context)) != value.get("encryption_context_hash")
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed job envelope commitment is invalid"
        )
    return dict(value)


def unseal_openrouter_job_credential_v2(
    envelope: Mapping[str, Any],
    *,
    vault: EncryptedArtifactVaultV2,
) -> Dict[str, str]:
    normalized = validate_openrouter_sealed_job_envelope_v2(envelope)
    _packed, document = _decode_storage_document(
        normalized["ciphertext_blob_b64"]
    )
    plaintext = vault.decrypt_storage_document(document)
    try:
        secret_document = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed credential document is invalid"
        ) from exc
    if (
        not isinstance(secret_document, Mapping)
        or canonical_json(dict(secret_document)).encode("utf-8") != plaintext
        or set(secret_document)
        != {
            "schema_version",
            "key_ref_hash",
            "miner_hotkey_hash",
            "credential_kind",
            "credential_slot",
            "credential_value_hash",
            "credential",
        }
        or secret_document.get("schema_version")
        != "leadpoet.openrouter_stored_secret.v2"
        or secret_document.get("key_ref_hash") != normalized["key_ref_hash"]
        or secret_document.get("miner_hotkey_hash")
        != normalized["miner_hotkey_hash"]
        or secret_document.get("credential_kind")
        != normalized["credential_kind"]
        or secret_document.get("credential_slot")
        != normalized["credential_slot"]
        or secret_document.get("credential_value_hash")
        != normalized["credential_value_hash"]
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed secret scope differs"
        )
    credential = str(secret_document.get("credential") or "")
    if credential_value_hash(credential) != normalized["credential_value_hash"]:
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed credential hash differs"
        )
    aad = json.loads(
        base64.b64decode(str(document.get("aad_b64") or ""), validate=True)
    )
    if (
        not isinstance(aad, Mapping)
        or aad.get("artifact_kind") != "openrouter_credential"
        or aad.get("purpose") != "research_lab.openrouter_credential.v2"
        or not _JOB_ID_RE.fullmatch(str(aad.get("job_id") or ""))
        or aad.get("plaintext_hash") != sha256_bytes(plaintext)
    ):
        raise OpenRouterCredentialV2Error(
            "OpenRouter sealed credential artifact scope differs"
        )
    return {
        "job_id": str(normalized["job_id"]),
        "credential_slot": str(normalized["credential_slot"]),
        "credential_value_hash": str(normalized["credential_value_hash"]),
        "key_ref_hash": str(normalized["key_ref_hash"]),
        "credential": credential,
    }


class OpenRouterRegistrationAuthorityV2:
    """Run the unchanged OpenRouter registration checks inside the coordinator."""

    def __init__(
        self,
        *,
        broker: ProviderBrokerV2,
        transport: BrokeredProviderTransportV2,
        retry_policy_hashes: Mapping[str, str],
        vault: EncryptedArtifactVaultV2,
    ) -> None:
        self._broker = broker
        self._transport = transport
        self._retry_policy_hashes = dict(retry_policy_hashes)
        self._vault = vault

    def execute(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        required = {
            "schema_version",
            "miner_hotkey",
            "key_label",
            "runtime_credential",
            "management_credential",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != required
            or payload.get("schema_version")
            != OPENROUTER_REGISTRATION_REQUEST_SCHEMA_VERSION
        ):
            raise OpenRouterCredentialV2Error(
                "OpenRouter registration request is invalid"
            )
        miner_hotkey = str(payload.get("miner_hotkey") or "")
        runtime = unseal_openrouter_ingress_credential_v2(
            payload["runtime_credential"],
            miner_hotkey=miner_hotkey,
            vault=self._vault,
        )
        management = unseal_openrouter_ingress_credential_v2(
            payload["management_credential"],
            miner_hotkey=miner_hotkey,
            vault=self._vault,
        )
        if (
            runtime["credential_kind"] != "runtime"
            or management["credential_kind"] != "management"
        ):
            raise OpenRouterCredentialV2Error(
                "OpenRouter registration credential roles differ"
            )
        attempts: list[Dict[str, Any]] = []
        artifacts: set[str] = set()
        leased_count = 0
        try:
            for lease in (runtime, management):
                self._broker.provision_job_credential(
                    job_id=context.job_id,
                    slot=lease["credential_slot"],
                    credential=lease["credential"],
                    credential_value_hash_expected=lease[
                        "credential_value_hash"
                    ],
                )
                leased_count += 1
            with self._transport.scope(
                job_id=context.job_id,
                purpose=context.purpose,
                logical_operation_id="openrouter-registration:%s" % context.job_id,
                retry_policy_hashes=self._retry_policy_hashes,
                default_timeout_ms=15000,
                terminal_sink=lambda item: attempts.append(dict(item)),
                artifact_sink=lambda item: artifacts.add(str(item)),
            ):
                preflight_doc = preflight_openrouter_key(
                    runtime["credential"],
                    timeout_seconds=12,
                )
                privacy_proof_doc = verify_openrouter_workspace_privacy(
                    runtime_key=runtime["credential"],
                    management_key=management["credential"],
                    timeout_seconds=15,
                    stage="key_registration",
                )
        finally:
            if leased_count:
                released = self._broker.release_job_credentials(context.job_id)
                if int(released.get("released_slot_count") or 0) != leased_count:
                    raise OpenRouterCredentialV2Error(
                        "OpenRouter registration credential release failed"
                    )
        key_hash = str(preflight_doc["key_hash"])
        management_key_hash = str(privacy_proof_doc["management_key_hash"])
        key_ref = openrouter_key_ref(
            miner_hotkey=miner_hotkey,
            key_hash=key_hash,
            management_key_hash=management_key_hash,
        )
        envelopes = [
            seal_openrouter_persistent_credential_v2(
                credential=lease["credential"],
                credential_kind=lease["credential_kind"],
                key_ref=key_ref,
                miner_hotkey=miner_hotkey,
                job_id=context.job_id,
                purpose=context.purpose,
                vault=self._vault,
            )
            for lease in (runtime, management)
        ]
        output = {
            "schema_version": OPENROUTER_REGISTRATION_RESULT_SCHEMA_VERSION,
            "key_ref": key_ref,
            "key_hash": key_hash,
            "management_key_hash": management_key_hash,
            "preflight_doc": dict(preflight_doc),
            "privacy_proof_doc": dict(privacy_proof_doc),
            "credential_envelopes": envelopes,
        }
        artifacts.update(
            str(item)
            for envelope in envelopes
            for item in (
                envelope["envelope_hash"],
                envelope["ciphertext_blob_hash"],
                envelope["credential_value_hash"],
            )
        )
        return ExecutionResultV2(
            output=output,
            transport_attempts=tuple(attempts),
            artifact_hashes=tuple(sorted(artifacts)),
        )

    def preflight(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        required = {
            "schema_version",
            "key_ref_hash",
            "miner_hotkey_hash",
            "credential_value_hash",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != required
            or payload.get("schema_version")
            != OPENROUTER_PREFLIGHT_REQUEST_SCHEMA_VERSION
            or any(
                not _HASH_RE.fullmatch(str(payload.get(field) or ""))
                for field in (
                    "key_ref_hash",
                    "miner_hotkey_hash",
                    "credential_value_hash",
                )
            )
            or context.provider_credential_ref_hashes.get("openrouter")
            != payload.get("credential_value_hash")
        ):
            raise OpenRouterCredentialV2Error(
                "OpenRouter credit preflight request is invalid"
            )
        attempts: list[Dict[str, Any]] = []
        artifacts: set[str] = set()

        def execute_with_credential(credential: str) -> Dict[str, Any]:
            with self._transport.scope(
                job_id=context.job_id,
                purpose=context.purpose,
                logical_operation_id="openrouter-credit-preflight:%s"
                % context.job_id,
                retry_policy_hashes=self._retry_policy_hashes,
                default_timeout_ms=12000,
                terminal_sink=lambda item: attempts.append(dict(item)),
                artifact_sink=lambda item: artifacts.add(str(item)),
            ):
                return preflight_openrouter_key(
                    credential,
                    timeout_seconds=12,
                )

        preflight_doc = self._broker.use_job_credential(
            job_id=context.job_id,
            slot="openrouter",
            callback=execute_with_credential,
        )
        output = {
            "schema_version": OPENROUTER_PREFLIGHT_RESULT_SCHEMA_VERSION,
            "key_ref_hash": str(payload["key_ref_hash"]),
            "miner_hotkey_hash": str(payload["miner_hotkey_hash"]),
            "credential_value_hash": str(payload["credential_value_hash"]),
            "preflight_doc": dict(preflight_doc),
        }
        return ExecutionResultV2(
            output=output,
            transport_attempts=tuple(attempts),
            artifact_hashes=tuple(sorted(artifacts)),
        )
