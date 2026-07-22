"""Coordinator-local encryption and persistence state for hidden V2 artifacts."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import secrets
import threading
from typing import Any, Dict, Mapping

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json
from leadpoet_canonical.attested_v2 import transport_root, validate_transport_attempt


ARTIFACT_ENVELOPE_SCHEMA_VERSION = "leadpoet.encrypted_artifact.v2"
ARTIFACT_MASTER_KEY_SLOT = "artifact_master_key"
ARTIFACT_MASTER_KEY_HASH_DOMAIN = b"leadpoet-artifact-master-key-v2:"
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_IN_MEMORY_ARTIFACTS = 2048
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ArtifactVaultV2Error(RuntimeError):
    """A hidden artifact is unencrypted, uncommitted, or not durably locked."""


def artifact_master_key_reference_hash(key: bytes) -> str:
    if not isinstance(key, (bytes, bytearray)) or len(key) != 32:
        raise ArtifactVaultV2Error("artifact master key must be 32 bytes")
    return "sha256:" + hashlib.sha256(
        ARTIFACT_MASTER_KEY_HASH_DOMAIN + bytes(key)
    ).hexdigest()


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ArtifactVaultV2Error("artifact retention timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ArtifactVaultV2Error("artifact retention timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


class EncryptedArtifactVaultV2:
    """Seal plaintext immediately and expose ciphertext only to the parent."""

    def __init__(
        self,
        *,
        master_key: bytes,
        boot_identity_hash: str,
        retention_days: int = 365,
        clock: Any = lambda: datetime.now(timezone.utc),
    ) -> None:
        if len(bytes(master_key)) != 32:
            raise ArtifactVaultV2Error("artifact master key must be 32 bytes")
        if not _HASH_RE.fullmatch(str(boot_identity_hash or "")):
            raise ArtifactVaultV2Error("artifact vault boot identity is invalid")
        self._aead = AESGCM(bytes(master_key))
        self._boot_identity_hash = str(boot_identity_hash)
        self._retention_days = max(1, int(retention_days))
        self._clock = clock
        self._artifacts = {}  # type: Dict[str, Dict[str, Any]]
        self._persisted_artifacts = {}  # type: Dict[str, Dict[str, Any]]
        self._lock = threading.RLock()

    def seal(
        self,
        plaintext: bytes,
        *,
        job_id: str,
        purpose: str,
        artifact_kind: str,
    ) -> Dict[str, Any]:
        payload = bytes(plaintext)
        if not payload or len(payload) > MAX_ARTIFACT_BYTES:
            raise ArtifactVaultV2Error("artifact plaintext is outside limit")
        now = self._clock()
        retain_until = _timestamp(now + timedelta(days=self._retention_days))
        plaintext_hash = sha256_bytes(payload)
        aad_document = {
            "schema_version": ARTIFACT_ENVELOPE_SCHEMA_VERSION,
            "boot_identity_hash": self._boot_identity_hash,
            "job_id": str(job_id),
            "purpose": str(purpose),
            "artifact_kind": str(artifact_kind),
            "plaintext_hash": plaintext_hash,
            "object_lock_mode": "COMPLIANCE",
            "retain_until": retain_until,
        }
        aad = canonical_json(aad_document).encode("utf-8")
        nonce = secrets.token_bytes(12)
        ciphertext = self._aead.encrypt(nonce, payload, aad)
        ciphertext_hash = sha256_bytes(ciphertext)
        artifact_id = sha256_json(
            {
                **aad_document,
                "nonce_b64": base64.b64encode(nonce).decode("ascii"),
                "ciphertext_hash": ciphertext_hash,
            }
        )
        record = {
            "artifact_id": artifact_id,
            "plaintext_hash": plaintext_hash,
            "ciphertext_hash": ciphertext_hash,
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "aad_b64": base64.b64encode(aad).decode("ascii"),
            "encryption_context_hash": sha256_bytes(aad),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
            "artifact_kind": str(artifact_kind),
            "job_id": str(job_id),
            "purpose": str(purpose),
            "object_lock_mode": "COMPLIANCE",
            "retain_until": retain_until,
            "persistence": None,
        }
        with self._lock:
            existing = self._artifacts.get(artifact_id)
            if existing is not None:
                return self.descriptor(artifact_id)
            if len(self._artifacts) >= MAX_IN_MEMORY_ARTIFACTS:
                raise ArtifactVaultV2Error("artifact vault capacity is full")
            self._artifacts[artifact_id] = record
        return self.descriptor(artifact_id)

    def descriptor(self, artifact_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._artifacts.get(str(artifact_id or ""))
            if record is None:
                record = self._persisted_artifacts.get(str(artifact_id or ""))
            if record is None:
                raise ArtifactVaultV2Error("encrypted artifact is unavailable")
            return {
                field: record[field]
                for field in (
                    "artifact_id",
                    "plaintext_hash",
                    "ciphertext_hash",
                    "artifact_kind",
                    "job_id",
                    "purpose",
                    "object_lock_mode",
                    "retain_until",
                    "encryption_context_hash",
                )
            } | {"persisted": record["persistence"] is not None}

    def export_ciphertext(self, artifact_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._artifacts.get(str(artifact_id or ""))
            if record is None:
                raise ArtifactVaultV2Error("encrypted artifact is unavailable")
            document = {
                "schema_version": ARTIFACT_ENVELOPE_SCHEMA_VERSION,
                "artifact_id": record["artifact_id"],
                "plaintext_hash": record["plaintext_hash"],
                "ciphertext_hash": record["ciphertext_hash"],
                "nonce_b64": record["nonce_b64"],
                "aad_b64": record["aad_b64"],
                "encryption_context_hash": record["encryption_context_hash"],
                "ciphertext_b64": record["ciphertext_b64"],
                "object_lock_mode": record["object_lock_mode"],
                "retain_until": record["retain_until"],
            }
            return {
                "storage_document": document,
                "storage_document_hash": sha256_json(document),
            }

    def release_transient(self, artifact_id: str) -> None:
        """Drop an envelope after an authenticated durable readback.

        Persistent cache/checkpoint rows carry the complete authenticated
        ciphertext and can be reopened with the KMS-released master key. Keeping
        every exported envelope in enclave memory would otherwise impose an
        artificial per-boot provider-call ceiling.
        """

        with self._lock:
            record = self._artifacts.get(str(artifact_id or ""))
            if record is None:
                raise ArtifactVaultV2Error("encrypted artifact is unavailable")
            if record["persistence"] is not None:
                raise ArtifactVaultV2Error("persisted Object Lock artifact is not transient")
            del self._artifacts[str(artifact_id)]

    def decrypt_storage_document(
        self,
        storage_document: Mapping[str, Any],
    ) -> bytes:
        """Verify and open an envelope created by this persistent master key."""

        fields = {
            "schema_version",
            "artifact_id",
            "plaintext_hash",
            "ciphertext_hash",
            "nonce_b64",
            "aad_b64",
            "encryption_context_hash",
            "ciphertext_b64",
            "object_lock_mode",
            "retain_until",
        }
        if not isinstance(storage_document, Mapping) or set(storage_document) != fields:
            raise ArtifactVaultV2Error("encrypted storage document fields are invalid")
        document = dict(storage_document)
        if document.get("schema_version") != ARTIFACT_ENVELOPE_SCHEMA_VERSION:
            raise ArtifactVaultV2Error("encrypted storage document schema is invalid")
        try:
            nonce = base64.b64decode(str(document["nonce_b64"]), validate=True)
            aad = base64.b64decode(str(document["aad_b64"]), validate=True)
            ciphertext = base64.b64decode(
                str(document["ciphertext_b64"]), validate=True
            )
            aad_document = json.loads(aad.decode("utf-8"))
        except Exception as exc:
            raise ArtifactVaultV2Error(
                "encrypted storage document encoding is invalid"
            ) from exc
        if (
            len(nonce) != 12
            or not isinstance(aad_document, Mapping)
            or canonical_json(dict(aad_document)).encode("utf-8") != aad
            or sha256_bytes(aad) != document["encryption_context_hash"]
            or sha256_bytes(ciphertext) != document["ciphertext_hash"]
            or aad_document.get("schema_version")
            != ARTIFACT_ENVELOPE_SCHEMA_VERSION
            or aad_document.get("plaintext_hash") != document["plaintext_hash"]
            or aad_document.get("object_lock_mode") != document["object_lock_mode"]
            or aad_document.get("retain_until") != document["retain_until"]
        ):
            raise ArtifactVaultV2Error("encrypted storage document commitment differs")
        expected_id = sha256_json(
            {
                **dict(aad_document),
                "nonce_b64": document["nonce_b64"],
                "ciphertext_hash": document["ciphertext_hash"],
            }
        )
        if expected_id != document["artifact_id"]:
            raise ArtifactVaultV2Error("encrypted storage document ID differs")
        try:
            plaintext = self._aead.decrypt(nonce, ciphertext, aad)
        except Exception as exc:
            raise ArtifactVaultV2Error(
                "encrypted storage document authentication failed"
            ) from exc
        if sha256_bytes(plaintext) != document["plaintext_hash"]:
            raise ArtifactVaultV2Error("encrypted storage plaintext hash differs")
        return plaintext

    def confirm_persistence(
        self,
        *,
        artifact_id: str,
        artifact_ref: str,
        observed_storage_document: Mapping[str, Any],
        response_headers: Mapping[str, Any],
        transport_attempts: Any,
    ) -> Dict[str, Any]:
        if not str(artifact_ref or "").strip():
            raise ArtifactVaultV2Error("artifact reference is required")
        headers = {str(name).lower(): str(value) for name, value in response_headers.items()}
        object_lock_mode = headers.get("x-amz-object-lock-mode", "").upper()
        if object_lock_mode != "COMPLIANCE":
            raise ArtifactVaultV2Error("artifact must use COMPLIANCE Object Lock")
        observed_retain_until = _parse_utc_timestamp(
            headers.get("x-amz-object-lock-retain-until-date", "")
        )
        normalized_attempts = []
        for attempt in transport_attempts:
            validate_transport_attempt(attempt)
            normalized_attempts.append(dict(attempt))
        if len(normalized_attempts) != 2 or [
            item["method"] for item in normalized_attempts
        ] != ["GET", "HEAD"]:
            raise ArtifactVaultV2Error(
                "artifact persistence requires authenticated GET and HEAD"
            )
        with self._lock:
            record = self._artifacts.get(str(artifact_id or ""))
            persisted_record = self._persisted_artifacts.get(str(artifact_id or ""))
            if record is None and persisted_record is not None:
                persistence = persisted_record["persistence"]
                if sha256_json(dict(observed_storage_document)) != persistence.get(
                    "storage_document_hash"
                ):
                    raise ArtifactVaultV2Error("persisted artifact ciphertext differs")
                immutable_fields = {
                    "artifact_ref": str(artifact_ref),
                    "ciphertext_hash": str(
                        observed_storage_document.get("ciphertext_hash") or ""
                    ),
                    "object_lock_mode": object_lock_mode,
                    "retain_until": _timestamp(observed_retain_until),
                }
                if any(
                    persistence.get(field) != value
                    for field, value in immutable_fields.items()
                ):
                    raise ArtifactVaultV2Error("artifact persistence is immutable")
                return {
                    **self.descriptor(artifact_id),
                    "artifact_ref": persistence["artifact_ref"],
                }
            if record is None:
                raise ArtifactVaultV2Error("encrypted artifact is unavailable")
            expected_document = self.export_ciphertext(artifact_id)["storage_document"]
            if dict(observed_storage_document) != expected_document:
                raise ArtifactVaultV2Error("persisted artifact ciphertext differs")
            required_retain_until = _parse_utc_timestamp(record["retain_until"])
            if observed_retain_until < required_retain_until:
                raise ArtifactVaultV2Error("persisted artifact retention is too short")
            normalized_retain_until = _timestamp(observed_retain_until)
            persistence = {
                "artifact_ref": str(artifact_ref),
                "ciphertext_hash": record["ciphertext_hash"],
                "object_lock_mode": object_lock_mode,
                "retain_until": normalized_retain_until,
                "storage_document_hash": sha256_json(expected_document),
                "transport_attempts": normalized_attempts,
                "transport_root": transport_root(normalized_attempts),
            }
            existing = record["persistence"]
            if existing is not None:
                immutable_fields = (
                    "artifact_ref",
                    "ciphertext_hash",
                    "object_lock_mode",
                    "retain_until",
                    "storage_document_hash",
                )
                if any(
                    existing.get(field) != persistence.get(field)
                    for field in immutable_fields
                ):
                    raise ArtifactVaultV2Error("artifact persistence is immutable")
                return {
                    **self.descriptor(artifact_id),
                    "artifact_ref": existing["artifact_ref"],
                }
            record["persistence"] = persistence
            self._persisted_artifacts[str(artifact_id)] = {
                field: record[field]
                for field in (
                    "artifact_id",
                    "plaintext_hash",
                    "ciphertext_hash",
                    "artifact_kind",
                    "job_id",
                    "purpose",
                    "object_lock_mode",
                    "retain_until",
                    "encryption_context_hash",
                    "persistence",
                )
            }
            del self._artifacts[str(artifact_id)]
            return {
                **self.descriptor(artifact_id),
                "artifact_ref": persistence["artifact_ref"],
            }

    def persistence_evidence(self, artifact_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._artifacts.get(str(artifact_id or ""))
            if record is None:
                record = self._persisted_artifacts.get(str(artifact_id or ""))
            if record is None or record["persistence"] is None:
                raise ArtifactVaultV2Error("encrypted artifact is not persisted")
            persistence = record["persistence"]
            return {
                **self.descriptor(artifact_id),
                "artifact_ref": persistence["artifact_ref"],
                "storage_document_hash": persistence["storage_document_hash"],
                "transport_attempts": [
                    dict(item) for item in persistence["transport_attempts"]
                ],
                "transport_root": persistence["transport_root"],
            }

    def job_artifacts(self, *, job_id: str, purpose: str) -> tuple[Dict[str, Any], ...]:
        with self._lock:
            ids = sorted(
                artifact_id
                for artifact_id, record in {
                    **self._persisted_artifacts,
                    **self._artifacts,
                }.items()
                if record["job_id"] == str(job_id)
                and record["purpose"] == str(purpose)
            )
        return tuple(self.descriptor(artifact_id) for artifact_id in ids)

    def require_persisted(self, artifact_ids: Any) -> None:
        for artifact_id in artifact_ids:
            if not self.descriptor(str(artifact_id))["persisted"]:
                raise ArtifactVaultV2Error("required encrypted artifact is not persisted")
