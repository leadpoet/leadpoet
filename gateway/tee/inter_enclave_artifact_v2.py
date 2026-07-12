"""Chunked, mutually attested artifact ingestion into the coordinator vault."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import re
import secrets
import threading
from typing import Any, Dict, Mapping

from leadpoet_canonical.attested_v2 import ROLE_PURPOSES, sha256_bytes, sha256_json


UPLOAD_SCHEMA_VERSION = "leadpoet.inter_enclave_artifact_upload.v2"
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_CHUNK_BYTES = 512 * 1024
MAX_ACTIVE_UPLOADS = 64
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_ARTIFACT_KINDS = frozenset(
    {
        "model_output",
        "model_trace",
        "provider_evidence_tape",
        "autoresearch_hidden_artifact",
    }
)


class InterEnclaveArtifactV2Error(RuntimeError):
    """An attested peer artifact upload is malformed, incomplete, or altered."""


def _scope(value: Any, field: str) -> str:
    text = str(value or "")
    if not text or len(text.encode("utf-8")) > 512 or "\x00" in text:
        raise InterEnclaveArtifactV2Error("%s is invalid" % field)
    return text


def _hash(value: Any, field: str) -> str:
    text = str(value or "").lower()
    if not _HASH_RE.fullmatch(text):
        raise InterEnclaveArtifactV2Error("%s is invalid" % field)
    return text


@dataclass
class _Upload:
    upload_id: str
    identity_hash: str
    peer_physical_role: str
    peer_boot_identity_hash: str
    job_id: str
    purpose: str
    artifact_kind: str
    plaintext_hash: str
    size_bytes: int
    content: bytearray = field(default_factory=bytearray)


class InterEnclaveArtifactIngestV2:
    """Accept plaintext only from an authenticated enclave TLS peer."""

    def __init__(self, *, vault: Any) -> None:
        self._vault = vault
        self._uploads: Dict[str, _Upload] = {}
        self._completed: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _peer(peer: Mapping[str, Any]) -> tuple[str, str, str]:
        physical_role = str(peer.get("physical_role") or "")
        service_role = str(peer.get("service_role") or "")
        boot = peer.get("boot_identity")
        if not isinstance(boot, Mapping):
            raise InterEnclaveArtifactV2Error("artifact peer boot is missing")
        boot_hash = _hash(boot.get("boot_identity_hash"), "artifact peer boot hash")
        if service_role not in {"gateway_scoring", "gateway_autoresearch"}:
            raise InterEnclaveArtifactV2Error("artifact peer role is not authorized")
        if boot.get("role") != service_role:
            raise InterEnclaveArtifactV2Error("artifact peer service role differs")
        return physical_role, service_role, boot_hash

    def begin(
        self,
        params: Mapping[str, Any],
        *,
        peer: Mapping[str, Any],
    ) -> Dict[str, Any]:
        required = {
            "schema_version",
            "job_id",
            "purpose",
            "artifact_kind",
            "plaintext_hash",
            "size_bytes",
        }
        if not isinstance(params, Mapping) or set(params) != required:
            raise InterEnclaveArtifactV2Error("artifact begin fields are invalid")
        if params.get("schema_version") != UPLOAD_SCHEMA_VERSION:
            raise InterEnclaveArtifactV2Error("artifact upload schema is invalid")
        physical_role, service_role, boot_hash = self._peer(peer)
        job_id = _scope(params.get("job_id"), "artifact job ID")
        purpose = _scope(params.get("purpose"), "artifact purpose")
        if purpose not in ROLE_PURPOSES[service_role]:
            raise InterEnclaveArtifactV2Error("artifact purpose is not authorized")
        artifact_kind = str(params.get("artifact_kind") or "")
        if artifact_kind not in _ARTIFACT_KINDS:
            raise InterEnclaveArtifactV2Error("artifact kind is not authorized")
        plaintext_hash = _hash(params.get("plaintext_hash"), "artifact plaintext hash")
        try:
            size_bytes = int(params.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise InterEnclaveArtifactV2Error("artifact size is invalid") from exc
        if size_bytes < 1 or size_bytes > MAX_ARTIFACT_BYTES:
            raise InterEnclaveArtifactV2Error("artifact size is outside limit")
        identity = sha256_json(
            {
                "schema_version": UPLOAD_SCHEMA_VERSION,
                "peer_physical_role": physical_role,
                "peer_boot_identity_hash": boot_hash,
                "job_id": job_id,
                "purpose": purpose,
                "artifact_kind": artifact_kind,
                "plaintext_hash": plaintext_hash,
                "size_bytes": size_bytes,
            }
        )
        with self._lock:
            for upload in self._uploads.values():
                if getattr(upload, "identity_hash", None) == identity:
                    return {
                        "status": "uploading",
                        "upload_id": upload.upload_id,
                        "uploaded_bytes": len(upload.content),
                    }
            upload_id = secrets.token_hex(16)
            if len(self._uploads) >= MAX_ACTIVE_UPLOADS:
                raise InterEnclaveArtifactV2Error("artifact upload capacity is full")
            upload = _Upload(
                upload_id=upload_id,
                identity_hash=identity,
                peer_physical_role=physical_role,
                peer_boot_identity_hash=boot_hash,
                job_id=job_id,
                purpose=purpose,
                artifact_kind=artifact_kind,
                plaintext_hash=plaintext_hash,
                size_bytes=size_bytes,
            )
            self._uploads[upload_id] = upload
        return {"status": "uploading", "upload_id": upload_id, "uploaded_bytes": 0}

    def put_chunk(
        self,
        params: Mapping[str, Any],
        *,
        peer: Mapping[str, Any],
    ) -> Dict[str, Any]:
        required = {"upload_id", "offset", "data_b64", "chunk_sha256"}
        if not isinstance(params, Mapping) or set(params) != required:
            raise InterEnclaveArtifactV2Error("artifact chunk fields are invalid")
        upload_id = str(params.get("upload_id") or "").lower()
        if not _UPLOAD_ID_RE.fullmatch(upload_id):
            raise InterEnclaveArtifactV2Error("artifact upload ID is invalid")
        try:
            offset = int(params.get("offset"))
            chunk = base64.b64decode(str(params.get("data_b64") or ""), validate=True)
        except Exception as exc:
            raise InterEnclaveArtifactV2Error("artifact chunk is invalid") from exc
        if not chunk or len(chunk) > MAX_CHUNK_BYTES:
            raise InterEnclaveArtifactV2Error("artifact chunk is outside limit")
        if sha256_bytes(chunk) != _hash(params.get("chunk_sha256"), "artifact chunk hash"):
            raise InterEnclaveArtifactV2Error("artifact chunk hash differs")
        physical_role, _service_role, boot_hash = self._peer(peer)
        with self._lock:
            upload = self._uploads.get(upload_id)
            if upload is None:
                raise InterEnclaveArtifactV2Error("artifact upload is unavailable")
            if (
                upload.peer_physical_role != physical_role
                or upload.peer_boot_identity_hash != boot_hash
            ):
                raise InterEnclaveArtifactV2Error("artifact upload peer differs")
            if offset != len(upload.content):
                raise InterEnclaveArtifactV2Error("artifact chunk offset differs")
            if len(upload.content) + len(chunk) > upload.size_bytes:
                raise InterEnclaveArtifactV2Error("artifact upload exceeds declared size")
            upload.content.extend(chunk)
            uploaded_bytes = len(upload.content)
        return {
            "status": "uploading",
            "upload_id": upload_id,
            "uploaded_bytes": uploaded_bytes,
        }

    def finish(
        self,
        params: Mapping[str, Any],
        *,
        peer: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(params, Mapping) or set(params) != {"upload_id"}:
            raise InterEnclaveArtifactV2Error("artifact finish fields are invalid")
        upload_id = str(params.get("upload_id") or "").lower()
        if not _UPLOAD_ID_RE.fullmatch(upload_id):
            raise InterEnclaveArtifactV2Error("artifact upload ID is invalid")
        physical_role, _service_role, boot_hash = self._peer(peer)
        with self._lock:
            completed = self._completed.get(upload_id)
            if completed is not None:
                return dict(completed)
            upload = self._uploads.get(upload_id)
            if upload is None:
                raise InterEnclaveArtifactV2Error("artifact upload is unavailable")
            if (
                upload.peer_physical_role != physical_role
                or upload.peer_boot_identity_hash != boot_hash
            ):
                raise InterEnclaveArtifactV2Error("artifact upload peer differs")
            payload = bytes(upload.content)
            if len(payload) != upload.size_bytes:
                raise InterEnclaveArtifactV2Error("artifact upload is incomplete")
            if sha256_bytes(payload) != upload.plaintext_hash:
                raise InterEnclaveArtifactV2Error("artifact plaintext hash differs")
            descriptor = self._vault.seal(
                payload,
                job_id=upload.job_id,
                purpose=upload.purpose,
                artifact_kind=upload.artifact_kind,
            )
            if descriptor.get("plaintext_hash") != upload.plaintext_hash:
                raise InterEnclaveArtifactV2Error("artifact vault commitment differs")
            result = {"status": "sealed", **dict(descriptor)}
            self._completed[upload_id] = result
            del self._uploads[upload_id]
        return dict(result)

    def cancel(
        self,
        params: Mapping[str, Any],
        *,
        peer: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(params, Mapping) or set(params) != {"upload_id"}:
            raise InterEnclaveArtifactV2Error("artifact cancel fields are invalid")
        upload_id = str(params.get("upload_id") or "").lower()
        physical_role, _service_role, boot_hash = self._peer(peer)
        with self._lock:
            upload = self._uploads.get(upload_id)
            if upload is not None and (
                upload.peer_physical_role != physical_role
                or upload.peer_boot_identity_hash != boot_hash
            ):
                raise InterEnclaveArtifactV2Error("artifact upload peer differs")
            self._uploads.pop(upload_id, None)
        return {"status": "cancelled", "upload_id": upload_id}


def seal_artifact_over_attested_tls_v2(
    *,
    client: Any,
    plaintext: bytes,
    job_id: str,
    purpose: str,
    artifact_kind: str,
) -> Dict[str, Any]:
    """Upload one artifact to the coordinator; every call is mTLS-protected."""

    payload = bytes(plaintext)
    if not payload or len(payload) > MAX_ARTIFACT_BYTES:
        raise InterEnclaveArtifactV2Error("artifact plaintext is outside limit")
    import secrets as _secrets

    def call(method: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        return client.call(
            target_physical_role="gateway_coordinator",
            method=method,
            params=params,
            channel_id=_secrets.token_hex(16),
        )

    started = call(
        "artifact_seal_begin",
        {
            "schema_version": UPLOAD_SCHEMA_VERSION,
            "job_id": str(job_id),
            "purpose": str(purpose),
            "artifact_kind": str(artifact_kind),
            "plaintext_hash": sha256_bytes(payload),
            "size_bytes": len(payload),
        },
    )
    upload_id = str(started.get("upload_id") or "")
    offset = int(started.get("uploaded_bytes") or 0)
    try:
        while offset < len(payload):
            chunk = payload[offset : offset + MAX_CHUNK_BYTES]
            progress = call(
                "artifact_seal_chunk",
                {
                    "upload_id": upload_id,
                    "offset": offset,
                    "data_b64": base64.b64encode(chunk).decode("ascii"),
                    "chunk_sha256": sha256_bytes(chunk),
                },
            )
            observed = int(progress.get("uploaded_bytes") or -1)
            if observed != offset + len(chunk):
                raise InterEnclaveArtifactV2Error("artifact upload progress differs")
            offset = observed
        result = call("artifact_seal_finish", {"upload_id": upload_id})
    except Exception:
        if _UPLOAD_ID_RE.fullmatch(upload_id):
            try:
                call("artifact_seal_cancel", {"upload_id": upload_id})
            except Exception:
                pass
        raise
    if (
        result.get("status") != "sealed"
        or result.get("plaintext_hash") != sha256_bytes(payload)
    ):
        raise InterEnclaveArtifactV2Error("artifact seal result differs")
    return dict(result)
