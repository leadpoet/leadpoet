"""Ciphertext-only parent adapter for V2 immutable artifact storage."""

from __future__ import annotations

import asyncio
from datetime import datetime
import os
import re
from typing import Any, Dict, Mapping, Optional

from gateway.utils.tee_client import coordinator_tee_client
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json


_ARTIFACT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ATTESTED_V2_ARTIFACT_KEY_PREFIX = "encrypted-artifacts"


class TEEArtifactStoreV2Error(RuntimeError):
    """The parent could not durably store or enclave-verify ciphertext."""


def _retain_until(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TEEArtifactStoreV2Error("artifact retention timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise TEEArtifactStoreV2Error("artifact retention timestamp lacks timezone")
    return parsed


def _object_key(prefix: str, artifact_id: str) -> str:
    normalized_prefix = str(prefix or "").strip("/")
    if not normalized_prefix or ".." in normalized_prefix:
        raise TEEArtifactStoreV2Error("artifact object prefix is invalid")
    if not _ARTIFACT_ID_RE.fullmatch(str(artifact_id or "")):
        raise TEEArtifactStoreV2Error("artifact id is invalid")
    return "%s/%s.json" % (normalized_prefix, artifact_id.split(":", 1)[1])


def _default_s3_client() -> Any:
    import boto3
    from botocore.config import Config

    region = str(
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or boto3.session.Session().region_name
        or "us-east-1"
    ).strip()
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
        ),
    )


async def persist_enclave_artifact_v2(
    artifact_id: str,
    *,
    bucket: str,
    key_prefix: str = ATTESTED_V2_ARTIFACT_KEY_PREFIX,
    client: Any = coordinator_tee_client,
    s3_client: Any = None,
    presign_expires_seconds: int = 300,
    attestation_job_id: str,
) -> Dict[str, Any]:
    """Upload only the encrypted envelope, then ask the enclave to read it back."""

    normalized_bucket = str(bucket or "").strip()
    if not normalized_bucket or not 60 <= int(presign_expires_seconds) <= 900:
        raise TEEArtifactStoreV2Error("artifact storage configuration is invalid")
    exported = await client.v2_export_encrypted_artifact(str(artifact_id))
    if not isinstance(exported, Mapping) or set(exported) != {
        "storage_document",
        "storage_document_hash",
    }:
        raise TEEArtifactStoreV2Error("encrypted artifact export is invalid")
    document = exported.get("storage_document")
    if not isinstance(document, Mapping) or document.get("artifact_id") != artifact_id:
        raise TEEArtifactStoreV2Error("encrypted artifact document is invalid")
    if sha256_json(dict(document)) != exported.get("storage_document_hash"):
        raise TEEArtifactStoreV2Error("encrypted artifact document hash mismatch")
    body = canonical_json(dict(document)).encode("utf-8")
    key = _object_key(key_prefix, str(artifact_id))
    if s3_client is None:
        s3_client = _default_s3_client()
    response = await asyncio.to_thread(
        s3_client.put_object,
        Bucket=normalized_bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        ObjectLockMode="COMPLIANCE",
        ObjectLockRetainUntilDate=_retain_until(document.get("retain_until")),
    )
    status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
    if status not in {200, 201}:
        raise TEEArtifactStoreV2Error("S3 ciphertext upload failed")
    params = {"Bucket": normalized_bucket, "Key": key}
    get_url = await asyncio.to_thread(
        s3_client.generate_presigned_url,
        "get_object",
        Params=params,
        ExpiresIn=int(presign_expires_seconds),
        HttpMethod="GET",
    )
    head_url = await asyncio.to_thread(
        s3_client.generate_presigned_url,
        "head_object",
        Params=params,
        ExpiresIn=int(presign_expires_seconds),
        HttpMethod="HEAD",
    )
    artifact_ref = "s3://%s/%s" % (normalized_bucket, key)
    verified = await client.v2_verify_encrypted_artifact_persistence(
        artifact_id=str(artifact_id),
        attestation_job_id=str(attestation_job_id),
        artifact_ref=artifact_ref,
        get_url=str(get_url),
        head_url=str(head_url),
    )
    if verified.get("status") != "persisted":
        raise TEEArtifactStoreV2Error(
            "enclave rejected artifact persistence: %s"
            % str(verified.get("failure_code") or "unknown")
        )
    return {
        "artifact_id": str(artifact_id),
        "artifact_ref": artifact_ref,
        "artifact_kind": "provider_response",
        "artifact_hash": str(document["ciphertext_hash"]),
        "encryption_context_hash": str(document["encryption_context_hash"]),
        "object_lock_mode": str(document["object_lock_mode"]),
        "retain_until": str(document["retain_until"]),
        "storage_document_hash": str(exported["storage_document_hash"]),
        "transport_root": str(verified["transport_root"]),
        "status": "persisted",
    }
