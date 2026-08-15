#!/usr/bin/env python3
"""Publish one immutable, exact-byte production-parity evidence object."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError


BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{1,61}[A-Za-z0-9]$")
KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9!_.*'()/-]{1,1023}$")


class EvidencePublicationError(RuntimeError):
    pass


def _client_error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code") or "")


def _existing_bytes(client: Any, *, bucket: str, key: str) -> bytes | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _client_error_code(exc) in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    body = response.get("Body")
    if body is None:
        raise EvidencePublicationError("existing parity evidence has no body")
    value = body.read()
    if not isinstance(value, bytes):
        raise EvidencePublicationError("existing parity evidence body is invalid")
    return value


def publish_exact(
    *,
    client: Any,
    bucket: str,
    key: str,
    payload: bytes,
    kms_key_id: str,
    object_lock_days: int = 0,
) -> dict[str, Any]:
    if (
        not BUCKET_RE.fullmatch(bucket)
        or not KEY_RE.fullmatch(key)
        or not str(kms_key_id).startswith("arn:aws:kms:")
        or not payload
        or len(payload) > 16 * 1024 * 1024
        or object_lock_days < 0
        or object_lock_days > 3650
    ):
        raise EvidencePublicationError("parity evidence publication input is invalid")
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    existing = _existing_bytes(client, bucket=bucket, key=key)
    if existing is not None:
        if existing != payload:
            raise EvidencePublicationError(
                "immutable parity evidence already exists with different bytes"
            )
        return {"bucket": bucket, "key": key, "sha256": digest, "created": False}

    request: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": payload,
        "ContentType": "application/json",
        "ServerSideEncryption": "aws:kms",
        "SSEKMSKeyId": kms_key_id,
        "IfNoneMatch": "*",
    }
    if object_lock_days:
        request.update(
            {
                "ObjectLockMode": "COMPLIANCE",
                "ObjectLockRetainUntilDate": datetime.now(timezone.utc)
                + timedelta(days=object_lock_days),
            }
        )
    try:
        client.put_object(**request)
    except ClientError as exc:
        if _client_error_code(exc) not in {
            "PreconditionFailed",
            "412",
            "ConditionalRequestConflict",
        }:
            raise
        existing = _existing_bytes(client, bucket=bucket, key=key)
        if existing != payload:
            raise EvidencePublicationError(
                "concurrent parity evidence publication differs"
            ) from exc
        return {"bucket": bucket, "key": key, "sha256": digest, "created": False}
    return {"bucket": bucket, "key": key, "sha256": digest, "created": True}


def _file_digest(path: Path, *, max_bytes: int) -> tuple[str, str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise EvidencePublicationError(
                    "parity evidence file exceeds its bounded size"
                )
            digest.update(chunk)
    if size <= 0:
        raise EvidencePublicationError("parity evidence file is empty")
    raw = digest.digest()
    return (
        "sha256:" + digest.hexdigest(),
        base64.b64encode(raw).decode("ascii"),
        size,
    )


def _head_file_identity(
    client: Any, *, bucket: str, key: str, version_id: str = ""
) -> Mapping[str, Any] | None:
    request: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "ChecksumMode": "ENABLED",
    }
    if version_id:
        request["VersionId"] = version_id
    try:
        value = client.head_object(**request)
    except ClientError as exc:
        if _client_error_code(exc) in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    if not isinstance(value, Mapping):
        raise EvidencePublicationError("existing parity evidence metadata is invalid")
    return value


def _matches_file_identity(
    value: Mapping[str, Any],
    *,
    digest: str,
    checksum: str,
    size: int,
    kms_key_id: str,
    require_object_lock: bool,
) -> bool:
    metadata = value.get("Metadata")
    retained_until = value.get("ObjectLockRetainUntilDate")
    if isinstance(retained_until, str):
        try:
            retained_until = datetime.fromisoformat(
                retained_until.replace("Z", "+00:00")
            )
        except ValueError:
            return False
    lock_matches = not require_object_lock or (
        value.get("ObjectLockMode") in {"GOVERNANCE", "COMPLIANCE"}
        and isinstance(retained_until, datetime)
        and retained_until.astimezone(timezone.utc) > datetime.now(timezone.utc)
    )
    return (
        isinstance(metadata, Mapping)
        and metadata.get("sha256") == digest.removeprefix("sha256:")
        and value.get("ChecksumSHA256") == checksum
        and value.get("ContentLength") == size
        and value.get("ServerSideEncryption") == "aws:kms"
        and value.get("SSEKMSKeyId") == kms_key_id
        and bool(str(value.get("VersionId") or ""))
        and lock_matches
    )


def publish_file_exact(
    *,
    client: Any,
    bucket: str,
    key: str,
    path: Path,
    kms_key_id: str,
    content_type: str,
    max_bytes: int,
    object_lock_days: int = 0,
) -> dict[str, Any]:
    if (
        not BUCKET_RE.fullmatch(bucket)
        or not KEY_RE.fullmatch(key)
        or not str(kms_key_id).startswith("arn:aws:kms:")
        or not re.fullmatch(r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+", content_type)
        or max_bytes <= 0
        or max_bytes > 5 * 1024 * 1024 * 1024
        or object_lock_days < 0
        or object_lock_days > 3650
    ):
        raise EvidencePublicationError("parity file publication input is invalid")
    digest, checksum, size = _file_digest(path, max_bytes=max_bytes)
    existing = _head_file_identity(client, bucket=bucket, key=key)
    if existing is not None:
        if not _matches_file_identity(
            existing,
            digest=digest,
            checksum=checksum,
            size=size,
            kms_key_id=kms_key_id,
            require_object_lock=object_lock_days > 0,
        ):
            raise EvidencePublicationError(
                "immutable parity evidence already exists with different bytes"
            )
        return {
            "bucket": bucket,
            "key": key,
            "sha256": digest,
            "size_bytes": size,
            "version_id": str(existing["VersionId"]),
            "created": False,
        }

    request: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "ContentLength": size,
        "ContentType": content_type,
        "ServerSideEncryption": "aws:kms",
        "SSEKMSKeyId": kms_key_id,
        "ChecksumAlgorithm": "SHA256",
        "ChecksumSHA256": checksum,
        "Metadata": {"sha256": digest.removeprefix("sha256:")},
        "IfNoneMatch": "*",
    }
    if object_lock_days:
        request.update(
            {
                "ObjectLockMode": "COMPLIANCE",
                "ObjectLockRetainUntilDate": datetime.now(timezone.utc)
                + timedelta(days=object_lock_days),
            }
        )
    try:
        with path.open("rb") as source:
            response = client.put_object(**request, Body=source)
    except ClientError as exc:
        if _client_error_code(exc) not in {
            "PreconditionFailed",
            "412",
            "ConditionalRequestConflict",
        }:
            raise
        existing = _head_file_identity(client, bucket=bucket, key=key)
        if existing is None or not _matches_file_identity(
            existing,
            digest=digest,
            checksum=checksum,
            size=size,
            kms_key_id=kms_key_id,
            require_object_lock=object_lock_days > 0,
        ):
            raise EvidencePublicationError(
                "concurrent parity evidence publication differs"
            ) from exc
        return {
            "bucket": bucket,
            "key": key,
            "sha256": digest,
            "size_bytes": size,
            "version_id": str(existing["VersionId"]),
            "created": False,
        }
    version_id = str(response.get("VersionId") or "")
    created_head = _head_file_identity(
        client,
        bucket=bucket,
        key=key,
        version_id=version_id,
    )
    if created_head is None or not _matches_file_identity(
        created_head,
        digest=digest,
        checksum=checksum,
        size=size,
        kms_key_id=kms_key_id,
        require_object_lock=object_lock_days > 0,
    ):
        raise EvidencePublicationError(
            "created parity evidence metadata did not read back exactly"
        )
    return {
        "bucket": bucket,
        "key": key,
        "sha256": digest,
        "size_bytes": size,
        "version_id": version_id,
        "created": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--kms-key-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--object-lock-days", type=int, default=0)
    parser.add_argument("--content-type", default="application/json")
    parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        result = publish_file_exact(
            client=boto3.client("s3", region_name=args.region),
            bucket=args.bucket,
            key=args.key,
            path=args.file,
            kms_key_id=args.kms_key_id,
            content_type=str(args.content_type),
            max_bytes=int(args.max_bytes),
            object_lock_days=args.object_lock_days,
        )
    except (
        OSError,
        ValueError,
        BotoCoreError,
        ClientError,
        EvidencePublicationError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
