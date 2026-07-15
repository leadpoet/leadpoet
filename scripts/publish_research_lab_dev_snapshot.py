#!/usr/bin/env python3
"""Publish one immutable Research Lab dev snapshot to S3, with READY last."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.config import (  # noqa: E402
    ResearchLabGitTreeConfig,
    ResearchLabGitTreeConfigError,
)
from research_lab.eval.snapshot_store import (  # noqa: E402
    POINTER_NAME,
    READY_NAME,
    RECORD_FAILURES_NAME,
    ProviderSnapshotStore,
    _parse_s3_root,
    build_snapshot_pointer_document,
    verify_snapshot_pointer_document,
)


def _content_type(path: Path) -> str:
    return "application/json" if path.suffix in {".json", ".jsonl"} else "application/octet-stream"


def _put(client: Any, *, bucket: str, key: str, body: bytes, kms_key_id: str, content_type: str) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        ServerSideEncryption="aws:kms",
        SSEKMSKeyId=kms_key_id,
    )


def _signed_document(kms: Any, *, document: dict[str, Any], hash_field: str, kms_key_id: str) -> dict[str, Any]:
    document_hash = str(document.get(hash_field) or "")
    signed = kms.sign(
        KeyId=kms_key_id,
        Message=document_hash.encode("utf-8"),
        MessageType="RAW",
        SigningAlgorithm="ECDSA_SHA_256",
    )
    return {
        **document,
        "kms_key_id": str(signed.get("KeyId") or kms_key_id),
        "signing_algorithm": str(signed.get("SigningAlgorithm") or "ECDSA_SHA_256"),
        "signature_b64": base64.b64encode(bytes(signed["Signature"])).decode("ascii"),
    }


def _publish_current_pointer(
    *,
    s3: Any,
    kms: Any,
    base_uri: str,
    target_uri: str,
    manifest_hash: str,
    ready_hash: str,
    recorded_at: str,
    kms_key_id: str,
) -> tuple[str, str]:
    base_bucket, base_prefix = _parse_s3_root(base_uri)
    pointer_uri = f"s3://{base_bucket}/{base_prefix}{POINTER_NAME}"
    pointer = build_snapshot_pointer_document(
        snapshot_uri=target_uri,
        manifest_hash=manifest_hash,
        ready_hash=ready_hash,
        recorded_at=recorded_at,
    )
    published_pointer = _signed_document(
        kms,
        document=pointer,
        hash_field="pointer_hash",
        kms_key_id=kms_key_id,
    )
    _put(
        s3,
        bucket=base_bucket,
        key=f"{base_prefix}{POINTER_NAME}",
        body=json.dumps(published_pointer, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        kms_key_id=kms_key_id,
        content_type="application/json",
    )
    verification = verify_snapshot_pointer_document(pointer_uri, require_signature=True)
    if not verification.get("passed"):
        raise RuntimeError(
            "published current pointer verification failed: "
            + "; ".join(verification.get("errors") or ())
        )
    if verification.get("snapshot_uri") != target_uri:
        raise RuntimeError("published current pointer resolved to a different snapshot")
    return pointer_uri, str(pointer["pointer_hash"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--s3-base-uri", required=True)
    parser.add_argument("--kms-key-id", required=True)
    parser.add_argument(
        "--skip-current-pointer",
        action="store_true",
        help="Publish and verify the immutable snapshot without advancing current.json",
    )
    args = parser.parse_args()

    try:
        expected_dev_icp_count = (
            ResearchLabGitTreeConfig.from_env().live_max_icps_per_node
        )
    except ResearchLabGitTreeConfigError as exc:
        print(f"ERROR: invalid Git-tree configuration: {exc}")
        return 1

    try:
        import boto3
    except Exception as exc:
        print(f"ERROR: boto3 is required: {exc}")
        return 1

    source = Path(args.source_dir).expanduser().resolve()
    local = ProviderSnapshotStore(str(source), mode="replay")
    verification = local.verify_ready_document(
        expected_dev_icp_count=expected_dev_icp_count,
        require_signature=False,
    )
    if not verification.get("passed"):
        print("ERROR: local snapshot is not READY: " + "; ".join(verification.get("errors") or ()))
        return 1
    manifest = local.load_manifest() or {}
    manifest_hash = str(manifest.get("manifest_hash") or "")
    if not manifest_hash.startswith("sha256:"):
        print("ERROR: local manifest hash is invalid")
        return 1

    base = str(args.s3_base_uri).rstrip("/")
    target_uri = f"{base}/{manifest_hash.split(':', 1)[1]}"
    bucket, prefix = _parse_s3_root(target_uri)
    s3 = boto3.client("s3")
    kms = boto3.client("kms")

    existing = s3.list_objects_v2(
        Bucket=bucket, Prefix=prefix, MaxKeys=1
    ).get("Contents") or []
    already_published = False
    if existing:
        remote = ProviderSnapshotStore(target_uri, mode="replay")
        remote_verification = remote.verify_ready_document(
            expected_dev_icp_count=expected_dev_icp_count,
            require_signature=True,
        )
        if not (
            remote_verification.get("passed")
            and remote_verification.get("manifest_hash") == manifest_hash
        ):
            print(
                "ERROR: immutable target already contains incomplete or different "
                f"data: {target_uri}"
            )
            return 1
        already_published = True
        ready_hash = str(remote_verification.get("ready_hash") or "")
    else:
        files = [path for path in sorted(source.rglob("*")) if path.is_file()]
        for path in files:
            relative = path.relative_to(source).as_posix()
            if relative in {READY_NAME, RECORD_FAILURES_NAME}:
                continue
            _put(
                s3,
                bucket=bucket,
                key=f"{prefix}{relative}",
                body=path.read_bytes(),
                kms_key_id=args.kms_key_id,
                content_type=_content_type(path),
            )

        remote = ProviderSnapshotStore(target_uri, mode="replay")
        manifest_verification = remote.verify_manifest(
            expected_icp_set_hash=str(manifest.get("icp_set_hash") or "")
        )
        if not manifest_verification.get("passed"):
            print(
                "ERROR: uploaded objects failed verification before READY: "
                + "; ".join(manifest_verification.get("errors") or ())
            )
            return 1

        ready = local.load_ready_document() or {}
        ready_hash = str(ready.get("ready_hash") or "")
        published_ready = _signed_document(
            kms,
            document=dict(ready),
            hash_field="ready_hash",
            kms_key_id=args.kms_key_id,
        )
        _put(
            s3,
            bucket=bucket,
            key=f"{prefix}{READY_NAME}",
            body=json.dumps(
                published_ready, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            kms_key_id=args.kms_key_id,
            content_type="application/json",
        )
        final = remote.verify_ready_document(
            expected_dev_icp_count=expected_dev_icp_count,
            require_signature=True,
        )
        if not final.get("passed"):
            print(
                "ERROR: published READY verification failed: "
                + "; ".join(final.get("errors") or ())
            )
            return 1

    if args.skip_current_pointer:
        if already_published:
            print(f"already_published={target_uri}")
        print(f"snapshot_uri={target_uri}")
        print(f"manifest_hash={manifest_hash}")
        print(f"ready_hash={ready_hash}")
        print("current_pointer_updated=false")
        return 0

    try:
        pointer_uri, pointer_hash = _publish_current_pointer(
            s3=s3,
            kms=kms,
            base_uri=base,
            target_uri=target_uri,
            manifest_hash=manifest_hash,
            ready_hash=ready_hash,
            recorded_at=str(manifest.get("recorded_at") or ""),
            kms_key_id=args.kms_key_id,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    if already_published:
        print(f"already_published={target_uri}")
    print(f"snapshot_uri={target_uri}")
    print(f"snapshot_pointer_uri={pointer_uri}")
    print(f"manifest_hash={manifest_hash}")
    print(f"ready_hash={ready_hash}")
    print(f"pointer_hash={pointer_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
