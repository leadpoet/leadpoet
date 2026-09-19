"""Publish and acquire exact-commit V2 release manifests.

The channel contains only manifests derived from the two independent parent
builders.  It cannot create evidence or approve a release.  Restart scripts
may consume it before shutdown, but still run the normal local preflight and
reproducible-build verification afterward.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.tee.release_manifest_v2 import (
    LOCAL_RELEASE_SCHEMA_VERSION,
    validate_release_manifest,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
SCHEMA_VERSION = "leadpoet.gateway_release_channel.v3"
DEFAULT_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
DEFAULT_PREFIX = "attested-v2/releases"
DEFAULT_RETENTION_DAYS = 365
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_LOCAL_COMMIT_ENV = "LEADPOET_LOCAL_RELEASE_COMMIT_SHA"
_LOCAL_GATEWAY_ENV = "LEADPOET_LOCAL_GATEWAY_RELEASE"


class ReleaseChannelV2Error(RuntimeError):
    """An independently built release channel is unavailable or inconsistent."""


def release_channel_key(commit_sha: str, *, prefix: str = DEFAULT_PREFIX) -> str:
    commit = str(commit_sha or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ReleaseChannelV2Error("release channel commit is invalid")
    normalized_prefix = str(prefix or "").strip("/")
    if not normalized_prefix or ".." in normalized_prefix.split("/"):
        raise ReleaseChannelV2Error("release channel prefix is invalid")
    return f"{normalized_prefix}/{commit}/release-channel-v2.json"


def _build_release_channel_v2(*, gateway_release_manifest: Mapping[str, Any], gateway_validator: Any) -> Dict[str, Any]:
    gateway = gateway_validator(gateway_release_manifest)
    commit = gateway["commit_sha"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "commit_sha": commit,
        "gateway_release_manifest": gateway,
    }
    hash_body = (
        {
            "schema_version": SCHEMA_VERSION,
            "commit_sha": commit,
            "gateway_release_hash": gateway["release_hash"],
        }
        if gateway["schema_version"] == LOCAL_RELEASE_SCHEMA_VERSION else body
    )
    return {**body, "channel_hash": sha256_json(hash_body)}


def build_release_channel_v2(
    *, gateway_release_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    """Build the current gateway-only release channel."""

    return _build_release_channel_v2(
        gateway_release_manifest=gateway_release_manifest,
        gateway_validator=validate_release_manifest,
    )


def _validate_release_channel_v2(
    value: Mapping[str, Any],
    *,
    expected_commit: Optional[str],
    gateway_validator: Any,
) -> Dict[str, Any]:
    fields = {"schema_version", "commit_sha", "gateway_release_manifest", "channel_hash"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReleaseChannelV2Error("release channel fields are invalid")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseChannelV2Error("release channel schema is invalid")
    normalized = _build_release_channel_v2(
        gateway_release_manifest=value["gateway_release_manifest"],
        gateway_validator=gateway_validator,
    )
    if value.get("commit_sha") != normalized["commit_sha"]:
        raise ReleaseChannelV2Error("release channel commit differs")
    if value.get("channel_hash") != normalized["channel_hash"]:
        raise ReleaseChannelV2Error("release channel hash differs")
    if expected_commit is not None and normalized["commit_sha"] != str(
        expected_commit
    ).lower():
        raise ReleaseChannelV2Error("release channel is for another commit")
    return normalized


def validate_release_channel_v2(
    value: Mapping[str, Any], *, expected_commit: Optional[str] = None
) -> Dict[str, Any]:
    """Validate a channel for the canonical current topology."""

    return _validate_release_channel_v2(
        value,
        expected_commit=expected_commit,
        gateway_validator=validate_release_manifest,
    )


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    descriptor = -1
    try:
        descriptor = os.open(
            str(Path(path)),
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= 4 * 1024 * 1024:
            raise ReleaseChannelV2Error(f"{label} is not a bounded regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(4 * 1024 * 1024 + 1)
        if not 0 < len(payload) <= 4 * 1024 * 1024:
            raise ReleaseChannelV2Error(f"{label} is not a bounded regular file")
        value = json.loads(payload)
    except ReleaseChannelV2Error:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseChannelV2Error(f"{label} is unavailable or invalid") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, Mapping):
        raise ReleaseChannelV2Error(f"{label} must be an object")
    return dict(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((canonical_json(dict(value)) + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def install_release_channel_v2(
    channel: Mapping[str, Any],
    *,
    expected_commit: str,
    gateway_output: Optional[Path] = None,
) -> Dict[str, Any]:
    normalized = validate_release_channel_v2(
        channel, expected_commit=expected_commit
    )
    if gateway_output is not None:
        _atomic_json(gateway_output, normalized["gateway_release_manifest"])
    return normalized


def local_release_inputs_match(
    *,
    expected_commit: str,
    gateway_output: Optional[Path],
) -> bool:
    try:
        if gateway_output is not None:
            gateway = validate_release_manifest(
                _load_json(gateway_output, "local gateway release manifest")
            )
            if gateway["commit_sha"] != expected_commit:
                return False
    except Exception:
        return False
    return gateway_output is not None


def fetch_release_channel_v2(
    *,
    bucket: str,
    commit_sha: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client: Any = None,
) -> Dict[str, Any]:
    local_values = (os.environ.get(_LOCAL_COMMIT_ENV), os.environ.get(_LOCAL_GATEWAY_ENV))
    if any(local_values):
        if not all(local_values):
            raise ReleaseChannelV2Error(
                "local release identity environment is incomplete"
            )
        local_commit, gateway_path = local_values
        if str(local_commit).lower() == str(commit_sha).lower():
            gateway = _load_json(Path(str(gateway_path)), "local gateway release")
            return validate_release_channel_v2(
                build_release_channel_v2(
                    gateway_release_manifest=gateway,
                ),
                expected_commit=str(commit_sha).lower(),
            )
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    try:
        response = s3_client.get_object(
            Bucket=str(bucket), Key=release_channel_key(commit_sha, prefix=prefix)
        )
        payload = response["Body"].read()
        value = json.loads(payload)
    except Exception as exc:
        raise ReleaseChannelV2Error(
            "approved release channel is unavailable"
        ) from exc
    return validate_release_channel_v2(value, expected_commit=commit_sha)


def release_channel_role_identities_v2(
    channel: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the exact compact role identities from one full channel."""

    normalized = validate_release_channel_v2(channel)
    gateway = normalized["gateway_release_manifest"]
    return {
        role: {
            "commit_sha": summary["commit_sha"],
            "pcr0": summary["pcr0"],
            "build_manifest_hash": summary["execution_manifest_hash"],
            "dependency_lock_hash": summary["dependency_lock_hash"],
        }
        for role, summary in sorted(gateway["roles"].items())
    }


def publish_release_channel_v2(
    channel: Mapping[str, Any],
    *,
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    s3_client: Any = None,
) -> Dict[str, Any]:
    normalized = validate_release_channel_v2(channel)
    if int(retention_days) < DEFAULT_RETENTION_DAYS:
        raise ReleaseChannelV2Error("release channel retention is too short")
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    key = release_channel_key(normalized["commit_sha"], prefix=prefix)
    payload = (canonical_json(normalized) + "\n").encode("ascii")
    try:
        existing = s3_client.get_object(Bucket=str(bucket), Key=key)["Body"].read()
    except Exception:
        existing = None
    if existing is not None:
        if existing == payload:
            observed = normalized
        else:
            try:
                value = json.loads(existing)
                observed = validate_release_channel_v2(
                    value, expected_commit=normalized["commit_sha"]
                )
            except Exception as exc:
                raise ReleaseChannelV2Error(
                    "immutable release channel already contains different bytes"
                ) from exc
            if release_channel_role_identities_v2(
                observed
            ) != release_channel_role_identities_v2(normalized):
                raise ReleaseChannelV2Error(
                    "immutable release channel already contains different roles"
                )
        return {"bucket": str(bucket), "key": key, **observed}
    retain_until = datetime.now(timezone.utc) + timedelta(days=int(retention_days))
    try:
        s3_client.put_object(
            Bucket=str(bucket),
            Key=key,
            Body=payload,
            ContentType="application/json",
            ObjectLockMode="COMPLIANCE",
            ObjectLockRetainUntilDate=retain_until,
            IfNoneMatch="*",
        )
        observed = s3_client.get_object(Bucket=str(bucket), Key=key)["Body"].read()
    except Exception as exc:
        raise ReleaseChannelV2Error("release channel publication failed") from exc
    if observed != payload:
        raise ReleaseChannelV2Error("published release channel readback differs")
    return {"bucket": str(bucket), "key": key, **normalized}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", type=Path)
    mode.add_argument("--publish", type=Path)
    mode.add_argument("--ensure", action="store_true")
    parser.add_argument("--gateway-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--gateway-output", type=Path)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    args = parser.parse_args(argv)

    if args.build:
        if not args.gateway_manifest or not args.output:
            raise ReleaseChannelV2Error("channel build inputs are incomplete")
        result = build_release_channel_v2(
            gateway_release_manifest=_load_json(
                args.gateway_manifest, "gateway release manifest"
            ),
        )
        _atomic_json(args.output, result)
    elif args.verify:
        result = validate_release_channel_v2(
            _load_json(args.verify, "release channel"),
            expected_commit=args.expected_commit,
        )
    elif args.publish:
        result = publish_release_channel_v2(
            _load_json(args.publish, "release channel"),
            bucket=args.bucket,
            prefix=args.prefix,
            retention_days=args.retention_days,
        )
    else:
        commit = str(args.expected_commit or "").lower()
        if not _COMMIT_RE.fullmatch(commit):
            raise ReleaseChannelV2Error("--ensure requires an exact commit")
        if local_release_inputs_match(
            expected_commit=commit,
            gateway_output=args.gateway_output,
        ):
            result = {"status": "local_verified", "commit_sha": commit}
        else:
            result = install_release_channel_v2(
                fetch_release_channel_v2(
                    bucket=args.bucket, commit_sha=commit, prefix=args.prefix
                ),
                expected_commit=commit,
                gateway_output=args.gateway_output,
            )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def cli(argv: Optional[Sequence[str]] = None) -> int:
    """Run the operator CLI without exposing expected retry tracebacks."""

    try:
        return main(argv)
    except ReleaseChannelV2Error as exc:
        print(f"Release channel unavailable: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(cli())
