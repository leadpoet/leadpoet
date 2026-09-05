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
import subprocess
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.tee.release_manifest_v2 import (
    HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    LOCAL_RELEASE_SCHEMA_VERSION,
    validate_historical_release_manifest,
    validate_prior_release_manifest,
    validate_release_manifest,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from validator_tee.host.release_v2 import (
    VALIDATOR_LOCAL_RELEASE_SCHEMA_VERSION,
    validate_validator_release_manifest,
)


SCHEMA_VERSION = "leadpoet.attested_release_channel.v2"
LINEAGE_SCHEMA_VERSION = "leadpoet.attested_release_lineage.v1"
DEFAULT_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
DEFAULT_PREFIX = "attested-v2/releases"
DEFAULT_RETENTION_DAYS = 365
MAX_LINEAGE_RELEASES = 512
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_LOCAL_COMMIT_ENV = "LEADPOET_LOCAL_RELEASE_COMMIT_SHA"
_LOCAL_GATEWAY_ENV = "LEADPOET_LOCAL_GATEWAY_RELEASE"
_LOCAL_VALIDATOR_ENV = "LEADPOET_LOCAL_VALIDATOR_RELEASE"
_LOCAL_PRIOR_LINEAGE_ENV = "LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE"


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


def _build_release_channel_v2(
    *,
    gateway_release_manifest: Mapping[str, Any],
    validator_release_manifest: Mapping[str, Any],
    gateway_validator: Any,
) -> Dict[str, Any]:
    gateway = gateway_validator(gateway_release_manifest)
    validator = validate_validator_release_manifest(validator_release_manifest)
    commit = gateway["commit_sha"]
    if validator["release"]["commit_sha"] != commit:
        raise ReleaseChannelV2Error(
            "gateway and validator release commits differ"
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "commit_sha": commit,
        "gateway_release_manifest": gateway,
        "validator_release_manifest": validator,
    }
    gateway_is_local = gateway["schema_version"] == LOCAL_RELEASE_SCHEMA_VERSION
    validator_is_local = (
        validator["schema_version"] == VALIDATOR_LOCAL_RELEASE_SCHEMA_VERSION
    )
    if gateway_is_local != validator_is_local:
        raise ReleaseChannelV2Error(
            "gateway and validator release identity modes differ"
        )
    hash_body = body
    if gateway_is_local:
        hash_body = {
            "schema_version": SCHEMA_VERSION,
            "commit_sha": commit,
            "gateway_release_hash": gateway["release_hash"],
            "validator_release_hash": validator["release_manifest_hash"],
        }
    return {**body, "channel_hash": sha256_json(hash_body)}


def build_release_channel_v2(
    *,
    gateway_release_manifest: Mapping[str, Any],
    validator_release_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a release channel for the canonical current topology."""

    return _build_release_channel_v2(
        gateway_release_manifest=gateway_release_manifest,
        validator_release_manifest=validator_release_manifest,
        gateway_validator=validate_release_manifest,
    )


def build_historical_release_channel_v2(
    *,
    gateway_release_manifest: Mapping[str, Any],
    validator_release_manifest: Mapping[str, Any],
    expected_topology_hash: str,
) -> Dict[str, Any]:
    """Build a channel for the explicitly selected known old topology."""

    return _build_release_channel_v2(
        gateway_release_manifest=gateway_release_manifest,
        validator_release_manifest=validator_release_manifest,
        gateway_validator=lambda value: validate_historical_release_manifest(
            value,
            expected_topology_hash=expected_topology_hash,
        ),
    )


def _validate_release_channel_v2(
    value: Mapping[str, Any],
    *,
    expected_commit: Optional[str],
    gateway_validator: Any,
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "commit_sha",
        "gateway_release_manifest",
        "validator_release_manifest",
        "channel_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReleaseChannelV2Error("release channel fields are invalid")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseChannelV2Error("release channel schema is invalid")
    normalized = _build_release_channel_v2(
        gateway_release_manifest=value["gateway_release_manifest"],
        validator_release_manifest=value["validator_release_manifest"],
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


def validate_historical_release_channel_v2(
    value: Mapping[str, Any],
    *,
    expected_topology_hash: str = HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    expected_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a channel for the explicitly selected known old topology."""

    return _validate_release_channel_v2(
        value,
        expected_commit=expected_commit,
        gateway_validator=lambda manifest: validate_historical_release_manifest(
            manifest,
            expected_topology_hash=expected_topology_hash,
        ),
    )


def validate_prior_release_channel_v2(
    value: Mapping[str, Any], *, expected_commit: Optional[str] = None
) -> Dict[str, Any]:
    """Validate a prior channel from the current or one known old topology."""

    return _validate_release_channel_v2(
        value,
        expected_commit=expected_commit,
        gateway_validator=validate_prior_release_manifest,
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
    validator_output: Optional[Path] = None,
) -> Dict[str, Any]:
    normalized = validate_release_channel_v2(
        channel, expected_commit=expected_commit
    )
    if gateway_output is not None:
        _atomic_json(gateway_output, normalized["gateway_release_manifest"])
    if validator_output is not None:
        _atomic_json(validator_output, normalized["validator_release_manifest"])
    return normalized


def local_release_inputs_match(
    *,
    expected_commit: str,
    gateway_output: Optional[Path],
    validator_output: Optional[Path],
) -> bool:
    try:
        if gateway_output is not None:
            gateway = validate_release_manifest(
                _load_json(gateway_output, "local gateway release manifest")
            )
            if gateway["commit_sha"] != expected_commit:
                return False
        if validator_output is not None:
            validator = validate_validator_release_manifest(
                _load_json(validator_output, "local validator release manifest")
            )
            if validator["release"]["commit_sha"] != expected_commit:
                return False
    except Exception:
        return False
    return gateway_output is not None or validator_output is not None


def fetch_release_channel_v2(
    *,
    bucket: str,
    commit_sha: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client: Any = None,
) -> Dict[str, Any]:
    local_values = (
        os.environ.get(_LOCAL_COMMIT_ENV),
        os.environ.get(_LOCAL_GATEWAY_ENV),
        os.environ.get(_LOCAL_VALIDATOR_ENV),
    )
    if any(local_values):
        if not all(local_values):
            raise ReleaseChannelV2Error(
                "local release identity environment is incomplete"
            )
        local_commit, gateway_path, validator_path = local_values
        if str(local_commit).lower() == str(commit_sha).lower():
            gateway = _load_json(Path(str(gateway_path)), "local gateway release")
            validator = _load_json(
                Path(str(validator_path)), "local validator release"
            )
            return validate_release_channel_v2(
                build_release_channel_v2(
                    gateway_release_manifest=gateway,
                    validator_release_manifest=validator,
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


def fetch_prior_release_channel_v2(
    *,
    bucket: str,
    commit_sha: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client: Any = None,
) -> Dict[str, Any]:
    """Fetch one immutable prior channel for bounded lineage ingestion."""

    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    try:
        response = s3_client.get_object(
            Bucket=str(bucket), Key=release_channel_key(commit_sha, prefix=prefix)
        )
        value = json.loads(response["Body"].read())
    except Exception as exc:
        raise ReleaseChannelV2Error(
            "approved prior release channel is unavailable"
        ) from exc
    return validate_prior_release_channel_v2(
        value,
        expected_commit=commit_sha,
    )


def fetch_historical_release_channel_v2(
    *,
    bucket: str,
    commit_sha: str,
    expected_topology_hash: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client: Any = None,
) -> Dict[str, Any]:
    """Fetch one channel through the explicit known historical validator."""

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
            "approved historical release channel is unavailable"
        ) from exc
    return validate_historical_release_channel_v2(
        value,
        expected_topology_hash=expected_topology_hash,
        expected_commit=commit_sha,
    )


def _build_release_lineage_v2(
    channels: Sequence[Mapping[str, Any]],
    *,
    current_commit: str,
    prior_commits: Sequence[str] = (),
    channel_validator: Any = None,
) -> Dict[str, Any]:
    commit = str(current_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ReleaseChannelV2Error("release lineage current commit is invalid")
    if not channels or len(channels) > MAX_LINEAGE_RELEASES:
        raise ReleaseChannelV2Error("release lineage size is invalid")
    prior = {str(item or "").lower() for item in prior_commits}
    if any(not _COMMIT_RE.fullmatch(item) for item in prior):
        raise ReleaseChannelV2Error("prior release lineage commit is invalid")
    releases: Dict[str, Any] = {}
    for value in channels:
        if channel_validator is None:
            claimed_commit = str(value.get("commit_sha") or "").lower()
            channel = (
                validate_prior_release_channel_v2(value)
                if claimed_commit in prior
                else validate_release_channel_v2(value)
            )
        else:
            channel = channel_validator(value)
        channel_commit = channel["commit_sha"]
        if channel_commit in releases:
            raise ReleaseChannelV2Error("release lineage commit is duplicated")
        gateway = channel["gateway_release_manifest"]
        roles = {}
        for role, summary in sorted(gateway["roles"].items()):
            roles[role] = {
                "commit_sha": summary["commit_sha"],
                "pcr0": summary["pcr0"],
                "build_manifest_hash": summary["execution_manifest_hash"],
                "dependency_lock_hash": summary["dependency_lock_hash"],
            }
        validator = channel["validator_release_manifest"]["release"]
        roles["validator_weights"] = {
            "commit_sha": validator["commit_sha"],
            "pcr0": validator["pcr0"],
            "build_manifest_hash": validator["app_manifest_hash"],
            "dependency_lock_hash": validator["dependency_lock_hash"],
        }
        releases[channel_commit] = {
            "channel_hash": channel["channel_hash"],
            "gateway_release_hash": gateway["release_hash"],
            "roles": roles,
        }
    current = releases.get(commit)
    if current is None:
        raise ReleaseChannelV2Error(
            "current release is absent from approved release lineage"
        )
    body = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "current_commit_sha": commit,
        "current_gateway_release_hash": current["gateway_release_hash"],
        "releases": {
            release_commit: releases[release_commit]
            for release_commit in sorted(releases)
        },
    }
    return {**body, "lineage_hash": sha256_json(body)}


def build_release_lineage_v2(
    channels: Sequence[Mapping[str, Any]],
    *,
    current_commit: str,
) -> Dict[str, Any]:
    """Compact exact current-topology channels for immutable configuration."""

    return _build_release_lineage_v2(
        channels,
        current_commit=current_commit,
        prior_commits=(),
        channel_validator=validate_release_channel_v2,
    )


def build_historical_release_lineage_v2(
    channels: Sequence[Mapping[str, Any]],
    *,
    current_commit: str,
    expected_topology_hash: str,
) -> Dict[str, Any]:
    """Compact only exact channels from one selected historical topology."""

    lineage = _build_release_lineage_v2(
        channels,
        current_commit=current_commit,
        channel_validator=lambda value: validate_historical_release_channel_v2(
            value,
            expected_topology_hash=expected_topology_hash,
        ),
    )
    from gateway.tee.release_lineage_v2 import (
        validate_historical_compact_release_lineage_v2,
    )

    return validate_historical_compact_release_lineage_v2(
        lineage,
        expected_topology_hash=expected_topology_hash,
        expected_current_commit=current_commit,
    )


def _local_release_lineage_entries(
    *,
    current_commit: str,
    bucket: str,
    prefix: str,
    s3_client: Any,
) -> Optional[Dict[str, Any]]:
    local_commit = str(os.environ.get(_LOCAL_COMMIT_ENV) or "").lower()
    if local_commit != current_commit:
        return None
    current_channel = fetch_release_channel_v2(
        bucket=bucket,
        commit_sha=current_commit,
        prefix=prefix,
        s3_client=s3_client,
    )
    current_lineage = build_release_lineage_v2(
        [current_channel],
        current_commit=current_commit,
    )
    releases = dict(current_lineage["releases"])
    prior_path = str(os.environ.get(_LOCAL_PRIOR_LINEAGE_ENV) or "").strip()
    if not prior_path:
        return releases
    from gateway.tee.release_lineage_v2 import (
        validate_prior_compact_release_lineage_v2,
    )

    prior = validate_prior_compact_release_lineage_v2(
        _load_json(Path(prior_path), "installed prior release lineage")
    )
    for commit, release in prior["releases"].items():
        if commit in releases and releases[commit] != release:
            raise ReleaseChannelV2Error(
                "local and installed release identities conflict"
            )
        releases[commit] = release
    return releases


def _compact_release_lineage_from_entries(
    releases: Mapping[str, Any],
    *,
    current_commit: str,
) -> Dict[str, Any]:
    current = releases.get(current_commit)
    if not isinstance(current, Mapping):
        raise ReleaseChannelV2Error(
            "current release is absent from local release lineage"
        )
    body = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "current_commit_sha": current_commit,
        "current_gateway_release_hash": current["gateway_release_hash"],
        "releases": {
            commit: releases[commit] for commit in sorted(releases)
        },
    }
    from gateway.tee.release_lineage_v2 import (
        validate_compact_release_lineage_v2,
    )

    return validate_compact_release_lineage_v2(
        {**body, "lineage_hash": sha256_json(body)},
        expected_current_commit=current_commit,
    )


def fetch_release_lineage_v2(
    *,
    bucket: str,
    current_commit: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client: Any = None,
    allowed_commits: Optional[Sequence[str]] = None,
    required_commits: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fetch and validate the selected immutable V2 release channels.

    An explicit ``required_commits`` set is the bounded production path. It
    reads exactly those content-addressed channel objects and does not enumerate
    the lifetime release catalog. The prefix scan remains for legacy callers
    that do not yet provide an explicit requirement set.
    """

    normalized_prefix = str(prefix or "").strip("/")
    if not normalized_prefix or ".." in normalized_prefix.split("/"):
        raise ReleaseChannelV2Error("release channel prefix is invalid")
    if required_commits is not None:
        if isinstance(required_commits, (str, bytes)):
            raise ReleaseChannelV2Error(
                "required release lineage commits are invalid"
            )
        required = tuple(required_commits)
        if not required or len(required) > MAX_LINEAGE_RELEASES:
            raise ReleaseChannelV2Error(
                "required release lineage size is invalid"
            )
        if any(
            not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit)
            for commit in required
        ):
            raise ReleaseChannelV2Error(
                "required release lineage commits are invalid"
            )
        if len(required) != len(set(required)):
            raise ReleaseChannelV2Error(
                "required release lineage commit is duplicated"
            )
        current = str(current_commit or "").lower()
        if not _COMMIT_RE.fullmatch(current) or current not in required:
            raise ReleaseChannelV2Error(
                "current release is absent from required release lineage"
            )
        if (
            allowed_commits is None
            or isinstance(allowed_commits, (str, bytes))
        ):
            raise ReleaseChannelV2Error(
                "required release lineage Git ancestry is unavailable"
            )
        allowed_values = tuple(allowed_commits)
        allowed = {str(commit or "").lower() for commit in allowed_values}
        if (
            not allowed_values
            or not allowed
            or any(
                not isinstance(commit, str)
                or commit != commit.lower()
                or not _COMMIT_RE.fullmatch(commit)
                for commit in allowed_values
            )
            or any(commit not in allowed for commit in required)
        ):
            raise ReleaseChannelV2Error(
                "required release lineage Git ancestry is invalid"
            )
        releases = _local_release_lineage_entries(
            current_commit=current,
            bucket=bucket,
            prefix=normalized_prefix,
            s3_client=s3_client,
        ) or {}
        missing = sorted(set(required) - set(releases))
        if missing:
            channels = [
                (
                    fetch_release_channel_v2(
                        bucket=bucket,
                        commit_sha=commit,
                        prefix=normalized_prefix,
                        s3_client=s3_client,
                    )
                    if commit == current
                    else fetch_prior_release_channel_v2(
                        bucket=bucket,
                        commit_sha=commit,
                        prefix=normalized_prefix,
                        s3_client=s3_client,
                    )
                )
                for commit in missing
            ]
            fetched = _build_release_lineage_v2(
                channels,
                current_commit=(current if current in missing else missing[0]),
                prior_commits=tuple(
                    commit for commit in missing if commit != current
                ),
            )
            for commit, release in fetched["releases"].items():
                if commit in releases and releases[commit] != release:
                    raise ReleaseChannelV2Error(
                        "local and fetched release identities conflict"
                    )
                releases[commit] = release
        selected = {commit: releases[commit] for commit in required}
        return _compact_release_lineage_from_entries(
            selected,
            current_commit=current,
        )
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    key_pattern = re.compile(
        rf"^{re.escape(normalized_prefix)}/([0-9a-f]{{40}})/"
        r"release-channel-v2\.json$"
    )
    allowed = None
    if allowed_commits is not None:
        allowed = {str(commit or "").lower() for commit in allowed_commits}
        if (
            not allowed
            or any(not _COMMIT_RE.fullmatch(commit) for commit in allowed)
            or str(current_commit).lower() not in allowed
        ):
            raise ReleaseChannelV2Error(
                "approved release lineage Git ancestry is invalid"
            )
    commits = []
    continuation_token = None
    try:
        while True:
            request = {
                "Bucket": str(bucket),
                "Prefix": normalized_prefix + "/",
                "MaxKeys": 1000,
            }
            if continuation_token is not None:
                request["ContinuationToken"] = continuation_token
            response = s3_client.list_objects_v2(**request)
            for item in response.get("Contents") or ():
                match = key_pattern.fullmatch(str(item.get("Key") or ""))
                if match and (
                    allowed is None or match.group(1) in allowed
                ):
                    commits.append(match.group(1))
                    if len(commits) > MAX_LINEAGE_RELEASES:
                        raise ReleaseChannelV2Error(
                            "approved release lineage is too large"
                        )
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                raise ReleaseChannelV2Error(
                    "approved release lineage pagination is invalid"
                )
    except ReleaseChannelV2Error:
        raise
    except Exception as exc:
        raise ReleaseChannelV2Error(
            "approved release lineage is unavailable"
        ) from exc
    if len(commits) != len(set(commits)):
        raise ReleaseChannelV2Error("approved release lineage is duplicated")
    selected_current = str(current_commit or "").lower()
    channels = [
        (
            fetch_release_channel_v2(
                bucket=bucket,
                commit_sha=commit,
                prefix=normalized_prefix,
                s3_client=s3_client,
            )
            if commit == selected_current
            else fetch_prior_release_channel_v2(
                bucket=bucket,
                commit_sha=commit,
                prefix=normalized_prefix,
                s3_client=s3_client,
            )
        )
        for commit in sorted(commits)
    ]
    return _build_release_lineage_v2(
        channels,
        current_commit=selected_current,
        prior_commits=tuple(
            commit for commit in commits if commit != selected_current
        ),
    )


def fetch_historical_release_lineage_v2(
    *,
    bucket: str,
    current_commit: str,
    expected_topology_hash: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client: Any = None,
    allowed_commits: Sequence[str],
    required_commits: Sequence[str],
) -> Dict[str, Any]:
    """Fetch a bounded all-old-topology lineage for an exact rollback."""

    normalized_prefix = str(prefix or "").strip("/")
    if not normalized_prefix or ".." in normalized_prefix.split("/"):
        raise ReleaseChannelV2Error("release channel prefix is invalid")
    if isinstance(required_commits, (str, bytes)) or isinstance(
        allowed_commits, (str, bytes)
    ):
        raise ReleaseChannelV2Error(
            "historical release lineage commits are invalid"
        )
    required = tuple(required_commits)
    allowed_values = tuple(allowed_commits)
    current = str(current_commit or "").lower()
    if (
        not required
        or len(required) > MAX_LINEAGE_RELEASES
        or len(required) != len(set(required))
        or any(
            not isinstance(commit, str)
            or commit != commit.lower()
            or not _COMMIT_RE.fullmatch(commit)
            for commit in required
        )
        or not _COMMIT_RE.fullmatch(current)
        or current not in required
    ):
        raise ReleaseChannelV2Error(
            "historical required release lineage is invalid"
        )
    allowed = {str(commit or "").lower() for commit in allowed_values}
    if (
        not allowed_values
        or not allowed
        or any(
            not isinstance(commit, str)
            or commit != commit.lower()
            or not _COMMIT_RE.fullmatch(commit)
            for commit in allowed_values
        )
        or any(commit not in allowed for commit in required)
    ):
        raise ReleaseChannelV2Error(
            "historical release lineage Git ancestry is invalid"
        )
    channels = [
        fetch_historical_release_channel_v2(
            bucket=bucket,
            commit_sha=commit,
            expected_topology_hash=expected_topology_hash,
            prefix=normalized_prefix,
            s3_client=s3_client,
        )
        for commit in required
    ]
    return build_historical_release_lineage_v2(
        channels,
        current_commit=current,
        expected_topology_hash=expected_topology_hash,
    )


def git_ancestor_commits_v2(
    *, repository: Path, current_commit: str
) -> Sequence[str]:
    commit = str(current_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ReleaseChannelV2Error("release lineage current commit is invalid")
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repository)), "rev-list", commit],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseChannelV2Error(
            "release lineage Git ancestry is unavailable"
        ) from exc
    commits = tuple(line.strip().lower() for line in result.stdout.splitlines())
    if (
        not commits
        or commits[0] != commit
        or any(not _COMMIT_RE.fullmatch(item) for item in commits)
    ):
        raise ReleaseChannelV2Error(
            "release lineage Git ancestry is invalid"
        )
    return commits


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
        if existing != payload:
            raise ReleaseChannelV2Error(
                "immutable release channel already contains different bytes"
            )
        return {"bucket": str(bucket), "key": key, **normalized}
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
    parser.add_argument("--validator-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--gateway-output", type=Path)
    parser.add_argument("--validator-output", type=Path)
    parser.add_argument("--lineage-output", type=Path)
    parser.add_argument("--lineage-repository", type=Path)
    parser.add_argument("--lineage-authority-commit")
    parser.add_argument("--lineage-required-commit", action="append")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    args = parser.parse_args(argv)

    if args.build:
        if not args.gateway_manifest or not args.validator_manifest or not args.output:
            raise ReleaseChannelV2Error("channel build inputs are incomplete")
        result = build_release_channel_v2(
            gateway_release_manifest=_load_json(
                args.gateway_manifest, "gateway release manifest"
            ),
            validator_release_manifest=_load_json(
                args.validator_manifest, "validator release manifest"
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
            validator_output=args.validator_output,
        ):
            result = {"status": "local_verified", "commit_sha": commit}
        else:
            result = install_release_channel_v2(
                fetch_release_channel_v2(
                    bucket=args.bucket, commit_sha=commit, prefix=args.prefix
                ),
                expected_commit=commit,
                gateway_output=args.gateway_output,
                validator_output=args.validator_output,
            )
        if args.lineage_output is not None:
            if args.lineage_repository is None:
                raise ReleaseChannelV2Error(
                    "--lineage-output requires --lineage-repository"
                )
            authority_commit = str(
                args.lineage_authority_commit or commit
            ).lower()
            if not _COMMIT_RE.fullmatch(authority_commit):
                raise ReleaseChannelV2Error(
                    "release lineage authority commit is invalid"
                )
            allowed_commits = git_ancestor_commits_v2(
                repository=args.lineage_repository,
                current_commit=authority_commit,
            )
            if commit not in allowed_commits:
                raise ReleaseChannelV2Error(
                    "selected release is absent from main release lineage"
                )
            lineage = fetch_release_lineage_v2(
                bucket=args.bucket,
                current_commit=commit,
                prefix=args.prefix,
                allowed_commits=allowed_commits,
                required_commits=args.lineage_required_commit,
            )
            _atomic_json(args.lineage_output, lineage)
            result = {
                **result,
                "lineage_hash": lineage["lineage_hash"],
                "lineage_release_count": len(lineage["releases"]),
            }
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
