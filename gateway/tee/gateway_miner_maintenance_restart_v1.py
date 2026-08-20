"""Bind a pre-hydration miner-maintenance change to one locked restart.

The first release containing the fixed-purpose miner-submission secret helper
cannot run that helper from the deployed N-1 checkout.  This module is instead
executed from a fully verified archive of the exact attested candidate while
the canonical gateway restart lock is held.  It updates only the existing
fixed-purpose setting and carries the resulting non-secret commitments through
that invocation in a sealed, unlinked memory file.  No cross-invocation local
receipt or restart authority is persisted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Optional, Sequence
import uuid

from gateway.tee.disable_gateway_miner_submissions_secret import (
    DEFAULT_RECOVERY_JOURNAL_PATH,
    EXPECTED_AWS_REGION,
    GATEWAY_SECRET_ID,
    TARGET_ENV_NAME,
    TARGET_ENV_VALUE,
    GatewayMinerSubmissionsDisableError,
    _FORBIDDEN_AWS_ENV_NAMES,
    _apply_gateway_miner_submissions_secret,
    _instance_role_secrets_client,
    _instance_role_aws_clients,
    _open_recovery_journal_parent_fd,
    _recover_orphan_transaction,
    _verify_protected_source,
    disable_gateway_miner_submissions_secret,
)
from gateway.tee.release_channel_v2 import (
    release_channel_key,
    validate_release_channel_v2,
)
from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.topology import ROLE_SPECS
from leadpoet_canonical.attested_v2 import sha256_json
from scripts.gateway_git_deploy import (
    DEFAULT_BRANCH,
    DEFAULT_REPO_URL,
    SCHEMA_VERSION as GIT_DEPLOYMENT_SCHEMA_VERSION,
    verify_materialized_tree,
)


SCHEMA_VERSION = "leadpoet.gateway_miner_maintenance_restart.v1"
DEFAULT_RELEASE_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
DEFAULT_RELEASE_PREFIX = "attested-v2/releases"
CANONICAL_GATEWAY_RESTART_LOCK_PATH = Path(
    "/home/ec2-user/.config/leadpoet/gateway-restart.lock"
)
CANONICAL_GATEWAY_ENV_PATH = Path(
    "/home/ec2-user/.config/leadpoet/gateway.env"
)
PROOF_FD_ENV_NAME = "GATEWAY_MINER_MAINTENANCE_PROOF_FD"
PROOF_FD_NUMBER = 190
CONTROLLER_WRAPPER_FD_NUMBER = 191
CONTROLLER_GIT_HELPER_FD_NUMBER = 192
CONTROLLER_EXACT_COMMIT_HELPER_FD_NUMBER = 193
CONTROLLER_MEMORY_GUARD_FD_NUMBER = 194
MAX_PROOF_BYTES = 32 * 1024
MAX_RELEASE_CHANNEL_BYTES = 4 * 1024 * 1024
MAX_RUNTIME_STATUS_BYTES = 256 * 1024
DEFAULT_RUNTIME_STATUS_URL = "http://127.0.0.1:8000/research-lab/status"
# These are minimum compatible ancestry floors, not an exhaustive release list.
SUPPORTED_N_MINUS_ONE_CONTROLLER_COMMITS = frozenset(
    {"0dd3a385a23a3af0fa17210bfe02a39cc4023952"}
)
LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT = (
    "0dd3a385a23a3af0fa17210bfe02a39cc4023952"
)
LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER = (
    "/home/ec2-user/.config/leadpoet/restart-controller/gateway/current/"
    "scripts/gateway_git_deploy.py"
)
RUNTIME_BUILD_IDENTITY_NAMES = (
    "ATTESTED_RUNTIME_COMMIT_SHA",
    "GITHUB_SHA",
    "GITHUB_COMMIT",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_TREE_RE = re.compile(r"^[0-9a-f]{40,64}$")
_VERSION_ID_RE = re.compile(r"^[A-Za-z0-9-]{32,64}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UNSAFE_GIT_ENV_NAMES = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
    }
)
_RESTART_AUTHORITY_NAMES = frozenset(
    {
        PROOF_FD_ENV_NAME,
        "GATEWAY_GIT_HELPER",
        "GATEWAY_EXACT_COMMIT_HELPER",
        "GATEWAY_HOST_MEMORY_GUARD_PATH",
    }
)
_PROOF_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_commit",
        "candidate_tree_hash",
        "candidate_blob_manifest_sha256",
        "pre_hydration_runtime_commit",
        "n_minus_one_controller_commit",
        "release_channel_hash",
        "release_channel_object_version_id",
        "release_channel_object_sha256",
        "release_channel_retain_until",
        "gateway_release_hash",
        "current_secret_version_id",
        "current_document_commitment",
        "current_hydrated_environment_commitment",
        "current_stage_topology_commitment",
        "controller_wrapper_sha256",
        "controller_git_helper_sha256",
        "controller_exact_commit_helper_sha256",
        "controller_memory_guard_sha256",
        "pre_hydration_live_process_commitment",
        "restart_invocation_id",
        "prepared_at",
        "proof_hash",
    }
)


class GatewayMinerMaintenanceRestartError(RuntimeError):
    """The fixed maintenance state cannot be bound safely to this restart."""


def _candidate_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_fixed_bootstrap_authority(
    environment: Mapping[str, str],
) -> None:
    if (
        str(environment.get("LEADPOET_GATEWAY_ENV_SECRET_ID") or GATEWAY_SECRET_ID)
        != GATEWAY_SECRET_ID
        or str(
            environment.get("GATEWAY_V2_RELEASE_BUCKET")
            or DEFAULT_RELEASE_BUCKET
        )
        != DEFAULT_RELEASE_BUCKET
        or str(
            environment.get("RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET")
            or DEFAULT_RELEASE_BUCKET
        )
        != DEFAULT_RELEASE_BUCKET
        or str(
            environment.get("GATEWAY_V2_RELEASE_PREFIX")
            or DEFAULT_RELEASE_PREFIX
        )
        != DEFAULT_RELEASE_PREFIX
        or str(
            environment.get("GATEWAY_ENV_FILE")
            or CANONICAL_GATEWAY_ENV_PATH
        )
        != str(CANONICAL_GATEWAY_ENV_PATH)
        or any(
            str(environment.get(name) or EXPECTED_AWS_REGION)
            != EXPECTED_AWS_REGION
            for name in ("AWS_REGION", "AWS_DEFAULT_REGION")
        )
        or str(
            environment.get("LEADPOET_AWS_INSTANCE_ROLE_ONLY") or "true"
        ).lower()
        != "true"
        or any(str(environment.get(name) or "") for name in _FORBIDDEN_AWS_ENV_NAMES)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance bootstrap authority differs from production"
        )


def _resolve_bootstrap_aws_clients(
    *,
    secrets_client: Any,
    release_s3_client: Any,
) -> tuple[Any, Any]:
    if secrets_client is None and release_s3_client is None:
        clients = _instance_role_aws_clients(
            environ={**os.environ, "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true"}
        )
        return clients["secretsmanager"], clients["s3"]
    if secrets_client is None or release_s3_client is None:
        raise GatewayMinerMaintenanceRestartError(
            "bootstrap AWS client authority is incomplete"
        )
    return secrets_client, release_s3_client


def _require_canonical_restart_lock_fd() -> None:
    configured = Path(
        os.environ.get("GATEWAY_RESTART_LOCK_FILE")
        or CANONICAL_GATEWAY_RESTART_LOCK_PATH
    )
    if configured != CANONICAL_GATEWAY_RESTART_LOCK_PATH:
        raise GatewayMinerMaintenanceRestartError(
            "canonical gateway restart lock path differs"
        )
    try:
        target = os.readlink("/proc/self/fd/9")
        metadata = os.fstat(9)
        if (
            target != str(CANONICAL_GATEWAY_RESTART_LOCK_PATH)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise GatewayMinerMaintenanceRestartError(
                "canonical gateway restart lock descriptor is unsafe"
            )
        fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.set_inheritable(9, True)
    except GatewayMinerMaintenanceRestartError:
        raise
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "canonical gateway restart lock is not held"
        ) from exc


def _read_bounded_proc_file(path: Path, *, max_bytes: int) -> bytes:
    descriptor: Optional[int] = None
    try:
        before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise GatewayMinerMaintenanceRestartError(
                "running gateway process identity is unsafe"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(65536, max_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > max_bytes:
                raise GatewayMinerMaintenanceRestartError(
                    "running gateway process metadata is too large"
                )
        final = os.fstat(descriptor)
        current = path.lstat()
        if (
            final.st_dev != opened.st_dev
            or final.st_ino != opened.st_ino
            or current.st_dev != opened.st_dev
            or current.st_ino != opened.st_ino
        ):
            raise GatewayMinerMaintenanceRestartError(
                "running gateway process changed while reading"
            )
        return b"".join(chunks)
    except GatewayMinerMaintenanceRestartError:
        raise
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "running gateway process metadata is unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_restart_authority_absent_from_environment_payload(
    environment_payload: bytes,
    *,
    expected_runtime_commit: Optional[str] = None,
    verified_controller_commit: Optional[str] = None,
    allow_legacy_n_minus_one_git_helper: bool = False,
) -> dict[str, Any]:
    environment_names: set[str] = set()
    environment_values: dict[str, str] = {}
    for record in environment_payload.split(b"\0"):
        if not record:
            continue
        raw_name, separator, raw_value = record.partition(b"=")
        if not separator:
            raise GatewayMinerMaintenanceRestartError(
                "running gateway process environment is malformed"
            )
        try:
            name = raw_name.decode("ascii")
        except UnicodeError as exc:
            raise GatewayMinerMaintenanceRestartError(
                "running gateway process environment is malformed"
            ) from exc
        if name in environment_names:
            raise GatewayMinerMaintenanceRestartError(
                "running gateway process environment has duplicate names"
            )
        environment_names.add(name)
        try:
            environment_values[name] = raw_value.decode("utf-8")
        except UnicodeError as exc:
            raise GatewayMinerMaintenanceRestartError(
                "running gateway process environment is malformed"
            ) from exc
    restart_authority_names = _RESTART_AUTHORITY_NAMES & environment_names
    legacy_git_helper_allowed = (
        allow_legacy_n_minus_one_git_helper
        and expected_runtime_commit == LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
        and verified_controller_commit == LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
        and restart_authority_names == {"GATEWAY_GIT_HELPER"}
        and environment_values.get("GATEWAY_GIT_HELPER")
        == LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER
    )
    if restart_authority_names and not legacy_git_helper_allowed:
        raise GatewayMinerMaintenanceRestartError(
            "running gateway process contains restart-only authority"
        )
    if _FORBIDDEN_AWS_ENV_NAMES & environment_names:
        raise GatewayMinerMaintenanceRestartError(
            "running gateway process contains delegated AWS authority"
        )
    if any(
        name in environment_values
        and environment_values[name] != EXPECTED_AWS_REGION
        for name in ("AWS_REGION", "AWS_DEFAULT_REGION")
    ) or (
        "LEADPOET_AWS_INSTANCE_ROLE_ONLY" in environment_values
        and environment_values["LEADPOET_AWS_INSTANCE_ROLE_ONLY"].lower()
        != "true"
    ):
        raise GatewayMinerMaintenanceRestartError(
            "running gateway process AWS authority differs"
        )
    runtime_build_identities = {
        name: environment_values.get(name)
        for name in RUNTIME_BUILD_IDENTITY_NAMES
    }
    if expected_runtime_commit is not None and (
        not _COMMIT_RE.fullmatch(expected_runtime_commit)
        or any(
            runtime_build_identities[name] != expected_runtime_commit
            for name in RUNTIME_BUILD_IDENTITY_NAMES
        )
    ):
        raise GatewayMinerMaintenanceRestartError(
            "running gateway process build identity differs"
        )
    return {
        "restart_authority_names": tuple(sorted(restart_authority_names)),
        "runtime_build_identities": runtime_build_identities,
    }


def _live_gateway_restart_authority_commitment(
    *,
    expected_runtime_commit: Optional[str] = None,
    verified_controller_commit: Optional[str] = None,
    allow_legacy_n_minus_one_git_helper: bool = False,
    proc_root: Path = Path("/proc"),
) -> str:
    proc_root = Path(proc_root)
    if not proc_root.is_dir():
        raise GatewayMinerMaintenanceRestartError(
            "running gateway process authority is unavailable"
        )
    candidates: list[tuple[int, tuple[str, ...], str]] = []
    for process_path in proc_root.iterdir():
        if not process_path.name.isdigit():
            continue
        try:
            process_metadata = process_path.stat()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if process_metadata.st_uid != os.geteuid():
            continue
        try:
            cmdline_payload = _read_bounded_proc_file(
                process_path / "cmdline",
                max_bytes=64 * 1024,
            )
        except GatewayMinerMaintenanceRestartError:
            if not process_path.exists():
                continue
            raise
        arguments = tuple(
            value.decode("utf-8", errors="strict")
            for value in cmdline_payload.split(b"\0")
            if value
        )
        is_gateway = (
            "gateway.main" in arguments
            and "-m" in arguments
        ) or any(Path(value).name == "main.py" for value in arguments[1:])
        if not is_gateway:
            continue
        stat_payload = _read_bounded_proc_file(
            process_path / "stat",
            max_bytes=64 * 1024,
        ).decode("ascii", errors="strict")
        _prefix, separator, remainder = stat_payload.rpartition(")")
        fields = remainder.strip().split()
        if separator != ")" or len(fields) <= 19 or not fields[19].isdigit():
            raise GatewayMinerMaintenanceRestartError(
                "running gateway process start identity is invalid"
            )
        candidates.append((int(process_path.name), arguments, fields[19]))
    if len(candidates) > 1:
        raise GatewayMinerMaintenanceRestartError(
            "running gateway process identity is ambiguous"
        )
    if not candidates:
        return sha256_json({"status": "absent"})
    process_id, arguments, start_time = candidates[0]
    environment_payload = _read_bounded_proc_file(
        proc_root / str(process_id) / "environ",
        max_bytes=4 * 1024 * 1024,
    )
    runtime_authority = (
        _require_restart_authority_absent_from_environment_payload(
            environment_payload,
            expected_runtime_commit=expected_runtime_commit,
            verified_controller_commit=verified_controller_commit,
            allow_legacy_n_minus_one_git_helper=(
                allow_legacy_n_minus_one_git_helper
            ),
        )
    )
    restart_authority_names = runtime_authority["restart_authority_names"]
    return sha256_json(
        {
            "status": "running",
            "pid": process_id,
            "start_time": start_time,
            "argv_sha256": "sha256:"
            + hashlib.sha256(b"\0".join(value.encode("utf-8") for value in arguments)).hexdigest(),
            "runtime_commit": expected_runtime_commit,
            "runtime_build_identities": runtime_authority[
                "runtime_build_identities"
            ],
            "restart_authority_names": list(restart_authority_names),
            "legacy_gateway_git_helper": (
                LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER
                if restart_authority_names == ("GATEWAY_GIT_HELPER",)
                else None
            ),
        }
    )


def _pre_hydration_live_process_commitment(
    tree_evidence: Mapping[str, Any],
) -> str:
    previous_sha = str(tree_evidence.get("previous_sha") or "").lower()
    controller = tree_evidence.get("controller_bundle")
    if not isinstance(controller, Mapping):
        raise GatewayMinerMaintenanceRestartError(
            "verified N-1 controller evidence is incomplete"
        )
    controller_commit = str(controller.get("controller_commit") or "").lower()
    if not _COMMIT_RE.fullmatch(previous_sha) or not _COMMIT_RE.fullmatch(
        controller_commit
    ):
        raise GatewayMinerMaintenanceRestartError(
            "pre-hydration runtime identity is invalid"
        )
    return _live_gateway_restart_authority_commitment(
        expected_runtime_commit=previous_sha,
        verified_controller_commit=controller_commit,
        allow_legacy_n_minus_one_git_helper=(
            previous_sha == LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
            and controller_commit == LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
        ),
    )


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_name, value in pairs:
        name = str(raw_name)
        if name in result:
            raise GatewayMinerMaintenanceRestartError(
                "miner-maintenance restart document contains duplicate fields"
            )
        result[name] = value
    return result


def _load_json(path: Path, *, label: str, max_bytes: int = MAX_PROOF_BYTES) -> dict[str, Any]:
    candidate = Path(path)
    try:
        path_stat = candidate.lstat()
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise GatewayMinerMaintenanceRestartError(f"{label} must be a regular file")
    if path_stat.st_size < 2 or path_stat.st_size > int(max_bytes):
        raise GatewayMinerMaintenanceRestartError(f"{label} size is invalid")
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_stat = os.fstat(descriptor)
        if (
            opened_stat.st_dev != path_stat.st_dev
            or opened_stat.st_ino != path_stat.st_ino
            or not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_size != path_stat.st_size
        ):
            raise GatewayMinerMaintenanceRestartError(
                f"{label} changed while opening"
            )
        payload = os.read(descriptor, int(max_bytes) + 1)
        final_stat = os.fstat(descriptor)
        if (
            final_stat.st_dev != opened_stat.st_dev
            or final_stat.st_ino != opened_stat.st_ino
            or final_stat.st_size != opened_stat.st_size
            or len(payload) != opened_stat.st_size
        ):
            raise GatewayMinerMaintenanceRestartError(
                f"{label} changed while reading"
            )
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(f"{label} is invalid") from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GatewayMinerMaintenanceRestartError(f"{label} is invalid") from exc
    if not isinstance(value, Mapping):
        raise GatewayMinerMaintenanceRestartError(f"{label} must be an object")
    return dict(value)


def _load_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = value.decode("utf-8")
        document = json.loads(
            decoded,
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GatewayMinerMaintenanceRestartError(f"{label} is invalid") from exc
    if not isinstance(document, Mapping):
        raise GatewayMinerMaintenanceRestartError(f"{label} must be an object")
    return dict(document)


def _s3_version_id(value: Any) -> str:
    version_id = str(value or "")
    encoded = version_id.encode("utf-8", errors="strict")
    if (
        version_id != version_id.strip()
        or version_id == "null"
        or not 1 <= len(encoded) <= 1024
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in version_id)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel object version is invalid"
        )
    return version_id


def _retention_timestamp(value: Any, *, now: datetime) -> tuple[datetime, str]:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel retention is invalid"
        )
    normalized = value.astimezone(timezone.utc)
    if normalized <= now:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel retention is not active"
        )
    return normalized, normalized.isoformat().replace("+00:00", "Z")


def _object_metadata(
    value: Mapping[str, Any],
    *,
    now: datetime,
    expected_version_id: Optional[str] = None,
) -> dict[str, Any]:
    version_id = _s3_version_id(value.get("VersionId"))
    if expected_version_id is not None and version_id != expected_version_id:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel object version changed"
        )
    if value.get("ObjectLockMode") != "COMPLIANCE":
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel lacks COMPLIANCE retention"
        )
    retain_until, retain_until_text = _retention_timestamp(
        value.get("ObjectLockRetainUntilDate"),
        now=now,
    )
    etag = str(value.get("ETag") or "")
    length = value.get("ContentLength")
    if not etag or len(etag) > 1024:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel object identity is invalid"
        )
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or not 2 <= length <= MAX_RELEASE_CHANNEL_BYTES
    ):
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel object size is invalid"
        )
    return {
        "version_id": version_id,
        "lock_mode": "COMPLIANCE",
        "retain_until": retain_until,
        "retain_until_text": retain_until_text,
        "etag": etag,
        "content_length": length,
    }


def _version_history(
    value: Mapping[str, Any],
    *,
    key: str,
) -> dict[str, Any]:
    if value.get("IsTruncated") is not False:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel object history is incomplete"
        )
    versions = [
        item
        for item in value.get("Versions") or []
        if isinstance(item, Mapping) and item.get("Key") == key
    ]
    delete_markers = [
        item
        for item in value.get("DeleteMarkers") or []
        if isinstance(item, Mapping) and item.get("Key") == key
    ]
    if len(versions) != 1 or delete_markers:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel object history is not a singleton"
        )
    version = dict(versions[0])
    if version.get("IsLatest") is not True:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel singleton is not latest"
        )
    version_id = _s3_version_id(version.get("VersionId"))
    etag = str(version.get("ETag") or "")
    size = version.get("Size")
    if (
        not etag
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 2 <= size <= MAX_RELEASE_CHANNEL_BYTES
    ):
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel history identity is invalid"
        )
    return {
        "version_id": version_id,
        "etag": etag,
        "content_length": size,
    }


def _require_same_object_identity(*values: Mapping[str, Any]) -> None:
    fields = ("version_id", "etag", "content_length")
    retained = [
        value
        for value in values
        if "lock_mode" in value and "retain_until" in value
    ]
    if not values or any(
        value.get(field) != values[0].get(field)
        for value in values[1:]
        for field in fields
    ) or any(
        value.get(field) != retained[0].get(field)
        for value in retained[1:]
        for field in ("lock_mode", "retain_until")
    ):
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel HEAD, GET, and history identities differ"
        )


def _require_six_release_identities(
    channel: Mapping[str, Any], *, expected_commit: str
) -> None:
    gateway_release = channel["gateway_release_manifest"]
    validator_release = channel["validator_release_manifest"]
    identities = [
        channel["commit_sha"],
        gateway_release["commit_sha"],
        *(
            gateway_release["roles"][role]["commit_sha"]
            for role in sorted(ROLE_SPECS)
        ),
        validator_release["release"]["commit_sha"],
    ]
    if len(identities) != 6 or set(identities) != {expected_commit}:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel identities differ from the candidate"
        )


def _fetch_locked_release_channel(
    *,
    commit_sha: str,
    s3_client: Any = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Read one singleton COMPLIANCE-locked channel by exact S3 VersionId."""

    commit = str(commit_sha or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayMinerMaintenanceRestartError("candidate commit is invalid")
    if s3_client is None:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel requires the validated instance-role S3 client"
        )
    observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    key = release_channel_key(commit, prefix=DEFAULT_RELEASE_PREFIX)
    arguments = {"Bucket": DEFAULT_RELEASE_BUCKET, "Key": key}
    try:
        history_before = _version_history(
            s3_client.list_object_versions(
                Bucket=DEFAULT_RELEASE_BUCKET,
                Prefix=key,
                MaxKeys=1000,
            ),
            key=key,
        )
        latest_before = _object_metadata(
            s3_client.head_object(**arguments),
            now=observed_now,
        )
        version_id = latest_before["version_id"]
        pinned_head = _object_metadata(
            s3_client.head_object(**arguments, VersionId=version_id),
            now=observed_now,
            expected_version_id=version_id,
        )
        response = s3_client.get_object(**arguments, VersionId=version_id)
        body = response["Body"]
        try:
            pinned_get = _object_metadata(
                response,
                now=observed_now,
                expected_version_id=version_id,
            )
            payload = body.read(MAX_RELEASE_CHANNEL_BYTES + 1)
        finally:
            body.close()
        if len(payload) != pinned_get["content_length"]:
            raise GatewayMinerMaintenanceRestartError(
                "approved release channel object body length differs"
            )
        latest_after = _object_metadata(
            s3_client.head_object(**arguments),
            now=observed_now,
            expected_version_id=version_id,
        )
        history_after = _version_history(
            s3_client.list_object_versions(
                Bucket=DEFAULT_RELEASE_BUCKET,
                Prefix=key,
                MaxKeys=1000,
            ),
            key=key,
        )
    except GatewayMinerMaintenanceRestartError:
        raise
    except Exception as exc:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel lock evidence is unavailable"
        ) from exc
    _require_same_object_identity(
        history_before,
        latest_before,
        pinned_head,
        pinned_get,
        latest_after,
        history_after,
    )
    channel = validate_release_channel_v2(
        _load_json_bytes(payload, label="approved release channel"),
        expected_commit=commit,
    )
    _require_six_release_identities(channel, expected_commit=commit)
    return {
        "channel": channel,
        "object_version_id": version_id,
        "object_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "object_lock_mode": "COMPLIANCE",
        "object_retain_until": pinned_get["retain_until_text"],
    }


def _safe_git_environment() -> dict[str, str]:
    if any(os.environ.get(name) for name in _UNSAFE_GIT_ENV_NAMES):
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git environment contains object-resolution overrides"
        )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _UNSAFE_GIT_ENV_NAMES
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _run_git(repo_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repo_root)), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git identity is unavailable"
        ) from exc
    if result.returncode != 0:
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git identity is unavailable"
        )
    return result.stdout.strip()


def _run_git_bytes(repo_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repo_root)), *arguments],
            check=False,
            capture_output=True,
            timeout=120,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git object is unavailable"
        ) from exc
    if result.returncode != 0:
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git object is unavailable"
        )
    return result.stdout


def _git_commit_exists(repo_root: Path, commit: str) -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(repo_root)),
                "cat-file",
                "-e",
                f"{commit}^{{commit}}",
            ],
            check=False,
            capture_output=True,
            timeout=120,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git object is unavailable"
        ) from exc
    return result.returncode == 0


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(repo_root)),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
            capture_output=True,
            timeout=120,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git identity is unavailable"
        ) from exc
    if result.returncode not in (0, 1):
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git identity is unavailable"
        )
    return result.returncode == 0


def _canonical_remote(value: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    return normalized[:-4] if normalized.endswith(".git") else normalized


def _require_unmodified_git_object_authority(repo_root: Path) -> None:
    repository = Path(repo_root).expanduser().resolve()
    if _run_git(
        repository,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace",
    ):
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git repository contains replacement refs"
        )
    for relative in ("info/grafts", "objects/info/alternates"):
        git_path = Path(_run_git(repository, "rev-parse", "--git-path", relative))
        if not git_path.is_absolute():
            git_path = repository / git_path
        try:
            metadata = git_path.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_size != 0
        ):
            raise GatewayMinerMaintenanceRestartError(
                "candidate Git repository contains graft or alternate authority"
            )


@contextmanager
def _open_private_parent_fd(path: Path) -> Iterator[tuple[int, str]]:
    candidate = Path(path)
    parts = candidate.parts
    if (
        not candidate.is_absolute()
        or len(parts) < 2
        or candidate.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in parts[1:])
    ):
        raise GatewayMinerMaintenanceRestartError(
            "private file path is invalid"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[
        tuple[int, Optional[int], Optional[str], tuple[int, int]]
    ] = []
    validation_error: Optional[BaseException] = None
    try:
        root_descriptor = os.open(parts[0], flags)
        root_metadata = os.fstat(root_descriptor)
        descriptors.append(
            (
                root_descriptor,
                None,
                None,
                (root_metadata.st_dev, root_metadata.st_ino),
            )
        )
        for index, part in enumerate(parts[1:-1], start=1):
            parent_descriptor = descriptors[-1][0]
            descriptor = os.open(part, flags, dir_fd=parent_descriptor)
            metadata = os.fstat(descriptor)
            writable = bool(metadata.st_mode & 0o022)
            sticky_root_directory = bool(
                metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid not in {0, os.geteuid()}
                or (writable and not sticky_root_directory)
                or (
                    index == len(parts) - 2
                    and (metadata.st_uid != os.geteuid() or writable)
                )
            ):
                os.close(descriptor)
                raise GatewayMinerMaintenanceRestartError(
                    "private file ancestry is unsafe"
                )
            descriptors.append(
                (
                    descriptor,
                    parent_descriptor,
                    part,
                    (metadata.st_dev, metadata.st_ino),
                )
            )
        yield descriptors[-1][0], candidate.name
    except GatewayMinerMaintenanceRestartError:
        raise
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "private file ancestry is unavailable"
        ) from exc
    finally:
        try:
            for descriptor, parent_descriptor, name, identity in descriptors[1:]:
                assert parent_descriptor is not None and name is not None
                current = os.stat(
                    name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(current.st_mode)
                    or (current.st_dev, current.st_ino) != identity
                    or (opened.st_dev, opened.st_ino) != identity
                ):
                    validation_error = GatewayMinerMaintenanceRestartError(
                        "private file ancestry changed"
                    )
                    break
        except OSError as exc:
            validation_error = GatewayMinerMaintenanceRestartError(
                "private file ancestry changed"
            )
            validation_error.__cause__ = exc
        for descriptor, _parent, _name, _identity in reversed(descriptors):
            os.close(descriptor)
        if validation_error is not None and sys.exc_info()[0] is None:
            raise validation_error


def _read_private_regular_file(
    path: Path,
    *,
    expected_mode: int,
    minimum_bytes: int,
    maximum_bytes: int,
    label: str,
) -> bytes:
    with _open_private_parent_fd(path) as (parent_fd, leaf_name):
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(
                leaf_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            opened = os.fstat(descriptor)
            current = os.stat(
                leaf_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or stat.S_IMODE(opened.st_mode) != expected_mode
                or not minimum_bytes <= opened.st_size <= maximum_bytes
                or opened.st_dev != current.st_dev
                or opened.st_ino != current.st_ino
            ):
                raise GatewayMinerMaintenanceRestartError(
                    f"{label} identity is unsafe"
                )
            chunks: list[bytes] = []
            observed_size = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(65536, maximum_bytes + 1 - observed_size),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                observed_size += len(chunk)
                if observed_size > maximum_bytes:
                    raise GatewayMinerMaintenanceRestartError(
                        f"{label} is too large"
                    )
            payload = b"".join(chunks)
            final = os.fstat(descriptor)
            final_path = os.stat(
                leaf_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                len(payload) != opened.st_size
                or final.st_dev != opened.st_dev
                or final.st_ino != opened.st_ino
                or final.st_size != opened.st_size
                or final_path.st_dev != opened.st_dev
                or final_path.st_ino != opened.st_ino
            ):
                raise GatewayMinerMaintenanceRestartError(
                    f"{label} changed while reading"
                )
            return payload
        except GatewayMinerMaintenanceRestartError:
            raise
        except OSError as exc:
            raise GatewayMinerMaintenanceRestartError(
                f"{label} is unavailable"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _read_hydrated_gateway_environment(path: Path) -> bytes:
    return _read_private_regular_file(
        path,
        expected_mode=0o600,
        minimum_bytes=1,
        maximum_bytes=4 * 1024 * 1024,
        label="hydrated gateway environment",
    )


def _require_hydrated_environment_commitment(
    *,
    path: Path,
    expected_commitment: str,
) -> str:
    commitment = "sha256:" + hashlib.sha256(
        _read_hydrated_gateway_environment(path)
    ).hexdigest()
    if commitment != str(expected_commitment):
        raise GatewayMinerMaintenanceRestartError(
            "hydrated gateway environment differs from durable secret state"
        )
    return commitment


def _replace_private_regular_file(
    *,
    path: Path,
    expected_current_payload: bytes,
    replacement_payload: bytes,
    mode: int,
) -> None:
    with _open_private_parent_fd(path) as (parent_fd, leaf_name):
        current_fd: Optional[int] = None
        temporary_fd: Optional[int] = None
        temporary_name = f".{leaf_name}.{uuid.uuid4().hex}"
        try:
            current_fd = os.open(
                leaf_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            current_metadata = os.fstat(current_fd)
            current_chunks: list[bytes] = []
            current_size = 0
            maximum_current_size = max(
                len(expected_current_payload),
                4 * 1024 * 1024,
            )
            while True:
                chunk = os.read(
                    current_fd,
                    min(65536, maximum_current_size + 1 - current_size),
                )
                if not chunk:
                    break
                current_chunks.append(chunk)
                current_size += len(chunk)
                if current_size > maximum_current_size:
                    raise GatewayMinerMaintenanceRestartError(
                        "installed gateway host wrapper is too large"
                    )
            current_payload = b"".join(current_chunks)
            if (
                not stat.S_ISREG(current_metadata.st_mode)
                or current_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(current_metadata.st_mode) != mode
                or current_payload != expected_current_payload
                or len(current_payload) != current_metadata.st_size
            ):
                raise GatewayMinerMaintenanceRestartError(
                    "installed gateway host wrapper changed before reconciliation"
                )
            current_path = os.stat(
                leaf_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                current_path.st_dev != current_metadata.st_dev
                or current_path.st_ino != current_metadata.st_ino
            ):
                raise GatewayMinerMaintenanceRestartError(
                    "installed gateway host wrapper changed before reconciliation"
                )
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=parent_fd,
            )
            written = 0
            while written < len(replacement_payload):
                count = os.write(temporary_fd, replacement_payload[written:])
                if count <= 0:
                    raise GatewayMinerMaintenanceRestartError(
                        "gateway host wrapper reconciliation write was incomplete"
                    )
                written += count
            os.fchmod(temporary_fd, mode)
            os.fsync(temporary_fd)
            os.close(temporary_fd)
            temporary_fd = None
            os.rename(
                temporary_name,
                leaf_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        except GatewayMinerMaintenanceRestartError:
            raise
        except OSError as exc:
            raise GatewayMinerMaintenanceRestartError(
                "installed gateway host wrapper could not be reconciled"
            ) from exc
        finally:
            if current_fd is not None:
                os.close(current_fd)
            if temporary_fd is not None:
                os.close(temporary_fd)
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def _read_exact_installed_file(
    path: Path,
    *,
    expected_mode: int,
    label: str,
    allow_open_fd_path: bool = False,
) -> bytes:
    candidate = Path(path)
    proc_fd_path = bool(
        allow_open_fd_path
        and re.fullmatch(r"/proc/[0-9]+/fd/[0-9]+", str(candidate))
    )
    if not proc_fd_path:
        return _read_private_regular_file(
            candidate,
            expected_mode=expected_mode,
            minimum_bytes=2,
            maximum_bytes=4 * 1024 * 1024,
            label=label,
        )
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(candidate, os.O_RDONLY)
        opened_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or opened_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(opened_metadata.st_mode) != expected_mode
            or opened_metadata.st_size < 2
            or opened_metadata.st_size > 4 * 1024 * 1024
        ):
            raise GatewayMinerMaintenanceRestartError(
                f"{label} identity is unsafe"
            )
        payload = os.pread(descriptor, 4 * 1024 * 1024 + 1, 0)
        final_metadata = os.fstat(descriptor)
        if (
            final_metadata.st_dev != opened_metadata.st_dev
            or final_metadata.st_ino != opened_metadata.st_ino
            or final_metadata.st_size != opened_metadata.st_size
        ):
            raise GatewayMinerMaintenanceRestartError(
                f"{label} changed while reading"
            )
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(f"{label} is unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(payload) != opened_metadata.st_size:
        raise GatewayMinerMaintenanceRestartError(f"{label} changed while reading")
    return payload


def _harden_installed_controller_directory(directory_path: Path) -> tuple[int, int]:
    path = Path(directory_path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: Optional[int] = None
    reopened: Optional[int] = None
    try:
        path_metadata = path.lstat()
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(path_metadata.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_gid != os.getegid()
            or opened.st_dev != path_metadata.st_dev
            or opened.st_ino != path_metadata.st_ino
            or stat.S_IMODE(opened.st_mode) not in {0o700, 0o775}
        ):
            raise GatewayMinerMaintenanceRestartError(
                "installed N-1 controller ancestry is unsafe"
            )
        if stat.S_IMODE(opened.st_mode) == 0o775:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        hardened = os.fstat(descriptor)
        if (
            hardened.st_dev != opened.st_dev
            or hardened.st_ino != opened.st_ino
            or hardened.st_uid != os.geteuid()
            or hardened.st_gid != os.getegid()
            or stat.S_IMODE(hardened.st_mode) != 0o700
        ):
            raise GatewayMinerMaintenanceRestartError(
                "installed N-1 controller ancestry could not be hardened"
            )
        os.close(descriptor)
        descriptor = None
        reopened = os.open(path, flags)
        verified = os.fstat(reopened)
        current = path.lstat()
        if (
            not stat.S_ISDIR(verified.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or verified.st_dev != hardened.st_dev
            or verified.st_ino != hardened.st_ino
            or verified.st_dev != current.st_dev
            or verified.st_ino != current.st_ino
            or verified.st_uid != os.geteuid()
            or verified.st_gid != os.getegid()
            or stat.S_IMODE(verified.st_mode) != 0o700
        ):
            raise GatewayMinerMaintenanceRestartError(
                "installed N-1 controller ancestry changed while hardening"
            )
        return (verified.st_dev, verified.st_ino)
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "installed N-1 controller ancestry is unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if reopened is not None:
            os.close(reopened)


def _verified_installed_controller_release_directory(
    directory_path: Path,
) -> tuple[int, int]:
    path = Path(directory_path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: Optional[int] = None
    try:
        path_metadata = path.lstat()
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(path_metadata.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_gid != os.getegid()
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_dev != path_metadata.st_dev
            or opened.st_ino != path_metadata.st_ino
        ):
            raise GatewayMinerMaintenanceRestartError(
                "installed N-1 controller target is invalid"
            )
        return (opened.st_dev, opened.st_ino)
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "installed N-1 controller target is unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verified_installed_controller_bundle(
    *,
    repo_root: Path,
    controller_current: Path,
    host_restart_path: Path,
    expected_commit: str,
    host_restart_is_open_fd: bool = False,
    reconcile_host_wrapper: bool = False,
) -> dict[str, Any]:
    current = Path(controller_current)
    controller_root = current.parent
    releases_root = controller_root / "releases"
    for directory_path in (controller_root.parent, controller_root, releases_root):
        _harden_installed_controller_directory(directory_path)
    try:
        link_metadata = current.lstat()
        link_target = os.readlink(current)
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "installed N-1 controller identity is unavailable"
        ) from exc
    match = re.fullmatch(r"releases/([0-9a-f]{40})", link_target)
    if (
        not stat.S_ISLNK(link_metadata.st_mode)
        or link_metadata.st_uid != os.geteuid()
        or match is None
    ):
        raise GatewayMinerMaintenanceRestartError(
            "installed N-1 controller link is invalid"
        )
    controller_commit = match.group(1)
    resolved = current.resolve(strict=True)
    expected_resolved = releases_root / controller_commit
    if resolved != expected_resolved:
        raise GatewayMinerMaintenanceRestartError(
            "installed N-1 controller target is invalid"
        )
    release_identity = _verified_installed_controller_release_directory(
        expected_resolved
    )
    if not _COMMIT_RE.fullmatch(expected_commit):
        raise GatewayMinerMaintenanceRestartError("candidate commit is invalid")
    if not _git_commit_exists(repo_root, expected_commit) or not _git_commit_exists(
        repo_root, controller_commit
    ):
        raise GatewayMinerMaintenanceRestartError(
            "installed N-1 controller Git object is unavailable"
        )
    if not any(
        _git_is_ancestor(repo_root, floor, controller_commit)
        for floor in SUPPORTED_N_MINUS_ONE_CONTROLLER_COMMITS
    ) or not _git_is_ancestor(repo_root, controller_commit, expected_commit):
        raise GatewayMinerMaintenanceRestartError(
            "installed N-1 controller is not compatible with the candidate"
        )
    controller_restart = _read_exact_installed_file(
        resolved / "gw_restart.sh",
        expected_mode=0o700,
        label="installed N-1 controller wrapper",
    )
    controller_helper = _read_exact_installed_file(
        resolved / "scripts/gateway_git_deploy.py",
        expected_mode=0o600,
        label="installed N-1 deployment helper",
    )
    controller_exact_restart = _read_exact_installed_file(
        resolved / "Leadpoet/utils/exact_commit_restart_v2.py",
        expected_mode=0o600,
        label="installed N-1 exact-commit helper",
    )
    controller_memory_guard = _read_exact_installed_file(
        resolved / "gateway/tee/host_memory_guard_v2.py",
        expected_mode=0o600,
        label="installed N-1 memory guard",
    )
    host_restart = _read_exact_installed_file(
        Path(host_restart_path),
        expected_mode=0o700,
        label="installed gateway host wrapper",
        allow_open_fd_path=host_restart_is_open_fd,
    )
    if (
        controller_restart
        != _run_git_bytes(repo_root, "show", f"{controller_commit}:gw_restart.sh")
        or controller_helper
        != _run_git_bytes(
            repo_root,
            "show",
            f"{controller_commit}:scripts/gateway_git_deploy.py",
        )
        or controller_exact_restart
        != _run_git_bytes(
            repo_root,
            "show",
            f"{controller_commit}:Leadpoet/utils/exact_commit_restart_v2.py",
        )
        or controller_memory_guard
        != _run_git_bytes(
            repo_root,
            "show",
            f"{controller_commit}:gateway/tee/host_memory_guard_v2.py",
        )
    ):
        raise GatewayMinerMaintenanceRestartError(
            "installed N-1 controller bytes differ from Git authority"
        )
    if host_restart != controller_restart:
        compatible_host_commits = {
            commit
            for commit in (
                SUPPORTED_N_MINUS_ONE_CONTROLLER_COMMITS
                | {controller_commit, expected_commit}
            )
            if host_restart
            == _run_git_bytes(repo_root, "show", f"{commit}:gw_restart.sh")
        }
        if not compatible_host_commits:
            raise GatewayMinerMaintenanceRestartError(
                "installed gateway host wrapper differs from Git authority"
            )
        if not reconcile_host_wrapper:
            raise GatewayMinerMaintenanceRestartError(
                "installed gateway host wrapper differs from current controller"
            )
        _replace_private_regular_file(
            path=Path(host_restart_path),
            expected_current_payload=host_restart,
            replacement_payload=controller_restart,
            mode=0o700,
        )
        host_restart = _read_exact_installed_file(
            Path(host_restart_path),
            expected_mode=0o700,
            label="reconciled gateway host wrapper",
        )
        if host_restart != controller_restart:
            raise GatewayMinerMaintenanceRestartError(
                "installed gateway host wrapper reconciliation failed"
            )
    final_link = current.lstat()
    final_target = os.readlink(current)
    if (
        final_link.st_dev != link_metadata.st_dev
        or final_link.st_ino != link_metadata.st_ino
        or final_target != link_target
        or _verified_installed_controller_release_directory(expected_resolved)
        != release_identity
    ):
        raise GatewayMinerMaintenanceRestartError(
            "installed N-1 controller changed while verifying"
        )
    payloads = {
        "wrapper": controller_restart,
        "git_helper": controller_helper,
        "exact_commit_helper": controller_exact_restart,
        "memory_guard": controller_memory_guard,
    }
    return {
        "controller_commit": controller_commit,
        "payloads": payloads,
        "commitments": {
            name: "sha256:" + hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
    }


def _verify_installed_controller(
    *,
    repo_root: Path,
    controller_current: Path,
    host_restart_path: Path,
    expected_commit: str,
    host_restart_is_open_fd: bool = False,
) -> str:
    """Return the exact verified controller commit for compatibility callers."""

    bundle = _verified_installed_controller_bundle(
        repo_root=repo_root,
        controller_current=controller_current,
        host_restart_path=host_restart_path,
        expected_commit=expected_commit,
        host_restart_is_open_fd=host_restart_is_open_fd,
    )
    return str(bundle["controller_commit"])


def _validate_candidate_identity(
    *,
    repo_root: Path,
    candidate_root: Path,
    plan_file: Path,
    expected_commit: str,
    controller_current: Path,
    host_restart_path: Path,
    host_restart_is_open_fd: bool = False,
    reconcile_host_wrapper: bool = False,
) -> dict[str, Any]:
    commit = str(expected_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayMinerMaintenanceRestartError("candidate commit is invalid")
    repository = Path(repo_root).expanduser().resolve()
    materialized = Path(candidate_root).expanduser().resolve()
    _require_unmodified_git_object_authority(repository)
    plan = _load_json(
        Path(plan_file),
        label="isolated N-1 deployment plan",
        max_bytes=64 * 1024,
    )
    required_plan = {
        "schema_version": GIT_DEPLOYMENT_SCHEMA_VERSION,
        "source": "github",
        "status": "prepared",
        "stage": "git_prepare",
        "mode": "pinned",
        "branch": DEFAULT_BRANCH,
        "target_sha": commit,
        "branch_head_sha": commit,
        "repo_root": str(repository),
    }
    if any(plan.get(name) != value for name, value in required_plan.items()):
        raise GatewayMinerMaintenanceRestartError(
            "isolated N-1 deployment plan differs from the exact candidate"
        )
    if _canonical_remote(str(plan.get("remote_url") or "")) != _canonical_remote(
        DEFAULT_REPO_URL
    ):
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git remote differs from the production repository"
        )
    previous_sha = str(plan.get("previous_sha") or "").lower()
    planned_tree = str(plan.get("tree_hash") or "").lower()
    if not _COMMIT_RE.fullmatch(previous_sha) or not _TREE_RE.fullmatch(planned_tree):
        raise GatewayMinerMaintenanceRestartError(
            "isolated N-1 deployment plan has invalid Git identities"
        )
    if _run_git(repository, "rev-parse", "HEAD") != previous_sha:
        raise GatewayMinerMaintenanceRestartError(
            "deployed checkout changed after isolated N-1 preparation"
        )
    if _run_git(repository, "rev-parse", "origin/main^{commit}") != commit:
        raise GatewayMinerMaintenanceRestartError(
            "fetched main no longer matches the exact candidate"
        )
    if _canonical_remote(
        _run_git(repository, "remote", "get-url", "origin")
    ) != _canonical_remote(DEFAULT_REPO_URL):
        raise GatewayMinerMaintenanceRestartError(
            "candidate Git remote differs from the production repository"
        )
    if _run_git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise GatewayMinerMaintenanceRestartError(
            "deployed checkout is not clean"
        )
    tree_evidence = verify_materialized_tree(
        repo_root=repository,
        materialized_root=materialized,
        target_sha=commit,
        strict_extras=True,
    )
    if tree_evidence.get("tree_hash") != planned_tree:
        raise GatewayMinerMaintenanceRestartError(
            "materialized candidate tree differs from the isolated N-1 plan"
        )
    controller_bundle = _verified_installed_controller_bundle(
        repo_root=repository,
        controller_current=controller_current,
        host_restart_path=host_restart_path,
        expected_commit=commit,
        host_restart_is_open_fd=host_restart_is_open_fd,
        reconcile_host_wrapper=reconcile_host_wrapper,
    )
    _require_unmodified_git_object_authority(repository)
    return {
        **tree_evidence,
        "previous_sha": previous_sha,
        "n_minus_one_controller_commit": str(
            controller_bundle["controller_commit"]
        ),
        "controller_bundle": controller_bundle,
    }


def _proof_body(
    *,
    candidate_commit: str,
    tree_evidence: Mapping[str, Any],
    release_evidence: Mapping[str, Any],
    final_secret_result: Mapping[str, str],
    restart_invocation_id: str,
    live_process_commitment: str,
) -> dict[str, str]:
    release_channel = release_evidence["channel"]
    gateway_release = validate_release_manifest(
        release_channel["gateway_release_manifest"]
    )
    controller = tree_evidence["controller_bundle"]
    commitments = controller["commitments"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "candidate_commit": str(candidate_commit),
        "candidate_tree_hash": str(tree_evidence["tree_hash"]),
        "candidate_blob_manifest_sha256": "sha256:"
        + str(tree_evidence["blob_manifest_sha256"]),
        "pre_hydration_runtime_commit": str(tree_evidence["previous_sha"]),
        "n_minus_one_controller_commit": str(
            tree_evidence["n_minus_one_controller_commit"]
        ),
        "release_channel_hash": str(release_channel["channel_hash"]),
        "release_channel_object_version_id": str(
            release_evidence["object_version_id"]
        ),
        "release_channel_object_sha256": str(
            release_evidence["object_sha256"]
        ),
        "release_channel_retain_until": str(
            release_evidence["object_retain_until"]
        ),
        "gateway_release_hash": str(gateway_release["release_hash"]),
        "current_secret_version_id": str(final_secret_result["current_version_id"]),
        "current_document_commitment": str(
            final_secret_result["current_document_commitment"]
        ),
        "current_hydrated_environment_commitment": str(
            final_secret_result["current_hydrated_environment_commitment"]
        ),
        "current_stage_topology_commitment": str(
            final_secret_result["current_stage_topology_commitment"]
        ),
        "controller_wrapper_sha256": str(commitments["wrapper"]),
        "controller_git_helper_sha256": str(commitments["git_helper"]),
        "controller_exact_commit_helper_sha256": str(
            commitments["exact_commit_helper"]
        ),
        "controller_memory_guard_sha256": str(commitments["memory_guard"]),
        "pre_hydration_live_process_commitment": str(
            live_process_commitment
        ),
        "restart_invocation_id": str(restart_invocation_id),
        "prepared_at": _utc_now(),
    }
    return {**body, "proof_hash": sha256_json(body)}


def _validate_proof_document(value: Mapping[str, Any]) -> dict[str, str]:
    if set(value) != _PROOF_FIELDS or value.get("schema_version") != SCHEMA_VERSION:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance invocation proof fields are invalid"
        )
    normalized = {name: str(value[name]) for name in _PROOF_FIELDS}
    body = {
        name: normalized[name]
        for name in _PROOF_FIELDS
        if name != "proof_hash"
    }
    commitment_fields = (
        "candidate_blob_manifest_sha256",
        "release_channel_hash",
        "release_channel_object_sha256",
        "gateway_release_hash",
        "current_document_commitment",
        "current_hydrated_environment_commitment",
        "current_stage_topology_commitment",
        "controller_wrapper_sha256",
        "controller_git_helper_sha256",
        "controller_exact_commit_helper_sha256",
        "controller_memory_guard_sha256",
        "pre_hydration_live_process_commitment",
    )
    if (
        not _COMMIT_RE.fullmatch(normalized["candidate_commit"])
        or not _TREE_RE.fullmatch(normalized["candidate_tree_hash"])
        or not _COMMIT_RE.fullmatch(
            normalized["pre_hydration_runtime_commit"]
        )
        or not _COMMIT_RE.fullmatch(normalized["n_minus_one_controller_commit"])
        or any(
            not _SHA256_RE.fullmatch(normalized[name])
            for name in commitment_fields
        )
        or not _s3_version_id(normalized["release_channel_object_version_id"])
        or not _VERSION_ID_RE.fullmatch(normalized["current_secret_version_id"])
        or not normalized["release_channel_retain_until"].endswith("Z")
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}",
            normalized["restart_invocation_id"],
        )
        or not normalized["prepared_at"].endswith("Z")
        or normalized["proof_hash"] != sha256_json(body)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance invocation proof commitments are invalid"
        )
    return normalized


def _serialized_proof(proof: Mapping[str, Any]) -> bytes:
    validated = _validate_proof_document(proof)
    payload = (
        json.dumps(validated, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    if not 2 <= len(payload) <= MAX_PROOF_BYTES:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance invocation proof size is invalid"
        )
    return payload


def _required_memfd_seals() -> int:
    names = ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    if not hasattr(os, "memfd_create") or any(
        not hasattr(fcntl, name) for name in names
    ):
        raise GatewayMinerMaintenanceRestartError(
            "sealed memory files are unavailable"
        )
    return sum(int(getattr(fcntl, name)) for name in names)


def _require_reserved_memfd_numbers_available() -> None:
    if os.environ.get(PROOF_FD_ENV_NAME):
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance proof descriptor collides with bootstrap"
        )
    for descriptor in (
        PROOF_FD_NUMBER,
        CONTROLLER_WRAPPER_FD_NUMBER,
        CONTROLLER_GIT_HELPER_FD_NUMBER,
        CONTROLLER_EXACT_COMMIT_HELPER_FD_NUMBER,
        CONTROLLER_MEMORY_GUARD_FD_NUMBER,
    ):
        try:
            os.fstat(descriptor)
        except OSError as exc:
            if exc.errno == errno.EBADF:
                continue
            raise GatewayMinerMaintenanceRestartError(
                "reserved miner-maintenance descriptor identity is unavailable"
            ) from exc
        raise GatewayMinerMaintenanceRestartError(
            "reserved miner-maintenance descriptor is already open"
        )


def _seal_payload_at_fd_number(
    *,
    payload: bytes,
    fd_number: int,
    name: str,
    max_bytes: int,
) -> int:
    if (
        not isinstance(payload, bytes)
        or not 2 <= len(payload) <= int(max_bytes)
        or not 190 <= int(fd_number) <= 199
        or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", str(name))
    ):
        raise GatewayMinerMaintenanceRestartError(
            "sealed memory file request is invalid"
        )
    required_seals = _required_memfd_seals()
    descriptor: Optional[int] = None
    try:
        descriptor = os.memfd_create(
            str(name),
            flags=int(getattr(os, "MFD_ALLOW_SEALING", 0x0002)),
        )
        os.fchmod(descriptor, 0o400)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise GatewayMinerMaintenanceRestartError(
                    "sealed memory file write was incomplete"
                )
            written += count
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
        if int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)) & required_seals != required_seals:
            raise GatewayMinerMaintenanceRestartError(
                "sealed memory file did not retain all required seals"
            )
        if descriptor != int(fd_number):
            os.dup2(descriptor, int(fd_number), inheritable=True)
            os.close(descriptor)
            descriptor = int(fd_number)
        os.set_inheritable(descriptor, True)
        return descriptor
    except GatewayMinerMaintenanceRestartError:
        raise
    except (OSError, ValueError) as exc:
        raise GatewayMinerMaintenanceRestartError(
            "sealed memory file could not be created"
        ) from exc
    finally:
        if descriptor is not None and descriptor != int(fd_number):
            os.close(descriptor)


def _read_sealed_payload_fd(
    fd_number: int,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    try:
        descriptor = int(fd_number)
    except (TypeError, ValueError) as exc:
        raise GatewayMinerMaintenanceRestartError(
            f"{label} descriptor is invalid"
        ) from exc
    if not 190 <= descriptor <= 199:
        raise GatewayMinerMaintenanceRestartError(
            f"{label} descriptor is invalid"
        )
    required_seals = _required_memfd_seals()
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or not 2 <= metadata.st_size <= int(max_bytes)
            or not os.get_inheritable(descriptor)
            or int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS))
            & required_seals
            != required_seals
        ):
            raise GatewayMinerMaintenanceRestartError(
                f"{label} descriptor identity is unsafe"
            )
        payload = os.pread(descriptor, int(max_bytes) + 1, 0)
        final = os.fstat(descriptor)
        if (
            final.st_dev != metadata.st_dev
            or final.st_ino != metadata.st_ino
            or final.st_size != metadata.st_size
            or len(payload) != metadata.st_size
        ):
            raise GatewayMinerMaintenanceRestartError(
                f"{label} changed while reading"
            )
    except GatewayMinerMaintenanceRestartError:
        raise
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            f"{label} is unavailable"
        ) from exc
    return payload


def _proof_from_fd(fd_number: int) -> dict[str, str]:
    payload = _read_sealed_payload_fd(
        fd_number,
        label="miner-maintenance invocation proof",
        max_bytes=MAX_PROOF_BYTES,
    )
    return _validate_proof_document(
        _load_json_bytes(
            payload,
            label="miner-maintenance invocation proof",
        )
    )


def _proof_fd_from_environment(
    environment: Mapping[str, str],
) -> Optional[int]:
    raw_value = str(environment.get(PROOF_FD_ENV_NAME) or "")
    try:
        os.fstat(PROOF_FD_NUMBER)
        descriptor_present = True
    except OSError as exc:
        if exc.errno != errno.EBADF:
            raise GatewayMinerMaintenanceRestartError(
                "miner-maintenance invocation proof descriptor is unavailable"
            ) from exc
        descriptor_present = False
    if descriptor_present:
        if raw_value != str(PROOF_FD_NUMBER):
            raise GatewayMinerMaintenanceRestartError(
                "miner-maintenance invocation proof pointer was downgraded"
            )
        return PROOF_FD_NUMBER
    if raw_value:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance invocation proof descriptor was lost"
        )
    return None


def _require_disabled_parent_environment(
    parent_environment: Mapping[str, str],
) -> None:
    if str(parent_environment.get(TARGET_ENV_NAME) or "").strip() != TARGET_ENV_VALUE:
        raise GatewayMinerMaintenanceRestartError(
            "restart parent did not hydrate disabled miner submissions"
        )


def _require_disabled_secret_readback(
    *,
    client: Any,
    expected_current_version_id: Optional[str] = None,
) -> dict[str, str]:
    current = disable_gateway_miner_submissions_secret(
        secrets_client=client,
        expected_current_version_id=expected_current_version_id,
    )
    if current.get("status") != "already_disabled":
        raise GatewayMinerMaintenanceRestartError(
            "durable gateway secret does not disable miner submissions"
        )
    return current


def _verify_locked_release_matches(
    *,
    deploy_commit: str,
    gateway_release_hash: str,
    release_s3_client: Any,
) -> dict[str, Any]:
    release_evidence = _fetch_locked_release_channel(
        commit_sha=str(deploy_commit).lower(),
        s3_client=release_s3_client,
    )
    if (
        release_evidence["channel"]["gateway_release_manifest"]["release_hash"]
        != str(gateway_release_hash)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "immutable release channel differs from the candidate"
        )
    return release_evidence


def _verify_proof_against_state(
    *,
    proof: Mapping[str, str],
    deploy_commit: str,
    candidate_tree_hash: str,
    gateway_release_hash: str,
    client: Any,
    release_s3_client: Any,
    tree_evidence: Optional[Mapping[str, Any]] = None,
    restart_invocation_id: Optional[str] = None,
    live_process_commitment: Optional[str] = None,
    hydrated_environment_path: Optional[Path] = None,
) -> dict[str, str]:
    validated = _validate_proof_document(proof)
    if (
        validated["candidate_commit"] != str(deploy_commit).lower()
        or validated["candidate_tree_hash"] != str(candidate_tree_hash).lower()
        or validated["gateway_release_hash"] != str(gateway_release_hash)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance invocation proof differs from the candidate"
        )
    if (
        restart_invocation_id is not None
        and validated["restart_invocation_id"] != str(restart_invocation_id)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance invocation proof differs from the restart"
        )
    if (
        live_process_commitment is not None
        and validated["pre_hydration_live_process_commitment"]
        != str(live_process_commitment)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "running gateway process differs from the invocation proof"
        )
    release_evidence = _verify_locked_release_matches(
        deploy_commit=deploy_commit,
        gateway_release_hash=gateway_release_hash,
        release_s3_client=release_s3_client,
    )
    release_channel = release_evidence["channel"]
    if (
        validated["release_channel_hash"] != release_channel["channel_hash"]
        or validated["release_channel_object_version_id"]
        != release_evidence["object_version_id"]
        or validated["release_channel_object_sha256"]
        != release_evidence["object_sha256"]
    ):
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance immutable release differs from the invocation proof"
        )
    try:
        prepared_retention = datetime.fromisoformat(
            validated["release_channel_retain_until"].replace("Z", "+00:00")
        )
        current_retention = datetime.fromisoformat(
            str(release_evidence["object_retain_until"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance release retention is invalid"
        ) from exc
    if current_retention < prepared_retention:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance release retention was shortened"
        )
    current = _require_disabled_secret_readback(
        client=client,
        expected_current_version_id=validated["current_secret_version_id"],
    )
    if (
        current.get("current_document_commitment")
        != validated["current_document_commitment"]
        or current.get("current_hydrated_environment_commitment")
        != validated["current_hydrated_environment_commitment"]
        or current.get("current_stage_topology_commitment")
        != validated["current_stage_topology_commitment"]
    ):
        raise GatewayMinerMaintenanceRestartError(
            "durable miner-maintenance state differs from the invocation proof"
        )
    if hydrated_environment_path is not None:
        _require_hydrated_environment_commitment(
            path=hydrated_environment_path,
            expected_commitment=validated[
                "current_hydrated_environment_commitment"
            ],
        )
    final_current = _require_disabled_secret_readback(
        client=client,
        expected_current_version_id=validated["current_secret_version_id"],
    )
    if (
        final_current.get("current_document_commitment")
        != validated["current_document_commitment"]
        or final_current.get("current_hydrated_environment_commitment")
        != validated["current_hydrated_environment_commitment"]
        or final_current.get("current_stage_topology_commitment")
        != validated["current_stage_topology_commitment"]
    ):
        raise GatewayMinerMaintenanceRestartError(
            "durable miner-maintenance state changed during hydration verification"
        )
    if tree_evidence is not None:
        controller = tree_evidence["controller_bundle"]
        commitments = controller["commitments"]
        if (
            validated["candidate_blob_manifest_sha256"]
            != "sha256:" + str(tree_evidence["blob_manifest_sha256"])
            or validated["pre_hydration_runtime_commit"]
            != str(tree_evidence["previous_sha"])
            or validated["n_minus_one_controller_commit"]
            != str(controller["controller_commit"])
            or validated["controller_wrapper_sha256"]
            != str(commitments["wrapper"])
            or validated["controller_git_helper_sha256"]
            != str(commitments["git_helper"])
            or validated["controller_exact_commit_helper_sha256"]
            != str(commitments["exact_commit_helper"])
            or validated["controller_memory_guard_sha256"]
            != str(commitments["memory_guard"])
        ):
            raise GatewayMinerMaintenanceRestartError(
                "verified N-1 controller differs from the invocation proof"
            )
    return {
        "status": "invocation_verified",
        "candidate_commit": validated["candidate_commit"],
        "current_secret_version_id": validated["current_secret_version_id"],
        "proof_hash": validated["proof_hash"],
        "release_channel_object_version_id": validated[
            "release_channel_object_version_id"
        ],
    }


def prepare_gateway_miner_maintenance_restart(
    *,
    repo_root: Path,
    candidate_root: Path,
    plan_file: Path,
    expected_commit: str,
    controller_current: Path,
    host_restart_path: Path,
    restart_invocation_id: str,
    recovery_journal_path: Path = DEFAULT_RECOVERY_JOURNAL_PATH,
    secrets_client: Any = None,
    release_s3_client: Any = None,
) -> dict[str, Any]:
    """Apply false and return one non-persistent invocation proof."""

    _require_canonical_restart_lock_fd()
    _require_fixed_bootstrap_authority(os.environ)
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}",
        str(restart_invocation_id),
    ):
        raise GatewayMinerMaintenanceRestartError(
            "gateway restart invocation identity is invalid"
        )
    secrets_client, release_s3_client = _resolve_bootstrap_aws_clients(
        secrets_client=secrets_client,
        release_s3_client=release_s3_client,
    )
    tree_evidence = _validate_candidate_identity(
        repo_root=repo_root,
        candidate_root=candidate_root,
        plan_file=plan_file,
        expected_commit=expected_commit,
        controller_current=controller_current,
        host_restart_path=host_restart_path,
        reconcile_host_wrapper=True,
    )
    _verify_protected_source()
    live_process_commitment = _pre_hydration_live_process_commitment(
        tree_evidence
    )
    release_evidence = _fetch_locked_release_channel(
        commit_sha=expected_commit,
        s3_client=release_s3_client,
    )
    channel = release_evidence["channel"]
    if str(channel.get("commit_sha") or "") != expected_commit:
        raise GatewayMinerMaintenanceRestartError(
            "approved release channel is for another candidate"
        )
    with _open_recovery_journal_parent_fd(
        recovery_journal_path
    ) as recovery_authority:
        _recover_orphan_transaction(
            secrets_client,
            recovery_journal_path=recovery_journal_path,
            recovery_journal_authority=recovery_authority,
        )
        observed = disable_gateway_miner_submissions_secret(
            secrets_client=secrets_client
        )
        version_id = str(observed["current_version_id"])
        if observed["status"] == "already_disabled":
            applied = observed
        elif observed["status"] == "verified":
            applied = _apply_gateway_miner_submissions_secret(
                secrets_client=secrets_client,
                expected_current_version_id=version_id,
                recovery_journal_path=recovery_journal_path,
                recovery_journal_authority=recovery_authority,
            )
        else:
            raise GatewayMinerMaintenanceRestartError(
                "miner-submission disable verification returned an invalid status"
            )
        final_version = str(applied["current_version_id"])
        final_result = disable_gateway_miner_submissions_secret(
            secrets_client=secrets_client,
            expected_current_version_id=final_version,
        )
        if final_result.get("status") != "already_disabled":
            raise GatewayMinerMaintenanceRestartError(
                "miner submissions were not disabled after exact readback"
            )
    proof = _proof_body(
        candidate_commit=expected_commit,
        tree_evidence=tree_evidence,
        release_evidence=release_evidence,
        final_secret_result=final_result,
        restart_invocation_id=restart_invocation_id,
        live_process_commitment=live_process_commitment,
    )
    _validate_proof_document(proof)
    return {
        "status": "prepared",
        "proof": proof,
        "tree_evidence": tree_evidence,
        "gateway_release_hash": str(
            channel["gateway_release_manifest"]["release_hash"]
        ),
    }


def verify_gateway_miner_maintenance_state(
    *,
    deploy_commit: str,
    candidate_tree_hash: str,
    gateway_release_hash: str,
    parent_environment: Mapping[str, str],
    secrets_client: Any = None,
    release_s3_client: Any = None,
    bind_live_process_to_proof: bool = True,
    hydrated_environment_path: Path = CANONICAL_GATEWAY_ENV_PATH,
) -> dict[str, str]:
    """Verify false hydration and durable state before production shutdown."""

    _require_fixed_bootstrap_authority(parent_environment)
    _require_disabled_parent_environment(parent_environment)
    secrets_client, release_s3_client = _resolve_bootstrap_aws_clients(
        secrets_client=secrets_client,
        release_s3_client=release_s3_client,
    )
    proof_fd = _proof_fd_from_environment(parent_environment)
    if proof_fd is not None:
        proof = _proof_from_fd(proof_fd)
        validated_proof = _validate_proof_document(proof)
        if bind_live_process_to_proof:
            proof_runtime_commit = validated_proof[
                "pre_hydration_runtime_commit"
            ]
            proof_controller_commit = validated_proof[
                "n_minus_one_controller_commit"
            ]
            live_process_commitment = (
                _live_gateway_restart_authority_commitment(
                    expected_runtime_commit=proof_runtime_commit,
                    verified_controller_commit=proof_controller_commit,
                    allow_legacy_n_minus_one_git_helper=(
                        proof_runtime_commit
                        == LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
                        and proof_controller_commit
                        == LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
                    ),
                )
            )
        else:
            _live_gateway_restart_authority_commitment()
            live_process_commitment = None
        return _verify_proof_against_state(
            proof=validated_proof,
            deploy_commit=deploy_commit,
            candidate_tree_hash=candidate_tree_hash,
            gateway_release_hash=gateway_release_hash,
            client=secrets_client,
            release_s3_client=release_s3_client,
            restart_invocation_id=parent_environment.get(
                "GATEWAY_RESTART_INVOCATION_ID"
            ),
            live_process_commitment=live_process_commitment,
            hydrated_environment_path=hydrated_environment_path,
        )
    _live_gateway_restart_authority_commitment()
    current = _require_disabled_secret_readback(client=secrets_client)
    _require_hydrated_environment_commitment(
        path=hydrated_environment_path,
        expected_commitment=str(
            current["current_hydrated_environment_commitment"]
        ),
    )
    final_current = _require_disabled_secret_readback(
        client=secrets_client,
        expected_current_version_id=str(current["current_version_id"]),
    )
    if (
        final_current.get("current_document_commitment")
        != current.get("current_document_commitment")
        or final_current.get("current_hydrated_environment_commitment")
        != current.get("current_hydrated_environment_commitment")
        or final_current.get("current_stage_topology_commitment")
        != current.get("current_stage_topology_commitment")
    ):
        raise GatewayMinerMaintenanceRestartError(
            "durable miner-maintenance state changed during hydration verification"
        )
    release_evidence = _verify_locked_release_matches(
        deploy_commit=deploy_commit,
        gateway_release_hash=gateway_release_hash,
        release_s3_client=release_s3_client,
    )
    return {
        "status": "durable_false_verified",
        "current_secret_version_id": str(current["current_version_id"]),
        "release_channel_object_version_id": str(
            release_evidence["object_version_id"]
        ),
    }


def _read_handoff_marker(
    *,
    path: Path,
    expected_commit: str,
    nonce: str,
) -> str:
    marker = Path(path)
    expected = (str(expected_commit) + " " + str(nonce) + "\n").encode("ascii")
    cancelled = (
        "failed:" + str(expected_commit) + " " + str(nonce) + "\n"
    ).encode("ascii")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(marker, flags)
        opened = os.fstat(descriptor)
        current = marker.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size not in {len(expected), len(cancelled)}
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            raise GatewayMinerMaintenanceRestartError(
                "miner-maintenance handoff marker identity is unsafe"
            )
        payload = os.read(descriptor, max(len(expected), len(cancelled)) + 1)
        final = os.fstat(descriptor)
        if (
            final.st_dev != opened.st_dev
            or final.st_ino != opened.st_ino
            or final.st_size != opened.st_size
            or len(payload) != opened.st_size
        ):
            raise GatewayMinerMaintenanceRestartError(
                "miner-maintenance handoff marker changed while reading"
            )
        if payload == expected:
            action = "continue"
        elif payload == cancelled:
            action = "cancel"
        else:
            raise GatewayMinerMaintenanceRestartError(
                "miner-maintenance handoff marker content is invalid"
            )
        final_path = marker.lstat()
        if (
            final_path.st_dev != opened.st_dev
            or final_path.st_ino != opened.st_ino
        ):
            raise GatewayMinerMaintenanceRestartError(
                "miner-maintenance handoff marker changed before removal"
            )
        marker.unlink()
        return action
    except GatewayMinerMaintenanceRestartError:
        raise
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance handoff marker is unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _wait_for_handoff_marker(
    *,
    path: Path,
    expected_commit: str,
    nonce: str,
    timeout_seconds: int,
) -> None:
    if (
        not re.fullmatch(
            r"/tmp/leadpoet-gateway-miner-maintenance-handoff\.[A-Za-z0-9._-]+",
            str(path),
        )
        or not re.fullmatch(r"[0-9a-f]{64}", str(nonce))
        or not 1 <= int(timeout_seconds) <= 300
    ):
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance handoff request is invalid"
        )
    deadline = time.monotonic() + int(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            path.lstat()
        except FileNotFoundError:
            time.sleep(0.1)
            continue
        action = _read_handoff_marker(
            path=path,
            expected_commit=expected_commit,
            nonce=nonce,
        )
        if action == "cancel":
            raise GatewayMinerMaintenanceRestartError(
                "paired operator cancelled the miner-maintenance handoff"
            )
        if action != "continue":
            raise GatewayMinerMaintenanceRestartError(
                "paired operator did not authorize the miner-maintenance handoff"
            )
        return
    raise GatewayMinerMaintenanceRestartError(
        "paired operator did not provide a bounded miner-maintenance handoff"
    )


def _close_bootstrap_tree(path: Path) -> None:
    root = Path(path)
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance bootstrap tree is unavailable"
        ) from exc
    if (
        not re.fullmatch(
            r"/tmp/gateway-miner-maintenance-bootstrap\.[A-Za-z0-9]+",
            str(root),
        )
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance bootstrap tree identity is unsafe"
        )
    shutil.rmtree(root)
    if root.exists():
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance bootstrap tree was not removed"
        )


def _leave_and_close_bootstrap_tree(path: Path) -> None:
    """Remove the bootstrap tree without leaving the exec process in it."""

    try:
        os.chdir("/")
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance bootstrap working directory is unavailable"
        ) from exc
    _close_bootstrap_tree(path)


def _install_controller_bundle_memfds(
    controller_bundle: Mapping[str, Any],
) -> None:
    payloads = controller_bundle["payloads"]
    assignments = (
        ("wrapper", CONTROLLER_WRAPPER_FD_NUMBER),
        ("git_helper", CONTROLLER_GIT_HELPER_FD_NUMBER),
        ("exact_commit_helper", CONTROLLER_EXACT_COMMIT_HELPER_FD_NUMBER),
        ("memory_guard", CONTROLLER_MEMORY_GUARD_FD_NUMBER),
    )
    for name, descriptor in assignments:
        payload = payloads.get(name)
        if not isinstance(payload, bytes):
            raise GatewayMinerMaintenanceRestartError(
                "verified N-1 controller bundle is incomplete"
            )
        _seal_payload_at_fd_number(
            payload=payload,
            fd_number=descriptor,
            name="leadpoet-" + name.replace("_", "-"),
            max_bytes=4 * 1024 * 1024,
        )


def bootstrap_gateway_miner_maintenance_restart(
    *,
    repo_root: Path,
    candidate_root: Path,
    bootstrap_root: Path,
    plan_file: Path,
    expected_commit: str,
    controller_current: Path,
    host_restart_path: Path,
    handoff_file: Path,
    handoff_nonce: str,
    restart_invocation_id: str,
    secrets_client: Any = None,
    release_s3_client: Any = None,
) -> None:
    """Prepare, fence, and exec the immutable installed N-1 controller."""

    proof_fd_open = False
    controller_fds_open = False
    cleaned = False
    try:
        _require_canonical_restart_lock_fd()
        _require_reserved_memfd_numbers_available()
        _require_fixed_bootstrap_authority(os.environ)
        secrets_client, release_s3_client = _resolve_bootstrap_aws_clients(
            secrets_client=secrets_client,
            release_s3_client=release_s3_client,
        )
        prepared = prepare_gateway_miner_maintenance_restart(
            repo_root=repo_root,
            candidate_root=candidate_root,
            plan_file=plan_file,
            expected_commit=expected_commit,
            controller_current=controller_current,
            host_restart_path=host_restart_path,
            restart_invocation_id=restart_invocation_id,
            secrets_client=secrets_client,
            release_s3_client=release_s3_client,
        )
        proof = prepared["proof"]
        _seal_payload_at_fd_number(
            payload=_serialized_proof(proof),
            fd_number=PROOF_FD_NUMBER,
            name="leadpoet-miner-maintenance-proof",
            max_bytes=MAX_PROOF_BYTES,
        )
        proof_fd_open = True
        if _proof_from_fd(PROOF_FD_NUMBER) != proof:
            raise GatewayMinerMaintenanceRestartError(
                "sealed miner-maintenance invocation proof differs"
            )
        print(
            "Prepared exact-candidate miner maintenance under the canonical restart lock",
            flush=True,
        )
        _wait_for_handoff_marker(
            path=handoff_file,
            expected_commit=expected_commit,
            nonce=handoff_nonce,
            timeout_seconds=300,
        )
        _require_canonical_restart_lock_fd()
        final_tree = _validate_candidate_identity(
            repo_root=repo_root,
            candidate_root=candidate_root,
            plan_file=plan_file,
            expected_commit=expected_commit,
            controller_current=controller_current,
            host_restart_path=host_restart_path,
            reconcile_host_wrapper=True,
        )
        _verify_protected_source()
        secrets_client, release_s3_client = _resolve_bootstrap_aws_clients(
            secrets_client=secrets_client,
            release_s3_client=release_s3_client,
        )
        _verify_proof_against_state(
            proof=_proof_from_fd(PROOF_FD_NUMBER),
            deploy_commit=expected_commit,
            candidate_tree_hash=str(final_tree["tree_hash"]),
            gateway_release_hash=str(prepared["gateway_release_hash"]),
            client=secrets_client,
            release_s3_client=release_s3_client,
            tree_evidence=final_tree,
            restart_invocation_id=restart_invocation_id,
            live_process_commitment=(
                _pre_hydration_live_process_commitment(final_tree)
            ),
        )
        _install_controller_bundle_memfds(final_tree["controller_bundle"])
        controller_fds_open = True
        for descriptor in (
            PROOF_FD_NUMBER,
            CONTROLLER_WRAPPER_FD_NUMBER,
            CONTROLLER_GIT_HELPER_FD_NUMBER,
            CONTROLLER_EXACT_COMMIT_HELPER_FD_NUMBER,
            CONTROLLER_MEMORY_GUARD_FD_NUMBER,
        ):
            os.set_inheritable(descriptor, True)
        _require_canonical_restart_lock_fd()
        _leave_and_close_bootstrap_tree(bootstrap_root)
        cleaned = True
        environment = dict(os.environ)
        environment.update(
            {
                PROOF_FD_ENV_NAME: str(PROOF_FD_NUMBER),
                "GATEWAY_GIT_HELPER": (
                    f"/proc/self/fd/{CONTROLLER_GIT_HELPER_FD_NUMBER}"
                ),
                "GATEWAY_EXACT_COMMIT_HELPER": (
                    f"/proc/self/fd/{CONTROLLER_EXACT_COMMIT_HELPER_FD_NUMBER}"
                ),
                "GATEWAY_HOST_MEMORY_GUARD_PATH": (
                    f"/proc/self/fd/{CONTROLLER_MEMORY_GUARD_FD_NUMBER}"
                ),
                "LEADPOET_GATEWAY_ENV_SECRET_ID": GATEWAY_SECRET_ID,
                "GATEWAY_V2_RELEASE_BUCKET": DEFAULT_RELEASE_BUCKET,
                "RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET": (
                    DEFAULT_RELEASE_BUCKET
                ),
                "GATEWAY_V2_RELEASE_PREFIX": DEFAULT_RELEASE_PREFIX,
                "AWS_REGION": EXPECTED_AWS_REGION,
                "AWS_DEFAULT_REGION": EXPECTED_AWS_REGION,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GATEWAY_RESTART_LOCK_HELD": "1",
                "GATEWAY_RESTART_PHASE": "prepare",
            }
        )
        for name in (
            "GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_PLAN",
            "GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT",
            "GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE",
            "GATEWAY_MINER_MAINTENANCE_HANDOFF_NONCE",
        ):
            environment.pop(name, None)
        os.execve(
            "/bin/bash",
            [
                "bash",
                f"/proc/self/fd/{CONTROLLER_WRAPPER_FD_NUMBER}",
                "--commit",
                str(expected_commit),
            ],
            environment,
        )
    finally:
        if not cleaned:
            try:
                _close_bootstrap_tree(bootstrap_root)
            except GatewayMinerMaintenanceRestartError:
                pass
        if controller_fds_open:
            for descriptor in (
                CONTROLLER_WRAPPER_FD_NUMBER,
                CONTROLLER_GIT_HELPER_FD_NUMBER,
                CONTROLLER_EXACT_COMMIT_HELPER_FD_NUMBER,
                CONTROLLER_MEMORY_GUARD_FD_NUMBER,
            ):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if proof_fd_open:
            try:
                os.close(PROOF_FD_NUMBER)
            except OSError:
                pass


def _fetch_runtime_status(
    *,
    url: str = DEFAULT_RUNTIME_STATUS_URL,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    if url != DEFAULT_RUNTIME_STATUS_URL:
        raise GatewayMinerMaintenanceRestartError(
            "running Research Lab status URL is not canonical"
        )
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        8000,
        timeout=timeout_seconds,
    )
    try:
        connection.request(
            "GET",
            "/research-lab/status",
            body=None,
            headers={"Host": "127.0.0.1:8000", "Connection": "close"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise GatewayMinerMaintenanceRestartError(
                "running Research Lab status response is not successful"
            )
        content_length = response.getheader("Content-Length")
        if content_length is not None and int(content_length) > MAX_RUNTIME_STATUS_BYTES:
            raise GatewayMinerMaintenanceRestartError(
                "running Research Lab status is too large"
            )
        payload = response.read(MAX_RUNTIME_STATUS_BYTES + 1)
    except GatewayMinerMaintenanceRestartError:
        raise
    except Exception as exc:
        raise GatewayMinerMaintenanceRestartError(
            "running Research Lab status is unavailable"
        ) from exc
    finally:
        connection.close()
    if len(payload) > MAX_RUNTIME_STATUS_BYTES:
        raise GatewayMinerMaintenanceRestartError(
            "running Research Lab status is too large"
        )
    return _load_json_bytes(payload, label="running Research Lab status")


def _require_runtime_miner_disabled(runtime_status: Mapping[str, Any]) -> None:
    if runtime_status.get("miner_submissions_enabled") is not False:
        raise GatewayMinerMaintenanceRestartError(
            "running gateway has miner submissions enabled"
        )


def verify_gateway_miner_maintenance_runtime_state(
    *,
    deploy_commit: str,
    candidate_tree_hash: str,
    gateway_release_hash: str,
    runtime_environment: Mapping[str, str],
    runtime_status: Mapping[str, Any],
    secrets_client: Any = None,
    release_s3_client: Any = None,
    hydrated_environment_path: Path = CANONICAL_GATEWAY_ENV_PATH,
) -> dict[str, str]:
    """Recheck the exact false state against the activated live runtime."""

    _require_fixed_bootstrap_authority(runtime_environment)
    _require_disabled_parent_environment(runtime_environment)
    _require_runtime_miner_disabled(runtime_status)
    result = verify_gateway_miner_maintenance_state(
        deploy_commit=deploy_commit,
        candidate_tree_hash=candidate_tree_hash,
        gateway_release_hash=gateway_release_hash,
        parent_environment=runtime_environment,
        secrets_client=secrets_client,
        release_s3_client=release_s3_client,
        bind_live_process_to_proof=False,
        hydrated_environment_path=hydrated_environment_path,
    )
    return {**result, "runtime_status": "disabled"}


def _active_tree_hash(repo_root: Path, expected_commit: str) -> str:
    commit = str(expected_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayMinerMaintenanceRestartError("candidate commit is invalid")
    if _run_git(repo_root, "rev-parse", "HEAD") != commit:
        raise GatewayMinerMaintenanceRestartError(
            "activated checkout differs from the expected candidate"
        )
    return _run_git(repo_root, "rev-parse", f"{commit}^{{tree}}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bootstrap-exec", action="store_true")
    mode.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/ec2-user/leadpoet_repo"),
    )
    parser.add_argument(
        "--controller-current",
        type=Path,
        default=Path(
            "/home/ec2-user/.config/leadpoet/restart-controller/gateway/current"
        ),
    )
    parser.add_argument(
        "--host-restart-path",
        type=Path,
        default=Path("/home/ec2-user/gw_restart.sh"),
    )
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--bootstrap-root", type=Path)
    parser.add_argument("--handoff-file", type=Path)
    parser.add_argument("--handoff-nonce")
    parser.add_argument("--release-manifest", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        expected_commit = str(args.expected_commit).lower()
        if args.bootstrap_exec:
            if (
                args.plan_file is None
                or args.bootstrap_root is None
                or args.handoff_file is None
                or args.handoff_nonce is None
                or args.release_manifest is not None
            ):
                parser.error(
                    "--bootstrap-exec requires plan, bootstrap root, handoff file, and nonce only"
                )
            restart_invocation_id = str(
                os.environ.get("GATEWAY_RESTART_INVOCATION_ID") or ""
            )
            bootstrap_gateway_miner_maintenance_restart(
                repo_root=args.repo_root,
                candidate_root=_candidate_root(),
                bootstrap_root=args.bootstrap_root,
                plan_file=args.plan_file,
                expected_commit=expected_commit,
                controller_current=args.controller_current,
                host_restart_path=args.host_restart_path,
                handoff_file=args.handoff_file,
                handoff_nonce=str(args.handoff_nonce),
                restart_invocation_id=restart_invocation_id,
            )
            raise GatewayMinerMaintenanceRestartError(
                "installed N-1 controller exec returned unexpectedly"
            )
        if (
            args.release_manifest is None
            or args.plan_file is not None
            or args.bootstrap_root is not None
            or args.handoff_file is not None
            or args.handoff_nonce is not None
        ):
            parser.error("--verify-runtime requires --release-manifest only")
        _verify_protected_source()
        release = validate_release_manifest(
            _load_json(
                args.release_manifest,
                label="activated gateway release manifest",
                max_bytes=4 * 1024 * 1024,
            )
        )
        if release["commit_sha"] != expected_commit:
            raise GatewayMinerMaintenanceRestartError(
                "activated gateway release is for another candidate"
            )
        result = verify_gateway_miner_maintenance_runtime_state(
            deploy_commit=expected_commit,
            candidate_tree_hash=_active_tree_hash(
                args.repo_root, expected_commit
            ),
            gateway_release_hash=release["release_hash"],
            runtime_environment=os.environ,
            runtime_status=_fetch_runtime_status(),
        )
    except (
        GatewayMinerMaintenanceRestartError,
        GatewayMinerSubmissionsDisableError,
    ) as exc:
        print(
            json.dumps(
                {"status": "failed_closed", "error": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    except Exception:
        print(
            '{"error":"unexpected miner-maintenance restart failure",'
            '"status":"failed_closed"}',
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
