"""Disable production Research Lab miner submissions without editing other state.

This operation has one fixed target: the production gateway environment secret's
``RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED`` setting.  Verification is read-only.
Applying requires the exact previously observed ``AWSCURRENT`` version, stages a
candidate under one unique operation-owned label, verifies its bytes and the
complete version-label topology, and only then moves ``AWSCURRENT`` with a
version fence. The prior Secrets Manager version and label topology are the
backup; secret contents are never written to local storage or output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence
import uuid


GATEWAY_SECRET_ID = "leadpoet/prod/gateway/env"
EXPECTED_AWS_ACCOUNT_ID = "493765492819"
EXPECTED_AWS_REGION = "us-east-1"
EXPECTED_GATEWAY_ROLE_NAME = "leadpoet-gateway-s3-cloudwatch-role"
TARGET_ENV_NAME = "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED"
TARGET_ENV_VALUE = "false"
# One known unrelated legacy shell key occurs exactly twice.  Its parsed raw
# assignment and semantic value must match; rendering preserves both records.
_LEGACY_IDENTICAL_SHELL_DUPLICATE_NAME = (
    "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED"
)
# Exact 0dd keeps this legacy pair in the canonical cache commitment, then its
# ENV_SECRET renderer strips both.  Process instance-role checks still reject it.
_LEGACY_PRESERVED_AWS_ENV_NAMES = frozenset(
    {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
)
RECOVERY_JOURNAL_SCHEMA_VERSION = "leadpoet.gateway_miner_disable_transaction.v1"
DEFAULT_RECOVERY_JOURNAL_PATH = Path(
    "/home/ec2-user/.config/leadpoet/gateway-miner-disable-transaction-v1.json"
)
MAX_RECOVERY_JOURNAL_BYTES = 64 * 1024
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_VERSION_ID_RE = re.compile(r"^[A-Za-z0-9-]{32,64}$")
_CUSTOM_STAGE_PREFIX = "LEADPOET_MINER_DISABLE_"
_CUSTOM_STAGE_RE = re.compile(r"^LEADPOET_MINER_DISABLE_[0-9a-f]{32}$")
_STAGE_LABEL_RE = re.compile(r"^[A-Za-z0-9_+=.@-]{1,256}$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_RESTART_AUTHORITY_NAMES = frozenset(
    {
        "GATEWAY_MINER_MAINTENANCE_PROOF_FD",
        "GATEWAY_GIT_HELPER",
        "GATEWAY_EXACT_COMMIT_HELPER",
        "GATEWAY_HOST_MEMORY_GUARD_PATH",
    }
)
_N_MINUS_ONE_HYDRATION_SKIP_NAMES = frozenset(
    {
        "GATEWAY_DEPLOY_COMMIT",
        "GATEWAY_ENV_FILE",
        "GATEWAY_PRIVATE_KEY_PATH",
        "ARWEAVE_KEYFILE_PATH",
        "GATEWAY_RESTART_GIT_SSH_COMMAND",
        "GATEWAY_PYTHON_BIN",
        "GATEWAY_RESTART_CONTROLLER_ROOT",
        "GATEWAY_RESTART_RECOVERY_LOCK_FILE",
        "GATEWAY_RESTART_INVOCATION_ID",
        "GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST",
        "GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT",
        "GATEWAY_V2_ARTIFACT_POLICY",
        "GATEWAY_V2_CONFIG_DIR",
        "GATEWAY_V2_DEFER_WORKER_FLEETS",
        "GATEWAY_V2_KMS_KEY_ID",
        "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
        "GATEWAY_V2_RELEASE_BUCKET",
        "GATEWAY_V2_RELEASE_ARCHIVE_ROOT",
        "GATEWAY_V2_RELEASE_LINEAGE",
        "GATEWAY_V2_RELEASE_MANIFEST",
        "GATEWAY_V2_RELEASE_PREFIX",
        "GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS",
        "GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS",
        "GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES",
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        "LEADPOET_GATEWAY_ENV_SECRET_ID",
        "LEADPOET_RESTART_INVOCATION_ID",
        "LEADPOET_SENTRY_API_TOKEN",
        "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT",
    }
)
_EC2_ROLE_ARN_RE = re.compile(
    rf"^arn:aws:sts::{EXPECTED_AWS_ACCOUNT_ID}:assumed-role/"
    rf"{re.escape(EXPECTED_GATEWAY_ROLE_NAME)}/i-[0-9a-f]+$"
)
_FORBIDDEN_AWS_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONFIG_FILE",
        "AWS_CA_BUNDLE",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL_STS",
        "AWS_ENDPOINT_URL_SECRETSMANAGER",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
        "AWS_METADATA_SERVICE_TIMEOUT",
        "AWS_METADATA_SERVICE_NUM_ATTEMPTS",
        "BOTO_CONFIG",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    }
)
_EXPECTED_AWS_ENDPOINTS = {
    "s3": frozenset(
        {
            "https://s3.amazonaws.com",
            "https://s3.us-east-1.amazonaws.com",
        }
    ),
    "secretsmanager": frozenset(
        {"https://secretsmanager.us-east-1.amazonaws.com"}
    ),
    "sts": frozenset({"https://sts.us-east-1.amazonaws.com"}),
}


class GatewayMinerSubmissionsDisableError(RuntimeError):
    """The fixed safety setting could not be changed without losing state."""


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for raw_name, value in pairs:
        name = str(raw_name)
        if name in decoded:
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret JSON contains duplicate names"
            )
        decoded[name] = value
    return decoded


def _decode_shell_target_value(raw_value: str) -> str:
    try:
        parts = shlex.split("VALUE=" + raw_value, posix=True)
    except ValueError as exc:
        raise GatewayMinerSubmissionsDisableError(
            "gateway miner-submission setting is malformed"
        ) from exc
    if len(parts) != 1 or not parts[0].startswith("VALUE="):
        raise GatewayMinerSubmissionsDisableError(
            "gateway miner-submission setting is malformed"
        )
    return parts[0].split("=", 1)[1]


def _decode_legacy_duplicate_value(raw_value: str) -> str:
    try:
        parts = shlex.split("VALUE=" + raw_value, posix=True)
    except ValueError as exc:
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret environment is malformed"
        ) from exc
    if len(parts) != 1 or not parts[0].startswith("VALUE="):
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret environment is malformed"
        )
    return parts[0].split("=", 1)[1]


def _shell_records(raw: str) -> tuple[list[tuple[str, str]], str]:
    if "\x00" in raw:
        if "\n" in raw or "\r" in raw:
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret environment format is unsupported"
            )
        delimiter = "\x00"
        pieces = raw.split(delimiter)
        records = [
            (piece, delimiter if index < len(pieces) - 1 else "")
            for index, piece in enumerate(pieces)
        ]
        return records, delimiter

    separators = re.findall(r"\r\n|\n|\r", raw)
    if len(set(separators)) > 1:
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret environment format is unsupported"
        )
    delimiter = separators[0] if separators else "\n"
    pieces = re.split(r"(\r\n|\n|\r)", raw)
    records = []
    for index in range(0, len(pieces), 2):
        record = pieces[index]
        separator = pieces[index + 1] if index + 1 < len(pieces) else ""
        records.append((record, separator))
    return records, delimiter


def _parse_shell_environment(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    raw_values: dict[str, str] = {}
    occurrences: dict[str, int] = {}
    for raw_record, _separator in _shell_records(raw)[0]:
        record = raw_record.strip()
        if not record or record.startswith("#"):
            continue
        candidate = record
        if candidate.startswith("export "):
            candidate = candidate[len("export ") :].lstrip()
        if "=" not in candidate:
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret environment is malformed"
            )
        name, raw_value = candidate.split("=", 1)
        name = name.strip()
        if not _ENVIRONMENT_NAME_RE.fullmatch(name):
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret environment contains an invalid name"
            )
        value = (
            _decode_shell_target_value(raw_value.strip())
            if name == TARGET_ENV_NAME
            else (
                _decode_legacy_duplicate_value(raw_value.strip())
                if name == _LEGACY_IDENTICAL_SHELL_DUPLICATE_NAME
                else raw_value
            )
        )
        if name in values:
            if (
                name != _LEGACY_IDENTICAL_SHELL_DUPLICATE_NAME
                or occurrences[name] != 1
                or raw_values[name] != raw_value
                or values[name] != value
            ):
                raise GatewayMinerSubmissionsDisableError(
                    "gateway secret environment contains duplicate names"
                )
            occurrences[name] += 1
            continue
        values[name] = value
        raw_values[name] = raw_value
        occurrences[name] = 1
    if not values:
        raise GatewayMinerSubmissionsDisableError("gateway secret environment is empty")
    return values


def _parse_environment(raw: str) -> tuple[dict[str, Any], str]:
    if not raw:
        raise GatewayMinerSubmissionsDisableError("gateway secret environment is empty")
    stripped = raw.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            decoded = json.loads(
                raw,
                object_pairs_hook=_json_object_without_duplicates,
            )
        except json.JSONDecodeError as exc:
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret JSON is malformed"
            ) from exc
        if not isinstance(decoded, Mapping) or not decoded:
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret JSON must contain a nonempty object"
            )
        for raw_name in decoded:
            if not _ENVIRONMENT_NAME_RE.fullmatch(str(raw_name)):
                raise GatewayMinerSubmissionsDisableError(
                    "gateway secret JSON contains an invalid name"
                )
        return dict(decoded), "json"
    return _parse_shell_environment(raw), "shell"


def _render_shell_environment(raw: str) -> str:
    records, delimiter = _shell_records(raw)
    rendered: list[str] = []
    replaced = False
    for record, separator in records:
        candidate = record.strip()
        if candidate.startswith("export "):
            candidate = candidate[len("export ") :].lstrip()
        name = candidate.split("=", 1)[0].strip() if "=" in candidate else ""
        if name == TARGET_ENV_NAME:
            if replaced:
                raise GatewayMinerSubmissionsDisableError(
                    "gateway secret environment contains duplicate names"
                )
            rendered.append(f"{TARGET_ENV_NAME}={TARGET_ENV_VALUE}{separator}")
            replaced = True
        else:
            rendered.append(record + separator)
    candidate_secret = "".join(rendered)
    if not replaced:
        if candidate_secret and not candidate_secret.endswith(("\n", "\r", "\x00")):
            candidate_secret += delimiter
        candidate_secret += f"{TARGET_ENV_NAME}={TARGET_ENV_VALUE}{delimiter}"
    return candidate_secret


def _render_environment(raw: str, *, document_format: str) -> str:
    if document_format == "json":
        decoded, observed_format = _parse_environment(raw)
        if observed_format != "json":
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret format changed during rendering"
            )
        decoded[TARGET_ENV_NAME] = TARGET_ENV_VALUE
        return json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    if document_format == "shell":
        return _render_shell_environment(raw)
    raise GatewayMinerSubmissionsDisableError(
        "gateway secret environment format is unsupported"
    )


def _secret_string(response: Mapping[str, Any]) -> str:
    value = response.get("SecretString")
    if not isinstance(value, str):
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret is not stored as text"
        )
    return value


def _version_id(response: Mapping[str, Any]) -> str:
    version_id = str(response.get("VersionId") or "")
    if not _VERSION_ID_RE.fullmatch(version_id):
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret version identity is invalid"
        )
    return version_id


def _version_stages(secrets_client: Any) -> dict[str, frozenset[str]]:
    try:
        description = secrets_client.describe_secret(SecretId=GATEWAY_SECRET_ID)
    except Exception as exc:
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret version stages are unavailable"
        ) from exc
    raw_stages = description.get("VersionIdsToStages")
    if not isinstance(raw_stages, Mapping):
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret version stages are invalid"
        )
    stages: dict[str, frozenset[str]] = {}
    label_holders: dict[str, str] = {}
    for raw_version, raw_labels in raw_stages.items():
        version = str(raw_version)
        if not _VERSION_ID_RE.fullmatch(version) or not isinstance(raw_labels, list):
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret version stages are invalid"
            )
        labels = frozenset(str(label) for label in raw_labels)
        if len(labels) != len(raw_labels) or any(
            not _STAGE_LABEL_RE.fullmatch(label) for label in labels
        ):
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret version stages are invalid"
            )
        for label in labels:
            if label in label_holders:
                raise GatewayMinerSubmissionsDisableError(
                    "gateway secret version stage is attached more than once"
                )
            label_holders[label] = version
        if labels:
            stages[version] = labels
    return stages


def _stage_holders(
    stages: Mapping[str, frozenset[str]],
    label: str,
) -> list[str]:
    return sorted(version for version, labels in stages.items() if label in labels)


def _require_unique_current(
    stages: Mapping[str, frozenset[str]],
    version_id: str,
) -> None:
    if _stage_holders(stages, "AWSCURRENT") != [version_id]:
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret current version changed or is ambiguous"
        )


def _validate_initial_topology(
    stages: Mapping[str, frozenset[str]],
    current_version_id: str,
) -> None:
    _require_unique_current(stages, current_version_id)
    previous_versions = _stage_holders(stages, "AWSPREVIOUS")
    if len(previous_versions) > 1 or previous_versions == [current_version_id]:
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret previous-version topology is invalid"
        )
    if any(
        label.startswith(_CUSTOM_STAGE_PREFIX)
        for labels in stages.values()
        for label in labels
    ):
        raise GatewayMinerSubmissionsDisableError(
            "another fixed-purpose miner disable operation is staged"
        )


def _read_current_secret(
    secrets_client: Any,
) -> tuple[str, str, dict[str, frozenset[str]]]:
    try:
        response = secrets_client.get_secret_value(
            SecretId=GATEWAY_SECRET_ID,
            VersionStage="AWSCURRENT",
        )
    except Exception as exc:
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret current version is unavailable"
        ) from exc
    version_id = _version_id(response)
    secret = _secret_string(response)
    stages = _version_stages(secrets_client)
    _require_unique_current(stages, version_id)
    return version_id, secret, stages


def _read_exact_secret(secrets_client: Any, version_id: str) -> str:
    try:
        response = secrets_client.get_secret_value(
            SecretId=GATEWAY_SECRET_ID,
            VersionId=version_id,
        )
    except Exception as exc:
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret exact version is unavailable"
        ) from exc
    if _version_id(response) != version_id:
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret exact version identity differs"
        )
    return _secret_string(response)


def _document_commitment(raw: str) -> str:
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _n_minus_one_hydrated_environment(raw: str) -> str:
    """Render the exact installed-0dd gateway environment cache bytes.

    Frozen 0dd reads ``SecretString`` through AWS CLI ``--output text``.  The
    CLI writes the scalar followed by one transport newline before the frozen
    renderer reads ``SECRET_TMP``.  Model that byte explicitly so a shell
    secret which already ends in a delimiter retains the same final blank
    record as the installed renderer.
    """

    transported_raw = raw + "\n"

    try:
        parsed = json.loads(transported_raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        lines: list[str] = []
        for key, value in parsed.items():
            if key in _N_MINUS_ONE_HYDRATION_SKIP_NAMES:
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, separators=(",", ":"))
            elif value is None:
                value = ""
            lines.append(f"{key}={value}")
        return "\n".join(lines) + "\n"

    lines = []
    for raw_line in transported_raw.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        candidate = line[7:].strip() if line.startswith("export ") else line
        try:
            parts = shlex.split(candidate, posix=True)
        except ValueError:
            parts = [candidate]
        assignment = parts[0] if len(parts) == 1 else candidate
        key = assignment.split("=", 1)[0].strip() if "=" in assignment else ""
        if key in _N_MINUS_ONE_HYDRATION_SKIP_NAMES:
            continue
        lines.append(raw_line)
    return "\n".join(lines) + ("\n" if lines else "")


def _n_minus_one_hydrated_environment_commitment(raw: str) -> str:
    return _document_commitment(_n_minus_one_hydrated_environment(raw))


def _topology_commitment(stages: Mapping[str, frozenset[str]]) -> str:
    document = {version: sorted(labels) for version, labels in sorted(stages.items())}
    return _document_commitment(
        json.dumps(document, sort_keys=True, separators=(",", ":"))
    )


def _recovery_journal_body(
    *,
    prior_version_id: str,
    candidate_version_id: str,
    custom_stage_label: str,
    initial_topology: Mapping[str, frozenset[str]],
    prior_document_commitment: str,
    candidate_document_commitment: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_JOURNAL_SCHEMA_VERSION,
        "secret_id": GATEWAY_SECRET_ID,
        "prior_version_id": prior_version_id,
        "candidate_version_id": candidate_version_id,
        "custom_stage_label": custom_stage_label,
        "initial_topology": {
            version: sorted(labels)
            for version, labels in sorted(initial_topology.items())
        },
        "prior_document_commitment": prior_document_commitment,
        "candidate_document_commitment": candidate_document_commitment,
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _validate_recovery_journal(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "secret_id",
        "prior_version_id",
        "candidate_version_id",
        "custom_stage_label",
        "initial_topology",
        "prior_document_commitment",
        "candidate_document_commitment",
        "created_at",
        "journal_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery journal fields are invalid"
        )
    body = {field: value[field] for field in fields if field != "journal_hash"}
    expected_hash = _document_commitment(
        json.dumps(body, sort_keys=True, separators=(",", ":"))
    )
    if (
        value.get("schema_version") != RECOVERY_JOURNAL_SCHEMA_VERSION
        or value.get("secret_id") != GATEWAY_SECRET_ID
        or not _VERSION_ID_RE.fullmatch(str(value.get("prior_version_id") or ""))
        or not _VERSION_ID_RE.fullmatch(str(value.get("candidate_version_id") or ""))
        or not _CUSTOM_STAGE_RE.fullmatch(str(value.get("custom_stage_label") or ""))
        or str(value.get("custom_stage_label"))
        != _custom_stage_label(str(value.get("candidate_version_id")))
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("prior_document_commitment") or ""),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("candidate_document_commitment") or ""),
        )
        or not str(value.get("created_at") or "").endswith("Z")
        or value.get("journal_hash") != expected_hash
    ):
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery journal commitments are invalid"
        )
    raw_topology = value.get("initial_topology")
    if not isinstance(raw_topology, Mapping):
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery topology is invalid"
        )
    topology: dict[str, frozenset[str]] = {}
    for raw_version, raw_labels in raw_topology.items():
        version = str(raw_version)
        if (
            not _VERSION_ID_RE.fullmatch(version)
            or not isinstance(raw_labels, list)
        ):
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery topology is invalid"
            )
        labels = frozenset(str(label) for label in raw_labels)
        if len(labels) != len(raw_labels) or any(
            not _STAGE_LABEL_RE.fullmatch(label) for label in labels
        ):
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery topology is invalid"
            )
        if labels:
            topology[version] = labels
    _validate_initial_topology(topology, str(value["prior_version_id"]))
    return {**dict(value), "initial_topology": topology}


@contextmanager
def _open_recovery_journal_parent_fd(path: Path) -> Iterator[tuple[int, str]]:
    candidate = Path(path).absolute()
    parts = candidate.parts
    if (
        not candidate.is_absolute()
        or len(parts) < 2
        or candidate.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in parts[1:])
    ):
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery journal path is invalid"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[tuple[int, Optional[int], Optional[str], tuple[int, int]]] = []
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
            try:
                descriptor = os.open(part, flags, dir_fd=parent_descriptor)
            except FileNotFoundError:
                os.mkdir(part, 0o700, dir_fd=parent_descriptor)
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
                raise GatewayMinerSubmissionsDisableError(
                    "miner-disable recovery journal ancestry is unsafe"
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
        for descriptor, parent_descriptor, name, identity in descriptors[1:]:
            assert parent_descriptor is not None and name is not None
            current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != identity
                or (opened.st_dev, opened.st_ino) != identity
            ):
                raise GatewayMinerSubmissionsDisableError(
                    "miner-disable recovery journal ancestry changed"
                )
    except GatewayMinerSubmissionsDisableError:
        raise
    except OSError as exc:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery journal ancestry is unavailable"
        ) from exc
    finally:
        for descriptor, _parent, _name, _identity in reversed(descriptors):
            os.close(descriptor)


def _ensure_recovery_journal_parent(path: Path) -> None:
    with _open_recovery_journal_parent_fd(path):
        return


def _write_recovery_journal(
    path: Path,
    body: Mapping[str, Any],
    *,
    parent_authority: Optional[tuple[int, str]] = None,
) -> None:
    if parent_authority is None:
        with _open_recovery_journal_parent_fd(path) as authority:
            _write_recovery_journal(path, body, parent_authority=authority)
        return
    parent_fd, leaf_name = parent_authority
    document = {
        **dict(body),
        "journal_hash": _document_commitment(
            json.dumps(dict(body), sort_keys=True, separators=(",", ":"))
        ),
    }
    _validate_recovery_journal(document)
    try:
        os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery journal already exists"
        )
    temporary_name = f".{leaf_name}.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        payload = (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        if len(payload) > MAX_RECOVERY_JOURNAL_BYTES:
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery journal is too large"
            )
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
        if os.write(descriptor, payload) != len(payload):
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery journal write was incomplete"
            )
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary_name,
                leaf_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery journal already exists"
            ) from exc
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass


def _read_recovery_journal(
    path: Path,
    *,
    parent_authority: Optional[tuple[int, str]] = None,
) -> tuple[dict[str, Any], tuple[int, int]]:
    if parent_authority is None:
        with _open_recovery_journal_parent_fd(path) as authority:
            return _read_recovery_journal(path, parent_authority=authority)
    parent_fd, leaf_name = parent_authority
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(leaf_name, flags, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        current = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
            or not 2 <= opened.st_size <= MAX_RECOVERY_JOURNAL_BYTES
        ):
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery journal identity is unsafe"
            )
        payload = os.read(descriptor, MAX_RECOVERY_JOURNAL_BYTES + 1)
        final = os.fstat(descriptor)
        if (
            final.st_dev != opened.st_dev
            or final.st_ino != opened.st_ino
            or final.st_size != opened.st_size
            or len(payload) != opened.st_size
        ):
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery journal changed while reading"
            )
    except OSError as exc:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery journal is unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        value = json.loads(payload, object_pairs_hook=_json_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery journal is invalid"
        ) from exc
    return _validate_recovery_journal(value), (opened.st_dev, opened.st_ino)


def _remove_recovery_journal(
    path: Path,
    identity: tuple[int, int],
    *,
    parent_authority: Optional[tuple[int, str]] = None,
) -> None:
    if parent_authority is None:
        with _open_recovery_journal_parent_fd(path) as authority:
            _remove_recovery_journal(
                path,
                identity,
                parent_authority=authority,
            )
        return
    parent_fd, leaf_name = parent_authority
    try:
        current = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != identity:
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery journal changed before removal"
            )
        os.unlink(leaf_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery journal could not be removed"
        ) from exc


def _validated_candidate(initial_secret: str) -> tuple[str, str, str]:
    initial_environment, document_format = _parse_environment(initial_secret)
    forbidden_aws_names = _FORBIDDEN_AWS_ENV_NAMES & set(initial_environment)
    # This shell-only exception preserves bytes for the exact 0dd cache; it does
    # not authorize either name in the helper process or rendered runtime env.
    legacy_preserved_aws_pair = (
        document_format == "shell"
        and forbidden_aws_names == _LEGACY_PRESERVED_AWS_ENV_NAMES
    )
    if (
        _FORBIDDEN_RESTART_AUTHORITY_NAMES & set(initial_environment)
        or (forbidden_aws_names and not legacy_preserved_aws_pair)
    ):
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret contains restart or AWS authority"
        )
    if any(
        name in initial_environment
        and str(initial_environment[name]) != EXPECTED_AWS_REGION
        for name in ("AWS_REGION", "AWS_DEFAULT_REGION")
    ) or (
        "LEADPOET_AWS_INSTANCE_ROLE_ONLY" in initial_environment
        and str(initial_environment["LEADPOET_AWS_INSTANCE_ROLE_ONLY"]).lower()
        != "true"
    ):
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret contains conflicting AWS authority"
        )
    target_present = TARGET_ENV_NAME in initial_environment
    raw_current = initial_environment.get(TARGET_ENV_NAME)
    if target_present and not isinstance(raw_current, str):
        raise GatewayMinerSubmissionsDisableError(
            "gateway miner-submission setting must be text"
        )
    current_value = str(raw_current).strip().lower() if target_present else None
    if current_value is not None and current_value not in _TRUE_VALUES | _FALSE_VALUES:
        raise GatewayMinerSubmissionsDisableError(
            "gateway miner-submission setting has an unknown boolean value"
        )
    if current_value == TARGET_ENV_VALUE:
        return initial_secret, document_format, "already_disabled"

    candidate_secret = _render_environment(
        initial_secret,
        document_format=document_format,
    )
    candidate_environment, candidate_format = _parse_environment(candidate_secret)
    if candidate_format != document_format:
        raise GatewayMinerSubmissionsDisableError(
            "candidate gateway secret format differs"
        )
    before_unrelated = {
        name: value
        for name, value in initial_environment.items()
        if name != TARGET_ENV_NAME
    }
    after_unrelated = {
        name: value
        for name, value in candidate_environment.items()
        if name != TARGET_ENV_NAME
    }
    if before_unrelated != after_unrelated:
        raise GatewayMinerSubmissionsDisableError(
            "candidate gateway secret changes unrelated values"
        )
    if candidate_environment.get(TARGET_ENV_NAME) != TARGET_ENV_VALUE:
        raise GatewayMinerSubmissionsDisableError(
            "candidate gateway secret does not disable miner submissions"
        )
    if candidate_secret == initial_secret:
        raise GatewayMinerSubmissionsDisableError(
            "candidate gateway secret did not change the target setting"
        )
    return candidate_secret, document_format, "verified"


def _custom_stage_label(candidate_version_id: str) -> str:
    label = _CUSTOM_STAGE_PREFIX + candidate_version_id.replace("-", "")
    if not _CUSTOM_STAGE_RE.fullmatch(label):
        raise GatewayMinerSubmissionsDisableError(
            "candidate custom stage identity is invalid"
        )
    return label


def _expected_staged_topology(
    initial_topology: Mapping[str, frozenset[str]],
    *,
    candidate_version_id: str,
    custom_stage_label: str,
) -> dict[str, frozenset[str]]:
    expected = dict(initial_topology)
    expected[candidate_version_id] = frozenset({custom_stage_label})
    return expected


def _expected_promoted_topology(
    initial_topology: Mapping[str, frozenset[str]],
    *,
    prior_version_id: str,
    candidate_version_id: str,
    custom_stage_label: str,
    include_custom_stage: bool,
) -> dict[str, frozenset[str]]:
    expected: dict[str, frozenset[str]] = {}
    for version, labels in initial_topology.items():
        updated = set(labels)
        updated.discard("AWSPREVIOUS")
        if version == prior_version_id:
            updated.discard("AWSCURRENT")
            updated.add("AWSPREVIOUS")
        if updated:
            expected[version] = frozenset(updated)
    candidate_labels = {"AWSCURRENT"}
    if include_custom_stage:
        candidate_labels.add(custom_stage_label)
    expected[candidate_version_id] = frozenset(candidate_labels)
    return expected


def _remove_custom_stage(
    secrets_client: Any,
    *,
    candidate_version_id: str,
    custom_stage_label: str,
) -> dict[str, frozenset[str]]:
    stages = _version_stages(secrets_client)
    holders = _stage_holders(stages, custom_stage_label)
    if not holders:
        return stages
    if holders != [candidate_version_id]:
        raise GatewayMinerSubmissionsDisableError(
            "candidate custom stage ownership differs"
        )
    try:
        secrets_client.update_secret_version_stage(
            SecretId=GATEWAY_SECRET_ID,
            VersionStage=custom_stage_label,
            RemoveFromVersionId=candidate_version_id,
        )
    except Exception:
        # Resolve an ambiguous API response from authoritative stage readback.
        pass
    stages = _version_stages(secrets_client)
    if _stage_holders(stages, custom_stage_label):
        raise GatewayMinerSubmissionsDisableError(
            "candidate custom stage cleanup failed"
        )
    return stages


def _cleanup_candidate_and_verify_original_topology(
    secrets_client: Any,
    *,
    initial_topology: Mapping[str, frozenset[str]],
    candidate_version_id: str,
    custom_stage_label: str,
) -> None:
    stages = _remove_custom_stage(
        secrets_client,
        candidate_version_id=candidate_version_id,
        custom_stage_label=custom_stage_label,
    )
    if stages != dict(initial_topology):
        raise GatewayMinerSubmissionsDisableError(
            "candidate cleanup did not preserve the original stage topology"
        )


def _fail_before_promotion(
    secrets_client: Any,
    *,
    initial_topology: Mapping[str, frozenset[str]],
    candidate_version_id: str,
    custom_stage_label: str,
    public_message: str,
    cause: BaseException,
) -> NoReturn:
    try:
        _cleanup_candidate_and_verify_original_topology(
            secrets_client,
            initial_topology=initial_topology,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
        )
    except Exception as cleanup_exc:
        raise GatewayMinerSubmissionsDisableError(
            "candidate cleanup could not restore the original stage topology"
        ) from cleanup_exc
    raise GatewayMinerSubmissionsDisableError(public_message) from cause


def _restore_original_topology(
    secrets_client: Any,
    *,
    initial_topology: Mapping[str, frozenset[str]],
    prior_version_id: str,
    prior_secret: str,
    candidate_version_id: str,
    custom_stage_label: str,
) -> None:
    try:
        stages = _version_stages(secrets_client)
        current_versions = _stage_holders(stages, "AWSCURRENT")
        if current_versions == [candidate_version_id]:
            try:
                secrets_client.update_secret_version_stage(
                    SecretId=GATEWAY_SECRET_ID,
                    VersionStage="AWSCURRENT",
                    MoveToVersionId=prior_version_id,
                    RemoveFromVersionId=candidate_version_id,
                )
            except Exception:
                pass
        elif current_versions != [prior_version_id]:
            raise GatewayMinerSubmissionsDisableError(
                "rollback current-version ownership differs"
            )

        stages = _version_stages(secrets_client)
        _require_unique_current(stages, prior_version_id)
        original_previous = _stage_holders(initial_topology, "AWSPREVIOUS")
        observed_previous = _stage_holders(stages, "AWSPREVIOUS")
        if original_previous:
            target_previous = original_previous[0]
            if observed_previous != [target_previous]:
                remove_from = (
                    observed_previous[0] if len(observed_previous) == 1 else None
                )
                if len(observed_previous) > 1:
                    raise GatewayMinerSubmissionsDisableError(
                        "rollback previous-version ownership is ambiguous"
                    )
                kwargs: dict[str, str] = {
                    "SecretId": GATEWAY_SECRET_ID,
                    "VersionStage": "AWSPREVIOUS",
                    "MoveToVersionId": target_previous,
                }
                if remove_from:
                    kwargs["RemoveFromVersionId"] = remove_from
                try:
                    secrets_client.update_secret_version_stage(**kwargs)
                except Exception:
                    pass
        elif observed_previous:
            if len(observed_previous) > 1:
                raise GatewayMinerSubmissionsDisableError(
                    "rollback previous-version ownership is ambiguous"
                )
            try:
                secrets_client.update_secret_version_stage(
                    SecretId=GATEWAY_SECRET_ID,
                    VersionStage="AWSPREVIOUS",
                    RemoveFromVersionId=observed_previous[0],
                )
            except Exception:
                pass
    except Exception:
        # Custom-label cleanup below is safe even when concurrent standard-label
        # drift makes a full rollback impossible.
        pass

    try:
        _remove_custom_stage(
            secrets_client,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
        )
        restored_version, restored_secret, restored_topology = _read_current_secret(
            secrets_client
        )
    except Exception as exc:
        raise GatewayMinerSubmissionsDisableError(
            "automatic version-stage rollback failed"
        ) from exc
    if (
        restored_version != prior_version_id
        or restored_secret != prior_secret
        or restored_topology != dict(initial_topology)
    ):
        raise GatewayMinerSubmissionsDisableError(
            "automatic version-stage rollback could not be verified"
        )


def _nontransaction_topology(
    stages: Mapping[str, frozenset[str]],
    *,
    custom_stage_label: str,
) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    removed = {"AWSCURRENT", "AWSPREVIOUS", custom_stage_label}
    for version, labels in stages.items():
        remaining = frozenset(set(labels) - removed)
        if remaining:
            result[version] = remaining
    return result


def _set_previous_version(
    secrets_client: Any,
    *,
    target_version_id: str | None,
) -> None:
    stages = _version_stages(secrets_client)
    observed = _stage_holders(stages, "AWSPREVIOUS")
    if len(observed) > 1:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery previous-version ownership is ambiguous"
        )
    if target_version_id is None:
        if observed:
            try:
                secrets_client.update_secret_version_stage(
                    SecretId=GATEWAY_SECRET_ID,
                    VersionStage="AWSPREVIOUS",
                    RemoveFromVersionId=observed[0],
                )
            except Exception:
                pass
    elif observed != [target_version_id]:
        kwargs = {
            "SecretId": GATEWAY_SECRET_ID,
            "VersionStage": "AWSPREVIOUS",
            "MoveToVersionId": target_version_id,
        }
        if observed:
            kwargs["RemoveFromVersionId"] = observed[0]
        try:
            secrets_client.update_secret_version_stage(**kwargs)
        except Exception:
            pass
    final = _stage_holders(_version_stages(secrets_client), "AWSPREVIOUS")
    expected = [] if target_version_id is None else [target_version_id]
    if final != expected:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery previous-version restoration failed"
        )


def _recover_orphan_transaction(
    secrets_client: Any,
    *,
    recovery_journal_path: Path,
    recovery_journal_authority: Optional[tuple[int, str]] = None,
) -> None:
    path = Path(recovery_journal_path)
    if recovery_journal_authority is None:
        with _open_recovery_journal_parent_fd(path) as authority:
            _recover_orphan_transaction(
                secrets_client,
                recovery_journal_path=path,
                recovery_journal_authority=authority,
            )
        return
    parent_fd, leaf_name = recovery_journal_authority
    try:
        os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    journal, identity = _read_recovery_journal(
        path,
        parent_authority=recovery_journal_authority,
    )
    prior_version_id = str(journal["prior_version_id"])
    candidate_version_id = str(journal["candidate_version_id"])
    custom_stage_label = str(journal["custom_stage_label"])
    initial_topology = dict(journal["initial_topology"])
    stages = _version_stages(secrets_client)
    current_versions = _stage_holders(stages, "AWSCURRENT")
    if len(current_versions) != 1:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery current-version ownership is ambiguous"
        )
    current_version_id = current_versions[0]
    prior_secret = _read_exact_secret(secrets_client, prior_version_id)
    if _document_commitment(prior_secret) != journal["prior_document_commitment"]:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery prior document differs"
        )
    expected_candidate, _document_format, candidate_status = _validated_candidate(
        prior_secret
    )
    if (
        candidate_status != "verified"
        or _document_commitment(expected_candidate)
        != journal["candidate_document_commitment"]
    ):
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery candidate authorization differs"
        )
    try:
        candidate_secret = _read_exact_secret(secrets_client, candidate_version_id)
    except GatewayMinerSubmissionsDisableError:
        candidate_secret = ""
    if candidate_secret:
        if (
            candidate_secret != expected_candidate
            or _document_commitment(candidate_secret)
            != journal["candidate_document_commitment"]
        ):
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery candidate document differs"
            )
    expected_nontransaction = _nontransaction_topology(
        initial_topology,
        custom_stage_label=custom_stage_label,
    )
    observed_nontransaction = _nontransaction_topology(
        stages,
        custom_stage_label=custom_stage_label,
    )
    if observed_nontransaction != expected_nontransaction:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery observed unrelated stage drift"
        )
    custom_holders = _stage_holders(stages, custom_stage_label)
    if custom_holders not in ([], [candidate_version_id]):
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery custom-stage ownership differs"
        )

    if current_version_id == prior_version_id:
        if not candidate_secret and stages != initial_topology:
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery pre-stage topology differs"
            )
        original_previous = _stage_holders(initial_topology, "AWSPREVIOUS")
        _set_previous_version(
            secrets_client,
            target_version_id=(original_previous[0] if original_previous else None),
        )
        if candidate_secret:
            _remove_custom_stage(
                secrets_client,
                candidate_version_id=candidate_version_id,
                custom_stage_label=custom_stage_label,
            )
        recovered_version, recovered_secret, recovered_topology = _read_current_secret(
            secrets_client
        )
        if (
            recovered_version != prior_version_id
            or recovered_secret != prior_secret
            or recovered_topology != initial_topology
        ):
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery could not restore the original topology"
            )
    elif current_version_id == candidate_version_id and candidate_secret:
        _set_previous_version(
            secrets_client,
            target_version_id=prior_version_id,
        )
        final_topology = _remove_custom_stage(
            secrets_client,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
        )
        expected_final = _expected_promoted_topology(
            initial_topology,
            prior_version_id=prior_version_id,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
            include_custom_stage=False,
        )
        recovered_version, recovered_secret, recovered_topology = _read_current_secret(
            secrets_client
        )
        if (
            final_topology != expected_final
            or recovered_topology != expected_final
            or recovered_version != candidate_version_id
            or recovered_secret != candidate_secret
        ):
            raise GatewayMinerSubmissionsDisableError(
                "miner-disable recovery could not complete the promoted topology"
            )
    else:
        raise GatewayMinerSubmissionsDisableError(
            "miner-disable recovery current version is unrelated"
        )
    _remove_recovery_journal(
        path,
        identity,
        parent_authority=recovery_journal_authority,
    )


def _disable_gateway_miner_submissions_secret(
    *,
    secrets_client: Any,
    expected_current_version_id: str = "",
    apply: bool = False,
    recovery_journal_path: Path | None = None,
    recovery_journal_authority: Optional[tuple[int, str]] = None,
) -> dict[str, str]:
    """Verify or atomically apply the one fixed production safety setting."""

    expected_version = str(expected_current_version_id or "").strip()
    if apply and not _VERSION_ID_RE.fullmatch(expected_version):
        raise GatewayMinerSubmissionsDisableError(
            "apply requires a valid expected current version identity"
        )
    if apply and recovery_journal_path is None:
        raise GatewayMinerSubmissionsDisableError(
            "apply requires the crash-recovery journal path"
        )
    if apply and recovery_journal_authority is None:
        assert recovery_journal_path is not None
        with _open_recovery_journal_parent_fd(recovery_journal_path) as authority:
            return _disable_gateway_miner_submissions_secret(
                secrets_client=secrets_client,
                expected_current_version_id=expected_current_version_id,
                apply=True,
                recovery_journal_path=recovery_journal_path,
                recovery_journal_authority=authority,
            )
    if (
        not apply
        and expected_version
        and not _VERSION_ID_RE.fullmatch(expected_version)
    ):
        raise GatewayMinerSubmissionsDisableError(
            "expected current version identity is invalid"
        )

    initial_version_id, initial_secret, initial_topology = _read_current_secret(
        secrets_client
    )
    _validate_initial_topology(initial_topology, initial_version_id)
    if expected_version and expected_version != initial_version_id:
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret differs from the expected current version"
        )
    candidate_secret, document_format, verification_status = _validated_candidate(
        initial_secret
    )
    base_result = {
        "secret_id": GATEWAY_SECRET_ID,
        "current_version_id": initial_version_id,
        "backup_version_id": initial_version_id,
        "document_format": document_format,
        "prior_document_commitment": _document_commitment(initial_secret),
        "candidate_document_commitment": _document_commitment(candidate_secret),
        "prior_stage_topology_commitment": _topology_commitment(initial_topology),
        "current_document_commitment": _document_commitment(initial_secret),
        "current_hydrated_environment_commitment": (
            _n_minus_one_hydrated_environment_commitment(initial_secret)
        ),
        "current_stage_topology_commitment": _topology_commitment(
            initial_topology
        ),
    }
    if verification_status == "already_disabled":
        return {"status": "already_disabled", **base_result}
    if not apply:
        return {"status": "verified", **base_result}

    reread_version_id, reread_secret, reread_topology = _read_current_secret(
        secrets_client
    )
    if (
        reread_version_id != initial_version_id
        or reread_secret != initial_secret
        or reread_topology != initial_topology
    ):
        raise GatewayMinerSubmissionsDisableError(
            "gateway secret changed concurrently before staging"
        )

    candidate_version_id = str(uuid.uuid4())
    custom_stage_label = _custom_stage_label(candidate_version_id)
    if candidate_version_id in initial_topology or any(
        custom_stage_label in labels for labels in initial_topology.values()
    ):
        raise GatewayMinerSubmissionsDisableError(
            "candidate version or custom stage identity collides"
        )
    journal_body = _recovery_journal_body(
        prior_version_id=initial_version_id,
        candidate_version_id=candidate_version_id,
        custom_stage_label=custom_stage_label,
        initial_topology=initial_topology,
        prior_document_commitment=_document_commitment(initial_secret),
        candidate_document_commitment=_document_commitment(candidate_secret),
    )
    journal_path = Path(recovery_journal_path)
    assert recovery_journal_authority is not None
    _write_recovery_journal(
        journal_path,
        journal_body,
        parent_authority=recovery_journal_authority,
    )
    _journal, journal_identity = _read_recovery_journal(
        journal_path,
        parent_authority=recovery_journal_authority,
    )
    try:
        response = secrets_client.put_secret_value(
            SecretId=GATEWAY_SECRET_ID,
            SecretString=candidate_secret,
            ClientRequestToken=candidate_version_id,
            VersionStages=[custom_stage_label],
        )
    except Exception as exc:
        _fail_before_promotion(
            secrets_client,
            initial_topology=initial_topology,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
            public_message="candidate gateway secret could not be staged",
            cause=exc,
        )
    try:
        if _version_id(response) != candidate_version_id:
            raise GatewayMinerSubmissionsDisableError(
                "staged gateway secret version identity differs"
            )
        persisted_candidate = _read_exact_secret(
            secrets_client,
            candidate_version_id,
        )
        current_version_id, current_secret, staged_topology = _read_current_secret(
            secrets_client
        )
        expected_staged_topology = _expected_staged_topology(
            initial_topology,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
        )
        if (
            persisted_candidate != candidate_secret
            or current_version_id != initial_version_id
            or current_secret != initial_secret
            or staged_topology != expected_staged_topology
        ):
            raise GatewayMinerSubmissionsDisableError(
                "candidate gateway secret failed exact staged readback"
            )
    except Exception as exc:
        _fail_before_promotion(
            secrets_client,
            initial_topology=initial_topology,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
            public_message="candidate gateway secret could not be verified",
            cause=exc,
        )

    promotion_error: BaseException | None = None
    try:
        secrets_client.update_secret_version_stage(
            SecretId=GATEWAY_SECRET_ID,
            VersionStage="AWSCURRENT",
            MoveToVersionId=candidate_version_id,
            RemoveFromVersionId=initial_version_id,
        )
    except Exception as exc:
        promotion_error = exc
    expected_promoted_with_custom = _expected_promoted_topology(
        initial_topology,
        prior_version_id=initial_version_id,
        candidate_version_id=candidate_version_id,
        custom_stage_label=custom_stage_label,
        include_custom_stage=True,
    )
    try:
        promoted_topology = _version_stages(secrets_client)
    except Exception as exc:
        _restore_original_topology(
            secrets_client,
            initial_topology=initial_topology,
            prior_version_id=initial_version_id,
            prior_secret=initial_secret,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
        )
        raise GatewayMinerSubmissionsDisableError(
            "promoted gateway stage topology is unavailable"
        ) from exc
    if promoted_topology != expected_promoted_with_custom:
        if _stage_holders(promoted_topology, "AWSCURRENT") == [candidate_version_id]:
            _restore_original_topology(
                secrets_client,
                initial_topology=initial_topology,
                prior_version_id=initial_version_id,
                prior_secret=initial_secret,
                candidate_version_id=candidate_version_id,
                custom_stage_label=custom_stage_label,
            )
            raise GatewayMinerSubmissionsDisableError(
                "promoted gateway stage topology differs"
            ) from promotion_error
        _fail_before_promotion(
            secrets_client,
            initial_topology=initial_topology,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
            public_message="version-fenced gateway secret promotion failed",
            cause=promotion_error
            or GatewayMinerSubmissionsDisableError(
                "promotion returned without the expected stage topology"
            ),
        )

    try:
        final_topology = _remove_custom_stage(
            secrets_client,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
        )
        final_version_id, final_secret, readback_topology = _read_current_secret(
            secrets_client
        )
        final_environment, final_format = _parse_environment(final_secret)
        expected_final_topology = _expected_promoted_topology(
            initial_topology,
            prior_version_id=initial_version_id,
            candidate_version_id=candidate_version_id,
            custom_stage_label=custom_stage_label,
            include_custom_stage=False,
        )
        if (
            final_topology != expected_final_topology
            or readback_topology != expected_final_topology
            or final_version_id != candidate_version_id
            or final_secret != candidate_secret
            or final_format != document_format
            or final_environment.get(TARGET_ENV_NAME) != TARGET_ENV_VALUE
        ):
            raise GatewayMinerSubmissionsDisableError(
                "promoted gateway secret failed exact readback"
            )
    except Exception as exc:
        try:
            _restore_original_topology(
                secrets_client,
                initial_topology=initial_topology,
                prior_version_id=initial_version_id,
                prior_secret=initial_secret,
                candidate_version_id=candidate_version_id,
                custom_stage_label=custom_stage_label,
            )
        except Exception as rollback_exc:
            raise GatewayMinerSubmissionsDisableError(
                "promoted verification and automatic topology rollback failed"
            ) from rollback_exc
        if isinstance(exc, GatewayMinerSubmissionsDisableError):
            raise
        raise GatewayMinerSubmissionsDisableError(
            "promoted gateway secret could not be verified"
        ) from exc

    _remove_recovery_journal(
        journal_path,
        journal_identity,
        parent_authority=recovery_journal_authority,
    )
    return {
        "status": "updated",
        **base_result,
        "current_version_id": candidate_version_id,
        "candidate_version_id": candidate_version_id,
        "current_document_commitment": _document_commitment(final_secret),
        "current_hydrated_environment_commitment": (
            _n_minus_one_hydrated_environment_commitment(final_secret)
        ),
        "current_stage_topology_commitment": _topology_commitment(
            readback_topology
        ),
    }


def disable_gateway_miner_submissions_secret(
    *,
    secrets_client: Any,
    expected_current_version_id: str = "",
) -> dict[str, str]:
    """Read-only verification of the fixed production safety setting."""

    return _disable_gateway_miner_submissions_secret(
        secrets_client=secrets_client,
        expected_current_version_id=expected_current_version_id,
        apply=False,
    )


def _apply_gateway_miner_submissions_secret(
    *,
    secrets_client: Any,
    expected_current_version_id: str,
    recovery_journal_path: Path,
    recovery_journal_authority: Optional[tuple[int, str]] = None,
) -> dict[str, str]:
    """Internal transaction entry used only by the FD9-paired controller."""

    return _disable_gateway_miner_submissions_secret(
        secrets_client=secrets_client,
        expected_current_version_id=expected_current_version_id,
        apply=True,
        recovery_journal_path=recovery_journal_path,
        recovery_journal_authority=recovery_journal_authority,
    )


def _instance_role_aws_clients(
    *,
    environ: Mapping[str, str] | None = None,
    session_factory: Any | None = None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    if any(environment.get(name) for name in _FORBIDDEN_AWS_ENV_NAMES):
        raise GatewayMinerSubmissionsDisableError(
            "static or delegated AWS credential configuration is forbidden"
        )
    if str(environment.get("LEADPOET_AWS_INSTANCE_ROLE_ONLY") or "").lower() != "true":
        raise GatewayMinerSubmissionsDisableError(
            "instance-role-only mode was not explicitly selected"
        )
    if session_factory is None:
        import boto3
        import botocore.session

        botocore_session = botocore.session.get_session()
        # Never consult ~/.aws/config or ~/.aws/credentials.  In particular,
        # this prevents a default-profile credential_process from executing
        # before the resolved credential method can be checked below.
        botocore_session.set_config_variable("config_file", os.devnull)
        botocore_session.set_config_variable("credentials_file", os.devnull)
        session = boto3.session.Session(
            botocore_session=botocore_session,
            region_name=EXPECTED_AWS_REGION,
        )
    else:
        try:
            session = session_factory(region_name=EXPECTED_AWS_REGION)
        except Exception as exc:
            raise GatewayMinerSubmissionsDisableError(
                "gateway EC2 instance-role credentials are unavailable"
            ) from exc
    try:
        credentials = session.get_credentials()
    except Exception as exc:
        raise GatewayMinerSubmissionsDisableError(
            "gateway EC2 instance-role credentials are unavailable"
        ) from exc
    if credentials is None or credentials.method != "iam-role":
        raise GatewayMinerSubmissionsDisableError(
            "gateway EC2 instance-role credentials are unavailable"
        )
    try:
        sts = session.client("sts")
        secretsmanager = session.client("secretsmanager")
        s3 = session.client("s3")
    except Exception as exc:
        raise GatewayMinerSubmissionsDisableError(
            "gateway EC2 instance-role identity is unavailable"
        ) from exc
    for service, client in {
        "sts": sts,
        "secretsmanager": secretsmanager,
        "s3": s3,
    }.items():
        endpoint = str(
            getattr(getattr(client, "meta", None), "endpoint_url", "") or ""
        ).rstrip("/")
        if endpoint not in _EXPECTED_AWS_ENDPOINTS[service]:
            raise GatewayMinerSubmissionsDisableError(
                "gateway AWS service endpoint identity differs"
            )
    try:
        identity = sts.get_caller_identity()
    except Exception as exc:
        raise GatewayMinerSubmissionsDisableError(
            "gateway EC2 instance-role identity is unavailable"
        ) from exc
    if str(
        identity.get("Account") or ""
    ) != EXPECTED_AWS_ACCOUNT_ID or not _EC2_ROLE_ARN_RE.fullmatch(
        str(identity.get("Arn") or "")
    ):
        raise GatewayMinerSubmissionsDisableError(
            "gateway EC2 instance-role identity differs"
        )
    return {"secretsmanager": secretsmanager, "s3": s3, "sts": sts}


def _instance_role_secrets_client(
    *,
    environ: Mapping[str, str] | None = None,
    session_factory: Any | None = None,
) -> Any:
    return _instance_role_aws_clients(
        environ=environ,
        session_factory=session_factory,
    )["secretsmanager"]


def _verify_protected_source() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "gateway" / "tee" / "protected_workflows.json"
    try:
        from gateway.tee.protected_workflows import load_manifest, verify_manifest

        verify_manifest(root, load_manifest(manifest_path))
    except Exception as exc:
        raise GatewayMinerSubmissionsDisableError(
            "fixed-purpose operation source binding is invalid"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the fixed production miner-submission disable without "
            "displaying secret values. Mutation is available only through "
            "the paired canonical restart controller."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-current-version-id",
        default="",
        help="Exact AWSCURRENT VersionId from the immediately prior verification.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.apply:
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "error": (
                        "standalone mutation is forbidden; use the paired "
                        "canonical restart controller"
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    try:
        _verify_protected_source()
        result = disable_gateway_miner_submissions_secret(
            secrets_client=_instance_role_secrets_client(),
            expected_current_version_id=str(args.expected_current_version_id),
        )
    except GatewayMinerSubmissionsDisableError as exc:
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
            '{"error":"unexpected fixed-purpose operation failure",'
            '"status":"failed_closed"}',
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
