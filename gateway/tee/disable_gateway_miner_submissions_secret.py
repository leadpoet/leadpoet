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
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Mapping, NoReturn, Sequence
import uuid


GATEWAY_SECRET_ID = "leadpoet/prod/gateway/env"
EXPECTED_AWS_ACCOUNT_ID = "493765492819"
EXPECTED_AWS_REGION = "us-east-1"
EXPECTED_GATEWAY_ROLE_NAME = "leadpoet-gateway-s3-cloudwatch-role"
TARGET_ENV_NAME = "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED"
TARGET_ENV_VALUE = "false"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_VERSION_ID_RE = re.compile(r"^[A-Za-z0-9-]{32,64}$")
_CUSTOM_STAGE_PREFIX = "LEADPOET_MINER_DISABLE_"
_CUSTOM_STAGE_RE = re.compile(r"^LEADPOET_MINER_DISABLE_[0-9a-f]{32}$")
_STAGE_LABEL_RE = re.compile(r"^[A-Za-z0-9_+=.@-]{1,256}$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
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
    }
)


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
        if name in values:
            raise GatewayMinerSubmissionsDisableError(
                "gateway secret environment contains duplicate names"
            )
        values[name] = (
            _decode_shell_target_value(raw_value.strip())
            if name == TARGET_ENV_NAME
            else raw_value
        )
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


def _topology_commitment(stages: Mapping[str, frozenset[str]]) -> str:
    document = {version: sorted(labels) for version, labels in sorted(stages.items())}
    return _document_commitment(
        json.dumps(document, sort_keys=True, separators=(",", ":"))
    )


def _validated_candidate(initial_secret: str) -> tuple[str, str, str]:
    initial_environment, document_format = _parse_environment(initial_secret)
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


def disable_gateway_miner_submissions_secret(
    *,
    secrets_client: Any,
    expected_current_version_id: str = "",
    apply: bool = False,
) -> dict[str, str]:
    """Verify or atomically apply the one fixed production safety setting."""

    expected_version = str(expected_current_version_id or "").strip()
    if apply and not _VERSION_ID_RE.fullmatch(expected_version):
        raise GatewayMinerSubmissionsDisableError(
            "apply requires a valid expected current version identity"
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

    return {
        "status": "updated",
        **base_result,
        "current_version_id": candidate_version_id,
        "candidate_version_id": candidate_version_id,
    }


def _instance_role_secrets_client(
    *,
    environ: Mapping[str, str] | None = None,
    session_factory: Any | None = None,
) -> Any:
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

        session_factory = boto3.session.Session
    try:
        session = session_factory(region_name=EXPECTED_AWS_REGION)
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
        identity = session.client("sts").get_caller_identity()
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
    try:
        return session.client("secretsmanager")
    except Exception as exc:
        raise GatewayMinerSubmissionsDisableError(
            "gateway Secrets Manager client is unavailable"
        ) from exc


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
            "Verify or apply the fixed production miner-submission disable "
            "without displaying secret values."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Stage, verify, and version-fence the fixed false setting.",
    )
    parser.add_argument(
        "--expected-current-version-id",
        default="",
        help="Exact AWSCURRENT VersionId from the immediately prior verification.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        _verify_protected_source()
        result = disable_gateway_miner_submissions_secret(
            secrets_client=_instance_role_secrets_client(),
            expected_current_version_id=str(args.expected_current_version_id),
            apply=bool(args.apply),
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
