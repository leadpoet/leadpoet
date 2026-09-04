"""Bind a pre-hydration miner-maintenance change to one locked restart.

The first release containing the fixed-purpose miner-submission maintenance
helper cannot run that helper from the deployed N-1 checkout.  This module is
instead executed from a verified archive of the exact candidate
while the canonical gateway restart lock is held.  It acquires the retry-stable
production SOURCE_ADD restart guard, drains every lease, disables the existing
global miner setting, and carries only non-secret commitments through that
invocation in a sealed, unlinked memory file.  The guard is exact-released only
after candidate runtime verification and atomically restores the durable
SOURCE_ADD pause state captured before the restart. No cross-invocation local
receipt or restart authority is persisted. A failed restart remains paused.
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
import shlex
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
    _open_recovery_journal_parent_fd,
    _parse_environment,
    _read_current_secret,
    _recover_orphan_transaction,
    _verify_protected_source,
    disable_gateway_miner_submissions_secret,
)
from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.supabase_schema_preflight_v2 import (
    SOURCE_ADD_CLAIM_CONTROL_ROLLBACK_V1_CONTRACT_SHA256,
    SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256,
    SupabaseSchemaPreflightV2Error,
    _verify_source_add_claim_control_contract_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.production_parity_boundary_v2 import (
    PRODUCTION_SUPABASE_ORIGIN,
)
from scripts.gateway_git_deploy import (
    DEFAULT_BRANCH,
    DEFAULT_REPO_URL,
    SCHEMA_VERSION as GIT_DEPLOYMENT_SCHEMA_VERSION,
    verify_materialized_tree,
)


SCHEMA_VERSION = "leadpoet.gateway_miner_maintenance_restart.v5"
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
MAX_RUNTIME_STATUS_BYTES = 256 * 1024
DEFAULT_RUNTIME_STATUS_URL = "http://127.0.0.1:8000/research-lab/status"
SOURCE_ADD_PAUSE_RPC = "research_lab_source_add_set_paused"
SOURCE_ADD_ADMISSION_CONTRACT_RPC = (
    "research_lab_source_add_admission_control_contract_v1"
)
SOURCE_ADD_CLAIM_CONTROL_CONTRACT_RPC = (
    "research_lab_source_add_claim_control_contract_v2"
)
SOURCE_ADD_RESTART_QUIESCENCE_RPC = (
    "research_lab_source_add_restart_quiescence_v1"
)
SOURCE_ADD_RESTART_GUARD_STATE_RPC = (
    "research_lab_source_add_restart_guard_state_v2"
)
SOURCE_ADD_ACQUIRE_RESTART_GUARD_RPC = (
    "research_lab_source_add_acquire_restart_guard_v2"
)
SOURCE_ADD_RELEASE_RESTART_GUARD_RPC = (
    "research_lab_source_add_release_restart_guard_v2"
)
SOURCE_ADD_CONTROL_TABLE = "research_lab_source_add_control"
SOURCE_ADD_PAUSE_REASON = "canonical_restart_guard"
SOURCE_ADD_CONTROL_MAX_BYTES = 64 * 1024
SOURCE_ADD_CONTROL_TIMEOUT_SECONDS = 15.0
# The canonical paired restart can coordinate for 9,300 seconds before the
# bounded candidate startup work. Migration 172 deliberately caps the lease at
# the exact deadline plus this 5,100-second post-coordination safety margin.
SOURCE_ADD_CANONICAL_COORDINATION_DEADLINE_SECONDS = 9_300
SOURCE_ADD_RESTART_GUARD_SAFETY_MARGIN_SECONDS = 5_100
SOURCE_ADD_RESTART_GUARD_LEASE_SECONDS = (
    SOURCE_ADD_CANONICAL_COORDINATION_DEADLINE_SECONDS
    + SOURCE_ADD_RESTART_GUARD_SAFETY_MARGIN_SECONDS
)
SOURCE_ADD_RESTART_GUARD_AUTHORITY = (
    "leadpoet.production.gateway.canonical_restart.v1"
)
SOURCE_ADD_QUIESCENCE_TIMEOUT_SECONDS = 900.0
SOURCE_ADD_QUIESCENCE_POLL_SECONDS = 1.0
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
        "current_secret_version_id",
        "current_document_commitment",
        "current_hydrated_environment_commitment",
        "current_stage_topology_commitment",
        "source_add_control_commitment",
        "source_add_restart_guard_commitment",
        "source_add_restart_guard_generation",
        "source_add_restart_guard_owner_generation_commitment",
        "source_add_restart_guard_restore_paused",
        "source_add_quiescence_commitment",
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


class _SourceAddAuthorityRejected(GatewayMinerMaintenanceRestartError):
    """The SOURCE_ADD authority conclusively rejected a request."""


class _SourceAddClaimControlContractResponse:
    """In-memory urllib-shaped response for the shared exact verifier."""

    def __init__(self, value: Any) -> None:
        self._payload = json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def __enter__(self) -> "_SourceAddClaimControlContractResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def getcode(self) -> int:
        return 200

    def read(self) -> bytes:
        return self._payload


def _decode_secret_environment_value(
    values: Mapping[str, Any],
    *,
    document_format: str,
    name: str,
) -> str:
    raw_value = values.get(name)
    if not isinstance(raw_value, str):
        raise GatewayMinerMaintenanceRestartError(
            "gateway secret lacks SOURCE_ADD pause authority"
        )
    if document_format == "json":
        value = raw_value
    elif document_format == "shell":
        try:
            parts = shlex.split("VALUE=" + raw_value, posix=True)
        except ValueError as exc:
            raise GatewayMinerMaintenanceRestartError(
                "gateway secret SOURCE_ADD pause authority is malformed"
            ) from exc
        if len(parts) != 1 or not parts[0].startswith("VALUE="):
            raise GatewayMinerMaintenanceRestartError(
                "gateway secret SOURCE_ADD pause authority is malformed"
            )
        value = parts[0].split("=", 1)[1]
    else:
        raise GatewayMinerMaintenanceRestartError(
            "gateway secret SOURCE_ADD pause authority is malformed"
        )
    if not value or len(value.encode("utf-8")) > 64 * 1024 or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise GatewayMinerMaintenanceRestartError(
            "gateway secret SOURCE_ADD pause authority is malformed"
        )
    return value


def _source_add_pause_credentials(
    *,
    secrets_client: Any,
    expected_current_version_id: Optional[str] = None,
) -> tuple[str, str]:
    try:
        version_id, raw_secret, _stages = _read_current_secret(secrets_client)
        values, document_format = _parse_environment(raw_secret)
    except GatewayMinerSubmissionsDisableError:
        raise
    except Exception as exc:
        raise GatewayMinerMaintenanceRestartError(
            "gateway secret SOURCE_ADD pause authority is unavailable"
        ) from exc
    if (
        expected_current_version_id is not None
        and version_id != str(expected_current_version_id)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "gateway secret changed during SOURCE_ADD pause verification"
        )
    supabase_url = _decode_secret_environment_value(
        values,
        document_format=document_format,
        name="SUPABASE_URL",
    ).rstrip("/")
    service_role_key = _decode_secret_environment_value(
        values,
        document_format=document_format,
        name="SUPABASE_SERVICE_ROLE_KEY",
    )
    if supabase_url != PRODUCTION_SUPABASE_ORIGIN:
        raise GatewayMinerMaintenanceRestartError(
            "gateway secret SOURCE_ADD pause origin differs from production"
        )
    return supabase_url, service_role_key


def _source_add_control_request(
    *,
    method: str,
    path: str,
    service_role_key: str,
    payload: Optional[Mapping[str, Any]] = None,
    connection_factory: Any = http.client.HTTPSConnection,
    timeout_seconds: float = SOURCE_ADD_CONTROL_TIMEOUT_SECONDS,
) -> Any:
    if (
        method not in {"GET", "POST"}
        or not path.startswith("/rest/v1/")
        or "//" in path
        or not service_role_key
        or not 1.0 <= float(timeout_seconds) <= 30.0
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD pause request is invalid"
        )
    hostname = PRODUCTION_SUPABASE_ORIGIN.removeprefix("https://")
    if not hostname or "/" in hostname:
        raise GatewayMinerMaintenanceRestartError(
            "production SOURCE_ADD pause origin is invalid"
        )
    encoded = None
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer " + service_role_key,
        "apikey": service_role_key,
        "Connection": "close",
        "Host": hostname,
    }
    if payload is not None:
        encoded = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(encoded))
    connection = connection_factory(
        hostname,
        443,
        timeout=float(timeout_seconds),
    )
    try:
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        if not 200 <= int(response.status) < 300:
            raise _SourceAddAuthorityRejected(
                "SOURCE_ADD pause authority request was rejected"
            )
        content_length = response.getheader("Content-Length")
        if (
            content_length is not None
            and int(content_length) > SOURCE_ADD_CONTROL_MAX_BYTES
        ):
            raise GatewayMinerMaintenanceRestartError(
                "SOURCE_ADD pause authority response is too large"
            )
        response_payload = response.read(SOURCE_ADD_CONTROL_MAX_BYTES + 1)
    except GatewayMinerMaintenanceRestartError:
        raise
    except Exception as exc:
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD pause authority is unavailable"
        ) from exc
    finally:
        connection.close()
    if len(response_payload) > SOURCE_ADD_CONTROL_MAX_BYTES:
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD pause authority response is too large"
        )
    try:
        return json.loads(
            response_payload.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD pause authority response is invalid"
        ) from exc


def _normalized_source_add_control(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    fields = {"singleton", "paused", "reason", "actor_ref", "updated_at"}
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("singleton") is not True
        or not isinstance(value.get("paused"), bool)
        or not isinstance(value.get("reason"), str)
        or not str(value.get("reason"))
        or len(str(value.get("reason"))) > 500
        or not isinstance(value.get("actor_ref"), str)
        or not str(value.get("actor_ref"))
        or len(str(value.get("actor_ref"))) > 200
        or not isinstance(value.get("updated_at"), str)
        or not str(value.get("updated_at"))
    ):
        raise GatewayMinerMaintenanceRestartError(
            "durable SOURCE_ADD pause readback is invalid"
        )
    try:
        updated_at = datetime.fromisoformat(
            str(value["updated_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "durable SOURCE_ADD pause readback is invalid"
        ) from exc
    if updated_at.tzinfo is None:
        raise GatewayMinerMaintenanceRestartError(
            "durable SOURCE_ADD pause readback is invalid"
        )
    return {name: value[name] for name in sorted(fields)}


def _source_add_control_commitment(value: Any) -> str:
    return sha256_json(_normalized_source_add_control(value))


def _read_source_add_control(
    *,
    service_role_key: str,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, Any]:
    result = _source_add_control_request(
        method="GET",
        path=(
            f"/rest/v1/{SOURCE_ADD_CONTROL_TABLE}"
            "?select=singleton,paused,reason,actor_ref,updated_at"
            "&singleton=eq.true&limit=2"
        ),
        service_role_key=service_role_key,
        connection_factory=connection_factory,
    )
    return _normalized_source_add_control(result)


def _require_source_add_admission_control_contract(
    *,
    service_role_key: str,
    connection_factory: Any = http.client.HTTPSConnection,
) -> None:
    result = _source_add_control_request(
        method="POST",
        path=f"/rest/v1/rpc/{SOURCE_ADD_ADMISSION_CONTRACT_RPC}",
        service_role_key=service_role_key,
        payload={},
        connection_factory=connection_factory,
    )
    if result != {
        "schema_version": "leadpoet.source_add_admission_control_contract.v1",
        "control_row_present": True,
        "trigger_enabled": True,
        "pause_rpc": SOURCE_ADD_PAUSE_RPC,
        "admission_trigger": "trg_source_add_work_admission_control",
    }:
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD admission-control contract is unavailable"
        )


def _require_source_add_claim_control_contract(
    *,
    service_role_key: str,
    connection_factory: Any = http.client.HTTPSConnection,
) -> None:
    expected_url = (
        f"{PRODUCTION_SUPABASE_ORIGIN}/rest/v1/rpc/"
        f"{SOURCE_ADD_CLAIM_CONTROL_CONTRACT_RPC}"
    )

    def opener(request: Any, *, timeout: float) -> Any:
        if (
            request.full_url != expected_url
            or request.get_method() != "POST"
            or request.data != b"{}"
            or float(timeout) != SOURCE_ADD_CONTROL_TIMEOUT_SECONDS
        ):
            raise GatewayMinerMaintenanceRestartError(
                "SOURCE_ADD claim-control contract request is invalid"
            )
        value = _source_add_control_request(
            method="POST",
            path=(
                f"/rest/v1/rpc/{SOURCE_ADD_CLAIM_CONTROL_CONTRACT_RPC}"
            ),
            service_role_key=service_role_key,
            payload={},
            connection_factory=connection_factory,
        )
        return _SourceAddClaimControlContractResponse(value)

    try:
        contract = _verify_source_add_claim_control_contract_v2(
            headers={},
            supabase_url=PRODUCTION_SUPABASE_ORIGIN,
            opener=opener,
            timeout_seconds=SOURCE_ADD_CONTROL_TIMEOUT_SECONDS,
        )
    except (SupabaseSchemaPreflightV2Error, GatewayMinerMaintenanceRestartError) as exc:
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD claim-control contract is unavailable or invalid"
        ) from exc
    if contract.get("function_authority_sha256") != (
        SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD claim-control function authority differs"
        )
    if contract.get("rollback_v1_contract_sha256") != (
        SOURCE_ADD_CLAIM_CONTROL_ROLLBACK_V1_CONTRACT_SHA256
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD rollback claim-control authority differs"
        )


def _source_add_restart_guard_identity(
    restart_invocation_id: str,
) -> dict[str, str]:
    invocation_id = str(restart_invocation_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}", invocation_id):
        raise GatewayMinerMaintenanceRestartError(
            "gateway restart invocation identity is invalid"
        )
    actor_digest = hashlib.sha256(invocation_id.encode("utf-8")).hexdigest()
    guard_digest = hashlib.sha256(
        SOURCE_ADD_RESTART_GUARD_AUTHORITY.encode("utf-8")
    ).hexdigest()
    guard_id = "source_add_restart_guard:" + guard_digest
    owner_id = "source_add_restart_owner:" + actor_digest
    owner_commitment = "sha256:" + hashlib.sha256(
        owner_id.encode("utf-8")
    ).hexdigest()
    return {
        "guard_id": guard_id,
        "guard_commitment": "sha256:"
        + hashlib.sha256(guard_id.encode("utf-8")).hexdigest(),
        "owner_id": owner_id,
        "owner_commitment": owner_commitment,
        "actor_ref": "gateway-restart:" + actor_digest,
    }


def _source_add_owner_generation_commitment(
    owner_commitment: str,
    guard_generation: int,
) -> str:
    if (
        not _SHA256_RE.fullmatch(str(owner_commitment))
        or isinstance(guard_generation, bool)
        or not isinstance(guard_generation, int)
        or guard_generation < 0
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard owner generation is invalid"
        )
    payload = f"{owner_commitment}:{guard_generation}".encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source_add_guard_generation(value: Any, *, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 9_223_372_036_854_775_807
    ):
        raise GatewayMinerMaintenanceRestartError(label)
    return value


def _source_add_expected_guard_generation(value: Any, *, label: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return _source_add_guard_generation(value, label=label)
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise GatewayMinerMaintenanceRestartError(label)
    return _source_add_guard_generation(int(value), label=label)


def _source_add_expected_restore_paused(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise GatewayMinerMaintenanceRestartError(label)


def _source_add_guard_expiry(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise GatewayMinerMaintenanceRestartError(label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GatewayMinerMaintenanceRestartError(label) from exc
    if parsed.tzinfo is None:
        raise GatewayMinerMaintenanceRestartError(label)
    return parsed


def _normalized_source_add_restart_guard_state(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "paused",
        "guard_active",
        "guard_commitment",
        "owner_commitment",
        "guard_generation",
        "owner_generation_commitment",
        "guard_expires_at",
        "restore_paused",
    }
    label = "SOURCE_ADD restart guard state is invalid"
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema_version")
        != "leadpoet.source_add_restart_guard_state.v2"
        or not isinstance(value.get("paused"), bool)
        or not isinstance(value.get("guard_active"), bool)
        or not isinstance(value.get("guard_commitment"), str)
        or not isinstance(value.get("owner_commitment"), str)
        or not isinstance(value.get("owner_generation_commitment"), str)
        or (
            value.get("guard_expires_at") is not None
            and not isinstance(value.get("guard_expires_at"), str)
        )
        or (
            value.get("restore_paused") is not None
            and not isinstance(value.get("restore_paused"), bool)
        )
    ):
        raise GatewayMinerMaintenanceRestartError(label)
    generation = _source_add_guard_generation(
        value.get("guard_generation"), label=label
    )
    commitments = (
        str(value["guard_commitment"]),
        str(value["owner_commitment"]),
        str(value["owner_generation_commitment"]),
    )
    has_commitments = any(commitments)
    if has_commitments and (
        not all(_SHA256_RE.fullmatch(item) for item in commitments)
        or commitments[2]
        != _source_add_owner_generation_commitment(commitments[1], generation)
    ):
        raise GatewayMinerMaintenanceRestartError(label)
    if not has_commitments and (
        any(commitments)
        or value.get("guard_expires_at") is not None
        or value.get("restore_paused") is not None
    ):
        raise GatewayMinerMaintenanceRestartError(label)
    if value.get("guard_expires_at") is not None:
        _source_add_guard_expiry(value["guard_expires_at"], label=label)
    if value["guard_active"] is True and (
        value["paused"] is not True
        or not has_commitments
        or value.get("guard_expires_at") is None
        or not isinstance(value.get("restore_paused"), bool)
    ):
        raise GatewayMinerMaintenanceRestartError(label)
    return {name: value[name] for name in sorted(fields)}


def _normalized_source_add_restart_guard(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "paused",
        "guard_active",
        "guard_commitment",
        "owner_commitment",
        "guard_generation",
        "owner_generation_commitment",
        "guard_expires_at",
        "restore_paused",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema_version")
        != "leadpoet.source_add_restart_guard.v2"
        or value.get("paused") is not True
        or value.get("guard_active") is not True
        or not isinstance(value.get("guard_commitment"), str)
        or not _SHA256_RE.fullmatch(str(value.get("guard_commitment")))
        or not isinstance(value.get("owner_commitment"), str)
        or not _SHA256_RE.fullmatch(str(value.get("owner_commitment")))
        or not isinstance(value.get("owner_generation_commitment"), str)
        or not _SHA256_RE.fullmatch(
            str(value.get("owner_generation_commitment"))
        )
        or not isinstance(value.get("restore_paused"), bool)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard response is invalid"
        )
    generation = _source_add_guard_generation(
        value.get("guard_generation"),
        label="SOURCE_ADD restart guard response is invalid",
    )
    if value["owner_generation_commitment"] != (
        _source_add_owner_generation_commitment(
            str(value["owner_commitment"]), generation
        )
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard response is invalid"
        )
    _source_add_guard_expiry(
        value.get("guard_expires_at"),
        label="SOURCE_ADD restart guard response is invalid",
    )
    return {name: value[name] for name in sorted(fields)}


def _normalized_source_add_quiescence(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "paused",
        "guard_active",
        "guard_matches",
        "owner_matches",
        "generation_matches",
        "guard_commitment",
        "owner_commitment",
        "guard_generation",
        "owner_generation_commitment",
        "guard_expires_at",
        "leased_work_count",
        "quiescent",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema_version")
        != "leadpoet.source_add_restart_quiescence.v1"
        or not isinstance(value.get("paused"), bool)
        or not isinstance(value.get("guard_active"), bool)
        or not isinstance(value.get("guard_matches"), bool)
        or not isinstance(value.get("owner_matches"), bool)
        or not isinstance(value.get("generation_matches"), bool)
        or not isinstance(value.get("guard_commitment"), str)
        or not _SHA256_RE.fullmatch(str(value.get("guard_commitment")))
        or not isinstance(value.get("owner_commitment"), str)
        or not _SHA256_RE.fullmatch(str(value.get("owner_commitment")))
        or not isinstance(value.get("owner_generation_commitment"), str)
        or not _SHA256_RE.fullmatch(
            str(value.get("owner_generation_commitment"))
        )
        or (
            value.get("guard_expires_at") is not None
            and not isinstance(value.get("guard_expires_at"), str)
        )
        or not isinstance(value.get("quiescent"), bool)
        or isinstance(value.get("leased_work_count"), bool)
        or not isinstance(value.get("leased_work_count"), int)
        or int(value.get("leased_work_count")) < 0
        or value.get("quiescent")
        is not (
            value.get("paused") is True
            and value.get("guard_active") is True
            and value.get("guard_matches") is True
            and value.get("owner_matches") is True
            and value.get("generation_matches") is True
            and int(value.get("leased_work_count")) == 0
        )
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart-quiescence readback is invalid"
        )
    generation = _source_add_guard_generation(
        value.get("guard_generation"),
        label="SOURCE_ADD restart-quiescence readback is invalid",
    )
    if value["owner_generation_commitment"] != (
        _source_add_owner_generation_commitment(
            str(value["owner_commitment"]), generation
        )
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart-quiescence readback is invalid"
        )
    if value.get("guard_expires_at") is not None:
        _source_add_guard_expiry(
            value["guard_expires_at"],
            label="SOURCE_ADD restart-quiescence readback is invalid",
        )
    if value.get("guard_active") is True and value.get("guard_expires_at") is None:
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart-quiescence readback is invalid"
        )
    return {name: value[name] for name in sorted(fields)}


def _read_source_add_restart_guard_state(
    *,
    service_role_key: str,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, Any]:
    return _normalized_source_add_restart_guard_state(
        _source_add_control_request(
            method="POST",
            path=f"/rest/v1/rpc/{SOURCE_ADD_RESTART_GUARD_STATE_RPC}",
            service_role_key=service_role_key,
            payload={},
            connection_factory=connection_factory,
        )
    )


def _read_source_add_restart_quiescence(
    *,
    service_role_key: str,
    guard_id: str,
    owner_id: str,
    guard_generation: int,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, Any]:
    return _normalized_source_add_quiescence(
        _source_add_control_request(
            method="POST",
            path=f"/rest/v1/rpc/{SOURCE_ADD_RESTART_QUIESCENCE_RPC}",
            service_role_key=service_role_key,
            payload={
                "p_guard_generation": guard_generation,
                "p_guard_id": guard_id,
                "p_owner_id": owner_id,
            },
            connection_factory=connection_factory,
        )
    )


def _require_owned_source_add_guard_state(
    *,
    service_role_key: str,
    restart_invocation_id: str,
    expected_guard_commitment: Optional[str] = None,
    expected_guard_generation: Optional[Any] = None,
    expected_owner_generation_commitment: Optional[str] = None,
    expected_restore_paused: Optional[Any] = None,
    connection_factory: Any = http.client.HTTPSConnection,
) -> tuple[dict[str, str], dict[str, Any]]:
    identity = _source_add_restart_guard_identity(restart_invocation_id)
    state = _read_source_add_restart_guard_state(
        service_role_key=service_role_key,
        connection_factory=connection_factory,
    )
    generation = _source_add_guard_generation(
        state["guard_generation"],
        label="SOURCE_ADD restart guard ownership is invalid",
    )
    owner_generation = _source_add_owner_generation_commitment(
        identity["owner_commitment"], generation
    )
    if (
        state["paused"] is not True
        or state["guard_active"] is not True
        or state["guard_commitment"] != identity["guard_commitment"]
        or state["owner_commitment"] != identity["owner_commitment"]
        or state["owner_generation_commitment"] != owner_generation
        or _source_add_guard_expiry(
            state["guard_expires_at"],
            label="SOURCE_ADD restart guard ownership is invalid",
        )
        <= datetime.now(timezone.utc)
        or (
            expected_guard_commitment is not None
            and identity["guard_commitment"]
            != str(expected_guard_commitment)
        )
        or (
            expected_guard_generation is not None
            and generation
            != _source_add_expected_guard_generation(
                expected_guard_generation,
                label="SOURCE_ADD restart guard ownership is invalid",
            )
        )
        or (
            expected_owner_generation_commitment is not None
            and owner_generation
            != str(expected_owner_generation_commitment)
        )
        or (
            expected_restore_paused is not None
            and state["restore_paused"]
            is not _source_add_expected_restore_paused(
                expected_restore_paused,
                label="SOURCE_ADD restart restore state is invalid",
            )
        )
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard is not owned by this invocation"
        )
    return identity, state


def _source_add_quiescence_commitment(value: Any) -> str:
    normalized = _normalized_source_add_quiescence(value)
    return sha256_json(
        {
            name: normalized[name]
            for name in sorted(normalized)
            if name != "guard_expires_at"
        }
    )


def _wait_for_source_add_quiescence(
    *,
    secrets_client: Any,
    restart_invocation_id: str,
    expected_current_version_id: Optional[str] = None,
    expected_guard_commitment: Optional[str] = None,
    expected_guard_generation: Optional[Any] = None,
    expected_owner_generation_commitment: Optional[str] = None,
    expected_restore_paused: Optional[Any] = None,
    connection_factory: Any = http.client.HTTPSConnection,
    timeout_seconds: float = SOURCE_ADD_QUIESCENCE_TIMEOUT_SECONDS,
    poll_seconds: float = SOURCE_ADD_QUIESCENCE_POLL_SECONDS,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> dict[str, str]:
    if not 1.0 <= float(timeout_seconds) <= 900.0 or not 0.1 <= float(
        poll_seconds
    ) <= 5.0:
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart-quiescence wait policy is invalid"
        )
    _supabase_url, service_role_key = _source_add_pause_credentials(
        secrets_client=secrets_client,
        expected_current_version_id=expected_current_version_id,
    )
    guard, guard_state = _require_owned_source_add_guard_state(
        service_role_key=service_role_key,
        restart_invocation_id=restart_invocation_id,
        expected_guard_commitment=expected_guard_commitment,
        expected_guard_generation=expected_guard_generation,
        expected_owner_generation_commitment=(
            expected_owner_generation_commitment
        ),
        expected_restore_paused=expected_restore_paused,
        connection_factory=connection_factory,
    )
    generation = int(guard_state["guard_generation"])
    owner_generation = str(guard_state["owner_generation_commitment"])
    deadline = float(monotonic()) + float(timeout_seconds)
    while True:
        state = _read_source_add_restart_quiescence(
            service_role_key=service_role_key,
            guard_id=guard["guard_id"],
            owner_id=guard["owner_id"],
            guard_generation=generation,
            connection_factory=connection_factory,
        )
        if (
            state["paused"] is not True
            or state["guard_active"] is not True
            or state["guard_matches"] is not True
            or state["owner_matches"] is not True
            or state["generation_matches"] is not True
            or state["guard_commitment"] != guard["guard_commitment"]
            or state["owner_commitment"] != guard["owner_commitment"]
            or state["guard_generation"] != generation
            or state["owner_generation_commitment"] != owner_generation
            or _source_add_guard_expiry(
                state["guard_expires_at"],
                label="SOURCE_ADD restart guard expiry is invalid",
            )
            <= datetime.now(timezone.utc)
        ):
            raise GatewayMinerMaintenanceRestartError(
                "SOURCE_ADD restart guard was lost while waiting for quiescence"
            )
        if state["quiescent"] is True:
            return {
                "status": "quiescent",
                "source_add_restart_guard_commitment": guard[
                    "guard_commitment"
                ],
                "source_add_restart_guard_generation": str(generation),
                "source_add_restart_guard_owner_generation_commitment": (
                    owner_generation
                ),
                "source_add_restart_guard_restore_paused": (
                    "true" if guard_state["restore_paused"] else "false"
                ),
                "source_add_quiescence_commitment": (
                    _source_add_quiescence_commitment(state)
                ),
            }
        remaining = deadline - float(monotonic())
        if remaining <= 0:
            raise GatewayMinerMaintenanceRestartError(
                "SOURCE_ADD work did not quiesce before the restart deadline"
            )
        sleep(min(float(poll_seconds), remaining))


def _require_source_add_quiescent(
    *,
    secrets_client: Any,
    restart_invocation_id: str,
    expected_current_version_id: Optional[str] = None,
    expected_guard_commitment: Optional[str] = None,
    expected_guard_generation: Optional[Any] = None,
    expected_owner_generation_commitment: Optional[str] = None,
    expected_restore_paused: Optional[Any] = None,
    expected_quiescence_commitment: Optional[str] = None,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, str]:
    _supabase_url, service_role_key = _source_add_pause_credentials(
        secrets_client=secrets_client,
        expected_current_version_id=expected_current_version_id,
    )
    guard, guard_state = _require_owned_source_add_guard_state(
        service_role_key=service_role_key,
        restart_invocation_id=restart_invocation_id,
        expected_guard_commitment=expected_guard_commitment,
        expected_guard_generation=expected_guard_generation,
        expected_owner_generation_commitment=(
            expected_owner_generation_commitment
        ),
        expected_restore_paused=expected_restore_paused,
        connection_factory=connection_factory,
    )
    generation = int(guard_state["guard_generation"])
    owner_generation = str(guard_state["owner_generation_commitment"])
    state = _read_source_add_restart_quiescence(
        service_role_key=service_role_key,
        guard_id=guard["guard_id"],
        owner_id=guard["owner_id"],
        guard_generation=generation,
        connection_factory=connection_factory,
    )
    if (
        state["paused"] is not True
        or state["guard_active"] is not True
        or state["guard_matches"] is not True
        or state["owner_matches"] is not True
        or state["generation_matches"] is not True
        or state["guard_commitment"] != guard["guard_commitment"]
        or state["owner_commitment"] != guard["owner_commitment"]
        or state["guard_generation"] != generation
        or state["owner_generation_commitment"] != owner_generation
        or state["quiescent"] is not True
        or _source_add_guard_expiry(
            state["guard_expires_at"],
            label="SOURCE_ADD restart guard expiry is invalid",
        )
        <= datetime.now(timezone.utc)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD work is not quiescent for restart"
        )
    commitment = _source_add_quiescence_commitment(state)
    if (
        expected_quiescence_commitment is not None
        and commitment != str(expected_quiescence_commitment)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart quiescence differs from the invocation proof"
        )
    return {
        "status": "quiescent",
        "source_add_restart_guard_commitment": guard["guard_commitment"],
        "source_add_restart_guard_generation": str(generation),
        "source_add_restart_guard_owner_generation_commitment": (
            owner_generation
        ),
        "source_add_restart_guard_restore_paused": (
            "true" if guard_state["restore_paused"] else "false"
        ),
        "source_add_quiescence_commitment": commitment,
    }


def _acquire_source_add_restart_guard(
    *,
    service_role_key: str,
    restart_invocation_id: str,
    allow_takeover: bool,
    expected_guard_generation: Optional[Any] = None,
    expected_owner_generation_commitment: Optional[str] = None,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, Any]:
    identity = _source_add_restart_guard_identity(restart_invocation_id)
    before = _read_source_add_restart_guard_state(
        service_role_key=service_role_key,
        connection_factory=connection_factory,
    )
    before_generation = int(before["guard_generation"])
    before_has_guard = bool(before["guard_commitment"])
    same_owner = (
        before["guard_active"] is True
        and before["guard_commitment"] == identity["guard_commitment"]
        and before["owner_commitment"] == identity["owner_commitment"]
    )
    if not allow_takeover and not same_owner:
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard ownership changed before renewal"
        )
    if (
        expected_guard_generation is not None
        and before_generation
        != _source_add_expected_guard_generation(
            expected_guard_generation,
            label=(
                "SOURCE_ADD restart guard generation changed before renewal"
            ),
        )
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard generation changed before renewal"
        )
    before_owner_generation = (
        _source_add_owner_generation_commitment(
            identity["owner_commitment"], before_generation
        )
        if same_owner
        else ""
    )
    if (
        expected_owner_generation_commitment is not None
        and before_owner_generation
        != str(expected_owner_generation_commitment)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard owner changed before renewal"
        )
    result_generation = before_generation if same_owner else before_generation + 1
    expected_restore_paused = (
        before["restore_paused"]
        if same_owner or before_has_guard
        else before["paused"]
    )
    if not isinstance(expected_restore_paused, bool):
        expected_restore_paused = True
    result_owner_generation = _source_add_owner_generation_commitment(
        identity["owner_commitment"], result_generation
    )
    previous_expiry = (
        _source_add_guard_expiry(
            before["guard_expires_at"],
            label="SOURCE_ADD restart guard state is invalid",
        )
        if same_owner
        else None
    )
    try:
        acquired = _normalized_source_add_restart_guard(
            _source_add_control_request(
                method="POST",
                path=(
                    f"/rest/v1/rpc/{SOURCE_ADD_ACQUIRE_RESTART_GUARD_RPC}"
                ),
                service_role_key=service_role_key,
                payload={
                    "p_actor_ref": identity["actor_ref"],
                    "p_expected_generation": before_generation,
                    "p_guard_id": identity["guard_id"],
                    "p_lease_seconds": SOURCE_ADD_RESTART_GUARD_LEASE_SECONDS,
                    "p_owner_id": identity["owner_id"],
                },
                connection_factory=connection_factory,
            )
        )
    except _SourceAddAuthorityRejected:
        raise
    except (
        GatewayMinerMaintenanceRestartError,
        GatewayMinerSubmissionsDisableError,
    ) as exc:
        # The write may have committed even when its response was lost.  One
        # exact state read reconciles only this invocation's expected CAS
        # result; a rejection or another owner's state is never taken over in
        # a retry loop.
        try:
            reconciled = _read_source_add_restart_guard_state(
                service_role_key=service_role_key,
                connection_factory=connection_factory,
            )
            if (
                reconciled["paused"] is not True
                or reconciled["guard_active"] is not True
                or reconciled["guard_commitment"]
                != identity["guard_commitment"]
                or reconciled["owner_commitment"]
                != identity["owner_commitment"]
                or reconciled["guard_generation"] != result_generation
                or reconciled["owner_generation_commitment"]
                != result_owner_generation
                or reconciled["restore_paused"] is not expected_restore_paused
            ):
                raise GatewayMinerMaintenanceRestartError(
                    "SOURCE_ADD restart guard acquisition outcome is unknown"
                )
            acquired = {
                "schema_version": "leadpoet.source_add_restart_guard.v2",
                "paused": True,
                "guard_active": True,
                "guard_commitment": reconciled["guard_commitment"],
                "owner_commitment": reconciled["owner_commitment"],
                "guard_generation": reconciled["guard_generation"],
                "owner_generation_commitment": reconciled[
                    "owner_generation_commitment"
                ],
                "guard_expires_at": reconciled["guard_expires_at"],
                "restore_paused": reconciled["restore_paused"],
            }
        except GatewayMinerMaintenanceRestartError:
            raise exc
    expiry = _source_add_guard_expiry(
        acquired["guard_expires_at"],
        label="SOURCE_ADD restart guard response is invalid",
    )
    if (
        acquired["guard_commitment"] != identity["guard_commitment"]
        or acquired["owner_commitment"] != identity["owner_commitment"]
        or acquired["guard_generation"] != result_generation
        or acquired["owner_generation_commitment"]
        != result_owner_generation
        or acquired["restore_paused"] is not expected_restore_paused
        or expiry <= datetime.now(timezone.utc)
        or (previous_expiry is not None and expiry <= previous_expiry)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard RPC returned an unexpected state"
        )
    return acquired


def _pause_source_add_for_restart(
    *,
    secrets_client: Any,
    restart_invocation_id: str,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, str]:
    _supabase_url, service_role_key = _source_add_pause_credentials(
        secrets_client=secrets_client
    )
    _require_source_add_admission_control_contract(
        service_role_key=service_role_key,
        connection_factory=connection_factory,
    )
    _require_source_add_claim_control_contract(
        service_role_key=service_role_key,
        connection_factory=connection_factory,
    )
    guard = _source_add_restart_guard_identity(restart_invocation_id)
    rpc_result = _acquire_source_add_restart_guard(
        service_role_key=service_role_key,
        restart_invocation_id=restart_invocation_id,
        allow_takeover=True,
        connection_factory=connection_factory,
    )
    readback = _read_source_add_control(
        service_role_key=service_role_key,
        connection_factory=connection_factory,
    )
    if (
        readback["paused"] is not True
        or readback["reason"] != SOURCE_ADD_PAUSE_REASON
        or readback["actor_ref"] != guard["actor_ref"]
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD pause changed during guarded readback"
        )
    return {
        "status": "paused",
        "source_add_control_commitment": _source_add_control_commitment(
            readback
        ),
        "source_add_restart_guard_commitment": guard["guard_commitment"],
        "source_add_restart_guard_generation": str(
            rpc_result["guard_generation"]
        ),
        "source_add_restart_guard_owner_generation_commitment": str(
            rpc_result["owner_generation_commitment"]
        ),
        "source_add_restart_guard_restore_paused": (
            "true" if rpc_result["restore_paused"] else "false"
        ),
    }


def _renew_source_add_restart_guard(
    *,
    secrets_client: Any,
    restart_invocation_id: str,
    expected_current_version_id: Optional[str] = None,
    expected_guard_generation: Optional[Any] = None,
    expected_owner_generation_commitment: Optional[str] = None,
    expected_restore_paused: Optional[Any] = None,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, str]:
    _supabase_url, service_role_key = _source_add_pause_credentials(
        secrets_client=secrets_client,
        expected_current_version_id=expected_current_version_id,
    )
    acquired = _acquire_source_add_restart_guard(
        service_role_key=service_role_key,
        restart_invocation_id=restart_invocation_id,
        allow_takeover=False,
        expected_guard_generation=expected_guard_generation,
        expected_owner_generation_commitment=(
            expected_owner_generation_commitment
        ),
        connection_factory=connection_factory,
    )
    if (
        expected_restore_paused is not None
        and acquired["restore_paused"]
        is not _source_add_expected_restore_paused(
            expected_restore_paused,
            label="SOURCE_ADD restart restore state changed before renewal",
        )
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart restore state changed before renewal"
        )
    return {
        "status": "renewed",
        "source_add_restart_guard_commitment": str(
            acquired["guard_commitment"]
        ),
        "source_add_restart_guard_generation": str(
            acquired["guard_generation"]
        ),
        "source_add_restart_guard_owner_generation_commitment": str(
            acquired["owner_generation_commitment"]
        ),
        "source_add_restart_guard_restore_paused": (
            "true" if acquired["restore_paused"] else "false"
        ),
    }


def _normalized_source_add_restart_guard_release(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "released",
        "paused",
        "guard_active",
        "guard_generation",
        "owner_generation_commitment",
        "restored_pre_restart_state",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema_version")
        != "leadpoet.source_add_restart_guard_release.v2"
        or value.get("released") is not True
        or not isinstance(value.get("paused"), bool)
        or value.get("guard_active") is not False
        or value.get("restored_pre_restart_state") is not True
        or not isinstance(value.get("owner_generation_commitment"), str)
        or not _SHA256_RE.fullmatch(
            str(value.get("owner_generation_commitment"))
        )
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard release response is invalid"
        )
    _source_add_guard_generation(
        value.get("guard_generation"),
        label="SOURCE_ADD restart guard release response is invalid",
    )
    return {name: value[name] for name in sorted(fields)}


def _release_source_add_restart_guard(
    *,
    secrets_client: Any,
    restart_invocation_id: str,
    expected_current_version_id: Optional[str] = None,
    expected_guard_generation: Optional[Any] = None,
    expected_owner_generation_commitment: Optional[str] = None,
    expected_restore_paused: Optional[Any] = None,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, str]:
    _supabase_url, service_role_key = _source_add_pause_credentials(
        secrets_client=secrets_client,
        expected_current_version_id=expected_current_version_id,
    )
    guard, guard_state = _require_owned_source_add_guard_state(
        service_role_key=service_role_key,
        restart_invocation_id=restart_invocation_id,
        expected_guard_generation=expected_guard_generation,
        expected_owner_generation_commitment=(
            expected_owner_generation_commitment
        ),
        expected_restore_paused=expected_restore_paused,
        connection_factory=connection_factory,
    )
    generation = int(guard_state["guard_generation"])
    released = _normalized_source_add_restart_guard_release(
        _source_add_control_request(
            method="POST",
            path=f"/rest/v1/rpc/{SOURCE_ADD_RELEASE_RESTART_GUARD_RPC}",
            service_role_key=service_role_key,
            payload={
                "p_actor_ref": guard["actor_ref"],
                "p_guard_generation": generation,
                "p_guard_id": guard["guard_id"],
                "p_owner_id": guard["owner_id"],
            },
            connection_factory=connection_factory,
        )
    )
    expected_owner_generation = _source_add_owner_generation_commitment(
        guard["owner_commitment"], generation
    )
    if (
        released["guard_generation"] != generation
        or released["owner_generation_commitment"]
        != expected_owner_generation
        or released["paused"] is not guard_state["restore_paused"]
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard release response is invalid"
        )
    return {
        "status": (
            "released_restored_paused"
            if released["paused"]
            else "released_restored_active"
        ),
        "source_add_restart_guard_generation": str(generation),
        "source_add_restart_guard_owner_generation_commitment": (
            expected_owner_generation
        ),
        "source_add_restart_guard_restore_paused": (
            "true" if released["paused"] else "false"
        ),
    }


def _require_source_add_state(
    *,
    secrets_client: Any,
    expected_paused: bool,
    expected_current_version_id: Optional[str] = None,
    expected_control_commitment: Optional[str] = None,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, str]:
    _supabase_url, service_role_key = _source_add_pause_credentials(
        secrets_client=secrets_client,
        expected_current_version_id=expected_current_version_id,
    )
    _require_source_add_admission_control_contract(
        service_role_key=service_role_key,
        connection_factory=connection_factory,
    )
    control = _read_source_add_control(
        service_role_key=service_role_key,
        connection_factory=connection_factory,
    )
    if control["paused"] is not expected_paused:
        raise GatewayMinerMaintenanceRestartError(
            "durable SOURCE_ADD state differs from restart restoration"
        )
    commitment = _source_add_control_commitment(control)
    if (
        expected_control_commitment is not None
        and commitment != str(expected_control_commitment)
    ):
        raise GatewayMinerMaintenanceRestartError(
            "durable SOURCE_ADD pause differs from the invocation proof"
        )
    return {
        "status": "paused" if expected_paused else "active",
        "source_add_control_commitment": commitment,
    }


def _force_source_add_paused_after_restart_failure(
    *,
    secrets_client: Any,
    restart_invocation_id: str,
    expected_current_version_id: Optional[str] = None,
    connection_factory: Any = http.client.HTTPSConnection,
) -> None:
    """Fail closed if completion fails after guard release may have committed."""

    _supabase_url, service_role_key = _source_add_pause_credentials(
        secrets_client=secrets_client,
        expected_current_version_id=expected_current_version_id,
    )
    guard = _source_add_restart_guard_identity(restart_invocation_id)
    _source_add_control_request(
        method="POST",
        path=f"/rest/v1/rpc/{SOURCE_ADD_PAUSE_RPC}",
        service_role_key=service_role_key,
        payload={
            "p_actor_ref": guard["actor_ref"],
            "p_paused": True,
            "p_reason": "canonical_restart_completion_failed",
        },
        connection_factory=connection_factory,
    )
    control = _read_source_add_control(
        service_role_key=service_role_key,
        connection_factory=connection_factory,
    )
    if control["paused"] is not True:
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD fail-closed pause was not durable"
        )


def _require_source_add_paused(
    *,
    secrets_client: Any,
    expected_current_version_id: Optional[str] = None,
    expected_control_commitment: Optional[str] = None,
    connection_factory: Any = http.client.HTTPSConnection,
) -> dict[str, str]:
    return _require_source_add_state(
        secrets_client=secrets_client,
        expected_paused=True,
        expected_current_version_id=expected_current_version_id,
        expected_control_commitment=expected_control_commitment,
        connection_factory=connection_factory,
    )


def _require_runtime_source_add_closed(
    runtime_status: Mapping[str, Any],
    *,
    allow_legacy_missing_intake: bool = False,
) -> None:
    source_add = runtime_status.get("source_add")
    control = source_add.get("control") if isinstance(source_add, Mapping) else None
    intake_is_closed = (
        isinstance(source_add, Mapping)
        and (
            source_add.get("intake_enabled") is False
            or (
                allow_legacy_missing_intake
                and "intake_enabled" not in source_add
            )
        )
    )
    if (
        not isinstance(source_add, Mapping)
        or not isinstance(control, Mapping)
        or control.get("paused") is not True
        or control.get("unavailable") is not False
        or not intake_is_closed
        or source_add.get("effective_dispatcher_enabled") is not False
    ):
        raise GatewayMinerMaintenanceRestartError(
            "running gateway SOURCE_ADD intake is not durably paused"
        )


def _require_runtime_source_add_restored(
    runtime_status: Mapping[str, Any], *, expected_paused: bool
) -> None:
    if expected_paused:
        _require_runtime_source_add_closed(runtime_status)
        return
    source_add = runtime_status.get("source_add")
    control = source_add.get("control") if isinstance(source_add, Mapping) else None
    if (
        not isinstance(source_add, Mapping)
        or not isinstance(control, Mapping)
        or control.get("paused") is not False
        or control.get("unavailable") is not False
        or source_add.get("intake_enabled") is not True
        or source_add.get("effective_dispatcher_enabled") is not True
    ):
        raise GatewayMinerMaintenanceRestartError(
            "running gateway SOURCE_ADD intake did not restore active"
        )


def _require_pre_activation_runtime_source_add_closed() -> None:
    """Verify the still-running N-1 gateway before candidate activation."""

    _require_runtime_source_add_closed(
        _fetch_runtime_status(),
        allow_legacy_missing_intake=True,
    )


def _require_pre_hydration_runtime_source_add_closed(
    *,
    live_process_commitment: str,
) -> str:
    commitment = str(live_process_commitment)
    if not _SHA256_RE.fullmatch(commitment):
        raise GatewayMinerMaintenanceRestartError(
            "pre-hydration gateway process commitment is invalid"
        )
    if commitment == sha256_json({"status": "absent"}):
        return "gateway_absent"
    _require_runtime_source_add_closed(
        _fetch_runtime_status(),
        allow_legacy_missing_intake=True,
    )
    return "runtime_closed"


def _candidate_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_fixed_bootstrap_authority(
    environment: Mapping[str, str],
) -> None:
    if (
        str(environment.get("LEADPOET_GATEWAY_ENV_SECRET_ID") or GATEWAY_SECRET_ID)
        != GATEWAY_SECRET_ID
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


def _resolve_bootstrap_secrets_client(secrets_client: Any) -> Any:
    if secrets_client is not None:
        return secrets_client
    return _instance_role_secrets_client(
        environ={**os.environ, "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true"}
    )


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
    final_secret_result: Mapping[str, str],
    source_add_pause_result: Mapping[str, str],
    source_add_quiescence_result: Mapping[str, str],
    restart_invocation_id: str,
    live_process_commitment: str,
) -> dict[str, str]:
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
        "source_add_control_commitment": str(
            source_add_pause_result["source_add_control_commitment"]
        ),
        "source_add_restart_guard_commitment": str(
            source_add_pause_result[
                "source_add_restart_guard_commitment"
            ]
        ),
        "source_add_restart_guard_generation": str(
            source_add_pause_result[
                "source_add_restart_guard_generation"
            ]
        ),
        "source_add_restart_guard_owner_generation_commitment": str(
            source_add_pause_result[
                "source_add_restart_guard_owner_generation_commitment"
            ]
        ),
        "source_add_restart_guard_restore_paused": str(
            source_add_pause_result[
                "source_add_restart_guard_restore_paused"
            ]
        ),
        "source_add_quiescence_commitment": str(
            source_add_quiescence_result[
                "source_add_quiescence_commitment"
            ]
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
        "current_document_commitment",
        "current_hydrated_environment_commitment",
        "current_stage_topology_commitment",
        "source_add_control_commitment",
        "source_add_restart_guard_commitment",
        "source_add_restart_guard_owner_generation_commitment",
        "source_add_quiescence_commitment",
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
        or normalized["source_add_restart_guard_restore_paused"]
        not in {"true", "false"}
        or not _VERSION_ID_RE.fullmatch(normalized["current_secret_version_id"])
        or not re.fullmatch(
            r"[1-9][0-9]*",
            normalized["source_add_restart_guard_generation"],
        )
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


def _verify_proof_against_state(
    *,
    proof: Mapping[str, str],
    deploy_commit: str,
    candidate_tree_hash: str,
    client: Any,
    tree_evidence: Optional[Mapping[str, Any]] = None,
    restart_invocation_id: Optional[str] = None,
    live_process_commitment: Optional[str] = None,
    hydrated_environment_path: Optional[Path] = None,
) -> dict[str, str]:
    validated = _validate_proof_document(proof)
    if (
        validated["candidate_commit"] != str(deploy_commit).lower()
        or validated["candidate_tree_hash"] != str(candidate_tree_hash).lower()
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
    _require_source_add_paused(
        secrets_client=client,
        expected_current_version_id=validated["current_secret_version_id"],
        expected_control_commitment=validated[
            "source_add_control_commitment"
        ],
    )
    _require_source_add_quiescent(
        secrets_client=client,
        restart_invocation_id=validated["restart_invocation_id"],
        expected_current_version_id=validated["current_secret_version_id"],
        expected_guard_commitment=validated[
            "source_add_restart_guard_commitment"
        ],
        expected_guard_generation=validated[
            "source_add_restart_guard_generation"
        ],
        expected_owner_generation_commitment=validated[
            "source_add_restart_guard_owner_generation_commitment"
        ],
        expected_restore_paused=validated[
            "source_add_restart_guard_restore_paused"
        ],
        expected_quiescence_commitment=validated[
            "source_add_quiescence_commitment"
        ],
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
    _require_source_add_paused(
        secrets_client=client,
        expected_current_version_id=validated["current_secret_version_id"],
        expected_control_commitment=validated[
            "source_add_control_commitment"
        ],
    )
    _require_source_add_quiescent(
        secrets_client=client,
        restart_invocation_id=validated["restart_invocation_id"],
        expected_current_version_id=validated["current_secret_version_id"],
        expected_guard_commitment=validated[
            "source_add_restart_guard_commitment"
        ],
        expected_guard_generation=validated[
            "source_add_restart_guard_generation"
        ],
        expected_owner_generation_commitment=validated[
            "source_add_restart_guard_owner_generation_commitment"
        ],
        expected_restore_paused=validated[
            "source_add_restart_guard_restore_paused"
        ],
        expected_quiescence_commitment=validated[
            "source_add_quiescence_commitment"
        ],
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
        "source_add_restart_guard_commitment": validated[
            "source_add_restart_guard_commitment"
        ],
        "source_add_restart_guard_generation": validated[
            "source_add_restart_guard_generation"
        ],
        "source_add_restart_guard_owner_generation_commitment": validated[
            "source_add_restart_guard_owner_generation_commitment"
        ],
        "source_add_restart_guard_restore_paused": validated[
            "source_add_restart_guard_restore_paused"
        ],
        "source_add_quiescence_commitment": validated[
            "source_add_quiescence_commitment"
        ],
        "proof_hash": validated["proof_hash"],
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
    secrets_client = _resolve_bootstrap_secrets_client(secrets_client)
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
    source_add_pause_result = _pause_source_add_for_restart(
        secrets_client=secrets_client,
        restart_invocation_id=restart_invocation_id,
    )
    _require_pre_hydration_runtime_source_add_closed(
        live_process_commitment=live_process_commitment,
    )
    source_add_wait_result = _wait_for_source_add_quiescence(
        secrets_client=secrets_client,
        restart_invocation_id=restart_invocation_id,
        expected_guard_commitment=source_add_pause_result[
            "source_add_restart_guard_commitment"
        ],
        expected_guard_generation=source_add_pause_result[
            "source_add_restart_guard_generation"
        ],
        expected_owner_generation_commitment=source_add_pause_result[
            "source_add_restart_guard_owner_generation_commitment"
        ],
        expected_restore_paused=source_add_pause_result[
            "source_add_restart_guard_restore_paused"
        ],
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
    _require_source_add_paused(
        secrets_client=secrets_client,
        expected_current_version_id=str(final_result["current_version_id"]),
        expected_control_commitment=source_add_pause_result[
            "source_add_control_commitment"
        ],
    )
    source_add_quiescence_result = _require_source_add_quiescent(
        secrets_client=secrets_client,
        restart_invocation_id=restart_invocation_id,
        expected_current_version_id=str(final_result["current_version_id"]),
        expected_guard_commitment=source_add_pause_result[
            "source_add_restart_guard_commitment"
        ],
        expected_guard_generation=source_add_pause_result[
            "source_add_restart_guard_generation"
        ],
        expected_owner_generation_commitment=source_add_pause_result[
            "source_add_restart_guard_owner_generation_commitment"
        ],
        expected_restore_paused=source_add_pause_result[
            "source_add_restart_guard_restore_paused"
        ],
        expected_quiescence_commitment=source_add_wait_result[
            "source_add_quiescence_commitment"
        ],
    )
    _require_pre_hydration_runtime_source_add_closed(
        live_process_commitment=live_process_commitment,
    )
    proof = _proof_body(
        candidate_commit=expected_commit,
        tree_evidence=tree_evidence,
        final_secret_result=final_result,
        source_add_pause_result=source_add_pause_result,
        source_add_quiescence_result=source_add_quiescence_result,
        restart_invocation_id=restart_invocation_id,
        live_process_commitment=live_process_commitment,
    )
    _validate_proof_document(proof)
    return {
        "status": "prepared",
        "proof": proof,
        "tree_evidence": tree_evidence,
    }


def verify_gateway_miner_maintenance_state(
    *,
    deploy_commit: str,
    candidate_tree_hash: str,
    parent_environment: Mapping[str, str],
    secrets_client: Any = None,
    bind_live_process_to_proof: bool = True,
    acquire_source_add_restart_guard: bool = True,
    hydrated_environment_path: Path = CANONICAL_GATEWAY_ENV_PATH,
) -> dict[str, str]:
    """Verify false hydration and durable state before production shutdown."""

    _require_fixed_bootstrap_authority(parent_environment)
    _require_disabled_parent_environment(parent_environment)
    secrets_client = _resolve_bootstrap_secrets_client(secrets_client)
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
            client=secrets_client,
            restart_invocation_id=parent_environment.get(
                "GATEWAY_RESTART_INVOCATION_ID"
            ),
            live_process_commitment=live_process_commitment,
            hydrated_environment_path=hydrated_environment_path,
        )
    _live_gateway_restart_authority_commitment()
    current = _require_disabled_secret_readback(client=secrets_client)
    restart_invocation_id = str(
        parent_environment.get("GATEWAY_RESTART_INVOCATION_ID") or ""
    )
    source_add_acquisition: Optional[Mapping[str, str]] = None
    source_add_wait_result: Optional[Mapping[str, str]] = None
    if acquire_source_add_restart_guard:
        source_add_acquisition = _pause_source_add_for_restart(
            secrets_client=secrets_client,
            restart_invocation_id=restart_invocation_id,
        )
        source_add_wait_result = _wait_for_source_add_quiescence(
            secrets_client=secrets_client,
            restart_invocation_id=restart_invocation_id,
            expected_current_version_id=str(current["current_version_id"]),
            expected_guard_commitment=source_add_acquisition[
                "source_add_restart_guard_commitment"
            ],
            expected_guard_generation=source_add_acquisition[
                "source_add_restart_guard_generation"
            ],
            expected_owner_generation_commitment=source_add_acquisition[
                "source_add_restart_guard_owner_generation_commitment"
            ],
            expected_restore_paused=source_add_acquisition[
                "source_add_restart_guard_restore_paused"
            ],
        )
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
    source_add_pause = _require_source_add_paused(
        secrets_client=secrets_client,
        expected_current_version_id=str(current["current_version_id"]),
        expected_control_commitment=(
            source_add_acquisition["source_add_control_commitment"]
            if source_add_acquisition is not None
            else None
        ),
    )
    source_add_quiescence = _require_source_add_quiescent(
        secrets_client=secrets_client,
        restart_invocation_id=restart_invocation_id,
        expected_current_version_id=str(current["current_version_id"]),
        expected_guard_commitment=(
            source_add_acquisition[
                "source_add_restart_guard_commitment"
            ]
            if source_add_acquisition is not None
            else None
        ),
        expected_guard_generation=(
            source_add_acquisition[
                "source_add_restart_guard_generation"
            ]
            if source_add_acquisition is not None
            else None
        ),
        expected_owner_generation_commitment=(
            source_add_acquisition[
                "source_add_restart_guard_owner_generation_commitment"
            ]
            if source_add_acquisition is not None
            else None
        ),
        expected_restore_paused=(
            source_add_acquisition[
                "source_add_restart_guard_restore_paused"
            ]
            if source_add_acquisition is not None
            else None
        ),
        expected_quiescence_commitment=(
            source_add_wait_result[
                "source_add_quiescence_commitment"
            ]
            if source_add_wait_result is not None
            else None
        ),
    )
    return {
        "status": "durable_false_verified",
        "current_secret_version_id": str(current["current_version_id"]),
        "source_add_control_commitment": source_add_pause[
            "source_add_control_commitment"
        ],
        "source_add_quiescence_commitment": source_add_quiescence[
            "source_add_quiescence_commitment"
        ],
        "source_add_restart_guard_commitment": source_add_quiescence[
            "source_add_restart_guard_commitment"
        ],
        "source_add_restart_guard_generation": source_add_quiescence[
            "source_add_restart_guard_generation"
        ],
        "source_add_restart_guard_owner_generation_commitment": (
            source_add_quiescence[
                "source_add_restart_guard_owner_generation_commitment"
            ]
        ),
        "source_add_restart_guard_restore_paused": source_add_quiescence[
            "source_add_restart_guard_restore_paused"
        ],
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
        or not 1
        <= int(timeout_seconds)
        <= SOURCE_ADD_CANONICAL_COORDINATION_DEADLINE_SECONDS
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
    try:
        root_device = metadata.st_dev
        for directory, _names, _files, descriptor in os.fwalk(
            root,
            topdown=False,
            follow_symlinks=False,
        ):
            opened = os.fstat(descriptor)
            current = Path(directory).lstat()
            if (
                not stat.S_ISDIR(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or opened.st_dev != root_device
                or opened.st_dev != current.st_dev
                or opened.st_ino != current.st_ino
                or stat.S_IMODE(opened.st_mode) & 0o022
            ):
                raise GatewayMinerMaintenanceRestartError(
                    "miner-maintenance bootstrap tree member is unsafe"
                )
            os.fchmod(descriptor, 0o700)
            hardened = os.fstat(descriptor)
            final_path = Path(directory).lstat()
            if (
                hardened.st_dev != opened.st_dev
                or hardened.st_ino != opened.st_ino
                or stat.S_IMODE(hardened.st_mode) != 0o700
                or final_path.st_dev != opened.st_dev
                or final_path.st_ino != opened.st_ino
            ):
                raise GatewayMinerMaintenanceRestartError(
                    "miner-maintenance bootstrap tree changed while closing"
                )
        final_root = root.lstat()
        if (
            final_root.st_dev != metadata.st_dev
            or final_root.st_ino != metadata.st_ino
        ):
            raise GatewayMinerMaintenanceRestartError(
                "miner-maintenance bootstrap tree changed while closing"
            )
        shutil.rmtree(root)
    except GatewayMinerMaintenanceRestartError:
        raise
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance bootstrap tree could not be removed"
        ) from exc
    try:
        root.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise GatewayMinerMaintenanceRestartError(
            "miner-maintenance bootstrap tree removal is unverified"
        ) from exc
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


def _controller_exec_environment(
    parent_environment: Mapping[str, str],
) -> dict[str, str]:
    """Carry the already-proved maintenance control across controller exec."""

    environment = dict(parent_environment)
    environment[TARGET_ENV_NAME] = TARGET_ENV_VALUE
    return environment


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
) -> None:
    """Prepare, fence, and exec the immutable installed N-1 controller."""

    proof_fd_open = False
    controller_fds_open = False
    cleaned = False
    try:
        _require_canonical_restart_lock_fd()
        _require_reserved_memfd_numbers_available()
        _require_fixed_bootstrap_authority(os.environ)
        secrets_client = _resolve_bootstrap_secrets_client(secrets_client)
        prepared = prepare_gateway_miner_maintenance_restart(
            repo_root=repo_root,
            candidate_root=candidate_root,
            plan_file=plan_file,
            expected_commit=expected_commit,
            controller_current=controller_current,
            host_restart_path=host_restart_path,
            restart_invocation_id=restart_invocation_id,
            secrets_client=secrets_client,
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
            # The paired operator builds both exact runtime releases after this
            # bootstrap proves the durable pause.  Keep this wait on the same
            # bounded deadline as that existing paired coordination window.
            timeout_seconds=SOURCE_ADD_CANONICAL_COORDINATION_DEADLINE_SECONDS,
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
        secrets_client = _resolve_bootstrap_secrets_client(secrets_client)
        _verify_proof_against_state(
            proof=_proof_from_fd(PROOF_FD_NUMBER),
            deploy_commit=expected_commit,
            candidate_tree_hash=str(final_tree["tree_hash"]),
            client=secrets_client,
            tree_evidence=final_tree,
            restart_invocation_id=restart_invocation_id,
            live_process_commitment=(
                _pre_hydration_live_process_commitment(final_tree)
            ),
        )
        _require_pre_activation_runtime_source_add_closed()
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
        # The exact candidate has already proved the durable false value and
        # bound it into the sealed invocation proof.  The installed N-1 shell
        # otherwise inherits its pre-mutation parent environment until its
        # prepared runtime clone is loaded.  Carry only this fixed, proved
        # control across exec; the destructive-boundary verifier independently
        # rereads both the durable secret and sealed proof before shutdown.
        environment = _controller_exec_environment(os.environ)
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


def verify_gateway_miner_maintenance_shutdown_quiescence(
    *,
    deploy_commit: str,
    parent_environment: Mapping[str, str],
    secrets_client: Any = None,
) -> dict[str, str]:
    """Recheck the guarded zero-lease state at the destructive boundary."""

    commit = str(deploy_commit).lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayMinerMaintenanceRestartError("candidate commit is invalid")
    _require_fixed_bootstrap_authority(parent_environment)
    _require_disabled_parent_environment(parent_environment)
    if secrets_client is None:
        secrets_client = _instance_role_secrets_client(
            environ={
                **parent_environment,
                "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
            }
        )
    restart_invocation_id = str(
        parent_environment.get("GATEWAY_RESTART_INVOCATION_ID") or ""
    )
    proof_fd = _proof_fd_from_environment(parent_environment)
    proof: Optional[dict[str, str]] = None
    expected_version_id: Optional[str] = None
    expected_control_commitment: Optional[str] = None
    expected_guard_commitment: Optional[str] = None
    expected_guard_generation: Optional[str] = None
    expected_owner_generation_commitment: Optional[str] = None
    expected_restore_paused: Optional[str] = None
    expected_quiescence_commitment: Optional[str] = None
    if proof_fd is not None:
        proof = _validate_proof_document(_proof_from_fd(proof_fd))
        if (
            proof["candidate_commit"] != commit
            or proof["restart_invocation_id"] != restart_invocation_id
        ):
            raise GatewayMinerMaintenanceRestartError(
                "miner-maintenance invocation proof differs at shutdown"
            )
        expected_version_id = proof["current_secret_version_id"]
        expected_control_commitment = proof[
            "source_add_control_commitment"
        ]
        expected_guard_commitment = proof[
            "source_add_restart_guard_commitment"
        ]
        expected_guard_generation = proof[
            "source_add_restart_guard_generation"
        ]
        expected_owner_generation_commitment = proof[
            "source_add_restart_guard_owner_generation_commitment"
        ]
        expected_restore_paused = proof[
            "source_add_restart_guard_restore_paused"
        ]
        expected_quiescence_commitment = proof[
            "source_add_quiescence_commitment"
        ]
    current = _require_disabled_secret_readback(
        client=secrets_client,
        expected_current_version_id=expected_version_id,
    )
    if proof is not None and (
        current.get("current_document_commitment")
        != proof["current_document_commitment"]
        or current.get("current_stage_topology_commitment")
        != proof["current_stage_topology_commitment"]
    ):
        raise GatewayMinerMaintenanceRestartError(
            "durable miner-maintenance state differs at shutdown"
        )
    renewed = _renew_source_add_restart_guard(
        secrets_client=secrets_client,
        restart_invocation_id=restart_invocation_id,
        expected_current_version_id=str(current["current_version_id"]),
        expected_guard_generation=expected_guard_generation,
        expected_owner_generation_commitment=(
            expected_owner_generation_commitment
        ),
        expected_restore_paused=expected_restore_paused,
    )
    if (
        expected_guard_commitment is not None
        and renewed["source_add_restart_guard_commitment"]
        != expected_guard_commitment
    ):
        raise GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD restart guard differs at shutdown"
        )
    _require_source_add_paused(
        secrets_client=secrets_client,
        expected_current_version_id=str(current["current_version_id"]),
        expected_control_commitment=expected_control_commitment,
    )
    quiescence = _require_source_add_quiescent(
        secrets_client=secrets_client,
        restart_invocation_id=restart_invocation_id,
        expected_current_version_id=str(current["current_version_id"]),
        expected_guard_commitment=renewed[
            "source_add_restart_guard_commitment"
        ],
        expected_guard_generation=renewed[
            "source_add_restart_guard_generation"
        ],
        expected_owner_generation_commitment=renewed[
            "source_add_restart_guard_owner_generation_commitment"
        ],
        expected_restore_paused=renewed[
            "source_add_restart_guard_restore_paused"
        ],
        expected_quiescence_commitment=expected_quiescence_commitment,
    )
    return {
        **quiescence,
        "status": "shutdown_quiescence_verified",
        "current_secret_version_id": str(current["current_version_id"]),
    }


def _require_runtime_miner_disabled(runtime_status: Mapping[str, Any]) -> None:
    if runtime_status.get("miner_submissions_enabled") is not False:
        raise GatewayMinerMaintenanceRestartError(
            "running gateway has miner submissions enabled"
        )


def verify_gateway_miner_maintenance_runtime_state(
    *,
    deploy_commit: str,
    candidate_tree_hash: str,
    runtime_environment: Mapping[str, str],
    runtime_status: Mapping[str, Any],
    secrets_client: Any = None,
    hydrated_environment_path: Path = CANONICAL_GATEWAY_ENV_PATH,
) -> dict[str, str]:
    """Recheck the exact false state against the activated live runtime."""

    _require_fixed_bootstrap_authority(runtime_environment)
    _require_disabled_parent_environment(runtime_environment)
    _require_runtime_miner_disabled(runtime_status)
    _require_runtime_source_add_closed(runtime_status)
    secrets_client = _resolve_bootstrap_secrets_client(secrets_client)
    result = verify_gateway_miner_maintenance_state(
        deploy_commit=deploy_commit,
        candidate_tree_hash=candidate_tree_hash,
        parent_environment=runtime_environment,
        secrets_client=secrets_client,
        bind_live_process_to_proof=False,
        acquire_source_add_restart_guard=False,
        hydrated_environment_path=hydrated_environment_path,
    )
    restart_invocation_id = str(
        runtime_environment.get("GATEWAY_RESTART_INVOCATION_ID") or ""
    )
    try:
        released = _release_source_add_restart_guard(
            secrets_client=secrets_client,
            restart_invocation_id=restart_invocation_id,
            expected_current_version_id=result["current_secret_version_id"],
            expected_guard_generation=result[
                "source_add_restart_guard_generation"
            ],
            expected_owner_generation_commitment=result[
                "source_add_restart_guard_owner_generation_commitment"
            ],
            expected_restore_paused=result[
                "source_add_restart_guard_restore_paused"
            ],
        )
        restored_paused = _source_add_expected_restore_paused(
            released["source_add_restart_guard_restore_paused"],
            label="SOURCE_ADD restart restoration result is invalid",
        )
        released_control = _require_source_add_state(
            secrets_client=secrets_client,
            expected_paused=restored_paused,
            expected_current_version_id=result["current_secret_version_id"],
        )
        _require_runtime_source_add_restored(
            _fetch_runtime_status(),
            expected_paused=restored_paused,
        )
    except (
        GatewayMinerMaintenanceRestartError,
        GatewayMinerSubmissionsDisableError,
    ) as exc:
        try:
            _force_source_add_paused_after_restart_failure(
                secrets_client=secrets_client,
                restart_invocation_id=restart_invocation_id,
                expected_current_version_id=result[
                    "current_secret_version_id"
                ],
            )
        except (
            GatewayMinerMaintenanceRestartError,
            GatewayMinerSubmissionsDisableError,
        ) as pause_exc:
            raise GatewayMinerMaintenanceRestartError(
                "SOURCE_ADD restart completion failed and the fail-closed "
                "pause could not be verified"
            ) from pause_exc
        raise
    return {
        **result,
        "source_add_control_commitment": released_control[
            "source_add_control_commitment"
        ],
        "runtime_status": "disabled",
        "source_add_restart_guard_status": released["status"],
    }


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
    mode.add_argument("--verify-shutdown-quiescence", action="store_true")
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
        if args.verify_shutdown_quiescence:
            if (
                args.release_manifest is not None
                or args.plan_file is not None
                or args.bootstrap_root is not None
                or args.handoff_file is not None
                or args.handoff_nonce is not None
            ):
                parser.error(
                    "--verify-shutdown-quiescence accepts no bootstrap or release arguments"
                )
            _verify_protected_source()
            result = verify_gateway_miner_maintenance_shutdown_quiescence(
                deploy_commit=expected_commit,
                parent_environment=os.environ,
            )
        else:
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
