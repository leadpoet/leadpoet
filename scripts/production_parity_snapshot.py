#!/usr/bin/env python3
"""Capture, verify, and restore encrypted production-parity snapshots safely."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity import (  # noqa: E402
    MIGRATION_RE,
    SNAPSHOT_SCHEMA_VERSION,
    ProductionParityError,
    file_sha256,
    migration_sequence,
    migration_delta,
    production_database_host_hash,
    safe_database_target,
    sha256_bytes,
    sha256_json,
    validate_archive,
    validate_contract,
    validate_snapshot_manifest,
)


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PINNED_POSTGRES_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}$"
)
_POSTGRES_CLIENT_TOOLS = frozenset({"psql", "pg_dump", "pg_restore"})
_POSTGRES_ENVIRONMENT_KEYS = (
    "PGHOST",
    "PGPORT",
    "PGDATABASE",
    "PGUSER",
    "PGPASSWORD",
    "PGSSLMODE",
    "PGOPTIONS",
)
_POSTGRES_ARCHIVE_TARGET = "/leadpoet-parity.snapshot"
_POSTGRES_MIGRATION_TARGET = "/leadpoet-parity.migration.sql"
_SOURCE_ADD_RESTART_STATE_MIGRATION = (
    "scripts/174-research-lab-source-add-restart-state-restore.sql"
)
_SOURCE_ADD_PROVENANCE_LEG1_MIGRATION = (
    "scripts/175-research-lab-source-add-provenance-leg1.sql"
)
_SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_MIGRATION = (
    "scripts/176-research-lab-source-add-provenance-origin-repair.sql"
)
_SOURCE_ADD_PROVENANCE_AUTHORITY_ACL_MIGRATION = (
    "scripts/177-research-lab-source-add-provenance-authority-acl.sql"
)
_SOURCE_ADD_MINER_STATUS_MIGRATION = (
    "scripts/178-research-lab-source-add-miner-status.sql"
)
_SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS = (
    {
        "path": "scripts/72-research-lab-source-experiments.sql",
        "sequence": 72,
        "sha256": "sha256:9335ab9ab320d0b95783f585626e010a69e0be74ac5d10aa55bc388d0a2df0a9",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/74-research-lab-source-add-provenance-precheck.sql",
        "sequence": 74,
        "sha256": "sha256:c802039521fb85f222605cc2dd081f63e1b83676080ce19147a2ace954898196",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/78-research-lab-source-add-catalog-provisioning.sql",
        "sequence": 78,
        "sha256": "sha256:7693418cc05410d8b674d76ef9f571c01121d90e53160773a2e81729dd430c9b",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/79-research-lab-source-add-llm-leg2-evidence.sql",
        "sequence": 79,
        "sha256": "sha256:641c369aad18087a93b0f203abec295f2e8a155b7d6291127e830680f2975b27",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/82-research-lab-source-add-llm-only-leg2.sql",
        "sequence": 82,
        "sha256": "sha256:b18b7fbc350a4e597875ee167f99dfa12b2d3e212f4ef148f2562fab86144b42",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/84-expand-source-add-source-kinds.sql",
        "sequence": 84,
        "sha256": "sha256:c646f66bedb25182542c574917c166626a47020c2a0bb4c3289eff512c6e4c56",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/86-research-lab-attested-v2-authority.sql",
        "sequence": 86,
        "sha256": "sha256:71dfeac1bcad6c0532cd7412d2ce2530a5a8020de579b4701856329c8210a80e",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/96-research-lab-source-add-functional-workflow.sql",
        "sequence": 96,
        "sha256": "sha256:4ffe42ea3265d5ec65f94d4ae58e01e095db02969e058302227987b42115cdc0",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/145-research-lab-source-add-admission-control.sql",
        "sequence": 145,
        "sha256": "sha256:cc249443e62bd9868ada13fc12be84222fe2beb649e909c726f0bdd7f343ef21",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "sequence": 169,
        "sha256": "sha256:bd811a7d909e6bac3a007ad0dc560aa6d872aad037bdb2fed978a9d2614a1add",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/170-research-lab-source-add-provider-origin-uniqueness.sql",
        "sequence": 170,
        "sha256": "sha256:cc79c740d3a4dfa4da2e7e2072c3ac7aa0d76529c9db0f40d49dce40a5f904fc",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/171-research-lab-source-add-duplicate-privacy.sql",
        "sequence": 171,
        "sha256": "sha256:15954b986b79252a0d52a551980e07d667e06720a447dd34d974f3aeaaf5defa",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/172-research-lab-source-add-claim-control.sql",
        "sequence": 172,
        "sha256": "sha256:96e68911a74e37c20a48fd3ed6221e9342339b5cbf5a3fdeb7fa88fb2f2f0327",
        "transaction_mode": "candidate-file",
    },
    {
        "path": "scripts/173-research-lab-source-add-leg1-release-policy.sql",
        "sequence": 173,
        "sha256": "sha256:e5e277f3f15730ce6793e020a4b066a0590c85200d3665cd63f8f40cb2fa045f",
        "transaction_mode": "candidate-file",
    },
    {
        "path": _SOURCE_ADD_RESTART_STATE_MIGRATION,
        "sequence": 174,
        "sha256": "sha256:766c06dc0de169e065a3114d6d4ed554d13bfcb063fd7d428a2644b5a861cb0b",
        "transaction_mode": "candidate-file",
    },
    {
        "path": _SOURCE_ADD_PROVENANCE_LEG1_MIGRATION,
        "sequence": 175,
        "sha256": "sha256:aac95bcdd7ea7dfb263b721e879bb8f2332ea0015415ed3631ce09429843ac50",
        "transaction_mode": "candidate-file",
    },
    {
        "path": _SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_MIGRATION,
        "sequence": 176,
        "sha256": "sha256:9ec8d3d9bc9412c285ac780c42fd9aa283d3705cd54a155dac65313cc051d1f8",
        "transaction_mode": "candidate-file",
    },
    {
        "path": _SOURCE_ADD_PROVENANCE_AUTHORITY_ACL_MIGRATION,
        "sequence": 177,
        "sha256": "sha256:5589d4e10c0932c2b85913df425f4f19eb7542c5a8cc5560573e3c797f223b32",
        "transaction_mode": "candidate-file",
    },
    {
        "path": _SOURCE_ADD_MINER_STATUS_MIGRATION,
        "sequence": 178,
        "sha256": "sha256:3cbeaa65110d8efc9281a7c1c952c343dfed933a9c23be8e2083513d701f2b40",
        "transaction_mode": "candidate-file",
    },
)
_SCHEMA_ONLY_SOURCE_ADD_ACL_SCHEMA_VERSION = (
    "leadpoet.production_parity.schema_only_source_add_acl.v6"
)
_SOURCE_ADD_DUPLICATE_PRIVACY_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:26bf34c94725b855f81c2e48b6afbd72d68db36a4aeffb5642494a5da32233e0"
)
_SOURCE_ADD_POST_ACCEPT_LEG1_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:fe7df9f9336217f3e738f420fae0d9720959042080df431c1bcb2d4baa8ee954"
)
_SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256 = (
    "sha256:208de2069d2b44826fe466de01a2d1a91f4c762869227b39bdba969c8586be16"
)
_SOURCE_ADD_PROVENANCE_LEG1_VIEW_AUTHORITY_SHA256 = (
    "sha256:36380661634fee55bbdb69631d81ee0872f96de9d1373a253d1b02db242f037a"
)
_SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:700345ac44ebad77f4568e6c80458238129fd4af6c9ada66d7558d1bca5c9491"
)
_SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:1082a75d70849b072299929ff00999b5c78a69adc9c7b03e544640ed60b02ff8"
)
_SCHEMA_ONLY_SOURCE_ADD_SERVICE_FUNCTIONS = (
    "public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)",
    "public.research_lab_source_add_acquire_restart_guard_v2(text,text,bigint,integer,text)",
    "public.research_lab_source_add_admission_control_contract_v1()",
    "public.research_lab_source_add_admit(jsonb,text,text,text,text,integer,integer,integer)",
    "public.research_lab_source_add_admit_v2(jsonb,text,text,text,text,text,integer,integer,integer)",
    "public.research_lab_source_add_admit_v3(jsonb,text,text,text,text,text,integer,integer,integer,integer)",
    "public.research_lab_source_add_begin_provider_execution(text,uuid)",
    "public.research_lab_source_add_claim_control_contract_v1()",
    "public.research_lab_source_add_claim_control_contract_v2()",
    "public.research_lab_source_add_claim_work(text,integer)",
    "public.research_lab_source_add_configure_probe_v2(text,text,jsonb,jsonb,text,text,text)",
    "public.research_lab_source_add_configure_probe_v3(text,text,jsonb,jsonb,text,text,text)",
    "public.research_lab_source_add_duplicate_privacy_contract_v1()",
    "public.research_lab_source_add_enqueue_leg1_after_provenance_v1(text,text,text,text,text)",
    "public.research_lab_source_add_enqueue_provision_smoke(text,text,text,text,jsonb,jsonb)",
    "public.research_lab_source_add_enqueue_provision_smoke_v2(text,text,text,text,jsonb,jsonb)",
    "public.research_lab_source_add_finalize_leg1_v4(text,text,uuid,uuid,integer,jsonb,jsonb)",
    "public.research_lab_source_add_finalize_provision_v2(text,jsonb,jsonb,jsonb)",
    "public.research_lab_source_add_finalize_provision_v3(text,jsonb,jsonb,jsonb)",
    "public.research_lab_source_add_finalize_provision_smoke_v3(text,uuid,text,jsonb,jsonb,jsonb)",
    "public.research_lab_source_add_reject_current_builtin_v2(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)",
    "public.research_lab_source_add_reject_current_builtin_v3(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)",
    "public.research_lab_source_add_reserve_leg1_slot_v2(text,text,uuid,integer,integer)",
    "public.research_lab_source_add_finalize_leg1_v2(text,text,uuid,uuid,integer,jsonb,jsonb)",
    "public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)",
    "public.research_lab_source_add_post_accept_leg1_contract_v1()",
    "public.research_lab_source_add_reserve_leg1_slot_v3(text,text,uuid,integer,integer)",
    "public.research_lab_source_add_finalize_leg1_v3(text,text,uuid,uuid,integer,jsonb,jsonb)",
    "public.research_lab_source_add_post_accept_leg1_contract_v2()",
    "public.research_lab_source_add_post_accept_leg1_contract_v3()",
    "public.research_lab_source_add_post_accept_leg1_contract_v4()",
    "public.research_lab_source_add_provider_origin_hash_v1(text)",
    "public.research_lab_source_add_provider_origin_host_v1(text)",
    "public.research_lab_source_add_provider_origin_contract_v1()",
    "public.research_lab_source_add_finish_work(text,uuid,text,text,jsonb,text,jsonb,jsonb,jsonb,jsonb,jsonb,jsonb,timestamp with time zone,boolean)",
    "public.research_lab_source_add_release_restart_guard_v1(text,text,bigint,text)",
    "public.research_lab_source_add_release_restart_guard_v2(text,text,bigint,text)",
    "public.research_lab_source_add_requeue_provenance(text,text,text,text,text,text)",
    "public.research_lab_source_add_requeue_provenance_v2(text,text,text,text,text,text,text)",
    "public.research_lab_source_add_reconcile_provenance_leg1_v1()",
    "public.research_lab_source_add_reserve_leg1_slot_v4(text,text,uuid,integer,integer)",
    "public.research_lab_source_add_miner_status_contract_v1()",
    "public.research_lab_source_add_miner_status_page_v1(text,text,integer)",
    "public.research_lab_source_add_restart_guard_state_v1()",
    "public.research_lab_source_add_restart_guard_state_v2()",
    "public.research_lab_source_add_restart_quiescence_v1(text,text,bigint)",
    "public.research_lab_source_add_set_paused(boolean,text,text)",
)
# These trigger-only functions retain PostgreSQL's default PUBLIC EXECUTE in
# the exact migration chain.  Parity must reproduce that effective ACL rather
# than silently hardening the clone into behavior production does not have.
_SCHEMA_ONLY_SOURCE_ADD_PUBLIC_FUNCTIONS = (
    "public.prevent_research_lab_source_add_history_mutation()",
    "public.prevent_research_lab_source_add_provisioning_mutation()",
    "public.prevent_research_lab_source_add_reward_mutation()",
)
_SCHEMA_ONLY_SOURCE_ADD_NON_SERVICE_FUNCTIONS = (
    "public.assert_research_lab_source_add_provider_origin_owner(text,text,text)",
    "public.enforce_research_lab_source_add_acceptance_v2()",
    "public.enforce_research_lab_source_add_admission_control()",
    "public.enforce_research_lab_source_add_eligible_v2()",
    "public.enforce_research_lab_source_add_eligible_v3()",
    "public.enforce_research_lab_source_add_leg1_initial_event_v3()",
    "public.enforce_research_lab_source_add_leg1_initial_event_v2()",
    "public.enforce_research_lab_source_add_leg1_obligation_v2()",
    "public.enforce_research_lab_source_add_leg1_obligation_v3()",
    "public.enforce_research_lab_source_add_leg1_slot_v2()",
    "public.enforce_research_lab_source_add_leg1_slot_v3()",
    "public.enforce_research_lab_source_add_leg1_work_v2()",
    "public.enforce_research_lab_source_add_leg1_work_v3()",
    "public.enforce_research_lab_source_add_provider_origin_submission()",
    "public.enforce_research_lab_source_add_provision_provider_origin()",
    "public.enforce_research_lab_source_catalog_provider_origin()",
    "public.enforce_source_add_restart_restore_pause_v2()",
    "public.prevent_research_lab_source_add_provider_origin_mutation()",
    "public.release_research_lab_source_add_provider_origin_terminal()",
    "public.research_lab_source_add_enqueue_provenance_leg1_trigger_v1()",
    "public.research_lab_source_add_canonical_jsonb_v2(jsonb)",
    "public.research_lab_source_add_configure_probe(text,text,jsonb,jsonb,text,text,text)",
    "public.research_lab_source_add_final_approval_catalog_v2(text)",
    "public.research_lab_source_add_finalize_provision(text,jsonb,jsonb,jsonb)",
    "public.research_lab_source_add_reserve_leg1_slot(text,text,uuid,integer,integer)",
    "public.research_lab_source_add_finalize_leg1(text,text,uuid,uuid,integer,jsonb,jsonb)",
    "public.research_lab_source_add_finalize_provision_smoke(text,uuid,text,jsonb,jsonb,jsonb)",
    "public.research_lab_source_add_jsonb_hash_v2(jsonb)",
    "public.research_lab_source_add_provenance_leg1_authority_matches_v1(text,text,text,text,text)",
)
_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_SCHEMA_VERSION = (
    "leadpoet.production_parity.schema_only_source_add_maintenance.v2"
)
_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_REASON = (
    "production_parity_fast_schema_only_source_add_cutover"
)
_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_ACTOR = (
    "operator:production-parity-fast-clone"
)
_SCHEMA_ONLY_SOURCE_ADD_CUTOVER_MIGRATIONS = tuple(
    migration
    for migration in _SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS
    if migration["path"]
    in {
        _SOURCE_ADD_RESTART_STATE_MIGRATION,
        _SOURCE_ADD_PROVENANCE_LEG1_MIGRATION,
        _SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_MIGRATION,
        _SOURCE_ADD_PROVENANCE_AUTHORITY_ACL_MIGRATION,
    }
)
_POSTGRES_MOUNT_TARGETS = frozenset(
    {_POSTGRES_ARCHIVE_TARGET, _POSTGRES_MIGRATION_TARGET}
)
DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS = 900
DEFAULT_CANDIDATE_MIGRATION_TIMEOUT_SECONDS = 300
DEFAULT_PINNED_POSTGRES_STARTUP_TIMEOUT_SECONDS = 600
MAX_SNAPSHOT_IO_TIMEOUT_SECONDS = 72_000
FULL_SNAPSHOT_DISK_RESERVE_BYTES = 64 * 1024**3


@dataclass(frozen=True)
class _PostgresClientMount:
    source: Path
    target: str
    read_only: bool


def _snapshot_io_timeout_seconds(value: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value > MAX_SNAPSHOT_IO_TIMEOUT_SECONDS
    ):
        raise ProductionParityError("production snapshot timeout is invalid")
    return value


def _require_full_snapshot_disk_headroom(
    path: Path,
    *,
    total_relation_bytes: int,
    simultaneous_copies: int,
) -> dict[str, int]:
    if (
        not isinstance(total_relation_bytes, int)
        or isinstance(total_relation_bytes, bool)
        or total_relation_bytes <= 0
        or simultaneous_copies not in {1, 2}
    ):
        raise ProductionParityError("full snapshot disk estimate is invalid")
    required = (
        total_relation_bytes * simultaneous_copies
        + FULL_SNAPSHOT_DISK_RESERVE_BYTES
    )
    try:
        available = shutil.disk_usage(path).free
    except OSError as exc:
        raise ProductionParityError(
            "full snapshot disk headroom is unavailable"
        ) from exc
    if available < required:
        raise ProductionParityError(
            "full snapshot disk headroom is insufficient: "
            f"required_bytes={required} available_bytes={available}"
        )
    return {
        "required_free_bytes": required,
        "available_free_bytes": available,
    }


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProductionParityError(f"{description} is unreadable") from exc
    if not isinstance(value, dict):
        raise ProductionParityError(f"{description} must be an object")
    return value


def _postgres_env(dsn: str, *, read_only: bool) -> tuple[dict[str, str], str]:
    parsed = urlparse(str(dsn or ""))
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ProductionParityError("PostgreSQL DSN is invalid")
    database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ProductionParityError("PostgreSQL DSN has no database")
    env = os.environ.copy()
    env.update(
        {
            "PGHOST": parsed.hostname,
            "PGPORT": str(parsed.port or 5432),
            "PGDATABASE": database,
            "PGUSER": unquote(parsed.username or ""),
            "PGPASSWORD": unquote(parsed.password or ""),
            "PGSSLMODE": "require",
        }
    )
    if read_only:
        env["PGOPTIONS"] = (
            "-c default_transaction_read_only=on "
            "-c statement_timeout=300000 "
            "-c lock_timeout=5000"
        )
    return env, parsed.hostname.lower()


def _run(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout: int,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(env),
        input=stdin,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _postgres_client_environment(
    env: Mapping[str, str], *, include_postgres: bool
) -> dict[str, str]:
    filtered = {"PATH": os.environ.get("PATH") or os.defpath}
    if include_postgres:
        for key in _POSTGRES_ENVIRONMENT_KEYS:
            if key in env:
                filtered[key] = str(env[key])
    return filtered


def _postgres_client_mount_argument(mount: _PostgresClientMount) -> str:
    if type(mount.read_only) is not bool:
        raise ProductionParityError("pinned PostgreSQL client mount mode is invalid")
    if mount.target not in _POSTGRES_MOUNT_TARGETS:
        raise ProductionParityError("pinned PostgreSQL client mount target is invalid")
    source = Path(mount.source)
    try:
        if source.is_symlink():
            raise ProductionParityError(
                "pinned PostgreSQL client mount source must not be a symlink"
            )
        resolved = source.resolve(strict=True)
        source_stat = resolved.stat()
    except OSError as exc:
        raise ProductionParityError(
            "pinned PostgreSQL client mount source is unavailable"
        ) from exc
    if not stat.S_ISREG(source_stat.st_mode) or "," in str(resolved):
        raise ProductionParityError(
            "pinned PostgreSQL client mount source is invalid"
        )
    if not mount.read_only and mount.target != _POSTGRES_ARCHIVE_TARGET:
        raise ProductionParityError(
            "pinned PostgreSQL client writable mount is invalid"
        )
    value = f"type=bind,src={resolved},dst={mount.target}"
    if mount.read_only:
        value += ",readonly"
    return value


def _redact_postgres_client_diagnostic(
    payload: bytes | None, env: Mapping[str, str]
) -> bytes | None:
    if payload is None:
        return None
    redacted = payload
    for key in _POSTGRES_ENVIRONMENT_KEYS:
        value = str(env.get(key) or "").encode("utf-8")
        if value:
            redacted = redacted.replace(value, b"[redacted]")
    return redacted


def _cleanup_interrupted_postgres_client(container_name: str) -> None:
    cleanup_env = _postgres_client_environment({}, include_postgres=False)
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            cwd=ROOT,
            env=cleanup_env,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
        pass
    try:
        absence = subprocess.run(
            [
                "docker",
                "container",
                "ls",
                "--all",
                "--quiet",
                "--filter",
                f"name=^/{container_name}$",
            ],
            cwd=ROOT,
            env=cleanup_env,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        raise ProductionParityError(
            "pinned PostgreSQL client cleanup could not be proven"
        ) from exc
    if absence.returncode != 0 or absence.stdout.strip():
        raise ProductionParityError(
            "pinned PostgreSQL client cleanup could not be proven"
        )


def _run_postgres(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout: int,
    stdin: bytes | None = None,
    postgres_image: str | None = None,
    mounts: Sequence[_PostgresClientMount] = (),
) -> subprocess.CompletedProcess[bytes]:
    if postgres_image is None:
        if mounts:
            raise ProductionParityError(
                "host PostgreSQL client cannot use container mounts"
            )
        return _run(command, env=env, timeout=timeout, stdin=stdin)
    if not _PINNED_POSTGRES_IMAGE_RE.fullmatch(postgres_image):
        raise ProductionParityError(
            "PostgreSQL client image must be digest-pinned"
        )
    if not command or command[0] not in _POSTGRES_CLIENT_TOOLS:
        raise ProductionParityError("PostgreSQL client command is not allowed")

    mount_arguments: list[str] = []
    mount_targets: set[str] = set()
    for mount in mounts:
        if (
            command[0] == "pg_dump"
            and (mount.target != _POSTGRES_ARCHIVE_TARGET or mount.read_only)
        ) or (
            command[0] == "pg_restore"
            and (mount.target != _POSTGRES_ARCHIVE_TARGET or not mount.read_only)
        ) or (
            command[0] == "psql"
            and (mount.target != _POSTGRES_MIGRATION_TARGET or not mount.read_only)
        ):
            raise ProductionParityError(
                "pinned PostgreSQL client mount does not match its command"
            )
        if mount.target in mount_targets:
            raise ProductionParityError(
                "pinned PostgreSQL client mount target is duplicated"
            )
        mount_targets.add(mount.target)
        mount_arguments.extend(
            ["--mount", _postgres_client_mount_argument(mount)]
        )

    container_name = (
        f"leadpoet-parity-pg-client-{os.getpid()}-{secrets.token_hex(6)}"
    )
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "host",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=67108864",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-i",
    ]
    for key in _POSTGRES_ENVIRONMENT_KEYS:
        if key in env:
            docker_command.extend(["--env", key])
    docker_command.extend(mount_arguments)
    docker_command.extend(
        ["--entrypoint", command[0], postgres_image, *list(command[1:])]
    )
    try:
        result = subprocess.run(
            docker_command,
            cwd=ROOT,
            env=_postgres_client_environment(env, include_postgres=True),
            input=stdin,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        _cleanup_interrupted_postgres_client(container_name)
        raise
    if result.returncode == 0:
        return result
    return subprocess.CompletedProcess(
        args=result.args,
        returncode=result.returncode,
        stdout=_redact_postgres_client_diagnostic(result.stdout, env),
        stderr=_redact_postgres_client_diagnostic(result.stderr, env),
    )


def _create_pinned_archive(path: Path) -> Path:
    try:
        parent = path.parent.resolve(strict=True)
        resolved = parent / path.name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
                raise ProductionParityError(
                    "pinned PostgreSQL archive file is invalid"
                )
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProductionParityError(
            "pinned PostgreSQL archive file could not be created safely"
        ) from exc
    return resolved


def _require_success(result: subprocess.CompletedProcess[bytes], *, stage: str) -> bytes:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()[-800:]
        raise ProductionParityError(f"{stage} failed: {detail}")
    return result.stdout


def _schema_only_source_add_maintenance_sql(
    migration: Mapping[str, Any],
) -> bytes:
    expected = next(
        (
            candidate
            for candidate in _SCHEMA_ONLY_SOURCE_ADD_CUTOVER_MIGRATIONS
            if candidate["path"] == migration.get("path")
        ),
        None,
    )
    if expected is None or dict(migration) != expected:
        raise ProductionParityError(
            "schema-only SOURCE_ADD maintenance migration identity differs"
        )
    return f"""
BEGIN;
SET LOCAL lock_timeout = '5s';
DO $schema_only_source_add_maintenance$
DECLARE
    v_pause JSONB;
BEGIN
    IF pg_catalog.to_regclass('public.research_lab_source_add_control') IS NULL
       OR pg_catalog.to_regclass('public.research_lab_source_add_work_items') IS NULL THEN
        RAISE EXCEPTION 'schema-only SOURCE_ADD maintenance relations are unavailable';
    END IF;
    LOCK TABLE public.research_lab_source_add_control
        IN ACCESS EXCLUSIVE MODE NOWAIT;
    LOCK TABLE public.research_lab_source_add_work_items
        IN SHARE ROW EXCLUSIVE MODE NOWAIT;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_source_add_control
    ) THEN
        RAISE EXCEPTION 'schema-only SOURCE_ADD control state is not empty';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_source_add_work_items
    ) THEN
        RAISE EXCEPTION 'schema-only SOURCE_ADD work state is not empty';
    END IF;
    INSERT INTO public.research_lab_source_add_control (
        singleton, paused, reason, actor_ref
    ) VALUES (
        TRUE,
        FALSE,
        'production_parity_fast_schema_only_active_seed',
        '{_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_ACTOR}'
    );
    SELECT public.research_lab_source_add_set_paused(
        TRUE,
        '{_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_REASON}',
        '{_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_ACTOR}'
    ) INTO v_pause;
    IF COALESCE((v_pause->>'paused')::BOOLEAN, FALSE) IS NOT TRUE
       OR v_pause->>'reason' <> '{_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_REASON}'
       OR v_pause->>'actor_ref' <> '{_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_ACTOR}' THEN
        RAISE EXCEPTION 'schema-only SOURCE_ADD pause RPC readback differs';
    END IF;
END;
$schema_only_source_add_maintenance$;
SELECT pg_catalog.json_build_object(
    'schema_version', '{_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_SCHEMA_VERSION}',
    'initial_paused', FALSE,
    'pause_rpc', 'research_lab_source_add_set_paused',
    'control_rows', (
        SELECT COUNT(*) FROM public.research_lab_source_add_control
    ),
    'work_rows', (
        SELECT COUNT(*) FROM public.research_lab_source_add_work_items
    ),
    'paused', (
        SELECT paused FROM public.research_lab_source_add_control WHERE singleton
    ),
    'guard_active', (
        SELECT restart_guard_commitment <> ''
        FROM public.research_lab_source_add_control
        WHERE singleton
    ),
    'guard_generation', (
        SELECT restart_guard_generation
        FROM public.research_lab_source_add_control
        WHERE singleton
    ),
    'reason_bound', (
        SELECT reason = '{_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_REASON}'
        FROM public.research_lab_source_add_control
        WHERE singleton
    ),
    'actor_bound', (
        SELECT actor_ref = '{_SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_ACTOR}'
        FROM public.research_lab_source_add_control
        WHERE singleton
    )
)::TEXT;
COMMIT;
""".encode("utf-8")


def _require_schema_only_source_add_acl_migrations(
    candidate_migrations: Sequence[Mapping[str, Any]],
) -> None:
    observed = {str(item.get("path") or ""): dict(item) for item in candidate_migrations}
    for expected in _SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS:
        if observed.get(expected["path"]) != expected:
            raise ProductionParityError(
                "schema-only SOURCE_ADD ACL migration identity differs: "
                f"{expected['path']}"
            )


def _schema_only_source_add_acl_expectations() -> dict[str, dict[str, bool]]:
    groups = (
        (
            _SCHEMA_ONLY_SOURCE_ADD_SERVICE_FUNCTIONS,
            {
                "service_role_callable": True,
                "public_callable": False,
                "anon_callable": False,
                "authenticated_callable": False,
            },
        ),
        (
            _SCHEMA_ONLY_SOURCE_ADD_PUBLIC_FUNCTIONS,
            {
                "service_role_callable": True,
                "public_callable": True,
                "anon_callable": True,
                "authenticated_callable": True,
            },
        ),
        (
            _SCHEMA_ONLY_SOURCE_ADD_NON_SERVICE_FUNCTIONS,
            {
                "service_role_callable": False,
                "public_callable": False,
                "anon_callable": False,
                "authenticated_callable": False,
            },
        ),
    )
    expectations: dict[str, dict[str, bool]] = {}
    for signatures, privileges in groups:
        for signature in signatures:
            if signature in expectations:
                raise ProductionParityError(
                    "schema-only SOURCE_ADD ACL inventory is duplicated"
                )
            expectations[signature] = dict(privileges)
    return expectations


def _schema_only_source_add_acl_migration(
    path: str,
) -> Mapping[str, Any]:
    matches = [
        migration
        for migration in _SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS
        if migration["path"] == path
    ]
    if len(matches) != 1:
        raise ProductionParityError(
            "schema-only SOURCE_ADD ACL migration inventory differs"
        )
    return matches[0]


def _schema_only_source_add_acl_sql(
    candidate_migrations: Sequence[Mapping[str, Any]],
) -> bytes:
    """Recreate every migration-bound SOURCE_ADD function ACL in the clone."""

    _require_schema_only_source_add_acl_migrations(candidate_migrations)
    expectations = _schema_only_source_add_acl_expectations()
    expected_rows = ",\n    ".join(
        "(" + ", ".join(
            (
                f"'{signature}'",
                str(privileges["service_role_callable"]).upper(),
                str(privileges["public_callable"]).upper(),
                str(privileges["anon_callable"]).upper(),
                str(privileges["authenticated_callable"]).upper(),
            )
        ) + ")"
        for signature, privileges in expectations.items()
    )
    all_functions = tuple(expectations)
    revoke_all = "\n".join(
        f"REVOKE ALL ON FUNCTION {signature} "
        "FROM PUBLIC, anon, authenticated, service_role;"
        for signature in all_functions
    )
    grant_service = "\n".join(
        f"GRANT EXECUTE ON FUNCTION {signature} TO service_role;"
        for signature in _SCHEMA_ONLY_SOURCE_ADD_SERVICE_FUNCTIONS
    )
    grant_public = "\n".join(
        f"GRANT EXECUTE ON FUNCTION {signature} TO PUBLIC;"
        for signature in _SCHEMA_ONLY_SOURCE_ADD_PUBLIC_FUNCTIONS
    )
    duplicate_privacy_migration = _schema_only_source_add_acl_migration(
        "scripts/171-research-lab-source-add-duplicate-privacy.sql"
    )
    provenance_leg1_migration = _schema_only_source_add_acl_migration(
        _SOURCE_ADD_PROVENANCE_LEG1_MIGRATION
    )
    provenance_origin_repair_migration = _schema_only_source_add_acl_migration(
        _SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_MIGRATION
    )
    provenance_authority_acl_migration = _schema_only_source_add_acl_migration(
        _SOURCE_ADD_PROVENANCE_AUTHORITY_ACL_MIGRATION
    )
    miner_status_migration = _schema_only_source_add_acl_migration(
        _SOURCE_ADD_MINER_STATUS_MIGRATION
    )
    return f"""
BEGIN;
SET LOCAL lock_timeout = '5s';
CREATE TEMPORARY TABLE schema_only_source_add_expected_acl (
    signature TEXT PRIMARY KEY,
    service_role_callable BOOLEAN NOT NULL,
    public_callable BOOLEAN NOT NULL,
    anon_callable BOOLEAN NOT NULL,
    authenticated_callable BOOLEAN NOT NULL
) ON COMMIT DROP;
INSERT INTO schema_only_source_add_expected_acl (
    signature, service_role_callable, public_callable,
    anon_callable, authenticated_callable
) VALUES
    {expected_rows};
DO $schema_only_source_add_acl_inventory$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'service_role'
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'anon'
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'authenticated'
    ) OR pg_catalog.to_regclass(
        'public.research_lab_source_add_miner_status_v1'
    ) IS NULL OR NOT COALESCE((
        SELECT class.reloptions @> ARRAY[
            'security_invoker=true', 'security_barrier=true'
        ]
        FROM pg_catalog.pg_class AS class
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = class.relnamespace
        WHERE namespace.nspname = 'public'
          AND class.relname = 'research_lab_source_add_miner_status_v1'
          AND class.relkind = 'v'
    ), FALSE
    ) THEN
        RAISE EXCEPTION
            'schema-only SOURCE_ADD ACL roles or miner status view are unavailable';
    END IF;
    IF (SELECT COUNT(*) FROM schema_only_source_add_expected_acl)
           <> {len(expectations)}
       OR EXISTS (
            SELECT 1
            FROM schema_only_source_add_expected_acl AS expected
            WHERE pg_catalog.to_regprocedure(expected.signature) IS NULL
       )
       OR (
            SELECT COUNT(*)
            FROM pg_catalog.pg_proc AS function_row
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = function_row.pronamespace
            WHERE namespace.nspname = 'public'
              AND (
                    pg_catalog.strpos(function_row.proname, 'source_add') > 0
                    OR function_row.proname =
                        'enforce_research_lab_source_catalog_provider_origin'
              )
       ) <> {len(expectations)}
       OR EXISTS (
            SELECT 1
            FROM pg_catalog.pg_proc AS function_row
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = function_row.pronamespace
            WHERE namespace.nspname = 'public'
              AND (
                    pg_catalog.strpos(function_row.proname, 'source_add') > 0
                    OR function_row.proname =
                        'enforce_research_lab_source_catalog_provider_origin'
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM schema_only_source_add_expected_acl AS expected
                    WHERE pg_catalog.to_regprocedure(expected.signature)
                        = function_row.oid
              )
       ) THEN
        RAISE EXCEPTION 'schema-only SOURCE_ADD ACL function inventory differs';
    END IF;
END;
$schema_only_source_add_acl_inventory$;
{revoke_all}
{grant_service}
{grant_public}
REVOKE ALL ON TABLE public.research_lab_source_add_miner_status_v1
    FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE public.research_lab_source_add_miner_status_v1
    TO service_role;
DO $schema_only_source_add_acl_readback$
BEGIN
    IF (SELECT COUNT(*) FROM schema_only_source_add_expected_acl)
           <> {len(expectations)}
       OR EXISTS (
            SELECT 1
            FROM schema_only_source_add_expected_acl AS expected
            WHERE pg_catalog.to_regprocedure(expected.signature) IS NULL
       )
       OR (
            SELECT COUNT(*)
            FROM pg_catalog.pg_proc AS function_row
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = function_row.pronamespace
            WHERE namespace.nspname = 'public'
              AND (
                    pg_catalog.strpos(function_row.proname, 'source_add') > 0
                    OR function_row.proname =
                        'enforce_research_lab_source_catalog_provider_origin'
              )
       ) <> {len(expectations)}
       OR EXISTS (
            SELECT 1
            FROM pg_catalog.pg_proc AS function_row
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = function_row.pronamespace
            WHERE namespace.nspname = 'public'
              AND (
                    pg_catalog.strpos(function_row.proname, 'source_add') > 0
                    OR function_row.proname =
                        'enforce_research_lab_source_catalog_provider_origin'
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM schema_only_source_add_expected_acl AS expected
                    WHERE pg_catalog.to_regprocedure(expected.signature)
                        = function_row.oid
              )
       )
       OR NOT pg_catalog.has_table_privilege(
            'service_role',
            'public.research_lab_source_add_miner_status_v1',
            'SELECT'
       )
       OR pg_catalog.has_table_privilege(
            'anon',
            'public.research_lab_source_add_miner_status_v1',
            'SELECT'
       )
       OR pg_catalog.has_table_privilege(
            'authenticated',
            'public.research_lab_source_add_miner_status_v1',
            'SELECT'
       )
       OR EXISTS (
            SELECT 1
            FROM pg_catalog.pg_class AS class
            CROSS JOIN LATERAL pg_catalog.aclexplode(
                COALESCE(
                    class.relacl,
                    pg_catalog.acldefault('r', class.relowner)
                )
            ) AS privilege
            WHERE class.oid = pg_catalog.to_regclass(
                'public.research_lab_source_add_miner_status_v1'
            )
              AND privilege.grantee = 0
              AND privilege.privilege_type = 'SELECT'
       )
       OR NOT COALESCE((
            SELECT class.reloptions @> ARRAY[
                'security_invoker=true', 'security_barrier=true'
            ]
            FROM pg_catalog.pg_class AS class
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = class.relnamespace
            WHERE namespace.nspname = 'public'
              AND class.relname = 'research_lab_source_add_miner_status_v1'
              AND class.relkind = 'v'
       ), FALSE)
       OR EXISTS (
        SELECT 1
        FROM schema_only_source_add_expected_acl AS expected
        JOIN pg_catalog.pg_proc AS function_row
          ON function_row.oid = pg_catalog.to_regprocedure(expected.signature)
        WHERE pg_catalog.has_function_privilege(
                  'service_role', function_row.oid, 'EXECUTE'
              ) <> expected.service_role_callable
           OR EXISTS (
                SELECT 1
                FROM pg_catalog.aclexplode(
                    COALESCE(
                        function_row.proacl,
                        pg_catalog.acldefault('f', function_row.proowner)
                    )
                ) AS privilege
                WHERE privilege.grantee = 0
                  AND privilege.privilege_type = 'EXECUTE'
           ) <> expected.public_callable
           OR pg_catalog.has_function_privilege(
                  'anon', function_row.oid, 'EXECUTE'
              ) <> expected.anon_callable
           OR pg_catalog.has_function_privilege(
                  'authenticated', function_row.oid, 'EXECUTE'
              ) <> expected.authenticated_callable
    ) THEN
        RAISE EXCEPTION 'schema-only SOURCE_ADD ACL readback differs';
    END IF;
END;
$schema_only_source_add_acl_readback$;
WITH contracts AS (
    SELECT
        public.research_lab_source_add_duplicate_privacy_contract_v1()
            AS duplicate_privacy,
        public.research_lab_source_add_post_accept_leg1_contract_v4()
            AS provenance_leg1,
        public.research_lab_source_add_claim_control_contract_v2()
            AS claim_control
), actual_acl AS (
    SELECT
        expected.signature,
        pg_catalog.has_function_privilege(
            'service_role', function_row.oid, 'EXECUTE'
        ) AS service_role_callable,
        EXISTS (
            SELECT 1
            FROM pg_catalog.aclexplode(
                COALESCE(
                    function_row.proacl,
                    pg_catalog.acldefault('f', function_row.proowner)
                )
            ) AS privilege
            WHERE privilege.grantee = 0
              AND privilege.privilege_type = 'EXECUTE'
        ) AS public_callable,
        pg_catalog.has_function_privilege(
            'anon', function_row.oid, 'EXECUTE'
        ) AS anon_callable,
        pg_catalog.has_function_privilege(
            'authenticated', function_row.oid, 'EXECUTE'
        ) AS authenticated_callable
    FROM schema_only_source_add_expected_acl AS expected
    JOIN pg_catalog.pg_proc AS function_row
      ON function_row.oid = pg_catalog.to_regprocedure(expected.signature)
)
SELECT pg_catalog.json_build_object(
    'schema_version', '{_SCHEMA_ONLY_SOURCE_ADD_ACL_SCHEMA_VERSION}',
    'migration_count', {len(_SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS)},
    'migration_171_sha256', '{duplicate_privacy_migration['sha256']}',
    'migration_175_sha256', '{provenance_leg1_migration['sha256']}',
    'migration_176_sha256', '{provenance_origin_repair_migration['sha256']}',
    'migration_177_sha256', '{provenance_authority_acl_migration['sha256']}',
    'migration_178_sha256', '{miner_status_migration['sha256']}',
    'function_signature_count', (SELECT COUNT(*) FROM actual_acl),
    'service_role_function_count', (
        SELECT COUNT(*) FROM actual_acl WHERE service_role_callable
    ),
    'non_service_role_function_count', (
        SELECT COUNT(*) FROM actual_acl WHERE NOT service_role_callable
    ),
    'public_function_count', (
        SELECT COUNT(*) FROM actual_acl WHERE public_callable
    ),
    'anon_callable_function_count', (
        SELECT COUNT(*) FROM actual_acl WHERE anon_callable
    ),
    'authenticated_callable_function_count', (
        SELECT COUNT(*) FROM actual_acl WHERE authenticated_callable
    ),
    'miner_status_view_acl_bound',
        pg_catalog.has_table_privilege(
            'service_role',
            'public.research_lab_source_add_miner_status_v1',
            'SELECT'
        )
        AND NOT pg_catalog.has_table_privilege(
            'anon',
            'public.research_lab_source_add_miner_status_v1',
            'SELECT'
        )
        AND NOT pg_catalog.has_table_privilege(
            'authenticated',
            'public.research_lab_source_add_miner_status_v1',
            'SELECT'
        )
        AND COALESCE((
            SELECT class.reloptions @> ARRAY[
                'security_invoker=true', 'security_barrier=true'
            ]
            FROM pg_catalog.pg_class AS class
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = class.relnamespace
            WHERE namespace.nspname = 'public'
              AND class.relname = 'research_lab_source_add_miner_status_v1'
              AND class.relkind = 'v'
        ), FALSE),
    'function_acl_inventory', (
        SELECT pg_catalog.jsonb_object_agg(
            signature,
            pg_catalog.jsonb_build_object(
                'service_role_callable', service_role_callable,
                'public_callable', public_callable,
                'anon_callable', anon_callable,
                'authenticated_callable', authenticated_callable
            )
        )
        FROM actual_acl
    ),
    'duplicate_privacy_authority_bound',
        contracts.duplicate_privacy->>'function_authority_sha256'
            = '{_SOURCE_ADD_DUPLICATE_PRIVACY_FUNCTION_AUTHORITY_SHA256}',
    'duplicate_privacy_permissions_bound',
        contracts.duplicate_privacy->'permissions' = pg_catalog.jsonb_build_object(
            'service_role_exists', TRUE,
            'v3_service_role_callable', TRUE,
            'v2_service_role_callable', TRUE,
            'contract_service_role_callable', TRUE,
            'anon_callable', FALSE,
            'authenticated_callable', FALSE
        ),
    'post_accept_leg1_authority_bound',
        contracts.provenance_leg1->>'function_authority_sha256'
            = '{_SOURCE_ADD_POST_ACCEPT_LEG1_FUNCTION_AUTHORITY_SHA256}',
    'provenance_leg1_trigger_authority_bound',
        contracts.provenance_leg1->>'trigger_authority_sha256'
            = '{_SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256}',
    'provenance_leg1_view_authority_bound',
        contracts.provenance_leg1->>'view_authority_sha256'
            = '{_SOURCE_ADD_PROVENANCE_LEG1_VIEW_AUTHORITY_SHA256}',
    'provenance_origin_repair_authority_bound',
        contracts.provenance_leg1->>'repair_function_authority_sha256'
            = '{_SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_FUNCTION_AUTHORITY_SHA256}',
    'provenance_leg1_policy_bound',
        contracts.provenance_leg1->'public_trigger_fields'
            = pg_catalog.jsonb_build_array(
                'precheck_status',
                'provenance_artifact_hash',
                'provenance_precheck_passed',
                'provenance_receipt_hash',
                'provenance_result_hash',
                'submission_id'
            )
        AND contracts.provenance_leg1->>'schema_version'
            = 'leadpoet.source_add_post_accept_leg1_contract.v4'
        AND contracts.provenance_leg1->>'required_migration'
            = 'scripts/176-research-lab-source-add-provenance-origin-repair.sql'
        AND (contracts.provenance_leg1->>'daily_cap')::INTEGER = 50
        AND (contracts.provenance_leg1->>'leg1_alpha_percent')::NUMERIC = 0.2
        AND (contracts.provenance_leg1->>'leg1_reward_epochs')::INTEGER = 20
        AND contracts.provenance_leg1->>'approval_boundary'
            = 'provenance_precheck_passed'
        AND contracts.provenance_leg1->>'backfill_policy'
            = 'earliest_exact_attested_provenance_per_provider_origin'
        AND contracts.provenance_leg1->>'provider_origin_scope'
            = 'normalized_exact_host'
        AND contracts.provenance_leg1->'provider_origin_winner_order'
            = pg_catalog.jsonb_build_array(
                'provenance_created_at', 'submission_id'
            )
        AND (contracts.provenance_leg1->>'cancelled_intents_are_authority')
            ::BOOLEAN IS FALSE
        AND contracts.provenance_leg1->>'authority_view'
            = 'research_lab_source_add_provenance_leg1_authority_v1'
        AND contracts.provenance_leg1->'functions'
            = pg_catalog.jsonb_build_object(
                'configure_probe_v3', TRUE,
                'enqueue_leg1_after_provenance_v1', TRUE,
                'enqueue_provision_smoke_v2', TRUE,
                'finalize_leg1_v4', TRUE,
                'finalize_provision_smoke_v3', TRUE,
                'finalize_provision_v3', TRUE,
                'reject_current_builtin_v3', TRUE,
                'reconcile_provenance_leg1_v1', TRUE,
                'reserve_leg1_slot_v4', TRUE
            )
        AND contracts.provenance_leg1->'triggers'
            = pg_catalog.jsonb_build_object(
                'automatic_enqueue', TRUE,
                'eligible_v2', TRUE,
                'eligible_v3', TRUE,
                'leg1_initial_event_v3', TRUE,
                'leg1_obligation_v3', TRUE,
                'leg1_slot_v3', TRUE,
                'leg1_work_v3', TRUE
            )
        AND contracts.provenance_leg1->'columns'
            = pg_catalog.jsonb_build_object(
                'intent_approval_kind', TRUE,
                'intent_provenance_artifact_hash', TRUE,
                'intent_provenance_receipt_hash', TRUE,
                'slot_approval_kind', TRUE
            ),
    'post_accept_leg1_permissions_bound',
        contracts.provenance_leg1->'permissions' = pg_catalog.jsonb_build_object(
            'service_role_exists', TRUE,
            'candidate_callable', TRUE,
            'rollback_v2_callable', TRUE,
            'internal_not_callable', TRUE
        ),
    'claim_control_authority_bound',
        contracts.claim_control->>'function_authority_sha256'
            = '{_SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256}',
    'claim_control_permissions_bound',
        contracts.claim_control->'permissions' = pg_catalog.jsonb_build_object(
            'service_role_exists', TRUE,
            'service_role_callable', TRUE,
            'anon_callable', FALSE,
            'authenticated_callable', FALSE
        )
)::TEXT
FROM contracts;
NOTIFY pgrst, 'reload schema';
COMMIT;
""".encode("utf-8")


def restore_schema_only_source_add_acl_contract(
    *,
    target_dsn: str,
    production_host: str,
    candidate_migrations: Sequence[Mapping[str, Any]],
    postgres_image: str | None = None,
) -> dict[str, Any]:
    """Restore and prove exact SOURCE_ADD ACLs in a schema-only clone."""

    safe_database_target(target_dsn, production_host=production_host)
    env, _ = _postgres_env(target_dsn, read_only=False)
    env["PGSSLMODE"] = "disable"
    raw = _require_success(
        _run_postgres(
            ["psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1"],
            env=env,
            timeout=DEFAULT_CANDIDATE_MIGRATION_TIMEOUT_SECONDS,
            stdin=_schema_only_source_add_acl_sql(candidate_migrations),
            postgres_image=postgres_image,
        ),
        stage="schema-only SOURCE_ADD ACL reconstruction",
    ).decode("utf-8", "strict").strip()
    try:
        observed = json.loads(raw)
    except ValueError as exc:
        raise ProductionParityError(
            "schema-only SOURCE_ADD ACL readback is invalid"
        ) from exc
    observed_inventory = observed.pop("function_acl_inventory", None)
    expected_inventory = _schema_only_source_add_acl_expectations()
    duplicate_privacy_migration = _schema_only_source_add_acl_migration(
        "scripts/171-research-lab-source-add-duplicate-privacy.sql"
    )
    provenance_leg1_migration = _schema_only_source_add_acl_migration(
        _SOURCE_ADD_PROVENANCE_LEG1_MIGRATION
    )
    provenance_origin_repair_migration = _schema_only_source_add_acl_migration(
        _SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_MIGRATION
    )
    provenance_authority_acl_migration = _schema_only_source_add_acl_migration(
        _SOURCE_ADD_PROVENANCE_AUTHORITY_ACL_MIGRATION
    )
    miner_status_migration = _schema_only_source_add_acl_migration(
        _SOURCE_ADD_MINER_STATUS_MIGRATION
    )
    expected = {
        "schema_version": _SCHEMA_ONLY_SOURCE_ADD_ACL_SCHEMA_VERSION,
        "migration_count": len(_SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS),
        "migration_171_sha256": duplicate_privacy_migration["sha256"],
        "migration_175_sha256": provenance_leg1_migration["sha256"],
        "migration_176_sha256": provenance_origin_repair_migration["sha256"],
        "migration_177_sha256": provenance_authority_acl_migration["sha256"],
        "migration_178_sha256": miner_status_migration["sha256"],
        "function_signature_count": len(expected_inventory),
        "service_role_function_count": (
            len(_SCHEMA_ONLY_SOURCE_ADD_SERVICE_FUNCTIONS)
            + len(_SCHEMA_ONLY_SOURCE_ADD_PUBLIC_FUNCTIONS)
        ),
        "non_service_role_function_count": len(
            _SCHEMA_ONLY_SOURCE_ADD_NON_SERVICE_FUNCTIONS
        ),
        "public_function_count": len(_SCHEMA_ONLY_SOURCE_ADD_PUBLIC_FUNCTIONS),
        "anon_callable_function_count": len(
            _SCHEMA_ONLY_SOURCE_ADD_PUBLIC_FUNCTIONS
        ),
        "authenticated_callable_function_count": len(
            _SCHEMA_ONLY_SOURCE_ADD_PUBLIC_FUNCTIONS
        ),
        "miner_status_view_acl_bound": True,
        "duplicate_privacy_authority_bound": True,
        "duplicate_privacy_permissions_bound": True,
        "post_accept_leg1_authority_bound": True,
        "provenance_leg1_trigger_authority_bound": True,
        "provenance_leg1_view_authority_bound": True,
        "provenance_origin_repair_authority_bound": True,
        "provenance_leg1_policy_bound": True,
        "post_accept_leg1_permissions_bound": True,
        "claim_control_authority_bound": True,
        "claim_control_permissions_bound": True,
    }
    if observed != expected or observed_inventory != expected_inventory:
        raise ProductionParityError(
            "schema-only SOURCE_ADD ACL readback differs"
        )
    return {
        **expected,
        "function_acl_inventory_sha256": sha256_json(
            {"functions": expected_inventory}
        ),
    }


def _stage_schema_only_source_add_maintenance(
    *,
    env: Mapping[str, str],
    migration: Mapping[str, Any],
    postgres_image: str | None,
) -> dict[str, Any]:
    """Reconstruct omitted active control and pause through the production RPC."""

    sql = _schema_only_source_add_maintenance_sql(migration)
    raw = _require_success(
        _run_postgres(
            ["psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1"],
            env=env,
            timeout=DEFAULT_CANDIDATE_MIGRATION_TIMEOUT_SECONDS,
            stdin=sql,
            postgres_image=postgres_image,
        ),
        stage="schema-only SOURCE_ADD maintenance staging",
    ).decode("utf-8", "strict").strip()
    try:
        observed = json.loads(raw)
    except ValueError as exc:
        raise ProductionParityError(
            "schema-only SOURCE_ADD maintenance readback is invalid"
        ) from exc
    expected = {
        "schema_version": _SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_SCHEMA_VERSION,
        "initial_paused": False,
        "pause_rpc": "research_lab_source_add_set_paused",
        "control_rows": 1,
        "work_rows": 0,
        "paused": True,
        "guard_active": False,
        "guard_generation": 0,
        "reason_bound": True,
        "actor_bound": True,
    }
    if observed != expected:
        raise ProductionParityError(
            "schema-only SOURCE_ADD maintenance readback differs"
        )
    return {
        **expected,
        "migration_path": migration["path"],
        "migration_sha256": migration["sha256"],
    }


def _database_stats(
    env: Mapping[str, str], *, postgres_image: str | None = None
) -> dict[str, Any]:
    stats_sql = """
SELECT json_build_object(
  'server_version_num', current_setting('server_version_num'),
  'relation_count', COUNT(*),
  'total_relation_bytes', COALESCE(SUM(pg_total_relation_size(c.oid)), 0),
  'largest_relation_bytes', COALESCE(MAX(pg_total_relation_size(c.oid)), 0),
  'capture_utc_timestamp', (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::text || '+00:00',
  'capture_utc_date', (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date::text,
  'latest_completed_benchmark_date', NULL,
  'current_day_rebenchmark_run_count', 0,
  'current_day_benchmark_bundle_count', 0,
  'weight_history_scope', (
    SELECT json_build_object(
      'netuid', netuid,
      'start_epoch', MIN(epoch_id),
      'end_epoch', MAX(epoch_id),
      'expected_rows', COUNT(*)
    )
    FROM public.research_lab_finalized_allocation_epochs_v2
    GROUP BY netuid
    ORDER BY COUNT(*) DESC, netuid
    LIMIT 1
  ),
  'source_role', (
    SELECT json_build_object(
      'role_name', rolname,
      'transaction_read_only', current_setting('transaction_read_only') = 'on',
      'superuser', rolsuper,
      'bypass_rls', rolbypassrls,
      'replication', rolreplication,
      'table_write_capable', EXISTS (
        SELECT 1
        FROM pg_class AS writable_class
        JOIN pg_namespace AS writable_namespace
          ON writable_namespace.oid = writable_class.relnamespace
        WHERE writable_namespace.nspname = 'public'
          AND writable_class.relkind IN ('r', 'p')
          AND has_table_privilege(
            current_user,
            writable_class.oid,
            'INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER'
          )
      )
    )
    FROM pg_roles
    WHERE rolname = current_user
  )
)::text
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'm')
  AND n.nspname = 'public';
"""
    raw = _require_success(
        _run_postgres(
            ["psql", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-c", stats_sql],
            env=env,
            timeout=60,
            postgres_image=postgres_image,
        ),
        stage="production database shape read",
    ).decode("utf-8", "replace").strip()
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ProductionParityError(
            "production database shape response is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ProductionParityError(
            "production database shape response is not an object"
        )
    source_role = value.get("source_role")
    if isinstance(source_role, Mapping):
        role_name = str(source_role.pop("role_name", ""))
        source_role["role_hash"] = sha256_json({"role": role_name})
    return value


def _target_rebenchmark_date(stats: Mapping[str, Any]) -> date:
    try:
        captured = datetime.fromisoformat(
            str(stats.get("capture_utc_timestamp") or "")
        ).astimezone(timezone.utc)
    except ValueError as exc:
        raise ProductionParityError("production snapshot clock is invalid") from exc
    # The clone executes tomorrow's normal production workflow. Candidate
    # code creates that date's ICP set itself, so the test never deletes,
    # rewrites, or reuses a consumed production daily slot.
    return captured.date() + timedelta(days=1)


def _git(
    root: Path,
    *args: str,
    timeout: int = 60,
) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return _require_success(result, stage="production snapshot Git identity")


def _source_migrations(
    *, root: Path, source_sha: str, candidate_sha: str
) -> list[dict[str, Any]]:
    source = str(source_sha or "").strip().lower()
    if not _SHA_RE.fullmatch(source):
        raise ProductionParityError("snapshot producer runtime SHA is invalid")
    resolved = _git(root, "rev-parse", f"{source}^{{commit}}").decode().strip()
    if resolved != source:
        raise ProductionParityError("snapshot producer runtime SHA is unavailable")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source, candidate_sha],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if ancestor.returncode != 0:
        raise ProductionParityError(
            "snapshot producer runtime is not an ancestor of capture code"
        )
    tracked = _git(root, "ls-tree", "-r", "--name-only", source).decode(
        "utf-8", "strict"
    )
    migrations: list[dict[str, Any]] = []
    for path in sorted(line.strip() for line in tracked.splitlines() if line.strip()):
        if MIGRATION_RE.fullmatch(path) is None:
            continue
        sequence, _name = migration_sequence(path)
        payload = _git(root, "show", f"{source}:{path}")
        migrations.append(
            {
                "path": path,
                "sequence": sequence,
                "sha256": sha256_bytes(payload),
                "transaction_mode": (
                    "autocommit"
                    if path.endswith(".concurrent.sql")
                    else "candidate-file"
                ),
            }
        )
    if not migrations:
        raise ProductionParityError(
            "snapshot producer runtime migration inventory is empty"
        )
    return sorted(migrations, key=lambda item: (item["sequence"], item["path"]))


def capture_snapshot(
    *,
    contract_path: Path,
    archive_path: Path,
    manifest_path: Path,
    dsn: str,
    expected_production_host: str,
    ttl_hours: int,
    source_sha: str,
    capture_mode: str = "full",
    timeout_seconds: int = DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS,
    postgres_image: str | None = None,
) -> dict[str, Any]:
    timeout_seconds = _snapshot_io_timeout_seconds(timeout_seconds)
    contract = validate_contract(_load_json(contract_path, description="parity contract"))
    env, observed_host = _postgres_env(dsn, read_only=True)
    if observed_host != str(expected_production_host or "").strip().lower():
        raise ProductionParityError("snapshot source is not the expected production database")
    if ttl_hours < 1 or ttl_hours > 48:
        raise ProductionParityError("snapshot TTL must be between 1 and 48 hours")
    if capture_mode not in {"full", "schema-only"}:
        raise ProductionParityError("production snapshot capture mode is invalid")

    read_only = _require_success(
        _run_postgres(
            ["psql", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-c", "SHOW transaction_read_only"],
            env=env,
            timeout=(
                min(timeout_seconds, DEFAULT_PINNED_POSTGRES_STARTUP_TIMEOUT_SECONDS)
                if postgres_image is not None
                else 30
            ),
            postgres_image=postgres_image,
        ),
        stage="production read-only transaction check",
    ).decode("utf-8", "replace").strip()
    if read_only != "on":
        raise ProductionParityError("production snapshot session is not read-only")

    stats = _database_stats(env, postgres_image=postgres_image)
    source_role = stats.get("source_role")
    if (
        not isinstance(source_role, Mapping)
        or source_role.get("transaction_read_only") is not True
        or source_role.get("superuser") is not False
        or not isinstance(source_role.get("bypass_rls"), bool)
        or source_role.get("replication") is not False
        or source_role.get("table_write_capable") is not False
    ):
        raise ProductionParityError(
            "production snapshot credential is not a dedicated read-only role"
        )
    target_rebenchmark_date = _target_rebenchmark_date(stats)
    source_migrations = _source_migrations(
        root=ROOT,
        source_sha=source_sha,
        candidate_sha=contract["candidate_sha"],
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if capture_mode == "full":
        _require_full_snapshot_disk_headroom(
            archive_path.parent,
            total_relation_bytes=int(stats.get("total_relation_bytes") or 0),
            simultaneous_copies=2,
        )
    mounted_archive: Path | None = None
    dump_archive = str(archive_path)
    dump_mounts: tuple[_PostgresClientMount, ...] = ()
    if postgres_image is not None:
        mounted_archive = _create_pinned_archive(archive_path)
        dump_archive = _POSTGRES_ARCHIVE_TARGET
        dump_mounts = (
            _PostgresClientMount(
                source=mounted_archive,
                target=_POSTGRES_ARCHIVE_TARGET,
                read_only=False,
            ),
        )
    dump_command = [
            "pg_dump",
            "--format=custom",
            "--compress=6",
            "--schema=public",
            "--no-owner",
            "--no-acl",
            "--serializable-deferrable",
            "--file",
            dump_archive,
        ]
    if capture_mode == "schema-only":
        dump_command.insert(4, "--schema-only")
    dump_env = dict(env)
    dump_env["PGOPTIONS"] = (
        "-c default_transaction_read_only=on "
        f"-c statement_timeout={timeout_seconds * 1000} "
        "-c lock_timeout=5000"
    )
    result = _run_postgres(
        dump_command,
        env=dump_env,
        timeout=timeout_seconds,
        postgres_image=postgres_image,
        mounts=dump_mounts,
    )
    _require_success(result, stage="read-only production snapshot capture")
    post_stats = _database_stats(env, postgres_image=postgres_image)
    if (
        str(post_stats.get("capture_utc_date") or "")
        != str(stats.get("capture_utc_date") or "")
        or _target_rebenchmark_date(post_stats) != target_rebenchmark_date
    ):
        archive_path.unlink(missing_ok=True)
        raise ProductionParityError(
            "production snapshot crossed its target-day consistency boundary"
        )
    captured_at = datetime.fromisoformat(
        str(stats.get("capture_utc_timestamp") or "")
    ).astimezone(timezone.utc)
    body = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_environment": "production-read-only",
        "source_host_hash": production_database_host_hash(observed_host),
        "capture_sha": contract["candidate_sha"],
        "capture_contract_hash": contract["contract_hash"],
        "source_sha": str(source_sha).lower(),
        "captured_at": captured_at.isoformat(),
        "expires_at": (captured_at + timedelta(hours=ttl_hours)).isoformat(),
        "capture_transaction_read_only": True,
        "capture_mode": capture_mode,
        "archive": {
            "format": (
                "postgres-custom"
                if capture_mode == "full"
                else "postgres-schema-custom"
            ),
            "storage": "ephemeral-encrypted-volume",
            "persisted": False,
            "sha256": file_sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
        },
        "database": {
            "server_version_num": str(stats.get("server_version_num") or ""),
            "relation_count": int(stats.get("relation_count") or 0),
            "total_relation_bytes": int(stats.get("total_relation_bytes") or 0),
            "largest_relation_bytes": int(stats.get("largest_relation_bytes") or 0),
            "capture_utc_date": str(stats.get("capture_utc_date") or ""),
            "target_rebenchmark_date": target_rebenchmark_date.isoformat(),
            "latest_completed_benchmark_date": stats.get(
                "latest_completed_benchmark_date"
            ),
            "current_day_rebenchmark_run_count": int(
                stats.get("current_day_rebenchmark_run_count") or 0
            ),
            "current_day_benchmark_bundle_count": int(
                stats.get("current_day_benchmark_bundle_count") or 0
            ),
            "source_role": dict(source_role),
            "weight_history_scope": dict(stats.get("weight_history_scope") or {}),
        },
        "migrations": source_migrations,
        "data_classification": "production-confidential-ephemeral",
    }
    manifest = validate_snapshot_manifest(
        {**body, "manifest_hash": sha256_json(body)}, now=captured_at
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path.chmod(0o600)
    archive_path.chmod(0o600)
    return manifest


def verify_snapshot(
    *,
    contract_path: Path,
    manifest_path: Path,
    archive_path: Path,
    expected_production_host: str | None = None,
    postgres_image: str | None = None,
) -> dict[str, Any]:
    contract = validate_contract(_load_json(contract_path, description="parity contract"))
    manifest = validate_snapshot_manifest(
        _load_json(manifest_path, description="snapshot manifest")
    )
    if manifest["source_sha"] != contract["base_sha"]:
        raise ProductionParityError(
            "snapshot source commit differs from the parity contract"
        )
    if manifest["capture_sha"] != contract["candidate_sha"]:
        raise ProductionParityError(
            "snapshot capture commit differs from the parity contract"
        )
    if manifest["capture_contract_hash"] != contract["contract_hash"]:
        raise ProductionParityError(
            "snapshot capture contract differs from the parity contract"
        )
    if (
        expected_production_host is not None
        and manifest["source_host_hash"]
        != production_database_host_hash(expected_production_host)
    ):
        raise ProductionParityError(
            "snapshot source host differs from the configured production database"
        )
    validate_archive(archive_path, manifest)
    delta = migration_delta(
        snapshot_migrations=manifest["migrations"],
        candidate_migrations=contract["migrations"],
    )
    if postgres_image is None:
        listing = subprocess.run(
            ["pg_restore", "--list", str(archive_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        listing_stdout = listing.stdout
    else:
        listing = _run_postgres(
            ["pg_restore", "--list", _POSTGRES_ARCHIVE_TARGET],
            env={},
            timeout=120,
            postgres_image=postgres_image,
            mounts=(
                _PostgresClientMount(
                    source=archive_path,
                    target=_POSTGRES_ARCHIVE_TARGET,
                    read_only=True,
                ),
            ),
        )
        listing_stdout = listing.stdout.decode("utf-8", "replace")
    if listing.returncode != 0 or not listing_stdout.strip():
        raise ProductionParityError("snapshot archive is not a readable PostgreSQL custom dump")
    return {
        "manifest_hash": manifest["manifest_hash"],
        "archive_hash": manifest["archive"]["sha256"],
        "source_sha": manifest["source_sha"],
        "source_host_hash": manifest["source_host_hash"],
        "candidate_sha": contract["candidate_sha"],
        "migration_delta": delta,
        "archive_entries": len(listing_stdout.splitlines()),
    }


def restore_snapshot(
    *,
    root: Path,
    contract_path: Path,
    manifest_path: Path,
    archive_path: Path,
    target_dsn: str,
    production_host: str,
    timeout_seconds: int = DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS,
    postgres_image: str | None = None,
) -> dict[str, Any]:
    timeout_seconds = _snapshot_io_timeout_seconds(timeout_seconds)
    evidence = verify_snapshot(
        contract_path=contract_path,
        manifest_path=manifest_path,
        archive_path=archive_path,
        expected_production_host=production_host,
        postgres_image=postgres_image,
    )
    safe_database_target(target_dsn, production_host=production_host)
    env, _ = _postgres_env(target_dsn, read_only=False)
    env["PGSSLMODE"] = "disable"
    restore_env = dict(env)
    restore_env["PGOPTIONS"] = "-c check_function_bodies=off"
    manifest = validate_snapshot_manifest(
        _load_json(manifest_path, description="snapshot manifest")
    )
    if manifest["capture_mode"] == "full":
        _require_full_snapshot_disk_headroom(
            archive_path.parent,
            total_relation_bytes=manifest["database"]["total_relation_bytes"],
            simultaneous_copies=1,
        )
    _require_success(
        _run_postgres(
            [
                "pg_restore",
                *(
                    ["--dbname", env["PGDATABASE"]]
                    if postgres_image is None
                    else ["--dbname="]
                ),
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-acl",
                "--exit-on-error",
                "--jobs=4",
                (
                    _POSTGRES_ARCHIVE_TARGET
                    if postgres_image is not None
                    else str(archive_path)
                ),
            ],
            env=restore_env,
            timeout=timeout_seconds,
            postgres_image=postgres_image,
            mounts=(
                (
                    _PostgresClientMount(
                        source=archive_path,
                        target=_POSTGRES_ARCHIVE_TARGET,
                        read_only=True,
                    ),
                )
                if postgres_image is not None
                else ()
            ),
        ),
        stage="isolated production snapshot restore",
    )
    clone_migration_preconditions: list[dict[str, Any]] = []
    for migration in evidence["migration_delta"]:
        path = root / str(migration["path"])
        if not path.is_file() or file_sha256(path) != migration["sha256"]:
            raise ProductionParityError(
                f"candidate migration bytes differ: {migration['path']}"
            )
        if (
            manifest["capture_mode"] == "schema-only"
            and not clone_migration_preconditions
            and migration["path"]
            in {
                candidate["path"]
                for candidate in _SCHEMA_ONLY_SOURCE_ADD_CUTOVER_MIGRATIONS
            }
        ):
            clone_migration_preconditions.append(
                _stage_schema_only_source_add_maintenance(
                    env=env,
                    migration=migration,
                    postgres_image=postgres_image,
                )
            )
        _require_success(
            _run_postgres(
                [
                    "psql",
                    "-X",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-f",
                    (
                        _POSTGRES_MIGRATION_TARGET
                        if postgres_image is not None
                        else str(path)
                    ),
                ],
                env=env,
                timeout=DEFAULT_CANDIDATE_MIGRATION_TIMEOUT_SECONDS,
                postgres_image=postgres_image,
                mounts=(
                    (
                        _PostgresClientMount(
                            source=path,
                            target=_POSTGRES_MIGRATION_TARGET,
                            read_only=True,
                        ),
                    )
                    if postgres_image is not None
                    else ()
                ),
            ),
            stage=f"candidate migration {migration['path']}",
        )
    if clone_migration_preconditions:
        return {
            **evidence,
            "clone_migration_preconditions": clone_migration_preconditions,
        }
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--contract", type=Path, required=True)
    capture.add_argument("--archive", type=Path, required=True)
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("--dsn-env", default="LEADPOET_PARITY_PRODUCTION_READONLY_DSN")
    capture.add_argument("--expected-host-env", default="LEADPOET_PARITY_PRODUCTION_DB_HOST")
    capture.add_argument("--ttl-hours", type=int, default=24)
    capture.add_argument("--source-sha", required=True)
    capture.add_argument(
        "--mode", choices=("full", "schema-only"), default="full"
    )
    capture.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS,
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--archive", type=Path, required=True)

    restore = subparsers.add_parser("restore")
    restore.add_argument("--root", type=Path, default=ROOT)
    restore.add_argument("--contract", type=Path, required=True)
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--target-dsn-env", default="LEADPOET_PARITY_TARGET_DSN")
    restore.add_argument("--production-host-env", default="LEADPOET_PARITY_PRODUCTION_DB_HOST")
    restore.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS,
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            dsn = os.environ.get(args.dsn_env, "")
            host = os.environ.get(args.expected_host_env, "")
            if not dsn or not host:
                raise ProductionParityError("production snapshot source environment is incomplete")
            result = capture_snapshot(
                contract_path=args.contract,
                archive_path=args.archive,
                manifest_path=args.manifest,
                dsn=dsn,
                expected_production_host=host,
                ttl_hours=args.ttl_hours,
                source_sha=args.source_sha,
                capture_mode=args.mode,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "verify":
            result = verify_snapshot(
                contract_path=args.contract,
                manifest_path=args.manifest,
                archive_path=args.archive,
            )
        else:
            target_dsn = os.environ.get(args.target_dsn_env, "")
            production_host = os.environ.get(args.production_host_env, "")
            if not target_dsn or not production_host:
                raise ProductionParityError("snapshot restore environment is incomplete")
            result = restore_snapshot(
                root=args.root,
                contract_path=args.contract,
                manifest_path=args.manifest,
                archive_path=args.archive,
                target_dsn=target_dsn,
                production_host=production_host,
                timeout_seconds=args.timeout_seconds,
            )
    except (OSError, ValueError, ProductionParityError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
