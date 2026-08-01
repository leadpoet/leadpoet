#!/usr/bin/env python3.11
"""Exercise settlement-critical candidate migrations in disposable PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from gateway.research_lab.champion_settlement_v2 import (
    CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
    ChampionSettlementV2Error,
    _preliminary_finalized_bundle_authority_v1,
    build_chain_realized_settlement_package_v1,
    validate_legacy_settlement_migrations_v2,
)
from gateway.research_lab.attested_v2_store import (
    boot_storage_row,
    receipt_storage_row,
    transport_storage_row,
)
from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
)
from gateway.tee.coordinator_chain_realized_settlement_v1 import (
    OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
)
from gateway.tee.coordinator_executor_v2 import CoordinatorExecutorV2
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    build_receipt_graph,
    build_transport_attempt,
    sha256_json,
)
from leadpoet_canonical.legacy_settlement_v2 import (
    LEGACY_SETTLEMENT_SCHEMA_VERSION,
    validate_legacy_settlement_document_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    validate_published_weight_bundle_v2,
)
from leadpoet_verifier.economics import allocate_research_lab_epoch
from tests.restart_rehearsal.fixture_contract import (
    load_rehearsal_current_settlement_epoch_id,
    load_rehearsal_metagraph_hotkeys,
)
from tests.restart_rehearsal.sanitized_weight_fixture import (
    NOW,
    SanitizedWeightFixture,
)


ALLOCATION_CANDIDATE_MIGRATION = (
    "33-research-lab-candidate-evaluation-queue.sql"
)
ALLOCATION_SCHEMA_MIGRATION = "35-research-lab-emission-allocator.sql"
ALLOCATION_CONTAINMENT_MIGRATION = (
    "87-research-lab-source-add-allocation-containment.sql"
)
MIGRATIONS_BEFORE_TRANSPORT_FIX = (
    "86-research-lab-attested-v2-authority.sql",
    "89-research-lab-provider-evidence-cache-v2.sql",
    "90-research-lab-provider-outcome-checkpoints-v2.sql",
    "99-research-lab-v2-champion-settlement.sql",
    "104-research-lab-attested-result-replay-v2.sql",
    "125-research-lab-artifact-key-lineage.sql",
    "126-research-lab-chain-realized-settlement.sql",
    "127-research-lab-chain-unattributed-settlement.sql",
)
TRANSPORT_FIX_MIGRATION = "128-research-lab-chain-settlement-transport-purposes.sql"
TRANSPORT_TERMINAL_MIGRATION = (
    "129-research-lab-attested-local-transport.sql"
)
PROVIDER_OUTCOME_APPEND_MIGRATION = (
    "130-research-lab-provider-outcome-append.sql"
)
PROVIDER_OUTCOME_BACKPRESSURE_MIGRATION = (
    "131-research-lab-provider-outcome-backpressure.sql"
)
PROVIDER_OUTCOME_CONTENTION_STATUS_MIGRATION = (
    "133-research-lab-provider-outcome-contention-status.sql"
)
PROVIDER_OUTCOME_HEAD_CONTENTION_MIGRATION = (
    "134-research-lab-provider-outcome-head-contention.sql"
)
ACTIVE_MODEL_RESULT_REPLAY_MIGRATION = (
    "135-research-lab-active-model-result-replay.sql"
)
CHAMPION_LIFETIME_CREDIT_MIGRATION = (
    "132-research-lab-champion-lifetime-credit.sql"
)
EXPECTED_FINALIZED_VIEW_COLUMNS = (
    "bundle_hash",
    "schema_version",
    "netuid",
    "epoch_id",
    "block",
    "validator_hotkey",
    "root_receipt_hash",
    "weights_hash",
    "snapshot_hash",
    "bundle_doc",
    "weight_submission_event_hash",
    "publication_receipt_hash",
    "transparency_event_hash",
    "durable_readback_hash",
    "publication_doc",
    "weight_finalization_event_hash",
    "finalization_receipt_hash",
    "extrinsic_authorization_hash",
    "extrinsic_hash",
    "finalized_block",
    "finalized_block_hash",
    "state_transition_hash",
    "finalization_doc",
)
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
SYSTEM_BINARY_DIRS = tuple(
    Path(value)
    for value in ("/usr/local/sbin", "/usr/sbin", "/sbin", "/usr/bin", "/bin")
)
ALLOCATION_MIGRATION_PREREQUISITES_SQL = """
CREATE SCHEMA auth;
CREATE FUNCTION auth.role()
RETURNS TEXT
LANGUAGE SQL
STABLE
AS $$ SELECT current_user::TEXT $$;
CREATE TABLE public.research_evaluation_score_bundles (
    score_bundle_id TEXT PRIMARY KEY,
    score_bundle_doc JSONB NOT NULL DEFAULT '{}'::JSONB
);
CREATE TABLE public.research_loop_tickets (
    ticket_id UUID PRIMARY KEY
);
CREATE TABLE public.research_loop_receipts (
    receipt_id UUID PRIMARY KEY
);
"""
GIT_TREE_CANDIDATE_PREREQUISITES_SQL = """
ALTER TABLE public.research_lab_candidate_artifacts
    ADD COLUMN git_tree_id TEXT NULL,
    ADD COLUMN git_tree_node_id TEXT NULL,
    ADD COLUMN git_tree_root_commit TEXT NULL,
    ADD COLUMN git_tree_node_commit TEXT NULL,
    ADD COLUMN git_tree_lineage_hash TEXT NULL;
"""


class PostgresContractProbeError(RuntimeError):
    """The candidate migration-backed V2 contract is not production-ready."""


def _sql_without_comments(value: str) -> str:
    value = re.sub(r"/\*.*?\*/", " ", value, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", value)


def _validate_required_migration_declarations(
    source_root: Path,
) -> dict[str, int]:
    documents: dict[str, str] = {}

    def migration_sql(name: str) -> str:
        if name not in documents:
            path = source_root / name
            if not path.is_file():
                raise PostgresContractProbeError(
                    "required migration is missing: %s" % name
                )
            documents[name] = _sql_without_comments(path.read_text(encoding="utf-8"))
        return documents[name]

    for migration, relation, columns in REQUIRED_SUPABASE_V2_SCHEMA:
        sql = migration_sql(migration)
        declaration = re.compile(
            r"\b(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)"
            r"|ALTER\s+TABLE)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
            r"(?:public\.)?%s\b" % re.escape(relation),
            flags=re.IGNORECASE,
        )
        declaration_match = declaration.search(sql)
        if declaration_match is None:
            raise PostgresContractProbeError(
                "required relation is not declared migration=%s relation=%s"
                % (migration, relation)
            )
        for column in columns:
            statement_end = sql.find(";", declaration_match.start())
            declaration_statement = sql[
                declaration_match.start() : None if statement_end < 0 else statement_end
            ]
            inherited_view_columns = (
                "VIEW" in declaration_statement.upper()
                and re.search(r"\b[a-z][a-z0-9_]*\s*\.\s*\*", declaration_statement)
                is not None
            )
            if (
                re.search(r"\b%s\b" % re.escape(column), sql) is None
                and not inherited_view_columns
            ):
                raise PostgresContractProbeError(
                    "required column is not declared migration=%s "
                    "relation=%s column=%s" % (migration, relation, column)
                )
    for migration, function_name in REQUIRED_SUPABASE_V2_RPCS:
        sql = migration_sql(migration)
        declaration = re.compile(
            r"\bCREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+"
            r"(?:public\.)?%s\s*\(" % re.escape(function_name),
            flags=re.IGNORECASE,
        )
        if declaration.search(sql) is None:
            raise PostgresContractProbeError(
                "required RPC is not declared migration=%s rpc=%s"
                % (migration, function_name)
            )
    return {
        "migration_count": len(documents),
        "relation_probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA),
        "rpc_probe_count": len(REQUIRED_SUPABASE_V2_RPCS),
    }


class DisposablePostgres:
    def __init__(self, *, state_root: Path):
        self.state_root = state_root
        self.root = Path(tempfile.mkdtemp(prefix="leadpoet-postgres-v2-", dir="/tmp"))
        self.data = self.root / "data"
        self.socket = self.root / "socket"
        self.port = 55432
        self.database = "leadpoet_rehearsal"
        self.started = False
        account = pwd.getpwnam("postgres")
        os.chown(self.root, account.pw_uid, account.pw_gid)
        self.socket.mkdir()
        os.chown(self.socket, account.pw_uid, account.pw_gid)

    @staticmethod
    def _binary(name: str) -> str:
        if not IDENTIFIER_RE.fullmatch(name):
            raise PostgresContractProbeError(
                "postgres binary name is invalid: %s" % name
            )
        candidates = []
        resolved = shutil.which(name)
        if resolved is not None:
            candidates.append(Path(resolved))
        candidates.extend(directory / name for directory in SYSTEM_BINARY_DIRS)
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        raise PostgresContractProbeError("postgres binary is unavailable: %s" % name)

    def _as_postgres(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            self._binary("runuser"),
            "-u",
            "postgres",
            "--",
            *argv,
        ]
        return subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=check,
        )

    def start(self) -> None:
        self._as_postgres(
            [
                self._binary("initdb"),
                "--pgdata",
                str(self.data),
                "--auth=trust",
                "--no-locale",
                "--encoding=UTF8",
            ]
        )
        self._as_postgres(
            [
                self._binary("pg_ctl"),
                "--pgdata",
                str(self.data),
                "--log",
                str(self.root / "postgres.log"),
                "--options",
                "-k %s -p %d -c listen_addresses=''" % (self.socket, self.port),
                "--wait",
                "start",
            ]
        )
        self.started = True
        self.psql(
            """
            CREATE ROLE anon NOLOGIN;
            CREATE ROLE authenticated NOLOGIN;
            CREATE ROLE service_role NOLOGIN;
            CREATE DATABASE leadpoet_rehearsal;
            """,
            database="postgres",
        )

    def stop(self) -> None:
        if self.started:
            self._as_postgres(
                [
                    self._binary("pg_ctl"),
                    "--pgdata",
                    str(self.data),
                    "--wait",
                    "--mode",
                    "fast",
                    "stop",
                ],
                check=False,
            )
            self.started = False
        shutil.rmtree(self.root, ignore_errors=True)

    def psql(
        self,
        sql: str,
        *,
        database: str | None = None,
        check: bool = True,
        tuples_only: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        argv = [
            self._binary("psql"),
            "--no-psqlrc",
            "--host",
            str(self.socket),
            "--port",
            str(self.port),
            "--username",
            "postgres",
            "--dbname",
            database or self.database,
            "--set",
            "ON_ERROR_STOP=1",
        ]
        if tuples_only:
            argv.extend(["--tuples-only", "--no-align"])
        return self._as_postgres(argv, input_text=sql, check=check)

    def apply_migration(self, path: Path) -> None:
        result = self.psql(path.read_text(encoding="utf-8"), check=False)
        if result.returncode != 0:
            raise PostgresContractProbeError(
                "migration failed path=%s stderr=%s"
                % (path.name, result.stderr.strip())
            )


def _json_insert_sql(table: str, row: Mapping[str, Any]) -> str:
    if not IDENTIFIER_RE.fullmatch(table):
        raise PostgresContractProbeError(
            "fixture table identifier is invalid: %s" % table
        )
    columns = tuple(row)
    if not columns or any(not IDENTIFIER_RE.fullmatch(name) for name in columns):
        raise PostgresContractProbeError(
            "fixture row columns are invalid for %s" % table
        )
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    if "$leadpoet$" in payload:
        raise PostgresContractProbeError("fixture JSON delimiter collision")
    selected = ",".join(columns)
    return (
        "INSERT INTO public.%s (%s) "
        "SELECT %s FROM pg_catalog.json_populate_record("
        "NULL::public.%s, $leadpoet$%s$leadpoet$::json);\n"
        % (table, selected, selected, table, payload)
    )


def _deterministic_seed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(row), "created_at": NOW}


def _provider_outcome_append_sql(row: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    if "$leadpoet$" in payload:
        raise PostgresContractProbeError(
            "provider outcome checkpoint JSON delimiter collision"
        )
    return (
        "SELECT public.append_research_lab_provider_outcome_checkpoint_v2("
        "$leadpoet$%s$leadpoet$::jsonb)::text;\n" % payload
    )


def _provider_outcome_append_contract(
    database: DisposablePostgres,
) -> dict[str, Any]:
    key_hash = "sha256:" + "a" * 64

    def rollback_count() -> int:
        database.psql("SELECT pg_catalog.pg_stat_clear_snapshot();")
        return int(
            database.psql(
                """
                SELECT xact_rollback
                FROM pg_catalog.pg_stat_database
                WHERE datname = pg_catalog.current_database();
                """,
                tuples_only=True,
            ).stdout.strip()
        )

    def row(
        *,
        sequence: int,
        checkpoint_hash: str,
        previous_checkpoint_hash: str,
        suffix: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "leadpoet.provider_outcome_checkpoint_row.v2",
            "artifact_master_key_ref_hash": key_hash,
            "utc_day": "2026-07-10",
            "sequence": sequence,
            "checkpoint_hash": checkpoint_hash,
            "previous_checkpoint_hash": previous_checkpoint_hash,
            "state_document_hash": "sha256:" + suffix * 64,
            "checkpoint_artifact_id": "sha256:" + suffix.upper().lower() * 64,
            "encrypted_checkpoint_doc": {
                "schema_version": "leadpoet.encrypted_artifact.v2",
                "fixture": suffix,
            },
        }

    first_hash = "sha256:" + "b" * 64
    first = row(
        sequence=1,
        checkpoint_hash=first_hash,
        previous_checkpoint_hash="",
        suffix="c",
    )
    inserted = json.loads(
        database.psql(
            _provider_outcome_append_sql(first),
            tuples_only=True,
        ).stdout.strip()
    )
    if inserted != {"status": "inserted", "checkpoint_hash": first_hash}:
        raise PostgresContractProbeError(
            "provider outcome first append result differs"
        )
    existing = json.loads(
        database.psql(
            _provider_outcome_append_sql(first),
            tuples_only=True,
        ).stdout.strip()
    )
    if existing != {"status": "existing", "checkpoint_hash": first_hash}:
        raise PostgresContractProbeError(
            "provider outcome idempotent append result differs"
        )
    rollback_count_before_contention = rollback_count()

    siblings = (
        row(
            sequence=2,
            checkpoint_hash="sha256:" + "d" * 64,
            previous_checkpoint_hash=first_hash,
            suffix="e",
        ),
        row(
            sequence=2,
            checkpoint_hash="sha256:" + "f" * 64,
            previous_checkpoint_hash=first_hash,
            suffix="1",
        ),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda value: database.psql(
                    _provider_outcome_append_sql(value),
                    check=False,
                    tuples_only=True,
                ),
                siblings,
            )
        )
    if any(result.returncode != 0 for result in results):
        raise PostgresContractProbeError(
            "provider outcome expected contention surfaced as a SQL error"
        )
    outcomes = [json.loads(result.stdout.strip()) for result in results]
    accepted = [
        outcome for outcome in outcomes if outcome.get("status") == "inserted"
    ]
    rejected = [
        outcome
        for outcome in outcomes
        if outcome.get("status") in {"busy", "conflict"}
    ]
    if len(accepted) != 1 or len(rejected) != 1:
        raise PostgresContractProbeError(
            "provider outcome concurrent append did not select one head"
        )
    accepted_hash = accepted[0].get("checkpoint_hash")
    accepted_row = next(
        (
            dict(candidate)
            for candidate in siblings
            if candidate["checkpoint_hash"] == accepted_hash
        ),
        None,
    )
    if accepted_row is None:
        raise PostgresContractProbeError(
            "provider outcome append accepted an unknown candidate"
        )
    rejected_outcome = rejected[0]
    if set(rejected_outcome) not in (
        {"status", "checkpoint_hash"},
        {"status", "checkpoint_hash", "head_checkpoint_row"},
    ):
        raise PostgresContractProbeError(
            "provider outcome contention response fields differ"
        )
    rejected_hash = rejected_outcome.get("checkpoint_hash")
    if (
        rejected_hash not in {candidate["checkpoint_hash"] for candidate in siblings}
        or rejected_hash == accepted_hash
    ):
        raise PostgresContractProbeError(
            "provider outcome contention response lost candidate identity"
        )
    if rejected_outcome["status"] == "busy":
        if set(rejected_outcome) != {"status", "checkpoint_hash"}:
            raise PostgresContractProbeError(
                "provider outcome busy response fields differ"
            )
    elif (
        set(rejected_outcome)
        != {"status", "checkpoint_hash", "head_checkpoint_row"}
        or rejected_outcome.get("head_checkpoint_row") != accepted_row
    ):
        raise PostgresContractProbeError(
            "provider outcome concurrent conflict omitted its durable head"
        )
    row_count = int(
        database.psql(
            """
            SELECT pg_catalog.count(*)
            FROM public.research_lab_provider_outcome_checkpoints_v2
            WHERE artifact_master_key_ref_hash =
                  'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
              AND utc_day = DATE '2026-07-10';
            """,
            tuples_only=True,
        ).stdout.strip()
    )
    if row_count != 2:
        raise PostgresContractProbeError(
            "provider outcome lineage contains an unexpected row count"
        )

    stale_row = next(
        dict(candidate)
        for candidate in siblings
        if candidate["checkpoint_hash"] != accepted_hash
    )
    stale = json.loads(
        database.psql(
            _provider_outcome_append_sql(stale_row),
            tuples_only=True,
        ).stdout.strip()
    )
    if stale != {
        "status": "conflict",
        "checkpoint_hash": stale_row["checkpoint_hash"],
        "head_checkpoint_row": accepted_row,
    }:
        raise PostgresContractProbeError(
            "provider outcome stale append did not return the exact durable head"
        )

    empty_conflict_row = row(
        sequence=2,
        checkpoint_hash="sha256:" + "4" * 64,
        previous_checkpoint_hash="sha256:" + "5" * 64,
        suffix="6",
    )
    empty_conflict_row["artifact_master_key_ref_hash"] = "sha256:" + "9" * 64
    empty_conflict = json.loads(
        database.psql(
            _provider_outcome_append_sql(empty_conflict_row),
            tuples_only=True,
        ).stdout.strip()
    )
    if empty_conflict != {
        "status": "conflict",
        "checkpoint_hash": empty_conflict_row["checkpoint_hash"],
        "head_checkpoint_row": None,
    }:
        raise PostgresContractProbeError(
            "provider outcome empty-lineage conflict response differs"
        )

    third_hash = "sha256:" + "2" * 64
    third = row(
        sequence=3,
        checkpoint_hash=third_hash,
        previous_checkpoint_hash=accepted_hash,
        suffix="3",
    )
    lock_sql = """
        BEGIN;
        SELECT pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtext('research_lab_provider_outcome_checkpoint_v2'),
            pg_catalog.hashtext(
                'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                || ':2026-07-10'
            )
        );
        SELECT pg_catalog.pg_sleep(2);
        COMMIT;
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        holder = executor.submit(database.psql, lock_sql)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            granted = int(
                database.psql(
                    """
                    SELECT pg_catalog.count(*)
                    FROM pg_catalog.pg_locks
                    WHERE locktype = 'advisory' AND granted;
                    """,
                    tuples_only=True,
                ).stdout.strip()
            )
            if granted:
                break
            time.sleep(0.02)
        else:
            raise PostgresContractProbeError(
                "provider outcome contention fixture did not acquire its lock"
            )
        started = time.monotonic()
        busy = database.psql(
            _provider_outcome_append_sql(third),
            check=False,
            tuples_only=True,
        )
        busy_elapsed = time.monotonic() - started
        busy_result = (
            json.loads(busy.stdout.strip())
            if busy.returncode == 0 and busy.stdout.strip()
            else {}
        )
        if busy.returncode != 0 or busy_result != {
            "status": "busy",
            "checkpoint_hash": third_hash,
        }:
            raise PostgresContractProbeError(
                "provider outcome contention did not return the busy contract"
            )
        if busy_elapsed >= 1.0:
            raise PostgresContractProbeError(
                "provider outcome contention occupied a database session"
            )
        holder.result(timeout=3.0)

    third_inserted = json.loads(
        database.psql(
            _provider_outcome_append_sql(third),
            tuples_only=True,
        ).stdout.strip()
    )
    if third_inserted != {
        "status": "inserted",
        "checkpoint_hash": third_hash,
    }:
        raise PostgresContractProbeError(
            "provider outcome append did not recover after contention"
        )
    rollback_count_after_contention = rollback_count()
    if rollback_count_after_contention != rollback_count_before_contention:
        raise PostgresContractProbeError(
            "provider outcome expected contention rolled back a transaction"
        )
    return {
        "first_checkpoint_hash": first_hash,
        "candidate_sibling_hashes": sorted(
            item["checkpoint_hash"] for item in siblings
        ),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "row_count": row_count + 1,
        "contention_rollback_delta": (
            rollback_count_after_contention
            - rollback_count_before_contention
        ),
        "durable_head_conflict_verified": True,
        "empty_head_conflict_verified": True,
    }


def _settlement_fixture(
    *,
    candidate_sha: str,
    epoch_id: int,
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    dict[str, Any],
    SanitizedWeightFixture,
]:
    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
    )
    bundle = fixture.bundle()
    verified = validate_published_weight_bundle_v2(bundle)
    coordinator_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "gateway_coordinator"
    )
    weight_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "validator_weights"
    )
    bundle_row = {
        "bundle_hash": verified["bundle_hash"],
        "schema_version": bundle["schema_version"],
        "netuid": verified["netuid"],
        "epoch_id": verified["epoch_id"],
        "block": verified["block"],
        "validator_hotkey": verified["validator_hotkey"],
        "root_receipt_hash": verified["root_receipt_hash"],
        "weights_hash": verified["weights_hash"],
        "snapshot_hash": verified["snapshot_hash"],
        "bundle_doc": bundle,
    }
    durable_readback_hash = sha256_json(bundle_row)
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": verified["bundle_hash"],
        "root_receipt_hash": verified["root_receipt_hash"],
        "durable_readback_hash": durable_readback_hash,
        "transparency_event_hash": "sha256:" + "d" * 64,
    }
    publication_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="gateway.weights.publication.v2",
        job_id="postgres-contract-publication",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json({"kind": "publication", "epoch_id": epoch_id}),
        output_root=sha256_json(publication_doc),
        parents=[verified["root_receipt_hash"]],
        sequence=800,
    )
    submission_event_hash = sha256_json(
        {
            "bundle_hash": verified["bundle_hash"],
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "transparency_event_hash": publication_doc["transparency_event_hash"],
            "durable_readback_hash": durable_readback_hash,
        }
    )
    publication_row = {
        "weight_submission_event_hash": submission_event_hash,
        "bundle_hash": verified["bundle_hash"],
        "publication_receipt_hash": publication_receipt["receipt_hash"],
        "transparency_event_hash": publication_doc["transparency_event_hash"],
        "durable_readback_hash": durable_readback_hash,
        "publication_doc": publication_doc,
    }
    finalization_doc = {
        "schema_version": "leadpoet.weight_finalization.v2",
        "validator_hotkey": verified["validator_hotkey"],
        "netuid": verified["netuid"],
        "epoch_id": verified["epoch_id"],
        "weights_hash": verified["weights_hash"],
        "weight_receipt_hash": verified["weight_receipt_hash"],
        "weight_submission_event_hash": submission_event_hash,
        "extrinsic_authorization_hash": "sha256:" + "e" * 64,
        "extrinsic_hash": "0x" + "f" * 64,
        "finalized_block": int(verified["block"]) + 1,
        "finalized_block_hash": "1" * 64,
        "state_transition_hash": "sha256:" + "2" * 64,
    }
    finalization_receipt = fixture.receipt(
        role="validator_weights",
        purpose="validator.weights.finalized.v2",
        job_id="postgres-contract-finalization",
        key=fixture.weight_key,
        boot=weight_boot,
        config_hash=str(weight_boot["config_hash"]),
        input_root=sha256_json({"kind": "finalization", "epoch_id": epoch_id}),
        output_root=sha256_json(finalization_doc),
        parents=[verified["weight_receipt_hash"]],
        sequence=801,
    )
    finalization_event_hash = sha256_json(
        {
            "weight_submission_event_hash": submission_event_hash,
            "bundle_hash": verified["bundle_hash"],
            "finalization_receipt_hash": finalization_receipt["receipt_hash"],
            "extrinsic_authorization_hash": finalization_doc[
                "extrinsic_authorization_hash"
            ],
            "extrinsic_hash": finalization_doc["extrinsic_hash"],
            "finalized_block": finalization_doc["finalized_block"],
            "finalized_block_hash": finalization_doc["finalized_block_hash"],
            "state_transition_hash": finalization_doc["state_transition_hash"],
        }
    )
    finalization_row = {
        "weight_finalization_event_hash": finalization_event_hash,
        "weight_submission_event_hash": submission_event_hash,
        "bundle_hash": verified["bundle_hash"],
        "finalization_receipt_hash": finalization_receipt["receipt_hash"],
        "extrinsic_authorization_hash": finalization_doc[
            "extrinsic_authorization_hash"
        ],
        "extrinsic_hash": finalization_doc["extrinsic_hash"],
        "finalized_block": finalization_doc["finalized_block"],
        "finalized_block_hash": finalization_doc["finalized_block_hash"],
        "state_transition_hash": finalization_doc["state_transition_hash"],
        "finalization_doc": finalization_doc,
    }
    root_receipt = next(
        receipt
        for receipt in bundle["receipt_graph"]["receipts"]
        if receipt["receipt_hash"] == verified["root_receipt_hash"]
    )
    rows = [
        *[
            (
                "research_lab_attested_boot_identities_v2",
                boot_storage_row(identity),
            )
            for identity in bundle["receipt_graph"]["boot_identities"]
        ],
        (
            "research_lab_attested_execution_receipts_v2",
            receipt_storage_row(root_receipt),
        ),
        (
            "research_lab_attested_execution_receipts_v2",
            receipt_storage_row(publication_receipt),
        ),
        (
            "research_lab_attested_execution_receipts_v2",
            receipt_storage_row(finalization_receipt),
        ),
        ("research_lab_attested_weight_bundles_v2", bundle_row),
        ("research_lab_attested_publication_events_v2", publication_row),
        ("research_lab_attested_weight_finalizations_v2", finalization_row),
    ]
    return rows, verified, fixture


def _relation_contract(database: DisposablePostgres) -> dict[str, Any]:
    result = database.psql(
        """
        SELECT pg_catalog.json_build_object(
            'relations',
            COALESCE(
                (
                    SELECT pg_catalog.json_object_agg(name, relation)
                    FROM (
                        SELECT
                            class.relname AS name,
                            pg_catalog.json_build_object(
                                'kind', class.relkind,
                                'columns', pg_catalog.json_agg(
                                    attribute.attname
                                    ORDER BY attribute.attnum
                                )
                            ) AS relation
                        FROM pg_catalog.pg_class class
                        JOIN pg_catalog.pg_namespace namespace
                          ON namespace.oid = class.relnamespace
                        JOIN pg_catalog.pg_attribute attribute
                          ON attribute.attrelid = class.oid
                         AND attribute.attnum > 0
                         AND NOT attribute.attisdropped
                        WHERE namespace.nspname = 'public'
                          AND class.relkind IN ('r', 'p', 'v', 'm')
                        GROUP BY class.relname, class.relkind
                        ORDER BY class.relname
                    ) relations
                ),
                '{}'::json
            ),
            'rpcs',
            COALESCE(
                (
                    SELECT pg_catalog.json_agg(name ORDER BY name)
                    FROM (
                        SELECT DISTINCT procedure.proname AS name
                        FROM pg_catalog.pg_proc procedure
                        JOIN pg_catalog.pg_namespace namespace
                          ON namespace.oid = procedure.pronamespace
                        WHERE namespace.nspname = 'public'
                    ) procedures
                ),
                '[]'::json
            )
        )::text;
        """,
        tuples_only=True,
    )
    return json.loads(result.stdout.strip())


def _measured_settlement_receipt_contract(
    *,
    authority: Mapping[str, Any],
    verified_bundle: Mapping[str, Any],
    fixture: SanitizedWeightFixture,
) -> dict[str, Any]:
    calculation = authority["bundle_doc"]["weight_snapshot"][
        "calculation_snapshot"
    ]
    hotkeys = list(calculation["metagraph_hotkeys"])
    hotkeys[0] = str(verified_bundle["validator_hotkey"])
    finalized_block = int(authority["finalized_block"])
    weights = [
        [int(uid), int(weight)]
        for uid, weight in zip(
            verified_bundle["uids"],
            verified_bundle["weights_u16"],
        )
    ]
    observation = {
        "schema_version": CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
        "netuid": int(verified_bundle["netuid"]),
        "epoch_id": int(verified_bundle["epoch_id"]),
        "official_subnet_epoch_id": int(verified_bundle["epoch_id"]),
        "cutover_mapping_hash": sha256_json({"cutover": "rehearsal"}),
        "close_block": finalized_block + 50,
        "close_block_hash": "3" * 64,
        "close_state_root": "4" * 64,
        "next_epoch_block": finalized_block + 51,
        "next_epoch_block_hash": "5" * 64,
        "validator_hotkey": str(verified_bundle["validator_hotkey"]),
        "validator_uid": 0,
        "metagraph_hotkeys": hotkeys,
        "weights": weights,
        "weights_storage_key": "0x01",
        "last_update_storage_key": "0x02",
        "last_update_block": finalized_block,
        "last_update_block_hash": str(authority["finalized_block_hash"]),
        "last_update_official_subnet_epoch_id": int(
            verified_bundle["epoch_id"]
        ),
        "active_source_epoch_id": int(verified_bundle["epoch_id"]),
        "weights_vector_hash": sha256_json(
            {
                "uids": [item[0] for item in weights],
                "weights_u16": [item[1] for item in weights],
            }
        ),
    }
    package = build_chain_realized_settlement_package_v1(
        observation=observation,
        authority=authority,
    )
    executor = CoordinatorExecutorV2(
        chain_realized_settlement_resolver=lambda _payload, _context: package
    )

    async def execute() -> Any:
        return await executor(
            OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
            {
                "schema_version": (
                    "leadpoet.chain_realized_settlement_request.v1"
                ),
                "netuid": int(verified_bundle["netuid"]),
                "epoch_id": int(verified_bundle["epoch_id"]),
            },
            ExecutionContextV2(
                job_id="restart-rehearsal-chain-settlement",
                purpose="research_lab.chain_realized_epoch_settlement.v1",
                epoch_id=int(verified_bundle["epoch_id"]),
            ),
        )

    measured = asyncio.run(execute())
    if measured.output != package:
        raise PostgresContractProbeError(
            "measured chain settlement output differs"
        )
    if measured.receipt_output != package["settlement_doc"]:
        raise PostgresContractProbeError(
            "measured chain settlement receipt projection differs"
        )
    if sha256_json(measured.receipt_output) != package["settlement_hash"]:
        raise PostgresContractProbeError(
            "measured chain settlement receipt root differs"
        )
    coordinator_boot = next(
        identity
        for identity in authority["bundle_doc"]["receipt_graph"][
            "boot_identities"
        ]
        if identity["physical_role"] == "gateway_coordinator"
    )
    receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.chain_realized_epoch_settlement.v1",
        job_id="restart-rehearsal-chain-settlement",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json(
            {
                "schema_version": (
                    "leadpoet.chain_realized_settlement_request.v1"
                ),
                "netuid": int(verified_bundle["netuid"]),
                "epoch_id": int(verified_bundle["epoch_id"]),
            }
        ),
        output_root=package["settlement_hash"],
        parents=[verified_bundle["root_receipt_hash"]],
        sequence=802,
    )
    return {
        "settlement_hash": package["settlement_hash"],
        "credit_count": len(package["credits"]),
        "package": package,
        "receipt": receipt,
    }


def _settlement_persistence_rows(
    *,
    package: Mapping[str, Any],
    receipt_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settlement_doc = dict(package["settlement_doc"])
    settlement_hash = str(package["settlement_hash"])
    settlement_row = {
        "netuid": int(settlement_doc["netuid"]),
        "epoch_id": int(settlement_doc["epoch_id"]),
        "schema_version": str(settlement_doc["schema_version"]),
        "settlement_hash": settlement_hash,
        "settlement_receipt_hash": receipt_hash,
        "settlement_doc": settlement_doc,
    }
    credit_rows = []
    for item in package["credits"]:
        document = dict(item["credit_doc"])
        credit_rows.append(
            {
                "netuid": int(document["netuid"]),
                "epoch_id": int(document["epoch_id"]),
                "settlement_hash": settlement_hash,
                "schema_version": str(document["schema_version"]),
                "obligation_kind": str(document["obligation_kind"]),
                "obligation_source_id": str(
                    document["obligation_source_id"]
                ),
                "miner_hotkey": str(document["miner_hotkey"]),
                "miner_uid": int(document["miner_uid"]),
                "observed_chain_alpha_percent": str(
                    document["observed_chain_alpha_percent"]
                ),
                "lab_attributed_alpha_percent": str(
                    document["lab_attributed_alpha_percent"]
                ),
                "scheduled_alpha_percent": str(
                    document["scheduled_alpha_percent"]
                ),
                "credited_alpha_percent": str(
                    document["credited_alpha_percent"]
                ),
                "champion_credit_policy": str(
                    document["champion_credit_policy"]
                ),
                "credit_hash": str(item["credit_hash"]),
                "credit_receipt_hash": receipt_hash,
                "credit_doc": document,
            }
        )
    credit_rows.sort(key=lambda row: str(row["credit_hash"]))
    return settlement_row, credit_rows


def _json_rpc_sql(
    function_name: str,
    first: Mapping[str, Any],
    second: Sequence[Mapping[str, Any]],
) -> str:
    if not IDENTIFIER_RE.fullmatch(function_name):
        raise PostgresContractProbeError("fixture RPC identifier is invalid")
    first_json = json.dumps(dict(first), sort_keys=True, separators=(",", ":"))
    second_json = json.dumps(
        [dict(item) for item in second],
        sort_keys=True,
        separators=(",", ":"),
    )
    if "$leadpoet$" in first_json or "$leadpoet$" in second_json:
        raise PostgresContractProbeError("fixture RPC JSON delimiter collision")
    return (
        "SELECT public.%s("
        "$leadpoet$%s$leadpoet$::jsonb,"
        "$leadpoet$%s$leadpoet$::jsonb"
        ")::text;\n"
        % (function_name, first_json, second_json)
    )


def _historical_v1_settlement_rows(
    *,
    fixture: SanitizedWeightFixture,
    coordinator_boot: Mapping[str, Any],
    netuid: int,
    epoch_id: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    credit_doc = {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_obligation_credit.v1"
        ),
        "netuid": netuid,
        "epoch_id": epoch_id,
        "obligation_kind": "champion",
        "obligation_source_id": "grandfathered-champion",
        "miner_hotkey": "lab-hotkey",
        "miner_uid": 2,
        "observed_chain_alpha_percent": "5",
        "lab_attributed_alpha_percent": "5",
        "scheduled_alpha_percent": "5",
        "credited_alpha_percent": "5",
        "attribution_doc": {"fixture": "grandfathered"},
        "observation_doc": {"fixture": "grandfathered"},
    }
    credit_hash = sha256_json(credit_doc)
    settlement_doc = {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_epoch_settlement.v1"
        ),
        "netuid": netuid,
        "epoch_id": epoch_id,
        "credit_hashes": [credit_hash],
        "observation_summary": {"fixture": "grandfathered"},
    }
    settlement_hash = sha256_json(settlement_doc)
    receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.chain_realized_epoch_settlement.v1",
        job_id="restart-rehearsal-grandfathered-settlement",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json(
            {"kind": "grandfathered-settlement", "epoch_id": epoch_id}
        ),
        output_root=settlement_hash,
        parents=[],
        sequence=803,
    )
    settlement_row = {
        "netuid": netuid,
        "epoch_id": epoch_id,
        "schema_version": settlement_doc["schema_version"],
        "settlement_hash": settlement_hash,
        "settlement_receipt_hash": receipt["receipt_hash"],
        "settlement_doc": settlement_doc,
    }
    credit_row = {
        "netuid": netuid,
        "epoch_id": epoch_id,
        "settlement_hash": settlement_hash,
        "schema_version": credit_doc["schema_version"],
        "obligation_kind": credit_doc["obligation_kind"],
        "obligation_source_id": credit_doc["obligation_source_id"],
        "miner_hotkey": credit_doc["miner_hotkey"],
        "miner_uid": credit_doc["miner_uid"],
        "observed_chain_alpha_percent": (
            credit_doc["observed_chain_alpha_percent"]
        ),
        "lab_attributed_alpha_percent": (
            credit_doc["lab_attributed_alpha_percent"]
        ),
        "scheduled_alpha_percent": credit_doc["scheduled_alpha_percent"],
        "credited_alpha_percent": credit_doc["credited_alpha_percent"],
        "credit_hash": credit_hash,
        "credit_receipt_hash": receipt["receipt_hash"],
        "credit_doc": credit_doc,
    }
    return receipt_storage_row(receipt), settlement_row, credit_row


def _single_relation_row(
    database: DisposablePostgres,
    *,
    relation: str,
    where_sql: str,
) -> dict[str, Any]:
    if not IDENTIFIER_RE.fullmatch(relation):
        raise PostgresContractProbeError(
            "fixture relation identifier is invalid: %s" % relation
        )
    result = database.psql(
        """
        SELECT pg_catalog.row_to_json(row_value)::TEXT
        FROM public.%s row_value
        WHERE %s;
        """
        % (relation, where_sql),
        tuples_only=True,
    )
    rows = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise PostgresContractProbeError(
            "fixture relation returned %d rows: %s" % (len(rows), relation)
        )
    return rows[0]


def _load_coordinator_release_identity(
    path: Path,
    *,
    candidate_sha: str,
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        roles = document["gateway_roles"]
        identity = roles["gateway_coordinator"]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise PostgresContractProbeError(
            "candidate coordinator release identity is unavailable"
        ) from exc
    if (
        not isinstance(document, dict)
        or document.get("commit_sha") != candidate_sha
        or not isinstance(roles, dict)
        or not isinstance(identity, dict)
        or identity.get("commit_sha") != candidate_sha
    ):
        raise PostgresContractProbeError(
            "candidate coordinator release identity commit differs"
        )
    required = {
        "commit_sha",
        "dependency_lock_hash",
        "execution_manifest_hash",
        "pcr0",
    }
    if required - set(identity):
        raise PostgresContractProbeError(
            "candidate coordinator release identity fields are incomplete"
        )
    return dict(identity)


def _historical_compute_reimbursements(
    *,
    source_root: Path,
    source_epoch: int,
) -> list[dict[str, Any]]:
    metagraph_hotkeys = load_rehearsal_metagraph_hotkeys(source_root)
    return [
        {
            "uid": 2,
            "miner_hotkey": metagraph_hotkeys[2],
            "source_id": "reimbursement_schedule:restart-rehearsal-compute-2",
            "island": "generalist",
            "status": "active",
            "start_epoch": source_epoch,
            "epoch_count": 20,
            "target_reimbursement_microusd": 1_000_000,
            "eligible_compute_microusd": 1_000_000,
        },
        {
            "uid": 3,
            "miner_hotkey": metagraph_hotkeys[3],
            "source_id": "reimbursement_schedule:restart-rehearsal-compute-3",
            "island": "generalist",
            "status": "active",
            "start_epoch": source_epoch,
            "epoch_count": 20,
            "target_reimbursement_microusd": 3_000_000,
            "eligible_compute_microusd": 3_000_000,
        },
    ]


def _historical_compute_allocation_seed_rows(
    *,
    database: DisposablePostgres,
    source_root: Path,
    candidate_sha: str,
    current_epoch: int,
    netuid: int,
    coordinator_release_identity: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Persist and read back one finalized prior compute allocation."""

    source_epoch = int(current_epoch) - 1
    if source_epoch < 0:
        raise PostgresContractProbeError(
            "historical compute source epoch is unavailable"
        )
    policy = {
        "policy_id": "restart-rehearsal-no-burn-v2",
        "enabled": True,
        "research_lab_emission_percent": 20.0,
        "reward_epochs": 20,
        "reimbursement_epochs": 20,
        "reimbursement_max_cost_multiplier_with_champions": 2.0,
        "champion_threshold_points": 1.0,
        "champion_min_alpha_percent": 7.0,
        "champion_extra_alpha_percent_per_point": 0.3,
        "champion_max_alpha_percent": 30.0,
        "champion_placeholder_alpha_percent": 0.0001,
        "champion_queue_trigger_ratio": 0.5,
        "usd_per_0_1_percent_epoch": 1.0,
        "enable_conservative": False,
        "enable_champ_cap": False,
    }
    reimbursements = _historical_compute_reimbursements(
        source_root=source_root,
        source_epoch=source_epoch,
    )
    allocation = allocate_research_lab_epoch(
        source_epoch,
        policy,
        reimbursements,
        [],
    )
    if (
        allocation.get("allocation_hash") != sha256_json(
            {
                key: value
                for key, value in allocation.items()
                if key != "allocation_hash"
            }
        )
        or float(allocation.get("reimbursement_alpha_percent") or 0) != 20.0
        or float(allocation.get("unallocated_percent") or 0) != 0.0
    ):
        raise PostgresContractProbeError(
            "historical compute allocation did not conserve the Lab cap"
        )

    allocation_hash = str(allocation["allocation_hash"])
    snapshot_row = {
        "allocation_id": "lab_allocation:" + allocation_hash,
        "schema_version": "1.0",
        "epoch": source_epoch,
        "netuid": int(netuid),
        "policy_id": str(policy["policy_id"]),
        "snapshot_status": "active",
        "lab_cap_alpha_percent": allocation["lab_cap_percent"],
        "reimbursement_alpha_percent": allocation[
            "reimbursement_alpha_percent"
        ],
        "champion_alpha_percent": allocation["champion_alpha_percent"],
        "queued_champion_alpha_percent": allocation[
            "queued_champion_alpha_percent"
        ],
        "unallocated_alpha_percent": allocation["unallocated_percent"],
        "input_hash": allocation["input_hash"],
        "allocation_hash": allocation_hash,
        "allocation_doc": allocation,
        "source_add_alpha_percent": allocation.get(
            "source_add_alpha_percent",
            0,
        ),
    }

    def hash_ref(label: str) -> str:
        return sha256_json({"fixture": label, "epoch": source_epoch})

    settlement_body = {
        "schema_version": LEGACY_SETTLEMENT_SCHEMA_VERSION,
        "netuid": int(netuid),
        "epoch_id": source_epoch,
        "allocation_hash": allocation_hash,
        "allocation_doc": allocation,
        "validator_hotkey": (
            "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
        ),
        "legacy_bundle_weights_hash": hash_ref("legacy-weights").split(
            ":", 1
        )[1],
        "legacy_bundle_block": source_epoch * 360 + 99,
        "chain_compare_hash": hash_ref("chain-compare"),
        "chain_vector_tolerance_u16": 1,
        "chain_target_block": (source_epoch + 1) * 360 - 1,
        "chain_target_block_hash": hash_ref("chain-target-block"),
        "chain_finalized_head_block": (source_epoch + 1) * 360,
        "validator_uid": 0,
        "weights_storage_key_hash": hash_ref("weights-storage-key"),
        "audit_event_hash": hash_ref("audit-event"),
        "audit_payload_hash": hash_ref("audit-payload"),
        "checkpoint_merkle_root": hash_ref("checkpoint-merkle-root"),
        "checkpoint_number": 1,
        "checkpoint_event_sequence": 1,
        "arweave_tx_id": "R" * 43,
    }
    settlement_doc = validate_legacy_settlement_document_v2(
        {
            **settlement_body,
            "settlement_hash": sha256_json(settlement_body),
        }
    )

    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=source_epoch,
    )
    coordinator_config_hash = sha256_json(
        {
            "candidate_sha": candidate_sha,
            "fixture": "historical-compute-coordinator-config",
        }
    )
    coordinator_boot = fixture._boot(
        role="gateway_coordinator",
        key=fixture.coordinator_key,
        config_hash=coordinator_config_hash,
        release_identity=coordinator_release_identity,
    )
    settlement_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.legacy_finalized_allocation.v2",
        job_id="restart-rehearsal-historical-compute-settlement",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=coordinator_config_hash,
        input_root=sha256_json(
            {
                "kind": "historical-compute-allocation",
                "allocation_hash": allocation_hash,
            }
        ),
        output_root=sha256_json(settlement_doc),
        parents=[],
        sequence=804,
    )
    settlement_receipt_hash = str(settlement_receipt["receipt_hash"])
    migration_row = {
        "netuid": int(netuid),
        "epoch_id": source_epoch,
        "schema_version": LEGACY_SETTLEMENT_SCHEMA_VERSION,
        "allocation_hash": allocation_hash,
        "settlement_hash": settlement_doc["settlement_hash"],
        "settlement_receipt_hash": settlement_receipt_hash,
        "allocation_doc": allocation,
        "settlement_doc": settlement_doc,
    }
    graph = build_receipt_graph(
        root_receipt_hash=settlement_receipt_hash,
        boot_identities=[coordinator_boot],
        receipts=[settlement_receipt],
        transport_attempts=[],
        host_operations=[],
    )
    validated = validate_legacy_settlement_migrations_v2(
        [migration_row],
        receipt_graphs={settlement_receipt_hash: graph},
    )
    if (
        len(validated) != 1
        or validated[0].get("allocation_hash") != allocation_hash
        or validated[0].get("allocation_doc") != allocation
    ):
        raise PostgresContractProbeError(
            "historical compute finalized authority did not validate"
        )

    database.psql(
        "".join(
            (
                _json_insert_sql(
                    "research_lab_emission_allocation_snapshots",
                    _deterministic_seed_row(snapshot_row),
                ),
                _json_insert_sql(
                    "research_lab_attested_boot_identities_v2",
                    _deterministic_seed_row(
                        boot_storage_row(coordinator_boot)
                    ),
                ),
                _json_insert_sql(
                    "research_lab_attested_execution_receipts_v2",
                    _deterministic_seed_row(
                        receipt_storage_row(settlement_receipt)
                    ),
                ),
                _json_insert_sql(
                    "research_lab_legacy_finalized_allocation_migrations_v2",
                    _deterministic_seed_row(migration_row),
                ),
            )
        )
    )

    rows = {
        "research_lab_emission_allocation_current": [
            _single_relation_row(
                database,
                relation="research_lab_emission_allocation_current",
                where_sql="epoch = %d AND netuid = %d"
                % (source_epoch, int(netuid)),
            )
        ],
        "research_lab_legacy_finalized_allocation_migrations_v2": [
            _single_relation_row(
                database,
                relation=(
                    "research_lab_legacy_finalized_allocation_migrations_v2"
                ),
                where_sql="epoch_id = %d AND netuid = %d"
                % (source_epoch, int(netuid)),
            )
        ],
        "research_lab_attested_boot_identities_v2": [
            _single_relation_row(
                database,
                relation="research_lab_attested_boot_identities_v2",
                where_sql="boot_identity_hash = '%s'"
                % coordinator_boot["boot_identity_hash"],
            )
        ],
        "research_lab_attested_execution_receipts_v2": [
            _single_relation_row(
                database,
                relation="research_lab_attested_execution_receipts_v2",
                where_sql="receipt_hash = '%s'" % settlement_receipt_hash,
            )
        ],
    }
    return rows


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    declaration_counts = _validate_required_migration_declarations(args.source_root)
    coordinator_release_identity = _load_coordinator_release_identity(
        args.release_build_input,
        candidate_sha=args.candidate_sha,
    )
    database = DisposablePostgres(state_root=args.state_root)
    try:
        database.start()
        scripts = args.source_root / "scripts"
        applied = []
        database.psql(ALLOCATION_MIGRATION_PREREQUISITES_SQL)
        database.apply_migration(
            scripts / ALLOCATION_CANDIDATE_MIGRATION
        )
        applied.append(ALLOCATION_CANDIDATE_MIGRATION)
        database.psql(GIT_TREE_CANDIDATE_PREREQUISITES_SQL)
        database.apply_migration(scripts / ALLOCATION_SCHEMA_MIGRATION)
        applied.append(ALLOCATION_SCHEMA_MIGRATION)
        for name in MIGRATIONS_BEFORE_TRANSPORT_FIX:
            database.apply_migration(scripts / name)
            applied.append(name)
            if name == "86-research-lab-attested-v2-authority.sql":
                database.apply_migration(
                    scripts / ALLOCATION_CONTAINMENT_MIGRATION
                )
                applied.append(ALLOCATION_CONTAINMENT_MIGRATION)

        fixture = SanitizedWeightFixture(
            candidate_sha=args.candidate_sha,
            epoch_id=args.epoch_id,
        )
        attempt = fixture.source_attempt(
            category="chain-settlement-contract",
            job_id="postgres-contract-transport",
            purpose="research_lab.chain_weight_observation.v1",
            sequence=700,
            provider_id="bittensor_chain",
            host="entrypoint-finney.opentensor.ai",
            method="WSS",
        )
        attempt_sql = _json_insert_sql(
            "research_lab_attested_transport_attempts_v2",
            transport_storage_row(attempt),
        )
        rejected = database.psql(attempt_sql, check=False)
        if rejected.returncode == 0:
            raise PostgresContractProbeError(
                "pre-128 V1 transport evidence unexpectedly persisted"
            )
        if (
            "research_lab_attested_transport_attempts_v2_purpose_check"
            not in rejected.stderr
        ):
            raise PostgresContractProbeError(
                "pre-128 transport rejection differed: %s" % rejected.stderr.strip()
            )

        database.apply_migration(scripts / TRANSPORT_FIX_MIGRATION)
        applied.append(TRANSPORT_FIX_MIGRATION)
        database.psql(attempt_sql)

        contract_result = database.psql(
            """
            SELECT
                public.research_lab_attested_transport_purpose_contract_v2()
                ::text;
            """,
            tuples_only=True,
        )
        transport_contract = json.loads(contract_result.stdout.strip())
        definition = str(transport_contract.get("constraint_definition") or "")
        if (
            transport_contract.get("constraint_valid") is not True
            or "research_lab.chain_weight_observation.v1" not in definition
            or "research_lab.chain_realized_epoch_settlement.v1" not in definition
        ):
            raise PostgresContractProbeError(
                "post-128 transport purpose contract is incomplete"
            )

        local_attempt = build_transport_attempt(
            request_id="f" * 32,
            logical_operation_id="provider-preflight-local-cache",
            job_id="postgres-contract-local-transport",
            purpose="research_lab.provider_preflight.v2",
            provider_id="exa",
            attempt_number=0,
            method="POST",
            destination_host="api.exa.ai",
            destination_port=443,
            path_hash=sha256_json({"path": "/search"}),
            nonsecret_headers_hash=sha256_json({"accept": "application/json"}),
            body_hash=sha256_json({"query": "rehearsal"}),
            credential_ref_hash=sha256_json({"credential": "attested-local"}),
            retry_policy_hash=sha256_json({"retry": "provider-preflight"}),
            timeout_ms=30000,
            started_at="2026-07-10T00:00:00Z",
            terminal_status="attested_local_response",
            http_status=200,
            response_hash=sha256_json({"response": "cached"}),
            request_artifact_hash=sha256_json({"artifact": "local-request"}),
            response_artifact_hash=sha256_json({"artifact": "local-response"}),
            tls_peer_chain_hash=None,
            tls_protocol=None,
            failure_code=None,
            completed_at="2026-07-10T00:00:01Z",
        )
        local_attempt_sql = _json_insert_sql(
            "research_lab_attested_transport_attempts_v2",
            transport_storage_row(local_attempt),
        )
        local_rejected = database.psql(local_attempt_sql, check=False)
        if local_rejected.returncode == 0:
            raise PostgresContractProbeError(
                "pre-129 attested local transport unexpectedly persisted"
            )
        if (
            "check constraint" not in local_rejected.stderr.lower()
            or "transport_attempts" not in local_rejected.stderr
        ):
            raise PostgresContractProbeError(
                "pre-129 attested local rejection differed: %s"
                % local_rejected.stderr.strip()
            )

        database.apply_migration(scripts / TRANSPORT_TERMINAL_MIGRATION)
        applied.append(TRANSPORT_TERMINAL_MIGRATION)
        database.psql(local_attempt_sql)
        terminal_contract_result = database.psql(
            """
            SELECT
                public.research_lab_attested_transport_terminal_contract_v2()
                ::text;
            """,
            tuples_only=True,
        )
        terminal_contract = json.loads(
            terminal_contract_result.stdout.strip()
        )
        terminal_constraints = terminal_contract.get("constraints")
        if not isinstance(terminal_constraints, Mapping) or set(
            terminal_constraints
        ) != {
            "research_lab_transport_terminal_status_v2_check",
            "research_lab_transport_terminal_shape_v2_check",
        }:
            raise PostgresContractProbeError(
                "post-129 transport terminal contract is incomplete"
            )
        for constraint in terminal_constraints.values():
            definition = str(constraint.get("constraint_definition") or "")
            if (
                constraint.get("constraint_valid") is not True
                or "attested_local_response" not in definition
            ):
                raise PostgresContractProbeError(
                    "post-129 transport terminal constraint is invalid"
                )

        database.apply_migration(scripts / PROVIDER_OUTCOME_APPEND_MIGRATION)
        applied.append(PROVIDER_OUTCOME_APPEND_MIGRATION)
        database.apply_migration(
            scripts / PROVIDER_OUTCOME_BACKPRESSURE_MIGRATION
        )
        applied.append(PROVIDER_OUTCOME_BACKPRESSURE_MIGRATION)
        pre_contention_contract = database.psql(
            """
            SELECT public.research_lab_provider_outcome_contention_contract_v2()
                   ::text;
            """,
            check=False,
        )
        if (
            pre_contention_contract.returncode == 0
            or "research_lab_provider_outcome_contention_contract_v2"
            not in pre_contention_contract.stderr
            or "does not exist" not in pre_contention_contract.stderr
        ):
            raise PostgresContractProbeError(
                "pre-133 provider outcome contention contract did not fail closed"
            )
        pre_head_contract = database.psql(
            """
            SELECT public.research_lab_provider_outcome_contention_contract_v3()
                   ::text;
            """,
            check=False,
        )
        if (
            pre_head_contract.returncode == 0
            or "research_lab_provider_outcome_contention_contract_v3"
            not in pre_head_contract.stderr
            or "does not exist" not in pre_head_contract.stderr
        ):
            raise PostgresContractProbeError(
                "pre-134 provider outcome head contract did not fail closed"
            )

        rows, verified, fixture = _settlement_fixture(
            candidate_sha=args.candidate_sha,
            epoch_id=args.epoch_id,
        )
        database.psql("".join(_json_insert_sql(table, row) for table, row in rows))
        view_result = database.psql(
            """
            SELECT pg_catalog.row_to_json(authority)::text
            FROM public.research_lab_finalized_allocation_epochs_v2 authority;
            """,
            tuples_only=True,
        )
        view_rows = [
            json.loads(line) for line in view_result.stdout.splitlines() if line.strip()
        ]
        if len(view_rows) != 1:
            raise PostgresContractProbeError(
                "finalized allocation view returned %d rows" % len(view_rows)
            )
        view_row = view_rows[0]
        if tuple(view_row) != EXPECTED_FINALIZED_VIEW_COLUMNS:
            raise PostgresContractProbeError(
                "finalized allocation view columns differ: %s" % ",".join(view_row)
            )
        if "weight_receipt_hash" in view_row:
            raise PostgresContractProbeError(
                "finalized allocation view synthesized weight_receipt_hash"
            )
        authority = _preliminary_finalized_bundle_authority_v1(view_row)
        if authority["weight_receipt_hash"] != verified["weight_receipt_hash"]:
            raise PostgresContractProbeError(
                "settlement authority weight receipt differs"
            )
        measured_settlement = _measured_settlement_receipt_contract(
            authority=authority,
            verified_bundle=verified,
            fixture=fixture,
        )
        package = measured_settlement.pop("package")
        settlement_receipt = measured_settlement.pop("receipt")
        if (
            package["settlement_doc"]["schema_version"]
            != "leadpoet.research_lab_chain_realized_epoch_settlement.v3"
            or not package["credits"]
        ):
            raise PostgresContractProbeError(
                "marked lifetime settlement fixture is incomplete"
            )
        coordinator_boot = next(
            identity
            for identity in authority["bundle_doc"]["receipt_graph"][
                "boot_identities"
            ]
            if identity["physical_role"] == "gateway_coordinator"
        )
        historical_receipt, historical_settlement, historical_credit = (
            _historical_v1_settlement_rows(
                fixture=fixture,
                coordinator_boot=coordinator_boot,
                netuid=int(verified["netuid"]),
                epoch_id=int(verified["epoch_id"]) - 1,
            )
        )
        activation_row = {
            "netuid": int(verified["netuid"]),
            "schema_version": (
                "leadpoet.research_lab_chain_realized_"
                "settlement_activation.v1"
            ),
            "first_epoch_id": int(verified["epoch_id"]) - 1,
            "source_bundle_hash": str(verified["bundle_hash"]),
            "source_bundle_epoch_id": int(verified["epoch_id"]) - 1,
            "source_finalized_block": int(authority["finalized_block"]) - 1,
        }
        database.psql(
            "".join(
                (
                    _json_insert_sql(
                        "research_lab_attested_execution_receipts_v2",
                        historical_receipt,
                    ),
                    _json_insert_sql(
                        "research_lab_chain_realized_settlement_activation_v1",
                        activation_row,
                    ),
                    _json_insert_sql(
                        "research_lab_chain_realized_epoch_settlements_v1",
                        historical_settlement,
                    ),
                    _json_insert_sql(
                        "research_lab_chain_realized_obligation_credits_v1",
                        historical_credit,
                    ),
                )
            )
        )

        settlement_row, credit_rows = _settlement_persistence_rows(
            package=package,
            receipt_hash=str(settlement_receipt["receipt_hash"]),
        )
        lifetime_rpc = (
            "persist_research_lab_chain_realized_lifetime_settlement_v2"
        )
        persistence_sql = _json_rpc_sql(
            lifetime_rpc,
            settlement_row,
            credit_rows,
        )
        pre_lifetime = database.psql(persistence_sql, check=False)
        if pre_lifetime.returncode == 0 or (
            lifetime_rpc not in pre_lifetime.stderr
            or "does not exist" not in pre_lifetime.stderr
        ):
            raise PostgresContractProbeError(
                "pre-132 lifetime persistence did not fail closed: %s"
                % pre_lifetime.stderr.strip()
            )

        database.apply_migration(scripts / CHAMPION_LIFETIME_CREDIT_MIGRATION)
        applied.append(CHAMPION_LIFETIME_CREDIT_MIGRATION)
        historical_result = database.psql(
            """
            SELECT pg_catalog.json_build_object(
                'schema_version', schema_version,
                'champion_credit_policy', champion_credit_policy,
                'document_has_policy', credit_doc ? 'champion_credit_policy',
                'credited_alpha_percent', credited_alpha_percent::TEXT
            )::text
            FROM public.research_lab_chain_realized_obligation_credits_v1
            WHERE netuid = 71
              AND epoch_id = %d
              AND obligation_source_id = 'grandfathered-champion';
            """
            % (int(verified["epoch_id"]) - 1),
            tuples_only=True,
        )
        historical_contract = json.loads(historical_result.stdout.strip())
        if historical_contract != {
            "schema_version": (
                "leadpoet.research_lab_chain_realized_obligation_credit.v1"
            ),
            "champion_credit_policy": "scheduled_bonus_v1",
            "document_has_policy": False,
            "credited_alpha_percent": "5.000000000000",
        }:
            raise PostgresContractProbeError(
                "migration 132 changed grandfathered settlement credit"
            )

        database.psql(
            _json_insert_sql(
                "research_lab_attested_execution_receipts_v2",
                receipt_storage_row(settlement_receipt),
            )
        )
        first_persistence = json.loads(
            database.psql(
                persistence_sql,
                tuples_only=True,
            ).stdout.strip()
        )
        repeated_persistence = json.loads(
            database.psql(
                persistence_sql,
                tuples_only=True,
            ).stdout.strip()
        )
        expected_credit_hashes = sorted(
            str(row["credit_hash"]) for row in credit_rows
        )
        expected_persistence = {
            "schema_version": (
                "leadpoet.research_lab_chain_realized_"
                "settlement_persistence.v1"
            ),
            "netuid": int(verified["netuid"]),
            "epoch_id": int(verified["epoch_id"]),
            "settlement_hash": str(package["settlement_hash"]),
            "settlement_receipt_hash": str(
                settlement_receipt["receipt_hash"]
            ),
            "credit_count": len(credit_rows),
            "credit_hashes": expected_credit_hashes,
        }
        if (
            first_persistence != expected_persistence
            or repeated_persistence != expected_persistence
        ):
            raise PostgresContractProbeError(
                "lifetime settlement persistence is not exact and idempotent"
            )
        lifetime_contract_result = database.psql(
            """
            SELECT
                public.research_lab_champion_lifetime_credit_contract_v1()
                ::text;
            """,
            tuples_only=True,
        )
        lifetime_contract = json.loads(
            lifetime_contract_result.stdout.strip()
        )
        lifetime_constraints = lifetime_contract.get("constraints")
        if (
            lifetime_contract.get("schema_version")
            != (
                "leadpoet.research_lab_champion_"
                "lifetime_credit_contract.v1"
            )
            or lifetime_contract.get("champion_credit_policy")
            != "accelerated_lifetime_cap_v1"
            or lifetime_contract.get("credit_policy_column") is not True
            or not isinstance(lifetime_constraints, Mapping)
            or set(lifetime_constraints)
            != {
                "research_lab_chain_settlement_schema_check",
                "research_lab_chain_settlement_champion_policy_check",
                "research_lab_chain_credit_schema_policy_check",
                "research_lab_chain_credit_policy_amount_check",
            }
            or any(
                constraint.get("validated") is not True
                for constraint in lifetime_constraints.values()
            )
        ):
            raise PostgresContractProbeError(
                "post-131 lifetime credit contract is incomplete"
            )

        database.apply_migration(
            scripts / PROVIDER_OUTCOME_CONTENTION_STATUS_MIGRATION
        )
        applied.append(PROVIDER_OUTCOME_CONTENTION_STATUS_MIGRATION)
        contention_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_provider_outcome_contention_contract_v2()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if contention_contract != {
            "schema_version": (
                "leadpoet.provider_outcome_contention_contract.v2"
            ),
            "lock_contention_status": "busy",
            "stale_lineage_status": "conflict",
        }:
            raise PostgresContractProbeError(
                "post-133 provider outcome contention contract differs"
            )
        pre_head_contract = database.psql(
            """
            SELECT public.research_lab_provider_outcome_contention_contract_v3()
                   ::text;
            """,
            check=False,
        )
        if (
            pre_head_contract.returncode == 0
            or "research_lab_provider_outcome_contention_contract_v3"
            not in pre_head_contract.stderr
            or "does not exist" not in pre_head_contract.stderr
        ):
            raise PostgresContractProbeError(
                "pre-134 provider outcome head contract did not fail closed"
            )

        database.apply_migration(
            scripts / PROVIDER_OUTCOME_HEAD_CONTENTION_MIGRATION
        )
        applied.append(PROVIDER_OUTCOME_HEAD_CONTENTION_MIGRATION)
        head_contention_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_provider_outcome_contention_contract_v3()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if head_contention_contract != {
            "schema_version": (
                "leadpoet.provider_outcome_contention_contract.v3"
            ),
            "lock_contention_status": "busy",
            "stale_lineage_status": "conflict",
            "candidate_checkpoint_hash": True,
            "conflict_head_checkpoint_row": "encrypted_or_null",
        }:
            raise PostgresContractProbeError(
                "post-134 provider outcome head contract differs"
            )

        database.apply_migration(
            scripts / ACTIVE_MODEL_RESULT_REPLAY_MIGRATION
        )
        applied.append(ACTIVE_MODEL_RESULT_REPLAY_MIGRATION)
        active_model_replay_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_active_model_replay_contract_v2()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        replay_constraints = active_model_replay_contract.get("constraints")
        replay_constraint_definitions = (
            "\n".join(
                str(constraint.get("constraint_definition") or "")
                for constraint in replay_constraints.values()
            )
            if isinstance(replay_constraints, Mapping)
            else ""
        )
        if (
            active_model_replay_contract.get("schema_version")
            != "leadpoet.active_model_replay_contract.v2"
            or active_model_replay_contract.get("operation")
            != "attest_active_private_model"
            or active_model_replay_contract.get("purpose")
            != "research_lab.active_private_model.v2"
            or not isinstance(replay_constraints, Mapping)
            or set(replay_constraints)
            != {
                "research_lab_attested_execution_results_v2_operation_check",
                "research_lab_attested_execution_results_v2_purpose_check",
                "research_lab_attested_execution_results_v2_operation_purpose_check",
            }
            or any(
                constraint.get("constraint_valid") is not True
                for constraint in replay_constraints.values()
            )
            or "attest_active_private_model"
            not in replay_constraint_definitions
            or "research_lab.active_private_model.v2"
            not in replay_constraint_definitions
        ):
            raise PostgresContractProbeError(
                "post-135 active-model replay contract differs"
            )
        provider_outcome_append = _provider_outcome_append_contract(database)
        historical_compute_seed_rows = (
            _historical_compute_allocation_seed_rows(
                database=database,
                source_root=args.source_root,
                candidate_sha=args.candidate_sha,
                current_epoch=args.epoch_id,
                netuid=int(verified["netuid"]),
                coordinator_release_identity=coordinator_release_identity,
            )
        )

        tampered = copy.deepcopy(view_row)
        tampered["finalization_doc"]["weight_receipt_hash"] = "sha256:" + "0" * 64
        try:
            _preliminary_finalized_bundle_authority_v1(tampered)
        except ChampionSettlementV2Error as exc:
            if "weight_receipt_hash" not in str(exc):
                raise PostgresContractProbeError(
                    "tampered authority failed for the wrong reason: %s" % exc
                ) from exc
        else:
            raise PostgresContractProbeError(
                "tampered settlement authority was accepted"
            )

        contract = _relation_contract(database)
        view_columns = contract["relations"][
            "research_lab_finalized_allocation_epochs_v2"
        ]["columns"]
        if tuple(view_columns) != EXPECTED_FINALIZED_VIEW_COLUMNS:
            raise PostgresContractProbeError(
                "catalog and finalized view projections differ"
            )
        return {
            "schema_version": "leadpoet.restart_rehearsal.postgres_contract.v1",
            "candidate_sha": args.candidate_sha,
            "applied_migrations": applied,
            "relations": contract["relations"],
            "rpcs": contract["rpcs"],
            "checks": {
                "pre_128_transport_rejected": True,
                "post_128_transport_persisted": True,
                "transport_contract_valid": True,
                "pre_129_attested_local_transport_rejected": True,
                "post_129_attested_local_transport_persisted": True,
                "transport_terminal_contract_valid": True,
                "pre_133_provider_outcome_contract_rejected": True,
                "post_133_provider_outcome_contract_valid": True,
                "pre_134_provider_outcome_head_contract_rejected": True,
                "post_134_provider_outcome_head_contract_valid": True,
                "post_135_active_model_replay_contract_valid": True,
                "provider_outcome_append_atomic": True,
                "provider_outcome_contention_zero_rollback": True,
                "provider_outcome_conflict_head_exact": True,
                "pre_132_lifetime_credit_rejected": True,
                "post_132_lifetime_credit_persisted": True,
                "lifetime_credit_rpc_idempotent": True,
                "grandfathered_credit_unchanged": True,
                "lifetime_credit_contract_valid": True,
                "finalized_view_projection_exact": True,
                "finalized_view_seed_available": True,
                "historical_compute_schema_migrations_applied": True,
                "historical_compute_finalized_authority_seed_available": True,
                "historical_compute_allocation_conserved": True,
                "historical_compute_release_identity_bound": True,
                "settlement_authority_parsed": True,
                "measured_settlement_receipt_projection_exact": True,
                "tampered_weight_receipt_rejected": True,
                "required_schema_migrations_declared": True,
            },
            "seed_rows": {
                "research_lab_finalized_allocation_epochs_v2": [view_row],
                **historical_compute_seed_rows,
            },
            "measured_settlement": measured_settlement,
            "champion_lifetime_credit": {
                "policy": "accelerated_lifetime_cap_v1",
                "credit_count": len(credit_rows),
                "credit_hashes": expected_credit_hashes,
                "historical_contract": historical_contract,
                "persistence": first_persistence,
            },
            "provider_outcome_append": provider_outcome_append,
            "provider_outcome_contention_contract": head_contention_contract,
            "required_schema_declarations": declaration_counts,
        }
    finally:
        database.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--release-build-input", type=Path, required=True)
    parser.add_argument("--epoch-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.candidate_sha):
        raise SystemExit("candidate SHA must be lowercase full-length hex")
    if args.epoch_id is None:
        args.epoch_id = load_rehearsal_current_settlement_epoch_id(
            args.source_root
        )
    args.state_root.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_probe(args)
    except Exception as exc:
        diagnostic = {
            "schema_version": "leadpoet.restart_rehearsal.failure.v1",
            "stage": "migration_backed_v2_settlement_contract",
            "candidate_sha": args.candidate_sha,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(
            "REHEARSAL_POSTGRES_CONTRACT_ERROR "
            + json.dumps(diagnostic, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        raise
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.output.chmod(0o600)
    print(
        "REHEARSAL_POSTGRES_CONTRACT_OK "
        + json.dumps(result["checks"], sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
