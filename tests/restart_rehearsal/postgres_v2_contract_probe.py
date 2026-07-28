#!/usr/bin/env python3.11
"""Exercise settlement-critical candidate migrations in disposable PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from gateway.research_lab.champion_settlement_v2 import (
    CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
    ChampionSettlementV2Error,
    _preliminary_finalized_bundle_authority_v1,
    build_chain_realized_settlement_package_v1,
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
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.weight_authority_v2 import (
    validate_published_weight_bundle_v2,
)
from tests.restart_rehearsal.sanitized_weight_fixture import (
    SanitizedWeightFixture,
)


MIGRATIONS_BEFORE_TRANSPORT_FIX = (
    "86-research-lab-attested-v2-authority.sql",
    "99-research-lab-v2-champion-settlement.sql",
    "104-research-lab-attested-result-replay-v2.sql",
    "126-research-lab-chain-realized-settlement.sql",
    "127-research-lab-chain-unattributed-settlement.sql",
)
TRANSPORT_FIX_MIGRATION = "128-research-lab-chain-settlement-transport-purposes.sql"
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


def _settlement_fixture(
    *,
    candidate_sha: str,
    epoch_id: int,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
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
    return rows, verified


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
    return {
        "settlement_hash": package["settlement_hash"],
        "credit_count": len(package["credits"]),
    }


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    declaration_counts = _validate_required_migration_declarations(args.source_root)
    database = DisposablePostgres(state_root=args.state_root)
    try:
        database.start()
        scripts = args.source_root / "scripts"
        applied = []
        for name in MIGRATIONS_BEFORE_TRANSPORT_FIX:
            database.apply_migration(scripts / name)
            applied.append(name)

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

        rows, verified = _settlement_fixture(
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
                "finalized_view_projection_exact": True,
                "settlement_authority_parsed": True,
                "measured_settlement_receipt_projection_exact": True,
                "tampered_weight_receipt_rejected": True,
                "required_schema_migrations_declared": True,
            },
            "measured_settlement": measured_settlement,
            "required_schema_declarations": declaration_counts,
        }
    finally:
        database.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--epoch-id", type=int, default=24208)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.candidate_sha):
        raise SystemExit("candidate SHA must be lowercase full-length hex")
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
