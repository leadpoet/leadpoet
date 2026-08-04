from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import pytest

from gateway.research_lab.attested_v2_store import (
    _REPLAYABLE_EXECUTION_PAIRS,
)
from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS
from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
    build_allocation_settlement_frontier_bootstrap_v2,
    frontier_bootstrap_artifact_hashes_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.attested_v2 import ROLE_PURPOSES, sha256_json


ROOT = Path(__file__).resolve().parents[1]
FRONTIER_MIGRATION = (
    ROOT / "scripts/137-research-lab-allocation-settlement-frontier.sql"
)
MIGRATION_NAME = "scripts/139-research-lab-allocation-frontier-bootstrap.sql"
MIGRATION = ROOT / MIGRATION_NAME
SQL = MIGRATION.read_text(encoding="utf-8")
HISTORICAL_SOURCE_MIGRATION = (
    ROOT
    / "scripts/140-research-lab-allocation-frontier-historical-source.sql"
)
HISTORICAL_SOURCE_SQL = HISTORICAL_SOURCE_MIGRATION.read_text(encoding="utf-8")
SOURCE_CONTRACT_MIGRATION_NAME = (
    "scripts/141-research-lab-allocation-frontier-source-contract.sql"
)
SOURCE_CONTRACT_SQL = (ROOT / SOURCE_CONTRACT_MIGRATION_NAME).read_text(
    encoding="utf-8"
)
SOURCE_CATALOG_REPLAY_MIGRATION_NAME = (
    "scripts/142-research-lab-source-catalog-result-replay.sql"
)
SOURCE_CATALOG_REPLAY_SQL = (
    ROOT / SOURCE_CATALOG_REPLAY_MIGRATION_NAME
).read_text(encoding="utf-8")
RPC = "persist_research_lab_allocation_frontier_bootstrap_v2"
CONTRACT_RPC = "research_lab_allocation_frontier_bootstrap_contract_v2"
SOURCE_CONTRACT_RPC = (
    "research_lab_allocation_frontier_historical_source_contract_v1"
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _literal(value: object) -> str:
    return "'%s'::jsonb" % json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("'", "''")


def _text(value: str) -> str:
    return "'%s'" % value.replace("'", "''")


def _row_documents(
    *,
    epoch: int = 200,
    settlement_frontier: str = "missing",
) -> tuple[dict, dict, dict, dict, dict]:
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=epoch,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    source_state = {
        "epoch": epoch,
        "netuid": 71,
    }
    if settlement_frontier == "null":
        source_state["settlement_frontier"] = None
    elif settlement_frontier == "non_null":
        source_state["settlement_frontier"] = {"unexpected": True}
    elif settlement_frontier != "missing":
        raise ValueError("unknown settlement frontier fixture")
    source_state_hash = sha256_json(source_state)
    allocation_receipt_hash = _sha("1")
    allocation_result = {
        "allocation": {"epoch": epoch, "netuid": 71},
        "source_state": source_state,
        "source_state_hash": source_state_hash,
    }
    allocation_row = {
        "receipt_hash": allocation_receipt_hash,
        "schema_version": "leadpoet.attested_execution_result.v2",
        "role": "gateway_coordinator",
        "operation": "research_lab_allocation",
        "purpose": "research_lab.allocation.v2",
        "job_id": "allocation:%d" % epoch,
        "epoch_id": epoch,
        "sequence": 0,
        "release_hash": _sha("a"),
        "input_root": _sha("b"),
        "output_root": _sha("c"),
        "artifact_root": _sha("d"),
        "result_hash": sha256_json(allocation_result),
        "artifact_hashes": [source_state_hash],
        "result_doc": allocation_result,
    }
    allocation_receipt = {
        "receipt_hash": allocation_receipt_hash,
        "role": allocation_row["role"],
        "purpose": allocation_row["purpose"],
        "job_id": allocation_row["job_id"],
        "epoch_id": epoch,
        "sequence": 0,
        "input_root": allocation_row["input_root"],
        "output_root": allocation_row["output_root"],
        "artifact_root": allocation_row["artifact_root"],
        "receipt_status": "succeeded",
        "receipt_doc": {"parent_receipt_hashes": []},
    }
    bootstrap = build_allocation_settlement_frontier_bootstrap_v2(
        netuid=71,
        bootstrap_epoch=epoch + 1,
        allocation_source_receipt_hash=allocation_receipt_hash,
        source_state_hash=source_state_hash,
        frontier=frontier,
    )
    bootstrap_receipt_hash = _sha("2")
    bootstrap_row = {
        "receipt_hash": bootstrap_receipt_hash,
        "schema_version": "leadpoet.attested_execution_result.v2",
        "role": "gateway_coordinator",
        "operation": "allocation_settlement_frontier_bootstrap_v2",
        "purpose": "research_lab.allocation_settlement_frontier_bootstrap.v2",
        "job_id": "allocation-frontier-bootstrap:%d" % (epoch + 1),
        "epoch_id": epoch + 1,
        "sequence": 0,
        "release_hash": _sha("e"),
        "input_root": _sha("f"),
        "output_root": _sha("3"),
        "artifact_root": _sha("4"),
        "result_hash": sha256_json(bootstrap),
        "artifact_hashes": list(frontier_bootstrap_artifact_hashes_v2(bootstrap)),
        "result_doc": bootstrap,
    }
    bootstrap_receipt = {
        "receipt_hash": bootstrap_receipt_hash,
        "role": bootstrap_row["role"],
        "purpose": bootstrap_row["purpose"],
        "job_id": bootstrap_row["job_id"],
        "epoch_id": bootstrap_row["epoch_id"],
        "sequence": 0,
        "input_root": bootstrap_row["input_root"],
        "output_root": bootstrap_row["output_root"],
        "artifact_root": bootstrap_row["artifact_root"],
        "receipt_status": "succeeded",
        "receipt_doc": {
            "parent_receipt_hashes": [allocation_receipt_hash],
        },
    }
    return (
        allocation_row,
        allocation_receipt,
        bootstrap_row,
        bootstrap_receipt,
        frontier,
    )


def test_migration_is_additive_private_and_declares_current_contract() -> None:
    assert SQL.lstrip().startswith("-- Measured, bounded activation")
    assert re.search(r"\bBEGIN\s*;", SQL)
    assert re.search(r"\bCOMMIT\s*;", SQL)
    assert not re.search(r"^\s*(?:UPDATE|DELETE\s+FROM)\s+", SQL, re.M)
    assert "pg_advisory_xact_lock(139, requested_netuid)" in SQL
    assert "allocation_frontier_bootstrap_already_initialized" in SQL
    assert "allocation_frontier_bootstrap_source_invalid" in SQL
    assert "parent_receipt_hashes" in SQL
    assert len(RPC.encode("utf-8")) <= 63
    assert (MIGRATION_NAME, RPC) in REQUIRED_SUPABASE_V2_RPCS
    assert (MIGRATION_NAME, CONTRACT_RPC) in REQUIRED_SUPABASE_V2_RPCS

    for role, expected_purposes in ROLE_PURPOSES.items():
        match = re.search(
            rf"role = '{re.escape(role)}' AND purpose IN \((.*?)\n\s*\)\)",
            SQL,
            re.DOTALL,
        )
        assert match is not None, role
        assert set(re.findall(r"'([^']+)'", match.group(1))) == set(
            expected_purposes
        )

    pair_clause = SOURCE_CATALOG_REPLAY_SQL.split(
        "research_lab_attested_exec_results_v2_op_purpose_check",
        1,
    )[1].split(") NOT VALID;", 1)[0]
    for operation, purpose in _REPLAYABLE_EXECUTION_PAIRS:
        assert "operation = '%s'" % operation in pair_clause
        assert "'%s'" % purpose in pair_clause


def test_historical_source_migration_accepts_only_absent_or_null_frontier() -> None:
    assert HISTORICAL_SOURCE_SQL.lstrip().startswith(
        "-- Accept the historical allocation-source shape"
    )
    assert "COALESCE(" in HISTORICAL_SOURCE_SQL
    assert "'settlement_frontier'," in HISTORICAL_SOURCE_SQL
    assert ") IS DISTINCT FROM 'null'::JSONB" in HISTORICAL_SOURCE_SQL
    assert "pg_get_functiondef" in HISTORICAL_SOURCE_SQL
    assert "allocation_frontier_bootstrap_historical_source_guard_missing" in (
        HISTORICAL_SOURCE_SQL
    )


def test_historical_source_contract_is_guarded_and_required_by_preflight() -> None:
    assert "pg_get_functiondef" in SOURCE_CONTRACT_SQL
    assert "COALESCE(" in SOURCE_CONTRACT_SQL
    assert "allocation_frontier_historical_source_contract_missing" in (
        SOURCE_CONTRACT_SQL
    )
    assert "settlement_frontier_compatibility" in SOURCE_CONTRACT_SQL
    assert "missing_or_null" in SOURCE_CONTRACT_SQL
    assert len(SOURCE_CONTRACT_RPC.encode("utf-8")) <= 63
    assert (
        SOURCE_CONTRACT_MIGRATION_NAME,
        SOURCE_CONTRACT_RPC,
    ) in REQUIRED_SUPABASE_V2_RPCS
    assert not re.search(
        r"^\s*(?:UPDATE|DELETE\s+FROM)\s+",
        HISTORICAL_SOURCE_SQL,
        re.M,
    )


@pytest.mark.parametrize(
    ("settlement_frontier", "expect_success"),
    (
        pytest.param("missing", True, id="historical-missing"),
        pytest.param("null", True, id="explicit-null"),
        pytest.param("non_null", False, id="non-null-rejected"),
    ),
)
def test_migration_persists_one_exact_measured_bootstrap(
    settlement_frontier: str,
    expect_success: bool,
) -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for the PostgreSQL migration contract")
    info = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=15
    )
    if info.returncode != 0:
        pytest.skip("Docker daemon is unavailable")

    container = "leadpoet-frontier-bootstrap-%s" % uuid.uuid4().hex[:12]

    def psql(
        statement: str,
        *,
        expect_success: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container,
                "psql",
                "-X",
                "-A",
                "-t",
                "-U",
                "postgres",
                "-d",
                "leadpoet",
                "-v",
                "ON_ERROR_STOP=1",
            ],
            input=statement,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if expect_success:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0, result.stdout
        return result

    setup_sql = """
CREATE ROLE anon NOLOGIN;
CREATE ROLE authenticated NOLOGIN;
CREATE ROLE service_role NOLOGIN;
CREATE TABLE public.research_lab_attested_execution_receipts_v2 (
    receipt_hash TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    purpose TEXT NOT NULL,
    job_id TEXT NOT NULL,
    epoch_id BIGINT NOT NULL,
    sequence INTEGER NOT NULL,
    input_root TEXT NOT NULL,
    output_root TEXT NOT NULL,
    artifact_root TEXT NOT NULL,
    receipt_status TEXT NOT NULL,
    receipt_doc JSONB NOT NULL
);
CREATE TABLE public.research_lab_attested_execution_results_v2 (
    receipt_hash TEXT PRIMARY KEY REFERENCES
        public.research_lab_attested_execution_receipts_v2(receipt_hash),
    schema_version TEXT NOT NULL,
    role TEXT NOT NULL,
    operation TEXT NOT NULL,
    purpose TEXT NOT NULL,
    job_id TEXT NOT NULL,
    epoch_id BIGINT NOT NULL,
    sequence INTEGER NOT NULL,
    release_hash TEXT NOT NULL,
    input_root TEXT NOT NULL,
    output_root TEXT NOT NULL,
    artifact_root TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    artifact_hashes JSONB NOT NULL,
    result_doc JSONB NOT NULL
);
CREATE FUNCTION public.prevent_research_lab_attested_v2_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'append-only relation';
END;
$$;
"""

    def insert(table: str, value: dict) -> None:
        psql(
            "INSERT INTO public.%s SELECT * FROM "
            "jsonb_populate_record(NULL::public.%s, %s);"
            % (table, table, _literal(value))
        )

    def persist(
        frontier: dict,
        receipt_hash: str,
        source_state_hash: str,
        *,
        rpc: str = RPC,
        expect_success: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return psql(
            "SELECT public.%s(%s,%s,%s);"
            % (
                rpc,
                _literal(frontier),
                _text(receipt_hash),
                _text(source_state_hash),
            ),
            expect_success=expect_success,
        )

    (
        allocation_row,
        allocation_receipt,
        bootstrap_row,
        bootstrap_receipt,
        frontier,
    ) = _row_documents(settlement_frontier=settlement_frontier)
    source_state_hash = bootstrap_row["result_doc"]["source_state_hash"]

    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                container,
                "--env",
                "POSTGRES_PASSWORD=postgres",
                "--env",
                "POSTGRES_DB=leadpoet",
                "postgres:15",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        for _ in range(80):
            startup = subprocess.run(
                ["docker", "logs", container],
                capture_output=True,
                text=True,
                timeout=5,
            )
            ready = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "pg_isready",
                    "-U",
                    "postgres",
                    "-d",
                    "leadpoet",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if (
                "PostgreSQL init process complete; ready for start up."
                in startup.stdout + startup.stderr
                and ready.returncode == 0
            ):
                break
            time.sleep(0.25)
        else:
            raise AssertionError("PostgreSQL 15 did not become ready")

        psql(setup_sql)
        psql(FRONTIER_MIGRATION.read_text(encoding="utf-8"))
        psql(SQL)
        psql(SQL)
        psql(HISTORICAL_SOURCE_SQL)
        psql(HISTORICAL_SOURCE_SQL)
        insert("research_lab_attested_execution_receipts_v2", allocation_receipt)
        insert("research_lab_attested_execution_results_v2", allocation_row)
        insert("research_lab_attested_execution_receipts_v2", bootstrap_receipt)
        insert("research_lab_attested_execution_results_v2", bootstrap_row)

        if not expect_success:
            rejected_source = persist(
                frontier,
                bootstrap_row["receipt_hash"],
                source_state_hash,
                expect_success=False,
            )
            assert "allocation_frontier_bootstrap_source_invalid" in (
                rejected_source.stderr
            )
            return

        first = json.loads(
            persist(
                frontier,
                bootstrap_row["receipt_hash"],
                source_state_hash,
            ).stdout.strip()
        )
        replay = json.loads(
            persist(
                frontier,
                bootstrap_row["receipt_hash"],
                source_state_hash,
            ).stdout.strip()
        )
        assert first["status"] == "persisted"
        assert replay["status"] == "already_persisted"
        assert first["frontier_hash"] == frontier["frontier_hash"]

        regular = persist(
            frontier,
            bootstrap_row["receipt_hash"],
            source_state_hash,
            rpc="persist_research_lab_allocation_settlement_frontier_v2",
            expect_success=False,
        )
        assert "allocation_settlement_frontier_source_invalid" in regular.stderr

        wrong_parent = dict(bootstrap_receipt)
        wrong_parent["receipt_hash"] = _sha("5")
        wrong_parent["receipt_doc"] = {"parent_receipt_hashes": [_sha("6")]}
        wrong_bootstrap = dict(bootstrap_row)
        wrong_bootstrap["receipt_hash"] = wrong_parent["receipt_hash"]
        wrong_bootstrap["job_id"] = "allocation-frontier-bootstrap:wrong-parent"
        wrong_parent["job_id"] = wrong_bootstrap["job_id"]
        insert("research_lab_attested_execution_receipts_v2", wrong_parent)
        insert("research_lab_attested_execution_results_v2", wrong_bootstrap)
        rejected = persist(
            frontier,
            wrong_bootstrap["receipt_hash"],
            source_state_hash,
            expect_success=False,
        )
        assert "allocation_frontier_bootstrap_source_invalid" in rejected.stderr

        contract = json.loads(
            psql(
                "SELECT public.%s()::text;" % CONTRACT_RPC
            ).stdout.strip()
        )
        assert contract["schema_version"] == (
            "leadpoet.allocation_frontier_bootstrap_contract.v2"
        )
        assert len(contract["constraints"]) == 4
        assert all(
            item["constraint_valid"] is True
            for item in contract["constraints"].values()
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            capture_output=True,
            text=True,
            timeout=30,
        )
