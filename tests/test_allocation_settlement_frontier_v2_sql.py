from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "scripts/137-research-lab-allocation-settlement-frontier.sql"
SQL = (ROOT / MIGRATION).read_text(encoding="utf-8")
FRONTIER_TABLE = "research_lab_allocation_settlement_frontiers_v2"
ACTIVATION_TABLE = (
    "research_lab_allocation_settlement_frontier_activation_v2"
)
RPC = "persist_research_lab_allocation_settlement_frontier_v2"


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _frontier(*, epoch: int, predecessor: str | None) -> dict:
    return build_allocation_settlement_frontier_v2(
        mode=(
            "legacy_full_history_bootstrap"
            if predecessor is None
            else "bounded_delta_v1"
        ),
        netuid=71,
        allocation_epoch=epoch,
        predecessor_frontier_hash=predecessor,
        reward_checkpoints=(),
    )


def _source(*, frontier: dict, receipt_hash: str) -> tuple[dict, dict, str]:
    source_state = {
        "epoch": frontier["allocation_epoch"],
        "netuid": frontier["netuid"],
        "settlement_frontier": frontier,
    }
    source_state_hash = sha256_json(source_state)
    input_root = _sha("a")
    output_root = _sha("b")
    artifact_root = _sha("c")
    row = {
        "receipt_hash": receipt_hash,
        "role": "gateway_coordinator",
        "operation": "research_lab_allocation",
        "purpose": "research_lab.allocation.v2",
        "job_id": "allocation:%d" % frontier["allocation_epoch"],
        "epoch_id": frontier["allocation_epoch"],
        "sequence": 0,
        "input_root": input_root,
        "output_root": output_root,
        "artifact_root": artifact_root,
        "result_doc": {
            "source_state": source_state,
            "source_state_hash": source_state_hash,
        },
        "artifact_hashes": [source_state_hash, frontier["frontier_hash"]],
    }
    receipt = {
        "receipt_hash": receipt_hash,
        "role": row["role"],
        "purpose": row["purpose"],
        "job_id": row["job_id"],
        "epoch_id": row["epoch_id"],
        "sequence": row["sequence"],
        "input_root": input_root,
        "output_root": output_root,
        "artifact_root": artifact_root,
        "receipt_status": "succeeded",
    }
    return row, receipt, source_state_hash


def _literal(value: object) -> str:
    return "'%s'::jsonb" % json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("'", "''")


def _text(value: str) -> str:
    return "'%s'" % value.replace("'", "''")


def test_migration_is_additive_private_and_append_only() -> None:
    assert SQL.lstrip().startswith(
        "-- Bounded, signed allocation settlement frontier."
    )
    assert re.search(r"\bBEGIN\s*;", SQL)
    assert re.search(r"\bCOMMIT\s*;\s*$", SQL)
    assert not re.search(r"^\s*UPDATE\s+", SQL, flags=re.MULTILINE)
    assert not re.search(r"^\s*DELETE\s+FROM\s+", SQL, flags=re.MULTILINE)
    assert SQL.count("BEFORE UPDATE OR DELETE") == 2
    assert SQL.count("prevent_research_lab_attested_v2_mutation()") == 2
    assert SQL.count("ENABLE ROW LEVEL SECURITY") == 2
    assert SQL.count(
        "FROM PUBLIC, anon, authenticated, service_role"
    ) == 2
    assert "TO service_role" in SQL
    assert "ON DELETE RESTRICT" in SQL
    assert "requested_frontier IS NULL" in SQL
    assert "requested_mode IS NULL" in SQL
    assert "requested_checkpoint_count IS NULL" in SQL
    assert "IS DISTINCT FROM 'array'" in SQL


def test_gateway_preflight_requires_frontier_contract() -> None:
    requirements = {
        relation: (migration, tuple(columns))
        for migration, relation, columns in REQUIRED_SUPABASE_V2_SCHEMA
        if relation in {FRONTIER_TABLE, ACTIVATION_TABLE}
    }
    assert requirements == {
        FRONTIER_TABLE: (
            MIGRATION,
            (
                "netuid",
                "allocation_epoch",
                "settled_through_epoch",
                "schema_version",
                "frontier_hash",
                "predecessor_frontier_hash",
                "source_receipt_hash",
                "source_state_hash",
                "frontier_doc",
            ),
        ),
        ACTIVATION_TABLE: (
            MIGRATION,
            (
                "netuid",
                "schema_version",
                "first_allocation_epoch",
                "first_frontier_hash",
                "source_receipt_hash",
            ),
        ),
    }
    assert (MIGRATION, RPC) in REQUIRED_SUPABASE_V2_RPCS


def test_migration_enforces_one_monotonic_frontier_chain() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for the PostgreSQL migration contract")
    info = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=15
    )
    if info.returncode != 0:
        pytest.skip("Docker daemon is unavailable")

    container = "leadpoet-frontier-sql-%s" % uuid.uuid4().hex[:12]

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
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON TABLES TO service_role;
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
    receipt_status TEXT NOT NULL
);
CREATE TABLE public.research_lab_attested_execution_results_v2 (
    receipt_hash TEXT PRIMARY KEY REFERENCES
        public.research_lab_attested_execution_receipts_v2(receipt_hash),
    role TEXT NOT NULL,
    operation TEXT NOT NULL,
    purpose TEXT NOT NULL,
    job_id TEXT NOT NULL,
    epoch_id BIGINT NOT NULL,
    sequence INTEGER NOT NULL,
    input_root TEXT NOT NULL,
    output_root TEXT NOT NULL,
    artifact_root TEXT NOT NULL,
    result_doc JSONB NOT NULL,
    artifact_hashes JSONB NOT NULL
);
CREATE FUNCTION public.prevent_research_lab_attested_v2_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'append-only relation';
END;
$$;
"""

    def install_source(frontier: dict, receipt_hash: str) -> str:
        row, receipt, source_state_hash = _source(
            frontier=frontier,
            receipt_hash=receipt_hash,
        )
        psql(
            "INSERT INTO public.research_lab_attested_execution_receipts_v2 "
            "SELECT * FROM jsonb_populate_record(NULL::"
            "public.research_lab_attested_execution_receipts_v2, %s);"
            % _literal(receipt)
        )
        psql(
            "INSERT INTO public.research_lab_attested_execution_results_v2 "
            "SELECT * FROM jsonb_populate_record(NULL::"
            "public.research_lab_attested_execution_results_v2, %s);"
            % _literal(row)
        )
        return source_state_hash

    def persist(
        frontier: dict,
        receipt_hash: str,
        source_state_hash: str,
        *,
        expect_success: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return psql(
            "SELECT public.%s(%s,%s,%s);"
            % (
                RPC,
                _literal(frontier),
                _text(receipt_hash),
                _text(source_state_hash),
            ),
            expect_success=expect_success,
        )

    first = _frontier(epoch=100, predecessor=None)
    second = _frontier(epoch=102, predecessor=first["frontier_hash"])
    late = _frontier(epoch=101, predecessor=first["frontier_hash"])
    fork = _frontier(epoch=103, predecessor=first["frontier_hash"])
    sources = [
        (first, _sha("1")),
        (second, _sha("2")),
        (late, _sha("3")),
        (fork, _sha("4")),
        (first, _sha("5")),
    ]

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
            ready = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "psql",
                    "-X",
                    "-A",
                    "-t",
                    "-U",
                    "postgres",
                    "-d",
                    "leadpoet",
                    "-c",
                    "SELECT 1",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            raise AssertionError("PostgreSQL 15 did not become ready")

        psql(setup_sql)
        psql(SQL)
        psql(SQL)
        null_frontier = psql(
            "SELECT public.%s(NULL,%s,%s);"
            % (RPC, _text(_sha("6")), _text(_sha("7"))),
            expect_success=False,
        )
        assert "allocation_settlement_frontier_request_invalid" in (
            null_frontier.stderr
        )
        null_mode = dict(first)
        null_mode["mode"] = None
        null_mode_result = persist(
            null_mode,
            _sha("6"),
            _sha("7"),
            expect_success=False,
        )
        assert "allocation_settlement_frontier_request_invalid" in (
            null_mode_result.stderr
        )
        null_checkpoint_count = dict(first)
        null_checkpoint_count["reward_checkpoint_count"] = None
        null_checkpoint_result = persist(
            null_checkpoint_count,
            _sha("6"),
            _sha("7"),
            expect_success=False,
        )
        assert "allocation_settlement_frontier_request_invalid" in (
            null_checkpoint_result.stderr
        )
        null_source_receipt = psql(
            "SELECT public.%s(%s,NULL,%s);"
            % (RPC, _literal(first), _text(_sha("7"))),
            expect_success=False,
        )
        assert "allocation_settlement_frontier_request_invalid" in (
            null_source_receipt.stderr
        )
        direct_write = psql(
            "SET ROLE service_role; "
            "INSERT INTO public.%s (netuid) VALUES (71);"
            % FRONTIER_TABLE,
            expect_success=False,
        )
        assert "permission denied" in direct_write.stderr
        hashes = {
            receipt: install_source(frontier, receipt)
            for frontier, receipt in sources
        }

        inserted = persist(first, _sha("1"), hashes[_sha("1")])
        assert '"status": "persisted"' in inserted.stdout
        replay = persist(first, _sha("1"), hashes[_sha("1")])
        assert '"status": "already_persisted"' in replay.stdout
        conflicting_receipt = persist(
            first,
            _sha("5"),
            hashes[_sha("5")],
            expect_success=False,
        )
        assert "allocation_settlement_frontier_conflict" in (
            conflicting_receipt.stderr
        )
        persist(second, _sha("2"), hashes[_sha("2")])

        out_of_order = persist(
            late,
            _sha("3"),
            hashes[_sha("3")],
            expect_success=False,
        )
        assert "allocation_settlement_frontier_successor_invalid" in (
            out_of_order.stderr
        )
        forked = persist(
            fork,
            _sha("4"),
            hashes[_sha("4")],
            expect_success=False,
        )
        assert "allocation_settlement_frontier_successor_invalid" in (
            forked.stderr
        )

        mutation = psql(
            "UPDATE public.%s SET settled_through_epoch = 1;"
            % FRONTIER_TABLE,
            expect_success=False,
        )
        assert "append-only relation" in mutation.stderr
        count = psql(
            "SELECT count(*) FROM public.%s;" % FRONTIER_TABLE
        )
        assert count.stdout.strip() == "2"
        psql(
            "ALTER TABLE public.%s DISABLE TRIGGER ALL; "
            "DELETE FROM public.%s; "
            "ALTER TABLE public.%s ENABLE TRIGGER ALL;"
            % (ACTIVATION_TABLE, ACTIVATION_TABLE, ACTIVATION_TABLE)
        )
        missing_activation = persist(
            first,
            _sha("1"),
            hashes[_sha("1")],
            expect_success=False,
        )
        assert "allocation_settlement_frontier_activation_invalid" in (
            missing_activation.stderr
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            capture_output=True,
            text=True,
            timeout=30,
        )
