"""Disposable PostgreSQL proof for zero-setup runner handoff."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import contracts
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration


ROOT = Path(__file__).parents[2]
MIGRATION_359 = ROOT / "scripts/359-lab-arena-setup-failure-model-retry.sql"
MIGRATION = ROOT / "scripts/360-lab-arena-zero-setup-runner-handoff.sql"
ROUND = "arena-2026-09-25-handoff360"
SUBMISSION = "baseline-2026-09-25-handoff360"
MINER = "5" + "A" * 47
RUNNER_A = "5" + "B" * 47
RUNNER_B = "5" + "C" * 47
ASSIGNMENT = f"{ROUND}:{SUBMISSION}:1:0"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + (MIGRATION_359.name, MIGRATION.name)
    )


def _result(terminal: str, wall: float, *, setup_provider: bool = False) -> dict:
    result = {
        "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
        "resource_summary": {
            "wall_seconds": wall,
            "cpu_seconds": 0 if wall == 0 else 1,
            "max_rss_bytes": 0 if wall == 0 else 1024,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "provider_call_count": 0,
        },
        "started_at": "2026-09-25T00:00:00Z",
        "finished_at": "2026-09-25T00:00:30Z",
        "terminal_status": terminal,
    }
    if setup_provider:
        result["failure_diagnostic"] = {
            "stage": "provider_call",
            "error_class": "provider_unavailable",
            "reason": "provider_error",
        }
    return result


def _seed(
    connection,
    prior: dict,
    cause: str,
    *,
    kind: str = "execute",
    prior_has_ledger: bool = False,
) -> None:
    configuration = base_round_configuration()
    configuration.update(
        round_id=ROUND,
        baseline_hotkey=MINER,
        runner_hotkeys=[RUNNER_A],
        runner_slot_ceiling=20,
        parallel_twenty_icp_execution=True,
    )
    configuration = contracts.validate_round_configuration(configuration)
    source_ref = f"arena/{ROUND}/sources/{SUBMISSION}.tar.gz"
    participants = [{
        "submission_id": SUBMISSION,
        "miner_hotkey": MINER,
        "is_king": True,
        "source_ref": source_ref,
    }]
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds("
            "round_id,status,status_generation,stage_generation,configuration_doc,"
            "rewards_enabled,participants,benchmark_ref) VALUES "
            "(%s,%s,1,1,%s::jsonb,FALSE,%s::jsonb,%s)",
            (
                ROUND,
                "stage1" if kind == "execute" else "stage1_scoring",
                json.dumps(configuration),
                json.dumps(participants),
                f"arena/{ROUND}/benchmark.json",
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
            "source_ref,source_size_bytes,consent,frozen_at) "
            "VALUES (%s,%s,%s,'frozen',TRUE,%s::jsonb,%s,100,%s::jsonb,"
            "'2026-09-25T00:00:00Z')",
            (
                SUBMISSION,
                ROUND,
                MINER,
                json.dumps({"source_ref": source_ref}),
                source_ref,
                json.dumps({"public_rerun": True}),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,lease_generation,stage_generation,"
            "runner_hotkey,terminal_cause,result_doc) VALUES "
            "(%s,%s,%s,%s,%s,1,0,1,%s,'failed',1,1,%s,%s,%s::jsonb)",
            (
                f"{ASSIGNMENT}:1",
                ASSIGNMENT,
                ROUND,
                SUBMISSION,
                MINER,
                kind,
                RUNNER_A,
                cause,
                json.dumps(prior),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,lease_generation,stage_generation,"
            "previous_runner_hotkey) VALUES "
            "(%s,%s,%s,%s,%s,1,0,2,%s,'pending',0,1,%s)",
            (
                f"{ASSIGNMENT}:2",
                ASSIGNMENT,
                ROUND,
                SUBMISSION,
                MINER,
                kind,
                RUNNER_A,
            ),
        )
        if prior_has_ledger:
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd) "
                "VALUES ('reservation',%s,%s,%s,%s,1,%s,'deepline',"
                "'deepline.execute','host',1)",
                (
                    MINER,
                    ROUND,
                    SUBMISSION,
                    f"{ASSIGNMENT}:1",
                    "sha256:" + "d" * 64,
                ),
            )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _claim(connection, runner: str, suffix: str) -> dict:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena_claim_assignment("
            "%s,%s,20,20,%s,%s,%s,%s,420)",
            (
                ROUND,
                runner,
                [],
                suffix * 32,
                "sha256:" + suffix * 64,
                "sha256:" + suffix * 64,
            ),
        )
        response = cursor.fetchone()[0]
    connection.commit()
    return response


@pytest.mark.parametrize(
    ("cause", "prior"),
    (
        ("model_error", _result("model_error", 0)),
        ("provider_error", _result("provider_error", 0, setup_provider=True)),
    ),
)
def test_same_runner_cannot_consume_retry_after_exact_zero_setup_failure(
    database, cause, prior
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection, prior, cause)
        assert _claim(connection, RUNNER_A, "1") == {"status": "no_pending"}
        claim = _claim(connection, RUNNER_B, "2")
        assert claim["status"] == "leased"
        assert claim["run_id"] == f"{ASSIGNMENT}:2"
    finally:
        connection.close()


def test_real_model_failure_keeps_lone_runner_fallback(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection, _result("model_error", 10), "model_error")
        claim = _claim(connection, RUNNER_A, "3")
        assert claim["status"] == "leased"
        assert claim["run_id"] == f"{ASSIGNMENT}:2"
    finally:
        connection.close()


def test_source_owned_zero_failure_stays_bounded_model_error(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection, _result("model_error", 0), "model_error")
        claim = _claim(connection, RUNNER_B, "6")
        assert claim["status"] == "leased"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)",
                (
                    claim["run_id"],
                    "sha256:" + "6" * 64,
                    json.dumps(_result("model_error", 0)),
                    "model_error",
                    "",
                ),
            )
            completed = cursor.fetchone()[0]
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE assignment_id=%s AND attempt>2",
                (ASSIGNMENT,),
            )
            later_attempts = cursor.fetchone()[0]
        connection.commit()
        assert completed["status"] == "failed"
        assert "confirmation_attempt" not in completed
        assert later_attempts == 0
    finally:
        connection.close()


def test_zero_result_with_accounting_keeps_lone_runner_fallback(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(
            connection,
            _result("model_error", 0),
            "model_error",
            prior_has_ledger=True,
        )
        claim = _claim(connection, RUNNER_A, "5")
        assert claim["status"] == "leased"
        assert claim["run_id"] == f"{ASSIGNMENT}:2"
    finally:
        connection.close()


def test_score_retry_is_unchanged(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection, _result("model_error", 0), "model_error", kind="score")
        claim = _claim(connection, RUNNER_A, "4")
        assert claim["status"] == "leased"
        assert claim["kind"] == "score"
    finally:
        connection.close()


def test_migration_is_idempotent_and_preserves_claim_security(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        query = (
            "SELECT pg_get_functiondef(procedure.oid),owner.rolname,procedure.proacl,"
            "procedure.prosecdef,procedure.provolatile,procedure.proconfig "
            "FROM pg_proc AS procedure JOIN pg_namespace AS namespace "
            "ON namespace.oid=procedure.pronamespace JOIN pg_roles AS owner "
            "ON owner.oid=procedure.proowner WHERE namespace.nspname='public' "
            "AND procedure.proname='lab_arena_claim_assignment' "
            "AND procedure.pronargs=9"
        )
        with connection.cursor() as cursor:
            cursor.execute(query)
            before = cursor.fetchone()
            assert "lab_arena_zero_setup_runner_handoff_v1" in before[0]
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(query)
            after = cursor.fetchone()
        assert after == before
        assert after[1] == "lab_arena_owner"
        assert after[3:] == (True, "v", ["search_path=pg_catalog, public"])
    finally:
        connection.close()
