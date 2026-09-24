"""Disposable PostgreSQL proof for the narrow setup-failure model retry."""

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
MIGRATION = ROOT / "scripts/359-lab-arena-setup-failure-model-retry.sql"
ROUND = "arena-2026-09-25-retry359"
SUBMISSION = "baseline-2026-09-25-retry359"
MINER = "5" + "A" * 47
RUNNER = "5" + "B" * 47
ASSIGNMENT = f"{ROUND}:{SUBMISSION}:1:0"
LEASE_HASH = "sha256:" + "c" * 64


@pytest.fixture(scope="module")
def database():
    assert "227-lab-arena-champion-funding.sql" in CURRENT_SERVICE_MIGRATIONS
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + (MIGRATION.name,)
    )


def _result(terminal: str, wall_seconds: float, *, setup: bool = False) -> dict:
    result = {
        "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
        "resource_summary": {
            "wall_seconds": wall_seconds,
            "cpu_seconds": 0 if wall_seconds == 0 else 1.0,
            "max_rss_bytes": 0 if wall_seconds == 0 else 1024,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "provider_call_count": 0,
        },
        "started_at": "2026-09-25T00:00:00Z",
        "finished_at": "2026-09-25T00:00:30Z",
        "terminal_status": terminal,
    }
    if setup:
        result["failure_diagnostic"] = {
            "stage": "provider_call",
            "error_class": "provider_unavailable",
            "reason": "provider_error",
        }
    return result


def _seed(
    connection,
    *,
    prior_result: dict,
    prior_cause: str,
    current_attempt: int = 2,
    current_kind: str = "execute",
) -> str:
    configuration = base_round_configuration()
    configuration.update(
        round_id=ROUND,
        baseline_hotkey=MINER,
        runner_hotkeys=[RUNNER],
        runner_slot_ceiling=1,
    )
    configuration = contracts.validate_round_configuration(configuration)
    participants = [{
        "submission_id": SUBMISSION,
        "miner_hotkey": MINER,
        "is_king": True,
        "source_ref": f"arena/{ROUND}/sources/{SUBMISSION}.tar.gz",
    }]
    current_run = f"{ASSIGNMENT}:{current_attempt}"
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
                "stage1_scoring" if current_kind == "score" else "stage1",
                json.dumps(configuration),
                json.dumps(participants),
                f"arena/{ROUND}/benchmark.json",
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
            "source_ref,source_size_bytes,consent,frozen_at) VALUES "
            "(%s,%s,%s,'frozen',TRUE,%s::jsonb,%s,100,%s::jsonb,"
            "'2026-09-25T00:00:00Z'::timestamptz)",
            (
                SUBMISSION,
                ROUND,
                MINER,
                json.dumps({"source_ref": participants[0]["source_ref"]}),
                participants[0]["source_ref"],
                json.dumps({"public_rerun": True}),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,lease_generation,stage_generation,"
            "terminal_cause,result_doc) VALUES "
            "(%s,%s,%s,%s,%s,1,0,1,%s,'failed',1,1,%s,%s::jsonb)",
            (
                f"{ASSIGNMENT}:1",
                ASSIGNMENT,
                ROUND,
                SUBMISSION,
                MINER,
                current_kind,
                prior_cause,
                json.dumps(prior_result),
            ),
        )
        if current_attempt == 3:
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status,lease_generation,stage_generation,"
                "terminal_cause,result_doc) VALUES "
                "(%s,%s,%s,%s,%s,1,0,2,%s,'failed',1,1,'model_error',%s::jsonb)",
                (
                    f"{ASSIGNMENT}:2",
                    ASSIGNMENT,
                    ROUND,
                    SUBMISSION,
                    MINER,
                    current_kind,
                    json.dumps(_result("model_error", 30)),
                ),
            )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,lease_generation,stage_generation,"
            "runner_hotkey,lease_token_hash,lease_expires_at) VALUES "
            "(%s,%s,%s,%s,%s,1,0,%s,%s,'leased',1,1,%s,%s,"
            "clock_timestamp()+interval '10 minutes')",
            (
                current_run,
                ASSIGNMENT,
                ROUND,
                SUBMISSION,
                MINER,
                current_attempt,
                current_kind,
                RUNNER,
                LEASE_HASH,
            ),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()
    return current_run


def _complete(
    connection,
    run_id: str,
    result: dict,
    terminal: str,
    *,
    lease_hash: str = LEASE_HASH,
) -> dict:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)",
            (
                run_id,
                lease_hash,
                json.dumps(result),
                terminal,
                f"arena/{ROUND}/outputs/retry359.json" if terminal == "accepted" else "",
            ),
        )
        response = cursor.fetchone()[0]
    connection.commit()
    return response


def test_setup_failure_then_real_model_failure_gets_one_claimable_retry(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        attempt2 = _seed(
            connection,
            prior_result=_result("provider_error", 0, setup=True),
            prior_cause="provider_error",
        )
        model_failure = _result("model_error", 30)
        completed = _complete(connection, attempt2, model_failure, "model_error")
        assert completed == {
            "status": "failed",
            "idempotent": False,
            "run_id": attempt2,
            "attempt": 2,
            "confirmation_attempt": 3,
        }
        assert _complete(connection, attempt2, model_failure, "model_error") == {
            "status": "failed",
            "idempotent": True,
            "run_id": attempt2,
            "attempt": 2,
        }
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),min(status),min(previous_runner_hotkey) "
                "FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=3",
                (ASSIGNMENT,),
            )
            assert cursor.fetchone() == (1, "pending", RUNNER)
            cursor.execute(
                "SELECT public.lab_arena_claim_assignment(%s,%s,1,1,%s,%s,%s,%s,420)",
                (
                    ROUND,
                    RUNNER,
                    [],
                    "3" * 32,
                    "sha256:" + "4" * 64,
                    "sha256:" + "5" * 64,
                ),
            )
            claim = cursor.fetchone()[0]
        connection.commit()
        assert claim["status"] == "leased"
        assert claim["run_id"] == f"{ASSIGNMENT}:3"
        assert claim["attempt"] == 3
        accepted = _complete(
            connection,
            claim["run_id"],
            _result("accepted", 25),
            "accepted",
            lease_hash="sha256:" + "5" * 64,
        )
        assert accepted["status"] == "accepted"
        assert _complete(
            connection,
            claim["run_id"],
            _result("accepted", 25),
            "accepted",
            lease_hash="sha256:" + "5" * 64,
        )["idempotent"] is True
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("name", "prior_result", "prior_cause", "attempt", "kind", "result", "cause"),
    (
        (
            "two_model_failures",
            _result("model_error", 20),
            "model_error", 2, "execute", _result("model_error", 30), "model_error",
        ),
        (
            "two_setup_failures",
            _result("provider_error", 0, setup=True),
            "provider_error", 2, "execute",
            _result("provider_error", 0, setup=True), "provider_error",
        ),
        (
            "bad_requirements",
            _result("provider_error", 0, setup=True),
            "provider_error", 2, "execute", _result("model_error", 0), "model_error",
        ),
        (
            "budget",
            _result("provider_error", 0, setup=True),
            "provider_error", 2, "execute",
            _result("budget_exhausted", 30), "budget_exhausted",
        ),
        (
            "credentials",
            _result("provider_error", 0, setup=True),
            "provider_error", 2, "execute",
            _result("credential_error", 30), "credential_error",
        ),
        (
            "score_job",
            _result("provider_error", 0, setup=True),
            "provider_error", 2, "score", _result("judge_error", 30), "judge_error",
        ),
        (
            "later_attempt",
            _result("provider_error", 0, setup=True),
            "provider_error", 3, "execute", _result("model_error", 30), "model_error",
        ),
    ),
)
def test_retry_controls_do_not_create_an_extra_attempt(
    database, name, prior_result, prior_cause, attempt, kind, result, cause
):
    del name
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        run_id = _seed(
            connection,
            prior_result=prior_result,
            prior_cause=prior_cause,
            current_attempt=attempt,
            current_kind=kind,
        )
        completed = _complete(connection, run_id, result, cause)
        assert completed["status"] == "failed"
        assert "confirmation_attempt" not in completed
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE assignment_id=%s AND attempt>%s",
                (ASSIGNMENT, attempt),
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()


def test_migration_is_idempotent_and_preserves_function_security(database):
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
            "AND procedure.proname='lab_arena_complete_attempt' "
            "AND procedure.pronargs=5"
        )
        with connection.cursor() as cursor:
            cursor.execute(query)
            before = cursor.fetchone()
            assert "lab_arena_setup_failure_model_retry_v1" in before[0]
            assert "v_run.attempt < LEAST(5, 2 +" in before[0]
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(query)
            after = cursor.fetchone()
        assert after == before
        assert after[1] == "lab_arena_owner"
        assert after[3:] == (True, "v", ["search_path=pg_catalog, public"])
    finally:
        connection.close()
