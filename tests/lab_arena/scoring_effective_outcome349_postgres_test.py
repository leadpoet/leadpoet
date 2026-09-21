"""Real-PostgreSQL proof for effective scoring outcomes across retry namespaces."""

from __future__ import annotations

import json

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


MIGRATION = "349-lab-arena-scoring-effective-outcome.sql"
ROUND = "arena-2026-09-21-close349"
BASELINE = "baseline-close349"
MINER = "miner-close349"
BASELINE_HOTKEY = hotkey("close349-baseline")
MINER_HOTKEY = hotkey("close349-miner")


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + (MIGRATION,)
    )


def _seed(connection, *, retry_status: str) -> None:
    participants = [
        {"submission_id": BASELINE, "miner_hotkey": BASELINE_HOTKEY, "is_king": True},
        {"submission_id": MINER, "miner_hotkey": MINER_HOTKEY, "is_king": False},
    ]
    configuration = {"schema_version": "leadpoet.lab_arena.round_configuration.v1"}
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
            "participants) VALUES (%s,'stage1_scoring',4,4,%s::jsonb,%s::jsonb)",
            (ROUND, json.dumps(configuration), json.dumps(participants)),
        )
        cursor.executemany(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king) "
            "VALUES (%s,%s,%s,'frozen',%s)",
            [
                (BASELINE, ROUND, BASELINE_HOTKEY, True),
                (MINER, ROUND, MINER_HOTKEY, False),
            ],
        )
        cursor.executemany(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,stage_generation,scored_run_id,"
            "terminal_cause,output_ref) VALUES (%s,%s,%s,%s,%s,1,0,%s,'execute',"
            "'accepted',4,%s,'accepted',%s)",
            [
                (
                    f"{BASELINE}:execute",
                    f"{ROUND}:{BASELINE}:1:0",
                    ROUND,
                    BASELINE,
                    BASELINE_HOTKEY,
                    1,
                    None,
                    f"arena/{ROUND}/outputs/{BASELINE}.json",
                ),
                (
                    f"{MINER}:execute",
                    f"{ROUND}:{MINER}:1:0",
                    ROUND,
                    MINER,
                    MINER_HOTKEY,
                    1,
                    None,
                    f"arena/{ROUND}/outputs/{MINER}.json",
                ),
            ],
        )
        # The retry uses a new assignment namespace but judges the same
        # accepted execution. The old failure is retained as evidence.
        cursor.executemany(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,stage_generation,scored_run_id,"
            "terminal_cause,output_ref) VALUES (%s,%s,%s,%s,%s,1,0,%s,'score',"
            "%s,5,%s,%s,%s)",
            [
                (
                    f"{BASELINE}:score:old:2",
                    f"{ROUND}:{BASELINE}:1:0:score:old",
                    ROUND,
                    BASELINE,
                    BASELINE_HOTKEY,
                    2,
                    "failed",
                    f"{BASELINE}:execute",
                    "judge_error",
                    None,
                ),
                (
                    f"{BASELINE}:score:new:1",
                    f"{ROUND}:{BASELINE}:1:0:score:new",
                    ROUND,
                    BASELINE,
                    BASELINE_HOTKEY,
                    1,
                    retry_status,
                    f"{BASELINE}:execute",
                    "accepted" if retry_status == "accepted" else "judge_error",
                    "arena/score/accepted.json" if retry_status == "accepted" else None,
                ),
            ],
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _close(connection) -> dict:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena_close_scoring(%s::text,1::smallint)", (ROUND,)
        )
        result = cursor.fetchone()[0]
    connection.commit()
    return result


def test_accepted_retry_wins_over_old_failure_in_new_namespace(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection, retry_status="accepted")
        result = _close(connection)
        assert result == {
            "status": "closed",
            "round_status": "stage1_judged",
            "incomplete_assignments": 0,
            "stage_generation": 5,
        }
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,terminal_cause FROM public.lab_arena_runs "
                "WHERE run_id IN (%s,%s) ORDER BY run_id",
                (f"{BASELINE}:score:old:2", f"{BASELINE}:score:new:1"),
            )
            assert cursor.fetchall() == [("accepted", "accepted"), ("failed", "judge_error")]
            cursor.execute(
                "SELECT status,cancel_reason FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == ("stage1_judged", None)
    finally:
        connection.close()


def test_newest_retry_failure_still_cancels(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection, retry_status="failed")
        result = _close(connection)
        assert result["status"] == "cancelled"
        assert result["round_status"] == "cancelled"
        assert result["incomplete_assignments"] == 1
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,cancel_reason FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            status, reason = cursor.fetchone()
            assert status == "cancelled"
            assert reason == "scoring_incomplete:stage1:1"
    finally:
        connection.close()


def test_migration_is_narrow():
    body = open("scripts/" + MIGRATION, encoding="utf-8").read()
    assert "DELETE FROM" not in body
    assert "TRUNCATE " not in body
    assert "INSERT INTO public.lab_arena_" not in body
    assert "DROP TABLE" not in body
