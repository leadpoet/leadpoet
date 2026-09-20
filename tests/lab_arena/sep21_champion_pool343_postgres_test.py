"""Real-PostgreSQL proof for the bounded Sep21 champion pool transition."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lab_arena import contracts, rewards
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/343-arena-2026-09-21-champion-pool-30.sql"
ROUND = "arena-2026-09-21"
CONTROL = "arena-2026-09-20"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration(round_id: str, pool_percent: int = 25) -> dict:
    return {
        "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": round_id,
        "mode": "live",
        "rewards_enabled": True,
        "reward_constants": rewards.reward_constants_document(pool_percent),
        "unrelated_rule": {"preserve": ["exactly", 7]},
    }


def _seed(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds"
            "(round_id,status,configuration_doc,rewards_enabled) "
            "VALUES (%s,'open',%s::jsonb,TRUE),(%s,'published',%s::jsonb,TRUE)",
            (
                ROUND,
                json.dumps(_configuration(ROUND)),
                CONTROL,
                json.dumps(_configuration(CONTROL)),
            ),
        )
        for index, status in enumerate(("accepted", "uploading")):
            submission_id = f"sep21-miner-{index}"
            source_ref = f"arena/{ROUND}/sources/{submission_id}.tar.gz"
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions"
                "(submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
                "source_size_bytes,submission_doc) "
                "VALUES (%s,%s,%s,%s,FALSE,%s,1000,%s::jsonb)",
                (
                    submission_id,
                    ROUND,
                    hotkey(submission_id),
                    status,
                    source_ref,
                    json.dumps(
                        {
                            "source_ref": source_ref,
                            "source_size_bytes": 1000,
                            "consent": {"public_rerun": True},
                        }
                    ),
                ),
            )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _row(cursor, round_id: str) -> dict:
    cursor.execute(
        "SELECT to_jsonb(round_row) FROM public.lab_arena_rounds round_row "
        "WHERE round_id=%s",
        (round_id,),
    )
    return cursor.fetchone()[0]


def _submissions(cursor) -> list[dict]:
    cursor.execute(
        "SELECT COALESCE(jsonb_agg(to_jsonb(submission) ORDER BY submission_id),"
        "'[]'::jsonb) FROM public.lab_arena_submissions submission "
        "WHERE round_id=%s",
        (ROUND,),
    )
    return cursor.fetchone()[0]


def _without_pool_and_updated_at(row: dict) -> dict:
    normalized = copy.deepcopy(row)
    del normalized["updated_at"]
    del normalized["configuration_doc"]["reward_constants"]["pool_percent"]
    return normalized


def _execute(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.commit()


def test_open_unstarted_round_moves_to_30_and_replay_is_exact(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection)
        with connection.cursor() as cursor:
            before = _row(cursor, ROUND)
            control_before = _row(cursor, CONTROL)
            submissions_before = _submissions(cursor)

        _execute(connection)

        with connection.cursor() as cursor:
            after = _row(cursor, ROUND)
            assert _row(cursor, CONTROL) == control_before
            assert _submissions(cursor) == submissions_before
            cursor.execute(
                "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
                "'public.lab_arena_rounds'::regclass AND "
                "tgname='lab_arena_rounds_write_once'"
            )
            assert cursor.fetchone()[0] == "O"
        assert after["configuration_doc"]["reward_constants"] == (
            rewards.reward_constants_document(30)
        )
        assert _without_pool_and_updated_at(after) == _without_pool_and_updated_at(
            before
        )
        assert after["updated_at"] != before["updated_at"]

        _execute(connection)
        with connection.cursor() as cursor:
            assert _row(cursor, ROUND) == after
            assert _submissions(cursor) == submissions_before
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            "UPDATE public.lab_arena_rounds SET status='committed',"
            "configuration_doc=jsonb_set(configuration_doc,"
            "'{reward_constants,pool_percent}','30'::jsonb,FALSE) "
            f"WHERE round_id='{ROUND}'",
            "not open and unstarted",
        ),
        (
            "INSERT INTO public.lab_arena_runs"
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status) SELECT 'sep21-run','sep21-run',"
            "round_id,submission_id,miner_hotkey,1,0,1,'execute','pending' "
            "FROM public.lab_arena_submissions "
            f"WHERE round_id='{ROUND}' LIMIT 1",
            "execution or scoring plan already exists",
        ),
        (
            "UPDATE public.lab_arena_rounds SET benchmark_ref="
            f"'arena/{ROUND}/benchmark.json' WHERE round_id='{ROUND}'",
            "not open and unstarted",
        ),
        (
            "UPDATE public.lab_arena_submissions SET status='frozen',"
            "frozen_at=clock_timestamp() WHERE submission_id='sep21-miner-0'",
            "baseline or submission source is already frozen",
        ),
        (
            "UPDATE public.lab_arena_rounds SET reward_basis_hash="
            "'sha256:'||repeat('a',64) "
            f"WHERE round_id='{ROUND}'",
            "not open and unstarted",
        ),
        (
            "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
            "configuration_doc,'{reward_constants,epochs_per_reward_week}',"
            f"'141'::jsonb,FALSE) WHERE round_id='{ROUND}'",
            "reward constants differ",
        ),
    ),
)
def test_transition_rejects_changed_or_started_round(database, mutation, error):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(mutation)
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        with connection.cursor() as cursor:
            before = _row(cursor, ROUND)
            submissions_before = _submissions(cursor)

        with pytest.raises(psycopg2.Error, match=error):
            _execute(connection)
        connection.rollback()

        with connection.cursor() as cursor:
            assert _row(cursor, ROUND) == before
            assert _submissions(cursor) == submissions_before
            cursor.execute(
                "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
                "'public.lab_arena_rounds'::regclass AND "
                "tgname='lab_arena_rounds_write_once'"
            )
            assert cursor.fetchone()[0] == "O"
    finally:
        connection.close()


def test_migration_is_numbered_and_disables_only_the_round_write_once_trigger():
    body = MIGRATION.read_text(encoding="utf-8")
    assert MIGRATION.name.startswith("343-")
    assert MIGRATION.name not in CURRENT_SERVICE_MIGRATIONS
    assert "DISABLE TRIGGER lab_arena_rounds_write_once" in body
    assert "ENABLE TRIGGER lab_arena_rounds_write_once" in body
    assert "DISABLE TRIGGER USER" not in body
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
