"""Migration 307 against the exact post-306, partially executed Sep19 state."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena import sep19_preexecution_source_quota306_postgres_test as migration306
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/307-arena-2026-09-19-preexecution-source-activation.sql"
ROUND19 = migration306.ROUND19
ROUND20 = migration306.ROUND20
BASELINE = migration306.BASELINE
OLD_REF = migration306.NEW_REF
NEW_REF = f"arena/{ROUND19}/sources/{BASELINE}-preexecution307.tar.gz"
NEW_SIZE = 678_209
NEW_SHA = "b3624778756db5460e456fd054c3aae28d5b8e51f7b9213a2f60bf19a25a4196"
NEW_COMMIT = "f3fd3acc10bc58a95c0728c61fe0fc044cf57303"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def execute(conn):
    with conn.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    conn.commit()


def rows(cursor, table, where, order):
    cursor.execute(
        f"SELECT COALESCE(jsonb_agg(to_jsonb(r) ORDER BY {order}),'[]'::jsonb) "
        f"FROM public.{table} r WHERE {where}"
    )
    return cursor.fetchone()[0]


def snapshot(cursor):
    return {
        "runs": rows(
            cursor,
            "lab_arena_runs",
            "round_id='arena-2026-09-19'",
            "run_id",
        ),
        "ledger": rows(
            cursor,
            "lab_arena_ledger",
            "round_id='arena-2026-09-19'",
            "entry_id",
        ),
        "miners19": rows(
            cursor,
            "lab_arena_submissions",
            "round_id='arena-2026-09-19' "
            "AND submission_id<>'baseline-2026-09-19'",
            "submission_id",
        ),
        "submissions20": rows(
            cursor,
            "lab_arena_submissions",
            "round_id='arena-2026-09-20'",
            "submission_id",
        ),
    }


def seed_post306_with_six_accepted(conn):
    migration306.seed(conn)
    migration306.execute(conn)
    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            """
            WITH chosen AS (
              SELECT run_id, row_number() OVER (ORDER BY run_id) AS ordinal
              FROM public.lab_arena_runs
              WHERE round_id=%s AND submission_id<>%s
              ORDER BY run_id
              LIMIT 6
            )
            UPDATE public.lab_arena_runs AS run
            SET status='accepted', runner_hotkey=run.miner_hotkey,
                lease_token_hash='sha256:'||repeat('a',64), lease_generation=1,
                lease_expires_at=clock_timestamp()-interval '1 minute',
                claim_request_id='sep19-accepted-'||chosen.ordinal,
                claim_request_hash='sha256:'||repeat('b',64),
                claim_response='{"status":"ok"}'::jsonb,
                result_doc='{"schema_version":"test.accepted.v1"}'::jsonb,
                output_ref='arena/test/accepted-'||chosen.ordinal||'.json',
                terminal_cause='accepted', terminal_doc='{}'::jsonb,
                participation_accepted_at=clock_timestamp()
            FROM chosen WHERE run.run_id=chosen.run_id
            """,
            (ROUND19, BASELINE),
        )
        assert cursor.rowcount == 6
        cursor.execute("SET session_replication_role=origin")
    conn.commit()


def test_source_only_activation_preserves_six_accepted_runs_and_all_other_state(database):
    psycopg2, dsn = database
    conn = psycopg2.connect(**dsn)
    try:
        seed_post306_with_six_accepted(conn)
        with conn.cursor() as cursor:
            protected_before = snapshot(cursor)
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND19,),
            )
            round19_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND20,),
            )
            round20_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT to_jsonb(s) FROM public.lab_arena_submissions s "
                "WHERE submission_id=%s",
                (BASELINE,),
            )
            baseline_before = cursor.fetchone()[0]

        execute(conn)

        with conn.cursor() as cursor:
            assert snapshot(cursor) == protected_before
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND19,),
            )
            round19_after = cursor.fetchone()[0]
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND20,),
            )
            round20_after = cursor.fetchone()[0]
            cursor.execute(
                "SELECT to_jsonb(s) FROM public.lab_arena_submissions s "
                "WHERE submission_id=%s",
                (BASELINE,),
            )
            baseline_after = cursor.fetchone()[0]
            cursor.execute(
                "SELECT status,count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s GROUP BY status ORDER BY status",
                (ROUND19,),
            )
            assert cursor.fetchall() == [("accepted", 6), ("pending", 94)]
            cursor.execute(
                "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgname IN "
                "('lab_arena_rounds_write_once','lab_arena_submissions_frozen') "
                "ORDER BY tgname"
            )
            assert [row[0] for row in cursor.fetchall()] == ["O", "O"]

        assert round19_after["configuration_doc"] == round19_before["configuration_doc"]
        assert round20_after == round20_before
        assert {
            key: value
            for key, value in round19_after.items()
            if key not in {"participants", "updated_at"}
        } == {
            key: value
            for key, value in round19_before.items()
            if key not in {"participants", "updated_at"}
        }
        assert {
            key: value
            for key, value in baseline_after.items()
            if key not in {"source_ref", "source_size_bytes", "submission_doc", "updated_at"}
        } == {
            key: value
            for key, value in baseline_before.items()
            if key not in {"source_ref", "source_size_bytes", "submission_doc", "updated_at"}
        }
        assert baseline_after["source_ref"] == NEW_REF
        assert baseline_after["source_size_bytes"] == NEW_SIZE
        assert baseline_after["submission_doc"]["source_sha256"] == NEW_SHA
        assert baseline_after["submission_doc"]["source_commit"] == NEW_COMMIT
        override = baseline_after["submission_doc"]["preexecution_source_override"]
        assert override["previous_source_ref"] == OLD_REF
        assert override["previous_source_sha256"] == migration306.NEW_SHA

        with conn.cursor() as cursor:
            replay_before = {
                "protected": snapshot(cursor),
                "rounds": rows(cursor, "lab_arena_rounds", "round_id IN "
                               "('arena-2026-09-19','arena-2026-09-20')", "round_id"),
                "baseline": rows(
                    cursor,
                    "lab_arena_submissions",
                    "submission_id='baseline-2026-09-19'",
                    "submission_id",
                ),
            }
        execute(conn)
        with conn.cursor() as cursor:
            assert {
                "protected": snapshot(cursor),
                "rounds": rows(cursor, "lab_arena_rounds", "round_id IN "
                               "('arena-2026-09-19','arena-2026-09-20')", "round_id"),
                "baseline": rows(
                    cursor,
                    "lab_arena_submissions",
                    "submission_id='baseline-2026-09-19'",
                    "submission_id",
                ),
            } == replay_before
    finally:
        conn.close()


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            "UPDATE public.lab_arena_runs SET status='leased', "
            "runner_hotkey=miner_hotkey,lease_generation=1,"
            "lease_token_hash='sha256:'||repeat('c',64),"
            "lease_expires_at=clock_timestamp()+interval '1 hour',"
            "claim_request_id='baseline-claimed',"
            "claim_request_hash='sha256:'||repeat('d',64) "
            "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
            "WHERE round_id='arena-2026-09-19' "
            "AND submission_id='baseline-2026-09-19')",
            "baseline execution was claimed",
        ),
        (
            "INSERT INTO public.lab_arena_ledger"
            "(entry_kind,miner_hotkey,round_id,submission_id,run_id,call_identity,"
            "provider,operation_id,funding_source,amount_microusd) "
            "SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,"
            "'sha256:'||repeat('e',64),'openrouter','openrouter.code_review','host',1 "
            "FROM public.lab_arena_runs WHERE round_id='arena-2026-09-19' "
            "AND submission_id='baseline-2026-09-19' LIMIT 1",
            "baseline or scoring ledger differs",
        ),
        (
            "INSERT INTO public.lab_arena_ledger"
            "(entry_kind,miner_hotkey,round_id,submission_id,call_identity,"
            "provider,operation_id,funding_source,amount_microusd) "
            "SELECT 'reservation',miner_hotkey,round_id,submission_id,"
            "'sha256:'||repeat('f',64),'openrouter','openrouter.score','host',1 "
            "FROM public.lab_arena_runs WHERE round_id='arena-2026-09-19' "
            "AND submission_id<>'baseline-2026-09-19' LIMIT 1",
            "baseline or scoring ledger differs",
        ),
        (
            "UPDATE public.lab_arena_runs SET status='leased',"
            "runner_hotkey=miner_hotkey,lease_generation=1,"
            "lease_token_hash='sha256:'||repeat('1',64),"
            "lease_expires_at=clock_timestamp()+interval '1 hour',"
            "claim_request_id='miner-active',"
            "claim_request_hash='sha256:'||repeat('2',64) "
            "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
            "WHERE round_id='arena-2026-09-19' AND status='pending' "
            "AND submission_id<>'baseline-2026-09-19')",
            "execution state or run plan differs",
        ),
        (
            "UPDATE public.lab_arena_rounds SET publication_doc='{}'::jsonb "
            "WHERE round_id='arena-2026-09-19'",
            "preexecution round state differs",
        ),
    ],
)
def test_activation_rejects_claimed_baseline_ledger_scoring_or_active_state(
    database, mutation, error
):
    psycopg2, dsn = database
    conn = psycopg2.connect(**dsn)
    try:
        seed_post306_with_six_accepted(conn)
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(mutation)
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        with pytest.raises(Exception, match=error):
            execute(conn)
        conn.rollback()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT source_ref FROM public.lab_arena_submissions "
                "WHERE submission_id=%s",
                (BASELINE,),
            )
            assert cursor.fetchone()[0] == OLD_REF
    finally:
        conn.close()
