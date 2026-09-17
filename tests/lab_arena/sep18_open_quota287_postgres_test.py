"""Disposable-PostgreSQL transition proof for migration 287."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena import scoring
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/287-arena-2026-09-18-openrouter-quota.sql"
ROUND = "arena-2026-09-18"
HISTORY = "arena-2026-09-17-history"
BASELINE_HOTKEY = "5" + "A" * 47
MINER_HOTKEY = "5" + "B" * 47


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration(round_id: str = ROUND) -> dict:
    return {
        "schema_version": "leadpoet.lab_arena.round_configuration.v1",
        "round_id": round_id,
        "mode": "live",
        "network_name": "finney",
        "netuid": 71,
        "rewards_enabled": True,
        "schedule": {
            "submission_open": "2026-09-17T00:00:00Z",
            "submission_cutoff": "2026-09-18T00:00:00Z",
            "benchmark_deadline": "2026-09-18T00:30:00Z",
            "stage_1_start": "2026-09-18T00:30:01Z",
            "stage_1_close": "2026-09-18T04:30:01Z",
            "stage_1_scoring_close": "2026-09-18T11:00:01Z",
            "stage_2_start": "2026-09-18T11:00:02Z",
            "stage_2_close": "2026-09-18T14:00:02Z",
            "final_scoring_close": "2026-09-18T20:30:02Z",
            "publication_deadline": "2026-09-18T20:30:03Z",
        },
        "stage_1_icp_count": 10,
        "stage_2_icp_count": 10,
        "finalist_count": 10,
        "max_challengers": 20,
        "runner_slot_ceiling": 20,
        "max_attempts_per_assignment": 2,
        "lease_ttl_seconds": 3600,
        "companies_per_icp": 5,
        "providers": ["scrapingdog", "deepline", "openrouter"],
        "call_quotas": {"scrapingdog": 30, "deepline": 30, "openrouter": 60},
        "scoring_call_quotas": {
            "scrapingdog": 150,
            "deepline": 40,
            "openrouter": 120,
        },
        "icp_wall_clock_seconds": 2700,
        "scoring_wall_clock_seconds": 900,
        "execution_cap_microusd": 80_000_000,
        "cost_per_company_microusd": 800_000,
        "sourcing_cost_eligibility_policy": "successful_calls_v1",
        "scoring_cap_microusd": 50_000_000,
        "baseline_hotkey": BASELINE_HOTKEY,
        "baseline_source_url": (
            "https://github.com/leadpoet/champion_model/archive/refs/heads/lab.tar.gz"
        ),
        "runner_hotkeys": [BASELINE_HOTKEY],
        "banned_hotkeys": [],
        "parallel_twenty_icp_execution": True,
        "checkpoint_deadline_policy": "atomic_checkpoint_45m_v1",
        "integrity_policy": "arena_integrity_v1",
        "contact_policy": "contacts_v1",
        "intent_details_policy": "intent_details_v1",
        "benchmark_disclosure_policy": "after_scoring_day2_v1",
        "scorer_policy": scoring.build_scorer_policy(
            scoring_adapter_version="qualification_contacts_v3",
            intent_details=True,
        ),
        "reward_constants": {"pool_percent": 25},
    }


def _migration_sql() -> str:
    sql = MIGRATION.read_text(encoding="utf-8")
    # Migration 278 established this repository convention for time-bounded
    # historical SQL. This is a SQL-projected time fixture for the durable
    # positive matrix; the separate verbatim test executes committed bytes.
    return sql.replace(
        "pg_catalog.clock_timestamp() >= v_cutoff",
        "'2026-09-17T23:00:00Z'::TIMESTAMPTZ >= v_cutoff",
    )


def _seed(connection) -> None:
    target = _configuration()
    historical = copy.deepcopy(target)
    historical["round_id"] = HISTORY
    historical["schedule"]["submission_open"] = "2026-09-16T00:00:00Z"
    historical["schedule"]["submission_cutoff"] = "2026-09-17T00:00:00Z"
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
            "rewards_enabled) VALUES (%s,'open',0,0,%s::jsonb,TRUE),"
            "(%s,'published',22,18,%s::jsonb,TRUE)",
            (ROUND, json.dumps(target), HISTORY, json.dumps(historical)),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king,"
            "source_ref,source_size_bytes,consent,submission_doc,"
            "code_review_status,code_review_doc,code_review_claim,"
            "code_review_started_at,code_review_attempts) VALUES "
            "('sep18-miner',%s,%s,'accepted',FALSE,"
            "'arena/arena-2026-09-18/sources/sep18-miner.tar.gz',123,"
            "'{\"public_rerun\":true}'::jsonb,"
            "jsonb_build_object('preserve',true,'source_sha256',%s,'source_commit',%s),"
            "'passed','{\"decision\":\"pass\"}'::jsonb,"
            "%s,'2026-09-17T20:00:00Z',1),"
            "('history-miner',%s,%s,'accepted',FALSE,"
            "'arena/arena-2026-09-17/sources/history-miner.tar.gz',456,"
            "'{\"public_rerun\":true}'::jsonb,"
            "jsonb_build_object('historical',true,'source_sha256',%s,'source_commit',%s),"
            "'passed',"
            "'{\"decision\":\"pass\",\"historical\":true}'::jsonb,"
            "%s,'2026-09-16T20:00:00Z',1)",
            (
                ROUND,
                MINER_HOTKEY,
                "a" * 64,
                "b" * 40,
                "sha256:" + "c" * 64,
                HISTORY,
                MINER_HOTKEY,
                "d" * 64,
                "e" * 40,
                "sha256:" + "f" * 64,
            ),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _snapshot(cursor, round_id: str) -> dict:
    cursor.execute(
        "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
        (round_id,),
    )
    row = cursor.fetchone()
    round_doc = row[0] if row is not None else None
    cursor.execute(
        "SELECT coalesce(jsonb_agg(to_jsonb(s) ORDER BY submission_id),'[]'::jsonb) "
        "FROM public.lab_arena_submissions s WHERE round_id=%s",
        (round_id,),
    )
    return {"round": round_doc, "submissions": cursor.fetchone()[0]}


def _execute(connection, sql: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(sql)
    connection.commit()


def _expect_failure(connection, mutation: str, message: str) -> None:
    _seed(connection)
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(mutation)
        cursor.execute("SET session_replication_role=origin")
    connection.commit()
    with connection.cursor() as cursor:
        before_target = _snapshot(cursor, ROUND)
        before_history = _snapshot(cursor, HISTORY)
    with pytest.raises(Exception, match=message):
        _execute(connection, _migration_sql())
    connection.rollback()
    with connection.cursor() as cursor:
        assert _snapshot(cursor, ROUND) == before_target
        assert _snapshot(cursor, HISTORY) == before_history
        cursor.execute(
            "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
            "'public.lab_arena_rounds'::regclass AND "
            "tgname='lab_arena_rounds_write_once'"
        )
        assert cursor.fetchone()[0] == "O"


def test_migration_287_exact_transition_replay_and_fail_closed_matrix(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        sql = _migration_sql()
        committed = MIGRATION.read_text(encoding="utf-8")
        assert "pg_catalog.clock_timestamp() >= v_cutoff" in committed
        assert "2026-09-18T00:00:00Z" in committed

        _seed(connection)
        with connection.cursor() as cursor:
            before_target = _snapshot(cursor, ROUND)
            before_history = _snapshot(cursor, HISTORY)
        _execute(connection, sql)
        with connection.cursor() as cursor:
            after_target = _snapshot(cursor, ROUND)
            after_history = _snapshot(cursor, HISTORY)
            cursor.execute(
                "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
                "'public.lab_arena_rounds'::regclass AND "
                "tgname='lab_arena_rounds_write_once'"
            )
            assert cursor.fetchone()[0] == "O"
        assert after_history == before_history
        assert after_target["submissions"] == before_target["submissions"]
        expected = copy.deepcopy(before_target["round"])
        expected["configuration_doc"]["call_quotas"]["openrouter"] = 200
        assert after_target["round"] == expected

        _execute(connection, sql)
        with connection.cursor() as cursor:
            assert _snapshot(cursor, ROUND) == after_target
            assert _snapshot(cursor, HISTORY) == after_history

        _seed(connection)
        after_cutoff_sql = committed.replace(
            "pg_catalog.clock_timestamp() >= v_cutoff",
            "'2026-09-18T00:00:00Z'::TIMESTAMPTZ >= v_cutoff",
        )
        with connection.cursor() as cursor:
            cutoff_target = _snapshot(cursor, ROUND)
            cutoff_history = _snapshot(cursor, HISTORY)
        with pytest.raises(Exception, match="before submission cutoff"):
            _execute(connection, after_cutoff_sql)
        connection.rollback()
        with connection.cursor() as cursor:
            assert _snapshot(cursor, ROUND) == cutoff_target
            assert _snapshot(cursor, HISTORY) == cutoff_history
            cursor.execute(
                "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
                "'public.lab_arena_rounds'::regclass AND "
                "tgname='lab_arena_rounds_write_once'"
            )
            assert cursor.fetchone()[0] == "O"

        cases = (
            (
                "DELETE FROM public.lab_arena_submissions "
                "WHERE round_id='arena-2026-09-18';"
                "DELETE FROM public.lab_arena_rounds "
                "WHERE round_id='arena-2026-09-18'",
                "sep18 open round missing",
            ),
            (
                "UPDATE public.lab_arena_rounds SET status='committed' "
                "WHERE round_id='arena-2026-09-18'",
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
                "configuration_doc,'{schedule,submission_cutoff}',"
                "'\"2026-09-18T01:00:00Z\"'::jsonb,false) "
                "WHERE round_id='arena-2026-09-18'",
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_rounds SET participants='[]'::jsonb "
                "WHERE round_id='arena-2026-09-18'",
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_rounds SET benchmark_ref="
                "'arena/arena-2026-09-18/benchmark.json' "
                "WHERE round_id='arena-2026-09-18'",
                "state differs",
            ),
            (
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king,submission_doc) "
                "VALUES ('baseline-2026-09-18','arena-2026-09-18',"
                "'%s','uploading',TRUE,'{}'::jsonb)" % BASELINE_HOTKEY,
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_submissions SET status='frozen',"
                "frozen_at='2026-09-17T22:00:00Z' "
                "WHERE submission_id='sep18-miner'",
                "state differs",
            ),
            (
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status) VALUES ("
                "'started-run','started-assignment','arena-2026-09-18',"
                "'sep18-miner','%s',1,0,1,'execute','pending')" % MINER_HOTKEY,
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
                "configuration_doc,'{call_quotas,openrouter}','61'::jsonb,false) "
                "WHERE round_id='arena-2026-09-18'",
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
                "configuration_doc,'{scoring_call_quotas,openrouter}','121'::jsonb,false) "
                "WHERE round_id='arena-2026-09-18'",
                "state differs",
            ),
        )
        for mutation, message in cases:
            _expect_failure(connection, mutation, message)
    finally:
        connection.close()


def test_migration_287_verbatim_committed_bytes_respect_real_cutoff(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    sql = MIGRATION.read_text(encoding="utf-8")
    before_cutoff = datetime.now(timezone.utc) < datetime(
        2026, 9, 18, tzinfo=timezone.utc
    )
    try:
        _seed(connection)
        with connection.cursor() as cursor:
            before_target = _snapshot(cursor, ROUND)
            before_history = _snapshot(cursor, HISTORY)
        if before_cutoff:
            _execute(connection, sql)
            with connection.cursor() as cursor:
                after_target = _snapshot(cursor, ROUND)
                after_history = _snapshot(cursor, HISTORY)
            expected = copy.deepcopy(before_target)
            expected["round"]["configuration_doc"]["call_quotas"][
                "openrouter"
            ] = 200
            assert after_target == expected
            assert after_history == before_history
            _execute(connection, sql)
            with connection.cursor() as cursor:
                assert _snapshot(cursor, ROUND) == after_target
                assert _snapshot(cursor, HISTORY) == after_history
                cursor.execute(
                    "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
                    "'public.lab_arena_rounds'::regclass AND "
                    "tgname='lab_arena_rounds_write_once'"
                )
                assert cursor.fetchone()[0] == "O"
        else:
            with pytest.raises(Exception, match="before submission cutoff"):
                _execute(connection, sql)
            connection.rollback()
            with connection.cursor() as cursor:
                assert _snapshot(cursor, ROUND) == before_target
                assert _snapshot(cursor, HISTORY) == before_history
    finally:
        connection.close()
