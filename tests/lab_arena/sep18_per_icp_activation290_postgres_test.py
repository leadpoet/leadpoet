"""Disposable PostgreSQL proof for the dated per-ICP activation."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.sep18_open_quota287_postgres_test import _configuration


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/290-arena-2026-09-18-per-icp-cost-activation.sql"
ROUND = "arena-2026-09-18"
NEXT = "arena-2026-09-19"
ARCHIVE_SHA = "7e1bb0747014a57bc50f48f9f822d1a7c936682d63f06564e978d23f54eb7fc1"
SOURCE_COMMIT = "e5341f85829ad196b4a1cb58b38a34155697c8d4"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + ("289-lab-arena-per-icp-cost-policy.sql",)
    )


def _hotkey(index: int) -> str:
    return "5" + chr(ord("A") + index) * 47


def _seed(connection) -> None:
    config = _configuration()
    config["call_quotas"]["openrouter"] = 200
    participants = [
        {
            "submission_id": (
                "baseline-2026-09-18" if index == 0 else "sep18-miner-%d" % index
            ),
            "miner_hotkey": _hotkey(index),
            "is_king": index == 0,
        }
        for index in range(5)
    ]
    next_config = copy.deepcopy(config)
    next_config["round_id"] = NEXT
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
            "(%s,'stage1',4,1,%s::jsonb,TRUE,%s::jsonb,%s),"
            "(%s,'open',0,0,%s::jsonb,TRUE,NULL,NULL)",
            (
                ROUND, json.dumps(config), json.dumps(participants),
                "arena/arena-2026-09-18/benchmark.json",
                NEXT, json.dumps(next_config),
            ),
        )
        for participant in participants:
            is_baseline = participant["is_king"]
            document = {
                "source_sha256": ARCHIVE_SHA if is_baseline else ("a" * 64),
                "source_commit": SOURCE_COMMIT if is_baseline else ("b" * 40),
            }
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
                "source_ref,source_size_bytes,frozen_at) "
                "VALUES (%s,%s,%s,'frozen',%s,%s::jsonb,%s,%s,clock_timestamp())",
                (
                    participant["submission_id"], ROUND,
                    participant["miner_hotkey"], is_baseline,
                    json.dumps(document),
                    "arena/%s/sources/%s.tar.gz" % (
                        ROUND, participant["submission_id"]
                    ),
                    604847 if is_baseline else 1000,
                ),
            )
            for position in range(20):
                stage = 1 if position < 10 else 2
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs("
                    "run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                    "stage,icp_position,attempt,kind,status,stage_generation) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,1,'execute','pending',1)",
                    (
                        "%s:%d" % (participant["submission_id"], position),
                        "%s:a:%d" % (participant["submission_id"], position),
                        ROUND, participant["submission_id"],
                        participant["miner_hotkey"], stage, position,
                    ),
                )
        for entry_kind, position, amount, suffix in (
            ("settlement", 0, 996000, "settled"),
            ("reservation", 1, 67000000, "dynamic"),
        ):
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,"
                "amount_microusd,entry_doc,terminal_response) "
                "VALUES (%s,%s,%s,'baseline-2026-09-18',%s,1,%s,"
                "'deepline','deepline.execute','host',%s,%s::jsonb,%s::jsonb)",
                (
                    entry_kind, _hotkey(0), ROUND,
                    "baseline-2026-09-18:%d" % position,
                    "sha256:" + (("1" if suffix == "settled" else "2") * 64),
                    amount,
                    json.dumps({"reserve_remaining_budget": True}),
                    json.dumps({"call_succeeded": True})
                    if entry_kind == "settlement" else None,
                ),
            )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _snapshot(cursor) -> dict:
    result = {}
    for table, order in (
        ("lab_arena_rounds", "round_id"),
        ("lab_arena_submissions", "submission_id"),
        ("lab_arena_runs", "run_id"),
        ("lab_arena_ledger", "entry_id"),
    ):
        cursor.execute(
            "SELECT COALESCE(jsonb_agg(to_jsonb(rows) ORDER BY %s),'[]'::jsonb) "
            "FROM public.%s AS rows" % (order, table)
        )
        result[table] = cursor.fetchone()[0]
    return result


def _execute(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.commit()


def test_activation_preserves_work_and_replays(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection)
        with connection.cursor() as cursor:
            before = _snapshot(cursor)
        _execute(connection)
        with connection.cursor() as cursor:
            after = _snapshot(cursor)
            cursor.execute(
                "SELECT round_id,configuration_doc FROM public.lab_arena_rounds "
                "ORDER BY round_id"
            )
            configs = dict(cursor.fetchall())
            cursor.execute(
                "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
                "'public.lab_arena_rounds'::regclass AND "
                "tgname='lab_arena_rounds_write_once'"
            )
            assert cursor.fetchone()[0] == "O"
        assert after["lab_arena_submissions"] == before["lab_arena_submissions"]
        assert after["lab_arena_runs"] == before["lab_arena_runs"]
        assert after["lab_arena_ledger"] == before["lab_arena_ledger"]
        for config in configs.values():
            assert config["sourcing_cost_eligibility_policy"] == (
                "successful_calls_per_icp_v1"
            )
            assert config["execution_icp_cap_microusd"] == 4_000_000
        _execute(connection)
        with connection.cursor() as cursor:
            replayed = _snapshot(cursor)
        assert replayed == after
    finally:
        connection.close()


@pytest.mark.parametrize(
    "mutation,message",
    (
        (
            "UPDATE public.lab_arena_rounds SET status='published' "
            "WHERE round_id='arena-2026-09-18'",
            "sep18 activation state differs",
        ),
        (
            "UPDATE public.lab_arena_runs SET kind='score',scored_run_id=run_id "
            "WHERE run_id='baseline-2026-09-18:0'",
            "execution snapshot is not safe",
        ),
        (
            "UPDATE public.lab_arena_ledger SET amount_microusd=4000000 "
            "WHERE entry_kind='settlement'",
            "execution snapshot is not safe",
        ),
        (
            "INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
            "VALUES ('refusal','5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',"
            "'arena-2026-09-18','baseline-2026-09-18',"
            "'baseline-2026-09-18:2',1,'sha256:'||repeat('3',64),"
            "'openrouter','openrouter.responses','host',0,'{\"reason\":\"money_cap\"}')",
            "execution snapshot is not safe",
        ),
        (
            "UPDATE public.lab_arena_submissions SET source_size_bytes=604846 "
            "WHERE submission_id='baseline-2026-09-18'",
            "frozen participant snapshot differs",
        ),
        (
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status) VALUES ("
            "'sep19-run','sep19-assignment','arena-2026-09-19',"
            "'baseline-2026-09-18','5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',"
            "1,0,1,'execute','pending')",
            "sep19 open activation state differs",
        ),
    ),
)
def test_activation_fails_atomically_on_changed_state(database, mutation, message):
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
            before = _snapshot(cursor)
        with pytest.raises(Exception, match=message):
            _execute(connection)
        connection.rollback()
        with connection.cursor() as cursor:
            assert _snapshot(cursor) == before
            cursor.execute(
                "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
                "'public.lab_arena_rounds'::regclass AND "
                "tgname='lab_arena_rounds_write_once'"
            )
            assert cursor.fetchone()[0] == "O"
    finally:
        connection.close()
