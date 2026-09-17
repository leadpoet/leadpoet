"""Disposable-PostgreSQL proof for the exact Sep18 schedule repair."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena import capacity, contracts, rewards
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.sep18_open_quota287_postgres_test import (
    BASELINE_HOTKEY,
    _configuration as quota_configuration,
)


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/288-arena-2026-09-18-schedule-capacity.sql"
ROUND = "arena-2026-09-18"
HISTORY = "arena-2026-09-17-history"
MINER_HOTKEYS = tuple("5" + char * 47 for char in "BCDE")
NEW_SCHEDULE = {
    "submission_open": "2026-09-17T00:00:00Z",
    "submission_cutoff": "2026-09-18T00:00:00Z",
    "benchmark_deadline": "2026-09-18T00:30:00Z",
    "stage_1_start": "2026-09-18T00:30:01Z",
    "stage_1_close": "2026-09-18T16:30:01Z",
    "stage_1_scoring_close": "2026-09-18T19:30:01Z",
    "stage_2_start": "2026-09-18T19:30:02Z",
    "stage_2_close": "2026-09-18T19:30:03Z",
    "final_scoring_close": "2026-09-18T22:30:03Z",
    "publication_deadline": "2026-09-18T22:30:04Z",
}


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration(round_id: str = ROUND) -> dict:
    config = quota_configuration(round_id)
    config["call_quotas"]["openrouter"] = 200
    config["scorer_image_digest"] = "sha256:" + "1" * 64
    config["scorer_image_reference"] = (
        "example.invalid/arena/judge@" + config["scorer_image_digest"]
    )
    config["reward_constants"] = rewards.reward_constants_document(25)
    return contracts.validate_round_configuration(config)


def _sql_at(timestamp: str) -> str:
    return MIGRATION.read_text(encoding="utf-8").replace(
        "pg_catalog.clock_timestamp() >= v_cutoff",
        "'%s'::TIMESTAMPTZ >= v_cutoff" % timestamp,
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
        for index, hotkey in enumerate(MINER_HOTKEYS):
            submission_id = "sep18-miner-%d" % index
            call_identity = "sha256:" + str(index + 2) * 64
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
                "source_size_bytes,consent,submission_doc,code_review_status,"
                "code_review_doc,code_review_claim,code_review_started_at,"
                "code_review_attempts) VALUES ("
                "%s,%s,%s,'accepted',FALSE,%s,123,'{\"public_rerun\":true}'::jsonb,"
                "jsonb_build_object('preserve',true,'source_sha256',%s,"
                "'source_commit',%s),'passed','{\"decision\":\"pass\"}'::jsonb,"
                "%s,'2026-09-17T20:00:00Z',1)",
                (
                    submission_id,
                    ROUND,
                    hotkey,
                    "arena/%s/sources/%s.tar.gz" % (ROUND, submission_id),
                    str(index + 3) * 64,
                    str(index + 4) * 40,
                    "sha256:" + str(index + 5) * 64,
                ),
            )
            for kind in ("reservation", "dispatch", "settlement"):
                entry_doc = {"review_status": "passed"} if kind == "settlement" else {}
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger("
                    "entry_kind,miner_hotkey,round_id,submission_id,call_identity,"
                    "provider,operation_id,funding_source,amount_microusd,entry_doc) "
                    "VALUES (%s,%s,%s,%s,%s,'openrouter','openrouter.code_review',"
                    "'miner_key',1,%s::jsonb)",
                    (
                        kind,
                        hotkey,
                        ROUND,
                        submission_id,
                        call_identity,
                        json.dumps(entry_doc),
                    ),
                )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
            "code_review_status,code_review_doc,code_review_claim,"
            "code_review_started_at,code_review_attempts) VALUES ("
            "'history-miner',%s,%s,'accepted',FALSE,'{\"historical\":true}'::jsonb,"
            "'passed','{\"decision\":\"pass\"}'::jsonb,%s,"
            "'2026-09-16T20:00:00Z',1)",
            (HISTORY, MINER_HOTKEYS[0], "sha256:" + "f" * 64),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _snapshot(cursor, round_id: str) -> dict:
    result = {}
    for name, table, order in (
        ("rounds", "lab_arena_rounds", "round_id"),
        ("submissions", "lab_arena_submissions", "submission_id"),
        ("runs", "lab_arena_runs", "run_id"),
        ("ledger", "lab_arena_ledger", "entry_id"),
    ):
        cursor.execute(
            "SELECT coalesce(jsonb_agg(to_jsonb(x) ORDER BY %s),'[]'::jsonb) "
            "FROM public.%s x WHERE round_id=%%s" % (order, table),
            (round_id,),
        )
        result[name] = cursor.fetchone()[0]
    return result


def _execute(connection, sql: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(sql)
    connection.commit()


def _assert_round_trigger_enabled(cursor) -> None:
    cursor.execute(
        "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
        "'public.lab_arena_rounds'::regclass AND "
        "tgname='lab_arena_rounds_write_once'"
    )
    assert cursor.fetchone()[0] == "O"


def _expect_failure(connection, mutation: str, message: str) -> None:
    _seed(connection)
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(mutation)
        cursor.execute("SET session_replication_role=origin")
    connection.commit()
    with connection.cursor() as cursor:
        before = _snapshot(cursor, ROUND)
        history = _snapshot(cursor, HISTORY)
    with pytest.raises(Exception, match=message):
        _execute(connection, _sql_at("2026-09-17T23:00:00Z"))
    connection.rollback()
    with connection.cursor() as cursor:
        assert _snapshot(cursor, ROUND) == before
        assert _snapshot(cursor, HISTORY) == history
        _assert_round_trigger_enabled(cursor)


def test_migration_288_transition_replay_contract_and_fail_closed_matrix(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection)
        config = _configuration()
        revised = copy.deepcopy(config)
        revised["schedule"] = NEW_SCHEDULE
        assert contracts.validate_round_configuration(revised) == revised
        assert capacity.daily_challenger_capacity(revised) == 9
        assert 2 * 10 * (2700 + 60) == 55_200
        assert 2 * 5 * (900 + 60) == 9_600

        with connection.cursor() as cursor:
            before = _snapshot(cursor, ROUND)
            history = _snapshot(cursor, HISTORY)
        _execute(connection, _sql_at("2026-09-17T23:00:00Z"))
        with connection.cursor() as cursor:
            after = _snapshot(cursor, ROUND)
            assert _snapshot(cursor, HISTORY) == history
        expected = copy.deepcopy(before)
        expected["rounds"][0]["configuration_doc"]["schedule"] = NEW_SCHEDULE
        assert after == expected
        _execute(connection, _sql_at("2026-09-17T23:00:00Z"))
        with connection.cursor() as cursor:
            assert _snapshot(cursor, ROUND) == after
            _assert_round_trigger_enabled(cursor)

        _seed(connection)
        with connection.cursor() as cursor:
            cutoff_before = _snapshot(cursor, ROUND)
        with pytest.raises(Exception, match="before submission cutoff"):
            _execute(connection, _sql_at("2026-09-18T00:00:00Z"))
        connection.rollback()
        with connection.cursor() as cursor:
            assert _snapshot(cursor, ROUND) == cutoff_before
            _assert_round_trigger_enabled(cursor)

        cases = (
            (
                "UPDATE public.lab_arena_rounds SET status='committed' "
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
                "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
                "configuration_doc,'{schedule,stage_1_close}',"
                "'\"2026-09-18T16:30:00Z\"'::jsonb,false) "
                "WHERE round_id='arena-2026-09-18'",
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_submissions SET status='frozen',"
                "frozen_at='2026-09-17T22:00:00Z' "
                "WHERE submission_id='sep18-miner-0'",
                "state differs",
            ),
            (
                "DELETE FROM public.lab_arena_ledger "
                "WHERE submission_id='sep18-miner-3';"
                "DELETE FROM public.lab_arena_submissions "
                "WHERE submission_id='sep18-miner-3'",
                "admission snapshot differs",
            ),
            (
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
                "source_size_bytes,consent,submission_doc,code_review_status) "
                "VALUES ('baseline-2026-09-18','arena-2026-09-18','%s','uploading',"
                "FALSE,'arena/arena-2026-09-18/sources/baseline.tar.gz',123,"
                "'{\"public_rerun\":true}'::jsonb,"
                "'{\"source_sha256\":\"%s\",\"source_commit\":\"%s\"}'::jsonb,"
                "'pending')" % (BASELINE_HOTKEY, "a" * 64, "b" * 40),
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_submissions SET code_review_status='error',"
                "code_review_doc='{\"reason\":\"test\"}'::jsonb "
                "WHERE submission_id='sep18-miner-0'",
                "admission snapshot differs",
            ),
            (
                "DELETE FROM public.lab_arena_ledger WHERE entry_id=("
                "SELECT min(entry_id) FROM public.lab_arena_ledger "
                "WHERE round_id='arena-2026-09-18')",
                "ledger snapshot differs",
            ),
            (
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status) VALUES ("
                "'started-run','started-assignment','arena-2026-09-18',"
                "'sep18-miner-0','%s',1,0,1,'execute','pending')"
                % MINER_HOTKEYS[0],
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
                "configuration_doc,'{call_quotas,openrouter}','199'::jsonb,false) "
                "WHERE round_id='arena-2026-09-18'",
                "state differs",
            ),
            (
                "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
                "configuration_doc,'{runner_slot_ceiling}','19'::jsonb,false) "
                "WHERE round_id='arena-2026-09-18'",
                "state differs",
            ),
        )
        for mutation, message in cases:
            _expect_failure(connection, mutation, message)

        _seed(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "DELETE FROM public.lab_arena_ledger WHERE round_id=%s", (ROUND,)
            )
            cursor.execute(
                "DELETE FROM public.lab_arena_submissions WHERE round_id=%s", (ROUND,)
            )
            cursor.execute("DELETE FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        with pytest.raises(Exception, match="open round missing"):
            _execute(connection, _sql_at("2026-09-17T23:00:00Z"))
        connection.rollback()
        with connection.cursor() as cursor:
            assert _snapshot(cursor, HISTORY)["submissions"][0]["submission_id"] == (
                "history-miner"
            )
            _assert_round_trigger_enabled(cursor)
    finally:
        connection.close()


def test_migration_288_verbatim_candidate_bytes_respect_real_cutoff(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    sql = MIGRATION.read_text(encoding="utf-8")
    before_cutoff = datetime.now(timezone.utc) < datetime(
        2026, 9, 18, tzinfo=timezone.utc
    )
    try:
        _seed(connection)
        with connection.cursor() as cursor:
            before = _snapshot(cursor, ROUND)
            history = _snapshot(cursor, HISTORY)
        if before_cutoff:
            _execute(connection, sql)
            with connection.cursor() as cursor:
                after = _snapshot(cursor, ROUND)
                assert _snapshot(cursor, HISTORY) == history
            expected = copy.deepcopy(before)
            expected["rounds"][0]["configuration_doc"]["schedule"] = NEW_SCHEDULE
            assert after == expected
            _execute(connection, sql)
            with connection.cursor() as cursor:
                assert _snapshot(cursor, ROUND) == after
                _assert_round_trigger_enabled(cursor)
        else:
            with pytest.raises(Exception, match="before submission cutoff"):
                _execute(connection, sql)
            connection.rollback()
            with connection.cursor() as cursor:
                assert _snapshot(cursor, ROUND) == before
                assert _snapshot(cursor, HISTORY) == history
                _assert_round_trigger_enabled(cursor)
    finally:
        connection.close()
