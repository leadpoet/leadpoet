"""Real-PostgreSQL proof for the Sep21 cutoff disclosure transition."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lab_arena import contracts
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/344-arena-september-cutoff-disclosure.sql"
ROUND = "arena-2026-09-21"
NEXT_ROUND = "arena-2026-09-22"
HISTORICAL_ROUND = "arena-2026-09-20"
OLD_POLICY = "after_scoring_day2_v1"
NEW_POLICY = "cutoff_public_v1"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration(round_id: str, *, policy: str = OLD_POLICY, mode: str = "live", cutoff: str | None = None) -> dict:
    date = round_id.removeprefix("arena-")
    return {
        "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": round_id,
        "mode": mode,
        "benchmark_disclosure_policy": policy,
        "schedule": {"submission_cutoff": cutoff or f"{date}T00:00:00Z"},
        "preserved_rule": {"value": ["unchanged", 344]},
    }


def _seed(connection, *, include_next_round: bool = True) -> None:
    baseline = "baseline-2026-09-21"
    miner = "sep21-frozen-miner"
    history_submission = "sep20-history-miner"
    baseline_hotkey = hotkey(baseline)
    miner_hotkey = hotkey(miner)
    history_hotkey = hotkey(history_submission)
    participants = [
        {
            "submission_id": baseline,
            "miner_hotkey": baseline_hotkey,
            "is_king": True,
        },
        {
            "submission_id": miner,
            "miner_hotkey": miner_hotkey,
            "is_king": False,
        },
    ]
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger, public.lab_arena_runs, "
            "public.lab_arena_submissions, public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,status_generation,stage_generation,configuration_doc,"
            "rewards_enabled,participants,benchmark_ref,evaluation_date,icp_set_date) "
            "VALUES (%s,'stage1',1,1,%s::jsonb,TRUE,%s::jsonb,%s,%s,%s)",
            (
                ROUND,
                json.dumps(_configuration(ROUND)),
                json.dumps(participants),
                f"arena/{ROUND}/benchmark.json",
                "2026-09-21",
                "2026-09-20",
            ),
        )
        if include_next_round:
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,status,configuration_doc,rewards_enabled) "
                "VALUES (%s,'open',%s::jsonb,TRUE)",
                (NEXT_ROUND, json.dumps(_configuration(NEXT_ROUND))),
            )
        historical_publication = {
            "schema_version": "leadpoet.lab_arena.publication.v1",
            "round_id": HISTORICAL_ROUND,
            "published_at": "2026-09-20T00:00:00Z",
            "result": {"aggregate_score": 91.25, "positive": True},
        }
        historical_participants = [
            {
                "submission_id": history_submission,
                "miner_hotkey": history_hotkey,
                "is_king": False,
            }
        ]
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,configuration_doc,rewards_enabled,participants,"
            "publication_doc,king_outcome,king_hotkey,published_at) "
            "VALUES (%s,'published',%s::jsonb,TRUE,%s::jsonb,%s::jsonb,"
            "'crowned',%s,%s::timestamptz)",
            (
                HISTORICAL_ROUND,
                json.dumps(_configuration(HISTORICAL_ROUND)),
                json.dumps(historical_participants),
                json.dumps(historical_publication),
                history_hotkey,
                "2026-09-20T00:00:00Z",
            ),
        )

        for submission_id, miner_key, is_king, round_id in (
            (baseline, baseline_hotkey, True, ROUND),
            (miner, miner_hotkey, False, ROUND),
            (history_submission, history_hotkey, False, HISTORICAL_ROUND),
        ):
            source_ref = f"arena/{round_id}/sources/{submission_id}.tar.gz"
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
                "source_size_bytes,submission_doc,frozen_at) VALUES "
                "(%s,%s,%s,'frozen',%s,%s,1000,%s::jsonb,clock_timestamp())",
                (
                    submission_id,
                    round_id,
                    miner_key,
                    is_king,
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

        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,result_doc,output_ref,terminal_cause) "
            "VALUES (%s,%s,%s,%s,%s,1,0,1,'execute','accepted',%s::jsonb,%s,'accepted')",
            (
                "sep21-baseline-accepted",
                "sep21-baseline-accepted-assignment",
                ROUND,
                baseline,
                baseline_hotkey,
                json.dumps({"schema_version": "test.accepted.v1", "score": 88.5}),
                "arena/sep21/baseline-accepted.json",
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status) VALUES "
            "('sep21-baseline-pending','sep21-baseline-pending-assignment',%s,%s,%s,1,1,1,'execute','pending')",
            (ROUND, baseline, baseline_hotkey),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status) VALUES "
            "('sep21-miner-pending','sep21-miner-pending-assignment',%s,%s,%s,1,0,1,'execute','pending')",
            (ROUND, miner, miner_hotkey),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,result_doc,output_ref,terminal_cause) "
            "VALUES ('sep20-history-accepted','sep20-history-assignment',%s,%s,%s,"
            "1,0,1,'execute','accepted','{}'::jsonb,'arena/sep20/history.json','accepted')",
            (HISTORICAL_ROUND, history_submission, history_hotkey),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "terminal_response,entry_doc) VALUES "
            "('settlement',%s,%s,%s,'sep21-baseline-accepted',1,%s,"
            "'openrouter','openrouter.execute','host',1234,%s::jsonb,%s::jsonb)",
            (
                baseline_hotkey,
                ROUND,
                baseline,
                "sha256:" + "a" * 64,
                json.dumps({"status": "confirmed", "charge_microusd": 1234}),
                json.dumps({"confirmed": True}),
            ),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _round(cursor, round_id: str) -> dict:
    cursor.execute(
        "SELECT to_jsonb(round_row) FROM public.lab_arena_rounds round_row "
        "WHERE round_id=%s",
        (round_id,),
    )
    return cursor.fetchone()[0]


def _bytes(cursor, table: str, order: str, where: str = "TRUE") -> list[str]:
    cursor.execute(
        "SELECT COALESCE(array_agg(encode(jsonb_send(to_jsonb(row_data)), 'hex') "
        f"ORDER BY {order}), ARRAY[]::text[]) FROM public.{table} row_data "
        f"WHERE {where}"
    )
    return cursor.fetchone()[0]


def _snapshot(cursor) -> dict[str, list[str]]:
    return {
        "rounds": _bytes(cursor, "lab_arena_rounds", "round_id"),
        "submissions": _bytes(cursor, "lab_arena_submissions", "submission_id"),
        "runs": _bytes(cursor, "lab_arena_runs", "run_id"),
        "ledger": _bytes(cursor, "lab_arena_ledger", "entry_id"),
    }


def _protected_round(round_row: dict) -> dict:
    normalized = copy.deepcopy(round_row)
    normalized.pop("updated_at")
    normalized["configuration_doc"].pop("benchmark_disclosure_policy")
    return normalized


def _execute(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.commit()


def _mutate(connection, statement: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(statement)
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _trigger_enabled(cursor) -> str:
    cursor.execute(
        "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
        "'public.lab_arena_rounds'::regclass AND "
        "tgname='lab_arena_rounds_write_once' AND NOT tgisinternal"
    )
    return cursor.fetchone()[0]


def test_transition_changes_only_two_policies_preserves_work_and_replays_exactly(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection)
        with connection.cursor() as cursor:
            before = _snapshot(cursor)
            sep21_before = _round(cursor, ROUND)
            sep22_before = _round(cursor, NEXT_ROUND)
            protected_rounds_before = _bytes(
                cursor,
                "lab_arena_rounds",
                "round_id",
                "round_id NOT IN ('arena-2026-09-21','arena-2026-09-22')",
            )

        _execute(connection)

        with connection.cursor() as cursor:
            after = _snapshot(cursor)
            sep21_after = _round(cursor, ROUND)
            sep22_after = _round(cursor, NEXT_ROUND)
            assert _trigger_enabled(cursor) == "O"
            assert sep21_after["configuration_doc"]["benchmark_disclosure_policy"] == NEW_POLICY
            assert sep22_after["configuration_doc"]["benchmark_disclosure_policy"] == NEW_POLICY
            assert _protected_round(sep21_after) == _protected_round(sep21_before)
            assert _protected_round(sep22_after) == _protected_round(sep22_before)
            assert sep21_after["updated_at"] != sep21_before["updated_at"]
            assert sep22_after["updated_at"] != sep22_before["updated_at"]
            assert after["submissions"] == before["submissions"]
            assert after["runs"] == before["runs"]
            assert after["ledger"] == before["ledger"]
            assert _bytes(
                cursor,
                "lab_arena_rounds",
                "round_id",
                "round_id NOT IN ('arena-2026-09-21','arena-2026-09-22')",
            ) == protected_rounds_before

        _execute(connection)
        with connection.cursor() as cursor:
            assert _snapshot(cursor) == after
            assert _trigger_enabled(cursor) == "O"
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("case", "include_next_round", "mutation"),
    (
        ("missing round", False, None),
        (
            "unexpected policy",
            True,
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "jsonb_set(configuration_doc,'{benchmark_disclosure_policy}',"
            "'\"future_policy\"'::jsonb,FALSE) WHERE round_id='arena-2026-09-22'",
        ),
        (
            "unexpected date",
            True,
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "jsonb_set(configuration_doc,'{schedule,submission_cutoff}',"
            "'\"2026-09-30T00:00:00Z\"'::jsonb,FALSE) WHERE round_id='arena-2026-09-22'",
        ),
        (
            "unexpected mode",
            True,
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "jsonb_set(configuration_doc,'{mode}','\"shadow\"'::jsonb,FALSE) "
            "WHERE round_id='arena-2026-09-22'",
        ),
        (
            "published old policy",
            True,
            "UPDATE public.lab_arena_rounds SET status='published' "
            "WHERE round_id='arena-2026-09-22'",
        ),
    ),
)
def test_invalid_transition_rolls_back_and_restores_write_once_trigger(
    database, case, include_next_round, mutation
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection, include_next_round=include_next_round)
        if mutation is not None:
            _mutate(connection, mutation)
        with connection.cursor() as cursor:
            before = _snapshot(cursor)

        with pytest.raises(psycopg2.Error, match="Daily disclosure transition"):
            _execute(connection)
        connection.rollback()

        with connection.cursor() as cursor:
            assert _snapshot(cursor) == before, case
            assert _trigger_enabled(cursor) == "O", case
    finally:
        connection.close()
