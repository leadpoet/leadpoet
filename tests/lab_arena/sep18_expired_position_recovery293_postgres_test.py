"""Disposable PostgreSQL proof for the exact Sep18 position-12 recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.sep18_budget_stop_recovery291_postgres_test import (
    BASELINE,
    BASELINE_HOTKEY,
    ROUND,
    _apply as _apply_291,
    _connect,
    _seed,
    _snapshot,
)


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/293-arena-2026-09-18-expired-position-recovery.sql"
TARGET_RUN = f"{ROUND}:{BASELINE}:2:12:3"
RETRY_RUN = f"{ROUND}:{BASELINE}:2:12:4"
RUNNER_HOTKEY = "5" + "C" * 47
EXTENDED_SCHEDULE = {
    "benchmark_deadline": "2026-09-18T00:30:00Z",
    "final_scoring_close": "2026-09-18T22:30:03Z",
    "publication_deadline": "2026-09-18T22:30:04Z",
    "stage_1_close": "2026-09-18T16:30:01Z",
    "stage_1_scoring_close": "2026-09-18T19:30:01Z",
    "stage_1_start": "2026-09-18T00:30:01Z",
    "stage_2_close": "2026-09-18T19:30:03Z",
    "stage_2_start": "2026-09-18T19:30:02Z",
    "submission_cutoff": "2026-09-18T00:00:00Z",
    "submission_open": "2026-09-17T00:00:00Z",
}


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + ("289-lab-arena-per-icp-cost-policy.sql",)
    )


def _apply_293(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(_migration_sql())
    connection.commit()


def _migration_sql() -> str:
    sql = MIGRATION.read_text(encoding="utf-8")
    live_guard = (
        "pg_catalog.clock_timestamp() + INTERVAL '45 minutes' > v_stage_close"
    )
    assert sql.count(live_guard) == 1
    return sql.replace(
        live_guard,
        "'2026-09-18T04:10:00Z'::TIMESTAMPTZ + "
        "INTERVAL '45 minutes' > v_stage_close",
    )


def test_expired_position_recovery_keeps_live_capacity_guard() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert (
        "pg_catalog.clock_timestamp() + INTERVAL '45 minutes' > v_stage_close"
        in sql
    )
    assert "'2026-09-18T04:10:00Z'" not in sql


def _seed_expired_state(connection) -> None:
    _seed(connection)
    _apply_291(connection)
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "terminal_cause='budget_exhausted',"
            "result_doc='{\"terminal_status\":\"budget_exhausted\"}'::jsonb,"
            "runner_hotkey=%s,lease_token_hash='sha256:'||repeat('7',64),"
            "lease_expires_at='2026-09-18T04:05:00Z'::timestamptz "
            "WHERE round_id=%s AND submission_id=%s AND stage=2 AND attempt=3 "
            "AND icp_position IN (10,11,13,14,19)",
            (RUNNER_HOTKEY, ROUND, BASELINE),
        )
        assert cursor.rowcount == 5
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "terminal_cause='lease_expired',result_doc=NULL,output_ref=NULL,"
            "terminal_doc=%s::jsonb,runner_hotkey=%s,"
            "lease_token_hash='sha256:'||repeat('8',64),lease_generation=3,"
            "lease_expires_at='2026-09-18T04:06:10.357793Z'::timestamptz "
            "WHERE run_id=%s",
            (
                json.dumps({"expired_at": "2026-09-18T04:06:49.130326Z"}),
                RUNNER_HOTKEY,
                TARGET_RUN,
            ),
        )
        assert cursor.rowcount == 1
        for call_index in range(10):
            identity = "sha256:" + f"{call_index + 100:064x}"
            amount = 90_616 if call_index == 9 else 1_000 + call_index
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,"
                "amount_microusd,entry_doc) VALUES ("
                "'reservation',%s,%s,%s,%s,2,%s,'openrouter',"
                "'openrouter.responses','host',%s,%s::jsonb)",
                (
                    BASELINE_HOTKEY,
                    ROUND,
                    BASELINE,
                    TARGET_RUN,
                    identity,
                    amount,
                    json.dumps(
                        {
                            "action_sequence": call_index,
                            "base_call_identity": identity,
                            "max_output_tokens": 100,
                            "model": "openai/gpt-5-mini",
                            "provider_attempt": 1,
                            "request_hash": "sha256:" + "a" * 64,
                        }
                    ),
                ),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,"
                "amount_microusd,entry_doc) VALUES ("
                "'dispatch',%s,%s,%s,%s,2,%s,'openrouter',"
                "'openrouter.responses','host',%s,"
                "'{\"dispatched_at\":\"2026-09-18T02:31:04Z\"}'::jsonb)",
                (
                    BASELINE_HOTKEY,
                    ROUND,
                    BASELINE,
                    TARGET_RUN,
                    identity,
                    amount,
                ),
            )
            if call_index < 9:
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger("
                    "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                    "call_identity,provider,operation_id,funding_source,"
                    "amount_microusd,entry_doc,terminal_response) VALUES ("
                    "'settlement',%s,%s,%s,%s,2,%s,'openrouter',"
                    "'openrouter.responses','host',%s,'{}'::jsonb,"
                    "'{\"status\":200}'::jsonb)",
                    (
                        BASELINE_HOTKEY,
                        ROUND,
                        BASELINE,
                        TARGET_RUN,
                        identity,
                        amount,
                    ),
                )
            else:
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger("
                    "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                    "call_identity,provider,operation_id,funding_source,"
                    "amount_microusd,entry_doc) VALUES ("
                    "'uncertain',%s,%s,%s,%s,2,%s,'openrouter',"
                    "'openrouter.responses','host',90616,%s::jsonb)",
                    (
                        BASELINE_HOTKEY,
                        ROUND,
                        BASELINE,
                        TARGET_RUN,
                        identity,
                        json.dumps(
                            {
                                "reason": "lease_expired",
                                "call": {"call_succeeded": False},
                            }
                        ),
                    ),
                )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled',"
            "status_generation=5,stage_generation=4,"
            "cancel_reason='execution_incomplete:stage1:1',"
            "configuration_doc=jsonb_set(configuration_doc,'{schedule}',%s::jsonb) "
            "WHERE round_id=%s",
            (json.dumps(EXTENDED_SCHEDULE), ROUND),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def test_expired_position_recovery_preserves_history_and_reaches_scoring(database):
    connection = _connect(database)
    try:
        _seed_expired_state(connection)
        with connection.cursor() as cursor:
            before = _snapshot(cursor)
        assert len(before["lab_arena_runs"]) == 116
        assert sum(row["status"] == "accepted" for row in before["lab_arena_runs"]) == 93
        assert sum(
            row["status"] == "accepted" and row["output_ref"] is not None
            for row in before["lab_arena_runs"]
        ) == 93
        _apply_293(connection)
        with connection.cursor() as cursor:
            after = _snapshot(cursor)
            cursor.execute(
                "SELECT status,status_generation,stage_generation,cancel_reason,"
                "configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            round_row = cursor.fetchone()
            cursor.execute(
                "SELECT run_id,assignment_id,stage,icp_position,attempt,status,"
                "stage_generation,previous_runner_hotkey FROM public.lab_arena_runs "
                "WHERE run_id=%s",
                (RETRY_RUN,),
            )
            retry = cursor.fetchone()
        assert round_row[:4] == ("stage1", 6, 5, None)
        assert round_row[4] == before["lab_arena_rounds"][0]["configuration_doc"]
        assert retry == (
            RETRY_RUN,
            f"{ROUND}:{BASELINE}:2:12",
            2,
            12,
            4,
            "pending",
            5,
            RUNNER_HOTKEY,
        )
        assert after["lab_arena_submissions"] == before["lab_arena_submissions"]
        assert after["lab_arena_ledger"] == before["lab_arena_ledger"]
        old_runs = {row["run_id"]: row for row in before["lab_arena_runs"]}
        assert {
            row["run_id"]: row
            for row in after["lab_arena_runs"]
            if row["run_id"] in old_runs
        } == old_runs

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_claim_assignment(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    ROUND,
                    BASELINE_HOTKEY,
                    10,
                    20,
                    [],
                    "293" + "0" * 29,
                    "sha256:" + "b" * 64,
                    "sha256:" + "c" * 64,
                    3600,
                ),
            )
            claim = cursor.fetchone()[0]
            assert claim["status"] == "leased"
            assert claim["run_id"] == RETRY_RUN
            assert claim["attempt"] == 4
            assert claim["stage_generation"] == 5
            cursor.execute(
                "SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)",
                (
                    RETRY_RUN,
                    "sha256:" + "c" * 64,
                    json.dumps({"terminal_status": "accepted"}),
                    "accepted",
                    f"arena/{ROUND}/outputs/recovery293/12.json",
                ),
            )
            assert cursor.fetchone()[0]["status"] == "accepted"
            cursor.execute(
                "SELECT public.lab_arena_close_parallel_execution_v1(%s)",
                (ROUND,),
            )
            assert cursor.fetchone()[0]["status"] == "closed"
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,status_generation,stage_generation,cancel_reason "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == ("stage1_closed", 7, 6, None)
            cursor.execute(
                "SELECT count(*) FILTER (WHERE status='accepted') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute'",
                (ROUND,),
            )
            assert cursor.fetchone()[0] == 94
            completed = _snapshot(cursor)
        _apply_293(connection)
        with connection.cursor() as cursor:
            assert _snapshot(cursor) == completed
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            "UPDATE public.lab_arena_ledger SET amount_microusd=90617 "
            "WHERE run_id='" + TARGET_RUN + "' AND entry_kind='uncertain'",
            "accounting state differs",
        ),
        (
            "UPDATE public.lab_arena_runs SET output_ref=NULL WHERE round_id='"
            + ROUND + "' AND status='accepted' AND output_ref IS NOT NULL AND "
            "run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='"
            + ROUND + "' AND status='accepted')",
            "execution state differs",
        ),
        (
            "UPDATE public.lab_arena_submissions SET source_ref='arena/"
            + ROUND + "/sources/wrong.tar.gz' WHERE submission_id='" + BASELINE + "'",
            "source state differs",
        ),
        (
            "UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb WHERE run_id='"
            + TARGET_RUN + "'",
            "target state differs",
        ),
        (
            "UPDATE public.lab_arena_runs SET result_doc=NULL WHERE run_id='"
            + ROUND + ":" + BASELINE + ":2:10:3'",
            "execution state differs",
        ),
    ),
)
def test_expired_position_recovery_rejects_changed_state_atomically(
    database, mutation, message
):
    connection = _connect(database)
    try:
        _seed_expired_state(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(mutation)
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        with connection.cursor() as cursor:
            before = _snapshot(cursor)
        with pytest.raises(Exception, match=message):
            _apply_293(connection)
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
