"""Credit retry is bounded by settled proof, frozen stage, and owner identity."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import round_config


SQL = Path(__file__).parents[2] / "scripts/403-lab-arena-credit-failure-retry.sql"
ROUND = "arena-2030-01-01"
OWNER = "5" * 48
OTHER = "6" * 48
BASELINE_OWNER = "7" * 48
RUNNER = "8" * 48
REQUEST = "sha256:" + "e" * 64


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _call(cursor, *, round_id=ROUND, submission="sub-a", owner=OWNER,
          request=REQUEST):
    cursor.execute(
        "SELECT public.lab_arena_retry_credit_failures_v1(%s,%s,%s,%s)",
        (round_id, submission, owner, request),
    )
    return cursor.fetchone()[0]


def _seed(cursor, *, stage1_count=10, stage2_count=10, parallel=False):
    config = round_config(ROUND, [RUNNER])
    config["stage_1_icp_count"] = stage1_count
    config["stage_2_icp_count"] = stage2_count
    if parallel:
        config["parallel_twenty_icp_execution"] = True
    config["schedule"].update({
            "stage_1_close": "2030-01-01T20:00:00Z",
            "stage_1_scoring_close": "2030-01-01T23:00:00Z",
            "stage_2_close": "2030-01-02T01:00:00Z",
            "final_scoring_close": "2030-01-02T04:00:00Z",
    })
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "INSERT INTO public.lab_arena_rounds "
        "(round_id,status,stage_generation,configuration_doc) "
        "VALUES (%s,'stage1',1,%s::jsonb)", (ROUND, json.dumps(config)),
    )
    for submission, hotkey, king in (
        ("sub-a", OWNER, False), ("sub-b", OTHER, False),
        ("baseline-a", BASELINE_OWNER, True),
    ):
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id,round_id,miner_hotkey,status,is_king,"
            "code_review_status,code_review_doc,code_review_claim,"
            "code_review_started_at,code_review_attempts,"
            "source_ref,source_size_bytes) "
            "VALUES (%s,%s,%s,'frozen',%s,'passed','{}'::jsonb,%s,"
            "clock_timestamp(),1,%s,4096)",
            (submission, ROUND, hotkey, king,
             "sha256:" + "a" * 64,
             f"arena/{ROUND}/sources/{submission}.tar.gz"),
        )
    cursor.execute("SET session_replication_role=origin")


def _failed(cursor, position, *, submission="sub-a", owner=OWNER,
            kind="execute", attempt=1, cause="credential_error",
            status="failed", generation=1, stage=1, cache_key=None,
            leader=None, company_refs=None):
    assignment = f"{ROUND}:{submission}:{stage}:{position}:{kind}"
    run_id = f"{assignment}:{attempt}"
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
        "stage,icp_position,attempt,kind,status,stage_generation,"
        "terminal_cause,result_doc,judgment_cache_key,"
        "judgment_group_leader,company_judgment_refs) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,"
        "%s,%s,%s::jsonb)",
        (run_id, assignment, ROUND, submission, owner, stage, position,
         attempt, kind, status, generation, cause,
         json.dumps({"terminal_status": cause}) if cause else None,
         cache_key, leader,
         json.dumps(company_refs) if company_refs is not None else None),
    )
    cursor.execute("SET session_replication_role=origin")
    return run_id


def _ledger(cursor, run_id, position, *, provider="openrouter", actual=0,
            status=402, proof=True, succeeded=False, kind="settlement",
            source="miner_key", stage=1, account_status=None):
    terminal = {"call_succeeded": succeeded, "status": status}
    if account_status is not None:
        terminal["account_failure_evidence"] = {
            "provider_status": account_status}
    if proof:
        terminal["credit_failure_proof"] = {
            "schema_version": "leadpoet.lab_arena.credit_failure_proof.v1",
            "provider": provider, "reason": "out_of_credit",
            "provider_status": status, "actual_microusd": actual,
        }
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,"
        "amount_microusd,terminal_response) "
        "VALUES (%s,%s,%s,'sub-a',%s,%s,%s,%s,'test',%s,%s,%s::jsonb)",
        (kind, OWNER, ROUND, run_id, stage,
         "sha256:" + format(position, "064x"),
         provider, source, actual, json.dumps(terminal)),
    )


def test_retry_preserves_history_and_replays_after_stage_moves(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            _seed(cursor)
            credit = _failed(cursor, 0)
            accepted = _failed(cursor, 1, cause="accepted", status="accepted")
            stale = _failed(cursor, 2, generation=0)
            charged = _failed(cursor, 3)
            unknown = _failed(cursor, 4)
            malformed = _failed(cursor, 8)
            forbidden = _failed(cursor, 9)
            second = _failed(cursor, 5)
            _failed(cursor, 5, attempt=2)
            baseline = _failed(cursor, 6, submission="baseline-a",
                               owner=BASELINE_OWNER)
            _ledger(cursor, credit, 0)
            _ledger(cursor, charged, 3, actual=7, status=200,
                    proof=False, succeeded=True)
            _ledger(cursor, charged, 13)
            _ledger(cursor, unknown, 4, proof=False)
            _ledger(cursor, malformed, 8)
            _ledger(cursor, malformed, 18, proof=False)
            _ledger(cursor, forbidden, 9, status=403, proof=False)
            _ledger(cursor, second, 5)
            assert _call(cursor, owner=OTHER)["status"] == "no_eligible"
            assert _call(cursor, round_id="arena-2030-01-02")["status"] == "no_eligible"
            assert _call(cursor, submission="baseline-a",
                         owner=BASELINE_OWNER)["status"] == "no_eligible"
            result = _call(cursor)
            assert result == {"status": "queued", "requeued_count": 2}
            cursor.execute(
                "SELECT credit_retry_parent_run_id,run_id,status,attempt,"
                "credit_retry_request_hash FROM public.lab_arena_runs "
                "WHERE credit_retry_request_hash=%s ORDER BY run_id", (REQUEST,),
            )
            rows = cursor.fetchall()
            assert [(row[0], row[1], row[2], row[3]) for row in rows] == [
                (credit, credit[:-1] + "2", "pending", 2),
                (charged, charged[:-1] + "2", "pending", 2),
            ]
            assert all(row[4] == REQUEST for row in rows)
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger "
                           "WHERE run_id IN (%s,%s)", (rows[0][1], rows[1][1]))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SAVEPOINT credit_lineage_test")
            with pytest.raises(psycopg.Error,
                               match="lab_arena_credit_retry_lineage_immutable"):
                cursor.execute(
                    "UPDATE public.lab_arena_runs "
                    "SET credit_retry_request_hash=%s WHERE run_id=%s",
                    ("sha256:" + "f" * 64, rows[0][1]),
                )
            cursor.execute("ROLLBACK TO SAVEPOINT credit_lineage_test")
            cursor.execute("RELEASE SAVEPOINT credit_lineage_test")
            lease_hash = "sha256:" + "1" * 64
            cursor.execute(
                "SELECT public.lab_arena_claim_assignment("
                "%s,%s,1,1,ARRAY[]::text[],%s,%s,%s,300)",
                (ROUND, RUNNER, "a" * 32, "sha256:" + "2" * 64,
                 lease_hash),
            )
            claimed = cursor.fetchone()[0]
            assert claimed["status"] == "leased", claimed
            assert claimed["run_id"] == rows[0][1]
            fresh_call = "sha256:" + "3" * 64
            cursor.execute(
                "SELECT public.lab_arena_reserve_call("
                "%s,%s,%s,'openrouter.chat','openrouter','miner_key',"
                "100,'{}'::jsonb,300)",
                (rows[0][1], lease_hash, fresh_call),
            )
            reservation = cursor.fetchone()[0]
            assert reservation["status"] == "reserved", reservation
            cursor.execute(
                "SELECT run_id,amount_microusd FROM public.lab_arena_ledger "
                "WHERE call_identity=%s AND entry_kind='reservation'",
                (fresh_call,),
            )
            assert cursor.fetchone() == (rows[0][1], 100)
            assert _call(cursor) == {"status": "replayed", "requeued_count": 2}
            assert _call(cursor, request="sha256:" + "d" * 64)["status"] == "no_eligible"
            concurrent = _failed(cursor, 7)
            _ledger(cursor, concurrent, 27)
            cursor.execute("SELECT count(*),coalesce(sum(amount_microusd),0) "
                           "FROM public.lab_arena_ledger")
            ledger_before = cursor.fetchone()
            connection.commit()
            concurrent_request = "sha256:" + "b" * 64
            def invoke():
                with psycopg.connect(**dsn) as other_connection:
                    with other_connection.cursor() as other_cursor:
                        return _call(other_cursor, request=concurrent_request)
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(lambda _: invoke(), range(2)))
            assert sorted(response["status"] for response in responses) == [
                "queued", "replayed"], responses
            assert all(response["requeued_count"] == 1 for response in responses)
            cursor.execute("SELECT count(*),coalesce(sum(amount_microusd),0) "
                           "FROM public.lab_arena_ledger")
            assert cursor.fetchone() == ledger_before
            cursor.execute("SELECT status,terminal_cause FROM public.lab_arena_runs "
                           "WHERE run_id IN (%s,%s,%s,%s)",
                           (credit, accepted, stale, baseline))
            assert {row[0] for row in cursor.fetchall()} == {"failed", "accepted"}
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1_scored' "
                           "WHERE round_id=%s", (ROUND,))
            cursor.execute("SET session_replication_role=origin")
            assert _call(cursor) == {"status": "replayed", "requeued_count": 2}
            cursor.execute(SQL.read_text())
            cursor.execute(SQL.read_text())
            assert _call(cursor) == {"status": "replayed", "requeued_count": 2}


def test_deadline_and_failure_controls(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            _seed(cursor)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1',"
                           "configuration_doc=jsonb_set(configuration_doc,"
                           "'{schedule,stage_1_close}',"
                           "'\"2020-01-01T00:00:00Z\"'::jsonb) "
                           "WHERE round_id=%s", (ROUND,))
            cursor.execute("SET session_replication_role=origin")
            assert _call(cursor, request="sha256:" + "c" * 64) == {
                "status": "no_eligible", "requeued_count": 0,
                "reason": "deadline_passed",
            }


def test_score_retry_keeps_one_live_cache_leader(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            _seed(cursor)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds "
                           "SET status='stage1_scoring' WHERE round_id=%s",
                           (ROUND,))
            cursor.execute("SET session_replication_role=origin")
            shared_key = "sha256:" + "a" * 64
            sole_key = "sha256:" + "b" * 64
            failed_shared = _failed(cursor, 0, kind="score",
                                    cache_key=shared_key, leader=True)
            _failed(cursor, 0, submission="sub-b", owner=OTHER, kind="score",
                    status="pending", cause=None, cache_key=shared_key,
                    leader=True)
            failed_sole = _failed(cursor, 1, kind="score",
                                  cache_key=sole_key, leader=True,
                                  company_refs=[])
            invalid401 = _failed(cursor, 2, kind="score")
            invalid429 = _failed(cursor, 3, kind="score")
            charged402 = _failed(cursor, 4, kind="score")
            conflicting_account = _failed(cursor, 5, kind="score")
            _ledger(cursor, failed_shared, 30)
            _ledger(cursor, failed_sole, 31)
            _ledger(cursor, invalid401, 32, status=401, proof=False)
            _ledger(cursor, invalid429, 33, status=429, proof=False)
            _ledger(cursor, charged402, 34, actual=7)
            _ledger(cursor, conflicting_account, 35, account_status=401)
            assert _call(cursor) == {"status": "queued", "requeued_count": 2}
            cursor.execute(
                "SELECT judgment_cache_key,judgment_group_leader "
                "FROM public.lab_arena_runs "
                "WHERE credit_retry_request_hash=%s ORDER BY judgment_cache_key",
                (REQUEST,),
            )
            assert cursor.fetchall() == [(shared_key, False), (sole_key, True)]
            cursor.execute(
                "SELECT company_judgment_refs FROM public.lab_arena_runs "
                "WHERE credit_retry_parent_run_id=%s", (failed_sole,),
            )
            assert cursor.fetchone()[0] == []
            lease_hash = "sha256:" + "4" * 64
            cursor.execute(
                "SELECT public.lab_arena_claim_assignment("
                "%s,%s,1,1,ARRAY[%s]::text[],%s,%s,%s,300)",
                (ROUND, RUNNER, OTHER, "c" * 32,
                 "sha256:" + "5" * 64, lease_hash),
            )
            claimed = cursor.fetchone()[0]
            assert claimed["status"] == "leased", claimed
            assert claimed["run_id"] == failed_sole[:-1] + "2"
            assert claimed["company_judgment_cache"]["schema_version"] == (
                "leadpoet.lab_arena.company_judgment_lease.v1")
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE judgment_cache_key=%s AND status IN "
                "('pending','leased','submitted') "
                "AND judgment_group_leader", (shared_key,),
            )
            assert cursor.fetchone()[0] == 1


@pytest.mark.parametrize("scenario", ["dynamic", "parallel"])
def test_stage_two_uses_frozen_positions_and_deadline(database, scenario):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            if scenario == "dynamic":
                _seed(cursor, stage1_count=8, stage2_count=12)
                position = 8
                round_status = "stage2"
            else:
                _seed(cursor, parallel=True)
                position = 10
                round_status = "stage1"
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status=%s,"
                "configuration_doc=jsonb_set(configuration_doc,"
                "'{schedule,stage_1_close}',"
                "'\"2020-01-01T00:00:00Z\"'::jsonb) "
                "WHERE round_id=%s", (round_status, ROUND),
            )
            cursor.execute("SET session_replication_role=origin")
            prior = _failed(cursor, position, stage=2)
            _ledger(cursor, prior, position, stage=2)
            result = _call(cursor)
            assert result == {"status": "queued", "requeued_count": 1}
            cursor.execute(
                "SELECT stage,icp_position,status FROM public.lab_arena_runs "
                "WHERE credit_retry_parent_run_id=%s", (prior,),
            )
            assert cursor.fetchone() == (2, position, "pending")
