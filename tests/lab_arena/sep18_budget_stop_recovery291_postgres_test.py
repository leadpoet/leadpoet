"""Disposable PostgreSQL proof for the Sep18 budget-stop recovery."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/291-arena-2026-09-18-budget-stop-recovery.sql"
ROUND = "arena-2026-09-18"
BASELINE = "baseline-2026-09-18"
BASELINE_HOTKEY = "5" + "A" * 47
PREVIOUS_RUNNER_HOTKEY = "5" + "B" * 47
SOURCE_REF = "arena/arena-2026-09-18/sources/baseline-2026-09-18.tar.gz"
SOURCE_COMMIT = "e5341f85829ad196b4a1cb58b38a34155697c8d4"
SOURCE_SHA256 = "7e1bb0747014a57bc50f48f9f822d1a7c936682d63f06564e978d23f54eb7fc1"
SOURCE_SIZE = 604847
RETRY_POSITIONS = (10, 11, 12, 13, 14, 19)
CAP_REFUSAL_POSITIONS = (10, 11, 13, 14, 19)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + ("289-lab-arena-per-icp-cost-policy.sql",)
    )


def _hotkey(index: int) -> str:
    return "5" + chr(ord("A") + index) * 47


def _baseline_run_id(position: int, attempt: int) -> str:
    stage = 1 if position < 10 else 2
    return f"{ROUND}:{BASELINE}:{stage}:{position}:{attempt}"


def _round_configuration() -> dict:
    from tests.lab_arena.sep18_open_quota287_postgres_test import _configuration as quota_configuration

    config = quota_configuration(ROUND)
    config["sourcing_cost_eligibility_policy"] = "successful_calls_per_icp_v1"
    config["execution_icp_cap_microusd"] = 4_000_000
    config["call_quotas"]["openrouter"] = 200
    return config


def _run_row(
    *,
    submission_id: str,
    miner_hotkey: str,
    position: int,
    attempt: int,
    status: str,
    terminal_cause: str,
    stage_generation: int = 2,
) -> tuple:
    stage = 1 if position < 10 else 2
    if submission_id == BASELINE:
        assignment_id = f"{ROUND}:{BASELINE}:{stage}:{position}"
    else:
        assignment_id = f"{submission_id}:assignment:{position}"
    run_id = f"{assignment_id}:{attempt}"
    accepted = status == "accepted"
    output_ref = (
        f"arena/{ROUND}/outputs/{submission_id}/{position}/attempt{attempt}.json"
        if accepted else None
    )
    result_doc = {"terminal_status": terminal_cause}
    terminal_doc = {"terminal_cause": terminal_cause, "attempt": attempt}
    return (
        run_id,
        assignment_id,
        ROUND,
        submission_id,
        miner_hotkey,
        stage,
        position,
        attempt,
        "execute",
        status,
        stage_generation,
        terminal_cause,
        json.dumps(result_doc),
        output_ref,
        json.dumps(terminal_doc),
    )


def _seed(connection) -> None:
    config = _round_configuration()
    participants = [
        {
            "submission_id": BASELINE if index == 0 else f"sep18-miner-{index}",
            "miner_hotkey": _hotkey(index),
            "is_king": index == 0,
            "source_ref": (
                SOURCE_REF if index == 0
                else f"arena/{ROUND}/sources/sep18-miner-{index}.tar.gz"
            ),
        }
        for index in range(5)
    ]
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
            "rewards_enabled,participants,benchmark_ref,cancel_reason) VALUES "
            "(%s,'cancelled',3,2,%s::jsonb,TRUE,%s::jsonb,%s,%s)",
            (
                ROUND,
                json.dumps(config),
                json.dumps(participants),
                f"arena/{ROUND}/benchmark.json",
                "execution_incomplete:stage1:6",
            ),
        )
        for participant in participants:
            is_baseline = participant["is_king"]
            source_size = SOURCE_SIZE if is_baseline else 1000
            document = {
                "source_ref": participant["source_ref"],
                "source_size_bytes": source_size,
                "consent": {"public_rerun": True},
            }
            if is_baseline:
                document.update({
                    "source_sha256": SOURCE_SHA256,
                    "source_commit": SOURCE_COMMIT,
                })
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
                "source_ref,source_size_bytes,consent,frozen_at) VALUES "
                "(%s,%s,%s,'frozen',%s,%s::jsonb,%s,%s,%s::jsonb,"
                "'2026-09-18T00:10:00Z'::timestamptz)",
                (
                    participant["submission_id"], ROUND,
                    participant["miner_hotkey"], is_baseline,
                    json.dumps(document), participant["source_ref"], source_size,
                    json.dumps({"public_rerun": True}),
                ),
            )
        run_values = []
        for participant in participants:
            if participant["is_king"]:
                for position in range(10):
                    run_values.append(_run_row(
                        submission_id=BASELINE, miner_hotkey=BASELINE_HOTKEY,
                        position=position, attempt=1, status="accepted",
                        terminal_cause="accepted",
                    ))
                for position in range(10, 20):
                    run_values.append(_run_row(
                        submission_id=BASELINE, miner_hotkey=BASELINE_HOTKEY,
                        position=position, attempt=1, status="failed",
                        terminal_cause="provider_error",
                    ))
                for position in (10, 11, 12, 13, 14, 19):
                    run_values.append(_run_row(
                        submission_id=BASELINE, miner_hotkey=BASELINE_HOTKEY,
                        position=position, attempt=2, status="failed",
                        terminal_cause="provider_error",
                    ))
                for position in (15, 16, 18):
                    run_values.append(_run_row(
                        submission_id=BASELINE, miner_hotkey=BASELINE_HOTKEY,
                        position=position, attempt=2, status="accepted",
                        terminal_cause="accepted",
                    ))
                run_values.append(_run_row(
                    submission_id=BASELINE, miner_hotkey=BASELINE_HOTKEY,
                    position=17, attempt=2, status="failed",
                    terminal_cause="model_error",
                ))
            else:
                for position in range(20):
                    run_values.append(_run_row(
                        submission_id=participant["submission_id"],
                        miner_hotkey=participant["miner_hotkey"], position=position,
                        attempt=1, status="accepted", terminal_cause="accepted",
                    ))
        cursor.executemany(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,stage_generation,terminal_cause,"
            "result_doc,output_ref,terminal_doc) VALUES ("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb)",
            run_values,
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET runner_hotkey=%s "
            "WHERE submission_id=%s AND attempt=2 AND status='failed' "
            "AND icp_position = ANY(%s)",
            (PREVIOUS_RUNNER_HOTKEY, BASELINE, list(RETRY_POSITIONS)),
        )
        for position in CAP_REFUSAL_POSITIONS:
            run_id = _baseline_run_id(position, 2)
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd,"
                "entry_doc) VALUES ('refusal',%s,%s,%s,%s,2,%s,'openrouter',"
                "'openrouter.responses','host',0,%s::jsonb)",
                (
                    BASELINE_HOTKEY, ROUND, BASELINE, run_id,
                    f"sha256:{position:064x}",
                    json.dumps({
                        "reason": "money_cap",
                        "attempt": 2,
                        "execution_icp_cap_microusd": 4_000_000,
                        "cost_per_company_cap_microusd": 800_000,
                    }),
                ),
            )
        # Position 12 has a settled provider retry and no money-cap refusal.
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc,terminal_response) VALUES ('settlement',%s,%s,%s,%s,2,%s,"
            "'openrouter','openrouter.responses','host',500000,%s::jsonb,%s::jsonb)",
            (
                BASELINE_HOTKEY, ROUND, BASELINE,
                _baseline_run_id(12, 2), "sha256:" + "c" * 64,
                json.dumps({"attempt": 2, "provider_retry": True}),
                json.dumps({"call_succeeded": True}),
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


def _apply(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.commit()


def _connect(database):
    psycopg2, dsn = database
    return psycopg2.connect(**dsn)


def test_budget_stop_recovery_adds_only_authorized_retries_and_replays(database):
    connection = _connect(database)
    try:
        _seed(connection)
        with connection.cursor() as cursor:
            before = _snapshot(cursor)
        assert len(before["lab_arena_runs"]) == 110
        assert sum(row["status"] == "accepted" for row in before["lab_arena_runs"]) == 93
        _apply(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,status_generation,stage_generation,cancel_reason,"
                "configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            round_row = cursor.fetchone()
            post_apply = _snapshot(cursor)
            cursor.execute(
                "SELECT icp_position,attempt,status,stage_generation,terminal_cause "
                "FROM public.lab_arena_runs WHERE submission_id=%s "
                "AND attempt=3 ORDER BY icp_position",
                (BASELINE,),
            )
            retries = cursor.fetchall()
        assert round_row[:4] == ("stage1", 4, 3, None)
        assert round_row[4] == before["lab_arena_rounds"][0]["configuration_doc"]
        assert post_apply["lab_arena_submissions"] == before["lab_arena_submissions"]
        assert post_apply["lab_arena_ledger"] == before["lab_arena_ledger"]
        old_runs = {row["run_id"]: row for row in before["lab_arena_runs"]}
        assert {row["run_id"]: row for row in post_apply["lab_arena_runs"] if row["run_id"] in old_runs} == old_runs
        assert len(post_apply["lab_arena_runs"]) == 116
        assert retries == [(position, 3, "pending", 3, None) for position in RETRY_POSITIONS]
        assert sum(row["entry_doc"].get("reason") == "money_cap" for row in before["lab_arena_ledger"]) == 5
        claimed_positions = []
        with connection.cursor() as cursor:
            for claim_number in range(6):
                cursor.execute(
                    "SELECT public.lab_arena_claim_assignment(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        ROUND, BASELINE_HOTKEY, 10, 20, [],
                        f"291{claim_number:029x}",
                        "sha256:" + "b" * 64,
                        "sha256:" + f"{claim_number + 1:064x}", 3600,
                    ),
                )
                claim = cursor.fetchone()[0]
                assert claim["status"] == "leased"
                assert claim["attempt"] == 3 and claim["stage_generation"] == 3
                claimed_positions.append(int(claim["icp_position"]))
                cursor.execute(
                    "SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)",
                    (
                        claim["run_id"],
                        "sha256:" + f"{claim_number + 1:064x}",
                        json.dumps({"terminal_status": "accepted"}),
                        "accepted",
                        f"arena/{ROUND}/outputs/recovery291/{claim['icp_position']}.json",
                    ),
                )
                assert cursor.fetchone()[0]["status"] == "accepted"
            cursor.execute(
                "SELECT previous_runner_hotkey,status FROM public.lab_arena_runs "
                "WHERE submission_id=%s AND attempt=3 ORDER BY icp_position",
                (BASELINE,),
            )
            retry_states = cursor.fetchall()
            cursor.execute(
                "SELECT public.lab_arena_close_parallel_execution_v1(%s)",
                (ROUND,),
            )
            close_result = cursor.fetchone()[0]
        assert sorted(claimed_positions) == list(RETRY_POSITIONS)
        assert retry_states == [(PREVIOUS_RUNNER_HOTKEY, "accepted")] * 6
        assert close_result["status"] == "closed"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,status_generation,stage_generation,cancel_reason "
                "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,)
            )
            assert cursor.fetchone() == ("stage1_closed", 5, 4, None)
            cursor.execute(
                "SELECT count(DISTINCT assignment_id),count(*) FILTER (WHERE status='accepted') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute'",
                (ROUND,),
            )
            assert cursor.fetchone() == (100, 99)
            after = _snapshot(cursor)
        _apply(connection)
        with connection.cursor() as cursor:
            replayed = _snapshot(cursor)
        assert replayed == after
    finally:
        connection.close()


@pytest.mark.parametrize(
    "mutation,message",
    (
        (
            "INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,"
            "submission_id,run_id,stage,call_identity,provider,operation_id,"
            "funding_source,amount_microusd,entry_doc) VALUES ('reservation',"
            "'" + BASELINE_HOTKEY + "','" + ROUND + "','" + BASELINE + "',"
            "'" + _baseline_run_id(12, 2) + "',2,'sha256:'||repeat('d',64),"
            "'openrouter','openrouter.responses','host',100000,'{}'::jsonb)",
            "execution snapshot differs",
        ),
        (
            "UPDATE public.lab_arena_submissions SET source_ref='arena/"
            + ROUND + "/sources/wrong.tar.gz' WHERE submission_id='" + BASELINE + "'",
            "source",
        ),
        (
            "UPDATE public.lab_arena_runs SET terminal_cause='model_error' WHERE "
            "run_id='" + _baseline_run_id(12, 2) + "'",
            "unfinished attempt",
        ),
        (
            "UPDATE public.lab_arena_rounds SET cancel_reason='execution_incomplete:stage1:5' "
            "WHERE round_id='" + ROUND + "'",
            "terminal state",
        ),
        (
            "UPDATE public.lab_arena_rounds SET participants=participants || "
            "jsonb_build_array(jsonb_build_object('submission_id','unexpected',"
            "'miner_hotkey','5ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')) "
            "WHERE round_id='" + ROUND + "'",
            "source or policy differs",
        ),
        (
            "UPDATE public.lab_arena_runs SET kind='score',scored_run_id=run_id "
            "WHERE run_id='" + _baseline_run_id(0, 1) + "'",
            "execution snapshot differs",
        ),
    ),
)
def test_budget_stop_recovery_rejects_changed_state_atomically(database, mutation, message):
    connection = _connect(database)
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
            _apply(connection)
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
